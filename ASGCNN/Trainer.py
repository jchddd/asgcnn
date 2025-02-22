import torch as th
from torch import nn
import numpy as np
import copy
import random
# from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, multilabel_confusion_matrix, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import warnings
from ASGCNN.Encoder import Graph_data_loader
from ASGCNN.Model import ASGCNN, CGCNN
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss

warnings.filterwarnings('ignore')


def setup_seed(seed):
    '''
    Function used to set the random seed

    Parameters:
        - seed: The random seed / int
    '''
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    th.use_deterministic_algorithms(True)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.enabled = False


class Trainer():
    def __init__(self,
                 Module,
                 Dataloader_train=None,
                 Dataloader_valid=None,
                 Dataloader_test=None,
                 init_lr=0.1,
                 metric='mae',
                 metric_para={},
                 optimizer='RMSprop',
                 scheduler='cos',
                 scheduler_para=1000,
                 weight_decay=0.0,
                 target_dims=[1]):
        '''
        Trainer

        Parameters:
            - Module: the Pytorch Module
            - Dataloader_train: Graph_data_loader for training set
            - Dataloader_valid: Graph_data_loader for validation set
            - Dataloader_test: Graph_data_loader for test set
            - init_lr: initial learning rate / float, default 0.1
            - metric: loss function type / str, in 'mae', 'mse', 'rmse', 'hyb', 'wmae'
            - metric_para: parameters for hybrid loss function except target_dims / dict, default {}
            - optimizer: optimizer type / str, in 'SGD', 'Momentum', 'RMSprop', 'Adam', 'AdamW'
            - scheduler: scheduler type / str, in 'cos', 'step'
            - scheduler_para: length of the periodicity of the learning rate changing / int, default 1000
             in 'cos', it means the number of epochs in a cycle of learning rate from large to small
             in 'step' it means the learning rate changes after how many epochs
            - weight_decay: weight_decay on optimizer / float, default 0.0
            - target_dims: dimensions for each target / float, default [1]
        '''

        # Module
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.Module = Module.to(self.device)
        # Dataloader
        self.Dataloader_train = Dataloader_train
        self.Dataloader_valid = Dataloader_valid
        self.Dataloader_test = Dataloader_test
        # Metric
        self.metric = metric
        metric_para.update({'target_dims': target_dims}) if metric == 'hyb' else None
        self.Metric = {'mae': nn.SmoothL1Loss, 'mse': nn.MSELoss, 'rmse': RMSELoss, 'hyb': HybLoss, 'wmae': wMAE}[metric](**metric_para).to(self.device)
        # Optimizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_para = scheduler_para
        self.weight_decay = weight_decay
        # Learning rate
        self.init_lr = init_lr
        self.last_epoch = -1
        self.last_lr = init_lr
        self.lrs = []
        # Loss and weight
        self.losses_train = []
        self.losses_valid = []
        self.best_weights = None
        # Snapshot
        self.snapshot = False
        self.snapshot_point = []
        self.Modules = []
        self.to_normal_ensemble = False
        # target
        self.target_dims = target_dims

    def _load_optimizer(self, optimizer, lr, init_lr):
        if optimizer == 'SGD':
            return th.optim.SGD(self._group_weight(init_lr), lr=lr, weight_decay=self.weight_decay)
        elif optimizer == 'Momentum':
            return th.optim.SGD(self._group_weight(init_lr), lr=lr, momentum=0.9, weight_decay=self.weight_decay)
        elif optimizer == 'RMSprop':
            return th.optim.RMSprop(self._group_weight(init_lr), lr=lr, weight_decay=self.weight_decay)
        elif optimizer == 'Adam':
            return th.optim.Adam(self._group_weight(init_lr), lr=lr, weight_decay=self.weight_decay)
        elif optimizer == 'AdamW':
            return th.optim.AdamW(self._group_weight(init_lr), lr=lr, weight_decay=self.weight_decay)

    def _group_weight(self, init_lr):
        group_decay = []
        group_no_decay = []
        for m in self.Module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                group_no_decay.extend([*m.parameters()])

        # assert len(list(self.Module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay, initial_lr=init_lr), dict(params=group_no_decay, initial_lr=init_lr, weight_decay=.0)]
        return groups

    def _load_scheduler(self, scheduler, optimizer, step_para, last_epoch):
        if scheduler == 'cos':
            return th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_para, eta_min=0.000001, last_epoch=last_epoch)
        elif scheduler == 'step':
            return th.optim.lr_scheduler.StepLR(optimizer, step_size=step_para, gamma=0.1, last_epoch=last_epoch)

    def init_snapshot_ensembling(self, cycle_time=5, epoch_each=200, weight_init=False):
        '''
        Initialize snapshot ensembling

        Parameters:
            - cycle_time: the number of cycles to collect the model / int, default 5
            - epoch_each: epoches for each cycle / int, default 200
            - weight_init: after the model is acquired, initialize the weights, equivalent to an ordinary ensemble model / bool, default False

        '''
        self.snapshot = True
        self.snapshot_point = [(i + 1) * epoch_each - 1 for i in range(cycle_time)]
        self.scheduler = 'cos'
        self.scheduler_para = epoch_each
        if weight_init:
            self.to_normal_ensemble = True

    def train(self, epoch, disable_tqdm=False):
        '''
        Strat training

        Parameter:
            - epoch: train epoch, it's theoretically possible to train continuously / int
        '''
        # init
        self.Dataloader_train.init_train()
        if self.last_epoch == -1:
            self.Module.apply(weight_init)
        Optimizer = self._load_optimizer(self.optimizer, self.last_lr, self.init_lr)
        Scheduler = self._load_scheduler(self.scheduler, Optimizer, self.scheduler_para, self.last_epoch)
        # start train
        with tqdm(range(epoch), leave=False, unit='e', desc='Train epoch', disable=disable_tqdm) as pbar:
            for e in range(epoch):  # loop epoch
                # init
                self.Module.train()
                metric_in_e = []
                for x, y in self.Dataloader_train:  # loop Dataloader
                    # calculate loss
                    yp = self._module_forward(x)
                    Optimizer.zero_grad()
                    if self.Dataloader_train.use_ELD:
                        w = self.Dataloader_train.get_LDS()
                        loss = self.Metric(yp, y, w)
                    elif not self.Dataloader_train.use_ELD:
                        loss = self.Metric(yp, y)
                    # execute backward
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.Module.parameters(), max_norm=20, norm_type=2)
                    Optimizer.step()
                    # update train loss
                    metric_in_e.append(float(loss.item()))
                self.losses_train.append(np.mean(metric_in_e))
                # update lr and Scheduler
                self.lrs.append(self.last_lr)
                Scheduler.step()
                self.last_lr = Scheduler.get_last_lr()[0]
                self.last_epoch += 1
                # calculate validation loss
                self.losses_valid.append(self._calculate_metric(self.Dataloader_valid))
                # judge and store the best weights
                if self.losses_valid[-1] < min(self.losses_valid):
                    self.best_weights = copy.deepcopy(Module.state_dict())
                # snapshot
                if self.snapshot and self.last_epoch in self.snapshot_point:
                    self.Modules.append(copy.deepcopy(self.Module))
                    if self.to_normal_ensemble:
                        self.Module.apply(weight_init)
                    Optimizer = self._load_optimizer(self.optimizer, self.init_lr, self.init_lr)
                    self.last_lr = self.init_lr
                    Scheduler = self._load_scheduler(self.scheduler, Optimizer, self.scheduler_para, -1)
                # update pbar
                pbar.update(1)

        # restore the best mode
        self._restore()

    def _module_forward(self, x, model=None):
        if model is None:
            model = self.Module
        if len(x) == 1:
            yp = model.forward(x[0])
        elif len(x) == 2:
            yp = model.forward(x[0], x[1])
        elif len(x) == 3:
            yp = model.forward(x[0], x[1], x[2])
        elif len(x) == 4:
            yp = model.forward(x[0], x[1], x[2], x[3])
        return yp

    def _calculate_metric(self, dataloader):
        self.Module.eval()
        metrics = []
        for x, y in dataloader:
            yp = self._module_forward(x)
            metrics.append(float(self.Metric(yp, y).item()))
        return np.mean(metrics)

    def _restore(self):
        if self.best_weights != None:
            self.Module.load_state_dict(self.best_weights)

    def save(self, path='model'):
        '''
        Function to save models
        
        Parameters:
            - path: path to where store modes / str
        '''
        if self.snapshot:
            for i, m in enumerate(self.Modules):
                m.name = m.name + str(i)
                path_dict = os.path.join(path, m.name + '_dict.pkl')
                path_model = os.path.join(path, m.name + '_model.pkl')
                th.save(m.state_dict(), path_dict)
                th.save(m, path_model)
        else:
            path_dict = os.path.join(path, self.Module.name + '_dict.pkl')
            path_model = os.path.join(path, self.Module.name + '_model.pkl')
            th.save(self.Module.state_dict(), path_dict)
            th.save(self.Module, path_model)

    def get_graph_vector(self, Dataloader):
        '''
        Function to extract graph pool vectors (structure represented feature) from a Dataloader

        Parameter:
            - Dataloader: data loader that is used to perform prediction
        '''
        graph_vectors = []
        if not self.snapshot:
            Dataloader.init_predict()
            self.Module.return_vp = True
            self.Module.eval()
            with th.no_grad():
                for x, y in Dataloader:
                    _, graph_vector = self._module_forward(x, self.Module)
                    graph_vectors.append(graph_vector)
            graph_vectors = th.cat(graph_vectors, dim=0)
            self.Module.return_vp = False
        elif self.snapshot:
            for i, model in enumerate(slef.Modules):
                graph_vectors.append([])
                Dataloader.init_predict()
                model.return_vp = True
                model.eval()
                with th.no_grad():
                    for x, y in Dataloader:
                        _, graph_vector = self._module_forward(x, model)
                        graph_vectors[i].append(graph_vector)
                graph_vectors[i] = th.cat(graph_vectors[i], dim=0)
                model.return_vp = False
        return graph_vectors
                
            
    def predict(self, Dataloader, to_class=False, return_uq=False):
        '''
        Function to use trained model to predict target from a Dataloader

        Parameters:
            - Dataloader: the Dataloader
            - to_class: converts the probability of each classes into class information from 0 to x
            - return_uq: whether to return uq values / bool, default False
        '''
        if not self.snapshot:
            return self._get_prediction(self.Module, Dataloader, to_class)

        elif self.snapshot and self.Module.task_typ == 'multy':
            predicts = []
            for model in self.Modules:
                predicts.append(self._get_prediction(model, Dataloader))
            predict = np.stack(predicts)

            yps = []
            i_s = 0
            i_e = self.target_dims[0]
            for ti in range(len(self.target_dims)):
                if ti != 0:
                    i_s += self.target_dims[ti - 1]
                    i_e += self.target_dims[ti]
                if   self.target_dims[ti] > 1:
                    ypc = np.mean(predict[:, :, i_s: i_e], axis=0)
                    if to_class:
                        _, ypc = th.max(th.from_numpy(ypc), 1)
                        ypc = ypc.unsqueeze(1).cpu().detach().numpy()
                    yps.append(ypc)
                elif self.target_dims[ti] == 1:
                    break
            ypr = np.mean(predict[:, :, i_s:], axis=0)
            yps.append(ypr) # [:, np.newaxis]
            uq = np.std(predict[:, :, i_s:], axis=0)
            predict = np.concatenate(yps, axis=1)

        elif self.snapshot and self.Module.task_typ == 'regre':
            predicts = []
            for model in self.Modules:
                predicts.append(self._get_prediction(model, Dataloader))
            predict = np.stack(predicts)
            uq = np.std(predict, axis=0)
            predict = np.mean(predict, axis=0)

        if return_uq:
            return predict, uq
        else:
            return predict

    def _get_prediction(self, model, Dataloader, to_class=False):
        Dataloader.init_predict()
        model.eval()
        predicts = []
        with th.no_grad():
            for x, y in Dataloader:
                predict = self._module_forward(x, model)
                predicts.append(predict)
        predict = th.cat(predicts, dim=0)
        if to_class:
            yps = []
            i_s = 0
            i_e = self.target_dims[0]
            for ti in range(len(self.target_dims)):
                if ti != 0:
                    i_s += self.target_dims[ti - 1]
                    i_e += self.target_dims[ti]
                if   self.target_dims[ti] > 1:
                    _, ypc = th.max(predict[:, i_s: i_e], 1)
                    yps.append(ypc.unsqueeze(1))
                elif self.target_dims[ti] == 1:
                    yps.append(predict[:, i_s: i_e])
            predict = th.cat(yps, 1)
        return predict.cpu().detach().numpy()

    def calculate_static(self):
        static = {'train': {}, 'valid': {}, 'test': {}}
        keys = ['train', 'valid', 'test']

        if self.Module.task_typ == 'multy':
            for i, k in enumerate(keys):
                dataloader = getattr(self, 'Dataloader_' + k)
                if dataloader is not None:
                    yp = th.Tensor(self.predict(dataloader))

                    i_s = 0
                    i_e = self.target_dims[0]
                    for ti in range(len(self.target_dims)):
                        if ti != 0:
                            i_s += self.target_dims[ti - 1]
                            i_e += self.target_dims[ti]
                        if   self.target_dims[ti] > 1:
                            _, ypc = th.max(yp[:, i_s: i_e], 1)
                            ypc = ypc.cpu().detach().numpy().reshape(-1)
                            yc = dataloader.target[:, ti].cpu().detach().numpy().reshape(-1)
                            static[k].update(
                                {'f1_' + str(ti): f1_score(yc, ypc, average='weighted'), 'ac_' + str(ti): accuracy_score(yc, ypc),
                                 'pc_' + str(ti): precision_score(yc, ypc, average='weighted'), 'rc_' + str(ti): recall_score(yc, ypc, average='weighted')}
                            )
                        elif self.target_dims[ti] == 1:
                            ypr = yp[:, i_s: i_e].cpu().detach().numpy().reshape(-1)
                            yr = dataloader.target[:, ti].cpu().detach().numpy().reshape(-1)
                            static[k].update(
                                {'r2_' + str(ti): r2_score(yr, ypr), 'mae_' + str(ti): mean_absolute_error(yr, ypr),
                                 'mse_' + str(ti): mean_squared_error(yr, ypr), 'rmse_' + str(ti): np.sqrt(mean_squared_error(yr, ypr))}
                            )
    
        elif self.Module.task_typ == 'regre':
            target_num = len(self.target_dims)
            for i, k in enumerate(keys):
                dataloader = getattr(self, 'Dataloader_' + k)
                if dataloader is not None:
                    yp = th.Tensor(self.predict(dataloader)).cpu().detach().numpy().reshape(-1)
                    y = dataloader.target.cpu().detach().numpy().reshape(-1)
                    y = y.reshape(-1, target_num)
                    yp = yp.reshape(-1, target_num)
                    for j in range(target_num):
                        static[k].update({'r2_' + str(j): r2_score(y[:, j], yp[:, j]), 'mae_' + str(j): mean_absolute_error(y[:, j], yp[:, j]),
                                          'mse_' + str(j): mean_squared_error(y[:, j], yp[:, j]), 'rmse_' + str(j): np.sqrt(mean_squared_error(y[:, j], yp[:, j]))})

        return static

    def confusion_matrix(self, dataloader):
        Dataloader = {'train': self.Dataloader_train, 'valid': self.Dataloader_valid, 'test': self.Dataloader_test}[dataloader]
        yp = th.Tensor(self.predict(Dataloader))
        cms = []
        mcms = []

        i_s = 0
        i_e = self.target_dims[0]
        for ti in range(len(self.target_dims)):
            if ti != 0:
                i_s += self.target_dims[ti - 1]
                i_e += self.target_dims[ti]
            if   self.target_dims[ti] > 1:
                _, ypc = th.max(yp[:, i_s: i_e], 1)
                ypc = ypc.cpu().detach().numpy().reshape(-1)
                yc = Dataloader.target[:, ti].cpu().detach().numpy().reshape(-1)
                cms.append(confusion_matrix(yc, ypc))
                mcms.append(multilabel_confusion_matrix(yc, ypc))
            elif self.target_dims[ti] == 1:
                break
                
        return cms, mcms

    def show_learn_curve(self):
        plt.figure(figsize=(12, 6))
        fontsize = 15
        plt.title('Loss Curve', fontsize=fontsize)
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Loss', fontsize=fontsize)
        plt.plot(range(len(self.losses_train)), self.losses_train, 'y', label='train', zorder=2)
        plt.plot(range(len(self.losses_valid)), self.losses_valid, 'b', label='valid', zorder=1)
        plt.xticks(fontsize=fontsize - 3)
        plt.yticks(fontsize=fontsize - 3)
        plt.legend()
        plt.show()
        
    def show_lr(self):
        plt.figure(figsize=(12, 6))
        fontsize = 15
        plt.title(" Learning Rate Curve", fontsize=fontsize)
        plt.xlabel('Epoch', fontsize=fontsize)
        plt.ylabel('Learning Rate', fontsize=fontsize)
        plt.plot(self.lrs)
        plt.xticks(fontsize=fontsize - 3)
        plt.yticks(fontsize=fontsize - 3)
        plt.show()

    def show_result(self, ti=0, dataset='test'):
        fontsize = 15
        if self.target_dims[ti] == 1:
            maxv = max(self.Dataloader_train.target[:, ti]).cpu().item()
            minv = min(self.Dataloader_train.target[:, ti]).cpu().item()
            maxv = maxv + (maxv - minv) * 0.1
            minv = minv - (maxv - minv) * 0.1
            # init figure
            plt.figure(figsize=(6, 6))
            plt.title('parity plot', fontsize=fontsize)
            plt.xlabel('DFT', fontsize=fontsize)
            plt.ylabel('predict', fontsize=fontsize)
            plt.plot([-10, 10], [-10, 10])
            plt.axis([minv, maxv, minv, maxv])
            plt.xticks(fontsize=fontsize - 3)
            plt.yticks(fontsize=fontsize - 3)
            # init
            print_name = ['train', 'valid', 'test']
            colors = ['y', 'b', 'r']
            # loop dataloader
            for i, l in enumerate(print_name):
                dataloader = getattr(self, 'Dataloader_' + l)
                if dataloader is not None:
                    # get yp
                    yp = self.predict(dataloader, True)[:, ti]
                    y = dataloader.target[:, ti].cpu().detach().numpy().reshape(-1)
                    # cal static
                    plt.scatter(6, 6, c=colors[i], label=print_name[i], ec='k',lw=0.6)
                    # plot scatter
                    plt.scatter(y, yp, c=colors[i], ec='k',lw=0.6)
            plt.legend()
            plt.show()
        elif self.target_dims[ti] > 1:
            i_s = 0
            i_e = self.target_dims[0]
            for i in range(len(self.target_dims)):
                if i != 0:
                    i_s += self.target_dims[i - 1]
                    i_e += self.target_dims[i]
                if i == ti:
                    break
            if dataset == 'test':
                y_true = self.Dataloader_test.target[:, ti].cpu()
                y_scores = self.predict(Dataloader=self.Dataloader_test)[:, i_s: i_e]
            elif dataset == 'valid':
                y_rue = self.Dataloader_valid.target[:, ti].cpu()
                y_scores = self.predict(Dataloader=self.Dataloader_valid)[:, i_s: i_e]
            elif dataset == 'train':
                y_rue = self.Dataloader_train.target[:, ti].cpu()
                y_scores = self.predict(Dataloader=self.Dataloader_train)[:, i_s: i_e]

            # binary label
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            n_classes = y_true_bin.shape[1]
            # calculate roc for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # calculate average roc
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            # plot roc for each class
            plt.figure(figsize=(10, 6))
            lw = 2
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
            # plot average roc
            plt.plot(fpr["macro"], tpr["macro"],
                     label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)
            # plot setting
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=fontsize)
            plt.ylabel('True Positive Rate', fontsize=fontsize)
            plt.title('Multi-class ROC curve', fontsize=fontsize)
            plt.xticks(fontsize=fontsize - 3)
            plt.yticks(fontsize=fontsize - 3)
            plt.legend(loc="lower right")
            plt.show()
    
    def show_uq(self, tir=0, i_s=0, i_e=10, dataset='test'):
        if dataset == 'test':
            dataloader = self.Dataloader_test
        elif dataset == 'valid':
            dataloader = self.Dataloader_valid
        elif dataset == 'train':
            dataloader = self.Dataloader_train
        
        ct = 0
        for td in self.target_dims:
            if td > 1:
                ct += 1
        
        yp, uq = self.predict(dataloader, True, True)
        true_values = dataloader.target[:, tir + ct].cpu()[i_s: i_e]
        predicted_values = yp[:, tir + ct][i_s: i_e]
        uncertainty = uq[:, tir][i_s: i_e]

        fontsize = 15
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.xticks(fontsize=fontsize - 3)
        plt.yticks(fontsize=fontsize - 3)
        ax.scatter(true_values, predicted_values, color='blue', label='Predictions')
        ax.errorbar(true_values, predicted_values, xerr=uncertainty, yerr=uncertainty, fmt='o', color='red', ecolor='lightgray', capsize=5, label='Uncertainty')
        ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], '--k', label='Perfect prediction')
        ax.set_title('Predictions vs True Values with Uncertainty', fontsize=fontsize)
        ax.set_xlabel('True Values', fontsize=fontsize)
        ax.set_ylabel('Predicted Values', fontsize=fontsize)
        ax.legend()
        plt.show()

    
    def load_pretrained(self, location='cpu'):
        for i in range(5):
            self.Modules.append(copy.deepcopy(self.Module))
            self.snapshot=True
        self.Modules[0].load_state_dict(th.load(r'pretrained\ASGCNN0_dict.pkl', map_location=th.device(location)))
        self.Modules[1].load_state_dict(th.load(r'pretrained\ASGCNN1_dict.pkl', map_location=th.device(location)))
        self.Modules[2].load_state_dict(th.load(r'pretrained\ASGCNN2_dict.pkl', map_location=th.device(location)))
        self.Modules[3].load_state_dict(th.load(r'pretrained\ASGCNN3_dict.pkl', map_location=th.device(location)))
        self.Modules[4].load_state_dict(th.load(r'pretrained\ASGCNN4_dict.pkl', map_location=th.device(location)))


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class wMAE(th.nn.Module):
    def __init__(self, beta=1, wi=0):
        super(wMAE, self).__init__()
        self.mae = nn.SmoothL1Loss(reduction='none', beta=beta)
        self.wi = wi

    def forward(self, yp, y, w=None):
        loss = self.mae(yp, y)
        if w is not None:
            loss = th.mean(loss * w[:, self.wi: self.wi + 1])
        else:
            loss = th.mean(loss)
        return loss


class RMSELoss(th.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, yp, y):
        criterion = nn.MSELoss()
        loss = th.sqrt(criterion(yp, y))
        return loss


class HybLoss(th.nn.Module):
    def __init__(self, weight=[0.4, 0.4, 0.6], target_dims=[5, 4, 1], loss_para=None):
        super(HybLoss, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.weight = th.Tensor(weight).to(self.device)
        self.target_dims = target_dims
        for ti, tl in enumerate(target_dims):
            tv = 0
            if   tl > 1:
                setattr(self, 'cel_' + str(ti), nn.CrossEntropyLoss())
                tv += tl
            elif tl == 1:
                beta = loss_para[ti] if loss_para is not None else 1
                setattr(self, 'mae_' + str(ti), wMAE(beta=beta, wi=tl))
                tv += tl

    def forward(self, yp, y, w=None):
        i_s = 0
        i_e = self.target_dims[0]
        for ti in range(len(self.target_dims)):
            if ti != 0:
                i_s += self.target_dims[ti - 1]
                i_e += self.target_dims[ti]
            if   self.target_dims[ti] > 1:
                loss = getattr(self, 'cel_' + str(ti))(yp[:, i_s: i_e], y[:, ti].long().squeeze())
            elif self.target_dims[ti] == 1:
                loss = getattr(self, 'mae_' + str(ti))(yp[:, i_s: i_e], y[:, ti: ti + 1], w)

            if ti == 0:
                loss_total = loss * self.weight[ti]
            else:
                loss_total += loss * self.weight[ti]
        return loss_total


def node_feature_forward_selection(model, trainer, element_encoder, train_epoch=100, target_metric=[['valid', 'mae_0']], max_run=10):
    """
    Forward selection

    Parameters:
        - model: the Module class
        - trainer: a Trainer with model and train and validation dataloader.
          Feature search after global features are added is not supported by now.
        - element_encoder: an Element_Encoder with all candidate features. 
          For ASGCNN, both types of graphs must use the same node features
        - train_epoch: training epoch / int, default 100
        - target_metric: metric as the selection criterions. / (n, 2) list, default [['valid', 'mae_0']]
          You can choose multiple metrics and use their sum as a criterion. Fill in the keys from the dictionary obtained by t.calculate_static()
          At present, this only supports judging by the minimum value. The better the model, the lower the value of the metric show be
        - max_run: maximum number of features to select / int, default 10
    Returns:
        - best features and scores
    """
    setup_seed(1234)
    all_features = element_encoder.features
    max_run = max_run if len(all_features) >= max_run else len(all_features)
    best_features = []
    best_scores = [np.inf] # np.NINF
    trainer.snapshot = False
    model_para = trainer.Module.model_args
    # loop to max_run 
    for i in range(max_run):
        # record feature and score for each loop
        check_features = []
        check_scores = []
        # forward select features 
        for feat in all_features:
            if feat not in best_features:
                setup_seed(1234)
                # reset element encoder
                element_encoder.features = best_features + [feat]
                element_encoder.assess_feature_properities()
                element_encoder.encode_element()
                # reset dataloader
                trainer.Dataloader_train.init_predict()
                trainer.Dataloader_train.init_train()
                trainer.Dataloader_valid.init_predict()
                trainer.Dataloader_valid.init_train()
                # reset trainer
                if   trainer.Module.name == 'CGCNN':
                    model_para.update({'node_feat_length': element_encoder.feature_lentgh_total})
                    trainer.Module=model(**model_para).to(trainer.device)
                    trainer.Dataloader_train.apply_feature([element_encoder], [None], disable_tqdm=True)
                    trainer.Dataloader_valid.apply_feature([element_encoder], [None], disable_tqdm=True)
                elif trainer.Module.name == 'ASGCNN':
                    model_para.update({'node_feat_length_adsb': element_encoder.feature_lentgh_total,
                                       'node_feat_length_slab': element_encoder.feature_lentgh_total})
                    trainer.Module=model(**model_para).to(trainer.device)
                    trainer.Dataloader_train.apply_feature([element_encoder, element_encoder], [None, None], disable_tqdm=True)
                    trainer.Dataloader_valid.apply_feature([element_encoder, element_encoder], [None, None], disable_tqdm=True)
                trainer.last_epoch = -1
                trainer.last_lr = trainer.init_lr
                # train
                trainer.train(train_epoch, disable_tqdm=True)
                static = trainer.calculate_static()
                # add check info
                check_features.append(feat)
                check_scores.append(sum([static[t[0]][t[1]] for t in target_metric]))
        # add best feature
        best_score = min(check_scores)
        best_scores.append(best_score)
        best_features.append(check_features[check_scores.index(best_score)])
        if   best_score <= best_scores[-2]:
            continue
        else:
            break

    return best_features, best_scores[1:]


def show_forawrd_selection(features, scores):
    plt.figure(figsize=(12,6))
    number_of_features = np.array(list(range(len(features)))) + 1
    plt.plot(number_of_features, scores, marker='o', color='grey', markersize=12, markeredgecolor='k', markerfacecolor='w', ls='--', zorder=0)

    y_text = max(scores) - (max(scores) - min(scores)) * 0.1
    for i in number_of_features:
        feature_text = '\n\n'.join([f for f in features[0: i]])
        plt.text(i + 0.1, y_text, feature_text, va='top', zorder=6)
    fontsize=15
    plt.xlim(0.8, max(number_of_features) + 0.8)
    plt.xticks(number_of_features, fontsize=fontsize - 3)
    plt.yticks(fontsize=fontsize - 3)
    plt.title('Feature Selection Process', fontsize=fontsize)
    plt.xlabel('Number of Features', fontsize=fontsize)
    plt.ylabel('Scores', fontsize=fontsize)


def optimize_hyperparameters(
    model_type='CGCNN',
    train_data_excel='',
    train_bins=[],
    valid_data_excel='',
    valid_bins=[],
    target_columns=[],
    para_load_train_data={},
    para_load_valid_data={},
    class_dim=[],
    regre_dim=[],
    task_type='regre',
    target_dims=[],
    random_seed=666,
    minmum_metric=['valid', 'mae_2'],
    train_epoch=20,
    max_evals=100,
    early_stop_iter=100,
    use_line_graph=False,
    global_feat_train=None,
    global_feat_valid=None,
    global_feat_dim=0,
    have_cluster=False
):
    '''
    Function used to perform hyperparameter optimization. 
    The hyperparameters are defined in advance and you only need to be passed in datasets and model related parameters

    Parameters:
        - model_type: type of the model / str, 'CGCNN' or 'ASGCNN', default = 'CGCNN'
        - train_data_excel: data excel or csv file for training set data / path, default = ''
        - train_bins: graph bins for training dataset / list, default = []
        - valid_data_excel: data excel or csv file for validation set data / path, default = ''
        - valid_bins: graph bins for validation dataset / list, default = []
        - target_columns: target value column names / list, default = []
        - para_load_train_data: other key and value pairs for parameters used to load training dataset except data_excel, encoders and target / dict, default = {}
        - para_load_valid_data: other key and value pairs for parameters used to load training dataset except data_excel, encoders and target / dict, default = {}
        - class_dim: class_dim for model setting / list, default = []
        - regre_dim: regre_dim for model setting / list, default = []
        - task_type: task_type for model setting / str, default = 'regre'
        - target_dim: target_dim for Trainer setting / list, default = []
        - random_seed: the random seed / int, default = 666
        - minmum_metric: metric used to minmum the objective function, give two keys to get value from t.calculate_static() / list, default = ['valid', 'mae_0']
        - train_epoch: training epoch at each evaluation / int, default = 20
        - max_evals: max evaluation times / int, default = 100
        - early_stop_iter: This parameter determines how many times the search stops without a drop in score / int, default = 100
        - use_line_graph: whether line graph is stored on the dataset / bool, default = False
        - train_global_feat: global feature for training set / np.array, default = None
          this will add global feature to pool layer. If you wang to add them to node or edge, do it during dataset construction.
        - valid_global_feat: global feature for validation set / np.array, default = None
        - global_feat_dim: global feature dimension / int, default = 0 
        - have_cluster: whether node cluster feature have been add to graph / bool, default = False
    '''

    def load_data():
        Loader_train = Graph_data_loader()
        Loader_train.load_data(
            data_excel=train_data_excel,
            target=target_columns,
            encoders=train_bins,
            **para_load_train_data
        )

        Loader_valid = Graph_data_loader()
        Loader_valid.load_data(
            data_excel=valid_data_excel,
            target=target_columns,
            encoders=valid_bins,
            **para_load_valid_data
        )

        return Loader_train, Loader_valid

    if global_feat_train is not None:
        main_conv_typ = ['CGConv', 'EGate', 'EAtt', 'MGConv']
    else:
        main_conv_typ = ['CGConv', 'EGate', 'EAtt']
    if have_cluster:
        pool_type = ['avg', 'sum', 'cluster']
    elif not have_cluster:
        pool_type = ['avg', 'sum']
    # general search parameters
    if model_type == 'CGCNN':
        search_params = {
            'embed_feat_length': hp.quniform('embed_feat_length', 16, 124, 4),
            'conv_num': hp.choice('conv_num', [1, 2, 3, 4, 5, 6]),
            'conv_typ': hp.choice('conv_typ', main_conv_typ),
            'pool_typ': hp.choice('pool_typ', pool_type),
            'att': hp.choice('att', [True, False]),
            'fcl_dim': hp.quniform('fcl_dim', 16, 124, 4),
            'fcl_dep': hp.choice('fcl_dep', [1, 2, 3, 4, 5, 6]),
            'mlp_acti': hp.choice('mlp_acti', ['elu', 'relu', 'prelu', 'selu', 'silu', 'celu', 'leakyrelu', 'sigmoid', 'logsigmoid', 'tanh', 'tanhshrink', 'softshrink']),
            'p_droput': hp.uniform('p_droput', 0., 0.3),
            'batch_size': hp.quniform('batch_size', 64, 512, 64),
            'lr': hp.choice('lr', [1, 2, 3, 4, 5, 6]),
            'weight_decay': hp.choice('weight_decay', [6, 7, 8, 9, 10, 11, 12]),
            'optimizer': hp.choice('optimizer', ['SGD', 'Momentum', 'RMSprop', 'Adam', 'AdamW'])
        }
    elif model_type == 'ASGCNN':
        search_params = {
            'embed_feat_length_adsb': hp.quniform('embed_feat_length_adsb', 16, 124, 4),
            'conv_num_adsb': hp.choice('conv_num_adsb', [1, 2, 3, 4, 5, 6]),
            'conv_typ_adsb': hp.choice('conv_typ_adsb', ['CGConv', 'EGate', 'EAtt']),
            'pool_typ_adsb': hp.choice('pool_typ_adsb', pool_type),
            'embed_feat_length_slab': hp.quniform('embed_feat_length_slab', 16, 124, 4),
            'conv_num_slab': hp.choice('conv_num_slab', [1, 2, 3, 4, 5, 6]),
            'conv_typ_slab': hp.choice('conv_typ_slab', main_conv_typ),
            'pool_typ_slab': hp.choice('pool_typ_slab', pool_type),
            'att': hp.choice('att', [True, False]),
            'fcl_dim': hp.quniform('fcl_dim', 16, 124, 4),
            'fcl_dep': hp.choice('fcl_dep', [1, 2, 3, 4, 5, 6]),
            'mlp_acti': hp.choice('mlp_acti', ['elu', 'relu', 'prelu', 'selu', 'silu', 'celu', 'leakyrelu', 'sigmoid', 'logsigmoid', 'tanh', 'tanhshrink', 'softshrink']),
            'p_droput': hp.uniform('p_droput', 0., 0.3),
            'batch_size': hp.quniform('batch_size', 64, 512, 64),
            'lr': hp.choice('lr', [1, 2, 3, 4, 5, 6]),
            'weight_decay': hp.choice('weight_decay', [6, 7, 8, 9, 10, 11, 12]),
            'optimizer': hp.choice('optimizer', ['SGD', 'Momentum', 'RMSprop', 'Adam', 'AdamW'])
        }
    # mul target related hypermaters
    search_params['mul_fcl'] = hp.choice('mul_fcl', [True, False])
    if   task_type == 'regre':
        num_targets = len(regre_dim)
    elif task_type == 'multy':
        num_targets = len(regre_dim) + len(class_dim)
    for i in range(num_targets):
        search_params['fcl_dim_' + str(i)] = hp.quniform('fcl_dim_' + str(i), 16, 124, 4)
        search_params['fcl_dep_' + str(i)] = hp.choice('fcl_dep_' + str(i), [1, 2, 3, 4, 5, 6])
    # line graph related hypermaters
    if   use_line_graph and model_type == 'CGCNN':
        search_params['conv_num_lg'] = hp.choice('conv_num_lg', [1, 2, 3, 4, 5, 6])
        search_params['conv_typ_lg'] = hp.choice('conv_typ_lg', ['CGConv', 'EGate', 'EAtt'])
    elif use_line_graph and model_type == 'ASGCNN':
        search_params['conv_num_adsb_lg'] = hp.choice('conv_num_adsb_lg', [1, 2, 3, 4, 5, 6])
        search_params['conv_typ_adsb_lg'] = hp.choice('conv_typ_adsb_lg', ['CGConv', 'EGate', 'EAtt'])
        search_params['conv_num_slab_lg'] = hp.choice('conv_num_slab_lg', [1, 2, 3, 4, 5, 6])
        search_params['conv_typ_slab_lg'] = hp.choice('conv_typ_slab_lg', ['CGConv', 'EGate', 'EAtt'])

    def objective_function(params):
        setup_seed(random_seed)
        Loader_train, Loader_valid = load_data()
        if global_feat_train is not None:
            Loader_train.add_global_feat(global_feat_train)
            Loader_valid.add_global_feat(global_feat_valid)
            use_global_feat = True
        else:
            use_global_feat = False
        Loader_train.batch_size = int(params['batch_size'])
        Loader_valid.batch_size = int(params['batch_size'])
        
        if model_type == 'CGCNN':
            node_feat_length = Loader_train.graphs[0][0].ndata['h_v'].shape[1]
            edge_feat_length = Loader_train.graphs[0][0].edata['h_e'].shape[1]
            angle_feat_length = Loader_train.line_graphs[0][0].edata['h_le'].shape[1] if use_line_graph else 0
            # fully connect layers
            if params['mul_fcl']:
                fcl_dims = []
                for i in range(num_targets):
                    fcl_dims.append([int(params['fcl_dim_' + str(i)])] * int(params['fcl_dep_' + str(i)]))
            elif not params['mul_fcl']: 
                fcl_dims = [int(params['fcl_dim'])] * int(params['fcl_dep'])
            # convolutional layers
            if not use_line_graph:
                conv_num = int(params['conv_num'])
                conv_typ = params['conv_typ']
            elif use_line_graph:
                conv_num = [int(params['conv_num_lg']), int(params['conv_num'])]
                conv_typ = [params['conv_typ_lg'], params['conv_typ']]
            # model
            model = CGCNN(
                node_feat_length=node_feat_length,
                edge_feat_length=edge_feat_length,
                angle_feat_length=angle_feat_length,
                embed_feat_length=int(params['embed_feat_length']),
                conv_num=conv_num,
                conv_typ=conv_typ,
                pool_typ=params['pool_typ'],
                att=params['att'],
                fcl_dims=fcl_dims,
                mlp_acti=params['mlp_acti'],
                p_droput=params['p_droput'],
                task_typ=task_type,
                class_dim=class_dim,
                regre_dim=regre_dim,
                use_global_feat=use_global_feat,
                global_feat_place='pool',
                global_feat_dim=global_feat_dim
            )
        elif model_type == 'ASGCNN':
            node_feat_length_adsb = Loader_train.graphs[0][0].ndata['h_v'].shape[1]
            edge_feat_length_adsb = Loader_train.graphs[0][0].edata['h_e'].shape[1]
            angle_feat_length_adsb = Loader_train.line_graphs[0][0].edata['h_le'].shape[1] if use_line_graph else 0
            node_feat_length_slab = Loader_train.graphs[1][0].ndata['h_v'].shape[1]
            edge_feat_length_slab = Loader_train.graphs[1][0].edata['h_e'].shape[1]
            angle_feat_length_slab = Loader_train.line_graphs[1][0].edata['h_le'].shape[1] if use_line_graph else 0
            # fully connect layers
            if params['mul_fcl']:
                fcl_dims = []
                for i in range(num_targets):
                    fcl_dims.append([int(params['fcl_dim_' + str(i)])] * int(params['fcl_dep_' + str(i)]))
            elif not params['mul_fcl']: 
                fcl_dims = [int(params['fcl_dim'])] * int(params['fcl_dep'])
            # convolutional layers
            if not use_line_graph:
                conv_num_adsb = int(params['conv_num_adsb'])
                conv_typ_adsb = params['conv_typ_adsb']
                conv_num_slab = int(params['conv_num_slab'])
                conv_typ_slab = params['conv_typ_slab']
            elif use_line_graph:
                conv_num_adsb = [int(params['conv_num_adsb_lg']), int(params['conv_num_adsb'])]
                conv_typ_adsb = [params['conv_typ_adsb_lg'], params['conv_typ_adsb']]
                conv_num_slab = [int(params['conv_num_slab_lg']), int(params['conv_num_slab'])]
                conv_typ_slab = [params['conv_typ_slab_lg'], params['conv_typ_slab']]
            # model            
            model = ASGCNN(
                node_feat_length_adsb=node_feat_length_adsb,
                edge_feat_length_adsb=edge_feat_length_adsb,
                angle_feat_length_adsb=angle_feat_length_adsb,
                node_feat_length_slab=node_feat_length_slab,
                edge_feat_length_slab=edge_feat_length_slab,
                angle_feat_length_slab=angle_feat_length_slab,
                embed_feat_length_adsb=int(params['embed_feat_length_adsb']),
                conv_num_adsb=conv_num_adsb,
                conv_typ_adsb=conv_typ_adsb,
                pool_typ_adsb=params['pool_typ_adsb'],
                embed_feat_length_slab=int(params['embed_feat_length_slab']),
                conv_num_slab=conv_num_slab,
                conv_typ_slab=conv_typ_slab,
                pool_typ_slab=params['pool_typ_slab'],
                att=params['att'],
                fcl_dims=fcl_dims,
                mlp_acti=params['mlp_acti'],
                p_droput=params['p_droput'],
                task_typ=task_type,
                class_dim=class_dim,
                regre_dim=regre_dim,
                use_global_feat=use_global_feat,
                global_feat_place='pool',
                global_feat_dim=global_feat_dim
            )

        if task_type == 'regre':
            metric = 'mae'
        elif task_type == 'multy':
            metric = 'hyb'
        t = Trainer(
            Module=model,
            Dataloader_train=Loader_train,
            Dataloader_valid=Loader_valid,
            init_lr=0.1**int(params['lr']),
            optimizer=params['optimizer'],
            weight_decay=0.1**int(params['weight_decay']),
            scheduler='step',
            target_dims=target_dims,
            metric=metric
        )
        t.train(train_epoch, disable_tqdm=True)
        
        return t.calculate_static()[minmum_metric[0]][minmum_metric[1]]

    trials=Trials()
    early_stop=no_progress_loss(early_stop_iter)
    params_best= fmin(objective_function,
                   space=search_params,
                   algo=tpe.suggest,
                   max_evals=max_evals,
                   verbose=True,
                   trials=trials,
                   early_stop_fn=early_stop)

    # parame translate
    params_best['att'] = {0: False, 1: True}[params_best['att']]
    params_best['batch_size'] = int(params_best['batch_size'])
    params_best['embed_feat_length_adsb'] = int(params_best['embed_feat_length_adsb'])
    params_best['embed_feat_length_slab'] = int(params_best['embed_feat_length_slab'])
    params_best['lr'] = '1E-' + str(int(params_best['lr']))
    params_best['mul_fcl'] = {0: False, 1: True}[params_best['mul_fcl']]
    dic_mlp_act = {
        0: 'elu',
        1: 'relu',
        2: 'prely',
        3: 'selu',
        4: 'silu',
        5: 'celu',
        6: 'leakyrelu',
        7: 'sigmoid',
        8: 'logsigmoid',
        9: 'tanh',
        10: 'tanhshrink',
        11: 'softshrink'
    }
    params_best['mlp_acti'] = dic_mlp_act[params_best['mlp_acti']]
    dic_opt = {
        0: 'SGD',
        1: 'Momentum',
        2: 'RMSprop',
        3: 'Adam',
        4: 'AdamW'
    }
    params_best['optimizer'] = dic_opt[params_best['optimizer']]
    params_best['weight_decay'] = '1E-' + str(int(params_best['weight_decay']))
    dic_conv_typ = {
        0: 'CGConv',
        1: 'EGate',
        2: 'EAtt',
        3: 'MGConv'
    }
    if 'conv_typ' in params_best.keys():
        params_best['conv_typ'] = dic_conv_typ[params_best['conv_typ']]
        if use_line_graph:
            params_best['conv_typ_lg'] = dic_conv_typ[params_best['conv_typ_lg']]
    else:
        params_best['conv_typ_adsb'] = dic_conv_typ[params_best['conv_typ_adsb']]
        params_best['conv_typ_slab'] = dic_conv_typ[params_best['conv_typ_slab']]
        if use_line_graph:
            params_best['conv_typ_adsb_lg'] = dic_conv_typ[params_best['conv_typ_adsb_lg']]
            params_best['conv_typ_slab_lg'] = dic_conv_typ[params_best['conv_typ_slab_lg']]
    dic_pool_typ = {
        0: 'avg',
        1: 'sum',
        2: 'cluster'
    }
    if 'pool_typ' in params_best.keys():
        params_best['pool_typ'] = dic_pool_typ[params_best['pool_typ']]
    else:
        params_best['pool_typ_adsb'] = dic_pool_typ[params_best['pool_typ_adsb']]
        params_best['pool_typ_slab'] = dic_pool_typ[params_best['pool_typ_slab']]
    
    return params_best, trials


# not_freeze_dict=['pred_adsb.layer.0.weight','pred_adsb.layer.0.bias','pred_adsb.layer.1.weight','pred_adsb.layer.1.bias',
#                 'pred_site.layer.0.weight','pred_site.layer.0.bias','pred_site.layer.1.weight','pred_site.layer.1.bias',]
#                  'fc_layers.0.layer.0.weight','fc_layers.0.layer.0.bias','fc_layers.0.layer.1.weight','fc_layers.0.layer.1.bias',
#                  'fc_layers.1.layer.0.weight','fc_layers.1.layer.0.bias','fc_layers.1.layer.1.weight','fc_layers.1.layer.1.bias',
#                  'fc_layers.2.layer.0.weight','fc_layers.2.layer.0.bias','fc_layers.2.layer.1.weight','fc_layers.2.layer.1.bias',
#                  'fc_atten_mlp.layer.0.weight','fc_atten_mlp.layer.0.bias','fc_atten_mlp.layer.1.weight','fc_atten_mlp.layer.1.bias',
#                  'fc_atten_bn.weight','fc_atten_bn.bias']
# def freeze_model(model, not_freeze_dict):
#    for (name,param) in model.named_parameters():
#        if name not in not_freeze_dict:
#            param.requires_grad=False
#        else:
#            pass
#    return model
# SA=freeze_model(SA,not_freeze_dict)
