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

        assert len(list(self.Module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay, initial_lr=init_lr), dict(params=group_no_decay, initial_lr=init_lr, weight_decay=.0)]
        return groups

    def _load_scheduler(self, scheduler, optimizer, step_para, last_epoch):
        if scheduler == 'cos':
            return th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_para, eta_min=0.000001, last_epoch=last_epoch)
        elif scheduler == 'step':
            return th.optim.lr_scheduler.StepLR(optimizer, step_size=step_para, gamma=0.1, last_epoch=last_epoch)

    def init_snapshot_ensembling(self, cycle_time=5, epoch_each=200):
        '''
        Initialize snapshot ensembling

        Parameters:
            - cycle_time: the number of cycles to collect the model / int, default 5
            - epoch_each: epoches for each cycle / int, default 200
        '''
        self.snapshot = True
        self.snapshot_point = [(i + 1) * epoch_each - 1 for i in range(cycle_time)]
        self.scheduler = 'cos'
        self.scheduler_para = epoch_each

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
                    # self.Module.apply(weight_init)
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
        plt.figure(figsize=(6, 4))
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.plot(range(len(self.losses_train)), self.losses_train, 'y', label='train', zorder=2)
        plt.plot(range(len(self.losses_valid)), self.losses_valid, 'b', label='valid', zorder=1)
        plt.legend()
        plt.show()
        
    def show_lr(self):
        plt.figure(figsize=(6, 4))
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('learning rate', fontsize=16)
        plt.plot(self.lrs)
        plt.show()

    def show_result(self, ti=0, dataset='test'):
        if self.target_dims[ti] == 1:
            maxv = max(self.Dataloader_train.target[:, ti]).cpu().item()
            minv = min(self.Dataloader_train.target[:, ti]).cpu().item()
            maxv = maxv + (maxv - minv) * 0.1
            minv = minv - (maxv - minv) * 0.1
            # init figure
            plt.figure(figsize=(6, 6))
            plt.xlabel('DFT', fontsize=16)
            plt.ylabel('predict', fontsize=16)
            plt.plot([-10, 10], [-10, 10])
            plt.axis([minv, maxv, minv, maxv])
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
            plt.figure()
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
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multi-class ROC curve')
            plt.legend(loc="lower right")
            plt.show()
    
    def show_uq(self, ti=0, i_s=0, i_e=10, dataset='test'):
        if dataset == 'test':
            dataloader = self.Dataloader_test
        elif dataset == 'valid':
            dataloader = self.Dataloader_valid
        elif dataset == 'train':
            dataloader = self.Dataloader_train
        
        yp, uq = self.predict(dataloader, True, True)
        true_values = dataloader.target[:, ti].cpu()[i_s: i_e]
        predicted_values = yp[:, ti][i_s: i_e]
        uncertainty = uq[:, ti][i_s: i_e]

        fig, ax = plt.subplots()
        ax.scatter(true_values, predicted_values, color='blue', label='Predictions')
        ax.errorbar(true_values, predicted_values, xerr=uncertainty, yerr=uncertainty, fmt='o', color='red', ecolor='lightgray', capsize=5, label='Uncertainty')
        ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], '--k', label='Perfect prediction')
        ax.set_title('Predictions vs True Values with Uncertainty')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
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

#not_freeze_dict=['pred_adsb.layer.0.weight','pred_adsb.layer.0.bias','pred_adsb.layer.1.weight','pred_adsb.layer.1.bias',
#                 'pred_site.layer.0.weight','pred_site.layer.0.bias','pred_site.layer.1.weight','pred_site.layer.1.bias',]
#                  'fc_layers.0.layer.0.weight','fc_layers.0.layer.0.bias','fc_layers.0.layer.1.weight','fc_layers.0.layer.1.bias',
#                  'fc_layers.1.layer.0.weight','fc_layers.1.layer.0.bias','fc_layers.1.layer.1.weight','fc_layers.1.layer.1.bias',
#                  'fc_layers.2.layer.0.weight','fc_layers.2.layer.0.bias','fc_layers.2.layer.1.weight','fc_layers.2.layer.1.bias',
#                  'fc_atten_mlp.layer.0.weight','fc_atten_mlp.layer.0.bias','fc_atten_mlp.layer.1.weight','fc_atten_mlp.layer.1.bias',
#                  'fc_atten_bn.weight','fc_atten_bn.bias']
#def freeze_model(model, not_freeze_dict):
#    for (name,param) in model.named_parameters():
#        if name not in not_freeze_dict:
#            param.requires_grad=False
#        else:
#            pass
#    return model
#SA=freeze_model(SA,not_freeze_dict)
