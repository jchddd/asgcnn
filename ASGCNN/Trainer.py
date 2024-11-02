import torch as th
from torch import nn
import numpy as np
import copy
import random
# from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, multilabel_confusion_matrix, confusion_matrix, r2_score, mean_absolute_error, \
    mean_squared_error
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
    '''
    Trainer

    Parameters:
        - Module: the Pytorch Module
        - Dataloader_train: Dataloader for training set
        - Dataloader_valid: Dataloader for validation set
        - Dataloader_test: Dataloader for test set
        - init_lr: initial learning rate / float, default 0.1
        - metric: loss function / str, 'mae', 'mse', 'rmse', 'hyb', 'wmae'
        - metric_para: weight no hybrid loss function / float, default 0.5
        - optimizer: optimizer type / str, 'SGD', 'Momentum', 'RMSprop', 'Adam', 'AdamW'
        - scheduler: scheduler type / str, 'cos', 'step'
        - scheduler_para: length of the periodicity of the learning rate changing / int, default 1000
        - weight_decay: weight_decay on optimizer / float, default 0.0
    '''

    def __init__(self,
                 Module,
                 Dataloader_train=None,
                 Dataloader_valid=None,
                 Dataloader_test=None,
                 init_lr=0.1,
                 metric='mae',
                 metric_para=0.5,
                 optimizer='RMSprop',
                 scheduler='cos',
                 scheduler_para=1000,
                 weight_decay=0.0):
        # Module
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.Module = Module.to(self.device)
        # Dataloader
        self.Dataloader_train = Dataloader_train
        self.Dataloader_valid = Dataloader_valid
        self.Dataloader_test = Dataloader_test
        # Metric
        self.metric = metric
        self.Metric = {'mae': nn.SmoothL1Loss(beta=0.5), 'mse': nn.MSELoss(), 'rmse': RMSELoss(), 'hyb': HybLoss(metric_para), 'wmae': wMAE()}[metric].to(self.device)
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

    def train(self, epoch):
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
        with tqdm(range(epoch), leave=False, unit='e', desc='Train epoch') as pbar:
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

    def predict(self, Dataloader, to_class=False, return_uq=False):
        '''
        Function to use trained model to predict target from a Dataloader

        Parameters:
            - Dataloader: the Dataloader
            - to_class: converts the 0 and 1 Boolean characteristics of a class into class information from 1 to x
            - return_uq: whether to return uq values / bool, default False
        '''
        if not self.snapshot:
            return self._get_prediction(self.Module, Dataloader, to_class)

        elif self.snapshot and self.Module.task_typ == 'multy':
            predicts = []
            for model in self.Modules:
                predicts.append(self._get_prediction(model, Dataloader))
            predict = np.stack(predicts)
            yp1 = np.mean(predict[:, :, :5], axis=0)
            yp2 = np.mean(predict[:, :, 5:9], axis=0)
            yp3 = np.mean(predict[:, :, 9], axis=0)
            uq = np.std(predict[:, :, 9], axis=0)
            if to_class:
                _, yp1 = th.max(th.from_numpy(yp1), 1)
                yp1 = yp1.unsqueeze(1).cpu().detach().numpy()
                _, yp2 = th.max(th.from_numpy(yp2), 1)
                yp2 = yp2.unsqueeze(1).cpu().detach().numpy()
            predict = np.concatenate([yp1, yp2, yp3[:, np.newaxis]], axis=1)

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
            yp = predict
            _, yp1 = th.max(yp[:, :5], 1)
            _, yp2 = th.max(yp[:, 5:9], 1)
            yp3 = yp[:, 9]
            predict = th.cat([yp1.unsqueeze(1), yp2.unsqueeze(1), yp3.unsqueeze(1)], 1)
        return predict.cpu().detach().numpy()

    def calculate_static(self):
        static = {'train': {}, 'valid': {}, 'test': {}}
        keys = ['train', 'valid', 'test']

        if self.Module.task_typ == 'multy':
            for i, dataloader in enumerate([self.Dataloader_train, self.Dataloader_valid, self.Dataloader_test]):
                yp = th.Tensor(self.predict(dataloader))

                _, yp1 = th.max(yp[:, :5], 1)
                yp1 = yp1.cpu().detach().numpy().reshape(-1)
                y1 = dataloader.target[:, 0].cpu().detach().numpy().reshape(-1)

                _, yp2 = th.max(yp[:, 5:9], 1)
                yp2 = yp2.cpu().detach().numpy().reshape(-1)
                y2 = dataloader.target[:, 1].cpu().detach().numpy().reshape(-1)

                yp3 = yp[:, 9].cpu().detach().numpy().reshape(-1)
                y3 = dataloader.target[:, 2].cpu().detach().numpy().reshape(-1)

                y1 = np.nan_to_num(y1, nan=6)
                yp1 = np.nan_to_num(yp1, nan=6)
                y2 = np.nan_to_num(y2, nan=6)
                yp2 = np.nan_to_num(yp2, nan=6)
                y3 = np.nan_to_num(y3, nan=6)
                yp3 = np.nan_to_num(yp3, nan=6)

                static[keys[i]] = {'f1_1': f1_score(y1, yp1, average='weighted'), 'f1_2': f1_score(y2, yp2, average='weighted'),
                                   'ac_1': accuracy_score(y1, yp1), 'ac_2': accuracy_score(y2, yp2),
                                   'pc_1': precision_score(y1, yp1, average='weighted'), 'pc_2': precision_score(y2, yp2, average='weighted'),
                                   'rc_1': recall_score(y1, yp1, average='weighted'), 'rc_2': recall_score(y2, yp2, average='weighted'),
                                   'r2': r2_score(y3, yp3), 'mae': mean_absolute_error(y3, yp3),
                                   'mse': mean_squared_error(y3, yp3), 'rmse': np.sqrt(mean_squared_error(y3, yp3))}

        elif self.Module.task_typ == 'regre':
            for i, dataloader in enumerate([self.Dataloader_train, self.Dataloader_valid, self.Dataloader_test]):
                yp = th.Tensor(self.predict(dataloader)).cpu().detach().numpy().reshape(-1)
                y = dataloader.target.cpu().detach().numpy().reshape(-1)
                y = y.reshape(-1, 1)
                yp = yp.reshape(-1, 1)
                static[keys[i]] = {'r2': r2_score(y, yp), 'mae': mean_absolute_error(y, yp),
                                   'mse': mean_squared_error(y, yp), 'rmse': np.sqrt(mean_squared_error(y, yp))}

        return static

    def confusion_matrix(self, dataloader):
        Dataloader = {'train': self.Dataloader_train, 'valid': self.Dataloader_valid, 'test': self.Dataloader_test}[dataloader]
        yp = th.Tensor(self.predict(Dataloader))

        _, yp1 = th.max(yp[:, :5], 1)
        yp1 = yp1.cpu().detach().numpy().reshape(-1)
        y1 = Dataloader.target[:, 0].cpu().detach().numpy().reshape(-1)

        _, yp2 = th.max(yp[:, 5:9], 1)
        yp2 = yp2.cpu().detach().numpy().reshape(-1)
        y2 = Dataloader.target[:, 1].cpu().detach().numpy().reshape(-1)

        cm1 = confusion_matrix(y1, yp1)
        cm2 = confusion_matrix(y2, yp2)
        mcm1 = multilabel_confusion_matrix(y1, yp1)
        mcm2 = multilabel_confusion_matrix(y2, yp2)

        return cm1, cm2, mcm1, mcm2

    def show_learn_curve(self):
        plt.figure(figsize=(6, 4))
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.plot(range(len(self.losses_train)), self.losses_train, 'y', label='train', zorder=2)
        plt.plot(range(len(self.losses_valid)), self.losses_valid, 'b', label='valid', zorder=1)
        plt.legend()
        plt.show()

    def show_result(self):
        # init figure
        plt.figure(figsize=(6, 6))
        plt.xlabel('DFT', fontsize=16)
        plt.ylabel('predict', fontsize=16)
        plt.plot([-10, 10], [-10, 10])
        plt.axis([-4, 4, -4, 4])
        # init
        print_name = ['train:', 'valid', 'test']
        colors = ['y', 'b', 'r']
        # loop dataloader
        for i, dataloader in enumerate([self.Dataloader_train, self.Dataloader_valid, self.Dataloader_test]):
            # get yp
            if self.Module.task_typ == 'regre':
                yp = self.predict(dataloader)
                y = dataloader.target.cpu().detach().numpy().reshape(-1)
            elif self.Module.task_typ == 'multy':
                yp = self.predict(dataloader, True)[:, 2]
                y = dataloader.target[:, 2].cpu().detach().numpy().reshape(-1)
            # cal static
            plt.scatter(6, 6, c=colors[i], label=print_name[i])
            # plot scatter
            plt.scatter(y, yp, c=colors[i])
        plt.legend()
        plt.show()
    
    def load_pretrained(self, location='cuda'):
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
    def __init__(self):
        super(wMAE, self).__init__()
        self.mae = nn.SmoothL1Loss(reduction='none', beta=0.5)

    def forward(self, yp, y, w=None):
        loss = self.mae(yp, y)
        if w is not None:
            loss = th.mean(loss * w)
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
    def __init__(self, weight=0.5):
        super(HybLoss, self).__init__()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.w1 = th.Tensor([weight]).to(self.device)
        self.w2 = th.Tensor([1 - weight]).to(self.device)
        self.cel1 = nn.CrossEntropyLoss()
        self.cel2 = nn.CrossEntropyLoss()
        self.mae = nn.SmoothL1Loss(reduction='none', beta=0.5)

    def forward(self, yp, y, w=None):
        loss1 = self.cel1(yp[:, :5], y[:, 0].long().squeeze())
        loss2 = self.cel2(yp[:, 5:9], y[:, 1].long().squeeze())
        loss3 = self.mae(yp[:, 9], y[:, 2])
        if w is None:
            loss3 = th.mean(loss3)
        else:
            loss3 = th.mean(loss3 * w)
        loss = self.w1 * (loss1 + loss2) + self.w2 * loss3
        return loss

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

#import hyperopt
#from hyperopt import hp, fmin, tpe, Trials, partial
#from hyperopt.early_stop import no_progress_loss
#
#search_params={
#    'conv_num': hp.choice('conv_num',[1,2,3,4,5]),
#    'adsb_eml': hp.quniform('adsb_eml',60,201,10),
#    'slab_eml': hp.quniform('slab_eml',60,201,10),
#    'fcly_1em': hp.quniform('fcly_1em',60,201,10),
#    'fcly_2em': hp.quniform('fcly_2em',10,161,10),
#    'fcly_3em': hp.quniform('fcly_3em',10,101,10),
#    'loss_w': hp.uniform('loss_w',0.001,1),
#}
#
#def run_opt_in_para_set(params):
#    setup_seed(66666)
#    Loader_train, Loader_valid, Loader_test = load_data()
#    
#    Loader_train.batch_size=256
#    fcl = [int(params['fcly_1em']),int(params['fcly_2em']),int(params['fcly_3em'])]
#    SA = AGCNN(101,6,int(params['adsb_eml']),101,8,int(params['slab_eml']),params['conv_num'],fcl,'Ce')
#    
#    t=Trainer_MC(SA, Loader_train, Loader_valid, Loader_test, 0.01, 'AdamW', 'cos', 200,0.001)
#    t.Metric = MulLoss_MC(1-params['loss_w'], params['loss_w'])
#    t.train_(50)
#    
#    return t.calculate_static()['valid']['mae']
#
#def param_hyperopt(max_evals=66):
#    trials=Trials()
#    
#    early_stop=no_progress_loss(100)
#    
#    params_best= fmin(run_opt_in_para_set,
#                    space=search_params,
#                    algo=tpe.suggest,
#                    max_evals=max_evals,
#                    verbose=True,
#                    trials=trials,
#                    early_stop_fn=early_stop)
#    return params_best, trials
#
#param_hyperopt(360)
