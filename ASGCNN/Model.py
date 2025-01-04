import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling
from dgl.nn.functional import edge_softmax


class CGCNN(nn.Module):
    def __init__(self,
                 node_feat_length,
                 edge_feat_length,
                 embed_feat_length,
                 angle_feat_length=0,
                 conv_num=3,
                 conv_typ='CGConv',
                 pool_typ='avg',
                 att=False,
                 fcl_dims=[10],
                 mlp_acti='silu',
                 p_droput=0.,
                 task_typ='regre',
                 class_dim=[5, 4],
                 regre_dim=[1],
                 use_global_feat=False,
                 global_feat_place='pool',
                 global_feat_dim=6,
                 return_vp=False):
        """
        CGCNN model

        Parameters:
            - node_feat_length / int
            - edge_feat_length / int
            - embed_feat_length: node and edge features are converted to features of the same dimension by embedding layers / int
            - angle_feat_length / int
            - conv_num: number of convolution layers / int, default 3. Turn to list when line graph is used.
            - conv_typ: choose convolution layers from 'CGConv', 'EGate', 'EAtt' / str, default 'CGConv'. Turn to list when line graph is used.
            - pool_typ: pooling function type from 'avg', 'sum', 'cluster' / str, default 'avg'
            - att: whether to add a self-attention layer after the pooling function / bool, default False
            - fcl_dims: dimensions of fully connected layer / list, or list of list, default [10]
              A single list means that all target predictions pass through the same fully connected layer, e.g. [30, 40, 20]
              Multiple lists represent a different full connection layer for each target prediction task, e.g. [[20, 10], [40]]
              An empty list can be passed in to represent the target predicted directly by the pooling feature, e.g. [], [[20, 10], []]
            - mlp_acti: activation function used in the perceptron / str, default 'silu'
            - p_droput: dropout rate at each layer on fully connected layers / float, default 0.
            - task_typ: task types choosen from 'regre' and 'multy',  / str, default 'regre'
              If 'regre', classification task-related command will not work, 'multy' means performing regression and classification tasks together
            - class_dim: the number of categories for each classification task / list, default [5, 4]
            - regre_dim: the dimensions of the predicted values for each set of regression tasks / list, default [1]
              If there is only one fully connected layer, it does not make sense to separate the regression tasks
              However, when there are multiple fully connected layers, each regression target quantity can be designed separately
            - use_global_feat: whether to use global features in the model / bool, default False
              Need to add global features to 'graph' and Data_Loader
            - global_feat_place: where to cat the global feature, chosen from 'pool' or 'linear' (feature before predict) / str, default 'pool'
              If add to 'linear' and fully connected layer is [], this feature addition will be ignored
            - global_feat_dim: feature length of global feature / int, default 6
            - return_vp: whether to return the pooling vector / bool, default False
        """
        super().__init__()
        # init
        self.name = 'CGCNN'
        self.task_typ = task_typ
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        # embedding
        self.node_embedding = nn.Linear(node_feat_length, embed_feat_length)
        self.edge_embedding = nn.Linear(edge_feat_length, embed_feat_length)
        # line_graph
        self.use_line_graph = False
        if   angle_feat_length > 0:
            self.use_line_graph = True
            self.angle_embedding = nn.Linear(angle_feat_length, embed_feat_length)
            conv_typ_ali = conv_typ[0]
            conv_num_ali = conv_num[0]
            self.conv_num_ali = conv_num_ali
            if   conv_typ_ali == 'CGConv':
                self.conv_layers_ali_g  = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length, mlp_acti) for _ in range(conv_num_ali)])
                self.conv_layers_ali_lg = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length, mlp_acti) for _ in range(conv_num_ali)])
            elif conv_typ_ali == 'EGate':
                self.conv_layers_ali_g  = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length, mlp_acti) for _ in range(conv_num_ali)])
                self.conv_layers_ali_lg = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length, mlp_acti) for _ in range(conv_num_ali)])
            elif conv_typ_ali == 'EAtt':
                self.conv_layers_ali_g  = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length, mlp_acti) for _ in range(conv_num_ali)])
                self.conv_layers_ali_lg = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length, mlp_acti) for _ in range(conv_num_ali)])
            conv_typ = conv_typ[1]
            conv_num = conv_num[1]
        elif angle_feat_length == 0:
            conv_typ = conv_typ
            conv_num = conv_num
        # convolution
        self.conv_typ = conv_typ
        if   conv_typ == 'CGConv':
            self.conv_layers = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length, mlp_acti) for _ in range(conv_num)])
        elif conv_typ == 'MGConv':
            self.conv_layers = nn.ModuleList([ConvFunc_MGENet(embed_feat_length, mlp_acti) for _ in range(conv_num)])
        elif conv_typ == 'EGate':
            self.conv_layers = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length, mlp_acti) for _ in range(conv_num)])
        elif conv_typ == 'EAtt':
            self.conv_layers = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length, mlp_acti) for _ in range(conv_num)])
        # pooling
        if   pool_typ == 'avg':
            self.pool = AvgPooling()
        elif pool_typ == 'sum':
            self.pool = SumPooling()
        elif pool_typ == 'cluster':
            self.pool = Cluster_Pooling()
        pool_vector_dim = embed_feat_length
        # self Attention
        self.att = att
        self.self_att = nn.Sequential(MLPLayer(embed_feat_length, 2 * embed_feat_length, mlp_acti), 
                             MLPLayer(2 * embed_feat_length, embed_feat_length, 'softmax'))
        # global_feat
        self.use_global_feat = use_global_feat
        self.global_feat_place = global_feat_place
        if use_global_feat:
            self.glob_embedding = nn.Linear(global_feat_dim, embed_feat_length, mlp_acti)
            if global_feat_place == 'pool':
                pool_vector_dim = embed_feat_length + embed_feat_length
        # fully connect layers
        ## one 
        if   len(fcl_dims) == 0:
            self.one_fcl = True
            self.fc_layers = None
            fcl_dims = [pool_vector_dim]
        elif type(fcl_dims[0]) == int:
            self.one_fcl = True
            fcl_dims = [pool_vector_dim] + fcl_dims
            self.fc_layers = nn.Sequential(*[MLPLayer(fcl_dims[_], fcl_dims[_ + 1], mlp_acti, p_droput) for _ in range(len(fcl_dims) - 1)])
        ## multy
        elif type(fcl_dims[0]) == list:
            self.one_fcl = False
            for fi, fcl_dims_i in enumerate(fcl_dims):
                fcl_dims[fi] = [pool_vector_dim] + fcl_dims_i
                if   len(fcl_dims[fi])  > 1:
                    setattr(self, 'fc_layers_' + str(fi), nn.Sequential(*[MLPLayer(fcl_dims[fi][_], fcl_dims[fi][_ + 1], mlp_acti, p_droput) for _ in range(len(fcl_dims[fi]) - 1)]))
                elif len(fcl_dims[fi]) == 1:
                    setattr(self, 'fc_layers_' + str(fi), None)
        # target dimension
        if   task_typ == 'regre':
            target_dims = regre_dim
            target_typs = ['r' for _ in range(len(regre_dim))]
            self.task_number = len(regre_dim)
        elif task_typ == 'multy':
            target_dims = class_dim + regre_dim
            target_typs = ['c' for _ in range(len(class_dim))] + ['r' for _ in range(len(regre_dim))]
            self.task_number = len(regre_dim) + len(class_dim)
        # get fcl output
        ## one fcl
        if   self.one_fcl:
            if   len(fcl_dims)  > 1:
                fcl_output_dim = fcl_dims[-1] + embed_global_feat_dim if (self.use_global_feat and global_feat_place == 'linear') else fcl_dims[-1]
            elif len(fcl_dims) == 1:
                fcl_output_dim = fcl_dims[-1]
            fcl_output_dims = [fcl_output_dim] * self.task_number
        ## multy fcl
        elif not self.one_fcl:
            fcl_output_dims = []
            for fi, fcl_dims_i in enumerate(fcl_dims):
                if   len(fcl_dims_i)  > 1:
                    fcl_output_dim = fcl_dims_i[-1] + embed_global_feat_dim if (self.use_global_feat and global_feat_place == 'linear') else fcl_dims_i[-1]
                elif len(fcl_dims_i) == 1:
                    fcl_output_dim = pool_vector_dim
                fcl_output_dims.append(fcl_output_dim)
        # predict target
        for ti in range(self.task_number):
            if   target_typs[ti] == 'c':
                setattr(self, 'pred_target_' + str(ti), nn.Sequential(nn.Linearnn.Linear(fcl_output_dims[ti], target_dims[ti]) ,get_activation('softmax')))
            elif target_typs[ti] == 'r':
                setattr(self, 'pred_target_' + str(ti), nn.Linear(fcl_output_dims[ti], target_dims[ti]))
        # return graph feature vector
        self.return_vp = return_vp

    def forward(self, bg):
        # init
        if self.use_line_graph:
            g  = bg[0]
            lg = bg[1]
        else:
            g = bg
        g = g.local_var()
        v = g.ndata.pop('h_v')
        e = g.edata.pop('h_e')
        f = g.global_feat if self.use_global_feat else None
        # embedding
        v = self.node_embedding(v)
        e = self.edge_embedding(e)
        f = self.glob_embedding(f) if self.use_global_feat else None
        # line graph
        if self.use_line_graph:
            lg = lg.local_var()
            a = lg.edata.pop('h_le')
            a = self.angle_embedding(a)
            for ic in range(self.conv_num_ali):
                e, a = self.conv_layers_ali_lg[ic](lg, e, a)
                v, e = self.conv_layers_ali_g[ic](g, v, e)
        # convolution
        for con_layer in self.conv_layers:
            if self.conv_typ == 'MGConv':
                v, e, f = con_layer(g, v, e, f)
            else:
                v, e = con_layer(g, v, e)
        # pooling
        vg = self.pool(g, v)
        vp = vg
        # self attention 
        if self.att:
            vg = self.self_att(vg) * vg
        # global feat at pool
        vg = torch.cat([vg, f], dim=1) if self.use_global_feat and self.global_feat_place == 'pool' else vg
        # fully connect layers
        if self.one_fcl and self.fc_layers is not None:
            vg = self.fc_layers(vg)
            if self.use_global_feat and self.global_feat_place == 'linear':
                vg = torch.cat([vg, f], dim=1)
        # predict target
        targets = []
        for ti in range(self.task_number):
            if       self.one_fcl:
                fcl_out = vg
            elif not self.one_fcl:
                fcl_layers = getattr(self, 'fc_layers_' + str(ti))
                if   fcl_layers is not None:
                    fcl_out = fcl_layers(vg)
                    if self.use_global_feat and self.global_feat_place == 'linear':
                        fcl_out = torch.cat([fcl_out, f], dim=1)
                elif fcl_layers is None:
                    fcl_out = vg
            targets.append(getattr(self, 'pred_target_' + str(ti))(fcl_out))
        target = torch.cat(targets, dim=1)
        # return graph feature vector
        if self.return_vp:
            return target, vp
        else:
            return target 


class ASGCNN(nn.Module):
    def __init__(self,
                 node_feat_length_adsb,
                 edge_feat_length_adsb,
                 embed_feat_length_adsb,
                 node_feat_length_slab,
                 edge_feat_length_slab,
                 embed_feat_length_slab,
                 angle_feat_length_adsb=0,
                 angle_feat_length_slab=0,
                 conv_num_adsb=3,
                 conv_typ_adsb='CGConv',
                 pool_typ_adsb='avg',
                 conv_num_slab=3,
                 conv_typ_slab='CGConv',
                 pool_typ_slab='avg',
                 att=False,
                 fcl_dims=[10],
                 mlp_acti='silu',
                 p_droput=0.,
                 task_typ='regre',
                 class_dim=[5, 4],
                 regre_dim=[1],
                 use_global_feat=False,
                 global_feat_place='pool',
                 global_feat_dim=6,
                 return_vp=False):
        """
        ASGCNN model

        Parameters:
            - node_feat_length_adsb: dimensions of node features on the first (adsorbate-site) graph / int
            - edge_feat_length_adsb: dimensions of edge features on the first (adsorbate-site) graph / int
            - embed_feat_length_adsb: dimensions of embedding features on the first (adsorbate-site) graph / int
            - angle_feat_length_adsb: dimensions of angle features on the first (adsorbate-site) graph / int
            - node_feat_length_slab: dimensions of node features on the second (slab) graph / int
            - edge_feat_length_slab: dimensions of edge features on the second (slab) graph / int
            - embed_feat_length_slab: dimensions of embedding features on the second (slab) graph / int
            - angle_feat_length_slab: dimensions of angle features on the secomd (slab) graph / int
            - conv_number: number of convolution layers / int, default 3. Turn to list when line graph is used.
            - conv_type: choose convolution layers from 'CGConv', 'EGate', 'EAtt' / str, default 'CGConv'. Turn to list when line graph is used.
            - pool_typ: pooling function type from 'avg', 'sum', 'cluster' / str, default 'avg'
            - att: whether to add a self-attention layer after the pooling function / bool, default False
            - fcl_dims: dimensions of fully connected layer / list, or list of list, default [10]
              A single list means that all target predictions pass through the same fully connected layer, e.g. [30, 40, 20]
              Multiple lists represent a different full connection layer for each target prediction task, e.g. [[20, 10], [40]]
              An empty list can be passed in to represent the target predicted directly by the pooling feature, e.g. [], [[20, 10], []]
            - mlp_acti: activation function used in the perceptron / str, default 'silu'
            - p_droput: dropout rate at each layer on fully connected layers / float, default 0.
            - task_typ: task types choosen from 'regre' and 'multy',  / str, default 'regre'
              If 'regre', classification task-related command will not work, 'multy' means performing regression and classification tasks together
            - class_dim: the number of categories for each classification task / list, default [5, 4]
            - regre_dim: the dimensions of the predicted values for each set of regression tasks / list, default [1]
              If there is only one fully connected layer, it does not make sense to separate the regression tasks
              However, when there are multiple fully connected layers, each regression target quantity can be designed separately
            - use_global_feat: whether to use global features in the model / bool, default False
              Need to add global features to 'graph' and Data_Loader
              In this model, global features are loaded on the second (slab) graph by default
            - global_feat_place: where to cat the global feature, chosen from 'pool' or 'linear' (feature before predict) / str, default 'pool'
              If add to 'linear' and fully connected layer is [], this feature addition will be ignored
            - global_feat_dim: feature length of global feature / int, default 6
            - return_vp: whether to return the pooling vector / bool, default False
        """
        super().__init__()
        # init
        self.name = 'ASGCNN'
        self.task_typ = task_typ
        self.model_args = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
        # embedding
        self.node_embedding_adsb = nn.Linear(node_feat_length_adsb, embed_feat_length_adsb)
        self.edge_embedding_adsb = nn.Linear(edge_feat_length_adsb, embed_feat_length_adsb)
        self.node_embedding_slab = nn.Linear(node_feat_length_slab, embed_feat_length_slab)
        self.edge_embedding_slab = nn.Linear(edge_feat_length_slab, embed_feat_length_slab)
        # line_graph
        ## adsb
        self.use_line_graph_adsb = False
        if   angle_feat_length_adsb > 0:
            self.use_line_graph_adsb = True
            conv_typ_ali_adsb = conv_typ_adsb[0]
            conv_num_ali_adsb = conv_num_adsb[0]
            self.conv_num_ali_adsb = conv_num_ali_adsb
            self.angle_embedding_adsb = nn.Linear(angle_feat_length_adsb, embed_feat_length_adsb)
            if   conv_typ_ali_adsb == 'CGConv':
                self.conv_layers_ali_g_adsb  = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_ali_adsb)])
                self.conv_layers_ali_lg_adsb = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_ali_adsb)])
            elif conv_typ_ali_adsb == 'EGate':
                self.conv_layers_ali_g_adsb  = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_ali_adsb)])
                self.conv_layers_ali_lg_adsb = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_ali_adsb)])
            elif conv_typ_ali_adsb == 'EAtt':
                self.conv_layers_ali_g_adsb  = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_ali_adsb)])
                self.conv_layers_ali_lg_adsb = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_ali_adsb)])
            conv_typ_adsb = conv_typ_adsb[1]
            conv_num_adsb = conv_num_adsb[1]
        elif angle_feat_length_adsb == 0:
            conv_typ_adsb = conv_typ_adsb
            conv_num_adsb = conv_num_adsb
        ## slab
        self.use_line_graph_slab = False
        if   angle_feat_length_slab > 0:
            self.use_line_graph_slab = True
            conv_typ_ali_slab = conv_typ_slab[0]
            conv_num_ali_slab = conv_num_slab[0]
            self.conv_num_ali_slab = conv_num_ali_slab
            self.angle_embedding_slab = nn.Linear(angle_feat_length_slab, embed_feat_length_slab)
            if   conv_typ_ali_slab == 'CGConv':
                self.conv_layers_ali_g_slab  = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_ali_slab)])
                self.conv_layers_ali_lg_slab = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_ali_slab)])
            elif conv_typ_ali_slab == 'EGate':
                self.conv_layers_ali_g_slab  = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_ali_slab)])
                self.conv_layers_ali_lg_slab = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_ali_slab)])
            elif conv_typ_ali_slab == 'EAtt':
                self.conv_layers_ali_g_slab  = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_ali_slab)])
                self.conv_layers_ali_lg_slab = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_ali_slab)])
            conv_typ_slab = conv_typ_slab[1]
            conv_num_slab = conv_num_slab[1]
        elif angle_feat_length_slab == 0:
            conv_typ_slab = conv_typ_slab
            conv_num_slab = conv_num_slab
        # convolution
        ## adsb
        if   conv_typ_adsb == 'CGConv':
            self.conv_layers_adsb = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_adsb)])
        elif conv_typ_adsb == 'EGate':
            self.conv_layers_adsb = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_adsb)])
        elif conv_typ_adsb == 'EAtt':
            self.conv_layers_adsb = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length_adsb, mlp_acti) for _ in range(conv_num_adsb)])
        ## slab
        self.conv_typ_slab = conv_typ_slab
        if   conv_typ_slab == 'CGConv':
            self.conv_layers_slab = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_slab)])
        elif conv_typ_slab == 'MGConv':
            self.conv_layers_slab = nn.ModuleList([ConvFunc_MGENet(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_slab)])
        elif conv_typ_slab == 'EGate':
            self.conv_layers_slab = nn.ModuleList([Edge_Gate_Convolution(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_slab)])
        elif conv_typ_slab == 'EAtt':
            self.conv_layers_slab = nn.ModuleList([Edge_Attn_Convolution(embed_feat_length_slab, mlp_acti) for _ in range(conv_num_slab)])
        # pooling
        ## adsb
        if   pool_typ_adsb == 'avg':
            self.pool_adsb = AvgPooling()
        elif pool_typ_adsb == 'sum':
            self.pool_adsb = SumPooling()
        elif pool_typ_adsb == 'cluster':
            self.pool_adsb = Cluster_Pooling()
        ## slab
        if   pool_typ_slab == 'avg':
            self.pool_slab = AvgPooling()
        elif pool_typ_slab == 'sum':
            self.pool_slab = SumPooling()
        elif pool_typ_slab == 'cluster':
            self.pool_slab = Cluster_Pooling()
        pool_vector_dim = embed_feat_length_adsb + embed_feat_length_slab
        # self Attention
        self.att = att
        self.self_att = nn.Sequential(MLPLayer(pool_vector_dim, 2 * pool_vector_dim, mlp_acti), 
                             MLPLayer(2 * pool_vector_dim, pool_vector_dim, 'softmax'))
        # global_feat
        self.use_global_feat = use_global_feat
        self.global_feat_place = global_feat_place
        if use_global_feat:
            self.glob_embedding = nn.Linear(global_feat_dim, embed_feat_length_slab, mlp_acti)
            if global_feat_place == 'pool':
                pool_vector_dim = pool_vector_dim + embed_feat_length_slab
        # fully connect layers
        ## one 
        if   len(fcl_dims) == 0:
            self.one_fcl = True
            self.fc_layers = None
            fcl_dims = [pool_vector_dim]
        elif type(fcl_dims[0]) == int:
            self.one_fcl = True
            fcl_dims = [pool_vector_dim] + fcl_dims
            self.fc_layers = nn.Sequential(*[MLPLayer(fcl_dims[_], fcl_dims[_ + 1], mlp_acti, p_droput) for _ in range(len(fcl_dims) - 1)])
        ## multy
        elif type(fcl_dims[0]) == list:
            self.one_fcl = False
            for fi, fcl_dims_i in enumerate(fcl_dims):
                fcl_dims[fi] = [pool_vector_dim] + fcl_dims_i
                if   len(fcl_dims[fi])  > 1:
                    setattr(self, 'fc_layers_' + str(fi), nn.Sequential(*[MLPLayer(fcl_dims[fi][_], fcl_dims[fi][_ + 1], mlp_acti, p_droput) for _ in range(len(fcl_dims[fi]) - 1)]))
                elif len(fcl_dims[fi]) == 1:
                    setattr(self, 'fc_layers_' + str(fi), None)
        # target dimension
        if   task_typ == 'regre':
            target_dims = regre_dim
            target_typs = ['r' for _ in range(len(regre_dim))]
            self.task_number = len(regre_dim)
        elif task_typ == 'multy':
            target_dims = class_dim + regre_dim
            target_typs = ['c' for _ in range(len(class_dim))] + ['r' for _ in range(len(regre_dim))]
            self.task_number = len(regre_dim) + len(class_dim)
        # get fcl output
        ## one fcl
        if   self.one_fcl:
            if   len(fcl_dims)  > 1:
                fcl_output_dim = fcl_dims[-1] + embed_feat_length_slab if (self.use_global_feat and global_feat_place == 'linear') else fcl_dims[-1]
            elif len(fcl_dims) == 1:
                fcl_output_dim = fcl_dims[-1]
            fcl_output_dims = [fcl_output_dim] * self.task_number
        ## multy fcl
        elif not self.one_fcl:
            fcl_output_dims = []
            for fi, fcl_dims_i in enumerate(fcl_dims):
                if   len(fcl_dims_i)  > 1:
                    fcl_output_dim = fcl_dims_i[-1] + embed_feat_length_slab if (self.use_global_feat and global_feat_place == 'linear') else fcl_dims_i[-1]
                elif len(fcl_dims_i) == 1:
                    fcl_output_dim = pool_vector_dim
                fcl_output_dims.append(fcl_output_dim)
        # predict target
        for ti in range(self.task_number):
            if   target_typs[ti] == 'c':
                setattr(self, 'pred_target_' + str(ti), nn.Sequential(nn.Linear(fcl_output_dims[ti], target_dims[ti]) ,get_activation('softmax')))
            elif target_typs[ti] == 'r':
                setattr(self, 'pred_target_' + str(ti), nn.Linear(fcl_output_dims[ti], target_dims[ti]))
        # return graph feature vector
        self.return_vp = return_vp

    def forward(self, bga, bgs):
        if self.use_line_graph_adsb or self.use_line_graph_slab:
            ga  = bga[0]
            lga = bga[1]
            gs  = bgs[0]
            lgs = bgs[1]
        else:
            ga = bga
            gs = bgs
        # adsb init
        ga = ga.local_var()
        va = ga.ndata.pop('h_v')
        ea = ga.edata.pop('h_e')
        # adsb embedding
        va = self.node_embedding_adsb(va)
        ea = self.edge_embedding_adsb(ea)
        # adsb line graph
        if self.use_line_graph_adsb:
            lga = lga.local_var()
            aa = lga.edata.pop('h_le')
            aa = self.angle_embedding_adsb(aa)
            for ic in range(self.conv_num_ali_adsb):
                ea, aa = self.conv_layers_ali_lg_adsb[ic](lga, ea, aa)
                va, ea = self.conv_layers_ali_g_adsb[ic](ga, va, ea)
        # adsb convolution
        for conv_layer in self.conv_layers_adsb:
            va, ea = conv_layer(ga, va, ea)
        # adsb pooling
        vap = self.pool_adsb(ga, va)
        # slab init
        gs = gs.local_var()
        vs = gs.ndata.pop('h_v')
        es = gs.edata.pop('h_e')
        gf = gs.global_feat if self.use_global_feat else None
        # slab embedding
        vs = self.node_embedding_slab(vs)
        es = self.edge_embedding_slab(es)
        gf = self.glob_embedding(gf) if self.use_global_feat else None
        # slab ling graph
        if self.use_line_graph_slab:
            lgs = lgs.local_var()
            ass = lgs.edata.pop('h_le')
            ass = self.angle_embedding_slab(ass)
            for ic in range(self.conv_num_ali_slab):
                es, ass = self.conv_layers_ali_lg_slab[ic](lgs, es, ass)
                vs, es = self.conv_layers_ali_g_slab[ic](gs, vs, es)
        # slab convolution
        for conv_layer in self.conv_layers_slab:
            if self.conv_typ_slab == 'MGConv':
                vs, es, gf = conv_layer(gs, vs, es, gf)
            else:
                vs, es = conv_layer(gs, vs, es)
        # slab pooling
        vsp = self.pool_slab(gs, vs)
        # cat graph features
        vt = torch.cat([vap, vsp], dim=1)
        vp = vt
        # self attention 
        if self.att:
            vt = self.self_att(vt) * vt
        # global feat at pool
        vt = torch.cat([vt, gf], dim=1) if self.use_global_feat and self.global_feat_place == 'pool' else vt
        # fully connect layers
        if self.one_fcl and self.fc_layers is not None:
            vt = self.fc_layers(vt)
            if self.use_global_feat and self.global_feat_place == 'linear':
                vt = torch.cat([vt, gf], dim=1)
        # predict target
        targets = []
        for ti in range(self.task_number):
            if       self.one_fcl:
                fcl_out = vt
            elif not self.one_fcl:
                fcl_layers = getattr(self, 'fc_layers_' + str(ti))
                if   fcl_layers is not None:
                    fcl_out = fcl_layers(vt)
                    if self.use_global_feat and self.global_feat_place == 'linear':
                        fcl_out = torch.cat([fcl_out, gf], dim=1)
                elif fcl_layers is None:
                    fcl_out = vt
            targets.append(getattr(self, 'pred_target_' + str(ti))(fcl_out))
        target = torch.cat(targets, dim=1)
        # return graph feature vector
        if self.return_vp:
            return target, vp
        else:
            return target  


class ASGCNN_pretrain(nn.Module):
    def __init__(self,
                 node_feat_length_adsb=101,
                 edge_feat_length_adsb=6,
                 embedding_length_adsb=110,
                 node_feat_length_slab=101,
                 edge_feat_length_slab=8,
                 embedding_length_slab=150,
                 conv_number=3,
                 fc_lens=[80,120,10],
                 mlp_acti='silu',
                 task_typ='multy',
                 conv_type='Ce',
                 p_droput=0,
                 ):
        super().__init__()
        # init
        self.name = 'ASGCNN'
        self.task_typ = task_typ
        # nn adsb
        self.node_embedding_adsb = MLPLayer(node_feat_length_adsb, embedding_length_adsb, mlp_acti)
        if conv_type == 'C':
            self.conv_layers_adsb = nn.ModuleList([ConvFunc_CGCNN(embedding_length_adsb, edge_feat_length_adsb) for _ in range(conv_number)])
        elif conv_type == 'Ce':
            self.conv_layers_adsb = nn.ModuleList([ConvFunc_CGCNN_edgeMLP(embedding_length_adsb, edge_feat_length_adsb) for _ in range(conv_number)])
        self.aver_pool_adsb = AvgPooling()
        # nn slab
        self.node_embedding_slab = MLPLayer(node_feat_length_slab, embedding_length_slab, mlp_acti)
        if conv_type == 'C':
            self.conv_layers_slab = nn.ModuleList([ConvFunc_CGCNN(embedding_length_slab, edge_feat_length_slab) for _ in range(conv_number)])
        elif conv_type == 'Ce':
            self.conv_layers_slab = nn.ModuleList([ConvFunc_CGCNN_edgeMLP(embedding_length_slab, edge_feat_length_slab) for _ in range(conv_number)])
        self.aver_pool_slab = AvgPooling()
        # nn FCLs
        fc_lens = [embedding_length_adsb + embedding_length_slab] + fc_lens
        self.fc_layers = nn.ModuleList([MLPLayer(fc_lens[_], fc_lens[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens) - 1)])
        # nn attention
        self.fc_atten_mlp = MLPLayer(embedding_length_adsb + embedding_length_slab, embedding_length_adsb + embedding_length_slab, 'elu')
        self.fc_atten_bn = nn.BatchNorm1d(embedding_length_adsb + embedding_length_slab)
        # nn target
        self.pred_adsb = MLPLayer(fc_lens[-1], 5, 'softmax')
        self.pred_site = MLPLayer(fc_lens[-1], 4, 'softmax')
        self.pred_target = nn.Linear(fc_lens[-1], 1)
        # nn vector
        self.return_vt = False

    def forward(self, ga, gs):
        # adsb init
        ga = ga.local_var()
        va = ga.ndata.pop('h_v')
        ea = ga.edata.pop('h_e')
        va = self.node_embedding_adsb(va)
        # adsb convolution
        for conv_layer in self.conv_layers_adsb:
            va = conv_layer(ga, va, ea)
        # adsb pooling
        va_s = self.aver_pool_adsb(ga, va)
        # slab init
        gs = gs.local_var()
        vs = gs.ndata.pop('h_v')
        es = gs.edata.pop('h_e')
        vs = self.node_embedding_slab(vs)
        # slab convolution
        for conv_layer in self.conv_layers_slab:
            vs = conv_layer(gs, vs, es)
        # slab pooling
        vs_s = self.aver_pool_slab(gs, vs)
        # attention
        vt = torch.cat([va_s, vs_s], dim=1)
        vt = self.fc_atten_bn(self.fc_atten_mlp(vt) * vt)
        # FCLs
        for fc_layer in self.fc_layers:
            vt = fc_layer(vt)
        # target
        if self.task_typ == 'regre':
            target = self.pred_target(vt)
        elif self.task_typ == 'multy':
            Cadsb = self.pred_adsb(vt)
            Csite = self.pred_site(vt)
            target = self.pred_target(vt)
            target = torch.cat([Cadsb, Csite, target], dim=1)
        if self.return_vt:
            return target, vt
        else:
            return target


class ConvFunc_CGCNN(nn.Module):
    def __init__(self, feature_length, mlpt_at='sigmoid', p_droput=0,  gate_at='softplus'):
        super().__init__()
        self.mlpt = MLPLayer(3 * feature_length, feature_length, mlpt_at, p_droput)
        self.gate = MLPLayer(3 * feature_length, feature_length, gate_at, p_droput)
        self.node_bn = nn.BatchNorm1d(feature_length)
        self.outp_at = get_activation(mlpt_at)

    def feats_cat_in_edge(self, edges):
        h = torch.cat((edges.src["h"], edges.dst["h"], edges.data["h"]), dim=1)
        return {"h_cat": h}

    def forward(self, g, node_feats, edge_feats):
        # init
        g = g.local_var()
        g.ndata['h'] = node_feats
        g.edata['h'] = edge_feats
        # prepare conv, cat node and edge
        g.apply_edges(self.feats_cat_in_edge)
        h_cat = g.edata.pop('h_cat')
        # conv
        h_mlpt = self.mlpt(h_cat)
        h_gate = self.gate(h_cat)
        h_node_update = h_mlpt * h_gate
        g.edata['h_node_update'] = h_node_update
        # update node
        g.update_all(
            fn.copy_e("h_node_update", "h_node_update"),
            fn.sum("h_node_update", "h_node_update")
        )
        return self.outp_at(self.node_bn(g.ndata['h_node_update']) + node_feats), edge_feats

    
class ConvFunc_MGENet(nn.Module):
    def __init__(self, feature_length, mlp_acti='softplus'):
        super().__init__()
        self.edge_update = nn.Sequential(*[MLPLayer(feature_length * 4, feature_length * 4, mlp_acti), 
                                           MLPLayer(feature_length * 4, feature_length, mlp_acti)])
        self.node_update = nn.Sequential(*[MLPLayer(feature_length * 3, feature_length * 3, mlp_acti), 
                                           MLPLayer(feature_length * 3, feature_length, mlp_acti)])
        self.glob_update = nn.Sequential(*[MLPLayer(feature_length * 3, feature_length * 3, mlp_acti), 
                                           MLPLayer(feature_length * 3, feature_length, mlp_acti)])
        self.aver_pool = AvgPooling()
        
    def feats_cat_in_edge(self, edges):
        h = torch.cat((edges.src["h"], edges.dst["h"], edges.data["h"]), dim=1)
        return {"h_cat": h}

    def forward(self, g, node_feats, edge_feats, global_feats):
        # init
        g = g.local_var()
        g.ndata['h'] = node_feats
        g.edata['h'] = edge_feats
        g.global_feats = global_feats
        batch_num_nodes = g.batch_num_nodes()
        batch_num_edges = g.batch_num_edges()
        # edge update
        g.apply_edges(self.feats_cat_in_edge)
        global_edges = torch.repeat_interleave(global_feats, batch_num_edges, dim=0)
        h_cat_edge = torch.cat([g.edata.pop('h_cat'), global_edges], dim=-1)
        g.edata['h_update'] = self.edge_update(h_cat_edge)
        # node update
        g.update_all(
            fn.copy_e('h_update', 'ave_he'),
            fn.mean('ave_he', 'h_ave_he')
        )
        h_cat_node = torch.cat([g.ndata.pop('h'), g.ndata['h_ave_he'], torch.repeat_interleave(global_feats, batch_num_nodes, dim=0)], dim=-1)
        g.ndata['h_update'] = self.node_update(h_cat_node)
        # glob update
        h_aver_node = self.aver_pool(g, g.ndata['h_update'])
        h_aver_edge = self.aver_pool(g, g.ndata['h_ave_he'])
        h_cat_glob = torch.cat([h_aver_node, h_aver_edge, global_feats], dim=-1)
        global_feats = self.glob_update(h_cat_glob)
        
        return g.ndata['h_update'], g.edata['h_update'], global_feats

    
class Edge_Gate_Convolution(nn.Module):
    def __init__(self, feat_length, mlp_act='silu'):
        super().__init__()
        self.edge_update_linear_src  = nn.Linear(feat_length, feat_length)
        self.edge_update_linear_dst  = nn.Linear(feat_length, feat_length)
        self.edge_update_linear_edge = nn.Linear(feat_length, feat_length)
        self.edge_update_acti        = get_activation(mlp_act)
        self.edge_update_batchnorm   = nn.BatchNorm1d(feat_length)
        self.edge_acti               = get_activation(mlp_act)

        self.node_update_linear_src  = nn.Linear(feat_length, feat_length)
        self.node_update_linear_dst  = nn.Linear(feat_length, feat_length)
        self.node_update_batchnorm   = nn.BatchNorm1d(feat_length)
        self.node_acti               = get_activation(mlp_act)
        
    def forward(self, g, node_feats, edge_feats):
        # init
        g = g.local_var()
        # update edge
        g.ndata['e_src'] = self.edge_update_linear_src(node_feats)
        g.ndata['e_dst'] = self.edge_update_linear_dst(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        edge_feats_update = self.edge_update_linear_edge(edge_feats) + g.edata.pop("e_nodes")
        # update node
        g.edata['e_update_sigmoid_gate'] = self.edge_update_acti(edge_feats_update)
        g.ndata['n_dst'] = self.node_update_linear_dst(node_feats)
        g.update_all(fn.v_mul_e('n_dst', 'e_update_sigmoid_gate', 'f'), fn.sum('f', 'n_gate_1'))
        g.update_all(fn.copy_e("e_update_sigmoid_gate", "f"), fn.sum("f", "n_gate_2"))
        g.ndata["n_gate"] = g.ndata["n_gate_1"] / (g.ndata["n_gate_2"] + 1e-6)
        node_feats_update = self.node_update_linear_src(node_feats) + g.ndata.pop("n_gate")
        # batch normal and residual
        edge_feats_update = self.edge_acti(self.edge_update_batchnorm(edge_feats_update)) + edge_feats
        node_feats_update = self.node_acti(self.node_update_batchnorm(node_feats_update)) + node_feats
        return node_feats_update, edge_feats_update


class Edge_Attn_Convolution(nn.Module):
    def __init__(self, feat_length, mlp_act='silu'):
        super().__init__()
        self.edge_update_linear_src  = nn.Linear(feat_length, feat_length)
        self.edge_update_linear_dst  = nn.Linear(feat_length, feat_length)
        self.edge_update_linear_edge = nn.Linear(feat_length, feat_length)
        self.edge_update_batchnorm   = nn.BatchNorm1d(feat_length)
        self.edge_acti               = get_activation(mlp_act)

        self.node_update_linear_src  = nn.Linear(feat_length, feat_length)
        self.node_update_linear_dst  = nn.Linear(feat_length, feat_length)
        self.node_update_batchnorm   = nn.BatchNorm1d(feat_length)
        self.node_acti               = get_activation(mlp_act)

        self.attn_weight             = None
        
    def forward(self, g, node_feats, edge_feats):
        # init
        g = g.local_var()
        # update edge
        g.ndata['e_src'] = self.edge_update_linear_src(node_feats)
        g.ndata['e_dst'] = self.edge_update_linear_dst(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        edge_feats_update = self.edge_update_linear_edge(edge_feats) + g.edata.pop("e_nodes")
        # update node
        g.edata['attn'] = edge_softmax(g, edge_feats_update.sum(dim=-1))
        self.attn_weight = g.edata['attn']
        g.ndata['n_dst'] = self.node_update_linear_dst(node_feats)
        g.update_all(fn.v_mul_e("n_dst", "attn", "f"), fn.sum("f", "na_dst"))
        node_feats_update = self.node_update_linear_src(node_feats) + g.ndata.pop("na_dst")
        # batch normal and residual
        edge_feats_update = self.edge_acti(self.edge_update_batchnorm(edge_feats_update)) + edge_feats
        node_feats_update = self.node_acti(self.node_update_batchnorm(node_feats_update)) + node_feats
        return node_feats_update, edge_feats_update
        

class MLPLayer(nn.Module):
    def __init__(self, input_length, output_length, acti_function='silu', p_droput=0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_length, output_length),
            nn.BatchNorm1d(output_length),
            get_activation(acti_function),
            nn.Dropout(p_droput)
        )

    def forward(self, x):
        return self.layer(x)

        
class Cluster_Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.sum_pool1 = SumPooling()

    def forward(self, g, node_feats):
        g = g.local_var()
        in_cluster = g.ndata['cluster']
        g.ndata['h_cluster'] = torch.mul(node_feats, in_cluster)
        h_cluster = g.ndata.pop('h_cluster')
        h_cluster = self.sum_pool1(g, h_cluster) / torch.sum(g.ndata.pop('cluster'))

        return h_cluster

        
def get_activation(name):
    act_name = name.lower()
    if act_name == 'elu':
        return nn.ELU(alpha=1.0)
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'relu6':
        return nn.ReLU6()
    elif act_name == 'prelu':
        return nn.PReLU(num_parameters=1, init=0.25)
    elif act_name == 'selu':
        return nn.SELU()
    elif act_name == 'silu':
        return nn.SiLU()
    elif act_name == 'celu':
        return nn.CELU(alpha=1.0)
    elif act_name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif act_name == 'sigmoid':
        return nn.Sigmoid()
    elif act_name == 'logsigmoid':
        return nn.LogSigmoid()
    elif act_name == 'tanh':
        return nn.Tanh()
    elif act_name == 'tanhshrink':
        return nn.Tanhshrink()
    elif act_name == 'softplus':
        return nn.Softplus(beta=1, threshold=20)
    elif act_name == 'softmax':
        return nn.Softmax(dim=1)
    elif act_name == 'softshrink':
        return nn.Softshrink()
    else:
        raise NameError("Not supported activation: {}".format(name))


class ConvFunc_CGCNN_edgeMLP(nn.Module):
    def __init__(self, node_feature_length, edge_feature_length, p_droput=0, mlp_ac='silu', mlp_at_ac='softplus'):
        super().__init__()
        self.mlp = MLPLayer(3 * node_feature_length, node_feature_length, mlp_ac, p_droput)
        self.screen = MLPLayer(3 * node_feature_length, node_feature_length, mlp_at_ac, p_droput)
        self.edgemlp = MLPLayer(edge_feature_length, node_feature_length, mlp_ac, p_droput)
        self.node_bn = nn.BatchNorm1d(node_feature_length)

    def feats_combine_in_edge(self, edges):
        h = torch.cat((edges.src["h"], edges.dst["h"], edges.data["hm"]), dim=1)
        return {"h_combine": h}

    def forward(self, g, node_feats, edge_feats):
        # init
        g = g.local_var()
        g.ndata['h'] = node_feats
        g.edata['h'] = edge_feats
        # prepare conv
        g.edata['hm'] = self.edgemlp(edge_feats)
        g.apply_edges(self.feats_combine_in_edge)
        h_combine = g.edata.pop('h_combine')
        # conv
        h_mlp = self.mlp(h_combine)
        h_screen = self.screen(h_combine)
        h_node_update = h_mlp * h_screen
        g.edata['h_node_update'] = h_node_update
        # update node
        g.update_all(
            fn.copy_e("h_node_update", "h_node_update"),
            fn.sum("h_node_update", "h_node_update")
        )
        return F.softplus(self.node_bn(g.ndata['h_node_update']) + node_feats)