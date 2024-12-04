import torch
from torch import nn
from torch.nn import functional as F
import dgl
import dgl.function as fn
from dgl.nn import AvgPooling, SumPooling


class CGCNN(nn.Module):
    def __init__(self,
                 node_feat_length,
                 edge_feat_length,
                 embed_feat_length,
                 conv_number,
                 fc_lens,
                 mlp_acti='silu',
                 conv_type='C',
                 att=False,
                 p_droput=0,
                 task_typ='regre',
                 class_dim=[5, 4],
                 regre_dim=[1],
                 return_vp=False):
        super().__init__()
        # init
        self.name = 'CGCNN'
        self.task_typ = task_typ
        # embedding
        self.node_embedding = MLPLayer(node_feat_length, embed_feat_length, mlp_acti)
        # convolution
        if conv_type == 'C':
            self.conv_layers = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length, edge_feat_length) for _ in range(conv_number)])
        elif conv_type == 'Ce':
            self.conv_layers = nn.ModuleList([ConvFunc_CGCNN_edgeMLP(embed_feat_length, edge_feat_length) for _ in range(conv_number)])
        # pooling
        self.aver_pool = AvgPooling()
        # self Attention
        self.att = att
        self.self_att = nn.ModuleList([MLPLayer(embed_feat_length, 2 * embed_feat_length, mlp_acti), 
                             MLPLayer(2 * embed_feat_length, embed_feat_length, 'softmax')])
        # fully connect layers
        if   type(fc_lens[0]) == int:
            self.one_fcl = True
            fc_lens = [embed_feat_length] + fc_lens
            self.fc_layers = nn.ModuleList([MLPLayer(fc_lens[_], fc_lens[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens) - 1)])
        elif type(fc_lens[0]) == list:
            self.one_fcl = False
            for fi, fc_lens_i in enumerate(fc_lens):
                fc_lens[fi] = [embed_feat_length] + fc_lens_i
        # predict target
        self.class_num = len(class_dim)
        self.regre_num = len(regre_dim)
        if   self.one_fcl:
            if task_typ == 'multy':
                for i, dim_class in enumerate(class_dim):
                    setattr(self, 'pred_class_' + str(i), MLPLayer(fc_lens[-1], dim_class, 'softmax'))
            for i, dim_regre in enumerate(regre_dim):
                setattr(self, 'pred_regre_' + str(i), nn.Linear(fc_lens[-1], dim_regre))
        elif not self.one_fcl:
            if task_typ == 'multy':
                for i, dim_class in enumerate(class_dim):
                    fc_lens_i = fc_lens[i]
                    if   len(fc_lens_i) == 1:
                        setattr(self, 'pred_class_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[-1], dim_class, 'softmax')]))
                    elif len(fc_lens_i)  > 1:
                        setattr(self, 'pred_class_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[_], fc_lens_i[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens_i) - 1)] + [MLPLayer(fc_lens_i[-1], dim_class, 'softmax')]))
            for i, dim_regre in enumerate(regre_dim):
                fc_lens_i = fc_lens[i + len(class_dim)] if task_typ == 'multy' else fc_lens[i]
                if  len(fc_lens_i) == 1:
                    setattr(self, 'pred_regre_' + str(i), nn.ModuleList([nn.Linear(fc_lens_i[-1], dim_regre)]))
                elif len(fc_lens_i)  > 1:
                    setattr(self, 'pred_regre_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[_], fc_lens_i[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens_i) - 1)] + [nn.Linear(fc_lens_i[-1], dim_regre)]))
        # return graph feature vector
        self.return_vp = return_vp

    def forward(self, g):
        # init
        g = g.local_var()
        v = g.ndata.pop('h_v')
        e = g.edata.pop('h_e')
        # embedding
        v = self.node_embedding(v)
        # convolution
        for con_layer in self.conv_layers:
            v = con_layer(g, v, e)
        # pooling
        vc = self.aver_pool(g, v)
        vp = vc
        # self attention 
        if self.att:
            for at_layer in self.self_att:
                vc = at_layer(vc)
        # fully connect layers
        if self.one_fcl:
            for fc_layer in self.fc_layers:
                vc = fc_layer(vc)
        # predict target
        targets = []
        if self.task_typ == 'multy':
            for i in range(self.class_num):
                if   self.one_fcl:
                    targets.append(getattr(self, 'pred_class_' + str(i))(vc))
                elif not self.one_fcl:
                    target = vc
                    for fcl in getattr(self, 'pred_class_' + str(i)):
                        target = fcl(target)
                    targets.append(target)
        for i in range(self.regre_num):
            if   self.one_fcl:
                targets.append(getattr(self, 'pred_regre_' + str(i))(vc))
            elif not self.one_fcl:
                target = vc
                for fcl in getattr(self, 'pred_regre_' + str(i)):
                    target = fcl(target)
                targets.append(target)
        target = torch.cat(targets, dim=1)
        # return graph feature vector
        if self.return_vp:
            return target, vp
        else:
            return target            


class SGCNN(nn.Module):
    def __init__(self,
                 node_feat_length_slab,
                 edge_feat_length_slab,
                 node_feat_length_bulk,
                 edge_feat_length_bulk,
                 embed_feat_length,
                 conv_number,
                 fc_lens,
                 mlp_acti='silu',
                 conv_type='C',
                 att=False,
                 p_droput=0,
                 task_typ='regre',
                 class_dim=[5, 4],
                 regre_dim=[1],
                 return_vp=False):
        super().__init__()
        # init
        self.name = 'SGCNN'
        self.task_typ = task_typ
        # embedding
        self.node_embedding_slab = MLPLayer(node_feat_length_slab, embed_feat_length, mlp_acti)
        self.node_embedding_bulk = MLPLayer(node_feat_length_bulk, embed_feat_length, mlp_acti)
        # convolution
        if conv_type == 'C':
            self.conv_layers_slab = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length, edge_feat_length_slab) for _ in range(conv_number)])
            self.conv_layers_bulk = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length, edge_feat_length_bulk) for _ in range(conv_number)])
        elif conv_type == 'Ce':
            self.conv_layers_slab = nn.ModuleList([ConvFunc_CGCNN_edgeMLP(embed_feat_length, edge_feat_length_slab) for _ in range(conv_number)])
            self.conv_layers_slab = nn.ModuleList([ConvFunc_CGCNN_edgeMLP(embed_feat_length, edge_feat_length_bulk) for _ in range(conv_number)])
        # pooling
        self.pool_slab = AvgPooling()
        self.pool_bulk = AvgPooling()
        # self Attention
        self.att = att
        self.self_att = nn.ModuleList([MLPLayer(2 * embed_feat_length, 3 * embed_feat_length, mlp_acti), 
                             MLPLayer(3 * embed_feat_length, 2 * embed_feat_length, 'softmax')])
        # fully connect layers
        if   type(fc_lens[0]) == int:
            self.one_fcl = True
            fc_lens = [2 * embed_feat_length] + fc_lens
            self.fc_layers = nn.ModuleList([MLPLayer(fc_lens[_], fc_lens[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens) - 1)])
        elif type(fc_lens[0]) == list:
            self.one_fcl = False
            for fi, fc_lens_i in enumerate(fc_lens):
                fc_lens[fi] = [embed_feat_length] + fc_lens_i
        # predict target
        self.class_num = len(class_dim)
        self.regre_num = len(regre_dim)
        if   self.one_fcl:
            if task_typ == 'multy':
                for i, dim_class in enumerate(class_dim):
                    setattr(self, 'pred_class_' + str(i), MLPLayer(fc_lens[-1], dim_class, 'softmax'))
            for i, dim_regre in enumerate(regre_dim):
                setattr(self, 'pred_regre_' + str(i), nn.Linear(fc_lens[-1], dim_regre))
        elif not self.one_fcl:
            if task_typ == 'multy':
                for i, dim_class in enumerate(class_dim):
                    fc_lens_i = fc_lens[i]
                    if   len(fc_lens_i) == 1:
                        setattr(self, 'pred_class_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[-1], dim_class, 'softmax')]))
                    elif len(fc_lens_i)  > 1:
                        setattr(self, 'pred_class_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[_], fc_lens_i[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens_i) - 1)] + [MLPLayer(fc_lens_i[-1], dim_class, 'softmax')]))
            for i, dim_regre in enumerate(regre_dim):
                fc_lens_i = fc_lens[i + len(class_dim)] if task_typ == 'multy' else fc_lens[i]
                if  len(fc_lens_i) == 1:
                    setattr(self, 'pred_regre_' + str(i), nn.ModuleList([nn.Linear(fc_lens_i[-1], dim_regre)]))
                elif len(fc_lens_i)  > 1:
                    setattr(self, 'pred_regre_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[_], fc_lens_i[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens_i) - 1)] + [nn.Linear(fc_lens_i[-1], dim_regre)]))
        # return graph feature vector
        self.return_vp = return_vp

    def forward(self, gs, gb):
        # slab init
        gs = gs.local_var()
        vs = gs.ndata.pop('h_v')
        es = gs.edata.pop('h_e')
        # slab embedding
        vs = self.node_embedding_slab(vs)
        # slab convolution
        for con_layer in self.conv_layers_slab:
            vs = con_layer(gs, vs, es)
        # slab pooling
        vsp = self.pool_slab(gs, vs)
        # bulk init
        gb = gb.local_var()
        vb = gb.ndata.pop('h_v')
        eb = gb.edata.pop('h_e')
        # bulk embedding
        vb = self.node_embedding_bulk(vb)
        # bulk convolution
        for con_layer in self.conv_layers_bulk:
            vb = con_layer(gb, vb, eb)
        # bulk pooling
        vbp = self.pool_bulk(gb, vb)
        # cat graph features
        vt = torch.cat([vsp, vbp], dim=1)
        vp = vt
        # self attention 
        if self.att:
            for at_layer in self.self_att:
                vt = at_layer(vt)
        # fully connect layers
        if self.one_fcl:
            for fc_layer in self.fc_layers:
                vt = fc_layer(vt)
        # predict target
        targets = []
        if self.task_typ == 'multy':
            for i in range(self.class_num):
                if   self.one_fcl:
                    targets.append(getattr(self, 'pred_class_' + str(i))(vt))
                elif not self.one_fcl:
                    target = vt
                    for fcl in getattr(self, 'pred_class_' + str(i)):
                        target = fcl(target)
                    targets.append(target)
        for i in range(self.regre_num):
            if   self.one_fcl:
                targets.append(getattr(self, 'pred_regre_' + str(i))(vt))
            elif not self.one_fcl:
                target = vt
                for fcl in getattr(self, 'pred_regre_' + str(i)):
                    target = fcl(target)
                targets.append(target)
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
                 conv_number,
                 fc_lens,
                 mlp_acti='silu',
                 conv_type='Ce',
                 att=False,
                 p_droput=0,
                 task_typ='regre',
                 class_dim=[5, 4],
                 regre_dim=[1],
                 return_vp=False):
        super().__init__()
        # init
        self.name = 'ASGCNN'
        self.task_typ = task_typ
        # embedding
        self.node_embedding_adsb = MLPLayer(node_feat_length_adsb, embed_feat_length_adsb, mlp_acti)
        self.node_embedding_slab = MLPLayer(node_feat_length_slab, embed_feat_length_slab, mlp_acti)
        # convolution
        if conv_type == 'C':
            self.conv_layers_adsb = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length_adsb, edge_feat_length_adsb) for _ in range(conv_number)])
            self.conv_layers_slab = nn.ModuleList([ConvFunc_CGCNN(embed_feat_length_slab, edge_feat_length_slab) for _ in range(conv_number)])
        elif conv_type == 'Ce':
            self.conv_layers_adsb = nn.ModuleList([ConvFunc_CGCNN_edgeMLP(embed_feat_length_adsb, edge_feat_length_adsb) for _ in range(conv_number)])
            self.conv_layers_slab = nn.ModuleList([ConvFunc_CGCNN_edgeMLP(embed_feat_length_slab, edge_feat_length_slab) for _ in range(conv_number)])
        # pooling
        self.pool_adsb = AvgPooling()            
        self.pool_slab = AvgPooling()
        embed_feat_length = embed_feat_length_adsb + embed_feat_length_slab
        # self Attention
        self.att = att
        self.self_att = nn.ModuleList([MLPLayer(2 * embed_feat_length, 3 * embed_feat_length, mlp_acti), 
                             MLPLayer(3 * embed_feat_length, 2 * embed_feat_length, 'softmax')])
        # fully connect layers
        if   type(fc_lens[0]) == int:
            self.one_fcl = True
            fc_lens = [embed_feat_length] + fc_lens
            self.fc_layers = nn.ModuleList([MLPLayer(fc_lens[_], fc_lens[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens) - 1)])
        elif type(fc_lens[0]) == list:
            self.one_fcl = False
            for fi, fc_lens_i in enumerate(fc_lens):
                fc_lens[fi] = [embed_feat_length] + fc_lens_i
        # predict target
        self.class_num = len(class_dim)
        self.regre_num = len(regre_dim)
        if   self.one_fcl:
            if task_typ == 'multy':
                for i, dim_class in enumerate(class_dim):
                    setattr(self, 'pred_class_' + str(i), MLPLayer(fc_lens[-1], dim_class, 'softmax'))
            for i, dim_regre in enumerate(regre_dim):
                setattr(self, 'pred_regre_' + str(i), nn.Linear(fc_lens[-1], dim_regre))
        elif not self.one_fcl:
            if task_typ == 'multy':
                for i, dim_class in enumerate(class_dim):
                    fc_lens_i = fc_lens[i]
                    if   len(fc_lens_i) == 1:
                        setattr(self, 'pred_class_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[-1], dim_class, 'softmax')]))
                    elif len(fc_lens_i)  > 1:
                        setattr(self, 'pred_class_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[_], fc_lens_i[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens_i) - 1)] + [MLPLayer(fc_lens_i[-1], dim_class, 'softmax')]))
            for i, dim_regre in enumerate(regre_dim):
                fc_lens_i = fc_lens[i + len(class_dim)] if task_typ == 'multy' else fc_lens[i]
                if  len(fc_lens_i) == 1:
                    setattr(self, 'pred_regre_' + str(i), nn.ModuleList([nn.Linear(fc_lens_i[-1], dim_regre)]))
                elif len(fc_lens_i)  > 1:
                    setattr(self, 'pred_regre_' + str(i), nn.ModuleList([MLPLayer(fc_lens_i[_], fc_lens_i[_ + 1], mlp_acti, p_droput) for _ in range(len(fc_lens_i) - 1)] + [nn.Linear(fc_lens_i[-1], dim_regre)]))
        # return graph feature vector
        self.return_vp = return_vp

    def forward(self, ga, gs):
        # adsb init
        ga = ga.local_var()
        va = ga.ndata.pop('h_v')
        ea = ga.edata.pop('h_e')
        # adsb embedding
        va = self.node_embedding_adsb(va)
        # adsb convolution
        for conv_layer in self.conv_layers_adsb:
            va = conv_layer(ga, va, ea)
        # adsb pooling
        vap = self.pool_adsb(ga, va)
        # slab init
        gs = gs.local_var()
        vs = gs.ndata.pop('h_v')
        es = gs.edata.pop('h_e')
        # slab embedding
        vs = self.node_embedding_slab(vs)
        # slab convolution
        for conv_layer in self.conv_layers_slab:
            vs = conv_layer(gs, vs, es)
        # slab pooling
        vsp = self.pool_slab(gs, vs)
        # cat graph features
        vt = torch.cat([vap, vsp], dim=1)
        vp = vt
        # self attention 
        if self.att:
            for at_layer in self.self_att:
                vt = at_layer(vt)
        # fully connect layers
        if self.one_fcl:
            for fc_layer in self.fc_layers:
                vt = fc_layer(vt)
        # predict target
        targets = []
        if self.task_typ == 'multy':
            for i in range(self.class_num):
                if   self.one_fcl:
                    targets.append(getattr(self, 'pred_class_' + str(i))(vt))
                elif not self.one_fcl:
                    target = vt
                    for fcl in getattr(self, 'pred_class_' + str(i)):
                        target = fcl(target)
                    targets.append(target)
        for i in range(self.regre_num):
            if   self.one_fcl:
                targets.append(getattr(self, 'pred_regre_' + str(i))(vt))
            elif not self.one_fcl:
                target = vt
                for fcl in getattr(self, 'pred_regre_' + str(i)):
                    target = fcl(target)
                targets.append(target)
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
    def __init__(self, node_feature_length, edge_feature_length, p_droput=0, mlp_ac='sigmoid', mlp_at_ac='softplus'):
        super().__init__()
        self.mlp = MLPLayer(2 * node_feature_length + edge_feature_length, node_feature_length, mlp_ac, p_droput)
        self.screen = MLPLayer(2 * node_feature_length + edge_feature_length, node_feature_length, mlp_at_ac, p_droput)
        self.node_bn = nn.BatchNorm1d(node_feature_length)

    def feats_combine_in_edge(self, edges):
        h = torch.cat((edges.src["h"], edges.dst["h"], edges.data["h"]), dim=1)
        return {"h_combine": h}

    def forward(self, g, node_feats, edge_feats):
        # init
        g = g.local_var()
        g.ndata['h'] = node_feats
        g.edata['h'] = edge_feats
        # prepare conv
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

# class AtomicTypeAvePooling(nn.Module):
#     def __init__(self, node_feature_length, p_droput=0, mlp_ac='elu'):
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.N_symbol = torch.Tensor([7]).to(self.device)
#         self.H_symbol = torch.Tensor([1]).to(self.device)
#         self.sum_pool1 = SumPooling()

#     def forward(self, g, node_feats):
#         g = g.local_var()
#         is_N = (g.ndata['symbol'] == self.N_symbol).long()
#         is_H = (g.ndata['symbol'] == self.H_symbol).long()
#         is_adsb = is_N + is_H
#         is_adsb = is_adsb.reshape(-1, 1)
#         g.ndata['is_adsb'] = is_adsb
#         g.ndata['h_adsb'] = torch.mul(node_feats, is_adsb)
#         h_adsb = g.ndata.pop('h_adsb')
#         h_adsb = self.sum_pool1(g, h_adsb) / torch.sum(g.ndata.pop('is_adsb'))

#         return h_adsb
