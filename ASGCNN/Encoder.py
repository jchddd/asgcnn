import numpy as np
import pandas as pd
import torch as th
import igraph as ig
import dgl
import os
import math
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure
from dgl.data.utils import save_graphs, load_graphs
from tqdm.notebook import tqdm as tqdm
# from tqdm import tqdm
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from scipy.ndimage import gaussian_filter1d, convolve1d
from scipy.signal.windows import triang
from collections import Counter


class Graph_data_loader():
    '''
    A graph data loader and reader.

    Parameter:
        - batch_size: predefine defined the batch size / int, default 64
    '''
    def __init__(self, batch_size=64):
        self.graphs = []
        self.target = 0
        self.index = []
        self.index_ori = []
        self.len = 0
        self.graph_typ_number = 0

        self.batch_size = batch_size
        self.start_index = 0
        self.random_shuffle = True
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.current_index = None
        self.use_ELD = False
        self.bin_index = None
        self.effective_label_density = None
        self.weights = None

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        # Stop check
        if self.start_index == self.len:
            self.start_index = 0
            raise StopIteration
        # Use random shuffle
        if self.start_index == 0 and self.random_shuffle == True:
            np.random.shuffle(self.index)
        # Get start and end index
        start_index = self.start_index
        end_index = self.start_index + self.batch_size
        if end_index > self.len:
            end_index = self.len
        self.start_index = end_index
        # Extract x, y data
        x = [dgl.batch([self.graphs[typ][i] for i in self.index[start_index:end_index]]) for typ in range(self.graph_typ_number)]
        y = self.target[self.index[start_index:end_index]]
        self.current_index = self.index[start_index:end_index]

        return x, y

    def load_data(self, data_excel, encoders, file_paths=None, file_columns=None, target=None, file_suffix='.vasp'):
        '''
        Creat graph data from excel files and vasp structures or load graph data from bin files

        Parameters:
            - data_excel: excel or csv that contains structure names and targets / str, path to the excel or csv file
            - encoders: Structure Encoders or bin files that store the graphs / list
            - file_path: file paths where the vasp structures are stored, using None for bin files reading / list
            - file_columns: column names in the excels that record the name of the vasp structures, using None for bin files reading / list
            - target: column names in the data_excel that record targets, using None for prediction dataset / list
            - file_suffix: suffix of vasp structures / str, default '.vasp'
        Cautions:
            - file_path, file_columns and encoders correspond one to one to different types of graphs of the same structure
        '''
        # init
        df = pd.read_csv(data_excel, index_col=0) if data_excel.split('.')[1] == 'csv' else pd.read_excel(data_excel, index_col=0)
        graph_typ_number = len(encoders)
        self.graph_typ_number = graph_typ_number

        # load graph
        # load graph from bin file
        if type(encoders[0]) == str:
            self.graphs = []  # empty stored list
            for typ in range(graph_typ_number):  # load graph from bin file
                graph_file = encoders[typ]
                graph_list = load_graphs(graph_file)[0]
                self.graphs.append(graph_list)
                for i in range(len(self.graphs[typ])):  # move graphs to device
                    self.graphs[typ][i] = self.graphs[typ][i].to(self.device)

        # load graph from vasp file
        else:
            self.graphs = [[] for typ in range(graph_typ_number)]  # empty stored list
            with tqdm(total=self.graph_typ_number * df.shape[0], unit='g', leave=False, desc='Load graph') as pbar:
                for typ in range(graph_typ_number):
                    structure_list = df[file_columns[typ]].values  # get file list
                    encoder = encoders[typ]
                    file_path = file_paths[typ]
                    for file in structure_list:  # read structure and encode
                        structure = Structure.from_file(os.path.join(file_path, file + file_suffix))
                        graph = encoder[structure]
                        graph = graph.to(self.device)
                        self.graphs[typ].append(graph)
                        pbar.update(1)

        # update infor
        self.len = len(self.graphs[0])
        self.index = np.array(range(self.len))
        self.index_ori = np.array(range(self.len))

        # read target values
        if target is None:  # empty target
            target = th.zeros([self.len, 1])
        elif type(target) == list:  # multy target
            targets = []
            for tar in target:
                target = df[tar].values
                target = th.Tensor(target).unsqueeze(1)
                targets.append(target)
            target = th.cat(targets, dim=1)
        else:  # single target
            target = df[target].values
            target = th.Tensor(target).unsqueeze(1)
        self.target = target.to(self.device)

    def apply_feature(self, encoder_element=[], encoder_edge=[]):
        '''
        Function to add features to the graph

        Parameters:
            - encoder_element: Encoder_elements for node feature encoding / list
            - encoder_edge: Encoder_edges for edge feature encoding / list
        '''
        with tqdm(iterable=range(self.graph_typ_number * self.len), unit='g', leave=False, desc='Apply feature') as pbar:
            for t in range(self.graph_typ_number):
                Encoder_element = encoder_element[t]
                Encoder_edge = encoder_edge[t]
                for g in self.graphs[t]:
                    if Encoder_element != None:
                        Encoder_element.apply_feature(g)
                    if Encoder_edge != None:
                        Encoder_edge.apply_feature(g)
                    pbar.update(1)

    def save_data(self, path, file_names):
        '''
        Function to save graphs to bin file

        Parameters:
            - path: path to where the bin file stored / str
            - file_names: file names for bin files / list, ['xxxx.bin', 'xxx.bin']
        '''
        for i in range(self.graph_typ_number):
            save_graphs(os.path.join(path, file_names[i]), self.graphs[i])

    def init_predict(self):
        self.start_index = 0
        self.index = self.index_ori.copy()
        self.len = len(self.index_ori)
        self.random_shuffle = False

    def init_train(self):
        self.start_index = 0
        self.random_shuffle = True

    def data_split(self, psamples):
        '''
        Seleting a random portion of the data for training

        Parameter:
            - psamples: the protion / float
        '''
        self.index = np.random.choice(self.index_ori, size=int(len(self.index_ori) * psamples), replace=False)
        self.len = len(self.index)

    def get_effective_label_density(self, bin_number=100, target_column=0, ks=5, sigma=2):
        '''
        Function to calculate effective label density for target property

        Parameters:
            - bin_number: total bin number for label distribution / int, default 100
            - target_column: target property column name / str
            - ks: kernel window length / int default, 5
            - sigma: sigma for gaussian kernel / float, 2
        '''
        self.use_ELD = True
        preds = self.target[:, target_column].cpu().detach().numpy().reshape(-1)
        sep_value = [min(preds) + (max(preds) - min(preds) + 0.001) / bin_number * (i + 1) for i in range(bin_number)]
        bin_index_per_label = []
        for target in preds:
            for i, sepv in enumerate(sep_value):
                if target < sepv:
                    bin_index_per_label.append(i)
                    break
        self.bin_index = np.array(bin_index_per_label)
        num_samples_of_bins = dict(Counter(bin_index_per_label))
        emp_label_dist = np.array([num_samples_of_bins.get(i, 0) for i in range(bin_number)])
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=ks, sigma=sigma)
        eff_label_dist = convolve1d(np.sqrt(emp_label_dist), weights=lds_kernel_window, mode='constant')
        self.effective_label_density = eff_label_dist
        weights = [np.float32(1 / x) for x in eff_label_dist]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        self.weights = th.Tensor(weights).unsqueeze(1).to(self.device)

    def get_LDS(self):
        bin_index = self.bin_index[self.current_index]
        return self.weights[bin_index]


class Encoder_structure():
    '''
    Encoder for encoding structure to graph

    Parameters:
        - cutoff: radius cutoff for searching neighbor atoms / float, default 3.36
        - pair_search_method: method for searching neighbor atoms / str, 'SGCNN', 'CGCNN', 'Voronoi' and 'Voronoi_incell'
          CGCNN: bonding is determined by the distance between atoms.
          SGCNN: bonding is determined by subtracting the radius from the distance between the atoms, a smaller cutoff should be used
          Voronoi: use the Voronoi method implemented on pymatgen to judge connection and determined distance lkie CGCNN
          Voronoi_incell: use the Vornoi method and regardless of the periodic
        - bond_restrict: add restrictions on bonding of specific elements / dict
          default {'H': {'bond': [{'N', 'H'}], 'radius': 1.6}} it means bonding of H is being restricted. 
          And H can only form bonds with N and H that are less than 1.6 Å in length.
        - bond_restrict_require: when the structure contains these specific elements, can bonding be restricted / dict, default {'N', 'H'}
        - element_restrict: selecting certain elements, only they and their neighbors will be preserved / dict, default {}
        - element_restrict_mode: mode for element restriction / str, 'atom', 'bond'
          This parameter determines the degree to which information about these elements' neighbors is retained
          atom (only): only bonds with their neighbors will be retained, bond (too): further retain bonds between their neighbors
        - distance_restrict: selecting particular elements, only atoms and the bonds to them within a distance are preserved / dict, default {}
          like {'one_v': ['N', 'H'], 'threshold': 0.3} means atoms within (0.3 + radius of the atom + radius of N or H) Å of H or N nodes are retained
        - dataset: name of element feature dataset / excel or csv file, default 'Feature.csv'
        - drop_repeat_edge: whether to drop repeat edge / bool, default False
          In general, there will be no repeat edges, this just to be on the safe side
    Cautions:
        - The element and distance bonding restrictions may conflict with each other and lead to errors, only one can be used at a time
    '''
    def __init__(self, cutoff=3.36,
                 pair_search_method='Voronoi',
                 bond_restrict={'H': {'bond': [{'N', 'H'}], 'radius': 1.6}},
                 bond_restrict_require={'N', 'H'},
                 element_restrict={},  # 'N','H'
                 element_restrict_mode='bond',
                 distance_restrict={},  # {'one_v': ['N', 'H'], 'threshold': 0.3}
                 dataset='ASGCNN/Element_feature.csv',
                 drop_repaet_edge=False,
                 ):
        self.cutoff = cutoff
        self.pair_search_method = pair_search_method
        assert self.pair_search_method in ['SGCNN', 'CGCNN', 'Voronoi', 'Voronoi_incell'], 'Undefined pair search method !'
        self.bond_restrict = bond_restrict
        self.bond_restrict_require = bond_restrict_require
        self.element_restrict = element_restrict
        self.element_restrict_mode = element_restrict_mode  # 'bond'-与限制元素成键才保留,'atom'-与限制元素及其成键原子之间的边会被保留
        self.distance_restrict = distance_restrict
        self.drop_repeat_edge = drop_repaet_edge

        self.dataset = pd.read_csv(dataset, index_col=0) if dataset.split('.')[1] == 'csv' else pd.read_excel(dataset, index_col=0)

        self.structure = None
        self.graph = None

    def encode_structure(self, structure):
        self.structure = structure
        neighbors = self.get_neighbors()
        edges = [n[:2] for n in neighbors]
        h_edges = [n[2] for n in neighbors]
        h_vertices = th.IntTensor([])
        for vertice in structure:
            h_vertices = th.cat([h_vertices, th.IntTensor([self.dataset.at[vertice.specie.symbol, 'atomic number']])])
        edges, h_edges, h_vertices = self.apply_edge_restrict(edges, h_edges, h_vertices)
        h_edges = th.Tensor(h_edges)
        graph = dgl.graph(([e[0] for e in edges], [e[1] for e in edges]))
        graph.ndata['symbol'] = h_vertices
        graph.edata['radius'] = h_edges
        self.graph = graph

    def get_neighbors(self, structure=None):
        if structure != None:
            self.structure = structure
        neighbors = []
        cutoff = self.cutoff
        method = self.pair_search_method
        if method.split('_')[0] == 'Voronoi':
            structure_voronoi = VoronoiConnectivity(self.structure, cutoff=cutoff)
            connectivity_array = structure_voronoi.connectivity_array
            for u in range(connectivity_array.shape[0]):
                for v in range(connectivity_array.shape[1]):
                    for im in range(connectivity_array.shape[2]):
                        if connectivity_array[u][v][im] != 0. and not all([(len(method.split('_')) == 2 and method.split('_')[1] == 'incell'), im != 13]):
                            v_im = structure_voronoi.get_sitej(v, im)
                            distance = v_im.distance(self.structure[u], jimage=[0, 0, 0])
                            if distance <= cutoff and distance > 0:
                                neighbors.append([u, v, distance])
        elif method == 'SGCNN' or method == 'CGCNN':
            cutoff = cutoff if method == 'CGCNN' else cutoff + 2.5
            connections = self.structure.get_all_neighbors(cutoff)
            for u in range(len(connections)):
                for e_v in connections[u]:
                    if method == 'CGCNN' and e_v.nn_distance <= cutoff:
                        neighbors.append([u, e_v.index, e_v.nn_distance])
                    elif method == 'SGCNN':
                        radius_u = self.dataset.at[self.structure[u].specie.symbol, 'radius']
                        radius_v = self.dataset.at[e_v.specie.symbol, 'radius']
                        if e_v.nn_distance - radius_u / 2 - radius_v / 2 <= cutoff:
                            neighbors.append([u, e_v.index, e_v.nn_distance])
        return neighbors

    def apply_edge_restrict(self, edges, h_edges, h_vertices):
        # init
        pop_index = []
        elements_str = [e.symbol for e in self.structure.species]
        # apply bond restrict
        if all([e in elements_str for e in self.bond_restrict_require]) and len(self.bond_restrict) > 0:  # = judge if bond_restrict_require if accomplished
            for i, edge in enumerate(edges):
                vertice = {elements_str[edge[0]], elements_str[edge[1]]}
                for v in edge:
                    symbol = elements_str[v]
                    if symbol in self.bond_restrict.keys():
                        if vertice not in self.bond_restrict[symbol]['bond'] or h_edges[i] > self.bond_restrict[symbol]['radius']:
                            pop_index.append(i)
                            break
        pop_index = list(set(pop_index))
        pop_index.sort(reverse=True)
        for i in pop_index:
            edges.pop(i)
            h_edges.pop(i)
        pop_index = []
        # apply element restrict
        if len(self.element_restrict) > 0:
            if self.element_restrict_mode == 'atom':
                atom_in_ele_restr = []
                for edge in edges:
                    if any([elements_str[edge[v]] in self.element_restrict for v in range(2)]):
                        atom_in_ele_restr += [edge[0], edge[1]]
                atom_in_ele_restr = set(atom_in_ele_restr)
                for i, edge in enumerate(edges):
                    if not all([edge[v] in atom_in_ele_restr for v in range(2)]):
                        pop_index.append(i)
            elif self.element_restrict_mode == 'bond':
                for i, edge in enumerate(edges):
                    if not any([elements_str[edge[v]] in self.element_restrict for v in range(2)]):
                        pop_index.append(i)
        # apply distance restrict
        if len(self.distance_restrict) > 0:
            distances_bond = []
            distances_cart = []
            indexes = []
            for i, edge in enumerate(edges):
                v_thre = [elements_str[edge[v]] in self.distance_restrict['one_v'] for v in range(2)]
                if any(v_thre) and not all(v_thre) and i not in pop_index:
                    radius_u = self.dataset.at[elements_str[edge[0]], 'radius']
                    radius_v = self.dataset.at[elements_str[edge[1]], 'radius']
                    bond_distance = h_edges[i] - (radius_u + radius_v)
                    distances_bond.append(bond_distance)
                    distances_cart.append(h_edges[i])
                    indexes.append(i)
            distances_bond_sort = distances_bond.copy()
            distances_bond_sort.sort()
            distances_cart_sort = distances_cart.copy()
            distances_cart_sort.sort()
            min_distance_bond = distances_bond_sort[0]
            min_distance_cart = distances_cart_sort[0]
            pop_atom = []
            for i in range(len(distances_bond)):
                if distances_bond[i] > min_distance_bond + self.distance_restrict['threshold']:
                    pop_bond_dist = True
                else:
                    pop_bond_dist = False
                if distances_cart[i] > min_distance_cart + self.distance_restrict['threshold']:
                    pop_bond_cart = True
                else:
                    pop_bond_cart = False
                if pop_bond_dist and pop_bond_cart:
                    pop_index.append(indexes[i])
        # delete edge and edge feature
        pop_index = list(set(pop_index))
        pop_index.sort(reverse=True)
        for i in pop_index:
            edges.pop(i)
            h_edges.pop(i)
        # drop repeat edge
        if self.drop_repeat_edge:
            edges_tup = [tuple(edge) for edge in edges]
            edges_dr = list(set(edges_tup))
            edges_dr = [list(edge) for edge in edges_dr]
            h_edges = [h_edges[edges.index(edge)] for edge in edges_dr]
            edges = edges_dr
        # delete empty nodes
        vertices_new = []
        for edge in edges:
            for v in edge:
                if v not in vertices_new:
                    vertices_new.append(v)
        if len(vertices_new) < h_vertices.shape[0]:
            for i, edge in enumerate(edges):
                for j, v in enumerate(edge):
                    edges[i][j] = vertices_new.index(edges[i][j])
            h_vertices = h_vertices[vertices_new]
        # return
        return edges, h_edges, h_vertices

    def __getitem__(self, item):
        self.encode_structure(item)
        return self.graph


class Encoder_edge():
    '''
    Encoder for encoding graph edge features

    Parameters:
        - features: list of features / list, 'radius', 'bond', 'category', 'distance'
          'radius': radius expanded on the basis function. 'bond': a simple int 1 represents bonding
          'category': bonding type according to two node types. 'distance': min node graph distances of this edge to certain nodes.
        - rbf_pares: radius basic function parameters / list, default [60, 5], number of basic sets and cutoff. None for use radius itself as feature
        - vertice_typs: category parameter. use atomic number to distinguish node type for construction of category feature / (x, 2) list
          default [[1, 7], []], means edges with nodes all in 1, 7, nodes all not in 1, 7, and one node in 1, 7 will be in three category 
        - distance_reference_element: distance parameter. the reference elements for calculating the distance feature / list, default [1, 7]
        - distance_feature_length: distance parameter. total length for distance feature / int, default 6
    '''
    def __init__(self, features=['radius'],
                 rbf_pares=[60, 5],
                 vertice_typs=[[1, 7], []],
                 distance_reference_element=[1, 7],
                 distance_feature_length=6,
                 ):
        self.edge_features = features
        assert all([h_e in ['radius', 'bond', 'category', 'distance'] for h_e in features]), 'Undefined edge feature !'
        self.rbf_paras = rbf_pares
        self.vertice_typs = vertice_typs
        self.distance_reference_element = distance_reference_element
        self.distance_feature_length = distance_feature_length
        self.h_edge_name = 'h_e'

        self.feature_lens = {}
        self.feature_length_total = 0
        self.combinations = []

        self.assess_feature_properities()

    def assess_feature_properities(self):
        for feature in self.edge_features:
            if feature == 'radius':
                if self.rbf_paras != None:
                    self.feature_lens['radius'] = self.rbf_paras[0]
                else:
                    self.feature_lens['radius'] = 1
            elif feature == 'bond':
                self.feature_lens['bond'] = 1
            elif feature == 'category':
                typ_number = len(self.vertice_typs)
                for typ_u in range(typ_number):
                    for typ_v in range(typ_number):
                        if {typ_u, typ_v} not in self.combinations:
                            self.combinations.append({typ_u, typ_v})
                combination_number = len(self.combinations)
                self.feature_lens['category'] = combination_number
            elif feature == 'distance':
                self.feature_lens['distance'] = self.distance_feature_length
        for feature_len in self.feature_lens.values():
            self.feature_length_total += feature_len

    def encode_edge(self, graph):
        num_edges = graph.num_edges()
        num_nodes = graph.num_nodes()
        symbols = graph.ndata['symbol'].cpu().numpy()
        radius = graph.edata['radius'].cpu().numpy()
        hs_e = {}
        for h_e in self.edge_features:
            if h_e == 'radius':
                if self.rbf_paras == None:
                    hs_e[h_e] = th.unsqueeze(graph.edata['radius'].cpu(), 1)
                else:
                    h = th.zeros((num_edges, self.rbf_paras[0]))
                    for i in range(num_edges):
                        h[i] = th.Tensor(bf_gaussian_radial_basis(radius[i], self.rbf_paras[0], self.rbf_paras[1]))
                    hs_e[h_e] = h

            elif h_e == 'bond':
                h = th.ones((num_edges, 1))
                hs_e[h_e] = h

            elif h_e == 'category':
                typ_number = len(self.vertice_typs)
                typ_element = {}
                for element in set(symbols):
                    for i, typ in enumerate(self.vertice_typs):
                        if int(element) in typ:
                            typ_element[str(int(element))] = i
                            break
                        elif typ == []:
                            typ_element[str(int(element))] = i

                h = th.zeros((num_edges, self.feature_lens['category']))
                for i in range(num_edges):
                    u_typ = typ_element[str(symbols[graph.edges()[0][i]])]
                    v_typ = typ_element[str(symbols[graph.edges()[1][i]])]
                    combination_index = self.combinations.index({u_typ, v_typ})
                    h[i][combination_index] = 1
                hs_e[h_e] = h

            elif h_e == 'distance':
                max_distance_length = 0
                g = ig.Graph()
                g.add_vertices(num_nodes)
                edges = [[graph.edges()[0][i], graph.edges()[1][i]] for i in range(num_edges)]
                g.add_edges(edges)
                dis_matrix = g.distances()
                h = th.zeros((num_edges, self.distance_feature_length))
                index_refer = []
                for i, symbol in enumerate(symbols):
                    if symbol in self.distance_reference_element:
                        index_refer.append(i)

                for i, edge in enumerate(edges):
                    if all([v in index_refer for v in edge]):
                        h[i][0] = 1
                    else:
                        id = max([min([dis_matrix[v][r] for r in index_refer]) for v in edge])
                        if len(set([min([dis_matrix[v][r] for r in index_refer]) for v in edge])) == 1:
                            id += 1
                        assert id < self.distance_feature_length, 'Encounter distance id ' + str(id) + ' the set distance feature length is insufficient'
                        h[i][id] = 1
                        if id + 1 > max_distance_length:
                            max_distance_length = id + 1
                hs_e[h_e] = h

        h_edges = th.cat(tuple(hs_e.values()), dim=1)
        return h_edges

    def apply_feature(self, graph):
        h_edges = self.encode_edge(graph)
        h_edges = h_edges.to(graph.device)
        graph.edata[self.h_edge_name] = h_edges


class Encoder_element():
    '''
    Encoder for encoding graph element features

    Parameters:
        - dataset: name for element feature dataset / excel or csv file, default 'Element_feature.csv'
        - elements: elements used in the study / list
        - features: seleted features / list, all available features on dataset
        - numerical_feature_length: feature length after numerical feature expand to one-hot type feature / int, default 10
        - one_hot_to_Gaussian: whether to use polycentric gaussians to replace one-hot feature / bool, default False
        - direct_use_feature_value: whether to direct use feature values as the feature / bool, default False
    '''
    def __init__(self, dataset='ASGCNN/Element_feature.csv',
                 elements=['Ag', 'Al', 'As', 'Au', 'B', 'Bi', 'Cd', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 'In', 'Ir', 'Mn', 'Mo', 'Nb',
                           'Ni', 'Os', 'Pb', 'Pd', 'Pt', 'Rh', 'Ru', 'Sb', 'Sc', 'Si', 'Sn', 'Ta', 'Ti', 'Tl', 'V', 'W', 'Y', 'Zn', 'Zr', 'N', 'H'],
                 features=['group', 'period', 'electronegativity', 'radius', 'affinity', 'weight'],
                 numerical_feature_length=10,
                 one_hot_to_Gaussian=False,
                 direct_use_feature_value=False,
                 bf_extend_magnitude=0.01,
                 floor_extend_magnitude=0.01
                 ):
        self.dataset = pd.read_csv(dataset, index_col=0) if dataset.split('.')[1] == 'csv' else pd.read_excel(dataset, index_col=0)
        drop_columns = [c for c in self.dataset.columns if c not in features and c != 'atomic number']
        self.dataset = self.dataset.loc[self.dataset.index.isin(elements)]
        self.dataset = self.dataset.drop(drop_columns, axis=1)
        assert not np.any(self.dataset.isnull()), 'Null values exist in the dataset !'
        self.elements = elements
        self.features = features
        self.direct_use_feature_value = direct_use_feature_value
        self.input_atomic_number = False
        self.h_vertice_name = 'h_v'
        self.bf_extend_magnitude = bf_extend_magnitude
        self.floor_extend_magnitude = floor_extend_magnitude

        self.feature_lens = {}
        self.feature_lentgh_total = 0
        self.features_encoderd = {}
        self.feature_scalar = {}

        self.numerical_features = ['electronegativity', 'ionization', 'affinity', 'volume', 'radius', 'weight', 'density', 'Zeff', 'melting', 'boiling',
                                   'polarizability', 'resistivity', 'fusion', 'vaporization', 'capacity']
        self.numerical_feature_length = 10
        self.one_hot_to_Gaussian = one_hot_to_Gaussian
        self.numerical_feature_interval = {}
        self.category_integer_features = ['group', 'period', 'atomic number', 'valence', 'd-electron']
        self.category_integer = {}

        self.assess_feature_properities()
        self.encode_element()

    def assess_feature_properities(self):
        self.feature_lens = {}
        self.feature_lentgh_total = 0
        for feature in self.features:
            all_value = list(set(self.dataset[feature].values))
            all_value.sort()
            if self.direct_use_feature_value:
                self.feature_lens[feature] = 1
                self.feature_lentgh_total += 1
                self.feature_scalar[feature] = StandardScaler().fit(np.array(all_value).reshape(-1, 1))
            else:
                if feature in self.numerical_features:
                    self.feature_lens[feature] = self.numerical_feature_length
                    self.feature_lentgh_total += self.numerical_feature_length
                    self.numerical_feature_interval[feature] = [min(all_value), max(all_value)]
                elif feature in self.category_integer_features:
                    self.feature_lens[feature] = len(all_value)
                    self.feature_lentgh_total += len(all_value)
                    self.category_integer[feature] = all_value

    def encode_element(self):
        for element in self.elements:
            feature_encoded = np.array([])
            for feature in self.features:
                feature_value = self.dataset.at[element, feature]
                if self.direct_use_feature_value:
                    feature_contact = np.array([[feature_value]])
                    feature_contact = self.feature_scalar[feature].transform(feature_contact)
                    feature_contact = feature_contact.reshape(1)
                else:
                    if feature in self.numerical_features:
                        minimum = self.numerical_feature_interval[feature][0]
                        maximum = self.numerical_feature_interval[feature][1]
                        spacing1 = abs((minimum - maximum) / (self.numerical_feature_length - 1))
                        spacing2 = abs((minimum - maximum) / (self.numerical_feature_length - 0))
                        if self.one_hot_to_Gaussian:
                            feature_contact = bf_polycentric_gaussians(feature_value, minimum, maximum, spacing1, self.bf_extend_magnitude)
                        else:
                            feature_contact = np.zeros(self.feature_lens[feature])
                            index_one_hot = np.floor((feature_value - (minimum - self.floor_extend_magnitude)) / spacing2) - 1
                            feature_contact[int(index_one_hot)] = 1
                    elif feature in self.category_integer_features:
                        category_integer = self.category_integer[feature]
                        feature_contact = np.zeros(len(category_integer))
                        index_one_hot = category_integer.index(feature_value)
                        feature_contact[index_one_hot] = 1
                assert self.feature_lens[feature] == len(feature_contact), 'Feature encode length error !'
                feature_encoded = np.concatenate((feature_encoded, feature_contact))
            self.features_encoderd[element] = th.from_numpy(feature_encoded)

    def decode_element(self, element):
        analytic_dict = {}
        index_start = 0
        feature_encoderd = self.features_encoderd[element]
        for feature in self.features:
            feature_len = self.feature_lens[feature]
            index_end = index_start + feature_len
            print(feature, ' : ', list(feature_encoderd[index_start:index_end].numpy()))
            index_start += feature_len

    def __getitem__(self, item):
        if self.input_atomic_number:
            item = self.dataset.loc[self.dataset['atomic number'] == item].index.values[0]
        return self.features_encoderd[item]

    def apply_feature(self, graph):
        num_nodes = graph.num_nodes()
        symbols = graph.ndata['symbol'].cpu().numpy()
        self.input_atomic_number = True
        h_vertices = th.zeros((num_nodes, self.feature_lentgh_total))
        for i in range(num_nodes):
            atomic_number = symbols[i]
            h_vertice = self[atomic_number]
            h_vertices[i] = h_vertice
        h_vertices = h_vertices.to(graph.device)
        graph.ndata[self.h_vertice_name] = h_vertices


def bf_polycentric_gaussians(values, center_min, center_max, center_spacing, maximum_extend_magnitude=0.01):
    '''
    A basis function consisting of Gaussian functions with equally spaced centers.

    Parameters:
        - values : float or list typ
        The target values that you want to project onto the basis function
        - center_min : flaot
        The minimum center of Gaussian function
        - center_max : float
        The maximum center of Gaussian function
        - center_spacing : float
        Centers of Gaussian functions will be creat from [center_min, center_max + center_spacing / 2) with a interval of center_spacing
        - maximum_extend_magnitude: float, default 0.01
        It is used to ensure that there is a Gaussian distribution near the maximum value When the center of the Gaussian basis is generated
    Return:
        - The vector that the target value is projected onto the basis function with a shape of (x, n) or (n)
        - x is the number of target valuse and n is the number of Gaussian functions
    Example:
        bf_polycentric_gaussians(3,2,3,0.5)
        [0.01831564 0.36787944 1.        ]
        bf_polycentric_gaussians([3,2.3],2,3,0.5)
        [[0.01831564 0.36787944 1.        ]
         [0.69767633 0.85214379 0.14085842]]
    '''
    values = np.array(values)
    centers = np.arange(center_min, center_max + maximum_extend_magnitude, center_spacing)
    return np.exp(-(values[..., np.newaxis] - centers) ** 2 / center_spacing ** 2)


def bf_gaussian_radial_basis(values, N_Gaussian=60, cutoff=6):
    '''
    A basis function using Gaussian radial basis.

    Parameters:
        - values : float or list typ
        The target values that you want to project onto the basis function
        - N_Gaussian : int, default 60
        The number of Gaussian basis sets
        - cutoff : float, default 6
        Cutoff value of the Gaussian basis sets
    Return:
        - The vector that the target value is projected onto the basis function with a shape of (x, n) or (n)
        - x is the number of target valuse and n is the number of Gaussian basis sets
    Example:
        bf_gaussian_radial_basis(3,6)
        [4.39941955e-01 2.39587913e-01 7.91383856e-02 1.58548533e-02 1.92659323e-03 1.41994195e-04]
    '''
    values = np.array(values)
    result = []
    for n in range(1, N_Gaussian + 1):
        phi = 1 - 6 * (values / cutoff) ** 5 + 15 * (values / cutoff) ** 4 - 10 * (values / cutoff) ** 3
        betak = (2 / N_Gaussian * (1 - np.exp(-cutoff))) ** (-2)
        miuk = np.exp(-cutoff) + (1 - np.exp(-cutoff)) / N_Gaussian * n
        gaussian = np.exp(-betak * (np.exp(-values) - miuk) ** 2)
        result.append(phi * gaussian)
    result = np.transpose(np.array(result))
    return result


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window
