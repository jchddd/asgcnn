# ASGCNN
This repository contains an implementation of the [ASGCNN](https://www.sciencedirect.com/science/article/pii/S0169433224012327) (Adsorbate-Site Graph Convolutional Neural Network) that predicts the adsorption energies with the help of classification tasks for adsorbate types and adsorption sites of slab structures.

<div align="center">
<img src="https://github.com/jchddd/asgcnn/blob/main/architecture.png"><br>
</div>

# Requirment
In parentheses is a version that is compatible after testing, current potential conflicts are from dgl and torch(torchdata).
- torch (2.1.0) (1.13.1 + cu117)
- torchdata (0.7.0)
- dgl (2.2.1) (1.0.1 + cu117)
- igraph
- networkx
- scikit-learn
- pymatgen
- matplotlib
- tqdm
- numpy
- pandas
- qmpy_rester
- hyperopt
# Overview
- **ASGCNN/**[**Encoder.py**](https://github.com/jchddd/asgcnn/blob/main/ASGCNN/Encoder.py):  Generate graph structure from VASP structure file and encode node and edge features.
- **ASGCNN/**[**Model.py**](https://github.com/jchddd/asgcnn/blob/main/ASGCNN/Model.py): Pytorch implementation of the ASGCNN model.
- **ASGCNN/**[**Traniner.py**](https://github.com/jchddd/asgcnn/blob/main/ASGCNN/Trainer.py): A module that calls the GNN model for training and prediction.
- **data**: Stores graph structures and targets for network training. Graphs are stored as .bin files in the dgl package.
- **figures**: Pictures drawn in Python in the article. Some of the drawings require custom [Jworkflow scripts](https://github.com/jchddd/scripts/tree/main/jworkflow). Some code cannot run directly due to data size limitations.
- **pretrained**: Pretrained models. There are five models learned in an ensemble method, and they predict together to provide the uncertainty of the prediction results.
- **structures**: VASP structure files for calculation and graph structure generation.
# Tutorials
- Query Heusler alloy data from OQMD: [Tutorial 1 - query data](https://github.com/jchddd/asgcnn/blob/main/tutorials/Tutorial%201%20-%20query%20data.ipynb)
- Analysis of VASP results and Batch construction of adsorption structures: [Jworkflow scripts](https://github.com/jchddd/scripts/tree/main/jworkflow)
- Load pre-trained models and view graph data characteristics : [Tutorial 2 -_load_pre-trained model](https://github.com/jchddd/asgcnn/blob/main/tutorials/Tutorial%202%20-%20load%20pre-trained%20model.ipynb)
- Load datasets and train an ASGCNN from scratch: [Tutorial 3 - model training](https://github.com/jchddd/asgcnn/blob/main/tutorials/Tutorial%203%20-%20model%20training.ipynb)
- Other supported model architectures: [Tutorial 4 - Other model architecture](https://github.com/jchddd/asgcnn/blob/main/tutorials/Tutorial%204%20-%20other%20model%20architecture.ipynb)
- Hyperparameter search, integrated model and other training methods: [Turorial 5 - training method](https://github.com/jchddd/asgcnn/blob/main/tutorials/Tutorial%205%20-%20training%20method.ipynb)
# Citation
If you are interested in our work, you can read our literature, and cite us using
```
@article{ZHOU2024160519,
title = {Machine-learning-accelerated screening of Heusler alloys for nitrogen reduction reaction with graph neural network},
journal = {Applied Surface Science},
volume = {669},
pages = {160519},
year = {2024},
issn = {0169-4332},
doi = {https://doi.org/10.1016/j.apsusc.2024.160519},
url = {https://www.sciencedirect.com/science/article/pii/S0169433224012327},
author = {Jing Zhou and Xiayong Chen and Xiao Jiang and Zean Tian and Wangyu Hu and Bowen Huang and Dingwang Yuan}
```
