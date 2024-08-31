# ASGCNN
This repository contains an implementation of the ASGCNN (Adsorbate-Site Graph Convolutional Neural Network) that predicts the adsorption energies with the help of classification tasks for adsorbate types and adsorption sites of slab structures.
# Requirment
- torch
- dgl
- igraph
- sklearn
- pymatgen
- matplotlib
- tqdm
- numpy
- pandas
# Overview
- ASGCNN/[Encoder.py](https://github.com/jchddd/asgcnn/blob/main/ASGCNN/Encoder.py):  Generate graph structure from VASP structure file and encode node and edge features.
- ASGCNN/[Model.py](https://github.com/jchddd/asgcnn/blob/main/ASGCNN/Model.py): Pytorch implementation of the ASGCNN model.
- ASGCNN/[Traniner.py](https://github.com/jchddd/asgcnn/blob/main/ASGCNN/Trainer.py): A module that calls the GNN model for training and prediction.
- data: Stores graph structures and target quantities for network training. Graphs stored as .bin files in the dgl package.
- figures: Pictures drawn in Python in the article. Some of the drawings require custom [scripts](https://github.com/jchddd/scripts). Some code cannot run directly due to data size limitations.
- pretrained: Pretrained models. There are a total of five models learned in an ensemble method, and they predict together to provide the uncertainty of the prediction results.
- structures: VASP structure files for calculation and graph structure generation.
# Tutorials
- xxx
- Batch construction of adsorption models and analysis of VASP results: This part is done through custom [scripts](https://github.com/jchddd/scripts)
- Use pre-trained model or train a new model from scratch: [tutorials.ipynb](https://github.com/jchddd/asgcnn/blob/main/tutorials.ipynb)
# Citation
If you use 'ASGCNN', please cite us using
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
