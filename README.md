# Code and data for the TGNN-based model
#  --------------------------------------------------------
This dataset and codes are attached to the article:
A temporal graph neural network for cross-scale modelling of polycrystals considering microstructure interaction
If you use the dataset or codes, please cite them.
#  --------------------------------------------------------
Author: Yuanzhe, Hu.

Affiliation: State Key Laboratory of Mechanical Systems and Vibration, Shanghai Jiao Tong University
#  --------------------------------------------------------
Code availability
"Code" folder: 
	GNN_LMSC_model.py: Neural network architecture of the TGNN-based model implemented by Pytorch Geometric. 
    		-Input Data structure
        	1. x: input strain increment to RVE
        	2. edge_index: grain connections in the format of adjacency list
        	3. init_ori: initial grain orientations [grain_number, 3] 


#  --------------------------------------------------------
Database availability 
"Data" folder
  demo_dataset_PN8_sq100.hdf5

Contents details:
...
