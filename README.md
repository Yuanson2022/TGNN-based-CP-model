# Code and data for the TGNN-based model
#  --------------------------------------------------------
The dataset and codes are attached to the article:
"A temporal graph neural network for cross-scale modelling of polycrystals considering microstructure interaction"

If you use the dataset or codes, please cite them.
#  --------------------------------------------------------
Author: Yuanzhe, Hu.  
Affiliation: State Key Laboratory of Mechanical Systems and Vibration, Shanghai Jiao Tong University
#  --------------------------------------------------------
Code availability  
"Code" folder:  
GNN_LMSC_model.py: Neural network architecture of the TGNN-based model implemented by Pytorch Geometric  
 -Input Data structure:  
	1. x: input strain increment to RVE  
	2. edge_index: grain connections in the format of adjacency list  
	3. init_ori: initial grain orientations [grain_number, 3]  

The used software packages can be found in "requirements.txt"  
#  --------------------------------------------------------
Database availability 

"Data" folder:  
  demo_dataset_PN8_sq100.hdf5  

Contents details:  
(1) It contains two strain paths for four loading cases, i.e., tension, compression, cyclic loading and arbitrary loading.  
(2) The data is organized as a four-dimensional tensor in the shape of [path_num, node_num, seqential_length, featrue_dimension].  
(3) This work assumes 100 grains in the RVE, and the 101 data in node_num records the homogenized response.  
(4) The feature dimension has 15 variables: 1-3 Euler angles for each grain; 4-9 Cauthy stress components; 10-15 accumulated strain 
