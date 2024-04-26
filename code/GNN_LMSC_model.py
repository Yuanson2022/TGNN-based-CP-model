import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# define datatype
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_long = torch.LongTensor

# model formulation
class GNN_LMSC_cell(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, units:int, gcn_type: str,
                    batch_size: int,  # this entry is unnecessary, kept only for backward compatibility
                    width=125, depth=4,):
            super(GNN_LMSC_cell, self).__init__()

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.units = units
            self.gcn_type = gcn_type
            self.batch_size = batch_size  # not needed
            self.depth = depth

            start_dim = units + in_channels
            inside_dim = start_dim
            self.qb =  Quadratic_block(inside_dim, width, depth)
        
            if gcn_type == 'GCNConv':
                self.gconv1 = GCN_block(inside_dim, inside_dim, layer = 1)
                self.gconv2 = GCN_block(inside_dim, inside_dim, layer = 1)
            
            if self.depth>0:
                inside_dim = width
            else:
                inside_dim = in_channels

            self.fc_alpha = nn.Linear(inside_dim,units)
            self.fc_beta = nn.Linear(inside_dim,units)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        ) -> torch.FloatTensor:

        # Input strain increment X-> [batch, seq_len, in_dim]
        # Hidden state H -> [batch, node_number, hidden_dim]

        h_t = H
    
        strain_norm = torch.norm(X[:,:,0:6],dim=2)
        x_input = X.clone()
        x_input[:,:,0:6] = (x_input[:,:,0:6].squeeze(1)/(strain_norm + 1e-15)).unsqueeze(1)

        cat_input = torch.cat([x_input.repeat(1,h_t.shape[1],1),h_t], dim = 2)

        # graph embedding
        if self.gcn_type == 'GCNConv':
            G_input1 = self.gconv1(cat_input, edge_index)
            G_input1 = F.relu(G_input1) 
            G_input2 = self.gconv2(cat_input, edge_index)
            G_input2 = F.relu(G_input2) 
        else:
            G_input1 = cat_input
            G_input2 = cat_input

        G_input1 = torch.tanh(self.qb(G_input1))
        G_input2 = torch.tanh(self.qb(G_input2))

        alpha = torch.exp(self.fc_alpha(G_input1))
        beta  = torch.tanh(self.fc_beta(G_input2))

        exp_f =  torch.exp(- alpha * strain_norm.unsqueeze(2).repeat(1,1,self.units))
        h = exp_f * (h_t  - beta) + beta

        return h.squeeze(1)


class GNN_LMSC_Model(nn.Module):
    def __init__(self, args, output_depth=3):
        super(GNN_LMSC_Model, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.seq_len = args.seq_len
        self.output_dim = args.node_output_dim
        self.batch_size = args.batch_size
        self.node_num = args.num_nodes
        self.input_dim = args.node_input_dim
        self.lmsc = GNN_LMSC_cell(args.node_input_dim, args.node_output_dim,
                                   units= args.hidden_dim, gcn_type=args.GCN_type, batch_size=args.batch_size)
        
        if args.out_type == 'FCNN':
            self.decoder = FCNN(args.layers, nn.ReLU)
        else:
            output_depth = args.out_depth
            self.decoder = Quadratic_block(args.hidden_dim, args.node_output_dim, output_depth)

    def forward(self, data):
        '''
        -Data structure
            1. x: input strain increment
            2. edge_index: grain connections in the format of adjacency list
            3. init_ori: initial grain orientations [grain_number, 3] 
        '''
        x, edge_index = data.x.type(dtype), data.edge_index
        edge_index = edge_index.to(device)

        # preprocessing for batch training
        # Input strain increment X-> [batch, seq_len, feature_dim]
        x = x.view(-1,self.seq_len, self.input_dim)
        edge_index = edge_index[:,0:edge_index.size(1)//self.batch_size]

        # Hidden state H -> [batch, node_number, hidden_dim]
        h0 = torch.zeros(x.size(0), self.node_num, self.hidden_dim)\
            .requires_grad_().to(device)
        hidden_out = torch.zeros(x.size(0), self.node_num, x.size(1),\
                                  self.hidden_dim).requires_grad_().to(device)
    
        # Assign initial orientation to hidden state
        h0[:,:,0:3] = data.init_ori.view(-1,self.node_num,3).to(device)
        h_last = h0.clone()

        # sequential prediciton
        for i in range(x.size(1)):
            x_input = x[:,i,:].unsqueeze(1).clone()  
            h_last = self.lmsc(x_input, edge_index, H = h_last)
            h_last = h_last.clone()
            hidden_out[:,:,i,:] = h_last.squeeze(1)
        
        # decode the hidden state to Output feature matrix
        # [batch, node_number, hidden_dim] -> [batch, node_number, out_dim]
        hidden_out = self.decoder(hidden_out)
        return hidden_out



class Quadratic_block(nn.Module):
    def __init__(self, input_dim, out_dim, depth):
        super(Quadratic_block, self,).__init__()
        self.modlist1 = nn.ModuleList()
        self.modlist2 = nn.ModuleList()
        self.depth = depth
        for i in range(depth):
            if i == depth-1:
                self.modlist1.append(torch.nn.Linear(input_dim, out_dim, bias=False))
                self.modlist2.append(torch.nn.Linear(input_dim, out_dim, bias=False))
            else:
                self.modlist1.append(torch.nn.Linear(input_dim, input_dim, bias=False))
                self.modlist2.append(torch.nn.Linear(input_dim, input_dim, bias=False))

    def forward(self, x):
        i = 0
        for m in self.modlist1:
            x1 = m(x)
            m2 = self.modlist2[i]
            x2 = m2(x)
            i += 1
            if i < self.depth:
                x1 = torch.tanh(x1)
                x2 = torch.tanh(x2)
            x = x1 * x2
        return x


class GCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, layer,\
                  improved=False, cached=False, add_self_loops=True):
        super(GCN_block, self,).__init__()
        self.modlist = nn.ModuleList()
        self.layer = layer
        for i in range(layer):
            if i == 0:
                self.modlist.append(GCNConv(in_channels, out_channels,\
                                             improved, cached, add_self_loops))
            else:
                self.modlist.append(GCNConv(out_channels, out_channels,\
                                             improved, cached, add_self_loops))

    def forward(self, x, edge_index):
        i = 0
        for m in self.modlist:
            x = m(x, edge_index)
            i += 1
            if i < self.layer:
                x = torch.relu(x)
        return x


class FCNN(torch.nn.Module):
    def __init__(self, layers, activation):
        super(FCNN, self).__init__()
   
        # parameters
        self.depth = len(layers) - 1        
        # set up layer order dict
        self.activation = activation
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out 