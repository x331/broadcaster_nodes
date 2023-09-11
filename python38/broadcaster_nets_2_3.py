from bc_data_pp_v2 import *

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.loader import ClusterData
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

import torch.nn as N

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class brGCN2_3(torch.nn.Module):
    def __init__(self,data,dataset,num_of_layers,br_multiplies, layer_out_dims,convType,slurmID,iters,clustering):
        super().__init__()
        self.dataset = dataset
        self.layers = N.ModuleList()
        self.num_of_layers = num_of_layers
        self.br_multiplies = br_multiplies
        self.layer_out_dims = layer_out_dims
        self.clustering = clustering
        self.name = [slurmID, torch.cuda.get_device_name(0), iters, clustering, num_of_layers, convType, layer_out_dims, br_multiplies]
        self.brMulti = []
        for layer_num in range(num_of_layers):
            if 0 in br_multiplies[layer_num]:
                self.brMulti.append(1)
            else:
                self.brMulti.append(len(br_multiplies[layer_num])+1)
                
            # self.brMulti.append( (2 if (len(br_multiplies[layer_num])==1 and br_multiplies[layer_num][0]!=0) else (3 if (len(br_multiplies[layer_num])==2 and br_multiplies[layer_num][0]!=0 and br_multiplies[layer_num][1]!=0) else 1))  )
            
            numFeatures = (dataset.num_node_features if layer_num==0 else layer_out_dims[layer_num-1])
            arg2 = dataset.num_classes if (layer_num == (num_of_layers-1)) else layer_out_dims[layer_num]
            if(convType ==0):
                self.layers.append(GCNConv(self.brMulti[layer_num]*numFeatures, arg2))
            elif(convType ==1):
                self.layers.append(SAGEConv(self.brMulti[layer_num]*numFeatures, arg2))
            elif(convType ==2):
                self.layers.append(GATConv(self.brMulti[layer_num]*numFeatures, arg2))
            else:
                print("out of range conv type")
            
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer_num in range(len(self.layers)):
            if 0 not in self.br_multiplies[layer_num]:
                x = bc_data_pp_2(x,([self.dataset.num_classes*x for x in self.br_multiplies[layer_num]]),device,self.clustering)
            #####            x = bc_data_pp_2(x,(self.dataset.num_classes*self.br_multiplies[layer_num]),device)
            # if self.brMulti[layer_num] == 2 :
            #     x = bc_data_pp_2(x,(self.dataset.num_classes*self.br_multiplies[layer_num][0]),device)
            # elif self.brMulti[layer_num] == 3 :
            #     x = bc_data_pp_3(x,(self.dataset.num_classes*self.br_multiplies[layer_num][0]), (self.dataset.num_classes*self.br_multiplies[layer_num][1]), device)
            if layer_num != 0:
                x = F.relu(x) 
                x = F.dropout(x, training=self.training)
            x = self.layers[layer_num](x,edge_index)
        return F.log_softmax(x, dim=1)