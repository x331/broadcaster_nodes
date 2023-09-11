from bc_data_pp_v2 import *

from BCMPLayer import *

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

import torch.nn as N

import kmeans




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class brGCN2_3(torch.nn.Module):
    def __init__(self,data,dataset,num_of_layers,br_multiplies, layer_out_dims,convType,slurmID,iters,clustering,gpu,argv):
        super().__init__()
        # slurmID = argv[0]
        in_dataset_name = argv[1]
        # iters= argv[2]
        # in_nets_number= argv[3]
        # convType= argv[4]
        # gpu_name = argv[5]
        # clustering_type = argv[6]
        # self.dataset= argv[7]
        # nets= argv[8]
        # data = argv[10]
        cluster_wait = argv[11]
        learn_rate = argv[12] 
        learn_decay = argv[13]
        faiss_res = argv[14]
            
        self.dataset = dataset
        self.layers = N.ModuleList()
        # self.layers = []
        self.num_of_layers = num_of_layers
        self.br_multiplies = br_multiplies
        self.layer_out_dims = layer_out_dims
        self.clustering = clustering

        self.brMulti = []
        self.kmean = []
        
        
        

        knumber = [[] for row in br_multiplies]
        for layer in range(len(br_multiplies)):
            for bc_set in br_multiplies[layer]:
                bc_set_number = round(data.x.shape[0]/bc_set) if bc_set != 0 else 0
                knumber[layer].append(bc_set_number)
                
        kproportional = [[] for row in br_multiplies]
        for layer in range(len(br_multiplies)):
            for bc_set in br_multiplies[layer]:
                bc_set_number = round(round(data.x.shape[0]/bc_set)/dataset.num_classes,2) if bc_set != 0 else 0
                kproportional[layer].append(bc_set_number)
                
        kper = br_multiplies
                
                
                
        self.name = [slurmID, learn_rate, learn_decay, gpu, iters, clustering, cluster_wait, num_of_layers, in_dataset_name, convType, layer_out_dims, kproportional, knumber, kper]
        
        
        
        
        for layer in range(len(br_multiplies)):
            layerSet = []
            for bc_set in br_multiplies[layer]:
                bc_set_number = round(data.x.shape[0]/bc_set) if bc_set != 0 else 0
                once = True if layer == 0 else False
                layerSet.append(kmeanMem(cluster_wait ,bc_set_number, device, clustering, once,faiss_res))
            self.kmean.append(layerSet)
        
        for layer_num in range(num_of_layers):
            if 0 in br_multiplies[layer_num]:
                self.brMulti.append(1)
            else:
                self.brMulti.append(len(br_multiplies[layer_num])+1)
            
            numFeatures = (dataset.num_node_features if layer_num==0 else layer_out_dims[layer_num-1])
            arg2 = dataset.num_classes if (layer_num == (num_of_layers-1)) else layer_out_dims[layer_num]
            # if(convType =='GCN'):
            #     # self.layers.append(GCNConv(self.brMulti[layer_num]*numFeatures, arg2))
            #     self.layers.append(BCMPLayer1((numFeatures, arg2),(numFeatures, arg2),(numFeatures, arg2),device))
            # elif(convType =='GraphSage'):
            #     # self.layers.append(SAGEConv(self.brMulti[layer_num]*numFeatures, arg2))
            #     self.layers.append(GCNConv(self.brMulti[layer_num]*numFeatures, arg2))
            # elif(convType =='GAT'):
            #     # self.layers.append(GATConv(self.brMulti[layer_num]*numFeatures, arg2))
            #     self.layers.append(GCNConv(self.brMulti[layer_num]*numFeatures, arg2))
            # else:
            #     print("out of range conv type",flush=True)
                
            if(convType =='GCN'):
                # self.layers.append(GCNConv(self.brMulti[layer_num]*numFeatures, arg2))
                self.layers.append(BCMPLayer3((numFeatures, arg2),(numFeatures, arg2),(numFeatures, arg2),GCNConv,device,k=self.brMulti[layer_num], kset=self.br_multiplies[layer_num]))
            elif(convType =='GraphSage'):
                # self.layers.append(SAGEConv(self.brMulti[layer_num]*numFeatures, arg2))
                self.layers.append(BCMPLayer3((numFeatures, arg2),(numFeatures, arg2),(numFeatures, arg2),SAGEConv,device,k=self.brMulti[layer_num]))
            elif(convType =='GAT'):
                # self.layers.append(GATConv(self.brMulti[layer_num]*numFeatures, arg2))
                self.layers.append(BCMPLayer3((numFeatures, arg2),(numFeatures, arg2),(numFeatures, arg2),GATConv,device,k=self.brMulti[layer_num]))
            else:
                print("out of range conv type",flush=True)
                
        self.lastLin = torch.nn.Linear(dataset.num_classes, dataset.num_classes, bias=False, device=device)
            
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer_num in range(len(self.layers)):
            # print(layer_num)
            if 0 not in self.br_multiplies[layer_num]:
                bc_features, bc_assigment = bc_data_pp_2(x,([round(data.x.shape[0]/x) for x in self.br_multiplies[layer_num]]), device,self.clustering, self.kmean[layer_num])
            else:
                bc_features, bc_assigment = None, None
            if layer_num != 0:
                x = F.relu(x) 
                x = F.dropout(x, training=self.training)
            # print('here')    
            x = self.layers[layer_num].forward(x,edge_index,bc_features,bc_assigment,bset=len(self.br_multiplies[layer_num]))
            # print(x.is_cuda)
            # print(x)
            # print(x.size())
        x = self.lastLin(x)
        return F.log_softmax(x, dim=1)