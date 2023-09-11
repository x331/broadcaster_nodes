from broadcaster_nets_2_3 import *

from torch_geometric.datasets import Planetoid
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
from torch_geometric.nn import GCNConv

import torch.nn as N

import itertools

import csv
from time import perf_counter

import copy


def netTest2layer(slurmID, in_dataset, iterations, in_nets, convType, clustering):
    
    dataset = in_dataset
    
    outcomes = []

    iters = iterations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)

    nets = in_nets

    for idx in range(len(nets)):
        outcomeP = {}
        outcomeT = {}
        for runs in range(iters):
            runTime = perf_counter()
            model = brGCN2_3(data,dataset,nets[idx][0],nets[idx][1],nets[idx][2],convType, slurmID, iters,clustering).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            for epoch in range(200):
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()

            model.eval()
            pred = model(data).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())
            rtime = perf_counter() - runTime
            print(f'loop:{runs} Accuracy: {acc:.4f} Time: {rtime:.4f}')
            outcomeP[runs] = acc
            outcomeT[runs] = rtime
        
        avgPerf = sum(outcomeP.values())/len(outcomeP.values())
        avgTime = sum(outcomeT.values())/len(outcomeT.values())
                                         
        print(f'average: {avgPerf},....{avgTime}')
        y = copy.deepcopy(model.name)
        y.extend([avgTime,avgPerf])
        outcomes.append(y)
 
    print(outcomes)

    for entry in outcomes:
        print(f'layers:{entry[2]}||gpu:{entry[1]}||clustering:{entry[3]}||convType:{entry[5]}||layer_out_dim:{entry[6]}||br_multiplies:{entry[7]}||time:{entry[8]}||average:{entry[9]}')
        
    with open('logs/'+slurmID, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["slurm id","gpu","iterations","clustering", "number of layers","conv type","hidden layer dims", "br multiplies", "avg time", f"{iterations}-avg accuracy"])
        csvwriter.writerows(outcomes)
        
        
         # [slurmID, iters, num_of_layers,layer_out_dims,br_multiplies]
            
