from broadcaster_nets_2_3 import *

import torch

from torch_geometric.datasets import Planetoid

import csv
from time import perf_counter

import copy

from netsGenerator import *

import faiss
import faiss.contrib.torch_utils

import statistics

import gc

from guppy import hpy

def netTest2layer(argv):
    slurmID = argv[0]
    in_dataset_name = argv[1]
    iters= argv[2]
    in_nets_number= argv[3]
    convType= argv[4]
    
    gpu_name = argv[5]
    clustering_type = argv[6]
    argv[7] = Planetoid(root='/tmp/'+argv[1], name=argv[1])
    dataset= argv[7]
    
    argv[8] = getnet(in_nets_number,argv)
    nets= argv[8]
    cluster_wait = argv[11]
    learn_rate = argv[12] 
    learn_decay = argv[13]
    argv[14] = faiss.StandardGpuResources()
    faiss_res = argv[14]

    outcomes = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    
    argv[10] = data
    
    with open(slurmID+'.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["slurm id", "learn_rate", "learn_decay" , "gpu", "iterations", "clustering type", "cluster wait", "number of layers", "dataset", "conv type","hidden layer dims", "k-proportional", "k-number", "nodes/cluster", f"{iters}-stddev","avg time",f"{iters}-avg peak train epoch" ,f"{iters}-avg peak train acc", f"{iters}-avg end train acc" , f"{iters}-avg test accuracy"])
        
    for idx in range(len(nets)):
        outcomeP = {}
        outcomeT = {}
        mOutcomeP = {}
        mOutcomePNumber = {}
        for runs in range(iters):
            runTime = perf_counter()
            model = brGCN2_3(data,dataset,nets[idx][0],nets[idx][1],nets[idx][2],convType, slurmID, iters,clustering_type,gpu_name,argv).to(device)
            # torch.autograd.set_detect_anomaly(True)
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=learn_decay)

            training_max_acc = 0
            training_max_acc_epoch_num = 0
            endTrainAvg = 0
            # for epoch in range(200):
            for epoch in range(200):
                model.train()
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                # loss.backward(retain_graph=True)
                loss.backward()

                optimizer.step()
                
                model.eval()
                pred = model(data).argmax(dim=1)
                correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
                acc = int(correct) / int(data.test_mask.sum())
                if training_max_acc < acc:
                    training_max_acc = acc
                    training_max_acc_epoch_num = epoch
                endTrainAvg = acc
                
                

            model.eval()
            pred = model(data).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())
            rtime = perf_counter() - runTime
            print(f'loop:{runs} Accuracy: {acc:.4f} Time: {rtime:.4f}',flush=True)
            outcomeP[runs] = acc
            mOutcomeP[runs] = training_max_acc
            mOutcomePNumber[runs] = training_max_acc_epoch_num
            outcomeT[runs] = rtime

            
        stddev = statistics.stdev(outcomeP.values()) if len(outcomeP.values()) >=2 else 0
        avgPerf = sum(outcomeP.values())/len(outcomeP.values())
        MavgPerf = sum(mOutcomeP.values())/len(mOutcomeP.values())
        MavgPerfNumber = sum(mOutcomePNumber.values())/len(mOutcomePNumber.values())
        avgTime = sum(outcomeT.values())/len(outcomeT.values())
                                         
        print(f'average: {avgPerf},...{stddev},... {model.name[13]}...{avgTime}, ... {MavgPerfNumber} ... {MavgPerf} ... {endTrainAvg}',flush=True)
        y = copy.deepcopy(model.name)
        
        del model
        torch.cuda.empty_cache()
        gc.collect
        
#         # for obj in gc.get_objects():
#         #     try:
#         #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#         #             print(type(obj), obj.size(),flush=True)
#         #     except:
#         #         pass
        # print("----------.....................c.......................--------",flush=True)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False),flush=True)
        # print("----------.....................d.......................--------",flush=True)
        # h = hpy()
        # print(h.heap(),flush=True)
        print("----------.....................e.......................--------",flush=True)

        
        y.extend([stddev,avgTime,MavgPerfNumber,MavgPerf,endTrainAvg,avgPerf])
        outcomes.append(y)
        
        with open(slurmID+'.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows([y])
 
    print(outcomes,flush=True)

    for entry in outcomes:
        print(f'layers:{entry[4]}||learn rate:{entry[1]}||learn decay:{entry[2]}||gpu:{entry[3]}||clustering_type:{entry[5]}||clustering_wait:{entry[6]}||num_layers:{entry[7]}||dataset:{entry[8]}||convType:{entry[9]}||layer_out_dim:{entry[10]}||k-proportional:{entry[11]}||k-number:{entry[12]}||nodes/clusters:{entry[13]}||stddev:{entry[14]}||time:{entry[15]}||train peak:{entry[16]}||avg peak train acc:{entry[17]}||end train acc:{entry[18]}||avg test acc:{entry[19]}',flush=True)
        
    # with open(slurmID+'.csv', 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(["slurm id", "learn_rate", "learn_decay" , "gpu", "iterations", "clustering type", "cluster wait", "number of layers", "dataset", "conv type","hidden layer dims", "br multiplies", f"{iters}-stddev","avg time", f"{iters}-avg accuracy"])
    #     csvwriter.writerows(outcomes)
        
    