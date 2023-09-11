from time import perf_counter
import sys
from layer2Test import netTest2layer

import torch

runTime0 = perf_counter()
argv = []
argv.insert(0, sys.argv[1])    #slurm id
argv.insert(1, sys.argv[2])    #in dataset name
argv.insert(2,int(sys.argv[3]))    #iters
argv.insert(3,int(sys.argv[4]))    #in net number
argv.insert(4,sys.argv[5])    #conv type
argv.insert(5,torch.cuda.get_device_name())    #gpu name
argv.insert(6,int(sys.argv[6])) # clus type
argv.insert(7,None)    #dataset
argv.insert(8,None)    #nets
# argv.insert(8,getnet(int(sys.argv[4])))    #nets
if sys.argv[5] == 'GCN':
    argv.insert(9,0)    #conv number
elif sys.argv[5] == 'GraphSage':
    argv.insert(9,1)    #conv number
elif sys.argv[5] == 'GAT':
    argv.insert(9,2)    #conv number
argv.insert(10,None) #data
argv.insert(11,int(sys.argv[7])) # clus wait
argv.insert(12,float(sys.argv[8])) # lr
argv.insert(13,float(sys.argv[9])) # lr decay
argv.insert(14,None) # faiss res

# print(f"{sys.argv[1]}; {sys.argv[2]}; inter:{sys.argv[3]}; net set:{sys.argv[4]}; conv type:{sys.argv[5]}, gpu:{torch.cuda.get_device_name(0)}, cluster:{sys.argv[6]}")
# netTest2layer(sys.argv[1], Planetoid(root='/tmp/'+sys.argv[2], name=sys.argv[2]),int(sys.argv[3]), getnet(int(sys.argv[4])), int(sys.argv[5]), int(sys.argv[6]), torch.cuda.get_device_name(0))
print(f"{argv[0]}; {argv[1]}; iter:{argv[2]}; net set:{argv[3]}; conv type:{argv[4]}, gpu:{argv[5]}, cluster:{argv[6]}, cluster wait:{argv[11]}, lr:{argv[12]}, lr decay:{argv[13]} ",flush=True)
netTest2layer(argv) 

print(perf_counter() - runTime0) 
