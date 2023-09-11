from torch_geometric.datasets import Planetoid
import torch
from layer2Test import netTest2layer
from netsGenerator import *
from time import perf_counter
import sys

runTime = perf_counter()
print(f"{sys.argv[1]}; {sys.argv[2]}; inter:{sys.argv[3]}; net set:{sys.argv[4]}; conv type:{sys.argv[5]}, gpu:{torch.cuda.get_device_name(0)}, cluster:{sys.argv[6]}")
netTest2layer(sys.argv[1], Planetoid(root='/tmp/'+sys.argv[2], name=sys.argv[2]),int(sys.argv[3]), getnet(int(sys.argv[4])), int(sys.argv[5]), int(sys.argv[6])) 

print(perf_counter() - runTime) 
