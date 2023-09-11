import torch
import numpy as np
import faiss
import faiss.contrib.torch_utils
import time
import matplotlib.pyplot as plt

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

d = 2
nb = 100
nc = 3
k = 1
max_iter = 100
res = faiss.StandardGpuResources()

# # make GT on torch cpu and test using IndexFlatL2
xb1 = (torch.rand(nb, d, dtype=torch.float32) + torch.Tensor([[1.0,0.0]])).to(device)
xb2 = (torch.rand(nb, d, dtype=torch.float32) + torch.Tensor([[-1.0,1.0]])).to(device)
xb3 = (torch.rand(nb, d, dtype=torch.float32) + torch.Tensor([[-1.0,-1.0]])).to(device)
xb = torch.cat([xb1,xb2,xb3],dim=0).to(device)

# xb = torch.rand(nb, d, dtype=torch.float32).to(device)
xq = torch.rand(nc, d, dtype=torch.float32).to(device)

# plt.figure()
# plt.scatter(xb[:,0].cpu().numpy(),xb[:,1].cpu().numpy())
# plt.scatter(xq[:,0].cpu().numpy(),xq[:,1].cpu().numpy(),color='r')
s = time.time()

xq_now = torch.zeros(nc, d, dtype=torch.float32).to(device)
for iter in range(max_iter):
    plt.figure()
    plt.scatter(xq[:,0].cpu().numpy(),xq[:,1].cpu().numpy(),color='r')
    
    D, I = faiss.knn_gpu(res, xb, xq, k)
    
    for i in range(nc):
        xq_now[i] = xb[torch.where(I==i)[0],:].mean(0)
        
    
    colors = ['g','b','pink']
    for i in range(nc):
        assignment = torch.where(I==i)[0]
        plt.scatter(xb[assignment,0].cpu().numpy(),xb[assignment,1].cpu().numpy(), color=colors[i])
        
    if torch.norm(xq-xq_now)<0.0001:
        break
    else:
        xq[:,:] = xq_now[:,:]
    

# t = time.time()
# print(t-s)