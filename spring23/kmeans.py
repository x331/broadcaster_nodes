# from functools import partial
import torch
import sys
import faiss
import faiss.contrib.torch_utils
import torch




def kmeansM(X,num_clusters,device,clustering, maxIters=2500, centroids=None):
    if clustering == 10:
        return kmeansFaissGood(X,k=num_clusters,device = device, maxIters=maxIters, centroids=centroids)
    else:
        print("!!!!!!!no clustering given!!!!!!!!")
        return

    
# newest re-implementaion of kmean's with faiss
def kmeansFaissGood(X,k,device, maxIters,centroids=None):
    res = faiss.StandardGpuResources()
    xq_now = torch.zeros(k, X.shape[1],device=device,dtype=X.dtype)
    if centroids == None:
        starterCen = torch.randint(0, X.shape[0]-1, (k,))
        xq = X[starterCen]
    else:
        xq = centroids;
    # counter=0
    for iter in range(maxIters):
        D, I = faiss.knn_gpu(res, X, xq, 1)
        for i in range(k):
            xq_now[i] = X[torch.where(I==i)[0],:].mean(0)
            
        if torch.norm(xq-xq_now)<0.0001:
            break
        else:
            # counter+=1
            xq[:,:] = xq_now[:,:]
    I = torch.flatten(I)
    return I , xq

class kmeanMem:
    def __init__(self, wait,num_clusters,device,clustering,once=False,kiters=1500):
        self.count = wait
        self.wait = wait
        self.clustering = num_clusters
        self.idexTobc = None
        self.device = device
        self.clusteringType = clustering
        self.once=once
        self.artifical_features = None
        self.kiters = 1500
        
    def cluster(self,X,clusNum):
        # print(f'{self.once}....{self.clustering}....{self.artifical_features}')
        if clusNum != self.clustering:
            print('kmeanMem mismatch')
            exit(-1)
        if self.once:
            if self.idexTobc == None:
                # print('a')
                self.idexTobc, self.artifical_features = kmeansM(X=X, num_clusters=self.clustering, device=self.device,clustering=self.clusteringType, maxIters=5000, centroids=self.artifical_features)
                
                while(torch.any(self.idexTobc.isnan())):
                    print("nan")
                    self.idexTobc, self.artifical_features = kmeansM(X=X, num_clusters=self.clustering, device=self.device,clustering=self.clusteringType, maxIters=5000, centroids=self.artifical_features)

            # print('b')
            # print(self.artifical_features)
            return self.idexTobc, self.artifical_features
        
        if self.count >= self.wait:
            # print('c')
            self.count = 1
            self.idexTobc, self.artifical_features = kmeansM(X=X, num_clusters=self.clustering, device=self.device,clustering=self.clusteringType, maxIters=2500, centroids=self.artifical_features)
            
            while(torch.any(self.idexTobc.isnan())):
                print("nan")
                self.idexTobc, self.artifical_features = kmeansM(X=X, num_clusters=self.clustering, device=self.device,clustering=self.clusteringType, maxIters=5000, centroids=self.artifical_features)
                
            return self.idexTobc, self.artifical_features
        
        elif self.clusteringType > -1:
            # print('d')
            self.count = self.count+1
#             rcentroids=torch.zeros(self.clustering,X.shape[1],device=self.device,dtype=X.dtype)
#             for centroid_id in range(self.clustering):
#                 k = torch.where(self.idexTobc==centroid_id)[0]
#                 j = torch.sum(X[k,:],dim=0)
#                 rcentroids[centroid_id] = torch.div(j,k.shape[0])
            for centroid_id in range(self.clustering):
                k = torch.where(self.idexTobc==centroid_id)[0]
                j = torch.sum(X[k,:],dim=0)
                self.artifical_features[centroid_id] = torch.div(j,k.shape[0])
       
            
            
            return self.idexTobc, self.artifical_features
        else:
            print("no clustering given")
            return
        