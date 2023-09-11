from functools import partial
import numpy as np
import torch
from tqdm import tqdm
import sys
import faiss
import faiss.contrib.torch_utils
import torch
import numpy as np
from fast_pytorch_kmeans import KMeans
# import os
# os.environ['CUDA_PATH'] = '/modules/apps/cuda/11.3.1' 
# import pykeops
# from pykeops.torch import LazyTensor
# # pykeops.clean_pykeops() #comment back in maybe when not run multiple times in a notebook?
# pykeops.verbose = True
# pykeops.build_type = 'Debug'
# pykeops.test_torch_bindings()



def kmeansM(X,num_clusters,device,clustering, maxIters=100, centroids=None, res=None, tol=1e-4  ):
    if clustering == 0:
        return kmeansOld(X=X,num_clusters=num_clusters,device=device)
    elif clustering == -1:
        return kmeansOld(X=X,num_clusters=num_clusters,device=device,iter_limit=maxIters,cluster_centers=centroids,tol=tol)
    elif clustering == 1:
        return kmeansFaiss(X,num_clusters,device,maxIters=maxIters)
    elif clustering == 2:
        return kmeansFast(X,num_clusters,device)
    # elif clustering == 3:
    #     return kmeanskeop(X,K=num_clusters,device = device, Niter=maxIters)
    elif clustering == 4:
        return kmeansFaissGpu(X,k=num_clusters,device = device, maxIters=maxIters, centroids=centroids)
    # elif clustering == 5:
    #     return kmeansFaissGpu1(X,k=num_clusters,device = device, maxIters=maxIters)
    # elif clustering == 6:
    #     return kmeansFaissGpu2(X,k=num_clusters,device = device, maxIters=maxIters)
    elif clustering == 7:
        return kmeansFaissGpuNoRedo(X,k=num_clusters,device = device, maxIters=maxIters, centroids=centroids)
    # elif clustering == 8:
    #     return kmeansFaissGpuNoRedo1(X,k=num_clusters,device = device, maxIters=maxIters)
    # elif clustering == 9:
    #     return kmeansFaissGpuNoRedo2(X,k=num_clusters,device = device, maxIters=maxIters)   
    if clustering == 10:
        return kmeansFaissGood(X,k=num_clusters,device = device, maxIters=maxIters, centroids=centroids, res = res)
    if clustering == 11:
        return kmeansFaissGoodResets(X,k=num_clusters,device = device, maxIters=maxIters, centroids=centroids, res = res)
    else:
        print("!!!!!!!no clustering given!!!!!!!!",flush=True)
        return
    
def kmeansFaissGood(x,k,device, maxIters,centroids=None, res=faiss.StandardGpuResources()):
    # res = faiss.StandardGpuResources()
    xq_now = torch.zeros(k, x.shape[1],device=device,dtype=x.dtype)
    xq = torch.zeros(k, x.shape[1],device=device,dtype=x.dtype)
    if centroids == None:
        # starterCen = torch.randint(0, x.shape[0]-1, (k,))
        # xq = x[starterCen]
        
        # for clust in range(k):
        #         starterCen = torch.randint(0, x.shape[0]-1, (int(x.shape[0]/k*2),))
        #         temp= x[starterCen].mean(0)
        #         xq[clust] = temp
        xq = x[torch.randperm(x.shape[0])[:k]] 

    else:
        xq = centroids;
    counter=0
    for iter in range(maxIters):
        D, I = faiss.knn_gpu(res, x, xq, 1)
        for i in range(k):
            xq_now[i] = x[torch.where(I==i)[0],:].mean(0)
            
        if torch.norm(xq-xq_now)<0.0001:
            break
        else:
            counter+=1
            # xq[:,:] = xq_now[:,:]
            # xq = xq_now
            xq = xq_now.detach().clone()
    I = torch.flatten(I)
    # print(f'{counter}.......{k}')
    # del xq_now
    return I , xq, counter

def kmeansFaissGoodResets(x,k,device, maxIters,centroids=None,res=faiss.StandardGpuResources()):
    # res = faiss.StandardGpuResources()
    xq_now = torch.zeros(k, x.shape[1],device=device,dtype=x.dtype)
    xq = torch.zeros(k, x.shape[1],device=device,dtype=x.dtype)
    restart=True
    if centroids == None:
        while(restart):
            # for clust in range(k):
            #     starterCen = torch.randint(0, x.shape[0]-1, (int(x.shape[0]/k/2),))
            #     temp= x[starterCen].mean(0)
            #     xq[clust] = temp
            # starterCen = torch.randint(0, x.shape[0]-1, (k,))
            # xq = x[starterCen]
            xq = x[torch.randperm(x.shape[0])[:k]] 
            # xq = x[torch.ones(x.shape[0]).multinomial(k, replacement=True)] 
            
            D, I = None,None
            
            for iter in range(2):
                D, I = faiss.knn_gpu(res, x, xq, 1)
                for i in range(k):
                    xq_now[i] = x[torch.where(I==i)[0],:].mean(0)
                    # xq[:,:] = xq_now[:,:]
                xq = xq_now.detach().clone()

            restart=False
            for idx in range(k):
                # if torch.where(I==idx)[0].shape[0] < x.shape[0]/k/5:
                if torch.where(I==idx)[0].shape[0] < 2:

                    restart = True
    else:
        xq = centroids;
        
    # while(restart):
    #     if centroids == None:
    #         for clust in range(k):
    #             starterCen = torch.randint(0, x.shape[0]-1, (x.shape[0]/k/2.5,))
    #             temp= x[starterCen].mean(0)
    #             xq[clust] = temp
    #         # starterCen = torch.randint(0, x.shape[0]-1, (k,))
    #         # xq = x[starterCen]  
    #     else:
    #         xq = centroids;
    #         # del centroids;
    #     D, I = faiss.knn_gpu(res, x, xq, 1)
    #     restart=False
    #     for idx in range(k):
    #         if torch.where(I==idx)[0].shape[0] < x.shape[0]/k/5:
    #             restart = True
    counter = 0
    for iter in range(maxIters):
        D, I = faiss.knn_gpu(res, x, xq, 1)
        for i in range(k):
            xq_now[i] = x[torch.where(I==i)[0],:].mean(0)
            
        if torch.norm(xq-xq_now)<0.0001:
            break
        else:
            counter+=1
            # xq[:,:] = xq_now[:,:]
            xq = xq_now.detach().clone()
            

    I = torch.flatten(I)
    # del xq_now
    # print(counter)
    return I , xq, counter


# # https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html#sphx-glr-auto-tutorials-kmeans-plot-kmeans-torch-py
# # def kmeanskeop(x, K, Niter=10, device='cuda'):
# def kmeanskeop(x, K, Niter=20, device='cuda'):
#     """Implements Lloyd's algorithm for the Euclidean metric."""
#     N, D = x.shape  # Number of samples, dimension of the ambient space

#     c = x[:K, :].clone()  # Simplistic initialization for the centroids

#     x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
#     c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

#     # K-means loop:
#     # - x  is the (N, D) point cloud,
#     # - cl is the (N,) vector of class labels
#     # - c  is the (K, D) cloud of cluster centroids
#     for i in range(Niter):

#         # E step: assign points to the closest cluster -------------------------
#         D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
#         cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

#         # M step: update the centroids to the normalized cluster average: ------
#         # Compute the sum of points per cluster:
#         c.zero_()
#         c.scatter_add_(0, cl[:, None].repeat(1, D), x)

#         # Divide by the number of points per cluster:
#         Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
#         c /= Ncl  # in-place division to compute the average
        
#     return cl, c
    
    
# https://www.kernel-operations.io/keops/_auto_benchmarks/benchmark_KNN.html#faiss-approximate-and-brute-force-methods
# Load FAISS on the GPU:
# (The library pre-allocates a cache file of around ~1Gb on the device.)
def KNN_faiss_gpu(
    K,
    metric ="euclidean",
    algorithm="flat",
    nlist=8192,
    nprobe=100,
    m=None,
    use_float16=False,
    res = faiss.StandardGpuResources(),
    deviceId = 0,
    **kwargs,
):
    def fit(x_train):
        # print(type(x_train))
        # print(x_train)

        D = x_train.shape[1]

        co = faiss.GpuClonerOptions()
        co.useFloat16 = use_float16

        if metric in ["euclidean", "angular"]:

            if algorithm == "flat":
                index = faiss.IndexFlatL2(D)  # May be used as quantizer
                index = faiss.index_cpu_to_gpu(res, deviceId, index, co)

            elif algorithm == "ivfflat":
                quantizer = faiss.IndexFlatL2(D)  # the other index
                faiss_metric = (
                    faiss.METRIC_L2
                    if metric == "euclidean"
                    else faiss.METRIC_INNER_PRODUCT
                )
                index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss_metric)
                index = faiss.index_cpu_to_gpu(res, deviceId, index, co)
                
                assert not index.is_trained
                index.train(x_train)  # add vectors to the index
                assert index.is_trained

        else:
            raise NotImplementedError(f"The '{metric}' distance is not supported.")

        # Pre-processing:
        # start = timer(use_torch=False)
        index.add(x_train)
        index.nprobe = nprobe
        # elapsed = timer(use_torch=False) - startel

        # Return an operator for actual KNN queries:
        def f(x_test):
            # start = timer(use_torch=False)
            # print(type(x_test))
            # print(x_test)
            distances, indices = index.search(x_test, K)
            # elapsed = timer(use_torch=False) - start
            return indices #elapsed

        return f #,elapsed

    return fit

KNN_faiss_gpu_Flat = partial(KNN_faiss_gpu, algorithm="flat")
KNN_faiss_gpu_IVFFlat_fast = partial(KNN_faiss_gpu, algorithm="ivfflat", nlist=500, nprobe=5)
KNN_faiss_gpu_IVFFlat_slow = partial(KNN_faiss_gpu, algorithm="ivfflat", nlist=400, nprobe=40)

def kmeansFaissGpuNoRedo(x, k, device, maxIters = 10, centroids=None):
    rcentroids_now = torch.zeros(k, x.shape[1],device=device,dtype=x.dtype)
    rcentroids = torch.zeros(k, x.shape[1],device=device,dtype=x.dtype)
    if centroids == None:
        
        # for clust in range(k):
        #     starterCen = torch.randint(0, x.shape[0]-1, (int(x.shape[0]/k*2),))
        #     temp= x[starterCen].mean(0)
        #     rcentroids[clust] = temp
            
        # starterCen = torch.randint(0, x.shape[0]-1, (k,))
        # rcentroids = x[starterCen]
        
        rcentroids = x[torch.randperm(x.shape[0])[:k]]
    else:
        rcentroids = centroids;
    counter = 0;
    for iters in range(maxIters):
        knn = KNN_faiss_gpu_Flat(1)(rcentroids)(x)
        for centroid_id in range(k):
            rcentroids_now[centroid_id] = x[torch.where(knn==centroid_id)[0],:].mean(0)
        if torch.norm(rcentroids-rcentroids_now)<0.0001:
            break
        else:
            counter+=1
            # rcentroids[:,:] = rcentroids_now[:,:]
            rcentroids = rcentroids_now.detach().clone()
    knn=torch.flatten(knn)
    # print(counter)
    # del rcentroids_now
    return knn, rcentroids, counter    

def kmeansFaissGpu(x, k, device, maxIters = 10, centroids=None):
    rcentroids_now = torch.zeros(k, x.shape[1],device=device,dtype=x.dtype)
    rcentroids = torch.zeros(k, x.shape[1],device=device,dtype=x.dtype)
    restart=True
    if centroids == None:
        while(restart):
            # for clust in range(k):
            #     starterCen = torch.randint(0, x.shape[0]-1, (int(x.shape[0]/k/2),))
            #     temp= x[starterCen].mean(0)
            #     rcentroids[clust] = temp
            # starterCen = torch.randint(0, x.shape[0]-1, (k,)) 
            # rcentroids = x[starterCen] 
            rcentroids = x[torch.randperm(x.shape[0])[:k]] 
            

            knn=None
            for i in range(2):
                knn = KNN_faiss_gpu_Flat(1)(rcentroids)(x)
                for centroid_id in range(k):
                    rcentroids_now[centroid_id] = x[torch.where(knn==centroid_id)[0],:].mean(0)
                rcentroids = rcentroids_now.detach().clone()

            restart=False
            for idx in range(k):
                # if torch.where(knn==idx)[0].shape[0] < x.shape[0]/k/5:
                if torch.where(knn==idx)[0].shape[0] < 2:
                    restart = True
                    
#     if centroids == None:
#         while(restart):
#             for clust in range(k):
#                 starterCen = torch.randint(0, x.shape[0]-1, (int(x.shape[0]/k*2),))
#                 temp= x[starterCen].mean(0)
#                 rcentroids[clust] = temp
                
#             knn = None
            
#             for iter in range(2):
#                 D, I = faiss.knn_gpu(res, x, xq, 1)
#                 for i in range(k):
#                     xq_now[i] = x[torch.where(I==i)[0],:].mean(0)
#                     # xq[:,:] = xq_now[:,:]
#                 xq = xq_now.detach().clone()

#             restart=False
#             for idx in range(k):
#                 if torch.where(I==idx)[0].shape[0] < x.shape[0]/k/5:
#                     restart = True
    else:
        rcentroids = centroids
        


    counter=0;
    for iters in range(maxIters):
        knn = KNN_faiss_gpu_Flat(1)(rcentroids)(x)
        for centroid_id in range(k):
            rcentroids_now[centroid_id] = x[torch.where(knn==centroid_id)[0],:].mean(0)
        if torch.norm(rcentroids-rcentroids_now)<0.0001:
            break
        else:
            counter+=1
            # rcentroids[:,:] = rcentroids_now[:,:]
            rcentroids = rcentroids_now.detach().clone()
    knn=torch.flatten(knn)
    # print(counter)
    # del rcentroids_now
    return knn, rcentroids, counter


# https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
def kmeansFaiss(X,num_clusters,device, maxIters):
    ncentroids = num_clusters
    niter = maxIters
    verbose = False
    d = X.shape[1]
    if device != 'cpu':
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True) #nredo=3
    else:
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=False)
    numpyX = X.detach().to("cpu").numpy()
    kmeans.train(numpyX)
    D, I = kmeans.index.search(numpyX, 1)
    return torch.squeeze(torch.from_numpy(I).to(device),1), torch.from_numpy(kmeans.centroids).to(device)



#https://github.com/subhadarship/kmeans_pytorch/blob/master/example.ipynb
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

def initialize(X, num_clusters, seed):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param seed: (int) seed for kmeans
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, num_clusters, replace=False)
    else:
        np.random.seed(seed) ; indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state

def kmeansOld(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=False,
        iter_limit=2500,
        # iter_limit=0,
        device=torch.device('cuda'),
        gamma_for_soft_dtw=0.001,
        seed=None,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if tqdm_flag:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    # elif distance == 'soft_dtw':
    #     sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
    #     pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    # if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
    if type(cluster_centers) == list or cluster_centers == None:  # ToDo: make this less annoyingly weird

        initial_state = initialize(X, num_clusters, seed=seed)
    else:
        if tqdm_flag:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:

        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster, initial_state, iteration

def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        tqdm_flag=True
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """
    if tqdm_flag:
        print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    elif distance == 'soft_dtw':
        sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
        pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu'), tqdm_flag=True):
    if tqdm_flag:
        print(f'device is :{device}')
    
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)
    
#     # print(data1.shape, data2.shape)

#     # N*1*M
#     A = data1.unsqueeze(dim=1)

#     # 1*N*M
#     B = data2.unsqueeze(dim=0)
    
#     # print(A.shape, B.shape)
    
#     dis = (A - B) ** 2.0
#     # return N*N matrix for pairwise distance
#     dis = dis.sum(dim=-1).squeeze()
    
    # #Mine
    A = data1
    B=data2
    N = A.size(dim=0)
    K = B.size(dim=0)
    dis = torch.sum(A*A,dim=-1,keepdim=True).repeat(1,K) -2*(A@B.T) + torch.sum(B.T*B.T,dim=0,keepdim=True).repeat(N,1)

    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def pairwise_soft_dtw(data1, data2, sdtw=None, device=torch.device('cpu')):
    if sdtw is None:
        raise ValueError('sdtw is None - initialize it with SoftDTW')

    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # (batch_size, seq_len, feature_dim=1)
    A = data1.unsqueeze(dim=2)

    # (cluster_size, seq_len, feature_dim=1)
    B = data2.unsqueeze(dim=2)

    distances = []
    for b in B:
        # (1, seq_len, 1)
        b = b.unsqueeze(dim=0)
        A, b = torch.broadcast_tensors(A, b)
        # (batch_size, 1)
        sdtw_distance = sdtw(b, A).view(-1, 1)
        distances.append(sdtw_distance)

    # (batch_size, cluster_size)
    dis = torch.cat(distances, dim=1)
    return dis

class kmeanMem:
    def __init__(self, wait,num_clusters,device,clustering,once=False,res=None,kiters=1500):
        self.count = wait
        self.wait = wait
        self.clustering = num_clusters
        self.idexTobc = None
        self.device = device
        self.clusteringType = clustering
        self.once=once
        self.artifical_features = None
        self.res = res
        self.kiters = kiters
        
    def cluster(self,X,clusNum):
        # print(f'{self.once}....{self.clustering}....{self.artifical_features}')
        if clusNum != self.clustering:
            print('kmeanMem mismatch',flush=True)
            exit(-1)
        if self.once:
            if self.idexTobc == None:
                # print('a')
                self.idexTobc, self.artifical_features, counter = kmeansM(X=X, num_clusters=self.clustering, device=self.device,clustering=self.clusteringType, maxIters=2*self.kiters, centroids=self.artifical_features,res=self.res,)
                

            # print('b')
            # print(self.artifical_features)
            return self.idexTobc, self.artifical_features
        
        if self.count >= self.wait:
            # print('c')
            self.count = 1
            self.idexTobc, self.artifical_features, counter = kmeansM(X=X, num_clusters=self.clustering, device=self.device,clustering=self.clusteringType, maxIters=self.kiters, centroids=self.artifical_features, res=self.res)
            if counter == self.kiters:
                temp = self.artifical_features
                self.artifical_features = None
                if self.kiters > 100:
                    self.kiters = round(self.kiters*.50)
                elif self.kiters > 50:
                    self.kiters = round(self.kiters*.96)
                elif self.kiters<=20:
                    self.kiters = 20
                return self.idexTobc, temp 
                
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