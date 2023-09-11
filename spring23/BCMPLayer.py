from typing import Optional

import torch

import torch_geometric.utils 
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class BCMPLayer1(torch.nn.Module):
    def __init__(self,dimWX,dimWZ,dimwalpha,device):
        # print("here i")
        super().__init__()
        self.WX = torch.nn.Linear(dimWX[0],dimWX[1],bias=False)
        self.WZ = torch.nn.Linear(dimWZ[0],dimWZ[1],bias=False)
        self.Walpha = torch.nn.Linear(dimwalpha[0],dimwalpha[1],bias=False)
        self.fuse = lambda Xprime,Zprime,Zalpha : torch.mean(torch.stack((Xprime,Zprime,Zalpha),dim=2),dim=2)
        self.fuse0 = lambda Xold,Xprime : torch.mean(torch.stack((Xold,Xprime),dim=2),dim=2)
        # self.fuse = lambda Xold,Xprime,Zprime,Zalpha : torch.sum(torch.stack((Xold,Xprime,Zprime,Zalpha),dim=2),dim=2)
        # self.fuse0 = lambda Xold,Xprime : torch.sum(torch.stack((Xold,Xprime),dim=2),dim=2)
        self.device = device    
        
    def forward(self,x,edge_index,bc_feature=None,bc_assigment=None): #case for if no bc 
        # print("here f")
        A = torch.sparse_coo_tensor(edge_index,torch.ones(edge_index.size(dim=1),device=self.device),(x.size(dim=0),x.size(dim=0)),device=self.device)
        if  bc_assigment ==None and bc_feature == None:
            Xprime = self.WX(x)
            Aselfloop = torch.eye(A.size(dim=0),device=self.device)+A#no longer sparse     make torch.eye spare and use torch.sparse.add
            D = torch.diag(torch.pow(torch.sum(Aselfloop,dim=0),-.5))
            Ahat = D@Aselfloop@D
            Ahat = Ahat.to_sparse()
            Xprime = torch.sparse.mm(Ahat,Xprime)
            return Xprime
        
        Z = bc_feature        
        bc_assigment_edges = torch.vstack((torch.arange(0,(bc_assigment.size(dim=0)),device=self.device),bc_assigment)) # non symetric
        B =  torch.sparse_coo_tensor(bc_assigment_edges,torch.ones(bc_assigment_edges.size(dim=1),device=self.device),(x.size(dim=0),bc_feature.size(dim=0)),device=self.device) #rectangle
        #square b                             
        # bc_assigment_edges = torch.tensor([torch.range(0,(bc_feature.size(dim=0))-1,device=self.device),torch.add(bc_assigment+x.size(dim=0))]device=self.device) # non symetric
        # # bc_assigment_edges_2 = torch.tensor([torch.add(bc_assigment+x.size(dim=0)),torch.range(0,(bc_feature.size(dim=0))-1,device=self.device)],device=self.device) # reverse symetric
        # # bc_assigment_edges = torch.cat((bc_assigment_edges,bc_assigment_edges_2),dim=1) # symetric
        # B =  torch.sparse_coo_tensor(bc_assigment_edges,torch.ones(bc_assigment_edges.size(dim=0)),(x.size(dim=0)+bc_feature.size(dim=0),x.size(dim=0)+bc_feature.size(dim=0),device=self.device) #square
        
        Xprime = self.WX(x)
        # print('X',Xprime.size())
        # Xold = Xprime
        Zprime = self.WZ(Z)
        Zalpha = self.Walpha(Z)
        Aselfloop = torch.eye(A.size(dim=0),device=self.device)+A#no longer sparse     make torch.eye spare and use torch.sparse.add
        Bselfloop = torch.eye(n=B.size(dim=0),m=B.size(dim=1),device=self.device)+B#no longer sparse
        D = torch.diag(torch.pow(torch.sum(Aselfloop,dim=0),-.5))
        Drow = torch.diag(torch.pow(torch.sum(Bselfloop,dim=1),-.5)) # include self-loop
        Dcol = torch.diag(torch.pow(torch.sum(torch.transpose(Bselfloop,0,1),dim=1),-.5)) # include self-loop 
        Ahat = D@Aselfloop@D
        Bhat = Drow@Bselfloop@Dcol
        Ahat = Ahat.to_sparse()
        Bhat = Bhat.to_sparse()
        # Xprime = torch_geometric.utils.spmm(Ahat.to_sparse(),Xprime,"sum")
        # Zprime = torch_geometric.utils.spmm(Bhat.to_sparse(),Zprime,"sum")
        # Zalpha = torch_geometric.utils.spmm(Ahat.to_sparse(),spmm(Bhat,Zalpha,"sum"),"sum")
        Xprime = torch.sparse.mm(Ahat,Xprime)
        Zprime = torch.sparse.mm(Bhat,Zprime)
        Zalpha = torch.sparse.mm(Ahat,torch.sparse.mm(Bhat,Zalpha))       
        return self.fuse(Xprime,Zprime,Zalpha)  
    
    
    
    
    
    
class BCMPLayer2(torch.nn.Module):
    def __init__(self,dimWX,dimWZ,dimwalpha,BaseLayer,device,norm1=None,norm2=None):  #implement norm
        #
        super().__init__()
        self.fuse = lambda Xprime,Zprime,Zalpha : torch.mean(torch.stack((Xprime,Zprime,Zalpha),dim=2),dim=2)
        self.fuse0 = lambda Xold,Xprime : torch.mean(torch.stack((Xold,Xprime),dim=2),dim=2)
        
        self.fuse_cat = lambda Xprime,Zprime,Zalpha : torch.cat((Xprime,Zprime,Zalpha),dim=1)
        self.fuse0_cat = lambda Xold,Xprime : torch.cat((Xold,Xprime),dim=1)
        
        self.layerNorm0 = torch.nn.LayerNorm([dimWX[1]])
        self.layerNorm1 = torch.nn.InstanceNorm2d(dimWX[1], affine=True)
       
        self.device = device 
        
        
        self.baselayer1 =  BaseLayer(dimWX[0],dimWX[1]) # example GCNConv
        self.baselayer5 =  BaseLayer(dimWX[0],dimWX[0],bias=False) # example GCNConv
        self.baselayer6 =  BaseLayer(dimWX[0]*2,dimWX[1]) # example GCNConv

        
        if BaseLayer != GATConv:
            self.baselayer2 =  BaseLayer(dimWX[0],dimWX[1],add_self_loops=False,normalize=False,bias=False) # example GCNConv #turn off bias and normilization
            # self.baselayer3 =  BaseLayer(dimWX[0],dimWX[0],add_self_loops=False) # example GCNConv #can get rid of
            self.baselayer4 =  BaseLayer(dimWX[1],dimWX[1],add_self_loops=False,normalize=False,bias=False) # example GCNConv
        else:
            self.baselayer2 =  BaseLayer(dimWX[0],dimWX[1],add_self_loops=False,bias=False) # example GCNConv #turn off bias and normilization
            self.baselayer4 =  BaseLayer(dimWX[1],dimWX[1],add_self_loops=False,bias=False) # example GCNConv
        
        # self.baselayer2 =  BaseLayer(dimWX[0],dimWX[1],add_self_loops=False,bias=False) # example GCNConv #turn off bias and normilization
        # self.baselayer4 =  BaseLayer(dimWX[1],dimWX[1],add_self_loops=False,bias=False)
        
        self.norm1 = norm1
        self.norm2 = norm2
        self.squeze = torch.nn.Linear(3*dimWX[1],dimWX[1],bias=False)
        self.squeze0 = torch.nn.Linear(2*dimWX[1],dimWX[1],bias=False)

    
    def forward(self,x,edge_index,bc_feature=None,bc_assigment=None,bset=0): #case for if no bc 
        if  bc_assigment ==None and bc_feature == None:
            Xprime = self.baselayer1.forward(x,edge_index)
            return Xprime
        
        Z = bc_feature #normalize
        bc_assigment_edges = None
        for b in range(bset):
            # bc_assigment_edges_add = torch.vstack((torch.arange(0,(x.size(dim=0)),device=self.device,dtype=torch.long), bc_assigment [b*x.size(dim=0):(b+1)*x.size(dim=0)].type(torch.long))) #switch this up and check \ this one is wrong
            bc_assigment_edges_add = torch.vstack((bc_assigment[b*x.size(dim=0):(b+1)*x.size(dim=0)].type(torch.long),torch.arange(0,(x.size(dim=0)),device=self.device,dtype=torch.long)))
            if b == 0:
                bc_assigment_edges = bc_assigment_edges_add
            else:
                bc_assigment_edges = torch.cat((bc_assigment_edges,bc_assigment_edges_add),dim=1)
        
        # xZ = torch.vstack((torch.zeros(tuple(x.size()),device = self.device),Z))
        xZ = torch.vstack((x,Z))
        
#         Xprime = self.baselayer6.forward(torch.cat((x,Z[bc_assigment.type(torch.cuda.LongTensor)]),dim=1),edge_index)
#         return Xprime
        
        
#         Xprime = self.baselayer5.forward(xZ,bc_assigment_edges)
#         Xprime = Xprime[0:x.size(dim=0), :]
#         Xprime = self.baselayer1.forward(Xprime,edge_index)
#         return Xprime

#         Xprime = self.baselayer1.forward(xZ,torch.cat((edge_index,bc_assigment_edges),1))
#         return Xprime[0:x.size(dim=0), :]
        
        Xprime = self.baselayer1.forward(x,edge_index)  # calculate magintude of the features of Xprime and Zprime and compare them
        Zprime = self.baselayer2.forward(xZ,bc_assigment_edges) #add_self_loops = False in init and use real x #might normalize by 1/number of clustering
        # Za1 = self.baselayer3(xZ,bc_assigment_edges) #can get rid of
        # Zalpha = self.baselayer4.forward(Za1,edge_index) #can get rid of
        
        Zalpha = self.baselayer4.forward(Zprime,edge_index)
        
        Zprime = Zprime[0:x.size(dim=0), :]
        Zalpha = Zalpha[0:x.size(dim=0), :]
        
        # print('-----------')
        # print('Xprime Norm', torch.norm(Xprime))
        # print('Xprime-1 Norm', torch.norm(Xprime[0]))
        # print('Zprime Norm', torch.norm(Zprime))
        # print('Zprime-1 Norm', torch.norm(Zprime[0]))
                                  
        # normalized = self.layerNorm0(torch.cat((Xprime,Zprime),dim=0)) #could try instansts norm
        normalized = self.layerNorm0(torch.cat((Xprime,Zprime,Zalpha),dim=0)) #could try instansts norm

        
        
        # print('-----------')
        # print(normalized.shape)
        
        Xprime = normalized[:x.shape[0]]
        Zprime = normalized[x.shape[0]:-x.shape[0]]
        Zalpha = normalized[-x.shape[0]:]

        # print(Xprime.shape)
        # print('NXprime Norm', torch.norm(Xprime))
        # print('NXprime-1 Norm', torch.norm(Xprime[0]))
        # print('NZprime Norm', torch.norm(Zprime))
        # print('NZprime-1 Norm', torch.norm(Zprime[0]))
        # print('NZalpha Norm', torch.norm(Zalpha))
        # print('NZalpha-1 Norm', torch.norm(Zalpha[0]))
                                  
              
        
        # return self.fuse0(Xprime,Zprime)
        # return self.fuse0(Xprime,Zalpha)
        # return self.fuse(Xprime,Zprime,Zalpha) 
        
        # return self.squeze0(self.fuse0_cat(Xprime,Zprime))
        # return self.squeze0(self.fuse0_cat(Xprime,Zalpha))
        return self.squeze( self.fuse_cat(Xprime,Zprime,Zalpha) ) 
        
#######################################################################################################################################################
#######################################################################################################################################################
#######################################################################################################################################################

        
class BCMPLayer3(torch.nn.Module):
    def __init__(self,dimWX,dimWZ,dimwalpha,BaseLayer,device,norm1=None,norm2=None,k=None,kset=[]):  #implement norm
        #
        super().__init__()     
        self.device = device
        self.k = k
        self.kset = kset
        self.baselayer1 =  BaseLayer(dimWX[0],dimWX[1])
        self.baselayer6 =  BaseLayer(dimWX[0]*k,dimWX[1])
        self.baselayer7 =  BaseLayer(dimWX[0],dimWX[1]) 
        self.baselayer8 =  BaseLayer(dimWX[0]*(k-1),dimWX[1]) 
        self.baselayer10 =  BaseLayer(dimWX[0]*(k-1),dimWX[1]) 
        self.lin1 = torch.nn.Linear(dimWX[1]*2,dimWX[1],bias=False)



    
    def forward(self,x,edge_index,bc_feature=None,bc_assigment=None,bset=0): #case for if no bc 
        if  bc_assigment ==None and bc_feature == None:
            Xprime = self.baselayer7.forward(x,edge_index)
            return Xprime
        
        Z = bc_feature
        
        # #zz1
        # Zreshaped = Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)]
        # for k in range(self.k-2):
        #     Zreshaped = torch.cat((Zreshaped,Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)]),dim=1)
        # #zz2
        # Zreshaped = torch.div(Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)],self.kset[0])
        # for k in range(self.k-2):
        #     Zreshaped = torch.cat((Zreshaped,torch.div(Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)],self.kset[k+1])),dim=1)
        # #zz3
        # Zreshaped = torch.div(Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)],x.shape[0]/self.kset[0])
        # for k in range(self.k-2):
        #     Zreshaped = torch.cat((Zreshaped,torch.div(Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)], x.shape[0]/self.kset[0])),dim=1)      ############need to fix   self.kset[0]/should be /self.kset[k+1]
        
        ##c1
        # Xprime = self.baselayer6.forward(torch.cat((x,Zreshaped),dim=1),edge_index)
        ##c2
        # Xprime = self.baselayer7.forward(x,edge_index) + self.baselayer8.forward(Zreshaped,edge_index)
        ##c3
        # Xprime = self.lin1(torch.cat((self.baselayer7.forward(x,edge_index), self.baselayer8.forward(Zreshaped,edge_index)),dim=1))
        
    
        # #z1
        # Zreshaped = Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)]
        # for k in range(self.k-2):
        #     Zreshaped = torch.stack((Zreshaped,Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)]),dim=0)
        #     if k == self.k-3 :
        #         Zreshaped = torch.mean(Zreshaped,dim=0)
        # #z2
        # Zreshaped = Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)]
        # for k in range(self.k-2):
        #     Zreshaped = Zreshaped+Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)]
        # #z3
        # Zreshaped = torch.div(Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)],self.kset[0])
        # for k in range(self.k-2):
        #     Zreshaped = torch.stack((Zreshaped,torch.div(Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)],self.kset[k+1])),dim=0)
        #     if k == self.k-3 :
        #         Zreshaped = torch.mean(Zreshaped,dim=0)
        # #z4       
        # Zreshaped = torch.div(Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)],self.kset[0])
        # for k in range(self.k-2):
        #     Zreshaped = Zreshaped + torch.div(Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)],self.kset[k+1])  
        # #z5
        # Zreshaped = torch.div(Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)],x.shape[0]/self.kset[0])
        # for k in range(self.k-2):
        #     Zreshaped = torch.stack((Zreshaped,torch.div(Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)],x.shape[0]/ self.kset[k+1])),dim=0)
        #     if k == self.k-3 :
        #         Zreshaped = torch.mean(Zreshaped,dim=0) 
        # #z6        
        Zreshaped = torch.div(Z[bc_assigment[:x.shape[0]].type(torch.cuda.LongTensor)],x.shape[0]/self.kset[0])
        for k in range(self.k-2):
            Zreshaped = Zreshaped + torch.div(Z[bc_assigment[(k+1)*x.shape[0]:(k+2)*x.shape[0]].type(torch.cuda.LongTensor)],x.shape[0]/self.kset[k+1])
        
        ##c4
        # Xprime = self.baselayer7.forward(torch.mean(torch.stack((x,Zreshaped),dim=0),dim=0),edge_index)
        ##c5
        Xprime = self.baselayer7.forward(x+Zreshaped,edge_index)
        

        

        return Xprime
        
        
    
    