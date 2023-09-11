from kmeans_old import *

def bc_data_pp_2(data, num_bc_nodes, device,clustering,kmean_set):
    x = data
    bc_features = torch.empty((0,),device=device)
    bc_assigment = torch.empty((0,),device=device)
    for bc_set in range(len(num_bc_nodes)):
        if num_bc_nodes[bc_set] == 0: 
            continue
        cluster_ids_x, artifical_features = kmean_set[bc_set].cluster(data,num_bc_nodes[bc_set])
        artifical_features = artifical_features.to(device)
        bc_features = torch.cat((bc_features,artifical_features),dim=0)
        bc_assigment = torch.cat((bc_assigment,cluster_ids_x),dim=0)
    return bc_features, bc_assigment

def bc_data_pp_2_2(data, num_bc_nodes, device,clustering,kmean_set):
    x = data
    bc_features = torch.empty((0,),device=device)
    bc_assigment = torch.empty((0,),device=device)
    for bc_set in range(len(num_bc_nodes)):
        if num_bc_nodes[bc_set] == 0: 
            continue
        cluster_ids_x, artifical_features = kmean_set[bc_set].cluster(data,num_bc_nodes[bc_set])
        artifical_features = artifical_features.to(device)
        bc_features = torch.cat((bc_features,artifical_features),dim=0)
        bc_assigment = torch.cat((bc_assigment,cluster_ids_x),dim=0)
    return bc_features, bc_assigment

def bc_data_pp_1_overlap(data, num_bc_nodes, device):
    self.data = data

    x, edge_index = self.data.x, self.data.edge_index

    cluster_ids_x, artifical_features = kmeans(X=x, num_clusters=num_bc_nodes, device=device) # tqdm_flag=False

    broadcasters_edge_list = torch.zeros(2,x.size()[0], dtype=torch.long, device=device )
    artifical_features = artifical_features.to(device)

    for idx in range(0, cluster_ids_x.size()[0]): 
        broadcasters_edge_list[0][idx] = cluster_ids_x[idx]+x.size()[0]
        broadcasters_edge_list[1][idx] = idx

    edges_inbtw_1 = torch.zeros(2,2*num_bc_nodes*x.size()[0], dtype=torch.long, device=device)
    edges_inbtw_counter = 0

    for c in range(edge_index.size()[1]):
        node1 = edge_index[0][c]
        node2 = edge_index[1][c]

        if broadcasters_edge_list[0][node1] != broadcasters_edge_list[0][node2]:
            edges_inbtw_1[0][edges_inbtw_counter] = broadcasters_edge_list[0][node2]
            edges_inbtw_1[1][edges_inbtw_counter] = node1
            edges_inbtw_counter +=1
            edges_inbtw_1[0][edges_inbtw_counter] = broadcasters_edge_list[0][node1]
            edges_inbtw_1[1][edges_inbtw_counter] = node2
            edges_inbtw_counter +=1

    x = torch.cat((x,artifical_features),dim=0)
    edge_index = torch.cat((edge_index,edges_inbtw_1,broadcasters_edge_list),dim=-1)
    edge_index = to_undirected(edge_index)

    self.data.x = x
    self.data.edge_index = edge_index

    self.data.y = torch.cat((self.data.y,torch.zeros(num_bc_nodes, dtype=torch.long, device=torch.device('cuda'))))

    falseCat = torch.zeros(num_bc_nodes, dtype=torch.bool, device = torch.device('cuda'))
    self.data.train_mask = torch.cat((self.data.train_mask,falseCat))
    self.data.val_mask = torch.cat((self.data.val_mask,falseCat))
    self.data.test_mask = torch.cat((self.data.test_mask,falseCat))
    return self.data

def bc_data_pp_1(data, num_bc_nodes, device):
    self.data = data
    x, edge_index = self.data.x, self.data.edge_index
    
    for bc_set in num_bc_nodes:
        cluster_ids_x, artifical_features = kmeans(X=x, num_clusters=bc_set, device=device) # tqdm_flag=False

        broadcasters_edge_list = torch.zeros(2,x.size()[0], dtype=torch.long, device=device )
        artifical_features = artifical_features.to(device)

        for idx in range(0, cluster_ids_x.size()[0]): 
            broadcasters_edge_list[0][idx] = cluster_ids_x[idx]+x.size()[0]
            broadcasters_edge_list[1][idx] = idx

        self.data.x = torch.cat((x,artifical_features),dim=0)
        self.data.edge_index = to_undirected(torch.cat((edge_index,broadcasters_edge_list),dim=-1))

    self.data.y = torch.cat((self.data.y,torch.zeros(sum(num_bc_nodes), dtype=torch.long, device=torch.device('cuda'))))
    falseCat = torch.zeros(sum(num_bc_nodes), dtype=torch.bool, device = torch.device('cuda'))
    self.data.train_mask = torch.cat((self.data.train_mask,falseCat))
    self.data.val_mask = torch.cat((self.data.val_mask,falseCat))
    self.data.test_mask = torch.cat((self.data.test_mask,falseCat))
    return self.data
