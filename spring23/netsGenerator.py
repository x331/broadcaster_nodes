def netTest(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[0]],[[387],[387]],[[0],[193,250,387]]]
    output_per_layer = [[hidden]]  
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest1(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[0]]]
    output_per_layer = [[16],[32],[64],[128]]
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest2(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[0]]]
    for br in [387,193,129,77,48,30]:
        br_per_layer.extend([[[0],[br]],[[br],[0]],[[br],[br]]])
    output_per_layer = [[hidden]]    
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest3(hidden): #64
    nets = []
    number_layers = [2]
    br_per_layer = []
    for br1 in [0,387,193,129,77,48,30,19]:
        for br2 in [0,387,193,129,77,48,30,19]:
            br_per_layer.append([[br1],[br2]])
    output_per_layer = [[hidden]]    
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest4(hidden): #961
    nets = []
    number_layers = [2]
    br_per_layer = []
    br_sets = []
    for br1 in [0,400,200,100,50,25]:
        br_sets.append([br1])
    for br1 in [400,200,100,50,25]:
        for br2 in [400,200,100,50,25]:
            br_sets.append([br1,br2])
    for idx1 in range(len(br_sets)):
        for idx2 in range(len(br_sets)):
            br_per_layer.append([br_sets[idx1],br_sets[idx2]])
    output_per_layer = [[hidden]]
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest5(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[0]],[[387],[387]],[[30],[30]]]
    output_per_layer = [[hidden]]  
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest6(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = []
    for br in [0,387,30,19]:
        br_per_layer.extend([[[0],[br]],[[br],[0]],[[br],[br]]])
    output_per_layer = [[hidden]]    
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest7(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[0]],[[387],[387]],[[12],[12]]]
    output_per_layer = [[hidden]]  
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest8(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = []
    br_sets = []
    for br1 in [0,400,25]:
        br_sets.append([br1])
    for br1 in [400,25]:
        for br2 in [400,25]:
            br_sets.append([br1,br2])
    for idx1 in range(len(br_sets)):
        if idx1%2 != 0 and idx1 != len(br_sets)-1 and idx1 != 0 :
            continue
        for idx2 in range(len(br_sets)):
            if idx2%2 == 0 :
                continue
            br_per_layer.append([br_sets[idx1],br_sets[idx2]])
    output_per_layer = [[hidden]]
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest9(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = []
    br_sets = []
    for br1 in [0,400,25]:
        br_sets.append([br1])
    for br1 in [400,25]:
        for br2 in [400,25]:
            br_sets.append([br1,br2])
    for idx1 in range(len(br_sets)):
        for idx2 in range(len(br_sets)):
            br_per_layer.append([br_sets[idx1],br_sets[idx2]])
    output_per_layer = [[hidden]]
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest10(hidden): #36
    nets = []
    number_layers = [2]
    br_per_layer = []
    for br1 in [0,400,200,100,50,25]:
        for br2 in [0,400,200,100,50,25]:
    # for br1 in [0,400,200]:
    #     for br2 in [0,400,200]:
            br_per_layer.append([[br1],[br2]])
    output_per_layer = [[hidden]]    
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest11(hidden): 
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[25]]]
    output_per_layer = [[hidden]]    
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest12(hidden):
    nets = []
    number_layers = [2]
    br_per_layer = []
    for br1 in [0,400,200,100]:
        for br2 in [0,400,200,100]:
    # for br1 in [0,400,200]:
    #     for br2 in [0,400,200]:
            br_per_layer.append([[br1],[br2]])
    output_per_layer = [[hidden]]    
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)

def netTest13(hidden): 
    nets = []
    number_layers = [2]
    br_per_layer = []
    br_sets = []
    br_set_first_layer = [[0],[400],[100]]
    for br1 in [0,400,100]:
        br_sets.append([br1])
    for br1 in [400,100]:
        for br2 in [400,100]:
            br_sets.append([br1,br2])
    for idx1 in range(len(br_set_first_layer)):
        for idx2 in range(len(br_sets)):
            br_per_layer.append([br_set_first_layer[idx1],br_sets[idx2]])
    output_per_layer = [[hidden]]
    return create_nets(number_layers,output_per_layer,br_per_layer,nets)


def create_nets(number_layers,output_per_layer,br_per_layer,nets):
    for layers in number_layers:
        for output_combo in output_per_layer:
            for br_combo in br_per_layer:
                nets.append([layers, list(br_combo), list(output_combo)])
    return nets

def getnet(net,argv):
    in_dataset_name = str(argv[1])
    convType= str(argv[4])
    learn_rate = argv[12] 
    learn_decay = argv[13]
    hidden = "bad"
    
    if in_dataset_name == 'Cora':
        if convType == 'GCN':
            hidden=32       
        elif convType == 'GAT':
            hidden=32
        elif convType == 'GraphSage':
            hidden=128
    elif in_dataset_name == 'CiteSeer':
        if convType == 'GCN':
            hidden=32       
        elif convType == 'GAT':
            hidden=128
        elif convType == 'GraphSage':
            hidden=128
    elif in_dataset_name == 'PubMed':
        if convType == 'GCN':
            hidden=32       
        elif convType == 'GAT':
            hidden=64
        elif convType == 'GraphSage':
            hidden=64
            
    nets = [netTest(hidden),netTest1(hidden),netTest2(hidden),netTest3(hidden),netTest4(hidden),netTest5(hidden),netTest6(hidden),netTest7(hidden),netTest8(hidden),netTest9(hidden),netTest10(hidden),netTest11(hidden),netTest12(hidden),netTest13(hidden)]
    return nets[net]


