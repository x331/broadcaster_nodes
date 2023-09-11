def netTest():
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[0]],[[1],[1]]]
    # br_per_layer = [[[0],[0]],[[1],[1]],[[1],[2,1]], [[1],[2,1,3]]]
    output_per_layer = [[8]]  
    for layers in number_layers:
        for output_combo in output_per_layer:
            for br_combo in br_per_layer:
                nets.append([layers, list(br_combo), list(output_combo)])
    return nets

def nets2():
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[0]],[[1],[1]],[[2],[2]],[[3],[3]],[[5],[5]],[[0],[1]],[[1],[0]],[[2],[0]],[[3],[0]],[[5],[0]],[[5],[1]],[[3],[1]],[[2],[1]],[[5],[2]],[[3],[2]],[[3],[1]],[[2,1],[2,1]],[[5,3],[2,1]],[[5,1],[0]],[[1,2],[3,5]]]
    output_per_layer = [[16],[32],[64]]  
    for layers in number_layers:
        for output_combo in output_per_layer:
            for br_combo in br_per_layer:
                nets.append([layers, list(br_combo), list(output_combo)])
    return nets

def nets3():
    nets = []
    number_layers = [2]
    br_per_layer = [[[0],[0]],[[2],[1]],[[2,1],[2,1]],[[0],[2,1]], [[2,1],[0]],[[5,1],[0]],[[0],[5,1]],[[5,3],[2,1]],[[1,2],[3,5]],[[3,5],[3,5]],[[5,3],[3,5]],[[0,0],[3,5]],[[3],[5]],[[10,5],[3,2]],[[2,3],[5,10]],[[0,0],[5,10]],[[5,10],[5,10]]]
    output_per_layer = [[32]]
    for layers in number_layers:
        for output_combo in output_per_layer:
            for br_combo in br_per_layer:
                nets.append([layers, list(br_combo), list(output_combo)])
    return nets

def getnet(net):
    nets = [netTest(),nets2(),nets3()]
    return nets[net]


