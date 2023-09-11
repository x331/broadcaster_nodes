#!/bin/bash

layerName="Null" ################
cluster_wait=0 ###########
clustering=-1 ############ 
iters=20     #############
test_set=13   ############ 
logFile=logs_z6c5/  

# test_set=3  ############
# iters=1    #############
# layerName="Null"  ################
# cluster_wait=1 ################
# clustering=3
# logFile=logs_test/  

for dataSet in Cora #CiteSeer PubMed
do
    for layer in  0 # 1 2
    do
        if [ $layer -eq 0 ]; then
            layerName="GCN"
            learn_rate=.01
            lr_decay=.001
        elif [ $layer -eq 1 ]; then
            layerName="GraphSage"
            learn_rate=.001
            lr_decay=.001
            if [ $dataSet == CiteSeer ]; then
                learn_rate=.01
                lr_decay=.0005
            fi
        elif [ $layer -eq 2 ]; then
            layerName="GAT"
            learn_rate=.001
            lr_decay=.001
            if [ $dataSet == Cora ]; then
                learn_rate=.01
            fi
        fi
        # echo ${layerName}_${dataSet}_test${test_set}_iter${iters}_clus${clustering}_wait${cluster_wait}_lr${learn_rate}_lrdecay${lr_decay}
        sbatch --job-name=bc${layerName}_${dataSet}_test${test_set}_iter${iters}_clus${clustering}_wait${cluster_wait}_lr${learn_rate}_lrdecay${lr_decay} --output ${logFile}br2_${layerName}_${dataSet}_test${test_set}_iter${iters}_clus${clustering}_wait${cluster_wait}_lr${learn_rate}_lrdecay${lr_decay}_%j.out --error ${logFile}br2_${layerName}_${dataSet}_test${test_set}_iter${iters}_clus${clustering}_wait${cluster_wait}_lr${learn_rate}_lrdecay${lr_decay}_%j.err /work/ssahibul_umass_edu/broadcaster_nodes/spring23/auto_test.sbatch $dataSet $iters $test_set $layerName $clustering $cluster_wait $learn_rate $lr_decay $logFile
        # exit 5  ###############################################################################################
    done
done


