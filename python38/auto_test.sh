#!/bin/bash


test_set=$1
iters=$2
layerName="Null"
clustering=$0

for dataSet in Cora CiteSeer PubMed
do
    for layer in 0 1 2
    do
        if [ $layer -eq 0 ]; then
            layerName="GCN"
        elif [ $layer -eq 1 ]; then
            layerName="GraphSage"
        elif [ $layer -eq 2 ]; then
            layerName="GAT"
        fi
        # echo $dataSet $iters $test_set $layer $layerName
        sbatch --job-name=bc${dataSet}${layerName}_${iters}_${test_set}_${clustering} --output logs/br2_${test_set}_${iters}_${clustering}_${dataSet}_${layerName}.out --error logs/br2_${test_set}_${iters}_${dataSet}_${layerName}.err /work/ssahibul_umass_edu/broadcaster_nodes/auto_test.sbatch $dataSet $iters $test_set $layer $layerName $clustering
    done
done

# echo "weee"
# layer=0
# layerName=GCN
# dataSet=Cora
# sbatch --job-name=bc${dataSet}${layerName}_${iters}_${test_set} --output logs/br2_${test_set}_${iters}_${dataSet}_${layerName}.out --error logs/br2_${test_set}_${iters}_${dataSet}_${layerName}.err /work/ssahibul_umass_edu/broadcaster_nodes/auto_test.sbatch Cora dataSet iters test_set layer layerName
# echo "helleo"

