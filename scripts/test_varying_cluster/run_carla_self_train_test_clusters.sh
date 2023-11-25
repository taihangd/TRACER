#!/bin/bash

# define parameter list
cluster_list=('MMVC' 'Strick' 'K-Means' 'agglomerative' 'HDBSCAN' 'PDBSCAN')

date_time='231123' # date time as the log name
dataset_name='carla' # need to set the corresponding dataset in the test python file

for cluster in "${cluster_list[@]}"
do
    echo "Running with cluster argument: ${cluster}"
    
    CUDA_VISIBLE_DEVICES=3 nohup /home/dth/miniconda3/envs/traj_rec/bin/python \
    -u  /home/dth/research/traj_recovery-main/test_cluster_carla.py \
    --select_cluster "$cluster" \
    > "./log/self_train_test_${dataset_name}_cluster_${cluster}_${date_time}_visualfeat.log" 2>&1 &

    pid=$! # get process ID
    echo "Current process ID: ${pid}"
    wait $pid # wait for the process to finish
    echo "Process ${pid} with cluster argument ${cluster} has been finished!"
done
