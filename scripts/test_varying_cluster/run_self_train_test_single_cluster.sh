CUDA_VISIBLE_DEVICES=3 nohup /home/dth/miniconda3/envs/traj_rec/bin/python \
-u /home/dth/research/traj_recovery-main/test_cluster_carla.py \
--select_cluster "Strick" \
> ./log/self_train_test_cluster_231119.log 2>&1 &