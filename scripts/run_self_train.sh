CUDA_VISIBLE_DEVICES=1 nohup /home/dth/miniconda3/envs/traj_rec/bin/python \
-u /home/dth/research/traj_recovery-main/self_train.py \
> ./log/carla_sampl_self_train_231110.log 2>&1 &