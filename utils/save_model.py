import os
import datetime
import pickle
import matplotlib.pyplot as plt
import torch


def save_model(model_path, model_state_dict, optimizer_state_dict, model_flag, current_epoch):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    out = os.path.join(model_path, "{}_checkpoint_{}.tar".format(model_flag, current_epoch))
    state = {'net': model_state_dict, 'optimizer': optimizer_state_dict, 'epoch': current_epoch}
    torch.save(state, out)

def save_record_feat(args, st_feat, car_feat, plate_feat):
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)

    curr_datetime = datetime.datetime.now()
    datetime_str = f"{curr_datetime.strftime('%Y-%m-%d_%H-%M-%S_')}"
    record_feat_file = os.path.join(args.res_path, datetime_str + args.record_feat_file)
    pickle.dump([st_feat, car_feat, plate_feat], open(record_feat_file, 'wb'))
    print(f'save {datetime_str} record features successfully!')
    return

def save_traj_rec_result(args, X, Y, vid_to_cid):
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)

    curr_datetime = datetime.datetime.now()
    datetime_str = f"{curr_datetime.strftime('%Y-%m-%d_%H-%M-%S_')}"
    traj_res_file = os.path.join(args.res_path, datetime_str + args.traj_res_file)
    pickle.dump([X, Y, vid_to_cid], open(traj_res_file, 'wb'))
    return

def plot_loss(loss_list, save_file=None):
    plt.figure(figsize=(5, 9), dpi=200)
    
    x = list(range(len(loss_list)))
    plt.plot(x, loss_list, 'r', lw=1)
    plt.title("training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend("train_loss")

    if save_file is not None:
        plt.savefig(save_file)
    
    return
