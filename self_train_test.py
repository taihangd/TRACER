import os
import time
import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from queue import Queue
from itertools import chain
import threading
from itertools import islice
from utils import yaml_config_hook
from modules import network, contrastive_loss
import eval
from datasets.urban_vehicle import *
from datasets.carla import *
from cluster import *
from utils.save_model import *


def get_sample_idx_pairs(snapshot_info_list, snapshot_label_list, num_sampl_group, num_group_pairs, batch_size):
    label_snapshot_idx_dict = {}
    for i, (snapshot_info, label) in enumerate(zip(snapshot_info_list, snapshot_label_list)):
        if label not in label_snapshot_idx_dict:
            label_snapshot_idx_dict[label] = []
        label_snapshot_idx_dict[label].append(i)
    del_label_list = [label for label, curr_label_snapshot_list in label_snapshot_idx_dict.items() if len(curr_label_snapshot_list) < 2]
    for label in del_label_list:
        del label_snapshot_idx_dict[label]
    
    # batch_size must be less than the training trajectory number
    label_list = list({label for label in label_snapshot_idx_dict.keys()})
    if len(label_list) < batch_size:
        return []
    # randomly select a batch labels
    label_group_list = [random.sample(label_list, batch_size) for i in range(num_sampl_group)]

    all_pairs_list = list()
    for label_group in label_group_list:
        curr_label_snapshot_idx_list = list()
        for label in label_group:
            curr_label_snapshot_idx_list.append(label_snapshot_idx_dict[label])
        
        curr_group_pairs_list = list()
        for _ in range(num_group_pairs): # sample data pair num_group_pairs times
            feat_idx_group = list()
            for curr_label_snapshot_idx in curr_label_snapshot_idx_list:
                feat_idx_group.append(random.choice(curr_label_snapshot_idx))
            curr_group_pairs_list.append(feat_idx_group)
        all_pairs_list.append(curr_group_pairs_list)

    return all_pairs_list

def compute_features(snapshot_list, model, device, cam_vec_idx_dict_keys_tensor, cam_vec_idx_dict_values_tensor, road_graph_node_emb):
    print('Computing features...')
    model.eval()

    _, car_feat_list, plate_feat_list, time_stamp_list, lon_list, lat_list, cam_id_list, label_list = list(zip(*snapshot_list))
    time_stamp = torch.stack(time_stamp_list, dim=0)
    lon = torch.stack(lon_list, dim=0)
    lat = torch.stack(lat_list, dim=0)
    cam_id = torch.stack(cam_id_list, dim=0)
    ts = torch.cat((time_stamp, lon, lat, cam_id), dim=1).float().to(device)
    
    # retrieve the corresponding node embedding
    cam_id_idx_tensor = torch.where(cam_vec_idx_dict_keys_tensor == cam_id.to(device))
    vec_idx_tensor = cam_vec_idx_dict_values_tensor[cam_id_idx_tensor[1]]
    node_vec = road_graph_node_emb[vec_idx_tensor]

    with torch.no_grad():
        st_feat = model.forward_extract_feat(ts, node_vec)
    feature_vector = st_feat.cpu().detach().numpy()
    car_feat_vector = torch.stack(car_feat_list, dim=0).cpu().detach().numpy().astype('float32')
    plate_feat_vector = torch.stack(plate_feat_list, dim=0).cpu().numpy().astype('float32')
    labels_vector = torch.stack(label_list, dim=0).cpu().numpy().astype('float32')

    # note that the data dimensions of traversing loader or list are different
    print("computing features: st features shape {}".format(feature_vector.shape))
    print("computing features: car features shape {}".format(car_feat_vector.shape))
    print("computing features: plate features shape {}".format(plate_feat_vector.shape))

    return feature_vector, car_feat_vector, plate_feat_vector, labels_vector

def gen_cluster_multimodal_feat(snapshot_info_list, Y, model, device, cam_vec_idx_dict_keys_tensor, cam_vec_idx_dict_values_tensor, road_graph_node_emb, temperature):
    # compute spatio-temporal features based on the current model
    st_feat, car_feat, plate_feat, _ = compute_features(snapshot_info_list, model, device, cam_vec_idx_dict_keys_tensor, cam_vec_idx_dict_values_tensor, road_graph_node_emb)         

    # traverse and record the number and index of each label
    label_dict = {}
    for idx, label in enumerate(Y):
        if label in label_dict.keys():
            label_dict[label].append(idx)
        else:
            label_dict[label] = [idx]
    cluster_id_list = list(label_dict.keys())
    label_idx_dict = {cluster_id: idx for idx, cluster_id in enumerate(cluster_id_list)} # map cluster id to cluster index
    cluster_num = len(cluster_id_list)
    cluster_result_st_feat_centroids = np.zeros((cluster_num, st_feat.shape[1]))
    cluster_result_car_feat_centroids = np.zeros((cluster_num, car_feat.shape[1]))
    cluster_result_plate_feat_centroids = np.zeros((cluster_num, plate_feat.shape[1]))
    cluster_result_st_feat_density = np.zeros(cluster_num)
    cluster_result_car_feat_density = np.zeros(cluster_num)
    cluster_result_plate_feat_density = np.zeros(cluster_num)
    for cluster_id in cluster_id_list:
        cluster_st_feat_list = [st_feat[idx] for idx in label_dict[cluster_id] if not np.all(st_feat[idx] == 0)]
        if len(cluster_st_feat_list) != 0:
            st_feat_centroid = np.mean(cluster_st_feat_list, axis=0)
            cluster_result_st_feat_centroids[label_idx_dict[cluster_id]] = st_feat_centroid
        cluster_car_feat_list = [car_feat[idx] for idx in label_dict[cluster_id] if not np.all(car_feat[idx] == 0)]
        if len(cluster_car_feat_list) != 0:
            car_feat_centroid = np.mean(cluster_car_feat_list, axis=0)
            cluster_result_car_feat_centroids[label_idx_dict[cluster_id]] = car_feat_centroid
        cluster_plate_feat_list = [plate_feat[idx] for idx in label_dict[cluster_id] if not np.all(plate_feat[idx] == 0)]
        if len(cluster_plate_feat_list) != 0:
            plate_feat_centroid = np.mean(cluster_plate_feat_list, axis=0)
            cluster_result_plate_feat_centroids[label_idx_dict[cluster_id]] = plate_feat_centroid

        # concentration estimation (phi)
        if len(cluster_st_feat_list) > 1:
            st_feat_dist_list = [np.linalg.norm(feat - st_feat_centroid) for feat in cluster_st_feat_list]
            cluster_result_st_feat_density[label_idx_dict[cluster_id]] = np.asarray(st_feat_dist_list).mean() / np.log(len(st_feat_dist_list) + 10)
        if len(cluster_car_feat_list) > 1:
            car_feat_dist_list = [np.linalg.norm(feat - car_feat_centroid) for feat in cluster_car_feat_list]
            cluster_result_car_feat_density[label_idx_dict[cluster_id]] = np.asarray(car_feat_dist_list).mean() / np.log(len(car_feat_dist_list) + 10)
        if len(cluster_plate_feat_list) > 1:
            plate_feat_dist_list = [np.linalg.norm(feat - plate_feat_centroid) for feat in cluster_plate_feat_list]
            cluster_result_plate_feat_density[label_idx_dict[cluster_id]] = np.asarray(plate_feat_dist_list).mean() / np.log(len(plate_feat_dist_list) + 10)

    # if cluster only has one point, use the max to estimate its concentration        
    st_feat_phi_max = np.max(cluster_result_st_feat_density)
    car_feat_phi_max = np.max(cluster_result_car_feat_density)
    plate_feat_phi_max = np.max(cluster_result_plate_feat_density)
    for cluster_id in cluster_id_list:
        if len(label_dict[cluster_id]) <= 1:
            cluster_result_st_feat_density[label_idx_dict[cluster_id]] = st_feat_phi_max 
            cluster_result_car_feat_density[label_idx_dict[cluster_id]] = car_feat_phi_max 
            cluster_result_plate_feat_density[label_idx_dict[cluster_id]] = plate_feat_phi_max 

    # post-process centroids and density
    cluster_result_st_feat_centroids = torch.tensor(cluster_result_st_feat_centroids).type(torch.float32).to(device)
    cluster_result_st_feat_centroids = nn.functional.normalize(cluster_result_st_feat_centroids, p=2, eps=1e-12, dim=1)    
    cluster_result_car_feat_centroids = torch.tensor(cluster_result_car_feat_centroids).type(torch.float32).to(device)
    cluster_result_car_feat_centroids = nn.functional.normalize(cluster_result_car_feat_centroids, p=2, eps=1e-12, dim=1)    
    cluster_result_plate_feat_centroids = torch.tensor(cluster_result_plate_feat_centroids).type(torch.float32).to(device)
    cluster_result_plate_feat_centroids = nn.functional.normalize(cluster_result_plate_feat_centroids, p=2, eps=1e-12, dim=1)    

    cluster_result_st_feat_density = cluster_result_st_feat_density.clip(np.percentile(cluster_result_st_feat_density, 10), np.percentile(cluster_result_st_feat_density, 90)) #clamp extreme values for stability
    cluster_result_st_feat_density = temperature * cluster_result_st_feat_density / (cluster_result_st_feat_density.mean() + 1e-12)  #scale the mean to temperature 
    cluster_result_st_feat_density = torch.tensor(cluster_result_st_feat_density).type(torch.float32).to(device)
    cluster_result_car_feat_density = cluster_result_car_feat_density.clip(np.percentile(cluster_result_car_feat_density, 10), np.percentile(cluster_result_car_feat_density, 90)) #clamp extreme values for stability
    cluster_result_car_feat_density = temperature * cluster_result_car_feat_density / (cluster_result_car_feat_density.mean() + 1e-12)  #scale the mean to temperature 
    cluster_result_car_feat_density = torch.tensor(cluster_result_car_feat_density).type(torch.float32).to(device)
    cluster_result_plate_feat_density = cluster_result_plate_feat_density.clip(np.percentile(cluster_result_plate_feat_density, 10), np.percentile(cluster_result_plate_feat_density, 90)) #clamp extreme values for stability
    cluster_result_plate_feat_density = temperature * cluster_result_plate_feat_density / (cluster_result_plate_feat_density.mean() + 1e-12)  #scale the mean to temperature 
    cluster_result_plate_feat_density = torch.tensor(cluster_result_plate_feat_density).type(torch.float32).to(device)

    return cluster_result_st_feat_centroids, cluster_result_car_feat_centroids, cluster_result_plate_feat_centroids, \
            cluster_result_st_feat_density, cluster_result_car_feat_density, cluster_result_plate_feat_density, label_idx_dict

def extract_car_feat(loader, device, preload_gpu_flag):
    time_vector = []
    car_feat_vector = []
    plate_feat_vector = []
    labels_vector = []
    snapshot_info_list = []
    for step, (car_feat, plate_feat, time_stamp, lon, lat, cam_id, y) in enumerate(loader):
        time_stamp = torch.unsqueeze(time_stamp, 1)
        lon = torch.unsqueeze(lon, 1)
        lat = torch.unsqueeze(lat, 1)
        cam_id = torch.unsqueeze(cam_id, 1)
        time_vector.extend(time_stamp.detach().numpy().astype('float32'))
        car_feat = F.normalize(car_feat, dim=-1)
        plate_feat = F.normalize(plate_feat, dim=-1)
        car_feat_vector.extend(car_feat.detach().numpy().astype('float32'))
        plate_feat_vector.extend(plate_feat.detach().numpy().astype('float32'))
        labels_vector.extend(y.numpy())
        for i, (curr_car_feat, curr_plate_feat, curr_time, curr_lon, curr_lat, curr_cam_id, curr_y) in \
            enumerate(zip(car_feat, plate_feat, time_stamp, lon, lat, cam_id, torch.unsqueeze(y, 1))):
            if preload_gpu_flag:
                snapshot_info_list.append([i+step*len(time_stamp), curr_car_feat.to(device), curr_plate_feat.to(device), 
                                        curr_time.to(device), curr_lon.to(device), curr_lat.to(device), 
                                        curr_cam_id.to(device), curr_y.to(device)])
            else:
                snapshot_info_list.append([i+step*len(time_stamp), curr_car_feat, curr_plate_feat, 
                                        curr_time, curr_lon, curr_lat, curr_cam_id, curr_y])
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Extract car features...")
    time_vector = np.array(time_vector)
    car_feat_vector = np.array(car_feat_vector)
    plate_feat_vector = np.array(plate_feat_vector)
    labels_vector = np.array(labels_vector)
    print("extract car features: features shape {}".format(car_feat_vector.shape))
    
    return time_vector, car_feat_vector, plate_feat_vector, labels_vector, snapshot_info_list

def extract_st_feat(snapshot_info_list, model, device, cam_vec_idx_dict_keys_tensor, cam_vec_idx_dict_values_tensor, road_graph_node_emb):
    model.eval()
    _, _, _, time_stamp_list, lon_list, lat_list, cam_id_list, _ = zip(*snapshot_info_list)
    time_stamp = torch.stack(time_stamp_list, dim=0)
    lon = torch.stack(lon_list, dim=0)
    lat = torch.stack(lat_list, dim=0)
    cam_id = torch.stack(cam_id_list, dim=0)

    ts = torch.cat((time_stamp, lon, lat, cam_id), dim=1).float().to(device)
    # retrieve the corresponding node embedding
    cam_id_idx_tensor = torch.where(cam_vec_idx_dict_keys_tensor == cam_id.to(device))
    vec_idx_tensor = cam_vec_idx_dict_values_tensor[cam_id_idx_tensor[1]]
    node_vec = road_graph_node_emb[vec_idx_tensor]

    with torch.no_grad():
        st_feat = model.forward_extract_feat(ts, node_vec)
    feature_vector = st_feat.cpu().detach().numpy()
    print("Extract spatio-temporal features: features shape {}".format(feature_vector.shape))
    
    return feature_vector

def update_train(model, all_pairs_list, 
                 snapshot_info_list, pred_labels, 
                 label_idx_dict, 
                 road_graph_node_emb, batch_size, 
                 device, cluster_result_centroids, 
                 cluster_result_densities, 
                 min_time, max_time,
                 lamda_dict_loss, 
                 lamda_proto_loss):
    print('Training model...')
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device) # define the dictionary loss function
    
    loss_epoch = 0
    pairs_num = len(all_pairs_list)
    for pair_idx, curr_group_pairs in enumerate(all_pairs_list):
        for curr_group_list1, curr_group_list2 in zip(curr_group_pairs, curr_group_pairs[1:]):
            curr_group_snapshot_info_list1 = [snapshot_info_list[i] for i in curr_group_list1]
            curr_group_snapshot_info_list2 = [snapshot_info_list[i] for i in curr_group_list2]
            _, x_i, plate_i, ts_i, lon_i, lat_i, cam_id_i, _ = list(zip(*curr_group_snapshot_info_list1))
            _, x_j, plate_j, ts_j, lon_j, lat_j, cam_id_j, _ = list(zip(*curr_group_snapshot_info_list2))
            x_i = torch.stack(x_i, dim=0).to(device)
            plate_i = torch.stack(plate_i, dim=0).to(device)
            ts_i = torch.stack(ts_i, dim=0).to(device)
            lon_i = torch.stack(lon_i, dim=0).to(device)
            lat_i = torch.stack(lat_i, dim=0).to(device)
            cam_id_i = torch.stack(cam_id_i, dim=0).to(device)
            label_i = torch.tensor(pred_labels[curr_group_list1]).reshape(batch_size, -1)
            x_j = torch.stack(x_j, dim=0).to(device)
            plate_j = torch.stack(plate_j, dim=0).to(device)
            ts_j = torch.stack(ts_j, dim=0).to(device)
            lon_j = torch.stack(lon_j, dim=0).to(device)
            lat_j = torch.stack(lat_j, dim=0).to(device)
            cam_id_j = torch.stack(cam_id_j, dim=0).to(device)
            label_j = torch.tensor(pred_labels[curr_group_list2]).reshape(batch_size, -1)
            batch_label = torch.cat((label_i, label_j), dim=0)

            # add the random time shift perturbation
            random_perturb_array = np.random.uniform(min_time, max_time, size=tuple(ts_i.shape))
            random_perturb_array = torch.tensor(random_perturb_array, dtype=torch.float32, device=device)
            ts_i = ts_i + random_perturb_array
            ts_j = ts_j + random_perturb_array
    
            ts_info_i = torch.cat((ts_i, lon_i, lat_i, cam_id_i), dim=1)
            ts_info_j = torch.cat((ts_j, lon_j, lat_j, cam_id_j), dim=1)

            with torch.no_grad():
                x_i = x_i.type(torch.float32).to(device)
                x_j = x_j.type(torch.float32).to(device)
                plate_i = plate_i.type(torch.float32).to(device)
                plate_j = plate_j.type(torch.float32).to(device)
                ts_info_i = ts_info_i.type(torch.float32).to(device)
                ts_info_j = ts_info_j.type(torch.float32).to(device)

            # retrieve the corresponding node embedding
            cam_id_i_idx_tensor = torch.where(cam_vec_idx_dict_keys_tensor == cam_id_i.to(device))
            cam_id_j_idx_tensor = torch.where(cam_vec_idx_dict_keys_tensor == cam_id_j.to(device))
            vec_i_idx_tensor = cam_vec_idx_dict_values_tensor[cam_id_i_idx_tensor[1]]
            vec_j_idx_tensor = cam_vec_idx_dict_values_tensor[cam_id_j_idx_tensor[1]]
            node_vec_i = road_graph_node_emb[vec_i_idx_tensor]
            node_vec_j = road_graph_node_emb[vec_j_idx_tensor]
            
            # spatio-temporal feature
            output, target, st_i, st_j = model(ts_info_i, ts_info_j, node_vec_i, node_vec_j)
            dict_loss = criterion(output, target)

            batch_label = torch.Tensor([label_idx_dict[label.item()] for label in batch_label]).to(torch.int).reshape(2*batch_size, -1).to(device)
            proto_loss = criterion_multi_modal_proto_noise_contra_estimation(st_i, st_j, x_i, x_j, plate_i, plate_j, batch_label, cluster_result_centroids, cluster_result_densities)
            loss = lamda_dict_loss * dict_loss + lamda_proto_loss * proto_loss

            # train model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Train model: Step [{pair_idx}/{pairs_num}]\t loss {loss.item()}\t")
            loss_epoch += loss.item()
    
    return model, loss_epoch


if __name__ == "__main__":
    dataset_config_file = "./config/uv_self_train.yaml"
    # dataset_config_file = "./config/uv-75_self_train.yaml"
    # dataset_config_file = "./config/uv-z_self_train.yaml"
    # dataset_config_file = "./config/carla_self_train.yaml"

    parser = argparse.ArgumentParser()
    config = yaml_config_hook(dataset_config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    # for experiments with varying parameters
    parser.add_argument("--cluster_sim_thres", default=None, type=float, help='similariry threshold for inc_cluster')
    parser.add_argument("--cluster_adj_pt_ratio", default=None, type=float, help='adjency points ratio for inc_cluster')
    parser.add_argument("--cluster_spher_distrib_coeff", default=None, type=float, help='spherical distribution coefficient for inc_cluster')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cluster_sim_thres is not None:
        cluster_sim_thres = args.cluster_sim_thres
    else:
        cluster_sim_thres = args.inc_cluster['sim_thres']
    if args.cluster_adj_pt_ratio is not None:
        cluster_adj_pt_ratio = args.cluster_adj_pt_ratio
    else:
        cluster_adj_pt_ratio = args.inc_cluster['adj_pt_ratio']
    if args.cluster_spher_distrib_coeff is not None:
        cluster_spher_distrib_coeff = args.cluster_spher_distrib_coeff
    else:
        cluster_spher_distrib_coeff = args.inc_cluster['spher_distrib_coeff']

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.dataset == "UrbanVehicle":
        test_dataset = uv_img_dataset(
            record_file=args.record_file,
            cam_file=args.cam_file,
            train=False,
            use_plate=args.use_plate,
            training_traj_id_list=args.training_traj_id_list,
            test_traj_id_list=args.test_traj_id_list
        )
    elif args.dataset == "Carla":
        test_dataset = carla_img_dataset(
            record_file=args.record_file,
            cam_file=args.cam_file,
            use_plate=False,
            train=False,
            training_traj_id_list=args.training_traj_id_list,
            test_traj_id_list=args.test_traj_id_list
        )
    else:
        raise NotImplementedError
    
    # use the default sampler and collate_fn function in test module
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    
    # load road graph node embedding
    node2vec_model = KeyedVectors.load(args.road_graph_node_vec_file)
    road_graph_node_emb = node2vec_model.wv.vectors
    road_graph_node_emb = torch.from_numpy(road_graph_node_emb).type(torch.float32).to(device)
    # generate camera id to road_graph_node_emb index list
    cam_list = pickle.load(open(args.cid_rid_correspondence_file, 'rb'))
    c2r_dict = {x["id"]: x['node_id'] for x in cam_list}
    cam_vec_idx_dict= {}
    for cam_id in c2r_dict.keys():
        rid = c2r_dict[cam_id]
        vec_idx = node2vec_model.wv.key_to_index[str(rid)]
        cam_vec_idx_dict[cam_id] = vec_idx
    cam_vec_idx_dict_keys_tensor = torch.tensor(list(cam_vec_idx_dict.keys()), device=device)
    cam_vec_idx_dict_values_tensor = torch.tensor(list(cam_vec_idx_dict.values()), device=device)
    st_proj_q = network.get_st_proj(args.batch_size, args.time_feat_dim, args.time_scaling_factor, 
                                    args.mapped_feat_dim, road_graph_node_emb, device, args.st_proj_name)
    st_proj_k = network.get_st_proj(args.batch_size, args.time_feat_dim, args.time_scaling_factor, 
                                    args.mapped_feat_dim, road_graph_node_emb, device, args.st_proj_name)
    model = network.Network_MoCo(st_proj_q, st_proj_k, args.moco['dim'], args.moco['k'], args.moco['m'], args.moco['t'], args.moco['mlp'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    model_fp = os.path.join(args.model_path, "sj_proj_model_checkpoint_{}.tar".format(args.test_epoch))
    checkpoint = torch.load(model_fp, map_location=device.type)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # loss definition
    criterion_multi_modal_proto_noise_contra_estimation = contrastive_loss.MultiModalPrototypicalLoss(
        args.batch_size, args.temperature, args.inc_cluster['weights'], device).to(device)

    print("### Creating features from model ###")
    extract_car_feat_start_time = time.time()
    time_arr, car_feat, plate_feat, gt_labels, snapshot_info_list = extract_car_feat(data_loader, device, args.preload_gpu_flag) # extract features
    print("extracting features consume time:", time.time() - extract_car_feat_start_time)

    # generate data batch
    feat_num = len(car_feat)
    batch_feat_num = args.data_batch_size
    batch_time_arr_list = [time_arr[i:i+batch_feat_num] for i in range(0, feat_num, batch_feat_num)]
    batch_feat_list = [
        [car_feat[i:i+batch_feat_num], plate_feat[i:i+batch_feat_num]] 
            for i in range(0, feat_num, batch_feat_num)
        ]
    batch_snapshot_info_lists = [snapshot_info_list[i:i+batch_feat_num] for i in range(0, feat_num, batch_feat_num)]
    batch_gt_labels_list = [gt_labels[i:i+batch_feat_num] for i in range(0, feat_num, batch_feat_num)]
    
    # initialize global interaction queue
    extracted_data_queue = Queue()
    network_model_queue = Queue()
    network_model_queue.put(model)
    stop_model_trainer_thread_flag = False
    pred_label = [-1] * feat_num

    # the function to perform model inference
    traj_rec_cluster_total_time = 0
    traj_rec_extract_feat_total_time = 0
    def traj_rec_inc_cluster():
        global traj_rec_cluster_total_time
        global traj_rec_extract_feat_total_time
        # fast incremental clustering
        cluster = IncCluster(args.inc_cluster['feat_dims'], args.inc_cluster['ngpu'], args.inc_cluster['useFloat16']) # initialization
        cumul_removed_feat_num = 0 # to record the number of removal features that are too far apart
        f_ids = [[], []]
        id_dict = {}
        cfs = {}
        global pred_label
        curr_label = 0
        pres_record_num = [0, 0, 0]
        
        snapshot_info_list_iter = iter(snapshot_info_list)
        for i, (curr_batch_time_arr, curr_batch_feat, curr_batch_snapshot_info_list, curr_batch_gt_label) in \
            enumerate(zip(batch_time_arr_list, batch_feat_list, batch_snapshot_info_lists, batch_gt_labels_list)):
            # update model
            while not network_model_queue.empty(): # always get the latest models
                model = network_model_queue.get()
                model.to(device)
                print('reload model successfully!')

            print("### Creating features from model ###")
            extract_feat_time = time.time()
            curr_batch_snapshot_info = list(islice(snapshot_info_list_iter, args.data_batch_size))
            st_feat = extract_st_feat(curr_batch_snapshot_info, model, device, cam_vec_idx_dict_keys_tensor, cam_vec_idx_dict_values_tensor, road_graph_node_emb) # only extract features
            extract_feat_time = time.time() - extract_feat_time
            print("extracting features consume time:", extract_feat_time)
            # clear GPU memory
            torch.cuda.empty_cache()

            print("### clustering... ###")
            cluster_time = time.time()
            curr_batch_feat = [st_feat, curr_batch_feat[0], curr_batch_feat[1]]
            f_ids, pred_label, id_dict, cfs, curr_label, pres_record_num = cluster.fit(
                cumul_removed_feat_num,
                curr_batch_feat,
                pred_label,
                weights=args.inc_cluster['weights'], 
                sim_thres=cluster_sim_thres,
                adj_pt_ratio=cluster_adj_pt_ratio,
                spher_distrib_coeff=cluster_spher_distrib_coeff,
                topK=args.inc_cluster['topK'],
                query_num=args.inc_cluster['query_num'],
                normalization=True,
                f_ids=f_ids,
                id_dict=id_dict,
                cfs=cfs,
                curr_label=curr_label,
                pres_record_num=pres_record_num,
            )
            cluster_time = time.time() - cluster_time
            print("clustering consume time:", cluster_time)

            eval_time = time.time()
            curr_batch_label = pred_label[i * batch_feat_num: (i + 1) * batch_feat_num]
            # estimate current batch snapshots label accuracy
            if not all(label == -1 for label in curr_batch_gt_label):
                precision, recall, fscore, expansion, vid_to_cid = eval.evaluate_prf(curr_batch_gt_label, curr_batch_label)
                print('precision/recall/fscore/expansion = {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(precision, recall, fscore, expansion))
            else:
                print('current batch GT labels are all -1')
            eval_time = time.time() - eval_time
            print(f'current batch data evaluation consume time: {eval_time}')
            if args.test_time_train:
                # put the data into queue
                extracted_data_queue.put([curr_batch_time_arr, curr_batch_feat, curr_batch_snapshot_info_list, curr_batch_label])

            # clear GPU memory
            torch.cuda.empty_cache()
            traj_rec_cluster_total_time = traj_rec_cluster_total_time + cluster_time
            traj_rec_extract_feat_total_time = traj_rec_extract_feat_total_time + extract_feat_time

        # clear the searchers' memory usage
        for feat_dim in set(args.inc_cluster['feat_dims']):
            cluster.searchers[feat_dim].reset()

        return pred_label
    
    # the function to train the spatio-temporal feature extraction model in parallel
    train_network_total_time = 0
    def model_trainer(extracted_data_batch_num, model):
        global stop_model_trainer_thread_flag
        global train_network_total_time
        time_arr_list = []
        extracted_feat_list = []
        batch_snapshot_info_lists = []
        est_label_list = []
        iter_num = 0
        while True:
            if stop_model_trainer_thread_flag:
                break
            time.sleep(args.sleep_time) # wait for the data processing thread
            while not extracted_data_queue.empty() or len(extracted_feat_list) == 0: # get the extracted data in queue
                time_arr, feat, batch_snapshot_info_list, labels = extracted_data_queue.get()
                time_arr_list.append(time_arr)
                extracted_feat_list.append(feat)
                batch_snapshot_info_lists.append(batch_snapshot_info_list)
                est_label_list.append(labels)
            if len(extracted_feat_list) == 0:
                continue
            training_feat_list = extracted_feat_list[-extracted_data_batch_num:]
            snapshot_infos_list = batch_snapshot_info_lists[-extracted_data_batch_num:]
            training_labels = est_label_list[-extracted_data_batch_num:]
            print(f'estimated label list length is {len(est_label_list)}')
            print(f'training label list length is {len(training_labels)}')

            curr_batch_snapshot_info_list = list(chain.from_iterable(snapshot_infos_list))
            curr_batch_label = list(chain.from_iterable(training_labels))

            # generate a list of sample pairs for one epoch
            gen_sample_pair_time = time.time()
            curr_batch_traj_num = len(set(curr_batch_label))
            num_sampl_group = int(curr_batch_traj_num * args.sampl_group_num_coef / args.batch_size)
            num_sampl_group = min(max(num_sampl_group, args.min_num_sampl_group), args.max_num_sampl_group)
            num_group_pairs = int(float(batch_feat_num) / curr_batch_traj_num)
            num_group_pairs = min(max(num_group_pairs, args.min_num_group_pairs), args.max_num_group_pairs)
            all_pairs_list = get_sample_idx_pairs(curr_batch_snapshot_info_list, curr_batch_label, num_sampl_group, num_group_pairs, args.batch_size)
            if len(all_pairs_list) == 0:
                print('the number of labels is less than the number of sampling groups')
                continue
            gen_sample_pair_time = time.time() - gen_sample_pair_time
            print("generate sample pairs consume time:", gen_sample_pair_time)
            
            gen_cluster_multimodal_feat_time = time.time()
            cluster_result_st_feat_centroids, \
            cluster_result_car_feat_centroids, \
            cluster_result_plate_feat_centroids, \
            cluster_result_st_feat_density, \
            cluster_result_car_feat_density, \
            cluster_result_plate_feat_density, \
            label_idx_dict = gen_cluster_multimodal_feat(curr_batch_snapshot_info_list, 
                                                        curr_batch_label, model, device, 
                                                        cam_vec_idx_dict_keys_tensor, 
                                                        cam_vec_idx_dict_values_tensor, 
                                                        road_graph_node_emb, 
                                                        args.temperature)
            gen_cluster_multimodal_feat_time = time.time() - gen_cluster_multimodal_feat_time
            print("cluster multimodal feature generation consume time:", gen_cluster_multimodal_feat_time)
            
            # update and train the network for one epoch during inference
            update_network_time = time.time()
            model, _ = update_train(model, all_pairs_list, snapshot_info_list, np.array(curr_batch_label), 
                                    label_idx_dict, road_graph_node_emb, args.batch_size, device, 
                                    [cluster_result_st_feat_centroids, cluster_result_car_feat_centroids, cluster_result_plate_feat_centroids], 
                                    [cluster_result_st_feat_density, cluster_result_car_feat_density, cluster_result_plate_feat_density],
                                    args.min_time, args.max_time, args.lamda_dict_loss, args.lamda_proto_loss)
            update_network_time = time.time() - update_network_time
            print("update train network consume time:", update_network_time)
            
            # put the updated model into queue
            train_network_total_time = train_network_total_time + gen_sample_pair_time + gen_cluster_multimodal_feat_time + update_network_time
            network_model_queue.put(model)
            save_model(args.model_path, model.state_dict(), optimizer.state_dict(), 'sj_proj_model_test', iter_num)
            iter_num += 1
            print('save self-training model during test dataset successfully!')

            # clear GPU memory
            torch.cuda.empty_cache()

            if stop_model_trainer_thread_flag:
                break

    train_model_thread = threading.Thread(target=model_trainer, args=(args.extracted_data_batch_num, model))
    traj_rec_thread = threading.Thread(target=traj_rec_inc_cluster)

    if args.test_time_train:
        train_model_thread.start()
    
    recover_traj_time = time.time()
    traj_rec_thread.start()
    traj_rec_thread.join()
    recover_traj_time = time.time() - recover_traj_time
    
    stop_model_trainer_thread_flag = True
    if args.test_time_train:
        train_model_thread.join()

    print("recover trajectories consume time:", recover_traj_time)
    print("cluster total consume time:", traj_rec_cluster_total_time)
    print("extract features total consume time:", traj_rec_extract_feat_total_time)
    print("network training consume time:", train_network_total_time)
    # evaluate inference results
    torch.cuda.empty_cache() # clear GPU memory
    precision, recall, fscore, expansion, vid_to_cid = eval.evaluate_prf(gt_labels, pred_label)
    print('precision/recall/fscore/expansion = {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(precision, recall, fscore, expansion))
    
    save_traj_rec_result(args, pred_label, gt_labels, vid_to_cid)
    print('save recovered trajectory results successfully!')
