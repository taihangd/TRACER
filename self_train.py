import os
import numpy as np
import argparse
import pickle
import random
import torch
from gensim.models import KeyedVectors
import eval
from modules import network, contrastive_loss
from modules.contrastive_loss import *
from datasets.urban_vehicle import *
from datasets.carla import *
from utils import yaml_config_hook
from utils.save_model import *
from cluster import *


def gen_snapshot_list(data_loader, device, preload_gpu_flag=False):
    snapshot_info_list = list()
    for idx, (img_feat, plate_feat, time, lon, lat, cam_id, label) in tqdm(enumerate(data_loader)):
        if preload_gpu_flag:
            snapshot_info_list.append((idx, img_feat[0].to(device), plate_feat[0].to(device), time.to(device), lon.to(device), lat.to(device), cam_id.to(device), label.to(device)))
        else:
            snapshot_info_list.append((idx, img_feat[0], plate_feat[0], time, lon, lat, cam_id, label))

    return snapshot_info_list

def get_sample_idx_pairs(snapshot_label_list, num_sampl_group, num_group_pairs, batch_size):
    label_snapshot_idx_dict = {}
    for i, label in enumerate(snapshot_label_list):
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

def extract_feat(loader, model, device, cam_vec_idx_dict, road_graph_node_emb):
    print('Computing features...')
    model.eval()
    feature_vector = []
    car_feat_vector = []
    plate_feat_vector = []
    labels_vector = []
    for step, (car_feat, plate_feat, time, lon, lat, cam_id, y) in enumerate(loader):
        time = torch.unsqueeze(time, 1)
        lon = torch.unsqueeze(lon, 1)
        lat = torch.unsqueeze(lat, 1)
        cam_id = torch.unsqueeze(cam_id, 1)
        ts = torch.cat((time, lon, lat, cam_id), dim=1).float().to(device)
        # retrieve the corresponding node embedding
        node_vec = torch.stack([road_graph_node_emb[cam_vec_idx_dict[id.item()]] for id in cam_id])
        with torch.no_grad():
            c = model.forward_extract_feat(ts, node_vec)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        car_feat_vector.extend(car_feat.detach().numpy().astype('float32'))
        plate_feat_vector.extend(plate_feat.detach().numpy().astype('float32'))
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    car_feat_vector = np.array(car_feat_vector)
    plate_feat_vector = np.array(plate_feat_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    
    return feature_vector, car_feat_vector, plate_feat_vector, labels_vector

def gen_cluster_multimodal_feat(eval_loader, pred_label, model, device, cam_vec_idx_dict, road_graph_node_emb, temperature):
    # compute spatio-temporal features based on the current model
    st_feat, car_feat, plate_feat, _ = extract_feat(eval_loader, model, device, cam_vec_idx_dict, road_graph_node_emb)         
    
    # traverse and record the number and index of each label
    label_dict = {}
    for idx, label in enumerate(pred_label):
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

def train(all_pairs_list, snapshot_info_list, 
          pred_labels, label_idx_dict, road_graph_node_emb, 
          batch_size, device, cluster_result_centroids, 
          cluster_result_densities, min_time, max_time, 
          lamda_dict_loss, lamda_proto_loss):
    # switch to train mode
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device) # define loss function (criterion)
    
    loss_epoch = 0
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
            label_i = torch.tensor(pred_labels[curr_group_list1], device=device).reshape(batch_size, -1)
            x_j = torch.stack(x_j, dim=0).to(device)
            plate_j = torch.stack(plate_j, dim=0).to(device)
            ts_j = torch.stack(ts_j, dim=0).to(device)
            lon_j = torch.stack(lon_j, dim=0).to(device)
            lat_j = torch.stack(lat_j, dim=0).to(device)
            cam_id_j = torch.stack(cam_id_j, dim=0).to(device)
            label_j = torch.tensor(pred_labels[curr_group_list2], device=device).reshape(batch_size, -1)
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

            print(f"Step [{pair_idx}/{len(all_pairs_list)}]\t loss: {loss.item()}\t")
            loss_epoch += loss.item()

    return loss_epoch


if __name__ == "__main__":
    # configuration file setting
    dataset_config_file = "./config/uv.yaml"
    # dataset_config_file = "./config/uv-75.yaml"
    # dataset_config_file = "./config/uv-z.yaml"
    # dataset_config_file = "./config/carla.yaml"
    
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(dataset_config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # prepare data
    if args.dataset == "UrbanVehicle":
        train_dataset = uv_img_dataset(
            record_file=args.record_file,
            cam_file=args.cam_file,
            train=True,
            use_plate=args.use_plate,
            training_traj_id_list=args.training_traj_id_list,
            test_traj_id_list=args.test_traj_id_list
        )
    elif args.dataset == "Carla":
        train_dataset = carla_img_dataset(
            record_file=args.record_file,
            cam_file=args.cam_file,
            use_plate=False,
            train=True,
            training_traj_id_list=args.training_traj_id_list,
            test_traj_id_list=args.test_traj_id_list
        )
    else:
        raise NotImplementedError

    # use the default sampler and collate_fn function in evaluation data module for cluster generation
    eval_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    # generate training feature dataloader by sequential reading
    train_data_loader_snapshots = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
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
    model = model.to(device)
    
    # optimizer and loss definition
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    criterion_multi_modal_proto_noise_contra_estimation = contrastive_loss.MultiModalPrototypicalLoss(
        args.batch_size, args.temperature, args.inc_cluster['weights'], device).to(device)

    num_sampl_group = int(train_dataset.traj_num * args.sampl_group_num_coef / args.batch_size)
    num_group_pairs = int(train_dataset.record_num / train_dataset.traj_num) # set the pairs number in the group
    snapshot_info_list = gen_snapshot_list(train_data_loader_snapshots, device, args.preload_gpu_flag)

    # fast incremental clustering initialization
    cluster = FastIncCluster(args.inc_cluster['feat_dims'], args.inc_cluster['ngpu'], args.inc_cluster['useFloat16'])

    # train main loop
    train_network_time = time.time()
    cluster_result_st_feat_centroids, cluster_result_st_feat_density = None, None
    cluster_result_car_feat_centroids, cluster_result_car_feat_density = None, None
    cluster_result_plate_feat_centroids, cluster_result_plate_feat_density = None, None
    cluster_result_centroids_validation, cluster_result_density_validation = None, None
    loss_list = list()
    for epoch in range(args.start_epoch, args.epochs):
        # get the initialized pseudo labels
        extract_feat_time = time.time()
        st_feat, car_feat, plate_feat, gt_labels = extract_feat(eval_data_loader, model, device, cam_vec_idx_dict, road_graph_node_emb) # extract features
        extract_feat_time = time.time() - extract_feat_time
        print("extracting features consume time:", extract_feat_time)
    
        cluster_time = time.time()
        # improved fast incremental SigCluster
        cumul_removed_feat_num = 0 # to record the number of removal features that are too far apart
        f_ids = [[], []]
        id_dict = {}
        cfs = {}
        curr_label = 0
        pres_record_num = [0, 0, 0]
        feat_num = len(car_feat)
        pred_label = [-1] * feat_num
        f_ids, pred_label, id_dict, cfs, curr_label, pres_record_num = cluster.fit(
            cumul_removed_feat_num,
            [st_feat, car_feat, plate_feat],
            pred_label,
            weights=args.inc_cluster['weights'], 
            sim_thres=args.inc_cluster['sim_thres'],
            adj_pt_ratio=args.inc_cluster['adj_pt_ratio'],
            spher_distrib_coeff=args.inc_cluster['spher_distrib_coeff'],
            topK=args.inc_cluster['topK'],
            query_num=args.inc_cluster['query_num'],
            normalization=True,
            f_ids=f_ids,
            id_dict=id_dict,
            cfs=cfs,
            curr_label=curr_label,
            pres_record_num=pres_record_num,
        )
        # clear the searchers' memory usage
        for feat_dim in set(args.inc_cluster['feat_dims']):
            cluster.searchers[feat_dim].reset()
        torch.cuda.empty_cache() # free up GPU memory
        cluster_time = time.time() - cluster_time
        print("clustering consume time:", cluster_time)
        
        # estimate current snapshot label accuracy
        eval_time = time.time()
        precision, recall, fscore, expansion, vid_to_cid = eval.evaluate_prf(gt_labels, pred_label)
        eval_time = time.time() - eval_time
        print('precision/recall/fscore/expansion = {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(precision, recall, fscore, expansion))
        print('evaluation consume time {}'.format(eval_time))

        if epoch >= args.warmup_epoch:
            gen_cluster_multimodal_feat_time = time.time()
            cluster_result_st_feat_centroids, \
            cluster_result_car_feat_centroids, \
            cluster_result_plate_feat_centroids, \
            cluster_result_st_feat_density, \
            cluster_result_car_feat_density, \
            cluster_result_plate_feat_density, \
            label_idx_dict = gen_cluster_multimodal_feat(eval_data_loader, pred_label, model, device, 
                                                        cam_vec_idx_dict, road_graph_node_emb, 
                                                        args.temperature)
            gen_cluster_multimodal_feat_time = time.time() - gen_cluster_multimodal_feat_time
            print(f'generate cluster multi-modal feature consume time: {gen_cluster_multimodal_feat_time}')

        # generate a list of sample pairs for one epoch
        get_sample_idx_pairs_time = time.time()
        all_pairs_list = get_sample_idx_pairs(pred_label, num_sampl_group, num_group_pairs, args.batch_size)
        get_sample_idx_pairs_time = time.time() - get_sample_idx_pairs_time
        print(f'get sample idx pairs consume time: {get_sample_idx_pairs_time}')

        # train for one epoch
        network_update_time = time.time()
        loss_epoch = train(all_pairs_list, snapshot_info_list, np.array(pred_label), 
                            label_idx_dict, road_graph_node_emb, args.batch_size, device, 
                            [cluster_result_st_feat_centroids, cluster_result_car_feat_centroids, cluster_result_plate_feat_centroids], 
                            [cluster_result_st_feat_density, cluster_result_car_feat_density, cluster_result_plate_feat_density], 
                            args.min_time, args.max_time, args.lamda_dict_loss, args.lamda_proto_loss)
        network_update_time = time.time() - network_update_time
        print(f'network update consume time: {network_update_time}')

        # save model
        save_model(args.model_path, model.state_dict(), optimizer.state_dict(), 'sj_proj_model', epoch)
        torch.cuda.empty_cache() # clean up in time to free up GPU memory
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(all_pairs_list)}")
        loss_list.append(loss_epoch)

        epoch_total_time = extract_feat_time + cluster_time + gen_cluster_multimodal_feat_time + get_sample_idx_pairs_time + network_update_time
        print(f'current epoch consume time: {epoch_total_time}')
    
    train_network_time = time.time() - train_network_time
    print("network training totally consume time:", train_network_time)

    if not os.path.exists(args.loss_pic_path):
        os.makedirs(args.loss_pic_path)
    loss_pic_file = os.path.join(args.loss_pic_path, args.loss_pic_file)
    plot_loss(loss_list, loss_pic_file)
    print('plot loss picture successfully!')
    
    save_model(args.model_path, model.state_dict(), optimizer.state_dict(), 'sj_proj_model', args.epochs)
    print('save model successfully!')

