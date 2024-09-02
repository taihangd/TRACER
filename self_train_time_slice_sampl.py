import os
import argparse
import random
import pickle
import torch
import numpy as np
import eval
from modules import network, contrastive_loss
from datasets.urban_vehicle_time_slice_sampl import *
from datasets.carla_tracklets_time_slice_sampl import *
from universal_functions import *
from utils.yaml_config_hook import *
from utils.save_model import *
from cluster import *


if __name__ == "__main__":
    # parse settings
    parser = argparse.ArgumentParser()
    # config file
    parser.add_argument('--cfg_file', 
                        help='experiment configure file name',
                        default='./config/carla_tracklets_self_train_time_slice_sampl.yaml')
    
    # configuration file setting
    initial_args, _ = parser.parse_known_args()
    dataset_config_file = initial_args.cfg_file
    config = yaml_config_hook(dataset_config_file)
    config = flatten_dict(config)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    nested_args = unflatten_dict(vars(args))
    for key, value in nested_args.items():
        setattr(args, key, value)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load road graph node embedding
    road_graph_node_emb, \
        cam_vec_idx_dict, \
            cam_vec_idx_dict_keys_tensor, \
                cam_vec_idx_dict_values_tensor = load_road_graph_node_embedding(args.road_graph_file, 
                                                                                args.road_graph_node_vec_file, 
                                                                                args.cid_rid_correspondence_file, 
                                                                                device)
    
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
    
    # fast incremental clustering initialization
    cluster = FastIncCluster(args.inc_cluster['feat_dims'], args.inc_cluster['ngpu'], args.inc_cluster['useFloat16'])
    
    # the snapshots of the beginning of time are loaded and clustered multiple epoches via self-supervised learning
    if args.dataset == "UrbanVehicle":
        train_dataset = uv_img_dataset(
            record_file=args.record_file,
            cam_file=args.cam_file,
            train=True,
            use_plate=args.use_plate,
            sampl_rate=args.sampl_rate, 
            time_gap_length=args.time_gap_length, 
            time_length=args.time_length, 
            fps=args.fps, 
        )
    elif args.dataset == "Carla":
        train_dataset = carla_img_dataset(
            record_file=args.record_file,
            cam_file=args.cam_file,
            train=True,
            use_plate=args.use_plate,
            sampl_rate=args.sampl_rate, 
            time_gap_length=args.time_gap_length, 
            time_length=args.time_length, 
            fps=args.fps, 
        )
    else:
        raise NotImplementedError

    # generate dataloader by reading snapshots with batch size
    extract_feats_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )

    # load tracklets related information, only used for carla dataset
    if args.dataset == 'Carla' and getattr(args, 'cluster_base_preprocess', True):
        load_tracklets_info_time = time.time()
        tracklets_index_dict = eval.load_corresp_tracklets_index_dict(train_dataset, args.tracklets_index_dict_file)
        _, gt_label_dict, _ = pickle.load(open(args.label_dict_file, 'rb'))
        load_tracklets_info_time = time.time() - load_tracklets_info_time
        print(f'load tracklets information consumes time: {load_tracklets_info_time}')

    # extract original features
    car_feats = train_dataset.car_feats_arr
    plate_feats = train_dataset.plate_feats_arr
    time_arr = train_dataset.time_arr
    lon_arr = train_dataset.lon_arr
    lat_arr = train_dataset.lat_arr
    camera_ids_arr = train_dataset.camera_ids_arr
    gt_labels = train_dataset.gt_labels_arr
   
    # train main loop
    train_network_time = time.time()
    cluster_result_st_feat_centroids, cluster_result_st_feat_concentration = None, None
    cluster_result_car_feat_centroids, cluster_result_car_feat_concentration = None, None
    cluster_result_plate_feat_centroids, cluster_result_plate_feat_concentration = None, None
    loss_list = list()
    for epoch in range(args.start_epoch, args.epochs):
        # extract spatio-temporal features
        extract_feat_time = time.time()
        st_feats = extract_st_feats(extract_feats_data_loader, model, device, cam_vec_idx_dict, road_graph_node_emb) # extract spatio-temporal features
        extract_feat_time = time.time() - extract_feat_time
        print("extracting features consume time:", extract_feat_time)

        # clustering to generate pseudo labels
        cluster_time = time.time()
        cumul_removed_feat_num = 0 # to record the number of removal features that are too far apart
        f_ids = [[], []]
        id_dict = {}
        css = {}
        cf_means = {}
        cf_means_no_norm = {}
        pred_label = [-1] * len(car_feats)
        curr_label = 0
        prev_record_num = [0, 0, 0]
        f_ids, pred_label, id_dict, css, cf_means, cf_means_no_norm, curr_label, prev_record_num = cluster.fit(
            cumul_removed_feat_num,
            [st_feats, car_feats, plate_feats],
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
            css=css,
            cf_means=cf_means,
            cf_means_no_norm=cf_means_no_norm,
            curr_label=curr_label,
            prev_record_num=prev_record_num,
        )
        # clear the searchers' memory usage
        for feat_dim in set(args.inc_cluster['feat_dims']):
            cluster.searchers[feat_dim].reset()
        torch.cuda.empty_cache() # free up GPU memory
        cluster_time = time.time() - cluster_time
        print("clustering consume time:", cluster_time)

        # estimate current snapshot label accuracy
        eval_time = time.time()
        if args.dataset == 'Carla' and getattr(args, 'cluster_base_preprocess', True):
            precision, recall, fscore, expansion, _ = eval.evaluate_snapshots(gt_labels, 
                                                                              pred_label, 
                                                                              camera_ids_arr, 
                                                                              gt_label_dict, 
                                                                              tracklets_index_dict, 
                                                                              0, 
                                                                              True)
        else:
            precision, recall, fscore, expansion, _ = eval.evaluate_prf(gt_labels, pred_label)
        eval_time = time.time() - eval_time
        print('precision/recall/fscore/expansion = {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(precision, recall, fscore, expansion))
        print('evaluation consume time {}'.format(eval_time))

        if epoch >= args.warmup_epoch:
            gen_cluster_multimodal_feat_time = time.time()
            cluster_result_st_feat_centroids, \
            cluster_result_car_feat_centroids, \
            cluster_result_plate_feat_centroids, \
            cluster_result_st_feat_concentration, \
            cluster_result_car_feat_concentration, \
            cluster_result_plate_feat_concentration, \
            label_idx_dict = gen_cluster_multimodal_feat(st_feats, car_feats, plate_feats, pred_label, device, args.temperature, alpha=args.alpha)
            gen_cluster_multimodal_feat_time = time.time() - gen_cluster_multimodal_feat_time
            print(f'generate cluster multi-modal feature consume time: {gen_cluster_multimodal_feat_time}')

        # generate a list of sample pairs for one epoch
        get_sample_idx_pairs_time = time.time()
        # generate the parameters to control sampling
        pred_traj_num = len(set(pred_label))
        num_sampl_group = int(pred_traj_num * args.sampl_group_num_coef / args.batch_size)
        num_group_pairs = max(int(len(car_feats) / pred_traj_num), 2) # set the pairs number in the group
        print(f'epoch {epoch} the number of sample group: {num_sampl_group}')
        print(f'epoch {epoch} the number of group pairs: {num_group_pairs}')
        all_pairs_list = get_sample_idx_pairs_sampl_win(pred_label, 
                                                        time_arr, 
                                                        train_dataset.sampl_time_list, 
                                                        num_sampl_group, 
                                                        num_group_pairs, 
                                                        args.batch_size)
        get_sample_idx_pairs_time = time.time() - get_sample_idx_pairs_time
        print(f'get sample idx pairs consume time: {get_sample_idx_pairs_time}')

        # train for one epoch
        network_update_time = time.time()
        loss_epoch = train(model, optimizer, 
                           all_pairs_list, 
                           car_feats, plate_feats, time_arr,
                           lon_arr, lat_arr, camera_ids_arr,
                           np.array(pred_label), label_idx_dict, road_graph_node_emb, 
                           args.batch_size, device, 
                           cam_vec_idx_dict_keys_tensor, 
                            cam_vec_idx_dict_values_tensor,
                            [cluster_result_st_feat_centroids, cluster_result_car_feat_centroids, cluster_result_plate_feat_centroids], 
                            [cluster_result_st_feat_concentration, cluster_result_car_feat_concentration, cluster_result_plate_feat_concentration], 
                            train_dataset.record_start_time, train_dataset.record_start_time+args.time_length * args.fps, 
                            criterion_multi_modal_proto_noise_contra_estimation, 
                            args.lamda_dict_loss, args.lamda_proto_loss)
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
    print('save final model successfully!')

