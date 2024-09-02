import os
import argparse
import random
import pickle
import torch
import numpy as np
from collections import deque
import eval
from modules import network
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

    # the snapshots of the beginning of time are loaded and clustered multiple epoches via self-supervised learning
    if args.dataset == "UrbanVehicle":
        test_dataset = uv_img_dataset(
            record_file=args.record_file,
            cam_file=args.cam_file,
            train=False,
            use_plate=args.use_plate,
            sampl_rate=args.sampl_rate, 
            time_gap_length=args.time_gap_length, 
            time_length=args.time_length, 
            fps=args.fps, 
        )
    elif args.dataset == "Carla":
        test_dataset = carla_img_dataset(
            record_file=args.record_file,
            cam_file=args.cam_file,
            train=False,
            use_plate=args.use_plate,
            sampl_rate=args.sampl_rate, 
            time_gap_length=args.time_gap_length, 
            time_length=args.time_length, 
            fps=args.fps, 
        )
    else:
        raise NotImplementedError

    # generate dataloader by reading snapshots with batch size
    time_interval_sampler = TimeIntervalSampler(test_dataset, args.test_time_interval * args.fps)
    extract_feats_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=time_interval_sampler,
        num_workers=args.workers,
    )

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
    model_fp = os.path.join(args.model_path, "sj_proj_model_checkpoint_{}.tar".format(args.test_epoch))
    checkpoint = torch.load(model_fp, map_location=device.type)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    
    # fast incremental clustering initialization
    cluster = FastIncCluster(args.inc_cluster['feat_dims'], args.inc_cluster['ngpu'], args.inc_cluster['useFloat16'])
    cumul_removed_feat_num = 0 # to record the number of removal features that are too far apart
    f_ids = [[], []]
    id_dict = {}
    css = {}
    cf_means = {}
    cf_means_no_norm = {}
    pred_labels = [-1] * test_dataset.record_num
    curr_label = 0
    prev_record_num = [0, 0, 0]

    # load tracklets related information, only used for carla dataset
    if args.dataset == 'Carla' and getattr(args, 'cluster_base_preprocess', True):
        load_tracklets_info_time = time.time()
        tracklets_index_dict = eval.load_corresp_tracklets_index_dict(test_dataset, args.tracklets_index_dict_file)
        _, gt_label_dict, _ = pickle.load(open(args.label_dict_file, 'rb'))
        load_tracklets_info_time = time.time() - load_tracklets_info_time
        print(f'load tracklets information consumes time: {load_tracklets_info_time}')

    # process trajectory recovery
    reconst_index_batch_thres = int(args.inc_cluster['reconst_index_feat_num_thres'] / args.data_batch_size)
    st_feat_list = []
    st_feats_queue = deque()
    extracted_data_count_list = []
    camera_ids = []
    gt_labels = []
    extract_feat_total_time = 0
    cluster_total_time = 0
    eval_total_time = 0
    for batch_idx, (curr_batch_car_feats, 
                    curr_batch_plate_feats, 
                    curr_batch_time, 
                    curr_batch_lon, 
                    curr_batch_lat, 
                    curr_batch_camera_ids, 
                    curr_batch_gt_label) in enumerate(extract_feats_data_loader):
        # extract spatio-temporal features
        extract_feat_time = time.time()
        curr_batch_st_feats = extract_st_feat_from_tensor(curr_batch_camera_ids, curr_batch_time, curr_batch_lon, curr_batch_lat, 
                                            model, device, cam_vec_idx_dict, road_graph_node_emb) # extract spatio-temporal features
        extract_feat_time = time.time() - extract_feat_time
        extract_feat_total_time += extract_feat_time
        print("extracting features consume time:", extract_feat_time)

        while len(st_feats_queue) > args.inc_cluster['reconst_index_batch_num']:
            st_feats_queue.popleft()
        # determine whether to delete the index
        if sum(extracted_data_count_list) > args.inc_cluster['feat_num_thres']:
            print('reconstruct the index...')
            reconstruct_index_time = time.time()

            # generate dataloader for index reconstruction
            reconst_index_batch_num = len(st_feats_queue)
            print(f'reconstruct index batch number: {reconst_index_batch_num}')
            start_batch_idx = batch_idx - reconst_index_batch_num
            end_batch_idx = batch_idx
            print(f'reconstruct index start batch index: {start_batch_idx}')
            print(f'reconstruct index end batch index: {end_batch_idx}')
            time_interval_range_sampler = TimeIntervalRangeSampler(time_interval_sampler, start_batch_idx, end_batch_idx)
            reconst_index_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_sampler=time_interval_range_sampler,
                num_workers=args.workers,
            )
            print(f'the length of reconst_index_data_loader: {len(reconst_index_data_loader)}')

            # clear the searchers' memory usage
            for feat_dim in set(args.inc_cluster['feat_dims']):
                cluster.searchers[feat_dim].reset()
            torch.cuda.empty_cache() # free up GPU memory
            print('clear the original index successfully!')

            # reconstruct index
            for reconst_batch_car_feats, reconst_batch_plate_feats, _, _, _, _, _ in reconst_index_data_loader:
                reconst_batch_st_feats = st_feats_queue.popleft()
                cluster.add_vector_in_index([reconst_batch_st_feats, reconst_batch_car_feats.float().numpy(), reconst_batch_plate_feats.float().numpy()],
                                            weights=args.inc_cluster['weights'], 
                                            normalization=True)
            
            # update the cumul_removed_feat_num parameter
            cumul_removed_feat_num = sum(extracted_data_count_list[:-reconst_index_batch_num])
            
            reconstruct_index_time = time.time() - reconstruct_index_time
            print('reconstruct index consumes time {}'.format(reconstruct_index_time))
            
        # clustering for trajectory recovery
        cluster_time = time.time()
        f_ids, pred_labels, id_dict, css, cf_means, cf_means_no_norm, curr_label, prev_record_num = cluster.fit(
            cumul_removed_feat_num,
            [curr_batch_st_feats, curr_batch_car_feats.numpy(), curr_batch_plate_feats.numpy()],
            pred_labels,
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
        cluster_time = time.time() - cluster_time
        cluster_total_time += cluster_time
        print("clustering consume time:", cluster_time)

        # estimate current snapshot label accuracy
        eval_time = time.time()
        prev_processed_record_num = sum(extracted_data_count_list)
        print(f'previous processed records number: {prev_processed_record_num}')
        if args.dataset == 'Carla' and getattr(args, 'cluster_base_preprocess', True):
            precision, recall, fscore, expansion, _ = eval.evaluate_snapshots(curr_batch_gt_label.numpy(), 
                                                                              pred_labels[prev_processed_record_num:prev_processed_record_num+len(curr_batch_car_feats)], 
                                                                              curr_batch_camera_ids.numpy(), 
                                                                              gt_label_dict, 
                                                                              tracklets_index_dict, 
                                                                              prev_processed_record_num, 
                                                                              True)
        else:
            if not all(label == -1 for label in curr_batch_gt_label.numpy()):
                precision, recall, fscore, expansion, _ = eval.evaluate_prf(curr_batch_gt_label.numpy(), 
                                                                            pred_labels[prev_processed_record_num:prev_processed_record_num+len(curr_batch_car_feats)])
            else:
                print('current batch GT labels are all -1')
        eval_time = time.time() - eval_time
        eval_total_time += eval_time
        print('precision/recall/fscore/expansion = {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(precision, recall, fscore, expansion))
        print('evaluation consume time {}'.format(eval_time))
        
        # update global labels and cache data
        camera_ids.extend(curr_batch_camera_ids.tolist())
        gt_labels.extend(curr_batch_gt_label.tolist())
        st_feats_queue.append(curr_batch_st_feats)
        st_feat_list.append(curr_batch_st_feats)
        extracted_data_count_list.append(len(curr_batch_car_feats))
        print(f'current batch data totally consumes time: {extract_feat_time + cluster_time}')
    
    print(f'feature extraction total consumes time: {extract_feat_total_time}')
    print(f'clustering total consumes time: {cluster_total_time}')
    print(f'evaluation total consumes time: {eval_total_time}')
    print(f'trajectory recovery total consumes time: {extract_feat_total_time + cluster_total_time}')
    torch.cuda.empty_cache() # clear GPU memory
    # estimate trajectory recovery results
    if args.dataset == 'Carla' and getattr(args, 'cluster_base_preprocess', True):
        precision, recall, fscore, expansion, vid_to_cid = eval.evaluate_snapshots(gt_labels, 
                                                                                    pred_labels, 
                                                                                    camera_ids, 
                                                                                    gt_label_dict, 
                                                                                    tracklets_index_dict, 
                                                                                    0, 
                                                                                    True)
    else:
        precision, recall, fscore, expansion, vid_to_cid = eval.evaluate_prf(gt_labels, pred_labels)
    print('final precision/recall/fscore/expansion = {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(precision, recall, fscore, expansion))

    save_traj_rec_result(args, pred_labels, gt_labels, vid_to_cid)
    if args.save_record:
        save_record_feat(args, np.vstack(st_feat_list), test_dataset.car_feats_arr, test_dataset.plate_feats_arr)
    print('save recovered trajectory results successfully!')
    print('Done!')
