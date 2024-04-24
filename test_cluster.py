import os
import time
import argparse
import pickle
import numpy as np
import torch
from gensim.models import KeyedVectors
from utils import yaml_config_hook
from modules import network, contrastive_loss
import eval
from datasets.urban_vehicle import *
from datasets.carla import *
from cluster import *
from comparison_cluster import *
from utils.save_model import *


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


if __name__ == "__main__":
    # configuration file setting
    dataset_config_file = "./config/uv_comparison_cluster.yaml"
    # # dataset_config_file = "./config/uv-75_comparison_cluster.yaml"
    # # dataset_config_file = "./config/uv-z_comparison_cluster.yaml"
    # # dataset_config_file = "./config/carla_comparison_cluster.yaml"

    # dataset_config_file = "./config/uv_train.yaml"
    # dataset_config_file = "./config/uv-75_train.yaml"
    # dataset_config_file = "./config/uv-z_train.yaml"
    # dataset_config_file = "./config/carla_train.yaml"

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
    model_fp = os.path.join(args.model_path, "sj_proj_model_checkpoint_{}.tar".format(args.test_epoch))
    checkpoint = torch.load(model_fp, map_location=device.type)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    
    # loss definition
    criterion_multi_modal_proto_noise_contra_estimation = contrastive_loss.MultiModalPrototypicalLoss(
        args.batch_size, args.temperature, args.sig_cluster['weights'], device).to(device)

    extract_feat_time = time.time()
    st_feat, car_feat, plate_feat, gt_labels = extract_feat(data_loader, model, device, cam_vec_idx_dict, road_graph_node_emb) # extract features
    extract_feat_time = time.time() - extract_feat_time
    print("extracting features consume time:", extract_feat_time)
    
    # clear the GPU memory usage
    road_graph_node_emb.to('cpu')
    model.to('cpu')
    torch.cuda.empty_cache()

    print("### clustering... ###")
    # get dataset name
    if 'uv_' in dataset_config_file:
        dataset_name = 'uv'
    elif 'uv-75' in dataset_config_file:
        dataset_name = 'uv-75'
    elif 'uv-z' in dataset_config_file:
        dataset_name = 'uv-z'
    elif 'carla' in dataset_config_file:
        dataset_name = 'carla'
    else:
        raise NotImplementedError

    # list all comparison clustering algorithms
    def kmeans_clustering(): # kmeans clustering
        pred_label = kmeans_missing_data(st_feat, car_feat, plate_feat, args)
        return pred_label, len(set(pred_label))

    def aggl_clustering(): # agglomerative clustering
        pred_label = aggl_cluster_missing_data(st_feat, car_feat, plate_feat, args)
        return pred_label, len(set(pred_label))

    def HDBSCAN_clustering(): # HDBSCAN clustering
        if dataset_name == 'uv-z' or dataset_name == 'carla':
            snapshot_feats = np.concatenate((st_feat, car_feat), axis=-1) # for uv-z and carla
        elif dataset_name == 'uv' or dataset_name == 'uv-75':
            snapshot_feats = np.concatenate((st_feat, car_feat, plate_feat), axis=-1) # for uv and uv-75
        else:
            return None, None
        pred_label, _, n_clusters = HDBSCAN_cluster(snapshot_feats, args)
        return pred_label, n_clusters
    
    def PDBSCAN_clustering(): # PDBSCAN clustering
        if dataset_name == 'uv-z' or dataset_name == 'carla':
            snapshot_feats = np.concatenate((st_feat, car_feat), axis=-1) # for uv-z and carla
        elif dataset_name == 'uv' or dataset_name == 'uv-75':
            snapshot_feats = np.concatenate((st_feat, car_feat, plate_feat), axis=-1) # for uv and uv-75
        else:
            return None, None
        _, pred_label, n_clusters = parallel_dbscan(snapshot_feats, args)
        return pred_label, n_clusters
    
    def MMVC_clustering(): # MMVC+ original SigCluster
        cluster = SigCluster(feature_dims=args.sig_cluster['feat_dims'], ngpu=args.sig_cluster['ngpu'])
        pred_label = cluster.fit(
            [[a, b, c] for a, b, c in zip(st_feat, car_feat, plate_feat)],
            initial_labels=None,
            weights=args.sig_cluster['weights'], 
            similarity_threshold=args.sig_cluster['sim_thres'],
            topK=args.sig_cluster['topK'],
            query_num=args.sig_cluster['query_num'],
            normalized=True,
        )
        return pred_label, len(set(pred_label))
    
    def inc_clustering(): # Strick clustering
        feat_num = len(st_feat)
        batch_feat_num = feat_num # processes all data in a single batch for default setting
        batch_feat_list = [
            [st_feat[i:i+batch_feat_num], car_feat[i:i+batch_feat_num], plate_feat[i:i+batch_feat_num]] 
                for i in range(0, feat_num, batch_feat_num)
            ]
        cluster = FastIncCluster(args.inc_cluster['feat_dims'], args.inc_cluster['ngpu'], args.inc_cluster['useFloat16']) # initialization
        cumul_removed_feat_num = 0
        f_ids = [[], []]
        id_dict = {}
        css = {}
        cf_means = {}
        cf_means_no_norm = {}
        pred_label = [-1] * feat_num
        curr_label = 0
        pres_record_num = [0, 0, 0]
        for curr_batch_feat in batch_feat_list:
            f_ids, pred_label, id_dict, css, cf_means, cf_means_no_norm, curr_label, pres_record_num = cluster.fit(
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
                css=css,
                cf_means=cf_means,
                cf_means_no_norm=cf_means_no_norm,
                curr_label=curr_label,
                pres_record_num=pres_record_num,
            )
        for feat_dim in set(args.inc_cluster['feat_dims']):
            cluster.searchers[feat_dim].reset()
        return pred_label, len(set(pred_label))

    def JRRQ_clustering(): # Joint Representation Range Query clustering, currently only supports single batch data processing
        feat_num = len(st_feat)
        batch_feat_num = feat_num # processes all data in a single batch for default setting
        batch_feat_list = [
            [st_feat[i:i+batch_feat_num], car_feat[i:i+batch_feat_num], plate_feat[i:i+batch_feat_num]] 
                for i in range(0, feat_num, batch_feat_num)
            ]
        cluster = JRRQ_Cluster(args.JRRQ['feat_dims'], args.JRRQ['ngpu'], args.JRRQ['useFloat16']) # initialization
        cumul_removed_feat_num = 0
        f_ids = [[], []]
        pred_label = [-1] * feat_num
        pres_record_num = [0, 0, 0]
        for curr_batch_feat in batch_feat_list:
            pred_label = cluster.fit(
                cumul_removed_feat_num,
                curr_batch_feat,
                pred_label,
                weights=args.JRRQ['weights'], 
                similarity_threshold=args.JRRQ['sim_thres'],
                topK=args.JRRQ['topK'],
                query_num=args.JRRQ['query_num'],
                normalization=True,
                f_ids=f_ids,
                pres_record_num=pres_record_num,
            )
        for feat_dim in set(args.JRRQ['feat_dims']):
            cluster.fast_inc_cluster_ret.searchers[feat_dim].reset()
        return pred_label, len(set(pred_label))

    def CDSC_clustering(): # Configurable Distribution Similarity Calculation clustering, currently only supports single batch data processing
        feat_num = len(st_feat)
        batch_feat_num = feat_num # processes all data in a single batch for default setting
        batch_feat_list = [
            [st_feat[i:i+batch_feat_num], car_feat[i:i+batch_feat_num], plate_feat[i:i+batch_feat_num]] 
                for i in range(0, feat_num, batch_feat_num)
            ]
        cluster = CDSC_Cluster(args.CDSC['feat_dims'], args.CDSC['ngpu'], args.CDSC['useFloat16']) # initialization
        f_ids = [[], []]
        id_dict = {}
        cfs = {}
        css = {}
        cf_means = {}
        cf_means_no_norm = {}
        pred_label = [-1] * feat_num
        curr_label = 0
        pres_record_num = [0, 0, 0]
        for curr_batch_feat in batch_feat_list:
            pred_label = cluster.fit(
                curr_batch_feat,
                pred_label,
                weights=args.CDSC['weights'], 
                sim_thres=args.CDSC['sim_thres'],
                adj_pt_ratio=args.CDSC['adj_pt_ratio'],
                spher_distrib_coeff=args.CDSC['spher_distrib_coeff'],
                topK=args.CDSC['topK'],
                query_num=args.CDSC['query_num'],
                normalization=True,
                f_ids=f_ids,
                id_dict=id_dict,
                cfs=cfs,
                css=css,
                cf_means=cf_means,
                cf_means_no_norm=cf_means_no_norm,
                curr_label=curr_label,
                pres_record_num=pres_record_num,
            )
        for feat_dim in set(args.CDSC['feat_dims']):
            cluster.sig_cluster_ret.searchers[feat_dim].reset()
        return pred_label, len(set(pred_label))

    def CCIU_clustering(): # Cluster Center Incremental Update clustering, currently only supports single batch data processing
        cluster = CCIU_Cluster(feature_dims=args.CCIU['feat_dims'], ngpu=args.CCIU['ngpu'], useFloat16=args.CCIU['useFloat16'])
        pred_label = cluster.fit(
            [[a, b, c] for a, b, c in zip(st_feat, car_feat, plate_feat)],
            initial_labels=None,
            weights=args.CCIU['weights'], 
            similarity_threshold=args.CCIU['sim_thres'],
            topK=args.CCIU['topK'],
            query_num=args.CCIU['query_num'],
            normalization=True,
        )
        return pred_label, len(set(pred_label))
    
    def JRRQ_CDSC_clustering(): # JRRQ-CDSC clustering, currently only supports single batch data processing
        feat_num = len(st_feat)
        batch_feat_num = feat_num # processes all data in a single batch for default setting
        batch_feat_list = [
            [st_feat[i:i+batch_feat_num], car_feat[i:i+batch_feat_num], plate_feat[i:i+batch_feat_num]] 
                for i in range(0, feat_num, batch_feat_num)
            ]
        cluster = JRRQ_CDSC_Cluster(args.JRRQ_CDSC['feat_dims'], args.JRRQ_CDSC['ngpu'], args.JRRQ_CDSC['useFloat16']) # initialization
        cumul_removed_feat_num = 0
        f_ids = [[], []]
        id_dict = {}
        cfs = {}
        css = {}
        cf_means = {}
        cf_means_no_norm = {}
        pred_label = [-1] * feat_num
        curr_label = 0
        pres_record_num = [0, 0, 0]
        for curr_batch_feat in batch_feat_list:
            pred_label = cluster.fit(
                cumul_removed_feat_num,
                curr_batch_feat,
                pred_label,
                weights=args.JRRQ_CDSC['weights'], 
                sim_thres=args.JRRQ_CDSC['sim_thres'],
                adj_pt_ratio=args.JRRQ_CDSC['adj_pt_ratio'],
                spher_distrib_coeff=args.JRRQ_CDSC['spher_distrib_coeff'],
                topK=args.JRRQ_CDSC['topK'],
                query_num=args.JRRQ_CDSC['query_num'],
                normalization=True,
                f_ids=f_ids,
                id_dict=id_dict,
                cfs=cfs,
                css=css,
                cf_means=cf_means,
                cf_means_no_norm=cf_means_no_norm,
                curr_label=curr_label,
                pres_record_num=pres_record_num,
            )
        for feat_dim in set(args.JRRQ_CDSC['feat_dims']):
            cluster.fast_inc_cluster_ret.searchers[feat_dim].reset()
        return pred_label, len(set(pred_label))

    def CDSC_CCIU_clustering(): # CDSC-CCIU clustering, currently only supports single batch data processing
        feat_num = len(st_feat)
        batch_feat_num = feat_num # processes all data in a single batch for default setting
        batch_feat_list = [
            [st_feat[i:i+batch_feat_num], car_feat[i:i+batch_feat_num], plate_feat[i:i+batch_feat_num]] 
                for i in range(0, feat_num, batch_feat_num)
            ]
        cluster = CDSC_CCIU_Cluster(args.CDSC_CCIU['feat_dims'], args.CDSC_CCIU['ngpu'], args.CDSC_CCIU['useFloat16']) # initialization
        f_ids = [[], []]
        id_dict = {}
        cfs = {}
        css = {}
        cf_means = {}
        cf_means_no_norm = {}
        pred_label = [-1] * feat_num
        curr_label = 0
        pres_record_num = [0, 0, 0]
        for curr_batch_feat in batch_feat_list:
            pred_label = cluster.fit(
                curr_batch_feat,
                pred_label,
                weights=args.CDSC_CCIU['weights'], 
                sim_thres=args.CDSC_CCIU['sim_thres'],
                adj_pt_ratio=args.CDSC_CCIU['adj_pt_ratio'],
                spher_distrib_coeff=args.CDSC_CCIU['spher_distrib_coeff'],
                topK=args.CDSC_CCIU['topK'],
                query_num=args.CDSC_CCIU['query_num'],
                normalization=True,
                f_ids=f_ids,
                id_dict=id_dict,
                cfs=cfs,
                css=css,
                cf_means=cf_means,
                cf_means_no_norm=cf_means_no_norm,
                curr_label=curr_label,
                pres_record_num=pres_record_num,
            )
        for feat_dim in set(args.CDSC_CCIU['feat_dims']):
            cluster.sig_cluster_ret.searchers[feat_dim].reset()
        return pred_label, len(set(pred_label))


    cluster_dict = {'K-Means': kmeans_clustering, 
                    'agglomerative': aggl_clustering,
                    'HDBSCAN': HDBSCAN_clustering,
                    'PDBSCAN': PDBSCAN_clustering,
                    'MMVC': MMVC_clustering,
                    'Strick': inc_clustering,
                    'JRRQ': JRRQ_clustering,
                    'CDSC': CDSC_clustering,
                    'CCIU': CCIU_clustering,
                    'JRRQ_CDSC': JRRQ_CDSC_clustering,
                    'CDSC_CCIU': CDSC_CCIU_clustering,
                    }
    selected_cluster = cluster_dict.get(args.select_cluster, None)
    if selected_cluster:
        cluster_time = time.time()
        pred_label, n_clusters = selected_cluster() # execute the selected clustering algorithm
        cluster_time = time.time() - cluster_time
        print("clustering consume time:", cluster_time)

    # evaluate inference results
    precision, recall, fscore, expansion, vid_to_cid = eval.evaluate_prf(gt_labels, pred_label)
    print('precision/recall/fscore/expansion = {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(precision, recall, fscore, expansion))
    
    save_traj_rec_result(args, pred_label, gt_labels, vid_to_cid)
    if args.save_record and args.select_cluster == 'Strick':
        save_record_feat(args, st_feat, car_feat, plate_feat)
    print('save recovered trajectory results successfully!')

