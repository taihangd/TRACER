import math
import gc
import time
import numpy as np
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
from numpy.linalg import norm
import faiss


def normalize(feature):
    return feature / (np.linalg.norm(feature) + 1e-12)
    
class FaissFlatSearcher:
    def __init__(self, ngpu=1, feat_dim=256, useFloat16=False):
        self.useFloat16 = useFloat16
        if ngpu:
            flat_config = []
            for i in range(ngpu):
                cfg = faiss.GpuIndexFlatConfig()
                cfg.useFloat16 = useFloat16
                cfg.device = i
                flat_config.append(cfg)
            res = [faiss.StandardGpuResources() for _ in range(ngpu)]
            indexes = [
                faiss.GpuIndexFlatIP(res[i], feat_dim, flat_config[i])
                for i in range(ngpu)
            ]
            self.index = faiss.IndexProxy()
            for sub_index in indexes:
                self.index.addIndex(sub_index)
        else:
            self.index = faiss.IndexFlatIP(feat_dim)
    
    def search_by_topk_by_blocks(self, query, gallery, topk, query_num):
        if query is None or len(query) == 0:
            return np.array([]), np.array([])
        
        self.index.add(gallery)

        topk_scores_list = []
        topk_idxs_list = []
        if query_num is None:
            query_num = len(query)
        block_num = int(np.ceil(len(query) / query_num))
        for i in range(block_num):
            topk_scores, topk_idxs = self.index.search(query[i * query_num: (i + 1) * query_num], topk)
            topk_scores_list.append(topk_scores.astype(np.float16) if self.useFloat16 else topk_scores) # be consistent with the input data type
            topk_idxs_list.append(topk_idxs.astype(np.int32)) # the default is int64, usually int32 is enough

        topk_scores = np.concatenate(topk_scores_list, axis=0)
        topk_idxs = np.concatenate(topk_idxs_list, axis=0)

        return topk_scores, topk_idxs

    def add(self, gallery):
        self.index.add(gallery)

    def remove(self, remove_feat_id): # remove ids is not supported by GPU index
        self.index.remove_ids(remove_feat_id)

    def reset(self):
        self.index.reset()

    def get_index_num(self):
        return self.index.ntotal

class FastIncCluster:
    def __init__(self, feature_dims=[256, 256], ngpu=1, useFloat16=False):
        self.searchers = {i: FaissFlatSearcher(ngpu, i, useFloat16) for i in set(feature_dims)}
        self.f_dims = feature_dims

    def fit(
        self,
        cumul_removed_feat_num,
        feats,
        cids,
        weights=[1],
        sim_thres=0.88,
        adj_pt_ratio=0.5,
        spher_distrib_coeff=1,
        topK=128,
        query_num=8192,
        normalization=True,
        f_ids=[[], []],
        id_dict={},
        css={},
        cf_means={},
        cf_means_no_norm={},
        curr_label=0,
        prev_record_num=[0, 0, 0],
    ):
        snapshots_retrieval_time = time.time()
        if isinstance(weights, float) or isinstance(weights, int):
            weights = [weights] * len(self.f_dims)
        if isinstance(topK, int):
            topK = [topK] * len(self.f_dims)
        else:
            assert len(topK) == len(self.f_dims)

        st_feat, car_feat, plate_feat = feats
        record_num = len(st_feat)
        if normalization: # have replaced the None type with the all-zero array
            faiss.normalize_L2(st_feat)
            faiss.normalize_L2(car_feat)
            faiss.normalize_L2(plate_feat)
            print("features normalization")
        
        # generate joint representation features
        weight_cumsum = np.cumsum(weights)
        norm_weight_st, norm_weight_vis = math.sqrt(weight_cumsum[0] / (weight_cumsum[0] + weights[1])), math.sqrt(weights[1] / (weight_cumsum[0] + weights[1]))
        norm_weight_st_vis, norm_weight_plate = math.sqrt(weight_cumsum[1] / (weight_cumsum[1] + weights[2])), math.sqrt(weights[2] / (weight_cumsum[1] + weights[2]))
        st_vis_feat = np.concatenate((st_feat * norm_weight_st, car_feat * norm_weight_vis), axis=1)
        
        st_vis_plate_feat_orig_idx_list = [i for i, feat in enumerate(plate_feat) if feat is not None and not np.all(feat == 0)]
        st_vis_plate_feat = np.concatenate((st_vis_feat[st_vis_plate_feat_orig_idx_list, :] * norm_weight_st_vis, plate_feat[st_vis_plate_feat_orig_idx_list, :] * norm_weight_plate), axis=-1)
        st_vis_feat_orig_idx_list = list(range(len(plate_feat)))

        cancat_feats = [st_vis_plate_feat, st_vis_feat]
        st_vis_plate_feat_idx_list = [i + prev_record_num[2] for i in st_vis_plate_feat_orig_idx_list]
        st_vis_feat_idx_list = [i + prev_record_num[2] for i in st_vis_feat_orig_idx_list]
        cancat_feats_idx_list = [st_vis_plate_feat_idx_list, st_vis_feat_idx_list]

        print("search topk")

        topk_scores_list = []
        topk_idxs_list = []
        for f, dim, topk in zip(cancat_feats, self.f_dims, topK): # for each modal
            if len(f) == 0:
                topk_scores_list.append([])
                topk_idxs_list.append([])
                continue
            topk_scores, topk_idxs = self.searchers[dim].search_by_topk_by_blocks(f, f, topk, query_num)
            topk_scores_list.append(topk_scores)
            topk_idxs_list.append(topk_idxs + cumul_removed_feat_num)

        # map back to original snapshot indexes
        for i, idx_list in enumerate(cancat_feats_idx_list):
            f_ids[i] += idx_list
        f_topks = [
            np.array(f_id)[topk_idxs]
            for f_id, topk_idxs in zip(f_ids, topk_idxs_list)
        ]
        # release topk_idxs_list
        del topk_idxs_list
        gc.collect()

        # fusion similarity
        fusion_sim_time = time.time()

        if len(st_vis_plate_feat_orig_idx_list) == 0:
            topks, topks_score = f_topks[1], topk_scores_list[1]
        else:
            topks = [[] for _ in range(record_num)]
            topks_score = [[] for _ in range(record_num)]

            # check if lengths are different and initialize padded arrays if necessary
            if len(st_vis_feat_orig_idx_list) != len(st_vis_plate_feat_orig_idx_list):
                padded_f_topks = np.full((len(st_vis_feat_orig_idx_list), f_topks[0].shape[1]), fill_value='-1', dtype=f_topks[0].dtype)
                padded_scores = np.full((len(st_vis_feat_orig_idx_list), topk_scores_list[0].shape[1]), fill_value=-1, dtype=topk_scores_list[0].dtype)
                padded_f_topks[st_vis_plate_feat_orig_idx_list] = f_topks[0]
                padded_scores[st_vis_plate_feat_orig_idx_list] = topk_scores_list[0]
                f_topks[0] = padded_f_topks
                topk_scores_list[0] = padded_scores
            # concatenate all f_topks and topk_scores_list
            all_topks = np.concatenate(f_topks, axis=1)
            all_topks_scores = np.concatenate(topk_scores_list, axis=1)
            # release f_topks and topk_scores_list
            del f_topks
            del topk_scores_list
            gc.collect()

            # remove -1 values and get unique elements
            if len(st_vis_feat_orig_idx_list) != len(st_vis_plate_feat_orig_idx_list):
                for i in range(record_num):
                    valid_indices = all_topks[i] != -1
                    unique_topks, unique_indices = np.unique(all_topks[i][valid_indices], return_index=True)
                    topks[i] = unique_topks
                    topks_score[i] = all_topks_scores[i][valid_indices][unique_indices]
            else:
                for i in range(record_num):
                    unique_topks, unique_indices = np.unique(all_topks[i], return_index=True)
                    topks[i] = unique_topks
                    topks_score[i] = all_topks_scores[i][unique_indices]

        snapshots_retrieval_time = time.time() - snapshots_retrieval_time # record the snapshots retrieval time
        print('fusion similarity consuming time: {}'.format(time.time() - fusion_sim_time))
        
        # clusters generation
        print("clusters generation")
        gen_cluster_time = time.time()
        sim_calc_time = 0
        organize_topk_time = 0
        closeness_calc_time = 0
        cluster_sim_calc_time = 0
        statistics_update_time = 0
        sim_select_num = 0
        sim_computation_num = 0
        data = [[a, b, c] if c is not None and not np.all(c == 0) else [a, b, None] for a, b, c in zip(st_feat, car_feat, plate_feat)]
        for record_idx1, (topk_idxs, topk_score) in tqdm(enumerate(zip(topks, topks_score))):
            start_sim_calc_time = time.time()
            
            record_idx1_total_seq = record_idx1 + prev_record_num[2]
            connect_flag = topk_score > sim_thres
            connect_flag[topk_idxs >= record_idx1_total_seq] = False
            
            connect_record_num = np.sum(connect_flag == True)
            if connect_record_num == 0:
                cids[record_idx1_total_seq] = curr_label
                id_dict[curr_label] = [record_idx1_total_seq]
                css[curr_label] = [0 if j is None else 1 for j in data[record_idx1]]
                cf_means[curr_label] = data[record_idx1]
                cf_means_no_norm[curr_label] = data[record_idx1]
                curr_label += 1
                continue
            
            start_organize_topk_time = time.time()
            connect_record_idx_array = np.argwhere(connect_flag)
            cid_sim_dict = defaultdict(list)
            for connect_record_idx in connect_record_idx_array[:, 0]:
                record_idx2 = topk_idxs[connect_record_idx]
                curr_sim = topk_score[connect_record_idx]
                topk_cid = cids[record_idx2]
                cid_sim_dict[topk_cid].append(curr_sim)
            organize_topk_time += time.time() - start_organize_topk_time

            start_closeness_calc_time = time.time()
            select_cid_list = []
            quick_calc_select_cid_list = []
            for cid, sim_list in cid_sim_dict.items():
                closeness = float(len(sim_list)) / len(id_dict[cid])
                if closeness > spher_distrib_coeff:
                    quick_calc_select_cid_list.append(cid)
                elif closeness > adj_pt_ratio:
                    select_cid_list.append(cid)
            closeness_calc_time += time.time() - start_closeness_calc_time
            # print('the number of the selected cid list is: {}'.format(len(select_cid_list)))
            # print('the number of the quick calculation selected cid list is: {}'.format(len(quick_calc_select_cid_list)))
            # print(f'current snapshot similarity computation times: {len(select_cid_list) - len(quick_calc_select_cid_list)}')

            start_cluster_sim_calc_time = time.time()
            max_sim = -1
            final_cid = -1
            for cid in quick_calc_select_cid_list:
                curr_sim = max(cid_sim_dict[cid])
                # curr_sim = sum(cid_sim_dict[cid]) / len(cid_sim_dict[cid])
                if curr_sim > max_sim:
                    max_sim = curr_sim
                    final_cid = cid
            for cid in select_cid_list:
                w_total = 0
                curr_sim = 0
                for w, a, b in zip(weights, data[record_idx1], cf_means[cid]):
                    if a is not None and b is not None:
                        curr_sim += a @ b * w
                        w_total += w
                curr_sim /= w_total
                if curr_sim > max_sim:
                    max_sim = curr_sim
                    final_cid = cid
            cluster_sim_calc_time += time.time() - start_cluster_sim_calc_time

            sim_calc_time += time.time() - start_sim_calc_time
            sim_select_num += len(quick_calc_select_cid_list)
            sim_computation_num += len(select_cid_list)

            start_statistics_update_time = time.time()
            if max_sim > sim_thres:
                cids[record_idx1_total_seq] = final_cid
                id_dict[final_cid].append(record_idx1_total_seq)
                cs = css[final_cid]
                cf_mean = cf_means[final_cid]
                cf_mean_no_norm = cf_means_no_norm[final_cid]
                for j, k in enumerate(data[record_idx1]):
                    if k is not None:
                        if cs[j]:
                            inc_mean_weight = cs[j] / (cs[j] + 1)
                            cs[j] += 1
                            cf_mean_no_norm[j] = cf_mean_no_norm[j] * inc_mean_weight + k * (1 - inc_mean_weight)
                            cf_mean[j] = normalize(cf_mean_no_norm[j])
                        else:
                            cs[j] = 1
                            cf_mean[j] = k
                            cf_mean_no_norm[j] = k
            else:
                cids[record_idx1_total_seq] = curr_label
                id_dict[curr_label] = [record_idx1_total_seq]
                css[curr_label] = [0 if j is None else 1 for j in data[record_idx1]]
                cf_means[curr_label] = data[record_idx1]
                cf_means_no_norm[curr_label] = data[record_idx1]
                curr_label += 1
            statistics_update_time = time.time() - start_statistics_update_time + statistics_update_time
        
        gen_cluster_time = time.time() - gen_cluster_time

        # output the running time of each module
        print(f'snapshots retrieval time: {snapshots_retrieval_time}')
        print(f'cluster generation total cosuming time: {gen_cluster_time}')
        print(f'similarity computation total cosuming time: {sim_calc_time}')
        print(f'topk organization total cosuming time: {organize_topk_time}')
        print(f'closeness computation total cosuming time: {closeness_calc_time}')
        print(f'cluster similarity computation total cosuming time: {cluster_sim_calc_time}')
        print(f'cluster statistics update total cosuming time: {statistics_update_time}')
        print(f'total similarity computation times: {sim_computation_num}')
        print(f'total selected times: {sim_select_num}')
        print(f'the number of snapshots is: {len(data)}')
        print("clustering finished!")

        # update the number of previous snapshots
        prev_record_num = [prev_record_num[0] + len(st_vis_plate_feat_idx_list), 
                           prev_record_num[1] + len(st_vis_feat_idx_list), 
                           prev_record_num[2] + record_num]

        return f_ids, cids, id_dict, css, cf_means, cf_means_no_norm, curr_label, prev_record_num
    
    def add_vector_in_index(
        self,
        feats,
        weights=[1],
        normalization=True,
    ):
        add_vector_in_index_time = time.time()
        if isinstance(weights, float) or isinstance(weights, int):
            weights = [weights] * len(self.f_dims)

        st_feat, car_feat, plate_feat = feats
        if normalization: # have replaced the None type with the all-zero array
            faiss.normalize_L2(st_feat)
            faiss.normalize_L2(car_feat)
            faiss.normalize_L2(plate_feat)
            print("features normalization")
        
        # generate joint representation features
        weight_cumsum = np.cumsum(weights)
        norm_weight_st, norm_weight_vis = math.sqrt(weight_cumsum[0] / (weight_cumsum[0] + weights[1])), math.sqrt(weights[1] / (weight_cumsum[0] + weights[1]))
        norm_weight_st_vis, norm_weight_plate = math.sqrt(weight_cumsum[1] / (weight_cumsum[1] + weights[2])), math.sqrt(weights[2] / (weight_cumsum[1] + weights[2]))
        st_vis_feat = np.concatenate((st_feat * norm_weight_st, car_feat * norm_weight_vis), axis=1)
        
        st_vis_plate_feat_idx_list = [i for i, feat in enumerate(plate_feat) if feat is not None and not np.all(feat == 0)]
        st_vis_plate_feat = np.concatenate((st_vis_feat[st_vis_plate_feat_idx_list, :] * norm_weight_st_vis, plate_feat[st_vis_plate_feat_idx_list, :] * norm_weight_plate), axis=-1)

        cancat_feats = [st_vis_plate_feat, st_vis_feat]

        for f, dim in zip(cancat_feats, self.f_dims): # for each modal
            if len(f) == 0: # if there is no plate features
                continue
            self.searchers[dim].add(f)
        
        add_vector_in_index_time = time.time() - add_vector_in_index_time
        print('add vectors in the index consumes time: {}'.format(add_vector_in_index_time))

        return

def gen_tracklets(feats, plate_feats, camera_ids, timestamps, label_cnt, snapshot_num_thres, time_thres, sim_thres):
    feat_num = len(feats)
    feat_label = [-1] * feat_num
    cluster_center = {}
    cluster_center_no_norm = {}
    cluster_pt_num = {}
    for i, (curr_feat, plate_feat) in enumerate(zip(feats, plate_feats)):
        prev_idx = max(i-snapshot_num_thres, -1) # decide how many snapshots to look back
        for j in range(i-1, prev_idx, -1): # check feature similarity
            # check the same camera constraints
            if camera_ids[i] != camera_ids[j]:
                continue
            
            # check time constraints
            if abs(timestamps[i] - timestamps[j]) > time_thres: 
                continue
            
            if not np.all(plate_feat == 0) and not np.all(plate_feats[j] == 0) and np.dot(plate_feat, plate_feats[j]) / (norm(plate_feat) * norm(plate_feats[j])) > sim_thres:
                curr_label = feat_label[j]
                feat_label[i] = curr_label
                inc_mean_weight = cluster_pt_num[curr_label] / (cluster_pt_num[curr_label] + 1)
                cluster_pt_num[curr_label] += 1
                cluster_center_no_norm[curr_label] = cluster_center_no_norm[curr_label] * inc_mean_weight + curr_feat * (1 - inc_mean_weight)
                cluster_center[curr_label] = normalize(cluster_center_no_norm[curr_label])
                break

            prev_feat_center = cluster_center[feat_label[j]]
            if np.dot(curr_feat, prev_feat_center) / (norm(curr_feat) * norm(prev_feat_center)) > sim_thres: # if similar, group them into previous category
                curr_label = feat_label[j]
                feat_label[i] = curr_label
                inc_mean_weight = cluster_pt_num[curr_label] / (cluster_pt_num[curr_label] + 1)
                cluster_pt_num[curr_label] += 1
                cluster_center_no_norm[curr_label] = cluster_center_no_norm[curr_label] * inc_mean_weight + curr_feat * (1 - inc_mean_weight)
                cluster_center[curr_label] = normalize(cluster_center_no_norm[curr_label])
                break
            
        if feat_label[i] == -1:
            curr_label = label_cnt
            feat_label[i] = curr_label
            cluster_pt_num[curr_label] = 1
            cluster_center_no_norm[curr_label] = curr_feat
            cluster_center[curr_label] = curr_feat
            label_cnt += 1

    return feat_label, label_cnt, cluster_center


if __name__ == "__main__":
    # generate random feature data, camera ID and timestamp
    np.random.seed(42)
    features = np.random.rand(100, 128) # 100 features, each feature is a 128-dimensional vector
    camera_ids = np.random.choice(['cam1', 'cam2', 'cam3'], 100) # 100 camera IDs
    timestamps = np.random.randint(0, 1000, 100) # 100 timestamps

    # record start time
    start_time = time.time()
    # call the one-time traversal clustering algorithm
    clusters = gen_tracklets(features, camera_ids, timestamps, time_thres=5, sim_thres=0.8)
    # record end time
    clustering_time = time.time() - start_time
    print(f'clustering consume time: {clustering_time}')

    # output clustering results
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx+1}:")
        print(f"  Camera ID: {cluster['camera_id']}")
        print(f"  Latest Time: {cluster['latest_time']}")
        print(f"  Number of Features: {len(cluster['features'])}")
