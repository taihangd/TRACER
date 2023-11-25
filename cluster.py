import numpy as np
import math
import time
from collections import Counter
from tqdm import tqdm
import faiss


def normalize(feature):
    return feature / (np.linalg.norm(feature) + 1e-12)
    
class FaissFlatSearcher:
    def __init__(self, ngpu=1, feat_dim=256, useFloat16=False):
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
            topk_scores_list.append(topk_scores)
            topk_idxs_list.append(topk_idxs)

        topk_scores = np.concatenate(topk_scores_list, axis=0)
        topk_idxs = np.concatenate(topk_idxs_list, axis=0)

        return topk_scores, topk_idxs

    def remove(self, remove_feat_id):
        self.index.remove_ids(remove_feat_id)

    def reset(self):
        self.index.reset()

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
        cfs={},
        curr_label=0,
        pres_record_num=[0, 0, 0],
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
        
        st_vis_feat_idx_list = list(range(len(plate_feat)))
        st_vis_plate_feat_idx_list = [i for i, feat in enumerate(plate_feat) if feat is not None and not np.all(feat == 0)]
        st_vis_plate_feat = np.concatenate((st_vis_feat * norm_weight_st_vis, plate_feat * norm_weight_plate), axis=-1)
        st_vis_plate_feat = st_vis_plate_feat[st_vis_plate_feat_idx_list, :]

        cancat_feats = [st_vis_plate_feat, st_vis_feat]
        cancat_feats_orig_idx_list = [st_vis_plate_feat_idx_list, st_vis_feat_idx_list]
        st_vis_plate_feat_idx_list = [i + pres_record_num[2] for i in st_vis_plate_feat_idx_list]
        st_vis_feat_idx_list = [i + pres_record_num[2] for i in st_vis_feat_idx_list]
        cancat_feats_idx_list = [st_vis_plate_feat_idx_list, st_vis_feat_idx_list]

        print("search topk")
        for i, idx_list in enumerate(cancat_feats_idx_list):
            f_ids[i] += idx_list

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
        f_topks = [
            [
                [
                    f_id[curr_topk_idx]
                    for curr_topk_idx in curr_topk_idxs
                    if curr_topk_idx < i + pres_record_num[modal_idx]
                ]
                for i, curr_topk_idxs in enumerate(topk_idxs)
            ]
            for modal_idx, (topk_idxs, f_id) in enumerate(zip(topk_idxs_list, f_ids))
        ]
        f_topks_score = [
            [
                [
                    curr_topk_scores[curr_topk_idxs_idx]
                    for curr_topk_idxs_idx, curr_topk_idx in enumerate(curr_topk_idxs)
                    if curr_topk_idx < i + pres_record_num[modal_idx]
                ]
                for i, (curr_topk_idxs, curr_topk_scores) in enumerate(zip(topk_idxs, topk_scores))
            ]
            for modal_idx, (topk_idxs, topk_scores) in enumerate(zip(topk_idxs_list, topk_scores_list))
        ]

        fusion_sim_time = time.time()
        topks = [[] for _ in range(record_num)]
        topks_score = [[] for _ in range(record_num)]
        # for the indexes combination, the features with plate have the higher assignment priority 
        for f_id, f_topk, f_topk_score in zip(cancat_feats_orig_idx_list, f_topks, f_topks_score):
            for i, topk, topk_score in zip(f_id, f_topk, f_topk_score):
                intersection_topk = list(set(topks[i]).intersection(topk))
                for j, k in enumerate(topk):
                    if k not in intersection_topk:
                        topks[i].append(k)
                        topks_score[i].append(topk_score[j])
        snapshots_retrieval_time = time.time() - snapshots_retrieval_time # record the similarity computation time
        print('fusion similarity consuming time: {}'.format(time.time() - fusion_sim_time))
        
        print("fast incremental clustering...")
        gen_cluster_time = time.time()
        sim_calc_time = 0
        statistics_update_time = 0
        sim_select_num = 0
        sim_computation_num = 0
        connect_flag_table = [np.array(topk_score) > sim_thres for topk_score in topks_score]
        data = [[a, b, c] if c is not None and not np.all(c == 0) else [a, b, None] for a, b, c in zip(st_feat, car_feat, plate_feat)]
        for record_idx1, connect_flag1 in tqdm(enumerate(connect_flag_table)):
            start_sim_calc_time = time.time()
            record_idx1_total_seq = record_idx1 + pres_record_num[2]
            connect_record_num = np.sum(connect_flag1 == True)
            if connect_record_num == 0:
                cids[record_idx1_total_seq] = curr_label
                id_dict[curr_label] = [record_idx1_total_seq]
                cfs[curr_label] = [[] if j is None else [j] for j in data[record_idx1]]
                curr_label += 1
                continue
            
            connect_record_idx_array = np.argwhere(connect_flag1 == True)
            topk_cid_list = []
            topk_sim_list = []
            cid_sim_dict = {}
            for connect_record_idx in connect_record_idx_array:
                record_idx2 = topks[record_idx1][connect_record_idx[0]]
                curr_sim = topks_score[record_idx1][connect_record_idx[0]]
                topk_cid_list.append(cids[record_idx2])
                topk_sim_list.append(curr_sim)
                if cids[record_idx2] not in cid_sim_dict.keys():
                    cid_sim_dict[cids[record_idx2]] = [curr_sim]
                else:
                    cid_sim_dict[cids[record_idx2]].append(curr_sim)
            
            counter = Counter(topk_cid_list)
            cid_times_list = counter.most_common() # get all connection
            select_cid_list = []
            quick_calc_select_cid_list = []
            for cid_times in cid_times_list:
                cid = cid_times[0]
                times = cid_times[1]
                if float(times) / len(id_dict[cid]) > adj_pt_ratio:
                    select_cid_list.append(cid)
                if float(times) / len(id_dict[cid]) > spher_distrib_coeff:
                    quick_calc_select_cid_list.append(cid)
            # print('the number of the selected cid list is: {}'.format(len(select_cid_list)))
            # print('the number of the quick calculation selected cid list is: {}'.format(len(quick_calc_select_cid_list)))
            # print(f'current snapshot similarity computation times: {len(select_cid_list) - len(quick_calc_select_cid_list)}')

            max_sim = 0
            final_cid = -1
            for cid in select_cid_list:
                if cid in quick_calc_select_cid_list:
                    curr_sim = np.max(cid_sim_dict[cid])
                else:
                    w_total = 0
                    curr_sim = 0
                    for w, a, cf_list in zip(weights, data[record_idx1], cfs[cid]):
                        if a is not None and len(cf_list) != 0:
                            b = normalize(np.mean(cf_list, axis=0))
                            curr_sim += a @ b * w
                            w_total += w
                    curr_sim /= w_total
                if curr_sim > max_sim:
                    max_sim = curr_sim
                    final_cid = cid

            sim_calc_time = time.time() - start_sim_calc_time + sim_calc_time
            sim_select_num += len(quick_calc_select_cid_list)
            sim_computation_num += len(select_cid_list) - len(quick_calc_select_cid_list)

            start_statistics_update_time = time.time()
            if max_sim > sim_thres:
                cids[record_idx1_total_seq] = final_cid
                id_dict[final_cid].append(record_idx1_total_seq)
                cf = cfs[final_cid]
                for j, k in enumerate(data[record_idx1]):
                    if k is not None:
                        if cf[j]:
                            cf[j].append(k)
                        else:
                            cf[j] = [k]
            else:
                cids[record_idx1_total_seq] = curr_label
                id_dict[curr_label] = [record_idx1_total_seq]
                cfs[curr_label] = [[] if j is None else [j] for j in data[record_idx1]]
                curr_label += 1

            statistics_update_time = time.time() - start_statistics_update_time + statistics_update_time
        
        gen_cluster_time = time.time() - gen_cluster_time

        print(f'snapshots retrieval time: {snapshots_retrieval_time}')
        print(f'cluster generation total cosuming time: {gen_cluster_time}')
        print(f'similarity computation total cosuming time: {sim_calc_time}')
        print(f'cluster statistics update total cosuming time: {statistics_update_time}')
        print(f'total similarity computation times: {sim_computation_num}')
        print(f'total selected times: {sim_select_num}')
        print(f'the number of snapshots is: {len(data)}')
        print("clustering finished!")

        # update the number of previous snapshots
        pres_record_num = [pres_record_num[0] + len(st_vis_plate_feat_idx_list), 
                           pres_record_num[1] + len(st_vis_feat_idx_list), 
                           pres_record_num[2] + record_num]

        return f_ids, cids, id_dict, cfs, curr_label, pres_record_num
