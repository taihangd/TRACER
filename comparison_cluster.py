import math
import time
import logging
import faiss
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.cluster import HDBSCAN
import sklearn.random_projection as rp


#%% the MMVC+ clustering algorithm
def normalize(feature):
    return feature / (np.linalg.norm(feature) + 1e-12)

class FlatSearcher:
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
            self.index = faiss.IndexFlatL2(feat_dim)

    def search_by_topk(self, query, gallery, topk=16):
        if query is None or len(query) == 0:
            return np.array([]), np.array([])
        
        self.index.reset()
        self.index.add(gallery)
        topk_scores, topk_idxs = self.index.search(query, topk)
        self.index.reset()
        
        return topk_scores, topk_idxs

    def search_by_topk_by_blocks(self, query, gallery, topk, query_num):
        if query is None or len(query) == 0:
            return np.array([]), np.array([])
        
        self.index.reset()
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
        
        self.index.reset()

        topk_scores = np.concatenate(topk_scores_list, axis=0)
        topk_idxs = np.concatenate(topk_idxs_list, axis=0)

        return topk_scores, topk_idxs

class SigCluster:
    def __init__(self, feature_dims=[256, 256], ngpu=1, useFloat16=False):
        self.searchers = {i: FlatSearcher(ngpu, i, useFloat16) for i in set(feature_dims)}
        self.f_dims = feature_dims

    def fit(
        self,
        data,
        initial_labels=None,
        weights=[1],
        similarity_threshold=0.88,
        topK=128,
        query_num=8192,
        normalized=True,
    ):
        snapshots_retrieval_time = time.time()
        if isinstance(weights, float) or isinstance(weights, int):
            weights = [weights] * len(self.f_dims)
        else:
            assert len(weights) == len(self.f_dims)
        if isinstance(topK, int):
            topK = [topK] * len(self.f_dims)
        else:
            assert len(topK) == len(self.f_dims)

        N = len(data)
        N_f = len(data[0])
        
        if normalized: # have replaced the None type with the all-zero array
            print("Normalize")
            data = [
                [None if np.all(j == 0) else normalize(j) for j in i] for i in tqdm(data)
            ]

        print("Search topk")
        data_ = list(zip(*data))
        fs = []
        f_ids = []
        for i in data_:
            tmp = [x for x in i if x is not None]
            if len(tmp) == 0: # if all snapshots for this feature dimension have no valid values
                continue
            f_id, f = zip(*((j, k) for j, k in enumerate(i) if k is not None))
            fs.append(np.array(f))
            f_ids.append(f_id)

        topk_idxs_list = []
        for f, dim, topk in zip(fs, self.f_dims, topK): # for each modal
            # topk_scores, topk_idxs = self.searchers[dim].search_by_topk(f, f, topk, query_num)
            topk_scores, topk_idxs = self.searchers[dim].search_by_topk_by_blocks(f, f, topk, query_num)
            topk_idxs_list.append(topk_idxs)
        f_topks = [
            [
                [
                    f_id[curr_topk_idx]
                    for curr_topk_idx in curr_topk_idxs
                    if curr_topk_idx < i
                ]
                for i, curr_topk_idxs in enumerate(topk_idxs)
            ]
            for topk_idxs, f_id in zip(topk_idxs_list, f_ids)
        ]
        assert all(len(i[0]) == 0 for i in f_topks)

        topks = [[] for _ in range(len(data))]
        for f_topk, f_id in zip(f_topks, f_ids):
            for i, topk in zip(f_id, f_topk):
                topks[i] += topk

        if not normalized: # have replaced the None type with the all-zero array
            data = [
                [None if np.all(j == 0) else normalize(j) for j in i] for i in tqdm(data)
            ]
        
        snapshots_retrieval_time = time.time() - snapshots_retrieval_time # record the similarity computation time

        print("Clustering")
        gen_cluster_time = time.time()
        cf_means = {}
        cfs = {}
        if initial_labels is None:
            cids = [-1] * N
        else:
            cids = initial_labels
            cid2records = defaultdict(list)
            for cid, record in zip(cids, data):
                if cid >= 0:
                    cid2records[cid].append(record)
            for cid, rs in cid2records.items():
                tmp = cfs[cid] = [[j for j in i if j is not None] for i in zip(*rs)]
                cf_means[cid] = [
                    normalize(np.mean(t, axis=0)) if len(t) else None for t in tmp
                ]
        
        sim_computation_num = 0
        sim_calc_time = 0
        statistics_update_time = 0
        for i, (record, topk) in enumerate(zip(tqdm(data), topks)):
            start_sim_calc_time = time.time()
            if cids[i] >= 0:
                continue
            cs = {cids[i] for i in topk}
            best_cid = -1
            best_sim = -1
            for c in cs:
                w_total = 0
                sim = 0
                for w, a, b in zip(weights, record, cf_means[c]):
                    if a is not None and b is not None:
                        sim += a @ b * w
                        w_total += w
                sim /= w_total
                if sim > best_sim:
                    best_sim = sim
                    best_cid = c
            
            sim_computation_num += len(cs)
            sim_calc_time = time.time() - start_sim_calc_time + sim_calc_time
            start_statistics_update_time = time.time()

            if best_cid >= 0 and best_sim >= similarity_threshold:
                cids[i] = best_cid
                cf = cfs[best_cid]
                cf_mean = cf_means[best_cid]
                for j, k in enumerate(record):
                    if k is not None:
                        if cf[j]:
                            cf[j].append(k)
                            cf_mean[j] = normalize(np.mean(cf[j], axis=0))
                        else:
                            cf[j] = [k]
                            cf_mean[j] = k
            else:
                cid = len(cf_means)
                cids[i] = cid
                cf_means[cid] = record
                cfs[cid] = [[] if j is None else [j] for j in record]

            statistics_update_time = time.time() - start_statistics_update_time + statistics_update_time

        gen_cluster_time = time.time() - gen_cluster_time

        # print(f'preprocessing time: {preprocess_time}')
        print(f'snapshots retrieval time: {snapshots_retrieval_time}')
        print(f'cluster generation total cosuming time: {gen_cluster_time}')
        print(f'similarity computation total cosuming time: {sim_calc_time}')
        print(f'cluster statistics update total cosuming time: {statistics_update_time}')
        print(f'similarity computation total times: {sim_computation_num}')
        print(f'the number of snapshots is: {len(data)}')
        print('Done!')

        return cids

#%% DBSCAN algorithms
# PDBSCAN
def parallel_dbscan(snapshot_feats, cfg=None):
    if cfg: # parse parameters
        n_components = cfg.pdbscan['random_proj_n_components']
        dense_output = cfg.pdbscan['random_proj_dense_output']
        random_state = cfg.pdbscan['random_proj_random_state']
        min_samples = cfg.pdbscan['min_samples']
        eps = cfg.pdbscan['eps']
    else: # set with default parameters
        n_components = 10
        dense_output = True
        random_state = 0
        min_samples = 5
        eps = 0.00001

    # random projection
    proj_trans = rp.SparseRandomProjection(n_components=n_components, dense_output=dense_output, random_state=random_state)
    feat_rp = proj_trans.fit_transform(snapshot_feats)
    print(feat_rp.shape)

    # original pdbscan library only support dimensionality 2 - 20, 
    # now set DBSCAN_MIN_DIMS/DBSCAN_MAX_DIMS to 2/200, then rebuild the dbscan library
    from dbscan import DBSCAN
    labels, core_samples_mask = DBSCAN(feat_rp, eps=eps, min_samples=min_samples)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    return core_samples_mask, labels, n_clusters

# HDBSCAN
def HDBSCAN_cluster(snapshot_feats, cfg=None):
    if cfg: # parse parameters
        n_components = cfg.hdbscan['random_proj_n_components']
        dense_output = cfg.hdbscan['random_proj_dense_output']
        random_state = cfg.hdbscan['random_proj_random_state']
        min_cluster_size = cfg.hdbscan['min_cluster_size']
        min_samples = cfg.hdbscan['min_samples']
        max_cluster_size = cfg.hdbscan['max_cluster_size']
        metric = cfg.hdbscan['metric']
        algorithm = cfg.hdbscan['algorithm']
        leaf_size = cfg.hdbscan['leaf_size']
        cluster_selection_method = cfg.hdbscan['cluster_selection_method']
    else: # set with default parameters
        n_components = 5
        dense_output = True
        random_state = 0
        min_cluster_size = 2
        min_samples = 1
        max_cluster_size = None
        metric = 'euclidean'
        algorithm = 'auto'
        leaf_size = 40
        cluster_selection_method = "eom"

    # generate random projection operator
    proj_trans = rp.SparseRandomProjection(n_components=n_components, 
                                    dense_output=dense_output, random_state=random_state)
    # random projection
    feat_rp = proj_trans.fit_transform(snapshot_feats)
    print(feat_rp.shape)

    # HDBSCAN algorithm
    hdbscan_cluster = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                              max_cluster_size=max_cluster_size, metric=metric, alpha=1.0, 
                              algorithm=algorithm, leaf_size=leaf_size, 
                              cluster_selection_method=cluster_selection_method)
    hdbscan_cluster.fit(feat_rp)
    # hdbscan_cluster.fit(snapshot_feats)

    cluster_label = hdbscan_cluster.labels_
    cluster_prob = hdbscan_cluster.probabilities_

    # # there may be problems with HDBSCAN boundary points, set confidence < 0.5 as noise points
    # cluster_label[cluster_prob < 0.5] = -1
    
    # the number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(cluster_label)) - (1 if -1 in cluster_label else 0)
    n_noise = list(cluster_label).count(-1)

    return cluster_label, cluster_prob, n_clusters

#%% K-Means clustering algorithm based on missing data
class KMeansMissData:
    def __init__(self, feats, use_plate, valid_mask, n_clusters, 
                 max_iterations=10, min_dist_change=10, 
                 resolve_empty='singleton', ngpu=1, 
                 weights=[1], feature_dims=[256, 256],
                 topK=1, query_num=8192, normalization=True,
    ):
        st_feat, car_feat, plate_feat = feats
        faiss.normalize_L2(st_feat)
        faiss.normalize_L2(car_feat)
        self.st_feat = st_feat
        self.car_feat = car_feat

        self.use_plate = use_plate
        if use_plate:
            faiss.normalize_L2(plate_feat)
            self.plate_feat = plate_feat

        # recombine different modal features
        if use_plate:
            self.joint_feat = np.concatenate((st_feat * math.sqrt(weights[0]), car_feat * math.sqrt(weights[1]), plate_feat * math.sqrt(weights[2])), axis=-1)
        else:
            weight_cumsum = np.cumsum(weights)
            norm_weight_st, norm_weight_vis = math.sqrt(weight_cumsum[0] / (weight_cumsum[0] + weights[1])), math.sqrt(weights[1] / (weight_cumsum[0] + weights[1]))
            st_vis_feat = np.concatenate((st_feat * norm_weight_st, car_feat * norm_weight_vis), axis=-1)
            self.joint_feat = st_vis_feat
        self.valid_mask = valid_mask
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.min_dist_change = min_dist_change
        self.resolve_empty = resolve_empty

        # initialize FlatSearcher with faiss
        self.searchers = {i: FlatSearcher(ngpu, i) for i in set(feature_dims)}
        self.f_dims = feature_dims
        
        self.weights = weights
        self.topK = topK
        self.query_num = query_num
        self.normalization = normalization
        
        (self.points_num, self.coords_num) = self.joint_feat.shape

        # initialize the similarity from data points to the assigned cluster centroids to zeros
        self.sim_array = np.ones(self.points_num)
    
    # initialize the cluster centroids randomly
    def init(self, seed=None, norm_init=False):
        if seed is not None:
            random.seed(seed)
        
        # # compute the mins and maxes of the columns - i.e. the min and max of each dimension
        if not norm_init:
            col_feat = [[self.joint_feat[i, j] for i in range(self.points_num) if self.valid_mask[i, j]] for j in range(self.coords_num)]
            self.mins = [min(feat) for feat in col_feat]
            self.maxs = [max(feat) for feat in col_feat]
        else:
            self.mins = [-1 for i in range(self.coords_num)]
            self.maxs = [1 for i in range(self.coords_num)]
        
        # Randomly initialize the cluster centroids
        self.centroids = [self.random_cluster_centroid() for k in range(self.n_clusters)]
        self.cluster_assignments = np.array([-1 for i in range(self.points_num)])
        self.mask_centroids = np.ones((self.n_clusters, self.coords_num))

    # Randomly place a new cluster centroids, picking uniformly between the min and max of each coordinate
    def random_cluster_centroid(self):
        centroid = []
        for coordinate in range(self.coords_num):
            value = random.uniform(self.mins[coordinate], self.maxs[coordinate])
            centroid.append(value)     
        return centroid    
    
    # perform the clustering, until there is no change
    def cluster(self):
        pre_dist_sum = None
        for curr_iter in range(self.max_iterations):
            print("========current iteration: {}========".format(curr_iter))
            
            if curr_iter != 0:
                update_cluster_start_time = time.time()
                for k in tqdm(range(self.n_clusters)): 
                    self.update_cluster(k)
                print("cluster update consume time:", time.time() - update_cluster_start_time)

            assignment_start_time = time.time()
            change, dist_sum = self.assignment()
            print("assignment consume time:", time.time() - assignment_start_time)
            if not change:
                break
            if pre_dist_sum is not None and abs(dist_sum - pre_dist_sum) < self.min_dist_change:
                break
            pre_dist_sum = dist_sum
            
        print("WARNING: did not converge, stopped after {} iterations.".format(self.max_iterations))

    # assign each data point to the closest cluster, 
    # return whether any reassignments were made
    def assignment(self):
        weights = self.weights
        topK = self.topK
        query_num = self.query_num
        if isinstance(topK, int):
            topK = [topK] * len(self.f_dims)
        else:
            assert len(topK) == len(self.f_dims)

        st_feat = self.st_feat
        car_feat = self.car_feat
        
        weight_cumsum = np.cumsum(weights)
        norm_weight_st, norm_weight_vis = math.sqrt(weight_cumsum[0] / (weight_cumsum[0] + weights[1])), math.sqrt(weights[1] / (weight_cumsum[0] + weights[1]))
        norm_weight_st_vis, norm_weight_plate = math.sqrt(weight_cumsum[1] / (weight_cumsum[1] + weights[2])), math.sqrt(weights[2] / (weight_cumsum[1] + weights[2]))
        st_vis_feat = np.concatenate((st_feat * norm_weight_st, car_feat * norm_weight_vis), axis=-1)

        if self.use_plate:
            plate_feat = self.plate_feat
            st_vis_plate_feat_idx_list = [i for i, feat in enumerate(plate_feat) if feat is not None and not np.all(feat == 0)]
            st_vis_plate_feat = np.concatenate((st_vis_feat * norm_weight_st_vis, plate_feat * norm_weight_plate), axis=-1)
            st_vis_plate_feat = st_vis_plate_feat[st_vis_plate_feat_idx_list, :]
            
            st_vis_feat_idx_list = [i for i in range(len(plate_feat)) if i not in st_vis_plate_feat_idx_list]
            st_vis_feat = st_vis_feat[st_vis_feat_idx_list, :]
            
            cancat_feats = [st_vis_plate_feat, st_vis_feat]
            cancat_feats_idx_list = st_vis_plate_feat_idx_list + st_vis_feat_idx_list
        else:
            cancat_feats = [st_vis_feat]
            cancat_feats_idx_list = list(range(len(st_vis_feat)))
        
        centroids = np.array(self.centroids, dtype=np.float32)
        topk_scores_list = []
        topk_idxs_list = []
        for i, (f, dim, topk) in enumerate(zip(cancat_feats, self.f_dims, topK)): # for each modal
            centroid_array = centroids[:, :dim].copy()
            faiss.normalize_L2(centroid_array)
            topk_scores, topk_idxs = self.searchers[dim].search_by_topk_by_blocks(f, centroid_array, topk, query_num)
            topk_scores_list.append(topk_scores)
            topk_idxs_list.append(topk_idxs)

        # get the topk score and the corresponding index
        score_list = []
        for topk_scores in topk_scores_list:
            score_list += topk_scores.flatten().tolist()
        idx_list = []
        for topk_idxs in topk_idxs_list:
            idx_list += topk_idxs.flatten().tolist()
        
        self.data_point_assignments = [[] for k in range(self.n_clusters)]
        change = False
        dist_sum = 0
        for i, (max_cosine_sim, new_cluster) in enumerate(zip(score_list, idx_list)):
            feat_idx = cancat_feats_idx_list[i]
            self.sim_array[feat_idx] = max_cosine_sim
            dist_sum += (1 - max_cosine_sim)
            
            orig_cluster = self.cluster_assignments[feat_idx]
            self.cluster_assignments[feat_idx] = new_cluster
            self.data_point_assignments[new_cluster].append(feat_idx)
            
            change = (change or orig_cluster != new_cluster)
        
        return change, dist_sum
    
    # Update the centroids to the mean of the points assigned to it. 
    # If for a coordinate there are no known values, we set this cluster's mask to 0 there.
    # If a cluster has no points assigned to it at all, we randomly re-initialize it.
    # Update for one specific cluster
    def update_cluster(self, k):
        known_coordinate_values = self.find_known_coord_values(k)
        
        if known_coordinate_values is None:
            # Reassign a datapoint to this cluster, as long as there are enough 
            # unique datapoints. Either furthest away (singleton) or random.
            if self.points_num >= self.n_clusters:
                if self.resolve_empty == 'singleton':
                    # Find the point currently furthest away from its centroid                    
                    index_furthest_away = self.find_point_least_similar()
                    old_cluster = self.cluster_assignments[index_furthest_away]
                    
                    # Add point to new cluster
                    self.centroids[k] = self.joint_feat[index_furthest_away]
                    self.mask_centroids[k] = self.valid_mask[index_furthest_away]
                    self.sim_array[index_furthest_away] = 1.0
                    self.cluster_assignments[index_furthest_away] = k
                    self.data_point_assignments[k] = [index_furthest_away]
                    
                    # Remove from old cluster and update
                    self.data_point_assignments[old_cluster].remove(index_furthest_away)
                    # print(k, old_cluster)
                    self.update_cluster(old_cluster)
                else:
                    # Randomly re-initialize this point
                    self.centroids[k] = self.random_cluster_centroid()
                    self.mask_centroids[k] = np.ones(self.coords_num)
        else:
            # For each coordinate set the centroid to the average, or to 0 if no values are observed
            for coordinate, coordinate_values in enumerate(known_coordinate_values):
                if len(coordinate_values) == 0:
                    new_coordinate = 0              
                    new_mask = 0
                else:
                    new_coordinate = sum(coordinate_values) / float(len(coordinate_values))
                    new_mask = 1
                
                self.centroids[k][coordinate] = new_coordinate
                self.mask_centroids[k][coordinate] = new_mask
    
    # For a given kth cluster centroid, construct a list of lists, each list consisting of
    # all known coordinate values of data points assigned to the centroid.
    # If no points are assigned to a cluster, return None.
    def find_known_coord_values(self, k):
        assigned_data_indexes = self.data_point_assignments[k]
        data_points = np.array([self.joint_feat[i] for i in assigned_data_indexes])
        masks = np.array([self.valid_mask[i] for i in assigned_data_indexes])
        
        if len(assigned_data_indexes) == 0:
            lists_known_coord_values = None
        else: 
            lists_known_coord_values = [
                [value for data_points_idx, value in enumerate(data_points.T[coordinate]) if masks[data_points_idx][coordinate]]
                for coordinate in range(self.coords_num)
            ]
            
        return lists_known_coord_values
        
    # Find data point furthest away from its current cluster centroid
    def find_point_least_similar(self):
        data_point_index = self.sim_array.argmin()
        return data_point_index

def kmeans_missing_data(st_feat, car_feat, plate_feat, cfg):  
    # generate mask
    mask1 = np.ones((len(plate_feat), st_feat.shape[1] + car_feat.shape[1]))
    mask2 = np.where(plate_feat != 0, 1, 0)
    if cfg.use_plate:
        mask = np.concatenate((mask1, mask2), axis=-1)
    else:
        mask = mask1

    use_plate = cfg.use_plate
    n_clusters = cfg.kmeas_missing_data['n_clusters']
    max_iter = cfg.kmeas_missing_data['max_iterations']
    min_dist_change = cfg.kmeas_missing_data['min_dist_change']
    resolve_empty = cfg.kmeas_missing_data['resolve_empty']
    ngpu = cfg.kmeas_missing_data['ngpu']
    weights = cfg.kmeas_missing_data['weights']
    feat_dims = cfg.kmeas_missing_data['feat_dims']
    topK = cfg.kmeas_missing_data['topK']
    query_num = cfg.kmeas_missing_data['query_num']
    normalization = cfg.kmeas_missing_data['normalization']
    feats = [st_feat, car_feat, plate_feat]
    kmeans = KMeansMissData(feats, use_plate, mask, n_clusters,
                            max_iter, min_dist_change=min_dist_change, 
                            resolve_empty=resolve_empty,
                            ngpu=ngpu, weights=weights, 
                            feature_dims=feat_dims,
                            topK=topK, query_num=query_num, 
                            normalization=normalization)
    print('KMeans for missing data instantiates object successfully!')
    norm_init = cfg.norm_init
    kmeans.init(cfg.seed, norm_init)
    print('KMeans for missing data initialization successfully!')
    # run missing data version kmeans
    kmeans.cluster()

    return kmeans.cluster_assignments

#%% agglomerative clustering based on missing data
def aggl_cluster_missing_data(st_feat, car_feat, plate_feat, cfg):
    # parse parameters
    ngpu = cfg.aggl_cluster_missing_data['ngpu']
    weights = cfg.aggl_cluster_missing_data['weights']
    feat_dims = cfg.aggl_cluster_missing_data['feat_dims']
    topK = cfg.aggl_cluster_missing_data['topK']
    query_num = cfg.aggl_cluster_missing_data['query_num']
    normalization = cfg.aggl_cluster_missing_data['normalization']
    use_plate = cfg.use_plate
    thres = cfg.aggl_cluster_missing_data['thres']

    # initialize FlatSearcher with faiss
    flat_searchers = {i: FlatSearcher(ngpu, i) for i in set(feat_dims)}

    # use faiss library to compute similarity between features
    if isinstance(topK, int):
        topK = [topK] * len(feat_dims)
    else:
        assert len(topK) == len(feat_dims)

    record_num = len(st_feat)
    if normalization: # have replaced the None type with the all-zero array
        faiss.normalize_L2(st_feat)
        faiss.normalize_L2(car_feat)
        faiss.normalize_L2(plate_feat)
    # convert all zero vector to none value
    weight_cumsum = np.cumsum(weights)
    norm_weight_st, norm_weight_vis = math.sqrt(weight_cumsum[0] / (weight_cumsum[0] + weights[1])), math.sqrt(weights[1] / (weight_cumsum[0] + weights[1]))
    norm_weight_st_vis, norm_weight_plate = math.sqrt(weight_cumsum[1] / (weight_cumsum[1] + weights[2])), math.sqrt(weights[2] / (weight_cumsum[1] + weights[2]))
    st_vis_feat = np.concatenate((st_feat * norm_weight_st, car_feat * norm_weight_vis), axis=1)
    st_vis_plate_feat = [np.concatenate((a * norm_weight_st_vis, b * norm_weight_plate), axis=0) if b is not None and not np.all(b == 0) else None for a, b in zip(st_vis_feat, plate_feat) ]
    cancat_feats = [st_vis_plate_feat, st_vis_feat]
    
    logging.info("Search topk")
    fs = []
    f_ids = []
    for curr_modal_feat in cancat_feats:
        if len([x for x in curr_modal_feat if x is not None and not np.all(x == 0)]) == 0: # if all snapshots for this feature dimension have no valid values
            continue
        f_id, f = zip(*((i, x) for i, x in enumerate(curr_modal_feat) if x is not None and not np.all(x == 0)))
        fs.append(np.array(f))
        f_ids.append(f_id)

    topk_scores_list = []
    topk_idxs_list = []
    for f, dim, topk in zip(fs, feat_dims, topK): # for each modal
        topk_scores, topk_idxs = flat_searchers[dim].search_by_topk_by_blocks(f, f, topk, query_num)
        topk_scores_list.append(topk_scores)
        topk_idxs_list.append(topk_idxs)
    f_topks = [
        [
            [f_id[curr_topk_idx] for curr_topk_idx in curr_topk_idxs]
            for curr_topk_idxs in topk_idxs
        ]
        for topk_idxs, f_id in zip(topk_idxs_list, f_ids)
    ]

    fusion_sim_time = time.time()
    topks = [[] for _ in range(record_num)]
    topks_score = [[] for _ in range(record_num)]
    # for multimodal data combinations, the smaller the index, the higher the assignment priority 
    for f_id, f_topk, f_topk_score in zip(f_ids, f_topks, topk_scores_list):
        for i, topk, topk_score in zip(f_id, f_topk, f_topk_score):
            intersection_topk = list(set(topks[i]).intersection(topk))
            for j, k in enumerate(topk):
                if k not in intersection_topk:
                    topks[i].append(k)
                    topks_score[i].append(topk_score[j])
    print('fusion similarity consuming time: {}'.format(time.time() - fusion_sim_time))
    
    # construct sparse similarity matrix
    # directly calculate a layer of linkage for clustering
    feat_connect_list = [[] for _ in range(record_num)]
    for feat_id1, (topk, topk_score) in tqdm(enumerate(zip(topks, topks_score))):
        for feat_id2, cosine_sim in zip(topk, topk_score):
            if feat_id1 == feat_id2:
                continue
            if cosine_sim > thres:
                feat_connect_list[feat_id1].append(feat_id2)
                feat_connect_list[feat_id2].append(feat_id1)

    # generate cluster results
    cluster_label_list = [-1] * record_num
    traverse_flag = [0] * record_num
    curr_cluster_label = 0
    for i in tqdm(range(record_num)):
        if traverse_flag[i]:
            continue
        
        queue = [i]
        while(queue):
            feat_id = queue.pop(0)
            if traverse_flag[feat_id]:
                continue

            cluster_label_list[feat_id] = curr_cluster_label
            traverse_flag[feat_id] = 1
            print('set {} label to feature id {}'.format(curr_cluster_label, feat_id))

            connect_list = feat_connect_list[feat_id]
            for connect_feat_id in connect_list:
                if traverse_flag[connect_feat_id]:
                    continue
                if connect_feat_id in queue:
                    continue
                queue.append(connect_feat_id)
        
        curr_cluster_label += 1

    # transfer format
    cluster_label_array = np.array(cluster_label_list)

    return cluster_label_array
