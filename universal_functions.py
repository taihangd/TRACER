import random
import pickle
import math
import copy
import torch
import numpy as np
import torch.nn as nn


class TimeIntervalSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, time_interval):
        self.dataset = dataset
        self.time_interval = time_interval
        self.batches = self._create_batches()

    def _create_batches(self):
        # get the sorted indices
        sorted_indices = list(range(len(self.dataset))) # the records have been sorted

        # split data into time intervals
        start_time = self.dataset.time_arr[sorted_indices[0]]
        batches = []
        current_batch = []
        for idx in sorted_indices:
            if self.dataset.time_arr[idx] < start_time + self.time_interval:
                current_batch.append(idx)
            else:
                batches.append(current_batch)
                current_batch = [idx]
                start_time = self.dataset.time_arr[idx]

        if current_batch:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    
class TimeIntervalRangeSampler(TimeIntervalSampler):
    def __init__(self, sampler, start, end):
        self.dataset = sampler.dataset
        self.time_interval = sampler.time_interval
        self.batches = sampler.batches[start: end]

def load_road_graph_node_embedding(road_graph_file, road_graph_node_vec_file, cid_rid_correspondence_file, device):
    # load road graph and node embeddings
    road_graph = pickle.load(open(road_graph_file, "rb"))
    road_graph_node_emb = pickle.load(open(road_graph_node_vec_file, 'rb'))
    road_graph_node_emb = road_graph_node_emb.to(device)
    # generate camera id to road_graph_node_emb index list
    cam_list = pickle.load(open(cid_rid_correspondence_file, 'rb'))
    c2r_dict = {x["id"]: x['node_id'] for x in cam_list}
    road_graph_node_id_list = list(road_graph.nodes())
    cam_vec_idx_dict= {}
    for cam_id, rid in c2r_dict.items():
        vec_idx = road_graph_node_id_list.index(rid)
        cam_vec_idx_dict[cam_id] = vec_idx
    cam_vec_idx_dict_keys_tensor = torch.tensor(list(cam_vec_idx_dict.keys()), device=device)
    cam_vec_idx_dict_values_tensor = torch.tensor(list(cam_vec_idx_dict.values()), device=device)

    return road_graph_node_emb, cam_vec_idx_dict, cam_vec_idx_dict_keys_tensor, cam_vec_idx_dict_values_tensor

def get_sample_idx_pairs_sampl_win(snapshot_label_list, time_arr, sampl_time_list, num_sampl_group, num_group_pairs, batch_size):
    label_snapshot_idx_dict = {}
    for i, label in enumerate(snapshot_label_list):
        if label not in label_snapshot_idx_dict:
            label_snapshot_idx_dict[label] = []
        label_snapshot_idx_dict[label].append(i)
    
    # batch_size must be less than the training trajectory number
    valid_label_snapshot_idx_dict = {label: idx_list for label, idx_list in label_snapshot_idx_dict.items() if len(idx_list) >= 2}
    valid_label_list = list(valid_label_snapshot_idx_dict.keys())
    if len(valid_label_list) < batch_size:
        for label, idx_list in list(label_snapshot_idx_dict.items()):
            if len(idx_list) < 2:
                label_snapshot_idx_dict[label].extend(idx_list * (2 - len(idx_list)))
        valid_label_snapshot_idx_dict = {label: idx_list for label, idx_list in label_snapshot_idx_dict.items() if len(idx_list) >= 2}
        valid_label_list = list(valid_label_snapshot_idx_dict.keys())

    # compute the group number for each time range
    groups_per_time_range = num_sampl_group // len(sampl_time_list)
    all_pairs_list = []

    for time_range in sampl_time_list:
        start_time, end_time = time_range
        time_range_label_snapshot_idx_dict = {label: [idx for idx in idx_list if start_time <= time_arr[idx] < end_time] for label, idx_list in valid_label_snapshot_idx_dict.items()}
        valid_time_range_label_snapshot_idx_dict = {label: idx_list for label, idx_list in time_range_label_snapshot_idx_dict.items() if len(idx_list) >= 2}
        time_range_valid_label_list = list(valid_time_range_label_snapshot_idx_dict.keys())
        
        if len(time_range_valid_label_list) < batch_size:
            for label, idx_list in list(time_range_label_snapshot_idx_dict.items()):
                if len(idx_list) < 2:
                    time_range_label_snapshot_idx_dict[label].extend(idx_list * (2 - len(idx_list)))
            valid_time_range_label_snapshot_idx_dict = {label: idx_list for label, idx_list in time_range_label_snapshot_idx_dict.items() if len(idx_list) >= 2}
            time_range_valid_label_list = list(valid_time_range_label_snapshot_idx_dict.keys())
        # skip this time range if not enough valid labels after extension
        if len(time_range_valid_label_list) < batch_size:
            continue

        # randomly select a batch of labels
        label_group_list = [random.sample(time_range_valid_label_list, batch_size) for _ in range(groups_per_time_range)]

        for label_group in label_group_list:
            curr_label_snapshot_idx_list = [valid_time_range_label_snapshot_idx_dict[label] for label in label_group]
            
            curr_group_pairs_list = []
            for _ in range(num_group_pairs): # sample data pair num_group_pairs times
                feat_idx_group = [random.choice(curr_label_snapshot_idx) for curr_label_snapshot_idx in curr_label_snapshot_idx_list]
                curr_group_pairs_list.append(feat_idx_group)
            
            all_pairs_list.append(curr_group_pairs_list)

    return all_pairs_list

def get_sample_idx_pairs(snapshot_label_list, num_sampl_group, num_group_pairs, batch_size):
    label_snapshot_idx_dict = {}
    for i, label in enumerate(snapshot_label_list):
        if label not in label_snapshot_idx_dict:
            label_snapshot_idx_dict[label] = []
        label_snapshot_idx_dict[label].append(i)
    
    # batch_size must be less than the training trajectory number
    valid_label_snapshot_idx_dict = {label: idx_list for label, idx_list in label_snapshot_idx_dict.items() if len(idx_list) >= 2}
    valid_label_list = list(valid_label_snapshot_idx_dict.keys())
    if len(valid_label_list) < batch_size:
        for label, idx_list in list(label_snapshot_idx_dict.items()):
            if len(idx_list) < 2:
                label_snapshot_idx_dict[label].extend(idx_list * (2 - len(idx_list)))
        valid_label_snapshot_idx_dict = {label: idx_list for label, idx_list in label_snapshot_idx_dict.items() if len(idx_list) >= 2}
        valid_label_list = list(valid_label_snapshot_idx_dict.keys())

    # randomly select a batch labels
    label_group_list = [random.sample(valid_label_list, batch_size) for i in range(num_sampl_group)]

    all_pairs_list = []
    for label_group in label_group_list:
        curr_label_snapshot_idx_list = [valid_label_snapshot_idx_dict[label] for label in label_group]
        
        curr_group_pairs_list = []
        for _ in range(num_group_pairs): # sample data pair num_group_pairs times
            feat_idx_group = [random.choice(curr_label_snapshot_idx) for curr_label_snapshot_idx in curr_label_snapshot_idx_list]
            curr_group_pairs_list.append(feat_idx_group)
        all_pairs_list.append(curr_group_pairs_list)

    return all_pairs_list

def extract_snapshot_feats(loader):
    print('Extracting features...')
    
    car_feat_vector = []
    plate_feat_vector = []
    time_vector = []
    lon_vector = []
    lat_vector = []
    cam_id_vector = []
    labels_vector = []
    for step, (car_feat, plate_feat, time, lon, lat, cam_id, label) in enumerate(loader):
        time = torch.unsqueeze(time, 1)
        lon = torch.unsqueeze(lon, 1)
        lat = torch.unsqueeze(lat, 1)
        cam_id = torch.unsqueeze(cam_id, 1)
        
        car_feat_vector.extend(car_feat.detach().numpy().astype('float32'))
        plate_feat_vector.extend(plate_feat.detach().numpy().astype('float32'))
        time_vector.extend(time.detach().numpy())
        lon_vector.extend(lon.detach().numpy())
        lat_vector.extend(lat.detach().numpy())
        cam_id_vector.extend(cam_id.detach().numpy())
        labels_vector.extend(label.numpy())
        
        if step % 20 == 0: # output the log information
            print(f"Step [{step}/{len(loader)}]\t Extracting features...")
    
    car_feat_vector = np.array(car_feat_vector)
    plate_feat_vector = np.array(plate_feat_vector)
    time_vector = np.array(time_vector)
    lon_vector = np.array(lon_vector)
    lat_vector = np.array(lat_vector)
    cam_id_vector = np.array(cam_id_vector)
    labels_vector = np.array(labels_vector)
    print("the shape of features: {}".format(car_feat_vector.shape))
    
    return car_feat_vector, plate_feat_vector, time_vector, lon_vector, lat_vector, cam_id_vector, labels_vector
    
def extract_st_feats(loader, model, device, cam_vec_idx_dict, road_graph_node_emb):
    print('Computing features...')
    model.eval()
    
    st_feat_vector = []
    for step, (_, _, time, lon, lat, cam_id, _) in enumerate(loader):
        time = torch.unsqueeze(time, 1)
        lon = torch.unsqueeze(lon, 1)
        lat = torch.unsqueeze(lat, 1)
        st_info = torch.cat((time, lon, lat), dim=1).float().to(device)
        
        cam_id = torch.unsqueeze(cam_id, 1)
        node_vec = torch.stack([road_graph_node_emb[cam_vec_idx_dict[id.item()]] for id in cam_id]) # retrieve the corresponding node embedding
        
        with torch.no_grad():
            st_feat = model.forward_extract_feat(st_info, node_vec)
        st_feat = st_feat.detach()
        st_feat_vector.extend(st_feat.cpu().detach().numpy())
        
        if step % 20 == 0: # output the log information
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    
    st_feat_vector = np.array(st_feat_vector)
    print("the shape of features: {}".format(st_feat_vector.shape))
    
    return st_feat_vector

def extract_st_feat_from_tensor(cam_ids, timestamp, lon, lat, model, device, cam_vec_idx_dict, road_graph_node_emb):
    model.eval()
    # prepare spatio-temporal information
    timestamp = torch.unsqueeze(timestamp, 1)
    lon = torch.unsqueeze(lon, 1)
    lat = torch.unsqueeze(lat, 1)
    st_info = torch.cat((timestamp, lon, lat), dim=1).float().to(device)

    # retrieve the corresponding node embedding
    cam_ids = torch.unsqueeze(cam_ids, 1)
    node_vec = torch.stack([road_graph_node_emb[cam_vec_idx_dict[id.item()]] for id in cam_ids])

    with torch.no_grad():
        st_feat = model.forward_extract_feat(st_info, node_vec)
    st_feat_vector = st_feat.cpu().detach().numpy()
    print("Extract spatio-temporal features: features shape {}".format(st_feat_vector.shape))
    
    return st_feat_vector

def extract_snapshot_feats_all(loader, model, device, cam_vec_idx_dict, road_graph_node_emb):
    print('Computing features...')
    model.eval()
    st_feat_vector = []
    car_feat_vector = []
    plate_feat_vector = []
    cam_id_vector = []
    labels_vector = []
    for step, (car_feat, plate_feat, time, lon, lat, cam_id, y) in enumerate(loader):
        time = torch.unsqueeze(time, 1)
        lon = torch.unsqueeze(lon, 1)
        lat = torch.unsqueeze(lat, 1)
        st_info = torch.cat((time, lon, lat), dim=1).float().to(device)
        
        # retrieve the corresponding node embedding
        node_vec = torch.stack([road_graph_node_emb[cam_vec_idx_dict[id.item()]] for id in cam_id])
        
        with torch.no_grad():
            c = model.forward_extract_feat(st_info, node_vec)
        c = c.detach()
        st_feat_vector.extend(c.cpu().detach().numpy())
        car_feat_vector.extend(car_feat.detach().numpy().astype('float32'))
        plate_feat_vector.extend(plate_feat.detach().numpy().astype('float32'))
        cam_id_vector.extend(cam_id.detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    
    st_feat_vector = np.array(st_feat_vector)
    car_feat_vector = np.array(car_feat_vector)
    plate_feat_vector = np.array(plate_feat_vector)
    cam_id_vector = np.array(cam_id_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(st_feat_vector.shape))
    
    return st_feat_vector, car_feat_vector, plate_feat_vector, cam_id_vector, labels_vector

def gen_cluster_multimodal_feat(st_feats, car_feats, plate_feats, pred_label, device, temperature, alpha):
    # traverse and record the number and index of each label
    label_dict = {}
    for idx, label in enumerate(pred_label):
        label_dict.setdefault(label, []).append(idx)

    cluster_id_list = list(label_dict.keys())
    label_idx_dict = {cluster_id: idx for idx, cluster_id in enumerate(cluster_id_list)} # map cluster id to cluster index
    cluster_num = len(cluster_id_list)

    # pre-allocate memory for centroids and concentrations
    cluster_result_st_feat_centroids = np.zeros((cluster_num, st_feats.shape[1]))
    cluster_result_car_feat_centroids = np.zeros((cluster_num, car_feats.shape[1]))
    cluster_result_plate_feat_centroids = np.zeros((cluster_num, plate_feats.shape[1]))
    cluster_result_st_feat_concentration = np.zeros(cluster_num)
    cluster_result_car_feat_concentration = np.zeros(cluster_num)
    cluster_result_plate_feat_concentration = np.zeros(cluster_num)

    for cluster_id in cluster_id_list:
        idx_list = label_dict[cluster_id]

        # compute centroids for each modality if there are non-zero features
        cluster_st_feat_list = st_feats[idx_list][~np.all(st_feats[idx_list] == 0, axis=1)]
        cluster_car_feat_list = car_feats[idx_list][~np.all(car_feats[idx_list] == 0, axis=1)]
        cluster_plate_feat_list = plate_feats[idx_list][~np.all(plate_feats[idx_list] == 0, axis=1)]
        if len(cluster_st_feat_list) > 0:
            st_feat_centroid = np.mean(cluster_st_feat_list, axis=0)
            cluster_result_st_feat_centroids[label_idx_dict[cluster_id]] = st_feat_centroid
        if len(cluster_car_feat_list) > 0:
            car_feat_centroid = np.mean(cluster_car_feat_list, axis=0)
            cluster_result_car_feat_centroids[label_idx_dict[cluster_id]] = car_feat_centroid
        if len(cluster_plate_feat_list) > 0:
            plate_feat_centroid = np.mean(cluster_plate_feat_list, axis=0)
            cluster_result_plate_feat_centroids[label_idx_dict[cluster_id]] = plate_feat_centroid
        
        # concentration estimation (phi)
        if len(cluster_st_feat_list) > 1:
            st_feat_dist_list = np.linalg.norm(cluster_st_feat_list - st_feat_centroid, axis=1)
            cluster_result_st_feat_concentration[label_idx_dict[cluster_id]] = np.mean(st_feat_dist_list) / np.log(len(st_feat_dist_list) + alpha)
        if len(cluster_car_feat_list) > 1:
            car_feat_dist_list = np.linalg.norm(cluster_car_feat_list - car_feat_centroid, axis=1)
            cluster_result_car_feat_concentration[label_idx_dict[cluster_id]] = np.mean(car_feat_dist_list) / np.log(len(car_feat_dist_list) + alpha)
        if len(cluster_plate_feat_list) > 1:
            plate_feat_dist_list = np.linalg.norm(cluster_plate_feat_list - plate_feat_centroid, axis=1)
            cluster_result_plate_feat_concentration[label_idx_dict[cluster_id]] = np.mean(plate_feat_dist_list) / np.log(len(plate_feat_dist_list) + alpha)
    
    # handle clusters with only one point by setting concentration to max value       
    st_feat_phi_max = np.max(cluster_result_st_feat_concentration)
    car_feat_phi_max = np.max(cluster_result_car_feat_concentration)
    plate_feat_phi_max = np.max(cluster_result_plate_feat_concentration)

    for cluster_id in cluster_id_list:
        if len(label_dict[cluster_id]) <= 1:
            idx = label_idx_dict[cluster_id]
            cluster_result_st_feat_concentration[idx] = st_feat_phi_max
            cluster_result_car_feat_concentration[idx] = car_feat_phi_max
            cluster_result_plate_feat_concentration[idx] = plate_feat_phi_max

    # post-process centroids and concentration
    def process_centroids_and_concentration(centroids, concentration, temperature):
        centroids = torch.tensor(centroids, dtype=torch.float32).to(device)
        centroids = nn.functional.normalize(centroids, p=2, eps=1e-12, dim=1)
        concentration = np.clip(concentration, np.percentile(concentration, 10), np.percentile(concentration, 90)) # trim extreme values for stability
        concentration = temperature * concentration / (np.mean(concentration) + 1e-12) #scale the mean to temperature
        concentration = torch.tensor(concentration, dtype=torch.float32).to(device)
        return centroids, concentration
    
    cluster_result_st_feat_centroids, cluster_result_st_feat_concentration = \
        process_centroids_and_concentration(cluster_result_st_feat_centroids, cluster_result_st_feat_concentration, temperature)
    cluster_result_car_feat_centroids, cluster_result_car_feat_concentration = \
        process_centroids_and_concentration(cluster_result_car_feat_centroids, cluster_result_car_feat_concentration, temperature)
    cluster_result_plate_feat_centroids, cluster_result_plate_feat_concentration = \
        process_centroids_and_concentration(cluster_result_plate_feat_centroids, cluster_result_plate_feat_concentration, temperature)
    
    return cluster_result_st_feat_centroids, cluster_result_car_feat_centroids, cluster_result_plate_feat_centroids, \
            cluster_result_st_feat_concentration, cluster_result_car_feat_concentration, cluster_result_plate_feat_concentration, label_idx_dict

def apply_random_perturbation(time_i, time_j, global_min_time, global_max_time, device):
    # calculate the minimum and maximum time of the snapshots
    min_time = min(time_i.min(), time_j.min())
    max_time = max(time_i.max(), time_j.max())
    
    # calculate the min and max time range for the perturbation
    perturb_min_time = global_min_time - min_time
    perturb_max_time = global_max_time - max_time
    
    # generate a single random perturbation value within the perturb_min_time and perturb_max_time range
    random_perturb_tensor = torch.empty(size=tuple(time_i.shape), device=device).uniform_(perturb_min_time.item(), perturb_max_time.item())
    
    # apply the perturbation
    time_i_perturbed = time_i + random_perturb_tensor
    time_j_perturbed = time_j + random_perturb_tensor
    
    return time_i_perturbed, time_j_perturbed

def train(model, optimizer, 
          all_pairs_list, 
          car_feats, plate_feats, time_arr, # tensor format by default
          lon_arr, lat_arr, camera_ids_arr,  
          labels, label_idx_dict, road_graph_node_emb, 
          batch_size, device, 
          cam_vec_idx_dict_keys_tensor, 
          cam_vec_idx_dict_values_tensor, 
          cluster_result_centroids, 
          cluster_result_densities, 
          min_time, max_time, 
          criterion_multi_modal_proto_noise_contra_estimation, 
          lamda_dict_loss, lamda_proto_loss):
    # switch to train mode
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean').to(device) # define loss function (criterion)
    
    loss_epoch = 0
    for pair_idx, curr_group_pairs in enumerate(all_pairs_list):
        for curr_group_list1, curr_group_list2 in zip(curr_group_pairs, curr_group_pairs[1:]):
            app_i = torch.as_tensor(car_feats[curr_group_list1], dtype=torch.float32, device=device)
            plate_i = torch.as_tensor(plate_feats[curr_group_list1], dtype=torch.float32, device=device)
            time_i = torch.as_tensor(time_arr[curr_group_list1], dtype=torch.float32, device=device).reshape(batch_size, -1)
            lon_i = torch.as_tensor(lon_arr[curr_group_list1], dtype=torch.float32, device=device).reshape(batch_size, -1)
            lat_i = torch.as_tensor(lat_arr[curr_group_list1], dtype=torch.float32, device=device).reshape(batch_size, -1)
            cam_id_i = torch.as_tensor(camera_ids_arr[curr_group_list1], dtype=torch.int32, device=device).reshape(batch_size, -1)

            app_j = torch.as_tensor(car_feats[curr_group_list2], dtype=torch.float32, device=device)
            plate_j = torch.as_tensor(plate_feats[curr_group_list2], dtype=torch.float32, device=device)
            time_j = torch.as_tensor(time_arr[curr_group_list2], dtype=torch.float32, device=device).reshape(batch_size, -1)
            lon_j = torch.as_tensor(lon_arr[curr_group_list2], dtype=torch.float32, device=device).reshape(batch_size, -1)
            lat_j = torch.as_tensor(lat_arr[curr_group_list2], dtype=torch.float32, device=device).reshape(batch_size, -1)
            cam_id_j = torch.as_tensor(camera_ids_arr[curr_group_list2], dtype=torch.int32, device=device).reshape(batch_size, -1)

            # add the random time shift perturbation
            time_i, time_j = apply_random_perturbation(time_i, time_j, min_time, max_time, device)

            # prepare the spatio-temporal information input
            st_info_i = torch.cat((time_i, lon_i, lat_i), dim=1)
            st_info_j = torch.cat((time_j, lon_j, lat_j), dim=1)

            # retrieve the corresponding node embedding
            cam_id_i_idx_tensor = torch.where(cam_vec_idx_dict_keys_tensor == cam_id_i.to(device))
            cam_id_j_idx_tensor = torch.where(cam_vec_idx_dict_keys_tensor == cam_id_j.to(device))
            vec_i_idx_tensor = cam_vec_idx_dict_values_tensor[cam_id_i_idx_tensor[1]]
            vec_j_idx_tensor = cam_vec_idx_dict_values_tensor[cam_id_j_idx_tensor[1]]
            node_vec_i = road_graph_node_emb[vec_i_idx_tensor]
            node_vec_j = road_graph_node_emb[vec_j_idx_tensor]
            
            # spatio-temporal feature
            output, target, st_i, st_j = model(st_info_i, st_info_j, node_vec_i, node_vec_j)

            # compute losses
            dict_loss = criterion(output, target)

            batch_label = torch.as_tensor([label_idx_dict[label] for label in labels[curr_group_list1 + curr_group_list2]], dtype=torch.int32, device=device).reshape(2 * batch_size, -1)
            proto_loss = criterion_multi_modal_proto_noise_contra_estimation(st_i, st_j, app_i, app_j, plate_i, plate_j, batch_label, cluster_result_centroids, cluster_result_densities)
            loss = lamda_dict_loss * dict_loss + lamda_proto_loss * proto_loss

            # train model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Step [{pair_idx}/{len(all_pairs_list)}]\t loss: {loss.item()}\t")
            loss_epoch += loss.item()

    return loss_epoch
