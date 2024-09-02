import json
import math
import base64
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


def from_base64(s):
    return np.frombuffer(base64.b64decode(s), np.float32)

class uv_img_dataset(Dataset):
    def __init__(self, 
                 record_file, 
                 cam_file, 
                 train=True, 
                 use_plate=True, 
                 sampl_rate=0.25,
                 time_gap_length=5500, 
                 time_length=44000,
                 fps=1,
                 self_supervision=True):
        super().__init__()
        self.record_file = record_file
        self.cam_file = cam_file
        self.train = train
        self.use_plate = use_plate
        self.sampl_rate = sampl_rate
        self.time_gap_length = time_gap_length

        # load records
        car_feats_list = []
        plate_feats_list = []
        time_list = []
        camera_ids_list = []
        gt_labels_list = []
        with open(record_file, 'r') as f: # the license plate text is not considered currently.
            for l in tqdm(f):
                r = json.loads(l)
                r['car_feature'] = from_base64(r['car_feature'])
                if r['plate_feature'] is not None:
                    r['plate_feature'] = from_base64(r['plate_feature'])
                car_feats_list.append(r['car_feature'])
                plate_feats_list.append(r['plate_feature'] if use_plate and r['plate_feature'] is not None else np.zeros(256)) # stay consistent with original features
                time_list.append(r['time'])
                camera_ids_list.append(r['camera_id'])
                gt_labels_list.append(r['vehicle_id'] if r['vehicle_id'] is not None else -1)

        # convert to arrays
        car_feats_arr = np.array(car_feats_list, dtype=np.float32)
        plate_feats_arr = np.array(plate_feats_list, dtype=np.float32)
        time_arr = np.array(time_list, dtype=np.float32)
        camera_ids_arr = np.array(camera_ids_list, dtype=np.int32)
        gt_labels_arr = np.array(gt_labels_list, dtype=np.int32)

        # sort tensors by time
        sorted_indices = np.argsort(time_arr)
        car_feats_arr = car_feats_arr[sorted_indices]
        plate_feats_arr = plate_feats_arr[sorted_indices]
        time_arr = time_arr[sorted_indices]
        camera_ids_arr = camera_ids_arr[sorted_indices]
        gt_labels_arr = gt_labels_arr[sorted_indices]

        # get the record_start time of records
        record_start_time = time_arr[0]
        self.record_start_time = record_start_time

        if train:
            time_slice_length = sampl_rate * time_gap_length * fps
            time_slice_num = int(math.ceil(time_length / time_gap_length))
            sampl_time_list = [
                [record_start_time + i * time_gap_length * fps, 
                 record_start_time + i * time_gap_length * fps + time_slice_length] 
                for i in range(time_slice_num)
            ]
            
            mask = np.zeros(len(car_feats_arr), dtype=bool)
            for start_time, end_time in sampl_time_list:
                mask |= (time_arr >= start_time) & (time_arr < end_time)
            if not self_supervision:
                mask &= (gt_labels_arr != -1)
            
            self.sampl_time_list = sampl_time_list
            self.sorted_indices = sorted_indices[mask]
            self.car_feats_arr = car_feats_arr[mask]
            self.plate_feats_arr = plate_feats_arr[mask]
            self.time_arr = time_arr[mask]
            self.camera_ids_arr = camera_ids_arr[mask]
            self.gt_labels_arr = gt_labels_arr[mask]
        else:
            self.sorted_indices = sorted_indices
            self.car_feats_arr = car_feats_arr
            self.plate_feats_arr = plate_feats_arr
            self.time_arr = time_arr
            self.camera_ids_arr = camera_ids_arr
            self.gt_labels_arr = gt_labels_arr
        
        # save the number of records and trajectories
        self.record_num = len(self.car_feats_arr)
        flattened_gt_labels_arr = self.gt_labels_arr.flatten()
        self.traj_num = len(np.unique(flattened_gt_labels_arr[flattened_gt_labels_arr != -1]))
        print(f'record number: {self.record_num}, trajectory number: {self.traj_num}')
        
        cameras = {}
        with open(cam_file, 'r') as f:
            for l in tqdm(f.readlines()):
                curr_cam_info = json.loads(l)
                cameras[curr_cam_info['camera_id']] = curr_cam_info['position']
        self.cameras = cameras

        # generate lon and lat array
        lon_arr = np.array([cameras[cam_id][0] for cam_id in self.camera_ids_arr], dtype=np.float32)
        lat_arr = np.array([cameras[cam_id][1] for cam_id in self.camera_ids_arr], dtype=np.float32)
        # note that we do not need to sort or call mask filtering again, because camera_ids_arr has already gone through these operations.
        self.lon_arr = lon_arr
        self.lat_arr = lat_arr

    def __len__(self):
        return len(self.car_feats_arr)  # dataset length
    
    def __getitem__(self, idx):
        # read out the number idx snapshot feature
        img_feat = self.car_feats_arr[idx]
        plate_feat = self.plate_feats_arr[idx]
        time = self.time_arr[idx]
        x = self.lon_arr[idx]
        y = self.lat_arr[idx]
        cam_id = self.camera_ids_arr[idx]
        label = self.gt_labels_arr[idx]

        return img_feat, plate_feat, time, x, y, cam_id, label
    