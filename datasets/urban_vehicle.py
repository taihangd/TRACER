import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset


def from_base64(s):
    import numpy as np
    import base64

    return np.frombuffer(base64.b64decode(s), np.float32)

class uv_img_dataset(Dataset):
    def __init__(self, record_file, cam_file, train=True, use_plate=True, training_traj_id_list=[], test_traj_id_list=[]):
        super().__init__()
        self.record_file = record_file
        self.cam_file = cam_file
        self.train = train
        self.use_plate = use_plate

        records = []
        with open(record_file, 'r') as f:
            for l in tqdm(f.readlines()):
                r = json.loads(l)
                r['car_feature'] = from_base64(r['car_feature'])
                if r['plate_feature'] is not None:
                    r['plate_feature'] = from_base64(r['plate_feature'])
                records.append(r)

        if train:
            train_record = [record for record in records if record['vehicle_id'] is not None and record['vehicle_id'] in training_traj_id_list]
            self.records = train_record
        else:
            test_record = [record for record in records if record['vehicle_id'] is None or record['vehicle_id'] in test_traj_id_list]
            self.records = test_record
        
        # save the number of records and trajectories
        self.record_num = len(self.records)
        record_label_dict = {record['vehicle_id'] for record in self.records}
        self.traj_num = len(record_label_dict)
        
        cameras = {}
        with open(cam_file, 'r') as f:
            for l in tqdm(f.readlines()):
                curr_cam_info = json.loads(l)
                cameras[curr_cam_info['camera_id']] = curr_cam_info['position']
        
        self.cameras = cameras

    def __len__(self):
        return len(self.records)  # dataset length
    
    def __getitem__(self, idx):
        # read out the number idx snapshot feature
        img_info = self.records[idx]
        label = img_info['vehicle_id']
        img_feat = img_info['car_feature']
        plate_feature = img_info['plate_feature']
        time = img_info['time']
        cam_id = img_info['camera_id']
        lon = self.cameras[cam_id][0]
        lat = self.cameras[cam_id][1]
        
        # to set the return variables as writeable variables
        img_feat_ = np.copy(img_feat)
        if self.use_plate and plate_feature is not None:
            plate_feature_ = np.copy(plate_feature)
        else:
            plate_feature_ = np.zeros(256)
        if label is not None:
            label_ = label
        else:
            label_ = -1

        return img_feat_, plate_feature_, time, lon, lat, cam_id, label_
