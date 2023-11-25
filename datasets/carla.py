import re
import numpy as np
import json
import pyproj
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from tqdm import tqdm
from torch.utils.data import Dataset


def from_base64(s):
    import numpy as np
    import base64

    return np.frombuffer(base64.b64decode(s), np.float32)

class carla_img_dataset(Dataset):
    def __init__(self, record_file, cam_file, use_plate=False, train=True, training_traj_id_list=[], test_traj_id_list=[]):
        super().__init__()
        self.record_file = record_file
        self.cam_file = cam_file
        self.train = train
        self.use_plate = use_plate

        # load records
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
        
        # load camera information
        cameras = {}
        with open(cam_file, 'r') as f:
            for line in tqdm(f.readlines()):
                line_info = re.split(' |,', line.rstrip('\n'))
                cam_id = int(line_info[0])
                lat = float(line_info[1])
                lon = float(line_info[2])
                cameras[cam_id] = [lon, lat]
        self.cameras = cameras
        
        # define latitude-longitude coordinate system and projected coordinate system
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=cameras[0][0],
                south_lat_degree=cameras[0][1],
                east_lon_degree=cameras[0][0],
                north_lat_degree=cameras[0][1],
            ),
        )
        source_crs = CRS.from_epsg(4326) # WGS84 latitude and longitude coordinate system
        target_crs = CRS.from_epsg(utm_crs_list[0].code) # Carla data in city New York, in UTM Zone 18N
        # create converter
        coord_system_transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

        # set the camera position range
        min_lon, min_lat = 1000, 1000
        max_lon, max_lat = -1000, -1000
        for cam_id, (lon, lat) in cameras.items():
            if lon > max_lon:
                max_lon = lon
            if lon < min_lon:
                min_lon = lon
            if lat > max_lat:
                max_lat = lat
            if lat < min_lat:
                min_lat = lat
        min_x, min_y = coord_system_transformer.transform(min_lon, min_lat)
        max_x, max_y = coord_system_transformer.transform(max_lon, max_lat)
        self.mean_x, self.mean_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        self.range_x, self.range_y = max_x - min_x, max_y - min_y
        print('mean x: {}, mean y: {}, x range: {}, y range: {}'.format(self.mean_x, self.mean_y, self.range_x, self.range_y))

        # generate cameras cartesian x, y coordinates
        cameras_xy = {}
        for cam_id, (lon, lat) in cameras.items():
            x, y = coord_system_transformer.transform(lon, lat) # unit is meter
            normalized_x = x - self.mean_x
            normalized_y = y - self.mean_y
            cameras_xy[cam_id] = [normalized_x, normalized_y]
        self.cameras_xy = cameras_xy

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
        x = self.cameras_xy[cam_id][0]
        y = self.cameras_xy[cam_id][1]

        # to set the return variables as writeable variables
        img_feat_ = np.copy(img_feat)
        if self.use_plate and plate_feature is not None:
            plate_feature_ = np.copy(plate_feature)
        else:
            plate_feature_ = np.zeros(2048) # default dimension
        if label is not None:
            label_ = label
        else:
            label_ = -1
        
        return img_feat_, plate_feature_, time, x, y, cam_id, label_
