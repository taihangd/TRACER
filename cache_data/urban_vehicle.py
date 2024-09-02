import argparse
import time
import pickle
import numpy as np
import json
import os
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec
from collections import defaultdict
import sys
sys.path.append(os.getcwd())
from utils import yaml_config_hook


def load_cam(cam_file):
    cameras=[]
    with open(cam_file) as file:
        for l in file:
            cameras.append(json.loads(l))
    return cameras
    
def load_map(data_path):
    map = json.load(open(os.path.join(data_path, 'map.json')))
    return map

def coo_dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx ** 2 + dy ** 2)

def gen_road_graph(data_path, road_graph_file):
    if not os.path.exists(road_graph_file):
        road_graph = nx.MultiDiGraph()

        road_graph_info = load_map(data_path)
        # generate road graph
        node_list = list()
        edge_list = list()
        for curr_info in road_graph_info:
            if curr_info['type'] == 'node':
                node_list.append(curr_info)
            if curr_info['type'] == 'way':
                edge_list.append(curr_info)
        
        for curr_node_info in node_list:
            road_graph.add_node(curr_node_info['id'], x=curr_node_info['xy'][0], y=curr_node_info['xy'][1])
        
        road_sec_id = 0
        for curr_edge_info in edge_list:
            oneway = curr_edge_info['oneway']
            road_level = curr_edge_info['level']
            for nodeID_idx in range(len(curr_edge_info['nodes'])-1):
                pre_node_id = curr_edge_info['nodes'][nodeID_idx]
                succ_node_id = curr_edge_info['nodes'][nodeID_idx+1]

                [x1, y1] = [road_graph.nodes[pre_node_id]['x'], road_graph.nodes[pre_node_id]['y']]
                [x2, y2] = [road_graph.nodes[succ_node_id]['x'], road_graph.nodes[succ_node_id]['y']]
                node_dist = coo_dist(x1, y1, x2, y2)
                road_graph.add_edge(pre_node_id, succ_node_id, id=road_sec_id, 
                                    node_id_list=[pre_node_id, succ_node_id], 
                                    level=road_level, oneway=oneway, length=node_dist)
                road_sec_id += 1
                if oneway == False:
                    road_graph.add_edge(succ_node_id, pre_node_id, id=road_sec_id, 
                                        node_id_list=[succ_node_id, pre_node_id], 
                                        level=road_level, oneway=oneway, length=node_dist)
                    road_sec_id += 1

        pickle.dump(road_graph, open(road_graph_file, "wb"))
    else:
        road_graph = pickle.load(open(road_graph_file, "rb"))

    return road_graph

def gen_road_graph_node_emb(road_graph_node_vec_file, road_graph, node2vec_param, device):
    if not os.path.exists(road_graph_node_vec_file):
        model = Node2Vec(from_networkx(road_graph).edge_index, 
                         embedding_dim=node2vec_param['dimensions'],
                         walk_length=node2vec_param['walk_length'],
                        context_size=node2vec_param['context_size'],
                        walks_per_node=node2vec_param['num_walks'],
                        num_negative_samples=node2vec_param['num_negative_samples'],
                        p=node2vec_param['p'],
                        q=node2vec_param['q'],
                        sparse=node2vec_param['sparse']).to(device)
        
        loader = model.loader(batch_size=node2vec_param['batch_size'], shuffle=node2vec_param['shuffle'], num_workers=node2vec_param['num_workers'])
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=node2vec_param['lr'])
        
        def train_node_emb():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        for epoch in range(node2vec_param['epochs']):
            loss = train_node_emb()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        node_embs = model().detach().cpu() # get the node embedding
        pickle.dump(node_embs, open(road_graph_node_vec_file, 'wb'))
    else:
        node_embs = pickle.load(open(road_graph_node_vec_file, 'rb'))

    return node_embs

# generate the corresponding relationships between cameras and road node
def gen_cid_to_rid_correspondence(cid_to_rid_file, cameras, road_graph):
    if not os.path.exists(cid_to_rid_file):
        cam_list = list()
        for cam in cameras:
            curr_cam_id = cam['camera_id']
            curr_cam_coo = cam['position']
            x1, y1 = curr_cam_coo[0], curr_cam_coo[1]

            cam_road_node = defaultdict(int)
            cam_road_node['id'] = curr_cam_id

            min_dist = np.inf
            for node in list(road_graph.nodes):
                curr_node = road_graph.nodes[node]
                x2, y2 = curr_node['x'], curr_node['y']
                dist = coo_dist(x1, y1, x2, y2)
                if dist < min_dist:
                    cam_road_node['node_id'] = node
                    min_dist = dist
            
            cam_list.append(cam_road_node)
        
        print(len(cam_list))
        pickle.dump(cam_list, open(cid_to_rid_file, "wb"))
    else:
        cam_list = pickle.load(open(cid_to_rid_file, "rb"))
    
    return cam_list


if __name__ == "__main__":
    dataset_config_file = "./config/uv_self_train_time_slice_sampl.yaml"
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(dataset_config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # parameter setting
    dataset = args.dataset
    proj_path = "./"
    cache_path = os.path.join(proj_path, 'cache_data', dataset)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load node2vec parameters
    node2vec_param = {'dimensions': args.node2vec['dimensions'], 
                    'walk_length': args.node2vec['walk_length'], 
                    'context_size': args.node2vec['context_size'], 
                    'num_walks': args.node2vec['num_walks'], 
                    'num_negative_samples': args.node2vec['num_negative_samples'], 
                    'p': args.node2vec['p'], 
                    'q': args.node2vec['q'], 
                    'sparse': args.node2vec['sparse'], 
                    'num_workers': args.node2vec['num_workers'], 
                    'batch_size': args.node2vec['batch_size'], 
                    'shuffle': args.node2vec['shuffle'], 
                    'lr': args.node2vec['lr'], 
                    'epochs': args.node2vec['epochs']}

    data_path = "./data/UrbanVehicle/"
    cam_node_file = args.cam_file
    road_graph_file = args.road_graph_file
    cid_to_rid_file = args.cid_rid_correspondence_file
    road_graph_node_vec_file = args.road_graph_node_vec_file

    # generate cache data
    gen_road_graph_time = time.time()
    road_graph = gen_road_graph(data_path, road_graph_file)
    gen_road_graph_time = time.time() - gen_road_graph_time
    print(f'generate road graph and statistics prior successfully! consuming time: {gen_road_graph_time}')

    # generate road graph node embedding
    gen_road_graph_node_emb_time = time.time()
    node_embs = gen_road_graph_node_emb(road_graph_node_vec_file, road_graph, node2vec_param, device)
    gen_road_graph_node_emb_time = time.time() - gen_road_graph_node_emb_time
    print(f'generate road graph embedding successfully! consuming time: {gen_road_graph_node_emb_time}')

    # generate correspondence relationship between camera id and road id
    gen_cam_dict_time = time.time()
    cameras = load_cam(cam_node_file)
    print('load camera information successfully!')
    cam_list = gen_cid_to_rid_correspondence(cid_to_rid_file, cameras, road_graph)
    c2r_dict = {x["id"]: x['node_id'] for x in cam_list}
    gen_cam_dict_time = time.time() - gen_cam_dict_time
    print(f'generate camera information successfully! consuming time: {gen_cam_dict_time}')

    print('Done!')
