import argparse
import time
import pickle
import numpy as np
import re
import os
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec
from collections import defaultdict
import sys
sys.path.append(os.getcwd())
from utils import yaml_config_hook


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367 * 1000

def gen_road_graph(road_graph_pkl_file, node_file, edge_file):
    ## construct road network
    if not os.path.exists(road_graph_pkl_file):
        ## read out edge and node information
        node_id_to_loc_dict = {}
        with open(node_file, 'r') as f:
            for line in f.readlines():
                line = re.split(' |\t|,', line.strip('\n'))
                node_id_to_loc_dict[int(line[0])] = (float(line[1]), float(line[2]))

        edge_dict = {}
        with open(edge_file, 'r') as f:
            for line in f.readlines():
                line = re.split(' |\t|,', line.strip('\n'))
                edge_id = int(line[0])
                pre_node_id = int(line[1])
                succ_nod_id = int(line[2])
                edge_dict[edge_id] = [pre_node_id, succ_nod_id]

        # build the road network map
        road_graph = nx.MultiDiGraph()

        for node_id, loc in node_id_to_loc_dict.items():
            lat = loc[0]
            lon = loc[1]
            # add node
            road_graph.add_node(node_id, id=node_id, lon=lon, lat=lat)

        for edge_id, node_info in edge_dict.items():           
            pre_node_id = node_info[0]
            succ_node_id = node_info[1]
            pre_node_lon, pre_node_lat = node_id_to_loc_dict[pre_node_id]
            succ_node_lon, succ_node_lat = node_id_to_loc_dict[succ_node_id]
            
            node_dist = haversine(pre_node_lon, pre_node_lat, succ_node_lon, succ_node_lat)
            road_graph.add_edge(pre_node_id, succ_node_id, id=edge_id, node_id_list=[pre_node_id, succ_node_id], 
                                level=5, oneway=False,  length=node_dist)

        pickle.dump(road_graph, open(road_graph_pkl_file, "wb"))
        print("generate and save road network graph pkl file successfully!")
    else:
        road_graph = pickle.load(open(road_graph_pkl_file, "rb"))
    
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

# generate correspondence between camera nodes and road nodes
def gen_cam_dict(road_graph, cid_rid_correspondence_pkl_file, cam_nodes_file):
    if not os.path.exists(cid_rid_correspondence_pkl_file): 
        cam_pos_dict = {}
        f = open(cam_nodes_file)
        line = f.readline()
        while line:
            if line == '\n':
                line = f.readline()
                continue
            node_info = re.split(' |\t', line.strip('\n'))
            cam_pos_dict[int(node_info[0])] = [float(node_info[1]), float(node_info[2])]
            line = f.readline()
        f.close()
            
        # generate camera id road id correspondence
        cid_rid_correspondence_list = list()
        for curr_cam_id in cam_pos_dict.keys():
            curr_cam_lat = cam_pos_dict[curr_cam_id][0]
            curr_cam_lon = cam_pos_dict[curr_cam_id][1]

            cam_road_node = defaultdict(int)
            cam_road_node['id'] = curr_cam_id

            min_dist = np.inf
            for node in list(road_graph.nodes):
                curr_node = road_graph.nodes[node]
                curr_node_lon, curr_node_lat = curr_node['lon'], curr_node['lat']
                dist = haversine(curr_node_lon, curr_node_lat, curr_cam_lon, curr_cam_lat)
                if dist < min_dist:
                    cam_road_node['node_id'] = node
                    min_dist = dist
            
            cid_rid_correspondence_list.append(cam_road_node)
        
        # save as .pkl file
        pickle.dump(cid_rid_correspondence_list, open(cid_rid_correspondence_pkl_file, "wb"))
        print("generarate and save camera id to road id correspondence info .pkl file successfully!")
    else:
        cid_rid_correspondence_list = pickle.load(open(cid_rid_correspondence_pkl_file, "rb"))
        print("read out camera id to road id correspondence info .pkl file successfully!")

    return cid_rid_correspondence_list


if __name__ == "__main__":
    dataset_config_file = "./config/carla_tracklets_self_train_time_slice_sampl.yaml"
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(dataset_config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    # parameter setting
    road_graph_file = args.road_graph_file
    road_graph_node_vec_file = args.road_graph_node_vec_file
    cid_to_rid_file = args.cid_rid_correspondence_file

    dataset = args.dataset
    data_path = os.path.join('./data/', dataset)
    cam_nodes_file = os.path.join(data_path, 'cam_nodes.txt')
    # node and edge files used to generate road graph
    node_file = os.path.join(data_path, 'road_nodes.txt')
    edge_file = os.path.join(data_path, 'road_edges.txt')
    
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

    ## generate road graph in NetworkX type
    gen_road_graph_time = time.time()
    road_graph = gen_road_graph(road_graph_file, node_file, edge_file)
    gen_road_graph_time = time.time() - gen_road_graph_time
    print(f'generate road network successfully! consuming time: {gen_road_graph_time}')

    # generate road graph node embedding
    gen_road_graph_node_emb_time = time.time()
    node_embs = gen_road_graph_node_emb(road_graph_node_vec_file, road_graph, node2vec_param, device)
    gen_road_graph_node_emb_time = time.time() - gen_road_graph_node_emb_time
    print(f'generate road graph embedding successfully! consuming time: {gen_road_graph_node_emb_time}')

    # generate correspondence between camera id and node id
    gen_cam_dict_time = time.time()
    cid_rid_correspondence_list = gen_cam_dict(road_graph, cid_to_rid_file, cam_nodes_file)
    gen_cam_dict_time = time.time() - gen_cam_dict_time
    print(f'save camera id to road id correspondence info .pkl file successfully! consuming time: {gen_cam_dict_time}')

    print('Done!')