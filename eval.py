import os
import time
import pickle
import logging
import numpy as np
from collections import Counter
from collections import defaultdict


def _eval(pair):
    pair.sort(key=lambda x: x[0])
    vid2cid = defaultdict(list)
    gt_size = defaultdict(int)
    cid_size = defaultdict(int)
    for i, j in pair:
        vid2cid[i].append(j)
        cid_size[j] += 1
        if i != -1:
            gt_size[i] += 1
    vid2cid.pop(-1, None)
    assert len(gt_size) == len(vid2cid)
    precision = 0
    recall = 0
    vid_to_cid = {}
    for vid, cids in vid2cid.items():
        cs = [i for i in cids if i != -1]
        if cs:
            cid, cnt = max(Counter(cs).items(), key=lambda x: x[1])
            precision += cnt / cid_size[cid] * gt_size[vid]
            recall += cnt
            vid_to_cid[vid] = cid
    gt_total = sum(gt_size.values())
    precision /= gt_total
    recall /= gt_total
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    expansion = sum(len(set(i)) for i in vid2cid.values()) / len(vid2cid)
    return precision, recall, fscore, expansion, vid_to_cid

def evaluate_prf(gt_labels, pred_labels, log=True, save_name=None):
    if not isinstance(gt_labels, list):
        gt_labels = gt_labels.tolist()
    if not isinstance(pred_labels, list):
        pred_labels = pred_labels.tolist()
    pair = [(-1 if i is None else i, j) for i, j in zip(gt_labels, pred_labels)]
    
    precision, recall, fscore, expansion, vid_to_cid = _eval(pair)
    if log:
        print("------------------")
        print(
            f"clusters: {len(set(pred_labels)-set([-1]))}\nprecision: {precision}\nrecall:    {recall}\nfscore:    {fscore}\nexpansion: {expansion}"
        )
        print("------------------")
    
    return precision, recall, fscore, expansion, vid_to_cid

def load_corresp_tracklets_index_dict(dataset, tracklets_index_dict_file):
    # load the tracklets index dictionary from the file
    with open(tracklets_index_dict_file, 'rb') as file:
        all_tracklets_index_dict = pickle.load(file)

    # create a mapping of new indices to original indices
    sorted_indices_map = dict(enumerate(dataset.sorted_indices))

    # create a new dictionary with the mapped indices
    new_all_tracklets_index_dict = {
        i: all_tracklets_index_dict[orig_idx]
        for i, orig_idx in sorted_indices_map.items()
    }

    print(f'load dictionary with tracklet number: {len(dataset)}')
    return new_all_tracklets_index_dict

def evaluate_snapshots(gt_labels, pred_labels, curr_batch_camera_ids_arr, gt_label_dict, all_tracklets_index_dict, prev_processed_record_num, log=True):
    if not isinstance(gt_labels, list):
        gt_labels = gt_labels.tolist()
    if not isinstance(pred_labels, list):
        pred_labels = pred_labels.tolist()
    pair = [(i, j) for i, j in zip(gt_labels, pred_labels)]

    precision, recall, fscore, expansion, _ = _eval(pair)
    if log:
        print("------------------")
        print(f"clusters: {len(set(pred_labels)-set([-1]))}\nprecision: {precision}\nrecall:    {recall}\nfscore:    {fscore}\nexpansion: {expansion}")
        print("------------------")
    
    # evaluate trajectory recovery at snapshot granularity
    if isinstance(curr_batch_camera_ids_arr, np.ndarray):
        curr_batch_camera_ids_arr = curr_batch_camera_ids_arr.reshape(-1)
    pred_label_all = []
    gt_label_all = []
    for record_idx, (cam_id, label) in enumerate(zip(curr_batch_camera_ids_arr, pred_labels)):
        gt_label_arr = gt_label_dict[cam_id]
        idx_list = all_tracklets_index_dict[record_idx+prev_processed_record_num]
        gt_label = list(gt_label_arr[idx_list])
        gt_label_all.extend(gt_label)

        pred_label = [label] * len(idx_list)
        pred_label_all.extend(pred_label)
    
    pair = [(i, j) for i, j in zip(gt_label_all, pred_label_all)]
    precision, recall, fscore, expansion, vid_to_cid = _eval(pair)
    if log:
        print("------------------")
        print(f"clusters at the snapshot granularity: {len(set(pred_labels)-set([-1]))}\nprecision: {precision}\nrecall:    {recall}\nfscore:    {fscore}\nexpansion: {expansion}")
        print("------------------")

    return precision, recall, fscore, expansion, vid_to_cid

