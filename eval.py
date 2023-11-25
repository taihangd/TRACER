import os
import time
import pickle
import logging
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
    pair = [
        (-1 if i is None else i, j)
        for i, j in zip(gt_labels, pred_labels)
    ]
    if save_name is not None:
        path = os.path.dirname(os.path.abspath(__file__))
        name = f'{path}/eval_history/{time.strftime("%Y%m%d_%H%M%S")}_{save_name}'
        pickle.dump(pair, open(name, "wb"))
        logging.info(f"saved to {name}")
    precision, recall, fscore, expansion, vid_to_cid = _eval(pair)
    if log:
        print("------------------")
        print(
            f"clusters: {len(set(pred_labels)-set([-1]))}\nprecision: {precision}\nrecall:    {recall}\nfscore:    {fscore}\nexpansion: {expansion}"
        )
        print("------------------")
    return precision, recall, fscore, expansion, vid_to_cid
