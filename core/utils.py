# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: util.py
# Time: 6/27/19 3:54 PM
# Description: 
# -------------------------------------------------------------------------------


from __future__ import print_function
import os
import logging
import torch
from torch.nn import init
from core.re_ranking import *


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).type_as(img).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


#######################################################################
# Evaluate
def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)

    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    index = np.argsort(score)
    index = index[::-1]

    query_index = np.argwhere(gl == ql)

    good_index = query_index

    CMC_tmp = compute_mAP(index, good_index)
    return CMC_tmp


def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def evaluate_CMC(query_features, query_labels, gallery_features, gallery_labels):
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    ap = 0.0
    for i in range(len(query_labels)):
        ap_tmp, CMC_tmp = evaluate(query_features[i], query_labels[i], gallery_features, gallery_labels)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_labels)  # average CMC
    return CMC, ap


#######################################################################
# Evaluate rerank
def evaluate_rerank(score, ql, gl):
    index = np.argsort(score)

    query_index = np.argwhere(gl == ql)
    good_index = query_index

    CMC_tmp = compute_mAP(index, good_index)
    return CMC_tmp


def evaluate_rerank_CMC(query_features, query_labels, gallery_features, gallery_labels):
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    ap = 0.0

    q_g_dist = np.dot(query_features, np.transpose(gallery_features))
    q_q_dist = np.dot(query_features, np.transpose(query_features))
    g_g_dist = np.dot(gallery_features, np.transpose(gallery_features))
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    for i in range(len(query_labels)):
        ap_tmp, CMC_tmp = evaluate_rerank(re_rank[i, :], query_labels[i], gallery_labels)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC / len(query_labels)  # average CMC

    return CMC, ap


#######################################################################
# Evaluate market
#
def evaluate_market(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    CMC_tmp = compute_mAP_market(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP_market(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


#####################################################################
# kaiming init
#
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def l2_norm(inputs, dim=1):
    norm = torch.norm(inputs, 2, dim, True)
    result = torch.div(inputs, norm)
    return result

# if __name__ == '__main__':
#     gallery = torch.FloatTensor(np.random.uniform(-1, 1, (5, 10)))
#     gallery_label = np.array([1, 2, 3, 4, 2])
#     query = torch.FloatTensor(np.random.uniform(-1, 1, (3, 10)))
#     query_label = np.array([2, 1, 3])
#
#     CMC = torch.IntTensor(len(gallery_label)).zero_()
#     ap = 0.0
#     for i in range(len(query_label)):
#         ap_tmp, CMC_tmp = evaluate(query[i], query_label[i], gallery, gallery_label)
#         if CMC_tmp[0] == -1:
#             continue
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#         # print(i, CMC_tmp[0])
#
#     CMC = CMC.float()
#     CMC = CMC / len(query_label)  # average CMC
#     print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[1], CMC[2], ap / len(query_label)))
