#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib.patches as patches

from dataloaders.visual_genome import VGDataLoader, VG, vg_collate
import numpy as np
import torch
from torch.autograd import Variable

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
import sys
import pickle as pkl

from collections import defaultdict
from graphviz import Digraph

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import faulthandler
faulthandler.enable()

def main():
    args = 'X -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/vgrel-motifnet-sgcls.tar -nepoch 50 -use_bias -multipred -cache motifnet_predcls1'
    sys.argv = args.split(' ')
    conf = ModelConfig()

    if conf.model == 'motifnet':
        from lib.rel_model import RelModel
    elif conf.model == 'stanford':
        from lib.rel_model_stanford import RelModelStanford as RelModel
    else:
        raise ValueError()

    train, val, test = VG.splits(
        num_val_im=conf.val_size, filter_duplicate_rels=True,
        use_proposals=conf.use_proposals,
        filter_non_overlap=conf.mode == 'sgdet',
    )
    if conf.test:
        val = test
    train_loader, val_loader = VGDataLoader.splits(
        train, val, mode='rel', batch_size=conf.batch_size,
        num_workers=conf.num_workers, num_gpus=conf.num_gpus
    )

    detector = RelModel(
        classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
        num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
        use_resnet=conf.use_resnet, order=conf.order,
        nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
        use_proposals=conf.use_proposals,
        pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
        pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
        pooling_dim=conf.pooling_dim,
        rec_dropout=conf.rec_dropout,
        use_bias=conf.use_bias,
        use_tanh=conf.use_tanh,
        limit_vision=conf.limit_vision
    )


    detector.cuda()
    ckpt = torch.load(conf.ckpt)

    optimistic_restore(detector, ckpt['state_dict'])

    evaluator = BasicSceneGraphEvaluator.all_modes(
        multiple_preds=conf.multi_pred)

    mode, N = 'test.multi_pred', 20
    recs = pkl.load(open('{}.{}.pkl'.format(mode, N), 'rb'))

    np.random.seed(0)
    # sorted_idxs = np.argsort(recs)
    selected_idxs = np.random.choice(range(len(recs)), size=100, replace=False)
    sorted_idxs = selected_idxs[np.argsort(np.array(recs)[selected_idxs])]
    print('Sorted idxs: {}'.format(sorted_idxs.tolist()))

    save_dir = '/nethome/bamos/2018-intel/data/2018-07-31/sgs.multi'

    for idx in selected_idxs:
        gt_entry = {
            'gt_classes': val.gt_classes[idx].copy(),
            'gt_relations': val.relationships[idx].copy(),
            'gt_boxes': val.gt_boxes[idx].copy(),
        }

        detector.eval()
        det_res = detector[vg_collate([test[idx]], num_gpus=1)]

        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = det_res
        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }


        unique_cnames = get_unique_cnames(gt_entry, test)
        save_img(idx, recs, test, gt_entry, det_res, unique_cnames, save_dir)
        save_gt_graph(idx, test, gt_entry, det_res, unique_cnames, save_dir)
        save_pred_graph(idx, test, pred_entry, det_res,
                        unique_cnames, save_dir,
                        multi_pred=conf.multi_pred, n_pred=20)

def get_unique_cnames(gt_entry, test):
    suffixes = []
    counts = defaultdict(int)
    for i in gt_entry['gt_classes']:
        ci = counts[i]
        if ci == 0:
            suffixes.append('-0')
        else:
            suffixes.append('-{}'.format(ci))
        counts[i] += 1

    unique_names = []
    for i, suffix in zip(gt_entry['gt_classes'], suffixes):
        name = test.ind_to_classes[i]
        if counts[i] > 1:
            name += suffix
        unique_names.append(name)
    return unique_names


def save_img(idx, recs, test, gt_entry, det_res, unique_cnames, save_dir):
    boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = det_res

    fig, ax = plt.subplots(figsize=(8,8))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    x0 = test[idx]['img']
    x0 = x0.clone().cpu().numpy()
    for c in range(3):
        x0[c] = x0[c] * std[c] + mean[c]
    x0 = np.transpose(x0, (1, 2, 0))
    ax.imshow(x0)
    ax.grid(False)
    ax.axis('off')

    for i, box in enumerate(gt_entry['gt_boxes']):
        box = box*test[idx]['scale']
        x1, y1, x2, y2 = box
        width = x2-x1
        height = y2-y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=3, edgecolor='blue', facecolor='none',
            label='xx'
        )
        ax.add_patch(rect)
        cls_name = unique_cnames[i]
        ax.text(
            x1, y1-8, cls_name, fontsize=10,
            bbox=dict(boxstyle="square", fc='white'),
        )

    ax.set_title('Test Recall: {:0.2f}'.format(recs[idx]), fontsize=26)
    fig.tight_layout()
    fig.savefig('{}/{}.img.png'.format(save_dir, idx))
    plt.close(fig)

def save_gt_graph(idx, test, gt_entry, det_res, unique_cnames, save_dir):
    boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = det_res
    dot = Digraph()
    dot.node_attr.update(color='lightblue2', style='filled')

    for cname in unique_cnames:
        dot.node(cname)

    for i, j, r in np.unique(gt_entry['gt_relations'], axis=0):
        clsi = unique_cnames[i]
        clsj = unique_cnames[j]
        relname = test.ind_to_predicates[r]
        dot.edge(clsi, clsj, relname)

    gt_fname = dot.render('{}/{}.gt'.format(save_dir, idx))

def save_pred_graph(idx, test, pred_entry, det_res, unique_cnames,
                    save_dir, multi_pred, n_pred):
    boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = det_res

    dot = Digraph()
    dot.node_attr.update(color='lightblue2', style='filled')

    for cname in unique_cnames:
        dot.node(cname)

    pred = pred_entry['rel_scores']
    if multi_pred:
        predflat = pred.reshape(-1)
        I = list(reversed(np.argsort(predflat)))
        rels_found = 0
        for i in I:
            nrel = pred.shape[1]
            row, col = int(np.floor(i/nrel)), i % nrel
            rel_type = col
            if rel_type > 0:
                ii, jj = pred_entry['pred_rel_inds'][row]
                clsii = unique_cnames[ii]
                clsjj = unique_cnames[jj]
                relname = test.ind_to_predicates[rel_type]
                dot.edge(clsii, clsjj, relname)
                rels_found += 1
                if rels_found == n_pred:
                    break
    else:
        top_pred_i = list(reversed(np.argsort(pred[:,1:].max(axis=1))))[:n_pred]
        for tripi in top_pred_i:
            rel_type = np.argmax(pred[tripi,1:])+1
            relname = test.ind_to_predicates[rel_type]
            ii, jj = pred_entry['pred_rel_inds'][tripi]
            clsii = unique_cnames[ii]
            clsjj = unique_cnames[jj]
            relname = test.ind_to_predicates[rel_type]
            dot.edge(clsii, clsjj, relname)

    pred_fname = dot.render('{}/{}.pred'.format(save_dir, idx))

if __name__ == '__main__':
    main()
