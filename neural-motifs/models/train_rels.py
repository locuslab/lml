"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""

import os, sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('{}/../'.format(script_dir))

from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

import pandas as pd
import time
import os

import pickle as pkl

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

from setproctitle import setproctitle
setproctitle('bamos.neural_motifs.train_rels')

conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()


# torch.backends.cudnn.enabled = False

# @profile
def main():
    fname = os.path.join(conf.save_dir, 'train_losses.csv')
    train_f = open(fname, 'w')
    train_f.write('iter,class_loss,rel_loss,total,recall20,recall50,recall100,recall20_con,recall50_con,recall100_con\n')
    train_f.flush()

    fname = os.path.join(conf.save_dir, 'val_losses.csv')
    val_f = open(fname, 'w')
    val_f.write('recall20,recall50,recall100,recall20_con,recall50_con,recall100_con\n')
    val_f.flush()

    train, val, _ = VG.splits(
        num_val_im=conf.val_size, filter_duplicate_rels=True,
        use_proposals=conf.use_proposals,
        filter_non_overlap=conf.mode == 'sgdet')
    train_loader, val_loader = VGDataLoader.splits(
        train, val, mode='rel',
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        num_gpus=conf.num_gpus
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
        limit_vision=conf.limit_vision,
        lml_topk=conf.lml_topk,
        lml_softmax=conf.lml_softmax,
        entr_topk=conf.entr_topk,
        ml_loss=conf.ml_loss
    )

    # Freeze the detector
    for n, param in detector.detector.named_parameters():
        param.requires_grad = False

    print(print_para(detector), flush=True)


    ckpt = torch.load(conf.ckpt)
    if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
        print("Loading EVERYTHING")
        start_epoch = ckpt['epoch']

        if not optimistic_restore(detector, ckpt['state_dict']):
            start_epoch = -1
            # optimistic_restore(
            #     detector.detector,
            #     torch.load('checkpoints/vgdet/vg-28.tar')['state_dict']
            # )
    else:
        start_epoch = -1
        optimistic_restore(detector.detector, ckpt['state_dict'])

        detector.roi_fmap[1][0].weight.data.copy_(
            ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap[1][3].weight.data.copy_(
            ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap[1][0].bias.data.copy_(
            ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap[1][3].bias.data.copy_(
            ckpt['state_dict']['roi_fmap.3.bias'])

        detector.roi_fmap_obj[0].weight.data.copy_(
            ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap_obj[3].weight.data.copy_(
            ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap_obj[0].bias.data.copy_(
            ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap_obj[3].bias.data.copy_(
            ckpt['state_dict']['roi_fmap.3.bias'])

    detector.cuda()


    print("Training starts now!")
    optimizer, scheduler = get_optim(
        detector, conf.lr * conf.num_gpus * conf.batch_size
    )
    best_eval = None
    for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
        rez = train_epoch(
            epoch, detector, train, train_loader, optimizer, conf, train_f)
        print("overall{:2d}: ({:.3f})\n{}".format(
            epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)

        mAp = val_epoch(detector, val, val_loader, val_f)
        scheduler.step(mAp)

        if conf.save_dir is not None:
            if best_eval is None or mAp > best_eval:
                torch.save({
                    'epoch': epoch,
                    'state_dict': detector.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                }, os.path.join(conf.save_dir, 'best-val.tar'))
                best_eval = mAp

        # if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 \
        #         for pg in optimizer.param_groups]):
        #     print("exiting training early", flush=True)
        #     break


def train_epoch(epoch_num, detector, train, train_loader, optimizer, conf, train_f):
    detector.train()
    tr = []
    total_iter = len(train_loader)
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(
            b, batch, detector, train, optimizer,
            verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        train_f.write(','.join(map(str, [
            epoch_num+b/total_iter,
            tr[-1].class_loss,
            tr[-1].rel_loss,
            tr[-1].total,
            tr[-1].recall20,
            tr[-1].recall50,
            tr[-1].recall100,
            tr[-1].recall20_con,
            tr[-1].recall50_con,
            tr[-1].recall100_con,
        ])) + '\n')
        train_f.flush()
        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, total_iter, time_per_batch,
                total_iter * time_per_batch / 60
            ))
            print(mn)
            print('-----------', flush=True)
            start = time.time()

    return pd.concat(tr, axis=1)



# @profile
def train_batch(batch_num, b, detector, train, optimizer, verbose=False):
    """
    :param b: contains:
        :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
        :param all_anchors: [num_anchors, 4] the boxes of all anchors
            that we'll be using
        :param all_anchor_inds: [num_anchors, 2] array of the indices
            into the concatenated RPN feature vector that give us all_anchors,
            each one (img_ind, fpn_idx)
        :param im_sizes: a [batch_size, 4] numpy array of
            (h, w, scale, num_good_anchors) for each image.

        :param num_anchors_per_img: int, number of anchors in total
             over the feature pyramid per img

        Training parameters:
        :param train_anchor_inds: a [num_train, 5] array of indices for
             the anchors that will be used to compute the training loss
             (img_ind, fpn_idx)
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """

    result, result_preds = detector[b]

    losses = {}
    losses['class_loss'] = F.cross_entropy(
        result.rm_obj_dists, result.rm_obj_labels
    )
    n_rel = len(train.ind_to_predicates)

    if conf.lml_topk is not None and conf.lml_topk:
        # Note: This still uses a maximum of 1 relationship per edge
        # in the graph. Adding them all requires changing the data loading
        # process.
        gt = result.rel_labels[:,-1]

        I = gt > 0
        gt = gt[I]
        n_pos = len(gt)

        reps = torch.cat(result.rel_reps)
        I_reps = I.unsqueeze(1).repeat(1, n_rel)
        reps = reps[I_reps].view(-1, n_rel)

        loss = []
        for i in range(n_pos):
            gt_i = gt[i]
            reps_i = reps[i]
            loss_i = -(reps_i[gt_i].log())
            loss.append(loss_i)

        loss = torch.cat(loss)
        loss = torch.sum(loss) / n_pos
        losses['rel_loss'] = loss
    elif conf.ml_loss:
        loss = []

        start = 0
        for i, rel_reps_i in enumerate(result.rel_reps):
            n = rel_reps_i.shape[0]

            # Get rid of the background labels here:
            reps = result.rel_dists[start:start+n,1:].contiguous().view(-1)
            gt = result.rel_labels[start:start+n,-1].data.cpu()
            I = gt > 0
            gt = gt[I]
            gt = gt - 1 # Hacky shift to get rid of background labels.
            r = (n_rel-1)*torch.arange(len(I))[I].long()
            gt_flat = r + gt
            gt_flat_onehot = torch.zeros(len(reps))
            gt_flat_onehot.scatter_(0, gt_flat, 1)
            loss_i = torch.nn.BCEWithLogitsLoss(size_average=False)(
                reps, Variable(gt_flat_onehot.cuda()))
            loss.append(loss_i)

            start += n

        loss = torch.cat(loss)
        loss = torch.sum(loss) / len(loss)
        losses['rel_loss'] = loss
    elif conf.entr_topk is not None and conf.entr_topk:
        # Note: This still uses a maximum of 1 relationship per edge
        # in the graph. Adding them all requires changing the data loading
        # process.
        loss = []

        start = 0
        for i, rel_reps_i in enumerate(result.rel_reps):
            n = rel_reps_i.shape[0]

            # Get rid of the background labels here:
            reps = result.rel_dists[start:start+n,1:].contiguous().view(-1)
            if len(reps) <= conf.entr_topk:
                # Nothing to do for small graphs.
                continue

            gt = result.rel_labels[start:start+n,-1].data.cpu()
            I = gt > 0
            gt = gt[I]
            gt = gt - 1 # Hacky shift to get rid of background labels.
            r = (n_rel-1)*torch.arange(len(I))[I].long()
            gt_flat = r + gt
            n_pos = len(gt_flat)

            if n_pos == 0:
                # Nothing to do if there is no ground-truth data.
                continue

            reps_sorted, J = reps.sort(descending=True)
            reps_sorted_last = reps_sorted[conf.entr_topk:]
            J_last = J[conf.entr_topk:]

            # Hacky way of removing the ground-truth from J.
            J_last_bool = J_last != gt_flat[0]
            for j in range(n_pos-1):
                J_last_bool *= (J_last != gt_flat[j+1])
            J_last_bool = J_last_bool.type_as(reps)

            loss_i = []
            for j in range(n_pos):
                yj = gt_flat[j]
                fyj = reps[yj]
                loss_ij = torch.log(
                    1. + torch.sum((reps_sorted_last-fyj).exp()*J_last_bool)
                )
                loss_i.append(loss_ij)

            loss_i = torch.cat(loss_i)
            loss_i = torch.sum(loss_i) / len(loss_i)
            loss.append(loss_i)

            start += n

        loss = torch.cat(loss)
        loss = torch.sum(loss) / len(loss)
        losses['rel_loss'] = loss
    else:
        losses['rel_loss'] = F.cross_entropy(
            result.rel_dists, result.rel_labels[:, -1]
        )
    loss = sum(losses.values())


    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()

    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    evaluator_con = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    assert conf.num_gpus == 1
    # assert conf.mode == 'predcls'
    for i, (pred_i, gt_idx) in enumerate(zip(result_preds, b.indexes)):
        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i = pred_i

        gt_entry = {
            'gt_classes': train.gt_classes[gt_idx].copy(),
            'gt_relations': train.relationships[gt_idx].copy(),
            'gt_boxes': train.gt_boxes[gt_idx].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and \
            np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i, # hack for now.
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
        evaluator_con[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )

    res = {x: y.data[0] for x, y in losses.items()}
    recalls = evaluator[conf.mode].result_dict[conf.mode + '_recall']
    recalls_con = evaluator_con[conf.mode].result_dict[conf.mode + '_recall']
    res.update({
        'recall20': np.mean(recalls[20]),
        'recall50': np.mean(recalls[50]),
        'recall100': np.mean(recalls[100]),
        'recall20_con': np.mean(recalls_con[20]),
        'recall50_con': np.mean(recalls_con[50]),
        'recall100_con': np.mean(recalls_con[100]),
    })

    res = pd.Series(res)
    return res


def val_epoch(detector, val, val_loader, val_f):
    print("=== Validating")
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    evaluator_con = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    n_val = len(val_loader)
    for val_b, batch in enumerate(val_loader):
        # print(val_b, n_val)
        val_batch(conf.num_gpus * val_b, detector, batch, val,
                  evaluator, evaluator_con)
    evaluator[conf.mode].print_stats()
    evaluator_con[conf.mode].print_stats()
    recalls = evaluator[conf.mode].result_dict[conf.mode + '_recall']
    recalls_con = evaluator_con[conf.mode].result_dict[conf.mode + '_recall']
    val_f.write('{},{},{},{},{},{}\n'.format(
        np.mean(recalls[20]),
        np.mean(recalls[50]),
        np.mean(recalls[100]),
        np.mean(recalls_con[20]),
        np.mean(recalls_con[50]),
        np.mean(recalls_con[100]),
    ))
    val_f.flush()
    return np.mean(recalls_con[100])


def val_batch(batch_num, detector, b, val, evaluator, evaluator_con):
    # Hack to remove `volatile`
    # Safe to do this here because it's in eval mode.
    b.imgs = Variable(b.imgs.data)
    b.gt_rels = Variable(b.gt_rels.data)
    b.gt_boxes = Variable(b.gt_boxes.data)
    b.gt_classes = Variable(b.gt_classes.data)

    _, det_res = detector[b]
    assert conf.num_gpus == 1
    # if conf.num_gpus == 1:
    #     det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry, pred_entry
        )
        evaluator_con[conf.mode].evaluate_scene_graph_entry(
            gt_entry, pred_entry
        )


def get_optim(detector, lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th.
    # It's a hack, but it helps stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() \
                 if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() \
                     if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    assert not (conf.adam and conf.rmsprop)

    if conf.adam:
        # optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
        optimizer = optim.Adam(params, lr=lr)
    elif conf.rmsprop:
        optimizer = optim.RMSprop(params, lr=lr)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.1,
        verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler


if __name__ == '__main__':
    main()
