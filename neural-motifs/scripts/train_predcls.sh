#!/bin/bash

cd $(dirname $0)/..

run_baseline() {
    SEED=$1
    # No manual seed is set internally, so just give the
    # output directory a different name.
    python3 models/train_rels.py -m predcls -model motifnet \
        -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 10 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
        -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar \
        -save_dir checkpoints/baseline_predcls.$SEED \
        -nepoch 30 -use_bias &> /dev/null &
}

run_lml() {
    TOPK=$1
    SEED=$2
    # No manual seed is set internally, so just give the
    # output directory a different name.
    python3 models/train_rels.py -m predcls -model motifnet \
            -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
            -p 10 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
            -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar \
            -save_dir checkpoints/lml_predcls.$TOPK.$SEED \
            -nepoch 30 -use_bias -lml_topk $TOPK &> /dev/null &
}

run_entr() {
    TOPK=$1
    SEED=$2
    # No manual seed is set internally, so just give the
    # output directory a different name.
    python3 models/train_rels.py -m predcls -model motifnet \
            -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
            -p 10 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
            -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar \
            -save_dir checkpoints/entr_predcls.$TOPK.$SEED \
            -nepoch 30 -use_bias -entr_topk $TOPK &> logs/entr.$TOPK.log &
}

run_ml() {
    SEED=$1
    # No manual seed is set internally, so just give the
    # output directory a different name.
    python3 models/train_rels.py -m predcls -model motifnet \
        -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 10 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
        -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar \
        -save_dir checkpoints/ml_predcls.$SEED \
        -nepoch 30 -use_bias -ml_loss &> logs/ml.log &
}

SEED=0

# export CUDA_VISIBLE_DEVICES=0
# run_baseline $SEED

# export CUDA_VISIBLE_DEVICES=1
# run_lml 20 $SEED

# export CUDA_VISIBLE_DEVICES=0
# run_lml 50 $SEED

# export CUDA_VISIBLE_DEVICES=1
# run_lml 100 $SEED

# export CUDA_VISIBLE_DEVICES=0
# run_entr 20 $SEED

# export CUDA_VISIBLE_DEVICES=1
# run_entr 50 $SEED

# export CUDA_VISIBLE_DEVICES=2
# run_entr 100 $SEED

export CUDA_VISIBLE_DEVICES=0
run_ml

wait
