#!/bin/bash

run_baseline() {
    SEED=$1
    python3 models/train_rels.py -m predcls -model motifnet \
        -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 10 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
        -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar \
        -save_dir checkpoints/baseline_predcls.$SEED \
        -nepoch 50 -use_bias &> /dev/null &
}

run_lml() {
    TOPK=$1
    SEED=$2
    python3 models/train_rels.py -m predcls -model motifnet \
        -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 10 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 \
        -ngpu 1 -ckpt checkpoints/vg-faster-rcnn.tar \
        -save_dir checkpoints/lml_predcls.$TOPK.$SEED \
        -nepoch 50 -use_bias -lml_topk $TOPK &> /dev/null &
}

SEED=2

# export CUDA_VISIBLE_DEVICES=1
# run_baseline $SEED

# export CUDA_VISIBLE_DEVICES=0
# run_lml 20 $SEED

export CUDA_VISIBLE_DEVICES=2
run_lml 50 $SEED

export CUDA_VISIBLE_DEVICES=3
run_lml 100 $SEED
