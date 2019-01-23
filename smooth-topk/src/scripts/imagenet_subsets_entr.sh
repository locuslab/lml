#!/bin/bash

lr_0=1
tau=1.0
mu=0.00025

cd $(dirname $0)/..

source ~/imagenet-fast.sh
export CUDA_VISIBLE_DEVICES=3

# ./main.py --dataset imagenet --loss entr \
#           --out-name ../xp/imagenet/im64k_entr_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --train-size 64000 --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali

# ./main.py --dataset imagenet --loss entr \
#           --out-name ../xp/imagenet/im128k_entr_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --train-size 128000 --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali

# ./main.py --dataset imagenet --loss entr \
#           --out-name ../xp/imagenet/im320k_entr_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --train-size 320000 --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali

# ./main.py --dataset imagenet --loss entr \
#           --out-name ../xp/imagenet/im640k_entr_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --train-size 640000 --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali

# ./main.py --dataset imagenet --loss entr \
#           --out-name ../xp/imagenet/imall_entr_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali
