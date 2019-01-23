#!/bin/bash

lr_0=1
tau=1.
mu=0.00025

cd $(dirname $0)/..

# source ~/imagenet-fast.sh
# source ~/.private
export CUDA_VISIBLE_DEVICES=2

# ./main.py --dataset imagenet --loss lml \
#           --out-name ../xp/imagenet/im64k_lml_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --train-size 64000 --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali

# ./main.py --dataset imagenet --loss lml \
#           --out-name ../xp/imagenet/im128k_lml_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --train-size 128000 --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali

# ./main.py --dataset imagenet --loss lml \
#           --out-name ../xp/imagenet/im320k_lml_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --train-size 320000 --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali

# ./main.py --dataset imagenet --loss lml \
#           --out-name ../xp/imagenet/im640k_lml_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --train-size 640000 --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali

# ./main.py --dataset imagenet --loss lml \
#           --out-name ../xp/imagenet/imall_lml_lr=${lr_0}_mu=${mu}_tau=${tau} \
#           --parallel-gpu --no-visdom \
#           --lr_0 $lr_0 --tau $tau --mu $mu --use_dali
