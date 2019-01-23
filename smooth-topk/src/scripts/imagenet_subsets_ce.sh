#!/bin/bash

# lr_0=0.1
lr_0=1.0
mu=2.5e-4

cd $(dirname $0)/..

export CUDA_VISIBLE_DEVICES=3

# python main.py --dataset imagenet --loss ce \
#        --out-name ../xp/imagenet/im64k_ce_lr=${lr_0}_mu=$mu \
#        --parallel-gpu --train-size 64000 --lr_0 $lr_0 --mu=$mu --no-visdom;

# python main.py --dataset imagenet --loss ce \
#        --out-name ../xp/imagenet/im128k_ce_lr=${lr_0}_mu=$mu \
#        --parallel-gpu --train-size 128000 --lr_0 $lr_0 --mu=$mu --no-visdom;

# python main.py --dataset imagenet --loss ce \
#        --out-name ../xp/imagenet/im320k_ce_lr=${lr_0}_mu=$mu \
#        --parallel-gpu --train-size 320000 --lr_0 $lr_0 --mu=$mu --no-visdom;

# python main.py --dataset imagenet --loss ce \
#        --out-name ../xp/imagenet/im640k_ce_lr=${lr_0}_mu=$mu \
#        --parallel-gpu --train-size 640000 --lr_0 $lr_0 --mu=$mu \
#        --no-visdom --use_dali

# python main.py --dataset imagenet --loss ce \
#        --out-name ../xp/imagenet/imall_ce_lr=${lr_0}_mu=$mu \
#        --parallel-gpu --lr_0 $lr_0 --mu=$mu --no-visdom;
