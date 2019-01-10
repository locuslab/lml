#!/bin/bash

# lr_0=1
# tau_0=0.1
# mu=0.00025

# export CUDA_VISIBLE_DEVICES=0,1

./main.py --dataset imagenet --loss lml --out-name ../xp/imagenet/im64k_lml \
    --parallel-gpu --train-size 64000 --no-visdom

./main.py --dataset imagenet --loss lml --out-name ../xp/imagenet/im128k_lml \
    --parallel-gpu --train-size 128000 --no-visdom

./main.py --dataset imagenet --loss lml --out-name ../xp/imagenet/im320k_lml \
    --parallel-gpu --train-size 320000 --no-visdom

./main.py --dataset imagenet --loss lml --out-name ../xp/imagenet/im640k_lml \
    --parallel-gpu --train-size 640000 --no-visdom

./main.py --dataset imagenet --loss lml --out-name ../xp/imagenet/imall_lml \
    --parallel-gpu --no-visdom
