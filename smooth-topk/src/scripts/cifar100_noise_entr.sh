#!/bin/bash

# echo "Using device" $device

mkdir -p logs

export CUDA_VISIBLE_DEVICES=0
seed=2
for p in 0.4 0.6; do
# for p in 0.0 0.2; do
    python3 main.py --dataset cifar100 --model densenet40-40 \
            --out-name ../xp/cifar100/cifar100_${p}_${seed}_entr \
            --loss entr --noise $p --seed $seed \
            --no-visdom --test-batch-size 64 &> /dev/null &
    export CUDA_VISIBLE_DEVICES=$(((CUDA_VISIBLE_DEVICES + 1) % 4))
done
