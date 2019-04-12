#!/bin/bash

# echo "Using device" $device

mkdir -p logs

# for p in 0.0 0.2 0.4 0.6 0.8 1.0; do
# for p in 0.2 0.4 0.6 0.8 1.0; do
for p in 0.2 0.4; do
    for seed in 0 3; do
        export CUDA_VISIBLE_DEVICES=$seed
        python3 main.py --dataset cifar100 --model densenet40-40 \
                --out-name ../xp/cifar100/cifar100_${p}_${seed}_ml \
                --loss ml --noise $p --seed $seed \
                --no-visdom --test-batch-size 64 &> /dev/null &
    done
    wait
done
