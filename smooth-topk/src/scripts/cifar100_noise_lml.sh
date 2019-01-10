#!/bin/bash

# echo "Using device" $device

mkdir -p logs

for seed in 0 1 2 3; do
    # for p in 0.0 0.2 0.4 0.6 0.8 1.0; do
    export CUDA_VISIBLE_DEVICES=$seed
    for p in 0.4; do
        python3 main.py --dataset cifar100 --model densenet40-40 \
                --out-name ../xp/cifar100/cifar100_${p}_${seed}_lml_v2 \
                --loss lml --noise $p --seed $seed \
                --no-visdom --test-batch-size 64 &> /dev/null &
    done
done
