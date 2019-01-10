#!/bin/bash

cd $(dirname $0)/..

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

for NCLS in 1000 10000; do
    for K in 5 50 100; do
        ./scripts/perf.py --n_classes $NCLS --k $K --n_trials 50
        ./scripts/perf.py --n_classes $NCLS --k $K --n_trials 50 --no-cuda
    done
done
