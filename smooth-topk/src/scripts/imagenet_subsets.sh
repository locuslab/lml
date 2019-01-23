#!/bin/bash

SRC_DIR=$(dirname $0)/..

function run_imagenet() {
    cd $SRC_DIR

    local GPUS=$1
    local LOSS=$2
    local SIZE_TAG=$3
    local lr_0=$4
    local tau=$5
    local mu=$6
    local seed=$7

    ARGS=""
    case "$SIZE_TAG" in
        "64k") ARGS+="--train-size 64000";;
        "128k") ARGS="--train-size 128000";;
        "320k") ARGS="--train-size 320000";;
        "640k") ARGS="--train-size 640000";;
        "all") ;;
        *) echo "Unrecognized size.";;
    esac

    export CUDA_VISIBLE_DEVICES=$GPUS
    setopt shwordsplit
    ./main.py --dataset imagenet --loss $LOSS \
              --out-name ../xp/imagenet/im${SIZE_TAG}_${LOSS}_lr=${lr_0}_mu=${mu}_tau=${tau} \
              --parallel-gpu $ARGS --no-visdom \
              --lr_0 $lr_0 --tau $tau --mu $mu --seed $seed \
              --use_dali
}
