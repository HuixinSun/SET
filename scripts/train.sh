#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/aitod/fcos_r50_set.py}
GPUS=${2:-1}
WORK_DIR=${3:-output/fcos_r50_set}

bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR"
