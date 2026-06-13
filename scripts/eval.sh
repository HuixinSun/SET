#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/aitod/fcos_r50_set.py}
CHECKPOINT=${2:-checkpoints/aitod_fcos_set_epoch12.pth}
GPUS=${3:-1}

bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" --eval bbox
