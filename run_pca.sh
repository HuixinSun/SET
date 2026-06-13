#!/usr/bin/env bash
set -e

BASE_CFG=${1:-configs/aitod/fcos_r50_baseline.py}
BASE_CKPT=${2:-checkpoints/aitod_fcos_r50_baseline_epoch12.pth}
SET_CFG=${3:-configs/aitod/fcos_r50_set.py}
SET_CKPT=${4:-checkpoints/aitod_fcos_set_epoch12.pth}
INDEX=${5:-0}
LAYER=${6:-0}
OUT_DIR=${7:-vis/pca}

python tools/vis/plot_pca.py \
  "$BASE_CFG" "$BASE_CKPT" \
  "$SET_CFG" "$SET_CKPT" \
  --index "$INDEX" --layer "$LAYER" --out-dir "$OUT_DIR"
