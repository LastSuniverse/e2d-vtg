#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/qwen_trac
T=/root/autodl-tmp/timelens
PY=/root/miniconda3/bin/python
export PYTHONPATH="$ROOT/code"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

while [[ ! -e "$T/FEATURES_READY" || ( ! -e "$T/MOD6.DONE" && ! -e "$T/MOD6.FAILED" ) ]]; do
  echo "waiting $(date -Is) features=$([[ -e "$T/FEATURES_READY" ]] && echo ready || echo wait) mod6=$([[ -e "$T/MOD6.DONE" ]] && echo done || ([[ -e "$T/MOD6.FAILED" ]] && echo failed || echo wait))"
  sleep 120
done

OUT="$ROOT/results_boundary_mamba/qwen"
mkdir -p "$OUT"
"$PY" -m boundary_mamba.train \
  --train-jsonl "$ROOT/data/uvcom_jsonl/charades_sta/charades_sta_train_tvr_format.jsonl" \
  --eval-jsonl "$ROOT/data/uvcom_jsonl/charades_sta/charades_sta_test_tvr_format.jsonl" \
  --train-video-cache "$ROOT/cache/uvcom_joint/charades_sta/train/video" \
  --eval-video-cache "$ROOT/cache/uvcom_joint/charades_sta/test/video" \
  --train-query-cache "$ROOT/cache/uvcom/charades_sta/train/query" \
  --eval-query-cache "$ROOT/cache/uvcom/charades_sta/test/query" \
  --video-key qid --video-dim 2048 --text-dim 2048 \
  --model-dim 256 --layers 3 --state-dim 16 --batch-size 16 --epochs 50 \
  --workers 4 --lr 2e-4 --seed 2018 --amp --output-dir "$OUT"

CKPT="$OUT/model_best.ckpt"
"$PY" -m boundary_mamba.eval --checkpoint "$CKPT" \
  --jsonl "$ROOT/data/uvcom_jsonl/charades_sta/charades_sta_test_tvr_format.jsonl" \
  --video-cache "$ROOT/cache/uvcom_joint/charades_sta/test/video" \
  --query-cache "$ROOT/cache/uvcom/charades_sta/test/query" \
  --video-key qid --video-dim 2048 --text-dim 2048 --model-dim 256 --layers 3 \
  --state-dim 16 --batch-size 16 --workers 4 --output "$OUT/official_test.json"

"$PY" -m boundary_mamba.eval --checkpoint "$CKPT" \
  --jsonl "$ROOT/data/uvcom_jsonl/charades_timelens/charades_timelens_test_tvr_format.jsonl" \
  --video-cache "$ROOT/cache/uvcom_joint/charades_timelens/test/video" \
  --query-cache "$ROOT/cache/uvcom/charades_timelens/test/query" \
  --video-key qid --video-dim 2048 --text-dim 2048 --model-dim 256 --layers 3 \
  --state-dim 16 --batch-size 16 --workers 4 --output "$OUT/timelens_test.json"
