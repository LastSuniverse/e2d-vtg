#!/bin/bash
set -u
BASE=/root/autodl-tmp/qwen_trac
CODE=$BASE/code
TRAIN=$BASE/qv_pipeline_20261021/official_train_available.jsonl
EVAL=$BASE/data/uvcom_jsonl/qvhighlights/highlight_val_release.jsonl
TV=$BASE/cache/uvcom_joint/qvhighlights/train/video
EV=$BASE/cache/uvcom_joint/qvhighlights/val/video
TQ=$BASE/cache/uvcom/qvhighlights/train/query
EQ=$BASE/cache/uvcom/qvhighlights/val/query

run_one(){
  name="$1"; shift
  out="$BASE/$name"; mkdir -p "$out"
  cd "$CODE"
  nohup /root/miniconda3/bin/python -u -m boundary_mamba.train_qv_retrieval \
    --train-jsonl "$TRAIN" --eval-jsonl "$EVAL" \
    --train-video-cache "$TV" --eval-video-cache "$EV" \
    --train-query-cache "$TQ" --eval-query-cache "$EQ" \
    --output-dir "$out" --epochs 200 --batch-size 32 --workers 4 \
    --expected-train 7202 --expected-eval 1550 \
    --video-dim 2050 --text-dim 2048 \
    --proposal-k 16 --train-topk 48 --jitter 12 --lr 0.0002 --amp "$@" \
    > "$out/train.log" 2>&1 &
  echo "$!" > "$out/train.pid"
  echo "START $name pid=$(cat $out/train.pid)"
}

run_one results_qv_qvretrieval_seed2021_20261101 --seed 2021
run_one results_qv_qvretrieval_seed2023_20261101 --seed 2023
