# Reproduction environment

All archived runs were trained in the environment described below.

## Hardware

- Single NVIDIA GPU (cloud instance), CUDA 12.8, driver managed by the host.
- Mixed precision (AMP) enabled for QVHighlights runs; ActivityNet runs used
  full precision (see `amp` in each run's `config.json`).

## Software

- OS: Linux x86_64 (Ubuntu-based container)
- Python 3.12 (Miniconda, `base` environment at `/root/miniconda3`)
- PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128
- transformers 5.20.1, tokenizers 0.23.3, accelerate 1.15.0,
  huggingface_hub 2.4.0
- numpy 2.3.2, pillow 11.3.0, av 19.0.2, tqdm, requests

The exact full package freeze of the training environment is in
[`environment/pip_freeze_20261116.txt`](environment/pip_freeze_20261116.txt).

## Representations

- Semantic encoder: frozen Qwen3-VL-Embedding-2B (instruct variant for
  query-conditioned arms). Frame features are 2048-d; the detector input
  appends a 2-d temporal embedding-of-frame (TEF), giving `video_dim = 2050`.
  The CLIP arm uses 512-d visual features projected to `video_dim = 514`
  (512 + 2 TEF), `text_dim = 2048` in all runs (CLIP text encoder output).
- Detector: dim 192, rank 64, 2.36M parameters; 200 epochs, batch size 32,
  lr 2e-4, seed as recorded per run.

## Launch commands

QVHighlights joint seeds (archived `results_qv_qvretrieval_seed*_20261101`):

```bash
python -u -m e2d_vtg.train_qvhighlights \
  --train-jsonl official_train_available.jsonl \
  --eval-jsonl highlight_val_release.jsonl \
  --train-video-cache cache/uvcom_joint/qvhighlights/train/video \
  --eval-video-cache cache/uvcom_joint/qvhighlights/val/video \
  --train-query-cache cache/uvcom/qvhighlights/train/query \
  --eval-query-cache cache/uvcom/qvhighlights/val/query \
  --output-dir results/qv/main_s2021 \
  --epochs 200 --batch-size 32 --workers 4 \
  --expected-train 7202 --expected-eval 1550 \
  --video-dim 2050 --text-dim 2048 \
  --proposal-k 16 --train-topk 48 --jitter 12 --lr 0.0002 --amp --seed 2021
```

(the original launcher script is kept at
[`environment/qv_seeds_qvretrieval.sh`](environment/qv_seeds_qvretrieval.sh))

ActivityNet ablations (archived `results_anet_ablation_*_s2022`): same entry
point family with `train_anet` settings; each run's `config.json` is the
authoritative argument list, e.g. `no_refinement: true` for the
"w/o refinement" row. The corresponding public entry point is
`e2d_vtg.train_activitynet`.

Evaluation: the per-epoch metrics in `metrics.jsonl` are computed on the
official validation split with the standardized top-1 interval protocol
(R1@0.3/0.5/0.7 and mIoU) implemented in `e2d_vtg.metrics`.
