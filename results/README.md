# E2D-VTG run archive

This directory archives the raw training outputs behind the paper
**"E2D-VTG: Embedding-to-Detection for Query-Conditioned Temporal Grounding"**
(CVPR 2027 submission). Every number in the paper can be traced to a
`best.json` / `metrics.jsonl` in one of the run directories below.

## Layout

```text
runs/
  qv/main_s2021/            QVHighlights joint, seed 2021 (200/200 epochs)
  qv/main_s2023/            QVHighlights joint, seed 2023 (200/200 epochs)
  anet/ablation_norefine_s2022/      ActivityNet, seed 2022, no refinement
  anet/ablation_nooffset_s2022/      ActivityNet, seed 2022, no boundary offset
  anet/ablation_notef_s2022/         ActivityNet, seed 2022, no TEF
  anet/ablation_nomultiscale_s2022/  ActivityNet, seed 2022, no multiscale
  anet/clip_arm_s2022/      ActivityNet CLIP arm (query-agnostic), seed 2022
logs/                       Training/evaluation logs of the archived runs
environment/                Exact pip freeze and launch scripts
```

Each run directory contains:

- `config.json`  full training arguments (data paths, dims, seed, ablation flags)
- `metrics.jsonl`  per-epoch R1@0.3 / R1@0.5 / R1@0.7 / mIoU on the official split
- `best.json`  best-epoch metrics (the row reported in the paper)
- `model_best.ckpt` / `last.ckpt`  checkpoints (published as a release asset,
  not inside git)

## Mapping to the paper

| Paper item | Dataset | Run directory |
|---|---|---|
| Table 1 (3-seed mean and best seed) | QVHighlights | `qv/main_s2021`, `qv/main_s2023` (+ seed 2022, see note) |
| Table 2 w/o refinement | ActivityNet | `anet/ablation_norefine_s2022` |
| Table 2 w/o boundary offset | ActivityNet | `anet/ablation_nooffset_s2022` |
| Table 2 w/o TEF | ActivityNet | `anet/ablation_notef_s2022` |
| Table 2 w/o multiscale | ActivityNet | `anet/ablation_nomultiscale_s2022` |
| Table 3 CLIP arm | ActivityNet | `anet/clip_arm_s2022` |

Archived best-epoch R1@0.5 values (percent):

| Run | R1@0.3 | R1@0.5 | R1@0.7 | mIoU | best epoch |
|---|---|---|---|---|---|
| qv/main_s2021 | 84.52 | 74.00 | 56.65 | 66.30 | 50 |
| qv/main_s2023 | 86.58 | 73.55 | 55.23 | 65.74 | 8 |
| anet/ablation_norefine_s2022 | 62.00 | 44.37 | 25.29 | 43.74 | 136 |
| anet/ablation_nooffset_s2022 | 64.83 | 46.52 | 27.24 | 45.74 | 6 |
| anet/ablation_notef_s2022 | 67.67 | 46.22 | 25.29 | 46.46 | 7 |
| anet/ablation_nomultiscale_s2022 | 67.12 | 47.32 | 27.01 | 46.83 | 6 |
| anet/clip_arm_s2022 | 61.07 | 42.64 | 23.68 | 42.72 | 6 |

Note: QVHighlights seed 2022 (`74.13 @ ep66`), the Charades-STA runs
(seeds 2021/2022/2023), the ActivityNet joint main runs (3 seeds), and the
Charades/QV ablation and video-only arm runs were executed on a second
instance whose results will be appended here once it is back online. The
published paper numbers are computed from all seeds as stated in the
manuscript.
