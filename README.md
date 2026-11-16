# E2D-VTG

Query-conditioned temporal grounding with an embedding-to-detection interface.
The project treats video temporal grounding as 1-D interval detection: a frozen
Qwen3-VL-Embedding encoder produces query-aware video vectors, and a compact
detector scores complete start-end pairs, refines the best proposals, and
predicts interval quality.

## Highlights

- Dense start-end relation field over valid one-second temporal cells.
- Query-conditioned multiscale temporal encoding with gated context branches.
- Top-K full-interval candidates, local boundary offsets, and IoU quality.
- Matched training and evaluation recipes for Charades-STA, QVHighlights, and
  ActivityNet-Captions.
- Official-split assertions and standardized top-1 `R1@0.3`, `R1@0.5`,
  `R1@0.7`, and `mIoU` reporting.

## Results archive

Training outputs behind the paper (per-epoch metrics, configs, logs,
and the reproduction environment) are archived under [results/](results) — see
[results/README.md](results/README.md). Trained checkpoints are attached to the
[v1.0.0 release](https://github.com/LastSuniverse/e2d-vtg/releases/tag/v1.0.0).

## Repository layout

```text
e2d_vtg/
  detector.py                 Dense retrieval and refinement detector
  data.py                     JSONL + NPZ feature-cache dataset and collate
  losses.py                   Additional loss definitions
  metrics.py                  Official top-1 interval metrics
  train_charades.py           Single-window official Charades training
  train_qvhighlights.py       Multi-window max-IoU QVHighlights training
  train_activitynet.py        Length-bucketed full ActivityNet training
  train_mixed.py              Optional multi-domain training
  eval_checkpoint.py          Standard checkpoint evaluator
tools/
  joint_encoder.py            Qwen3-VL-Embedding wrapper
  extract_*_keyframes.py      Resumable 1-fps keyframe extraction
  encode_*                    Feature-cache construction
  build_activitynet_queries.py Official ActivityNet query cache
tests/                        Model-level smoke checks
```

## Installation

Use Python 3.10 or newer. For detector-only use:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Feature extraction additionally requires a local `ffmpeg`/`ffprobe`
installation and the complete Qwen3-VL-Embedding model directory. The wrapper
loads the model-specific helper from `models/Qwen3-VL-Embedding-2B/scripts/`.

## Feature-cache format

Each training record is a JSONL object:

```json
{
  "qid": 100,
  "vid": "video-id",
  "duration": 31.2,
  "relevant_windows": [[12.0, 28.0]]
}
```

A video feature file is `video_dir/<qid>.npz` or `<vid>.npz`, depending on the
requested `--video-key`, and stores:

```text
features       float array, shape (T, D)
duration       scalar float
fps            scalar float
frame_count    scalar integer
```

A query feature file is `query_dir/<qid>.npz` and stores a pooled vector under
`pooler_output`. `data.py` appends two temporal-extent features unless
`--append-tef false` is requested, so a 2048-D embedding becomes a 2050-D
detector input.

## Keyframe protocol

Both keyframe extractors decode at 2 fps, resize width to 448 while preserving
aspect ratio, and keep the first frame in each integer second. They write a
metadata file, a SHA-256 source digest, and a `_DONE` marker only after the
frame count passes a duration sanity check. The process is resumable.

Example:

```bash
python -m tools.extract_charades_keyframes \
  --raw-root data/charades/videos \
  --out-root frames/charades_sta

python -m tools.extract_activitynet_keyframes \
  --raw-root data/activitynet/videos \
  --out-root frames/activitynet_captions \
  --ann-dir data/activitynet_captions/annotations \
  --workers 8
```

Qwen feature extraction is a GPU job:

```bash
python -m tools.encode_video_only \
  --jsonl data/qv/train.jsonl data/qv/val.jsonl \
  --split-names train val \
  --frames-roots frames/qv \
  --out-cache cache/qv_video_only \
  --model-dir models/Qwen3-VL-Embedding-2B
```

## Training

Commands are explicit about cache locations and expected official split sizes.
Do not silently train on partial caches.

```bash
python -m e2d_vtg.train_charades \
  --train-jsonl data/charades/train.jsonl \
  --eval-jsonl data/charades/test.jsonl \
  --train-video-cache cache/charades/train/video \
  --eval-video-cache cache/charades/test/video \
  --train-query-cache cache/charades/train/query \
  --eval-query-cache cache/charades/test/query \
  --output-dir results/charades_seed2021 \
  --video-dim 2050 --text-dim 2048 \
  --epochs 200 --seed 2021 --amp

python -m e2d_vtg.train_qvhighlights \
  --train-jsonl data/qvhighlights/train.jsonl \
  --eval-jsonl data/qvhighlights/val.jsonl \
  --train-video-cache cache/qvhighlights/train/video \
  --eval-video-cache cache/qvhighlights/val/video \
  --train-query-cache cache/qvhighlights/train/query \
  --eval-query-cache cache/qvhighlights/val/query \
  --output-dir results/qv_seed2021 \
  --expected-train 7202 --expected-eval 1550 \
  --video-dim 2050 --text-dim 2048 \
  --epochs 200 --seed 2021 --amp

python -m e2d_vtg.train_activitynet \
  --train-jsonl data/activitynet/train.jsonl \
  --eval-jsonl data/activitynet/val_2.jsonl \
  --train-video-cache cache/activitynet/train/video \
  --eval-video-cache cache/activitynet/val/video \
  --train-query-cache cache/activitynet/train/query \
  --eval-query-cache cache/activitynet/val/query \
  --output-dir results/activitynet_seed2021 \
  --expected-train 37421 --expected-eval 17031 \
  --max-video-len 760 --max-len 760 \
  --video-dim 2050 --text-dim 2048 \
  --epochs 200 --seed 2021 --amp
```

For reproducibility, the paper's reported seeds are 2021, 2022, and 2023.
QVHighlights uses max-IoU matching when an annotation contains multiple windows.
ActivityNet uses the complete official `train.json` and `val_2.json`.

## Evaluation

```bash
python -m e2d_vtg.eval_checkpoint \
  --checkpoint results/charades_seed2021/model_best.ckpt \
  --jsonl data/charades/test.jsonl \
  --video-cache cache/charades/test/video \
  --query-cache cache/charades/test/query \
  --output results/charades_seed2021/eval.json \
  --expected-records 3720
```

The evaluator refuses to score incomplete expected test sets. It writes top-1
predictions and the four standardized metrics.

## Results and checkpoints

The repository does not redistribute benchmark videos or checkpoints. Results
reported in the paper use official annotations, no private re-split, and fixed
seeds; feature caches and model checkpoints are excluded because of size.

## Tests

Run the lightweight smoke checks after changing the detector or training code:

```bash
pytest -q
```

## License

This repository is released under the MIT License. Dataset annotations and
videos remain governed by their respective benchmark licenses.

## Citation

Citation information will be added after public release of the paper.
