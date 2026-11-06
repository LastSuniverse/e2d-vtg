"""Fill an ActivityNet Captions video-feature cache from 1-fps frame directories.

Each frame directory must contain the expected JPEG keyframes plus ``meta.json``
with duration and source provenance. The encoder writes one resumable NPZ per
video with ``(T, 2048)`` float16 features, matching the detector feature-cache
contract. The 448-pixel JPEG protocol is unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .joint_encoder import JointEncoder

DEFAULT_ROOTS = [
    "frames/activitynet_captions_aligned_keyframes_v1",
    "frames/activitynet_captions_base_aligned_keyframes_v1",
    "frames/activitynet_captions_val_priority_base_v1",
    "frames/activitynet_captions_val_priority_existing_v1",
    "frames/activitynet_captions_1fps",
]


def split_map(ann_dir: Path) -> dict[str, str]:
    """Fill an ActivityNet Captions video-feature cache from 1-fps frame directories.

Each frame directory must contain the expected JPEG keyframes plus ``meta.json``
with duration and source provenance. The encoder writes one resumable NPZ per
video with ``(T, 2048)`` float16 features, matching the detector feature-cache
contract. The 448-pixel JPEG protocol is unchanged.
"""
    m: dict[str, str] = {}
    for f, name in (("train.json", "train"), ("val_2.json", "val"), ("val_1.json", "val")):
        p = ann_dir / f
        if not p.exists():
            continue
        for vid in json.loads(p.read_text()):
            m.setdefault(vid, name)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--ann-dir", default="data/activitynet_captions/annotations")
    ap.add_argument("--out-cache", default="cache/activitynet_v12_frames")
    ap.add_argument("--model-dir", default="models/Qwen3-VL-Embedding-2B")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=600)
    args = ap.parse_args()

    torch.set_num_threads(4)
    vmap = split_map(Path(args.ann_dir))
    out_root = Path(args.out_cache)

    # vid -> frame dir (first root that has it wins; all roots use one recipe)
    frames: dict[str, Path] = {}
    for r in args.roots:
        base = Path(r)
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_dir() and (p / "meta.json").exists():
                frames.setdefault(p.name, p)
    print(json.dumps({"frame_videos": len(frames),
                      "in_official": sum(1 for v in frames if v in vmap)}), flush=True)

    todo = []
    for vid, d in sorted(frames.items()):
        split = vmap.get(vid)
        if split is None:
            continue
        if (out_root / split / "video" / f"{vid}.npz").exists():
            continue
        todo.append((vid, split, d))
    if args.limit:
        todo = todo[:args.limit]
    by_split = {s: sum(1 for _, x, _ in todo if x == s) for s in ("train", "val")}
    print(json.dumps({"to_encode": len(todo), "by_split": by_split}), flush=True)

    enc = JointEncoder(args.model_dir, device="cuda")
    print("ENCODER_READY", flush=True)

    t0 = time.time()
    done = bad = 0
    frames_done = 0
    for vid, split, d in todo:
        try:
            meta = json.loads((d / "meta.json").read_text())
            names = sorted(p for p in d.iterdir() if p.suffix == ".jpg")
            if not names:
                bad += 1
                continue
            names = names[:args.max_frames]
            imgs = [Image.open(p).convert("RGB") for p in names]
            z = enc.encode_image(imgs, batch=args.batch).float().cpu().numpy()
            z = z[: len(names)].astype(np.float16)
            out = out_root / split / "video"
            out.mkdir(parents=True, exist_ok=True)
            dst = out / f"{vid}.npz"
            tmp = dst.with_suffix(".partial.npz")
            np.savez_compressed(
                tmp, features=z,
                duration=np.float32(meta.get("duration", len(names))),
                fps=np.float32(meta.get("fps", 1.0)),
                frame_count=np.int32(len(names)),
                source_rel=str(meta.get("source", "")))
            os.replace(tmp, dst)
            done += 1
            frames_done += len(names)
        except Exception as e:
            print("BAD %s: %s: %s" % (vid, type(e).__name__, e), flush=True)
            bad += 1
            continue
        if done % 25 == 0:
            el = time.time() - t0
            print("progress vids=%d/%d frames=%d bad=%d %.0fs (%.2f vid/s, %.0f img/s)"
                  % (done, len(todo), frames_done, bad, el,
                     done / max(el, 1e-9), frames_done / max(el, 1e-9)), flush=True)

    print("ANET_ENCODE_DONE vids=%d frames=%d bad=%d elapsed=%.0fs"
          % (done, frames_done, bad, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
