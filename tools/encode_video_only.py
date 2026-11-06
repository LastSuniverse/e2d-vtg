"""Build a video-only Qwen feature cache for CH/QV, matching the Anet VO cache.

The joint-vs-video-only controlled comparison exists only on ActivityNet; CH and
QV were encoded query-conditioned from the start. This closes that gap: it
encodes each clip's frames ONCE with the image-only path (JointEncoder.
encode_image -- no text in context) and writes one npz per video, byte-compatible
with cache/activitynet_v12_frames:

    features (T, 2048) float16, duration f32, fps f32, frame_count i32, source_rel str

Driven by the official jsonl manifests (vid -> split), so coverage can be
verified against the exact training/eval sets. Resumable: existing npz skipped.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True,
                    help="manifests, same order as --split-names")
    ap.add_argument("--split-names", nargs="+", required=True)
    ap.add_argument("--frames-roots", nargs="+", required=True)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--model-dir", default="models/Qwen3-VL-Embedding-2B")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=300)
    a = ap.parse_args()

    torch.set_num_threads(4)
    assert len(a.jsonl) == len(a.split_names)
    vmap: dict[str, str] = {}
    nq = 0
    for jl, split in zip(a.jsonl, a.split_names):
        for line in Path(jl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            vmap.setdefault(str(row["vid"]), split)
            nq += 1

    frames: dict[str, Path] = {}
    for r in a.frames_roots:
        base = Path(r)
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_dir() and any(p.glob("*.jpg")):
                frames.setdefault(p.name, p)

    out_root = Path(a.out_cache)
    todo = []
    for vid, split in sorted(vmap.items()):
        if (out_root / split / "video" / f"{vid}.npz").exists():
            continue
        d = frames.get(vid)
        if d is None:
            continue
        todo.append((vid, split, d))
    have = {v: (out_root / s / "video" / f"{v}.npz").exists() for v, s in vmap.items()}
    print(json.dumps({"queries": nq, "unique_vids": len(vmap),
                      "frame_dirs": len(frames),
                      "with_frames": sum(1 for v in vmap if v in frames),
                      "already_done": sum(1 for ok in have.values() if ok),
                      "to_encode": len(todo),
                      "by_split": {s: sum(1 for _, x, _ in todo if x == s)
                                   for s in set(a.split_names)}}), flush=True)
    if not todo:
        print("VIDEO_ONLY_DONE nothing to do", flush=True)
        return
    if a.limit:
        todo = todo[: a.limit]

    enc = JointEncoder(a.model_dir, device="cuda")
    print("ENCODER_READY", flush=True)

    t0 = time.time()
    done = bad = 0
    for vid, split, d in todo:
        try:
            meta = {}
            mp = d / "meta.json"
            if mp.exists():
                meta = json.loads(mp.read_text())
            names = sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")
            if not names:
                bad += 1
                continue
            names = names[: a.max_frames]
            imgs = [Image.open(p).convert("RGB") for p in names]
            z = enc.encode_image(imgs, batch=a.batch).float().cpu().numpy()
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
        except Exception as e:
            print("BAD %s: %s: %s" % (vid, type(e).__name__, e), flush=True)
            bad += 1
        if done % 25 == 0:
            el = time.time() - t0
            print("progress %d/%d %.2f vid/s eta %.1f h"
                  % (done, len(todo), done / el, (len(todo) - done) / (done / el) / 3600),
                  flush=True)
    print("VIDEO_ONLY_DONE done=%d bad=%d seconds=%.0f" % (done, bad, time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
