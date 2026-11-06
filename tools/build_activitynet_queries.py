"""Rebuild ActivityNet query caches from the OFFICIAL Captions annotations.

The existing cache/activitynet_v12_frames/{train,val}/query was built from
data/activitynet/annotations/{train,val}.json, where val.json is val_1 -- not
the official evaluation split. ActivityNet Captions evaluates on val_2.

train.json is fine (byte-identical to the friedrichor HF release), so train is
rebuilt here only to keep the manifest in sync with whichever videos currently
have features. val_2 is built fresh.

Video features are keyed by vid and the val_2 video set is a subset of val_1's
(4,885 of 4,917), so no video re-encoding is required: the same per-video
features serve both annotation sets.

Outputs, per split:
    <out>/<split>/manifest.jsonl     one row per (vid, sentence)
    <out>/<split>/query/<qid>.npz    {"pooler_output": (2048,) float32}
    <out>/<split>/QUERY_CACHE_META.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .joint_encoder import JointEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="data/activitynet_captions/annotations")
    ap.add_argument("--video-cache", default="cache/activitynet_v12_frames")
    ap.add_argument("--out", default="cache/activitynet_v12_official")
    ap.add_argument("--model-dir", default="models/Qwen3-VL-Embedding-2B")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir)
    vc = Path(args.video_cache)
    out_root = Path(args.out)

    # video features may live under either cached split; collect them all
    have: set[str] = set()
    for s in ("train", "val"):
        d = vc / s / "video"
        if d.is_dir():
            have |= {p.stem for p in d.glob("*.npz")}
    print(json.dumps({"videos_with_features": len(have)}), flush=True)

    enc = None
    for split, ann_name in (("train", "train.json"), ("val_2", "val_2.json")):
        ann = json.loads((ann_dir / ann_name).read_text())
        rows = []
        for vid, d in ann.items():
            if vid not in have:
                continue
            for i, (q, w) in enumerate(zip(d.get("sentences", []), d.get("timestamps", []))):
                rows.append({"qid": f"anet_{ann_name[:-5]}_{vid}_{i:03d}",
                             "vid": vid, "query": str(q).strip(),
                             "duration": float(d.get("duration", 0.0)),
                             "relevant_windows": [list(map(float, w))]})
        root = out_root / split
        qdir = root / "query"
        qdir.mkdir(parents=True, exist_ok=True)
        with (root / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        todo = [r for r in rows if not (qdir / (r["qid"] + ".npz")).exists()]
        print(json.dumps({"split": split, "videos": len({r['vid'] for r in rows}),
                          "queries": len(rows), "todo": len(todo)}), flush=True)
        if todo:
            if enc is None:
                enc = JointEncoder(args.model_dir, device="cuda")
                print("ENCODER_READY", flush=True)
            t0 = time.time()
            done = 0
            for j in range(0, len(todo), args.batch):
                chunk = todo[j:j + args.batch]
                z = enc.encode_text([r["query"] for r in chunk],
                                    batch=args.batch).float().cpu().numpy()
                for k, r in enumerate(chunk):
                    np.savez_compressed(qdir / (r["qid"] + ".npz"),
                                        pooler_output=z[k].astype(np.float32))
                done += len(chunk)
                if done % 2048 < len(chunk):
                    print("progress %d/%d %.0f/s" % (done, len(todo),
                                                     done / max(time.time() - t0, 1e-9)),
                          flush=True)
        (root / "QUERY_CACHE_META.json").write_text(json.dumps({
            "dataset": "ActivityNet Captions", "annotation": ann_name,
            "annotation_dir": str(ann_dir), "split": split,
            "videos": len({r["vid"] for r in rows}), "queries": len(rows),
            "video_feature_source": str(vc),
            "note": "restricted to videos that currently have features"}, indent=2))
        print("SPLIT_DONE %s videos=%d queries=%d" % (
            split, len({r["vid"] for r in rows}), len(rows)), flush=True)


if __name__ == "__main__":
    main()
