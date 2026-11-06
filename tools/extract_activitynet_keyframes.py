"""Extract resumable 1-fps keyframes for ActivityNet Captions videos.

The decoder runs at 2 fps, resizes width to 448 with aspect ratio preserved,
and keeps the first decoded frame in each integer second. A video directory is
published only after its frame count passes a duration sanity check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from pathlib import Path

SAMPLING_RECIPE = "qwen_joint_decode2fps_first_per_second_width448_jpegq3_v1"
STATE: dict = {}


def probe_duration(src: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(src)],
        capture_output=True, text=True, timeout=120)
    return float(out.stdout.strip())


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def work(item):
    vid, split, src, out_root = item
    src = Path(src)
    dest = Path(out_root) / split / vid
    if (dest / "_DONE").exists():
        return "skip"
    tmp = dest.with_name(dest.name + f".partial.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        dur = probe_duration(src)
        if not (dur > 0):
            raise ValueError("bad duration %r" % dur)
        digest = sha256_of(src)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-threads", "1", "-filter_threads", "1", "-i", str(src),
             "-map", "0:v:0", "-an", "-vf", "fps=2,scale=448:-2",
             "-threads", "1", "-q:v", "3", str(tmp / "f%06d.jpg")],
            check=True, timeout=1800)
        decoded = sorted(tmp.glob("f*.jpg"))
        for i, frame in enumerate(decoded):
            if i % 2:
                frame.unlink()
        kept = decoded[::2]
        for i, frame in enumerate(kept, 1):
            target = tmp / f"f{i:06d}.jpg"
            if frame != target:
                os.replace(frame, target)
        count = len(kept)
        if count == 0:
            raise ValueError("no frames decoded")
        expected = int(dur)
        if abs(count - expected) > max(3, 0.05 * expected):
            raise ValueError("frame count %d vs duration %.2f" % (count, dur))
        meta = dict(video_id=vid, split=split, path=str(dest), source=str(src),
                    source_size=src.stat().st_size, source_sha256=digest,
                    fps=1.0, width=448, jpeg_quality=3, num_frames=count,
                    duration=dur, decode_fps=2.0, frames_per_second=1,
                    sampling_recipe=SAMPLING_RECIPE, pixels_verified=True,
                    origin="captions_v13_raw_local",
                    timestamp_convention="n_over_2_grouped_by_integer_second_first",
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
        (tmp / "_DONE").touch()
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dest)
        return "ok"
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return "bad:%s:%s" % (vid, e)


def init(raw_root, out_root, splits):
    STATE["raw"] = raw_root
    STATE["out"] = out_root
    STATE["splits"] = splits


def work_star(item):
    return work(item)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/activitynet/Activity_Videos/v1-3")
    ap.add_argument("--out-root", default="frames/activitynet_captions_remaining_v1")
    ap.add_argument("--ann-dir", default="data/activitynet_captions/annotations")
    ap.add_argument("--skip-roots", nargs="*", default=[
        str(ROOT / "frames/activitynet_captions_aligned_keyframes_v1"),
        str(ROOT / "frames/activitynet_captions_base_aligned_keyframes_v1"),
        str(ROOT / "frames/activitynet_captions_val_priority_base_v1"),
        str(ROOT / "frames/activitynet_captions_val_priority_existing_v1"),
        str(ROOT / "frames/activitynet_captions_1fps"),
    ])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    splits: dict[str, str] = {}
    for f, name in (("train.json", "train"), ("val_2.json", "val"), ("val_1.json", "val")):
        p = Path(args.ann_dir) / f
        if p.exists():
            for vid in json.loads(p.read_text()):
                splits.setdefault(vid, name)

    done: set[str] = set()
    for r in args.skip_roots:
        base = Path(r)
        if base.is_dir():
            for m in base.rglob("meta.json"):
                done.add(m.parent.name)
    base = Path(args.out_root)
    if base.is_dir():
        for m in base.rglob("_DONE"):
            done.add(m.parent.name)

    raw: dict[str, Path] = {}
    for p in Path(args.raw_root).rglob("*"):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm"):
            raw.setdefault(p.stem, p)

    todo = [(vid, splits[vid], str(raw[vid]), args.out_root)
            for vid in sorted(raw) if vid in splits and vid not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(json.dumps({"already_framed": len(done), "raw": len(raw),
                      "todo": len(todo),
                      "by_split": {s: sum(1 for t in todo if t[1] == s)
                                   for s in ("train", "val")}}), flush=True)

    t0 = time.time()
    ok = bad = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work_star, todo), 1):
            if r == "ok":
                ok += 1
            elif r == "skip":
                pass
            else:
                bad += 1
                if bad <= 30:
                    print(r, flush=True)
            if i % 50 == 0:
                el = time.time() - t0
                print("progress %d/%d ok=%d bad=%d %.0fs (%.2f vid/s, eta %.0f min)"
                      % (i, len(todo), ok, bad, el, i / max(el, 1e-9),
                         (len(todo) - i) / max(i / max(el, 1e-9), 1e-9) / 60), flush=True)
    print("ANET_FRAMES_DONE ok=%d bad=%d elapsed=%.0fs" % (ok, bad, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
