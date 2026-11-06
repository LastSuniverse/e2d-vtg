"""Extract 1 fps keyframes for Charades videos (same recipe as ActivityNet).

The project's locked recipe: ffmpeg decodes at 2 fps, width 448, JPEG q3; the
first decoded frame of each integer second is kept and renamed f000001.jpg...
Each video gets a meta.json (duration probed, sha256 of the source) and is only
published after a frame-count vs duration sanity check. A `_DONE` marker makes
reruns skip finished videos.

Charades has no split directories here -- one flat output dir, since the
video-only encoder keys everything by video id.
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
    vid, src, out_root = item
    src = Path(src)
    dest = Path(out_root) / vid
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
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
             "-threads", "1", "-filter_threads", "1", "-i", str(src),
             "-map", "0:v:0", "-an", "-vf", "fps=2,scale=448:-2",
             "-threads", "1", "-q:v", "3", str(tmp / "f%06d.jpg")],
            check=True, timeout=900)
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
        meta = dict(video_id=vid, path=str(dest), source=str(src),
                    source_size=src.stat().st_size, source_sha256=sha256_of(src),
                    fps=1.0, width=448, jpeg_quality=3, num_frames=count,
                    duration=dur, decode_fps=2.0, frames_per_second=1,
                    sampling_recipe=SAMPLING_RECIPE, pixels_verified=True,
                    origin="charades_v1_480_hf_mirror",
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


def init(raw_root, out_root):
    STATE["raw"] = raw_root
    STATE["out"] = out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    raw = {p.stem: p for p in Path(a.raw_root).rglob("*")
           if p.suffix.lower() in (".mp4", ".mkv", ".webm")}
    todo = [(vid, str(p), a.out_root) for vid, p in sorted(raw.items())
            if not (Path(a.out_root) / vid / "_DONE").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(json.dumps({"raw_videos": len(raw), "todo": len(todo)}), flush=True)

    t0 = time.time()
    ok = bad = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=init,
                  initargs=(a.raw_root, a.out_root)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, todo), 1):
            if r == "ok":
                ok += 1
            elif r != "skip":
                bad += 1
                if bad <= 20:
                    print(r, flush=True)
            if i % 200 == 0:
                el = time.time() - t0
                print("progress %d/%d ok=%d bad=%d %.1f vid/s eta %.0f min"
                      % (i, len(todo), ok, bad, i / el,
                         (len(todo) - i) / (i / el) / 60), flush=True)
    print("CHARADES_FRAMES_DONE ok=%d bad=%d elapsed=%.0fs" % (ok, bad, time.time() - t0),
          flush=True)


if __name__ == "__main__":
    main()
