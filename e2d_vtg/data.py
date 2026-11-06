from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class BoundaryDataset(Dataset):
    """JSONL + NPZ dataset for Qwen joint or CLIP caches.

    Video caches may be keyed by qid (Qwen joint features) or vid (CLIP
    features). Query caches are keyed by qid and contain last_hidden_state or
    tokens. Missing-cache records are reported and skipped during construction.
    """

    def __init__(
        self,
        jsonl: str,
        video_dir: str,
        query_dir: str,
        video_key: str = "qid",
        max_video_len: int = 75,
        max_query_len: int = 64,
        strict: bool = True,
        text_key: str = "pooler_output",
        normalize_video: bool = True,
        normalize_text: bool = True,
        append_tef: bool = True,
        clip_len: float = 1.0,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.query_dir = Path(query_dir)
        self.video_key = video_key
        self.max_video_len = max_video_len
        self.max_query_len = max_query_len
        self.text_key = text_key
        self.normalize_video = normalize_video
        self.normalize_text = normalize_text
        self.append_tef = append_tef
        self.clip_len = float(clip_len)
        self.missing_video = 0
        self.missing_query = 0
        self.rows: List[dict] = []
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qid = int(row["qid"])
                vid = str(row["vid"])
                vname = f"qid{qid}.npz" if video_key == "qid" else f"{vid}.npz"
                qname = f"qid{qid}.npz"
                vp = self.video_dir / vname
                qp = self.query_dir / qname
                if not vp.exists():
                    self.missing_video += 1
                if not qp.exists():
                    self.missing_query += 1
                if strict and (not vp.exists() or not qp.exists()):
                    continue
                row = dict(row)
                row["_video_path"] = str(vp)
                row["_query_path"] = str(qp)
                self.rows.append(row)
        if not self.rows:
            raise RuntimeError(f"No usable records in {jsonl}; missing video={self.missing_video}, query={self.missing_query}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, object]:
        r = self.rows[i]
        vz = np.load(r["_video_path"])
        qz = np.load(r["_query_path"])
        video = np.asarray(vz["features"], dtype=np.float32)[: self.max_video_len]
        if self.text_key == "pooler_output":
            if "pooler_output" not in qz.files:
                raise KeyError(f"pooler_output missing from {r['_query_path']}")
            text = np.asarray(qz["pooler_output"], dtype=np.float32).reshape(1, -1)
            qlen = 1
        elif self.text_key == "last_hidden_state":
            if "last_hidden_state" not in qz.files:
                raise KeyError(f"last_hidden_state missing from {r['_query_path']}")
            text = np.asarray(qz["last_hidden_state"], dtype=np.float32)
            qlen = int(np.asarray(qz["token_len"]).item()) if "token_len" in qz.files else text.shape[0]
        elif self.text_key == "tokens":
            if "tokens" not in qz.files:
                raise KeyError(f"tokens missing from {r['_query_path']}")
            text = np.asarray(qz["tokens"], dtype=np.float32)
            qlen = text.shape[0]
        else:
            raise ValueError(f"Unsupported text_key: {self.text_key}")
        text = text[: self.max_query_len]
        qlen = min(qlen, text.shape[0])
        if self.normalize_video:
            video = video / (np.linalg.norm(video, axis=-1, keepdims=True) + 1e-5)
        if self.normalize_text:
            text = text / (np.linalg.norm(text, axis=-1, keepdims=True) + 1e-5)
        if self.append_tef:
            length = max(len(video), 1)
            starts = np.arange(len(video), dtype=np.float32) / length
            ends = starts + 1.0 / length
            video = np.concatenate([video, np.stack([starts, ends], axis=-1)], axis=-1)
        duration = float(r["duration"])
        gt = list(r["relevant_windows"][0])
        # Match UVCOM's l1 supervision exactly: GT seconds are normalized by
        # ctx_l * clip_len. Predictions are still decoded in annotation seconds
        # using the sample's duration, as in UVCOM inference.
        timebase = max(len(video) * self.clip_len, 1e-6)
        span = [gt[0] / timebase, gt[1] / timebase]
        return {
            "video": video,
            "text": text,
            "video_len": len(video),
            "text_len": qlen,
            "span": np.asarray(span, dtype=np.float32),
            "gt_seconds": np.asarray(gt, dtype=np.float32),
            "qid": int(r["qid"]),
            "vid": str(r["vid"]),
            "duration": duration,
        }


def collate_boundary(samples: List[Dict[str, object]]) -> Dict[str, Tensor]:
    if not samples:
        raise ValueError("empty batch")
    b = len(samples)
    vd = int(np.asarray(samples[0]["video"]).shape[-1])
    qd = int(np.asarray(samples[0]["text"]).shape[-1])
    tv = max(int(s["video_len"]) for s in samples)
    tq = max(int(s["text_len"]) for s in samples)
    video = torch.zeros(b, tv, vd, dtype=torch.float32)
    text = torch.zeros(b, tq, qd, dtype=torch.float32)
    vm = torch.zeros(b, tv, dtype=torch.bool)
    qm = torch.zeros(b, tq, dtype=torch.bool)
    lengths = torch.zeros(b, dtype=torch.long)
    spans = torch.zeros(b, 2, dtype=torch.float32)
    gt_seconds = torch.zeros(b, 2, dtype=torch.float32)
    qids = []
    vids = []
    durations = torch.zeros(b, dtype=torch.float32)
    for i, s in enumerate(samples):
        v = torch.from_numpy(np.asarray(s["video"], dtype=np.float32))
        q = torch.from_numpy(np.asarray(s["text"], dtype=np.float32))[: int(s["text_len"])]
        video[i, : v.shape[0]] = v
        text[i, : q.shape[0]] = q
        vm[i, : v.shape[0]] = True
        qm[i, : q.shape[0]] = True
        lengths[i] = v.shape[0]
        spans[i] = torch.from_numpy(np.asarray(s["span"], dtype=np.float32))
        gt_seconds[i] = torch.from_numpy(np.asarray(s["gt_seconds"], dtype=np.float32))
        durations[i] = float(s["duration"])
        qids.append(int(s["qid"]))
        vids.append(str(s["vid"]))
    return {"video": video, "text": text, "video_mask": vm, "text_mask": qm,
            "lengths": lengths, "spans": spans, "gt_seconds": gt_seconds, "durations": durations,
            "qids": qids, "vids": vids}
