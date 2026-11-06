"""Shared-evaluator audit for Retrieval+Refinement checkpoints.

Scores every official Charades-STA test qid and writes top-1 predictions and
the four standardized temporal grounding metrics. Can compare candidate ranking
rules on one frozen checkpoint without retraining the detector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import BoundaryDataset, collate_boundary
from .detector import RetrievalRefinement, decode_intervals, select_candidates
from .train_charades import move
from .metrics import standard_vtg_metrics


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--video-cache", required=True)
    p.add_argument("--query-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rank-score", choices=["combined", "quality", "retrieval"], default="combined")
    p.add_argument("--proposal-k", type=int, default=64)
    p.add_argument("--disable-refinement", action="store_true",
                   help="Use the top dense-retrieval interval without the local boundary refiner")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataset-name", default="Charades-STA official test")
    p.add_argument("--expected-records", type=int, default=3720)
    p.add_argument("--max-video-len", type=int, default=None)
    p.add_argument("--clip-len", type=float, default=None)
    p.add_argument("--gt-mode", choices=["first", "best"], default="first")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    ds = BoundaryDataset(a.jsonl, a.video_cache, a.query_cache, "qid", strict=True,
                         max_video_len=a.max_video_len or int(config.get("max_video_len", 75)),
                         clip_len=a.clip_len if a.clip_len is not None else float(config.get("clip_len", 1.0)))
    if len(ds) != a.expected_records or ds.missing_video or ds.missing_query:
        raise RuntimeError(f"Expected all {a.dataset_name} qids ({a.expected_records}); got {len(ds)}, "
                           f"missing video={ds.missing_video}, query={ds.missing_query}")
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=device.type == "cuda",
                        persistent_workers=a.workers > 0, collate_fn=collate_boundary)
    video_dim = int(config.get("video_dim", 2050))
    text_dim = int(config.get("text_dim", 2048))
    dim = int(config.get("dim", 192))
    rank = int(config.get("rank", 64))
    model = RetrievalRefinement(video_dim, text_dim, dim, rank,
                                max_len=int(config.get("max_len", 75))).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    predictions = []
    for raw in loader:
        batch = move(raw, device)
        out = model(batch["video"], batch["text"], batch["video_mask"], batch["text_mask"])
        candidates = select_candidates(out["relation_logits"], out["valid"], a.proposal_k)
        if a.disable_refinement:
            t = out["relation_logits"].shape[1]
            top = candidates[:, 0]
            spans = torch.stack([top[:, 0].float() / t,
                                 (top[:, 1].float() + 1.0) / t], -1)
            spans = spans[:, None] * batch["durations"][:, None, None]
        else:
            refined = model.refine_candidates(out["features"], out["query"],
                                              out["relation_logits"], candidates,
                                              batch["video_mask"].sum(-1))
            if a.rank_score == "quality":
                refined["score"] = torch.nn.functional.logsigmoid(refined["quality"].float())
            elif a.rank_score == "retrieval":
                t = out["relation_logits"].shape[1]
                ids = candidates[..., 0] * t + candidates[..., 1]
                refined["score"] = out["relation_logits"].flatten(1).gather(1, ids)
            spans, _ = decode_intervals(refined, batch["durations"], topk=1)
        for i, qid in enumerate(batch["qids"]):
            s, e = spans[i, 0].float().cpu().tolist()
            predictions.append({"qid": int(qid), "pred_relevant_windows": [[s, e, 1.0]]})

    ground_truth = [json.loads(line) for line in Path(a.jsonl).read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    if (len(predictions) != a.expected_records or len(ground_truth) != a.expected_records
            or {r["qid"] for r in predictions} != {r["qid"] for r in ground_truth}):
        raise RuntimeError("Predictions do not cover the exact requested benchmark qids")
    metrics = standard_vtg_metrics(predictions, ground_truth, gt_mode=a.gt_mode)
    result = {"checkpoint": str(a.checkpoint), "epoch": state.get("epoch"),
              "dataset": a.dataset_name, "records": len(predictions),
              "max_video_len": ds.max_video_len, "clip_len": ds.clip_len,
              "gt_mode": a.gt_mode,
              "rank_score": ("retrieval_only_no_refinement" if a.disable_refinement else a.rank_score),
              "proposal_k": a.proposal_k, **metrics}
    out_path = Path(a.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out_path.with_suffix(".predictions.jsonl")).write_text(
        "".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
