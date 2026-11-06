"""Multi-domain training on a merged Charades + QVHighlights split.

Motivation. Single-domain training produces a head whose scoring function is
tuned to one dataset's statistics: zero-shot transfer drops 20-40 R1@0.5,
while the interval proposals themselves stay largely intact (oracle@16 62-63
vs 80-86 in-domain). The question this run answers is whether seeing two
domains at once produces a scoring function that is less dataset-bound.

Design notes
  * Both datasets share the query-conditioned Qwen joint features, so the
    merged input space is homogeneous. ActivityNet is deliberately excluded:
    its cache is video-only, and mixing the two input semantics would make the
    result uninterpretable.
  * QVDataset + max-IoU over all annotated windows is used for every row, so
    QVHighlights keeps its 7 windows and Charades contributes its single one.
  * Length-bucketed batches: Charades clips are ~30 s, QV clips 150 s, and the
    relation field is O(T^2), so mixing them blindly would pad every batch to
    the longest member.
  * --max-len covers the longest interval width in bins, not the video length.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate_boundary
from .detector import RetrievalRefinement, make_training_candidates
from .train_qvhighlights import QVDataset, collate_qv, qv_loss, evaluate
from .train_activitynet import BucketSampler, move


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--train-video-cache", required=True)
    ap.add_argument("--train-query-cache", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--eval-video-cache", required=True)
    ap.add_argument("--eval-query-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--video-key", default="qid")
    ap.add_argument("--video-dim", type=int, default=2050)
    ap.add_argument("--text-dim", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max-video-len", type=int, default=160)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--clip-len", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--proposal-k", type=int, default=16)
    ap.add_argument("--train-topk", type=int, default=48)
    ap.add_argument("--jitter", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--expected-train", type=int, required=True)
    ap.add_argument("--expected-eval", type=int, required=True)
    ap.add_argument("--eval-every", type=int, default=5)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = QVDataset(a.train_jsonl, a.train_video_cache, a.train_query_cache,
                   max_video_len=a.max_video_len, clip_len=a.clip_len)
    ev = QVDataset(a.eval_jsonl, a.eval_video_cache, a.eval_query_cache,
                   max_video_len=a.max_video_len, clip_len=a.clip_len)
    print(json.dumps({"train": len(tr), "eval": len(ev),
                      "missing_train_video": tr.missing_video,
                      "missing_train_query": tr.missing_query,
                      "missing_eval_video": ev.missing_video,
                      "missing_eval_query": ev.missing_query}), flush=True)
    if len(tr) != a.expected_train or len(ev) != a.expected_eval:
        raise SystemExit("REFUSING: expected %d/%d got %d/%d"
                         % (a.expected_train, a.expected_eval, len(tr), len(ev)))

    lens = [min(float(r["duration"]), a.max_video_len) for r in tr.rows]
    sampler = BucketSampler(lens, a.batch_size, seed=a.seed)
    tl = DataLoader(tr, batch_sampler=sampler, num_workers=a.workers,
                    pin_memory=True, persistent_workers=a.workers > 0,
                    collate_fn=collate_qv)
    el = DataLoader(ev, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                    pin_memory=True, persistent_workers=a.workers > 0,
                    collate_fn=collate_qv)

    model = RetrievalRefinement(a.video_dim, a.text_dim, a.dim, a.rank,
                                max_len=a.max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl), eta_min=a.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    print(json.dumps({"batches_per_epoch": len(tl)}), flush=True)

    best = -1.0
    for epoch in range(1, a.epochs + 1):
        sampler.set_epoch(epoch)
        model.train(); t0 = time.time(); total = 0.0
        for step, raw in enumerate(tl, 1):
            b = move(raw, device)
            n_gt = b["gt_mask"].sum(-1)
            pick = torch.floor(torch.rand(len(n_gt), device=device) * n_gt).long()
            target = b["all_gt"][torch.arange(len(pick), device=device), pick]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                base = model(b["video"], b["text"], b["video_mask"], b["text_mask"])
                cand = make_training_candidates(base["relation_logits"].detach(), base["valid"],
                                                target, a.train_topk, a.jitter)
                base.update(model.refine_candidates(base["features"], base["query"],
                                                    base["relation_logits"], cand,
                                                    b["video_mask"].sum(-1)))
                loss = qv_loss(base, b["all_gt"], b["gt_mask"])["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss at epoch %d step %d" % (epoch, step))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            before = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if not scaler.is_enabled() or scaler.get_scale() >= before:
                sched.step()
            total += float(loss.detach())
            if step % 200 == 0:
                print("TRAIN %d %d/%d loss %.4f" % (epoch, step, len(tl), total / step), flush=True)
        row = {"epoch": epoch, "train_loss": total / len(tl), "seconds": time.time() - t0}
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            m, _ = evaluate(model, el, device, a.proposal_k)
            row.update(m)
            state = {"model": model.state_dict(), "epoch": epoch, "config": vars(a), "metrics": row}
            torch.save(state, out / "last.ckpt")
            if m["R1@0.5"] > best:
                best = m["R1@0.5"]
                torch.save(state, out / "model_best.ckpt")
                (out / "best.json").write_text(json.dumps(row, indent=2))
        else:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": vars(a), "metrics": row}, out / "last.ckpt")
        print(json.dumps(row), flush=True)
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    print("MIXED_TRAIN_DONE best=%.4f" % best, flush=True)


if __name__ == "__main__":
    main()
