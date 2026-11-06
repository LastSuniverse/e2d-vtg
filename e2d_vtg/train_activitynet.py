"""Full ActivityNet Captions training for the retrieval+refinement detector.

Differences from train_retrieval_refine.py (Charades) and
train_qv_retrieval.py (QVHighlights), both of which hardcode a sequence cap:

  * --max-video-len / --max-len are exposed. ActivityNet has a long tail
    (p99 = 237 s, max = 755 s), so the cap is set to cover every video and the
    model is trained on genuinely variable-length input; nothing is truncated.
  * Batches are built by a length-bucketed sampler. Variable-length input is
    padded to the longest item in the batch, and the relation field is O(T^2),
    so mixing a 750 s video into a batch of 60 s videos would waste ~150x the
    compute. Bucketing keeps padding waste near zero without truncating.
  * The split sizes are arguments, not hardcoded, so the official
    train.json / val_2.json counts are asserted explicitly.

Every query in the manifest is used; the only videos that can be absent are
those without a feature cache entry, and the run refuses to start unless
--expected-train / --expected-eval match exactly.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .data import BoundaryDataset, collate_boundary
from .detector import (RetrievalRefinement, make_training_candidates,
                               retrieval_refinement_loss, select_candidates,
                               decode_intervals)
from .metrics import standard_vtg_metrics


def move(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class BucketSampler(Sampler):
    """Group indices of similar length, shuffle inside and across buckets.

    Lengths come from the annotation duration (1 feature per second), which is
    what the loader sees after truncation, so bucketing is exact without
    opening the npz files.
    """

    def __init__(self, lengths, batch_size, pool_mult=50, seed=0):
        self.lengths = np.asarray(lengths, dtype=np.float32)
        self.bs = batch_size
        self.pool = max(self.bs, batch_size * pool_mult)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths))
        order = order[np.argsort(self.lengths[order], kind="stable")]
        batches = []
        for s in range(0, len(order), self.pool):
            chunk = order[s:s + self.pool].copy()
            rng.shuffle(chunk)
            for b in range(0, len(chunk), self.bs):
                batches.append(chunk[b:b + self.bs].tolist())
        rng.shuffle(batches)
        for b in batches:
            yield b


@torch.inference_mode()
def evaluate(model, loader, device, gt, proposal_k, gt_mode):
    model.eval()
    preds = []
    for raw in loader:
        b = move(raw, device)
        out = model(b["video"], b["text"], b["video_mask"], b["text_mask"])
        cand = select_candidates(out["relation_logits"], out["valid"], proposal_k)
        refined = model.refine_candidates(out["features"], out["query"],
                                          out["relation_logits"], cand,
                                          b["video_mask"].sum(-1))
        sp, _ = decode_intervals(refined, b["durations"], topk=1)
        for i, q in enumerate(b["qids"]):
            s, e = sp[i, 0].float().cpu().tolist()
            preds.append({"qid": int(q), "pred_relevant_windows": [[s, e, 1.0]]})
    m = standard_vtg_metrics(preds, gt, gt_mode=gt_mode)
    model.train()
    return m, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--train-video-cache", required=True)
    ap.add_argument("--eval-video-cache", required=True)
    ap.add_argument("--train-query-cache", required=True)
    ap.add_argument("--eval-query-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--video-dim", type=int, default=2050)
    ap.add_argument("--text-dim", type=int, default=2048)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--max-video-len", type=int, default=760)
    ap.add_argument("--max-len", type=int, default=760)
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
    ap.add_argument("--gt-mode", choices=["first", "best"], default="first")
    ap.add_argument("--eval-every", type=int, default=1)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = BoundaryDataset(a.train_jsonl, a.train_video_cache, a.train_query_cache,
                         "qid", strict=True, max_video_len=a.max_video_len,
                         clip_len=a.clip_len)
    ev = BoundaryDataset(a.eval_jsonl, a.eval_video_cache, a.eval_query_cache,
                         "qid", strict=True, max_video_len=a.max_video_len,
                         clip_len=a.clip_len)
    print(json.dumps({"train": len(tr), "eval": len(ev),
                      "missing_train_video": tr.missing_video,
                      "missing_train_query": tr.missing_query,
                      "missing_eval_video": ev.missing_video,
                      "missing_eval_query": ev.missing_query}), flush=True)
    if len(tr) != a.expected_train or len(ev) != a.expected_eval:
        raise SystemExit("REFUSING: expected %d/%d, got %d/%d"
                         % (a.expected_train, a.expected_eval, len(tr), len(ev)))
    if tr.missing_video or tr.missing_query or ev.missing_video or ev.missing_query:
        raise SystemExit("REFUSING: incomplete caches")

    # durations are the bin counts the loader will see (1 feature per second)
    tr_lens = [min(float(r["duration"]), a.max_video_len) for r in tr.rows]
    ev_lens = [min(float(r["duration"]), a.max_video_len) for r in ev.rows]
    sampler = BucketSampler(tr_lens, a.batch_size, seed=a.seed)
    tl = DataLoader(tr, batch_sampler=sampler, num_workers=a.workers,
                    pin_memory=True, persistent_workers=a.workers > 0,
                    collate_fn=collate_boundary)
    el = DataLoader(ev, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                    pin_memory=True, persistent_workers=a.workers > 0,
                    collate_fn=collate_boundary)
    gt = [json.loads(l) for l in Path(a.eval_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]

    model = RetrievalRefinement(a.video_dim, a.text_dim, a.dim, a.rank,
                                max_len=a.max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tl), eta_min=a.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps({**vars(a), "model_max_len": a.max_len}, indent=2))
    print(json.dumps({"batches_per_epoch": len(tl), "max_video_len": a.max_video_len,
                      "max_len": a.max_len}), flush=True)

    best = -1.0
    for epoch in range(1, a.epochs + 1):
        sampler.set_epoch(epoch)
        model.train(); t0 = time.time(); total = 0.0
        for step, raw in enumerate(tl, 1):
            b = move(raw, device)
            target = b["spans"].squeeze(1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
                base = model(b["video"], b["text"], b["video_mask"], b["text_mask"])
                cand = make_training_candidates(base["relation_logits"].detach(), base["valid"],
                                                target, a.train_topk, a.jitter)
                base.update(model.refine_candidates(base["features"], base["query"],
                                                    base["relation_logits"], cand,
                                                    b["video_mask"].sum(-1)))
                loss = retrieval_refinement_loss(base, target)["loss"]
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
            if step % 100 == 0:
                print("TRAIN %d %d/%d loss %.4f" % (epoch, step, len(tl), total / step), flush=True)
        row = {"epoch": epoch, "train_loss": total / len(tl), "seconds": time.time() - t0}
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            m, preds = evaluate(model, el, device, gt, a.proposal_k, a.gt_mode)
            row.update(m)
            print(json.dumps(row), flush=True)
            state = {"model": model.state_dict(), "epoch": epoch,
                     "config": {**vars(a), "model_max_len": a.max_len}, "metrics": row}
            torch.save(state, out / "last.ckpt")
            if m["R1@0.5"] > best:
                best = m["R1@0.5"]
                torch.save(state, out / "model_best.ckpt")
                (out / "best.json").write_text(json.dumps(row, indent=2))
                (out / "best_predictions.jsonl").write_text(
                    "".join(json.dumps(x) + "\n" for x in preds))
        else:
            print(json.dumps(row), flush=True)
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": {**vars(a), "model_max_len": a.max_len}, "metrics": row},
                       out / "last.ckpt")
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    print("ANET_TRAIN_DONE best=%.4f" % best, flush=True)


if __name__ == "__main__":
    main()
