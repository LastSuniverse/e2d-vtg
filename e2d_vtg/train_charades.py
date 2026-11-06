"""Train dense interval retrieval + query-conditioned proposal refinement."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import BoundaryDataset, collate_boundary
from .detector import (RetrievalRefinement, decode_intervals,
                               interval_iou, make_training_candidates,
                               retrieval_refinement_loss, select_candidates)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, proposal_k: int, rank_score: str = "combined"):
    model.eval()
    ious, proposal_ious = [], []
    for batch in loader:
        batch = move(batch, device)
        out = model(batch["video"], batch["text"], batch["video_mask"], batch["text_mask"])
        candidates = select_candidates(out["relation_logits"], out["valid"], proposal_k)
        refined = model.refine_candidates(out["features"], out["query"],
                                          out["relation_logits"], candidates,
                                          batch["video_mask"].sum(-1))
        if rank_score == "quality":
            refined["score"] = torch.nn.functional.logsigmoid(refined["quality"].float())
        elif rank_score == "retrieval":
            bsz, t, _ = out["relation_logits"].shape
            flat_logits = out["relation_logits"].flatten(1)
            indices = candidates[..., 0] * t + candidates[..., 1]
            refined["score"] = flat_logits.gather(1, indices)
        elif rank_score != "combined":
            raise ValueError(f"Unknown rank_score: {rank_score}")
        spans, _ = decode_intervals(refined, batch["durations"], topk=1)
        # BoundaryDataset stores normalized [0,1] spans.
        gt_norm = batch["spans"]
        lengths = batch["video_mask"].sum(-1).float().clamp_min(1)
        candidate_spans = torch.stack([
            candidates[..., 0].float() / lengths[:, None],
            (candidates[..., 1].float() + 1) / lengths[:, None],
        ], -1)
        proposal_ious.append(interval_iou(
            candidate_spans, gt_norm[:, None].expand_as(candidate_spans)).cpu())
        gt = batch["gt_seconds"]
        ious.append(interval_iou(spans[:, 0], gt).cpu())
    iou = torch.cat(ious)
    max_proposal_iou = torch.cat(proposal_ious).max(-1).values
    return {"R1@0.3": (iou >= .3).float().mean().item(),
            "R1@0.5": (iou >= .5).float().mean().item(),
            "R1@0.7": (iou >= .7).float().mean().item(),
            "mIoU": iou.mean().item(),
            f"proposal_R@{proposal_k}_0.3": (max_proposal_iou >= .3).float().mean().item(),
            f"proposal_R@{proposal_k}_0.5": (max_proposal_iou >= .5).float().mean().item(),
            f"proposal_R@{proposal_k}_0.7": (max_proposal_iou >= .7).float().mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--train-video-cache", required=True)
    p.add_argument("--eval-video-cache", required=True)
    p.add_argument("--train-query-cache", required=True)
    p.add_argument("--eval-query-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--video-key", choices=["qid", "vid"], default="qid")
    p.add_argument("--video-dim", type=int, default=2048)
    p.add_argument("--text-dim", type=int, default=2048)
    p.add_argument("--clip-len", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--proposal-k", type=int, default=64)
    p.add_argument("--rank-score", choices=["combined", "quality", "retrieval"], default="combined",
                   help="Ranking among retrieved proposals during test decoding")
    p.add_argument("--train-topk", type=int, default=48)
    p.add_argument("--jitter", type=int, default=12)
    p.add_argument("--rank-weight", type=float, default=0.0,
                   help="Weight of listwise soft-IoU ranking loss on the inference combined score")
    p.add_argument("--rank-temperature", type=float, default=0.08)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--amp", action="store_true")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = BoundaryDataset(a.train_jsonl, a.train_video_cache, a.train_query_cache,
                            a.video_key, strict=True, clip_len=a.clip_len)
    test = BoundaryDataset(a.eval_jsonl, a.eval_video_cache, a.eval_query_cache,
                           a.video_key, strict=True, clip_len=a.clip_len)
    if len(train) != 12404 or len(test) != 3720:
        raise RuntimeError(f"Expected full official Charades-STA train/test (12404/3720), got {len(train)}/{len(test)}")
    print(f"device={device} train={len(train)} official_test={len(test)} "
          f"missing_train(video={train.missing_video},query={train.missing_query}) "
          f"missing_test(video={test.missing_video},query={test.missing_query})", flush=True)
    loader_kw = dict(batch_size=a.batch_size, num_workers=a.workers,
                     pin_memory=device.type == "cuda", persistent_workers=a.workers > 0,
                     collate_fn=collate_boundary)
    tl = DataLoader(train, shuffle=True, **loader_kw)
    el = DataLoader(test, shuffle=False, **loader_kw)
    model = RetrievalRefinement(a.video_dim, a.text_dim, a.dim, a.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs * len(tl), eta_min=a.lr * .05)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device.type == "cuda")
    # A new budget changes the cosine trajectory from the first update. Keep
    # screening runs intact and refuse accidental checkpoint/log overwrites.
    outdir = Path(a.output_dir); outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "config.json").write_text(json.dumps(vars(a), indent=2))
    best, best_epoch = -1., 0
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); sums = {}
        for step, batch in enumerate(tl, 1):
            batch = move(batch, device)
            gt = batch["spans"]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=a.amp and device.type == "cuda"):
                base = model(batch["video"], batch["text"], batch["video_mask"], batch["text_mask"])
                candidates = make_training_candidates(base["relation_logits"].detach(),
                                                       base["valid"], gt, a.train_topk, a.jitter)
                refined = model.refine_candidates(base["features"], base["query"],
                                                  base["relation_logits"], candidates,
                                                  batch["video_mask"].sum(-1))
                base.update(refined)
                losses = retrieval_refinement_loss(
                    base, gt, rank_weight=a.rank_weight,
                    rank_temperature=a.rank_temperature)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, step {step}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if not scaler.is_enabled() or scaler.get_scale() >= scale_before:
                scheduler.step()
            for k, v in losses.items(): sums[k] = sums.get(k, 0.) + float(v.detach())
            if step % 200 == 0:
                print(f"epoch={epoch} step={step}/{len(tl)} loss={sums['loss']/step:.4f}", flush=True)
        metrics = evaluate(model, el, device, a.proposal_k, a.rank_score)
        row = {"epoch": epoch, "train_loss": sums.get("loss", 0.) / max(1, len(tl)),
               **metrics, "lr": scheduler.get_last_lr()[0], "seconds": time.time() - t0,
               "split_protocol": "Charades-STA official full train; official test every epoch",
               "input_protocol": "pooler_output; L2 normalize video/text; append two-dim TEF",
               "label_timebase": "ctx_l * clip_len; UVCOM-compatible",
               "seed": a.seed, "proposal_k": a.proposal_k, "rank_score": a.rank_score}
        print(json.dumps(row), flush=True)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                 "metrics": row, "config": vars(a)}
        torch.save(state, outdir / "last.ckpt")
        with (outdir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if metrics["R1@0.5"] > best:
            best, best_epoch = metrics["R1@0.5"], epoch
            torch.save(state, outdir / "model_best.ckpt")
        with (outdir / "best.json").open("w", encoding="utf-8") as f:
            json.dump({"best_epoch": best_epoch, "best_R1@0.5": best}, f)


if __name__ == "__main__":
    main()
