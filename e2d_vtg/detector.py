from __future__ import annotations

"""Dense interval retrieval followed by query-conditioned local refinement.

Unlike UVCOM, candidates are temporal windows enumerated on a start/end grid.
No learned slots, reference-point decoder, or Hungarian assignment is used.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RetrievalRefinement(nn.Module):
    def __init__(self, video_dim: int, text_dim: int, dim: int = 192,
                 rank: int = 64, max_len: int = 75) -> None:
        super().__init__()
        self.video_in = nn.Sequential(nn.Linear(video_dim, dim), nn.LayerNorm(dim))
        self.text_in = nn.Sequential(nn.Linear(text_dim, dim), nn.LayerNorm(dim))
        self.cross = nn.MultiheadAttention(dim, 6, dropout=0.1, batch_first=True)
        self.scales = nn.ModuleList([
            nn.Sequential(nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim),
                          nn.Conv1d(dim, dim, 1), nn.GELU())
            for k in (3, 7, 15)
        ])
        self.scale_gate = nn.Linear(dim, 3)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 6, 4 * dim, dropout=0.1,
                                       activation="gelu", batch_first=True,
                                       norm_first=True), 2)
        self.start = nn.Linear(dim, rank)
        self.end = nn.Linear(dim, rank)
        self.start_aux = nn.Linear(dim, 1)
        self.end_aux = nn.Linear(dim, 1)
        self.width_bias = nn.Embedding(max_len, 1)
        # Candidate descriptor: start, end, pooled interval, and query.
        self.refine = nn.Sequential(nn.LayerNorm(dim * 4 + 3),
                                    nn.Linear(dim * 4 + 3, dim * 2), nn.GELU(),
                                    nn.Dropout(0.1), nn.Linear(dim * 2, dim), nn.GELU())
        self.offset = nn.Linear(dim, 2)
        self.quality = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.rank_scale = rank ** -0.5
        self.dim = dim

    def encode(self, video: Tensor, text: Tensor, video_mask: Tensor,
               text_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        _, _, _ = video.shape
        v = self.video_in(video.float())
        qtok = self.text_in(text.float())
        q = (qtok * text_mask.unsqueeze(-1)).sum(1) / text_mask.sum(1, keepdim=True).clamp_min(1)
        cross, _ = self.cross(v, qtok, qtok, key_padding_mask=~text_mask, need_weights=False)
        x = v + cross
        branches = torch.stack([m(x.transpose(1, 2)).transpose(1, 2) for m in self.scales], 1)
        gate = torch.softmax(self.scale_gate(q), -1)[:, :, None, None]
        x = self.temporal(x + (branches * gate).sum(1), src_key_padding_mask=~video_mask)
        x = x * video_mask.unsqueeze(-1).to(x.dtype)
        return x, q, self.start(x), self.end(x)

    def retrieve(self, x: Tensor, q: Tensor, s: Tensor, e: Tensor,
                 video_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        b, t, _ = x.shape
        logits = torch.einsum("bir,bjr->bij", s, e) * self.rank_scale
        logits = logits + self.start_aux(x).squeeze(-1).unsqueeze(-1)
        logits = logits + self.end_aux(x).squeeze(-1).unsqueeze(-2)
        ei = torch.arange(t, device=x.device).view(1, 1, t)
        si = torch.arange(t, device=x.device).view(1, t, 1)
        width = (ei - si).clamp(0, self.width_bias.num_embeddings - 1)
        logits = logits + self.width_bias(width).squeeze(-1)
        valid = (video_mask[:, :, None] & video_mask[:, None, :]
                 & torch.ones(t, t, dtype=torch.bool, device=x.device).triu(0))
        logits = logits.masked_fill(~valid, -1e4)
        return logits, valid, ei, si

    def refine_candidates(self, x: Tensor, q: Tensor, logits: Tensor,
                          candidates: Tensor, lengths: Tensor) -> Dict[str, Tensor]:
        """Refine normalized [start_index,end_index] candidates, inclusive."""
        b, k, _ = candidates.shape
        t, d = x.shape[1], x.shape[2]
        left = candidates[..., 0].clamp(0, t - 1)
        right = candidates[..., 1].clamp(0, t - 1)
        prefix = F.pad(x.cumsum(1), (0, 0, 1, 0))
        rows = torch.arange(b, device=x.device)[:, None]
        seg = (prefix[rows, right + 1] - prefix[rows, left]) / (right - left + 1).unsqueeze(-1)
        xs, xe = x[rows, left], x[rows, right]
        width = ((right - left + 1).float() / lengths[:, None].clamp_min(1)).unsqueeze(-1)
        rel_s = (left.float() / lengths[:, None].clamp_min(1)).unsqueeze(-1)
        rel_e = ((right.float() + 1) / lengths[:, None].clamp_min(1)).unsqueeze(-1)
        desc = torch.cat([xs, xe, seg, q[:, None].expand(-1, k, -1), width, rel_s, rel_e], -1)
        h = self.refine(desc)
        delta = torch.tanh(self.offset(h)) * 0.25
        center = (rel_s.squeeze(-1) + rel_e.squeeze(-1)) * 0.5 + delta[..., 0]
        span = (rel_e.squeeze(-1) - rel_s.squeeze(-1)) * torch.exp(delta[..., 1].clamp(-1, 1))
        start = (center - span * 0.5).clamp(0, 1)
        end = (center + span * 0.5).clamp(0, 1)
        refined = torch.stack([start, end], -1)
        quality = self.quality(h).squeeze(-1)
        score = torch.log_softmax(logits.flatten(1), -1).gather(1, (left * t + right))
        score = score + F.logsigmoid(quality)
        return {"spans": refined, "score": score, "offset": delta,
                "quality": quality}

    def forward(self, video: Tensor, text: Tensor, video_mask: Tensor,
                text_mask: Tensor, candidates: Tensor | None = None) -> Dict[str, Tensor]:
        x, q, s, e = self.encode(video, text, video_mask, text_mask)
        logits, valid, _, _ = self.retrieve(x, q, s, e, video_mask)
        result = {"relation_logits": logits, "valid": valid, "features": x, "query": q}
        if candidates is not None:
            result.update(self.refine_candidates(x, q, logits, candidates, video_mask.sum(-1)))
        return result


def interval_iou(a: Tensor, b: Tensor) -> Tensor:
    inter = (torch.minimum(a[..., 1], b[..., 1]) - torch.maximum(a[..., 0], b[..., 0])).clamp_min(0)
    union = (a[..., 1] - a[..., 0]).clamp_min(0) + (b[..., 1] - b[..., 0]).clamp_min(0) - inter
    return inter / union.clamp_min(1e-6)


def make_training_candidates(logits: Tensor, valid: Tensor, gt: Tensor,
                             topk: int = 48, jitter: int = 12) -> Tensor:
    """Top retrieval windows plus GT-centered jitter to train the local head."""
    b, t, _ = logits.shape
    candidates = select_candidates(logits, valid, topk)
    # Deterministic offsets around GT make refinement supervision stable across epochs.
    offsets = torch.arange(-jitter // 2, jitter - jitter // 2, device=gt.device)
    lengths = valid.any(-1).sum(-1).float()
    centers = gt.mean(-1) * lengths
    widths = (gt[:, 1] - gt[:, 0]) * lengths
    starts = (centers[:, None] - widths[:, None] * 0.5 + offsets[None]).round().long()
    ends = (centers[:, None] + widths[:, None] * 0.5 + offsets[None]).round().long() - 1
    starts = starts.clamp(0, t - 1)
    ends = torch.maximum(ends.clamp(0, t - 1), starts)
    jittered = torch.stack([starts, ends], -1)
    all_cand = torch.cat([candidates, jittered], 1)
    # Remove padding-derived proposals while retaining GT jitter windows.
    lengths = valid.any(-1).sum(-1)[:, None]
    all_cand[..., 0] = torch.minimum(all_cand[..., 0], lengths - 1)
    all_cand[..., 0] = all_cand[..., 0].clamp_min(0)
    all_cand[..., 1] = torch.minimum(all_cand[..., 1], lengths - 1)
    return all_cand


def retrieval_refinement_loss(out: Dict[str, Tensor], gt: Tensor,
                             rank_weight: float = 0.0,
                             rank_temperature: float = 0.08) -> Dict[str, Tensor]:
    logits, valid = out["relation_logits"].float(), out["valid"]
    b, t, _ = logits.shape
    lengths = valid.any(-1).sum(-1).clamp_min(1).float()
    starts = torch.arange(t, device=logits.device, dtype=torch.float32)[None, :] / lengths[:, None]
    ends = (torch.arange(t, device=logits.device, dtype=torch.float32)[None, :] + 1) / lengths[:, None]
    ps, pe = starts[:, :, None], ends[:, None, :]
    gs, ge = gt[:, 0, None, None], gt[:, 1, None, None]
    inter = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp_min(0)
    union = (pe - ps).clamp_min(0) + (ge - gs).clamp_min(0) - inter
    iou = inter / union.clamp_min(1e-6)
    target = torch.softmax((iou / 0.08).masked_fill(~valid, -1e4).flatten(1), -1)
    retrieval = -(target * logits.flatten(1).log_softmax(-1)).sum(-1).mean()
    pred = out["spans"].float()
    cand_iou = interval_iou(pred, gt[:, None].expand_as(pred))
    qloss = F.binary_cross_entropy_with_logits(out["quality"].float(), cand_iou.detach())
    # Quality supervision already trains candidate ranking; direct soft-IoU
    # alignment improves the top proposal without unstable hard assignment.
    refine = 1.0 - cand_iou.mean()
    # Listwise supervision targets the exact combined score used at inference.
    # This is an isolated ablation: with rank_weight=0 the original objective
    # is unchanged; otherwise candidates with larger detached IoU receive more
    # probability mass, while gradients flow through retrieval and quality.
    target_rank = torch.softmax((cand_iou.detach() / rank_temperature), dim=-1)
    ranking = -(target_rank * F.log_softmax(out["score"].float(), dim=-1)).sum(-1).mean()
    total = retrieval + 0.5 * qloss + 0.5 * refine + rank_weight * ranking
    return {"loss": total, "retrieval": retrieval, "quality": qloss,
            "refine": refine, "ranking": ranking}


@torch.no_grad()
def select_candidates(logits: Tensor, valid: Tensor, topk: int = 64) -> Tensor:
    """Decode valid intervals, repeating the best when a short video has <K."""
    b, t, _ = logits.shape
    flat = logits.masked_fill(~valid, -1e4).flatten(1)
    rows = []
    for i in range(b):
        k = min(topk, int(valid[i].sum().item()))
        if k < 1:
            raise RuntimeError("Cannot decode intervals for an empty video")
        ids = flat[i].topk(k, dim=-1).indices
        if k < topk:
            ids = torch.cat([ids, ids[:1].expand(topk - k)], dim=0)
        rows.append(ids)
    ids = torch.stack(rows)
    starts, ends = ids // t, ids % t
    return torch.stack([starts, ends], -1)


@torch.no_grad()
def decode_intervals(out: Dict[str, Tensor], durations: Tensor, topk: int = 1) -> Tuple[Tensor, Tensor]:
    scores = out["score"]
    values, ids = scores.topk(min(topk, scores.shape[1]), -1)
    spans = out["spans"].gather(1, ids[..., None].expand(-1, -1, 2))
    return spans * durations[:, None, None], values
