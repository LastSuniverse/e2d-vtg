from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _focal_bce(logits: Tensor, target: Tensor, mask: Tensor, gamma: float = 2.0) -> Tensor:
    p = logits.sigmoid()
    pt = p * target + (1.0 - p) * (1.0 - target)
    alpha = 0.75 * target + 0.25 * (1.0 - target)
    loss = -alpha * (1.0 - pt).pow(gamma) * F.logsigmoid(logits * (2.0 * target - 1.0))
    return loss.masked_select(mask).mean()


def _span_iou(s: Tensor, e: Tensor, gs: Tensor, ge: Tensor) -> Tensor:
    inter = (torch.minimum(e, ge) - torch.maximum(s, gs)).clamp_min(0)
    union = (e - s).clamp_min(1e-6) + (ge - gs).clamp_min(1e-6) - inter
    return inter / union.clamp_min(1e-6)


class BoundaryMambaLoss(nn.Module):
    """Boundary, span-ranking, quality and expected-coordinate losses."""

    def __init__(self, sigma: float = 1.5, span_temperature: float = 12.0) -> None:
        super().__init__()
        self.sigma = sigma
        self.span_temperature = span_temperature

    def forward(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        start = outputs["start_logits"]
        end = outputs["end_logits"]
        span = outputs["span_logits"]
        quality = outputs["quality_logits"]
        mask = outputs["video_mask"]
        lengths = batch["lengths"].to(start.device)
        gt = batch["spans"].to(start.device)  # normalized [B,2]
        b, t = start.shape
        # Keep coordinate and target construction in fp32.  Under AMP the
        # previous exp(span_temperature * IoU) could overflow at IoU=1,
        # producing NaNs before GradScaler had a chance to react.
        idx = torch.arange(t, device=start.device, dtype=torch.float32)[None]
        # The dataset normalizes GT by ctx_l * clip_len. With clip_len=1 s,
        # token i is the cell [i/L, (i+1)/L]. Keep the supervision grid on
        # that same cell convention as inference and the other detectors.
        scale = lengths.float().clamp_min(1.0)[:, None]
        gt32 = gt.float()
        gs = gt32[:, 0:1] * scale
        ge = gt32[:, 1:2] * scale - 1.0
        target_s = torch.exp(-0.5 * ((idx - gs) / self.sigma).pow(2)).clamp_max(1.0)
        target_e = torch.exp(-0.5 * ((idx - ge) / self.sigma).pow(2)).clamp_max(1.0)
        boundary_mask = mask
        loss_start = _focal_bce(start, target_s, boundary_mask)
        loss_end = _focal_bce(end, target_e, boundary_mask)

        idx = torch.arange(t, device=start.device, dtype=torch.float32)[None, :]
        starts = idx / scale
        ends = (idx + 1.0) / scale
        ss = starts[:, :, None].expand(-1, -1, t)
        ee = ends[:, None, :].expand(-1, t, -1)
        iou = _span_iou(ss, ee, gt32[:, 0, None, None], gt32[:, 1, None, None])
        tri = torch.triu(torch.ones(t, t, dtype=torch.bool, device=start.device), diagonal=1)[None]
        pair_mask = mask[:, :, None] & mask[:, None, :] & tri
        log_probs = F.log_softmax(span.float().masked_fill(~pair_mask, -1e4).flatten(1), dim=-1)
        target_logit = (self.span_temperature * iou).masked_fill(~pair_mask, -1e4).flatten(1)
        target_prob = F.softmax(target_logit, dim=-1)
        loss_span = -(target_prob * log_probs).sum(-1).mean()

        quality_target = iou.clamp(0, 1)
        q_pred = quality.float().sigmoid()
        loss_quality = F.smooth_l1_loss(q_pred[pair_mask], quality_target[pair_mask])

        prob = span.float().masked_fill(~pair_mask, -1e4).flatten(1).softmax(-1).view(b, t, t)
        pred_s = (prob * ss).sum((1, 2))
        pred_e = (prob * ee).sum((1, 2))
        loss_expected = F.l1_loss(pred_s, gt[:, 0]) + F.l1_loss(pred_e, gt[:, 1])

        total = loss_start + loss_end + loss_span + 0.5 * loss_quality + 0.5 * loss_expected
        return {
            "loss": total,
            "loss_start": loss_start.detach(),
            "loss_end": loss_end.detach(),
            "loss_span": loss_span.detach(),
            "loss_quality": loss_quality.detach(),
            "loss_expected": loss_expected.detach(),
        }
