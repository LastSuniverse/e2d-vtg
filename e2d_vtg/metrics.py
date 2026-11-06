"""TimeLens-Bench-compatible top-1 temporal grounding metrics.

The benchmark protocol is intentionally small and explicit: R1 at IoU
thresholds 0.3/0.5/0.7 and mean IoU of the first ranked prediction.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence


STANDARD_KEYS = ("R1@0.3", "R1@0.5", "R1@0.7", "mIoU")


def temporal_iou(pred: Sequence[float], target: Sequence[float]) -> float:
    ps, pe = float(pred[0]), float(pred[1])
    gs, ge = float(target[0]), float(target[1])
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(1e-12, max(0.0, pe - ps) + max(0.0, ge - gs) - inter)
    return inter / union


def standard_vtg_metrics(
    predictions: Iterable[Mapping],
    ground_truth: Iterable[Mapping],
    gt_mode: str = "first",
) -> Dict[str, float]:
    """Compute the four TimeLens-Bench metrics from JSON-style rows.

    ``pred_relevant_windows`` must be ranked, with the first item being the
    model's top-1 prediction.  ``relevant_windows`` is read from the first
    ground-truth window, matching Charades-STA and the TimeLens Charades split.
    Missing predictions are scored as IoU zero rather than silently dropped.
    """
    if gt_mode not in {"first", "best"}:
        raise ValueError(f"Unknown GT mode: {gt_mode}")
    pred_by_qid = {int(row["qid"]): row for row in predictions}
    values = []
    for row in ground_truth:
        qid = int(row["qid"])
        gt_windows = row.get("relevant_windows") or []
        gt = gt_windows[0] if gt_windows else [0.0, 0.0]
        pred_windows = (pred_by_qid.get(qid) or {}).get("pred_relevant_windows") or []
        pred = pred_windows[0] if pred_windows else [0.0, 0.0]
        values.append(max((temporal_iou(pred, target) for target in gt_windows), default=0.0)
                      if gt_mode == "best" else temporal_iou(pred, gt))
    n = max(1, len(values))
    return {
        "R1@0.3": sum(v >= 0.3 for v in values) / n,
        "R1@0.5": sum(v >= 0.5 for v in values) / n,
        "R1@0.7": sum(v >= 0.7 for v in values) / n,
        "mIoU": sum(values) / n,
    }
