from __future__ import annotations

import torch

from e2d_vtg.detector import (
    RetrievalRefinement,
    make_training_candidates,
    retrieval_refinement_loss,
)
from tools.joint_encoder import group_by_second


def test_detector_and_training_loss_run_on_cpu() -> None:
    torch.manual_seed(0)
    batch, length, video_dim, text_dim = 2, 5, 16, 16
    model = RetrievalRefinement(video_dim, text_dim, dim=24, rank=8, max_len=length)
    video = torch.randn(batch, length, video_dim)
    text = torch.randn(batch, 2, text_dim)
    video_mask = torch.ones(batch, length, dtype=torch.bool)
    text_mask = torch.ones(batch, 2, dtype=torch.bool)

    output = model(video, text, video_mask, text_mask)
    assert output["relation_logits"].shape == (batch, length, length)
    assert output["valid"].shape == (batch, length, length)
    assert torch.isfinite(output["relation_logits"]).all()

    target = torch.tensor([[0.1, 0.7], [0.2, 0.8]])
    candidates = make_training_candidates(
        output["relation_logits"].detach(),
        output["valid"],
        target,
        topk=4,
        jitter=1,
    )
    lengths = video_mask.sum(-1)
    refined = model.refine_candidates(
        output["features"],
        output["query"],
        output["relation_logits"],
        candidates,
        lengths,
    )
    losses = retrieval_refinement_loss({**output, **refined}, target)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()


def test_group_by_second_keeps_first_frame() -> None:
    frames = [object(), object(), object()]
    grouped = group_by_second(frames, [0.0, 0.5, 1.0], fps=2.0, frames_per_second=1)
    assert len(grouped) == 2
    assert grouped[0][0] is frames[0]
    assert grouped[1][0] is frames[2]


