"""Qwen3-VL-Embedding feature encoder used by E2D-VTG.

The wrapper loads the embedding model through the release's official
``qwen3_vl_embedding.py`` helper and exposes three deterministic paths:

* ``encode`` fuses one instruction, one query, and one frame group.
* ``encode_text`` produces a pooled query vector.
* ``encode_image`` produces a query-agnostic video-only frame vector.

The paper protocol uses one keyframe per integer second at 448 pixels.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_INSTRUCTION = "Find the video segment that matches the description."


def _load_official(model_dir: str | Path, script: str):
    path = Path(model_dir) / "scripts" / script
    if not path.is_file():
        raise FileNotFoundError(
            f"official encoder helper not found: {path}. Download the complete "
            "Qwen3-VL-Embedding model directory before extracting features."
        )
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class JointEncoder:
    """Thin wrapper around the official Qwen3-VL-Embedding helper."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cuda",
        instruction: str = DEFAULT_INSTRUCTION,
    ) -> None:
        helper = _load_official(model_dir, "qwen3_vl_embedding.py")
        self.embedding = helper.Qwen3VLEmbedder(str(model_dir), torch_dtype=torch.bfloat16)
        self.embedding.model.to(device).eval()
        self.device = device
        self.instruction = instruction
        self.dim = int(self.embedding.model.config.text_config.hidden_size)

    def _conversations(self, texts, frame_groups):
        conversations = []
        for text, frames in zip(texts, frame_groups):
            content = [{"type": "image", "image": frame} for frame in frames]
            # Text is intentionally last. The pooled last-text token is the
            # only output that attends to the preceding visual tokens.
            content.append({"type": "text", "text": text})
            conversations.append(
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.instruction}],
                    },
                    {"role": "user", "content": content},
                ]
            )
        return conversations

    @torch.no_grad()
    def encode(self, texts, frame_groups, batch_size: int = 16) -> torch.Tensor:
        """Encode query-frame groups; returns L2-normalized ``(B, D)`` vectors."""
        if len(texts) != len(frame_groups):
            raise ValueError("texts and frame_groups must have equal length")
        outputs = []
        for start in range(0, len(texts), batch_size):
            conversations = self._conversations(
                texts[start : start + batch_size],
                frame_groups[start : start + batch_size],
            )
            inputs = self.embedding._preprocess_inputs(conversations)
            inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            output = self.embedding.model(**inputs)
            pooled = self.embedding._pooling_last(
                output.last_hidden_state, inputs["attention_mask"]
            )
            outputs.append(F.normalize(pooled, p=2, dim=-1))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_text(self, texts, batch_size: int = 32) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            inputs = self.embedding._preprocess_inputs(
                [
                    self.embedding.format_model_input(text=text)
                    for text in texts[start : start + batch_size]
                ]
            )
            inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            output = self.embedding.model(**inputs)
            pooled = self.embedding._pooling_last(
                output.last_hidden_state, inputs["attention_mask"]
            )
            outputs.append(F.normalize(pooled, p=2, dim=-1))
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_image(self, images, batch_size: int = 32) -> torch.Tensor:
        outputs = []
        for start in range(0, len(images), batch_size):
            inputs = self.embedding._preprocess_inputs(
                [
                    self.embedding.format_model_input(image=image)
                    for image in images[start : start + batch_size]
                ]
            )
            inputs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            output = self.embedding.model(**inputs)
            pooled = self.embedding._pooling_last(
                output.last_hidden_state, inputs["attention_mask"]
            )
            outputs.append(F.normalize(pooled, p=2, dim=-1))
        return torch.cat(outputs, dim=0)


def group_by_second(frames, timestamps, fps: float, frames_per_second: int | str):
    """Group decoded frames into integer-second bins.

    ``frames_per_second=1`` keeps the first decoded frame in each second and
    matches the paper's keyframe protocol. ``frames_per_second="all"`` keeps
    every decoded frame in its second.
    """
    if not frames:
        return []
    end_seconds = max(1, int(np.ceil((timestamps[-1] + 1.0 / fps) if timestamps else 1.0)))
    groups: list[list] = [[] for _ in range(end_seconds)]
    for frame, timestamp in zip(frames, timestamps):
        groups[min(int(timestamp), end_seconds - 1)].append(frame)
    if frames_per_second == 1:
        groups = [group[:1] for group in groups]
    return [group if group else [frames[0]] for group in groups]


def probe_duration(path: str | Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return float(result.stdout.strip())


def decode_video(path: str | Path, fps: float = 2.0, timeout: int = 1800):
    """Decode at ``fps`` and preserve 448-pixel width with aspect ratio."""
    workdir = Path(tempfile.mkdtemp(prefix="e2d_vtg_frames_"))
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(path),
                "-vf", f"fps={fps},scale=448:-2", "-q:v", "3",
                str(workdir / "f%06d.jpg"),
            ],
            check=True,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        names = sorted(workdir.glob("f*.jpg"))
        frames = [Image.open(name).convert("RGB") for name in names]
        timestamps = [index / fps for index in range(len(frames))]
        return frames, timestamps, probe_duration(path)
    finally:
        for name in workdir.iterdir():
            name.unlink()
        workdir.rmdir()
