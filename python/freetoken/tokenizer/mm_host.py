"""Host-side (CPU) image encoders for checkpoints whose vision tower is not built on the GPU.

``build_host_image_encoder(model_path)`` returns an object with
``encode(mm_inputs: dict) -> dict`` that replaces the HF processor's raw image tensors with
what the scheduler needs (``mm_embeds`` plus the grid the M-RoPE derives from), or None when
the checkpoint's images are encoded on the engine side (Gemma 4) or not at all.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)


def _model_type(model_path: str) -> str | None:
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            return json.load(f).get("model_type")
    except Exception:  # noqa: BLE001 -- hub ids / missing config: no host encoder
        return None


class _QwenHostEncoder:
    def __init__(self, tower: Any) -> None:
        self.tower = tower

    def encode(self, mm: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        grid = mm["image_grid_thw"]
        embeds = self.tower.encode(mm["pixel_values"], grid)
        return {
            "mm_embeds": embeds,
            "image_grid_thw": grid.to(torch.int64).cpu(),
            "merge_size": torch.tensor(self.tower.spatial_merge_size, dtype=torch.int64),
        }


def build_host_image_encoder(model_path: str | None) -> Any | None:
    if not model_path or _model_type(model_path) != "qwen4_exp":
        return None
    from freetoken.models.qwen4_exp.vision_cpu import Qwen4ExpCpuVision

    logger.info("loading the vision tower on the CPU for image input (%s)", model_path)
    tower = Qwen4ExpCpuVision.from_checkpoint(model_path)
    logger.info("vision tower ready (merge %d)", tower.spatial_merge_size)
    return _QwenHostEncoder(tower)


__all__ = ["build_host_image_encoder"]
