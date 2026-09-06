"""Host-side (CPU) image encoders for checkpoints whose vision tower is not built on the GPU.

``build_host_image_encoder(model_path)`` returns an object with
``encode(mm_inputs: dict) -> dict`` that replaces the HF processor's raw image tensors with
what the scheduler needs (``mm_embeds`` plus the grid the M-RoPE derives from), or None when
the checkpoint's images are encoded on the engine side (Gemma 4) or not at all.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
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
    """Runs the tower image by image so identical images (same file, same size cap) are
    encoded once: chat clients resend a conversation's images every turn and again for
    title / tag generation. ``FT_IMAGE_EMBED_CACHE`` = entries kept (default 32; an entry
    is at most a few MB; 0 disables)."""

    def __init__(self, tower: Any, cache_entries: int | None = None) -> None:
        self.tower = tower
        if cache_entries is None:
            cache_entries = int(os.environ.get("FT_IMAGE_EMBED_CACHE", "32"))
        self.capacity = max(0, cache_entries)
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(patches: torch.Tensor) -> str:
        return hashlib.sha1(patches.contiguous().cpu().numpy().tobytes()).hexdigest()

    def _encode_one(self, patches: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        if self.capacity == 0:
            return self.tower.encode(patches, grid.view(1, 3))
        key = self._key(patches)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return hit
        self.misses += 1
        embeds = self.tower.encode(patches, grid.view(1, 3))
        self._cache[key] = embeds
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return embeds

    def encode(self, mm: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        grid = mm["image_grid_thw"].to(torch.int64)
        pixel_values = mm["pixel_values"]
        parts = []
        offset = 0
        for g in grid:
            n = int(g.prod())
            parts.append(self._encode_one(pixel_values[offset : offset + n], g))
            offset += n
        if offset != pixel_values.shape[0]:
            raise ValueError(
                f"image_grid_thw covers {offset} patches but pixel_values has {pixel_values.shape[0]}"
            )
        return {
            "mm_embeds": torch.cat(parts, dim=0) if parts else pixel_values.new_zeros(0, 0),
            "image_grid_thw": grid.cpu(),
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
