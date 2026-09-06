"""Qwen vision tower on the CPU (Qwen3.8-Flash-Next and the Qwen3.5-MoE family).

These checkpoints ship ``model.visual.*`` in bf16 (~0.9 GB); the GPU budget on a 12 GB (or
6 GB) card has no room for it, and one image is a few seconds of CPU work, so the tower runs
in the tokenizer worker with transformers' own vision module (no reimplementation) and hands
the scheduler already-projected soft tokens (``[num_soft_tokens, text_hidden]``). The text
model then only scatters them at the placeholder rows and ropes with M-RoPE positions.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterable

import torch

VISUAL_PREFIXES = ("model.visual.", "visual.")

# HF model_type -> (module, class) of the vision tower transformers implements for it.
_VISION_CLASSES = {
    "qwen4_exp": ("transformers.models.qwen4_exp.modeling_qwen4_exp", "Qwen4ExpVisionModel"),
    "qwen3_5_moe": ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeVisionModel"),
}
HOST_VISION_MODEL_TYPES = tuple(_VISION_CLASSES)


def vision_model_class(model_type: str):
    """The transformers vision tower class for ``model_type`` (ValueError if unsupported)."""
    import importlib

    try:
        module, name = _VISION_CLASSES[model_type]
    except KeyError:
        raise ValueError(
            f"no CPU vision tower for model_type {model_type!r} (supported: {HOST_VISION_MODEL_TYPES})"
        ) from None
    return getattr(importlib.import_module(module), name)


def load_prefixed_state(
    model_path: str, prefixes: Iterable[str] = VISUAL_PREFIXES
) -> Dict[str, torch.Tensor]:
    """Every checkpoint tensor under one of ``prefixes`` (prefix stripped), read from the
    safetensors shards on the CPU. Uses ``model.safetensors.index.json`` when present, else
    scans every shard header."""
    from safetensors import safe_open

    prefixes = tuple(prefixes)
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    by_shard: Dict[str, list[str]] = {}
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        for name, shard in weight_map.items():
            if name.startswith(prefixes):
                by_shard.setdefault(os.path.join(model_path, shard), []).append(name)
    else:
        for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                names = [k for k in f.keys() if k.startswith(prefixes)]
            if names:
                by_shard[shard] = names
    state: Dict[str, torch.Tensor] = {}
    for shard, names in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as f:
            for name in names:
                key = next(name[len(p):] for p in prefixes if name.startswith(p))
                state[key] = f.get_tensor(name)
    if not state:
        raise FileNotFoundError(
            f"no tensors under {prefixes} in {model_path}: this checkpoint has no vision tower"
        )
    return state


class Qwen4ExpCpuVision:
    """The HF vision tower + merger, resident on the CPU (any model_type in
    ``HOST_VISION_MODEL_TYPES``; the name predates the Qwen3.5-MoE support)."""

    def __init__(self, model: Any, spatial_merge_size: int, dtype: torch.dtype) -> None:
        self.model = model
        self.spatial_merge_size = int(spatial_merge_size)
        self.dtype = dtype

    @classmethod
    def from_checkpoint(cls, model_path: str, dtype: torch.dtype = torch.float32) -> "Qwen4ExpCpuVision":
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_path)
        vc = getattr(hf, "vision_config", None)
        if vc is None:
            raise ValueError(f"{model_path}: config has no vision_config")
        if getattr(vc, "deepstack_visual_indexes", None):
            # DeepStack injects vision features into several early decoder layers; this path
            # only scatters the merger output at the placeholder rows.
            raise ValueError(f"{model_path}: DeepStack vision (deepstack_visual_indexes) is not supported")
        vc._attn_implementation = "sdpa"
        model = vision_model_class(getattr(hf, "model_type", "")) (vc).to(dtype).eval()
        model.load_state_dict(load_prefixed_state(model_path), strict=True)
        return cls(model, vc.spatial_merge_size, dtype)

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """``pixel_values [sum_patches, C*T*P*P]`` + ``image_grid_thw [N, 3]`` ->
        ``[sum(prod(grid) / merge**2), text_hidden]`` bf16 soft tokens (all images, in order)."""
        out = self.model(pixel_values.to(self.dtype), grid_thw=image_grid_thw.to(torch.long))
        return out.pooler_output.to(torch.bfloat16).contiguous()


CpuVisionTower = Qwen4ExpCpuVision

__all__ = [
    "CpuVisionTower",
    "HOST_VISION_MODEL_TYPES",
    "Qwen4ExpCpuVision",
    "VISUAL_PREFIXES",
    "load_prefixed_state",
    "vision_model_class",
]
