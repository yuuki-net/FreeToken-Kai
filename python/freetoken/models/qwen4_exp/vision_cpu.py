"""Qwen3.8-Flash-Next vision tower on the CPU.

The checkpoint ships ``model.visual.*`` in bf16 (~0.9 GB); the GPU budget on a 12 GB card
has no room for it, and one image is a few seconds of CPU work, so the tower runs in the
tokenizer worker with transformers' own ``Qwen4ExpVisionModel`` (no reimplementation) and
hands the scheduler already-projected soft tokens (``[num_soft_tokens, text_hidden]``).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterable

import torch

VISUAL_PREFIXES = ("model.visual.", "visual.")


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
    """The HF vision tower + merger, resident on the CPU."""

    def __init__(self, model: Any, spatial_merge_size: int, dtype: torch.dtype) -> None:
        self.model = model
        self.spatial_merge_size = int(spatial_merge_size)
        self.dtype = dtype

    @classmethod
    def from_checkpoint(cls, model_path: str, dtype: torch.dtype = torch.float32) -> "Qwen4ExpCpuVision":
        from transformers import AutoConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

        hf = AutoConfig.from_pretrained(model_path)
        vc = getattr(hf, "vision_config", None)
        if vc is None:
            raise ValueError(f"{model_path}: config has no vision_config")
        vc._attn_implementation = "sdpa"
        model = Qwen4ExpVisionModel(vc).to(dtype).eval()
        model.load_state_dict(load_prefixed_state(model_path), strict=True)
        return cls(model, vc.spatial_merge_size, dtype)

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """``pixel_values [sum_patches, C*T*P*P]`` + ``image_grid_thw [N, 3]`` ->
        ``[sum(prod(grid) / merge**2), text_hidden]`` bf16 soft tokens (all images, in order)."""
        out = self.model(pixel_values.to(self.dtype), grid_thw=image_grid_thw.to(torch.long))
        return out.pooler_output.to(torch.bfloat16).contiguous()


__all__ = ["Qwen4ExpCpuVision", "load_prefixed_state", "VISUAL_PREFIXES"]
