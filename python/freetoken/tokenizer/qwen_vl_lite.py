"""A Qwen-VL processor without the video processor.

``AutoProcessor`` assembles ``Qwen3VLProcessor`` with ``Qwen3VLVideoProcessor``, which needs
torchvision even when no video is ever passed. For images the processor's whole job is:
render the chat template (it emits one ``<|image_pad|>`` per image part), run the image
processor, and repeat each ``<|image_pad|>`` ``prod(grid_thw) // merge_size**2`` times. This
class does that with the PIL image processor and the tokenizer, so a server needs only Pillow.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

IMAGE_TOKEN = "<|image_pad|>"

# preprocessor_config.json image_processor_type -> module holding the PIL backend class
_PIL_IMAGE_PROCESSORS = {
    "Qwen2VLImageProcessor": ("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl", "Qwen2VLImageProcessorPil"),
}


def _image_processor_type(model_path: str) -> str | None:
    try:
        with open(os.path.join(model_path, "preprocessor_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:  # noqa: BLE001 -- no preprocessor config: not a VL checkpoint
        return None
    kind = str(cfg.get("image_processor_type") or "")
    return kind.removesuffix("Fast").removesuffix("Pil") or None


class QwenVLLiteProcessor:
    """Duck-types the slice of ``Qwen3VLProcessor`` the tokenizer worker uses:
    ``image_processor`` (with ``.size``), ``image_token``, ``apply_chat_template`` and
    ``__call__(text=, images=, return_tensors=, add_special_tokens=, size=)``."""

    image_token = IMAGE_TOKEN

    def __init__(self, tokenizer: Any, image_processor: Any) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.merge_size = int(getattr(image_processor, "merge_size", 2))

    @classmethod
    def from_pretrained(cls, model_path: str, tokenizer: Any) -> "QwenVLLiteProcessor | None":
        kind = _image_processor_type(model_path)
        target = _PIL_IMAGE_PROCESSORS.get(kind or "")
        if target is None:
            return None
        import importlib

        module, name = target
        proc_cls = getattr(importlib.import_module(module), name)
        return cls(tokenizer, proc_cls.from_pretrained(model_path))

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        # the checkpoint's own template renders {"type": "image"} parts as the image token
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Any],
        return_tensors: str = "pt",
        add_special_tokens: bool = False,
        size: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if len(text) != 1:
            raise ValueError("QwenVLLiteProcessor handles one prompt at a time")
        prompt = text[0]
        flat = [im for group in images for im in (group if isinstance(group, (list, tuple)) else [group])]
        if prompt.count(self.image_token) != len(flat):
            raise ValueError(
                f"{prompt.count(self.image_token)} image placeholder(s) in the prompt but "
                f"{len(flat)} image(s)"
            )
        kwargs: dict[str, Any] = {"return_tensors": return_tensors}
        if size is not None:
            kwargs["size"] = dict(size)
        out = self.image_processor(images=flat, **kwargs)
        grid = out["image_grid_thw"]
        counts = (grid.prod(-1) // (self.merge_size**2)).tolist()
        pieces = prompt.split(self.image_token)
        expanded = pieces[0]
        for n, piece in zip(counts, pieces[1:]):
            expanded += self.image_token * int(n) + piece
        enc = self.tokenizer(expanded, return_tensors=return_tensors, add_special_tokens=add_special_tokens)
        return {"input_ids": enc["input_ids"], "pixel_values": out["pixel_values"], "image_grid_thw": grid}


__all__ = ["QwenVLLiteProcessor", "IMAGE_TOKEN"]
