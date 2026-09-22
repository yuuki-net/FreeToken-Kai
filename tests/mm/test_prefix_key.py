"""The radix key of an image prompt is the image's own pixels: the same image reuses the prefix, a different one does not.

No checkpoint here: the family's image processor is replaced by one whose rows come from the pixels, which is
what content_hash reads. tests/tokenizer/test_mm_tokenize.py covers the real processors when weights are present.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import torch

from freetoken.kvcache.prefix_disk_store import fingerprint_digest, prefix_key
from freetoken.kvcache.radix_cache import RadixPrefixCache
from freetoken.mm import MM_PAD_SHIFT_VALUE
from freetoken.mm.config import MultimodalConfig
from freetoken.mm.processors.qwen_vl import QwenVLMMProcessor

IMAGE_TOKEN = 151655
HEAD = [11, 12, 13]  # prompt text before the image
TAIL = [14, 15]  # and after it
GRID = [1, 4, 4]  # 16 patches, merged 2x2 -> 4 image tokens
PATCH_DIM = 3 * 2 * 16 * 16
RED, BLUE = (200, 30, 30), (30, 30, 200)


def _png(color, compress_level=6):
    """One solid-color image; compress_level changes the file bytes without touching a pixel."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, format="PNG", compress_level=compress_level)
    return buf.getvalue()


class _PixelImageProcessor:
    """Stands in for the checkpoint's: one row per patch, filled from the image's own pixels."""

    def __call__(self, images, **kwargs):
        pixels = torch.frombuffer(bytearray(images.convert("RGB").tobytes()), dtype=torch.uint8).to(torch.float32) / 255
        rows = torch.zeros(GRID[1] * GRID[2], PATCH_DIM)
        rows[:, : pixels.numel()] = pixels
        return {"pixel_values": rows, "image_grid_thw": torch.tensor([GRID])}


def _processor():
    hf = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        image_token_id=IMAGE_TOKEN,
        vision_config=SimpleNamespace(spatial_merge_size=2, patch_size=16, temporal_patch_size=2, in_channels=3),
        text_config=SimpleNamespace(rope_parameters={"rope_theta": 1e7}, vocab_size=248320),
    )
    proc = QwenVLMMProcessor(hf, "/nonexistent", MultimodalConfig())
    proc._image_processor = lambda: _PixelImageProcessor()
    return proc


def _prompt(proc, image):
    """The prompt ids of HEAD + one image + TAIL, and the image's item."""
    ids = torch.tensor(HEAD + [IMAGE_TOKEN] + TAIL, dtype=torch.int64)
    r = proc.apply(ids, [image])
    return r.input_ids, r.mm_items[0]


def _cache_holding(ids):
    cache = RadixPrefixCache(torch.device("cpu"), page_size=1)
    cache.insert_prefix(ids, torch.arange(len(ids)))
    return cache


def test_the_image_span_is_a_run_of_the_content_pad_id():
    ids, item = _prompt(_processor(), _png(RED))
    (start, end), = item.offsets
    assert (start, end) == (len(HEAD), len(HEAD) + 4)
    span = ids[start:end]
    assert span.unique().tolist() == [item.pad_value] and item.pad_value >= MM_PAD_SHIFT_VALUE
    assert bool((ids[:start] < MM_PAD_SHIFT_VALUE).all()) and bool((ids[end:] < MM_PAD_SHIFT_VALUE).all())


def test_the_same_pixels_in_another_file_reuse_the_whole_prefix():
    proc = _processor()
    a, item_a = _prompt(proc, _png(RED))
    b, item_b = _prompt(proc, _png(RED, compress_level=1))
    assert _png(RED) != _png(RED, compress_level=1)  # different bytes, same pixels
    assert item_a.hash == item_b.hash and item_a.pad_value == item_b.pad_value
    assert torch.equal(a, b)
    assert _cache_holding(a).match_prefix(b).cuda_handle.cached_len == len(b)


def test_a_different_image_reuses_only_the_text_before_it():
    proc = _processor()
    a, item_a = _prompt(proc, _png(RED))
    b, item_b = _prompt(proc, _png(BLUE))
    assert item_a.hash != item_b.hash and item_a.pad_value != item_b.pad_value
    assert _cache_holding(a).match_prefix(b).cuda_handle.cached_len == len(HEAD)


def test_the_disk_cache_key_follows_the_image_content():
    proc = _processor()
    a, _ = _prompt(proc, _png(RED))
    same, _ = _prompt(proc, _png(RED, compress_level=1))
    other, _ = _prompt(proc, _png(BLUE))
    digest = fingerprint_digest({"model": "test"})
    assert prefix_key(digest, a) == prefix_key(digest, same)
    assert prefix_key(digest, a) != prefix_key(digest, other)
