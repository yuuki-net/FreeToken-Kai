"""Qwen-style multimodal rope (M-RoPE) for prompts with images.

Mirrors HF ``Qwen4ExpModel.get_rope_index`` / ``Qwen4ExpTextRotaryEmbedding``: a text run
takes consecutive positions on all three axes; an image's soft tokens take
``(t, h, w)`` grid positions offset by the running position, and the running position then
advances by ``max(h, w)`` (not by the token count), so every later token sits at
``logical + delta`` with ``delta = max_position + 1 - prompt_len <= 0``.

The cos/sin table uses FreeToken's cache layout (``[rows, rotary_dim]`` = ``[cos | sin]``,
float32) so the existing rope kernels index it by logical position unchanged.
"""

from __future__ import annotations

import torch


def image_rope_positions(
    input_ids: torch.Tensor, image_token_id: int, grid_thw: torch.Tensor, merge_size: int
) -> tuple[torch.Tensor, int]:
    """``(positions [3, L] int64, delta)`` for a prompt whose image placeholder runs (in
    order) correspond to ``grid_thw`` rows ``(t, h, w)`` in patch units (``merge_size``
    patches per soft token along h and w)."""
    ids = input_ids.reshape(-1).tolist()
    length = len(ids)
    pos = torch.empty(3, length, dtype=torch.int64)
    grids = [tuple(int(x) for x in g) for g in grid_thw.reshape(-1, 3).tolist()]
    gi, cur, i = 0, 0, 0
    while i < length:
        if ids[i] != image_token_id:
            j = i
            while j < length and ids[j] != image_token_id:
                j += 1
            pos[:, i:j] = torch.arange(cur, cur + (j - i), dtype=torch.int64)
            cur += j - i
            i = j
            continue
        if gi >= len(grids):
            raise ValueError("more image placeholder runs than images")
        t, h, w = grids[gi]
        gi += 1
        lh, lw = h // merge_size, w // merge_size
        n = t * lh * lw
        if i + n > length or any(ids[k] != image_token_id for k in range(i, i + n)):
            raise ValueError(
                f"image placeholder run at {i} is shorter than the image's {n} soft tokens"
            )
        tt = torch.arange(t).view(t, 1, 1).expand(t, lh, lw)
        hh = torch.arange(lh).view(1, lh, 1).expand(t, lh, lw)
        ww = torch.arange(lw).view(1, 1, lw).expand(t, lh, lw)
        pos[0, i : i + n] = tt.reshape(-1) + cur
        pos[1, i : i + n] = hh.reshape(-1) + cur
        pos[2, i : i + n] = ww.reshape(-1) + cur
        cur += max(lh, lw)
        i += n
    if gi != len(grids):
        raise ValueError("fewer image placeholder runs than images")
    delta = int(pos.max().item()) + 1 - length if length else 0
    return pos, delta


def mrope_cos_sin(
    pos3: torch.Tensor,
    rotary_dim: int,
    base: float,
    mrope_section: tuple[int, ...],
    *,
    interleaved: bool = True,
) -> torch.Tensor:
    """The ``[L, rotary_dim]`` float32 ``[cos | sin]`` table for 3-axis positions
    ``pos3 [3, L]``. Interleaved layout (HF ``apply_interleaved_mrope``): frequency ``i``
    takes the h axis when ``i % 3 == 1`` and ``i < 3 * section[1]``, the w axis when
    ``i % 3 == 2`` and ``i < 3 * section[2]``, the t axis otherwise. When all three axes
    agree (text) this equals the plain rope row for that position."""
    if not interleaved:
        raise NotImplementedError("only the interleaved M-RoPE layout is served")
    if len(mrope_section) != 3 or sum(mrope_section) != rotary_dim // 2:
        raise ValueError(
            f"mrope_section {mrope_section} must have 3 entries summing to rotary_dim/2 = {rotary_dim // 2}"
        )
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    freqs = pos3.to(torch.float32)[:, :, None] * inv_freq[None, None, :]  # [3, L, rd/2]
    out = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        length = mrope_section[axis] * 3
        idx = slice(offset, length, 3)
        out[:, idx] = freqs[axis][:, idx]
    return torch.cat((out.cos(), out.sin()), dim=-1)


__all__ = ["image_rope_positions", "mrope_cos_sin"]
