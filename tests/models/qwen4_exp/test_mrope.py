"""M-RoPE for image prompts: HF's position assignment and interleaved cos/sin, reproduced
in FreeToken's table layout."""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen4_exp.mrope import image_rope_positions, mrope_cos_sin

IMG = 7


def _prompt(text_before: int, grid, text_after: int, merge: int = 2):
    t, h, w = grid
    n = t * (h // merge) * (w // merge)
    ids = [1] * text_before + [IMG] * n + [2] * text_after
    return torch.tensor(ids, dtype=torch.int32), torch.tensor([grid])


def test_positions_follow_the_hf_rule():
    ids, grid = _prompt(3, (1, 4, 6), 2)  # image -> 2 x 3 soft tokens
    pos, delta = image_rope_positions(ids, IMG, grid, 2)
    # text: 0,1,2 on every axis
    assert pos[:, :3].tolist() == [[0, 1, 2]] * 3
    # image: t = 3, h = 3 + row, w = 3 + col, row-major
    assert pos[0, 3:9].tolist() == [3] * 6
    assert pos[1, 3:9].tolist() == [3, 3, 3, 4, 4, 4]
    assert pos[2, 3:9].tolist() == [3, 4, 5, 3, 4, 5]
    # text after: continues from 3 + max(2, 3) = 6
    assert pos[:, 9:].tolist() == [[6, 7]] * 3
    # delta = max + 1 - len = 8 - 11
    assert delta == -3


def test_two_images_consume_grids_in_order():
    ids = torch.tensor([1] + [IMG] * 4 + [5] + [IMG] * 2 + [2], dtype=torch.int32)
    grid = torch.tensor([[1, 4, 4], [1, 2, 4]])  # 2x2 = 4 tokens, then 1x2 = 2 tokens
    pos, delta = image_rope_positions(ids, IMG, grid, 2)
    assert pos[1, 1:5].tolist() == [1, 1, 2, 2] and pos[2, 1:5].tolist() == [1, 2, 1, 2]
    assert pos[:, 5].tolist() == [3, 3, 3]          # 1 + max(2,2) = 3
    assert pos[1, 6:8].tolist() == [4, 4] and pos[2, 6:8].tolist() == [4, 5]
    assert pos[:, 8].tolist() == [6, 6, 6]          # 4 + max(1,2) = 6
    assert delta == 7 - 9


@pytest.mark.parametrize("bad", ["short_run", "extra_grid", "missing_grid"])
def test_placeholder_grid_mismatch_is_an_error(bad):
    if bad == "short_run":
        ids, grid = torch.tensor([1, IMG, IMG, 2]), torch.tensor([[1, 4, 4]])
    elif bad == "extra_grid":
        ids, grid = torch.tensor([1, 2]), torch.tensor([[1, 2, 2]])
    else:
        ids, grid = torch.tensor([1, IMG, 2]), torch.tensor([], dtype=torch.long).reshape(0, 3)
    with pytest.raises(ValueError):
        image_rope_positions(ids, IMG, grid, 2)


def test_text_rows_equal_plain_rope_rows():
    rd, base = 64, 1e7
    pos3 = torch.arange(10).repeat(3, 1)
    table = mrope_cos_sin(pos3, rd, base, (11, 11, 10))
    inv = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    freqs = torch.arange(10, dtype=torch.float32)[:, None] * inv[None]
    plain = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    torch.testing.assert_close(table, plain)


def test_table_matches_hf_interleaved_rotary():
    tmod = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    tcfg = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfg = tcfg.Qwen4ExpTextConfig(
        hidden_size=512, num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    )
    rot = tmod.Qwen4ExpTextRotaryEmbedding(cfg)
    ids, grid = _prompt(2, (1, 4, 6), 3)
    pos3, _ = image_rope_positions(ids, IMG, grid, 2)
    cos, sin = rot(torch.zeros(1, ids.numel(), 8), pos3[:, None, :])
    table = mrope_cos_sin(pos3, 64, 1e7, (11, 11, 10))
    torch.testing.assert_close(table[:, :32], cos[0, :, :32], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(table[:, 32:], sin[0, :, :32], rtol=1e-5, atol=1e-5)
