"""A prompt with images may span several prefill chunks: each chunk takes exactly the soft
tokens of the placeholders it contains, and a placeholder-free chunk keeps an empty (not
None) slice so the request stays out of the shared prefix cache."""

from __future__ import annotations

import torch

from freetoken.scheduler.prefill import chunk_mm_embeds

# prompt: 3 text, 6 image rows, 2 text, 4 image rows, 1 text  (16 tokens, 10 soft tokens)
SLOTS = torch.tensor([False] * 3 + [True] * 6 + [False] * 2 + [True] * 4 + [False])
EMBEDS = torch.arange(10).float().unsqueeze(-1)  # row i -> value i


def test_chunks_take_their_placeholders_in_prompt_order():
    a = chunk_mm_embeds(EMBEDS, SLOTS, 0, 5)      # rows 0..4: placeholders 3,4 -> soft 0,1
    b = chunk_mm_embeds(EMBEDS, SLOTS, 5, 7)      # rows 5..11: placeholders 5..8 and 11 -> soft 2..5, 6
    c = chunk_mm_embeds(EMBEDS, SLOTS, 12, 4)     # rows 12..15: placeholders 12,13,14 -> soft 7,8,9
    assert a.squeeze(-1).tolist() == [0.0, 1.0]
    assert b.squeeze(-1).tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert c.squeeze(-1).tolist() == [7.0, 8.0, 9.0]
    assert torch.equal(torch.cat([a, b, c]), EMBEDS)


def test_placeholder_free_chunk_is_empty_not_none():
    head = chunk_mm_embeds(EMBEDS, SLOTS, 0, 3)
    assert head is not None and tuple(head.shape) == (0, 1)


def test_single_chunk_is_the_whole_thing():
    assert torch.equal(chunk_mm_embeds(EMBEDS, SLOTS, 0, 16), EMBEDS)
