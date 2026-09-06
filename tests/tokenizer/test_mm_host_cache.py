"""The host encoder encodes per image and remembers identical images."""

from __future__ import annotations

import torch

from freetoken.tokenizer.mm_host import _QwenHostEncoder


class FakeTower:
    spatial_merge_size = 2

    def __init__(self):
        self.calls = []

    def encode(self, patches, grid):
        self.calls.append((tuple(patches.shape), grid.tolist()))
        n = int(grid.prod()) // 4
        # a fingerprint of the input so a wrong slice would show up in the output
        return torch.full((n, 4), float(patches.float().mean()), dtype=torch.bfloat16)


def _mm(*imgs):
    pvs, grids = [], []
    for value, grid in imgs:
        n = grid[0] * grid[1] * grid[2]
        pvs.append(torch.full((n, 8), value, dtype=torch.float16))
        grids.append(grid)
    return {"pixel_values": torch.cat(pvs), "image_grid_thw": torch.tensor(grids)}


def test_images_are_encoded_separately_and_in_order():
    tower = FakeTower()
    enc = _QwenHostEncoder(tower, cache_entries=8)
    out = enc.encode(_mm((1.0, [1, 4, 4]), (2.0, [1, 2, 4])))
    assert tower.calls == [((16, 8), [[1, 4, 4]]), ((8, 8), [[1, 2, 4]])]
    assert out["mm_embeds"].float()[:, 0].tolist() == [1.0] * 4 + [2.0] * 2
    assert out["image_grid_thw"].tolist() == [[1, 4, 4], [1, 2, 4]]
    assert int(out["merge_size"]) == 2


def test_identical_image_is_encoded_once():
    tower = FakeTower()
    enc = _QwenHostEncoder(tower, cache_entries=8)
    enc.encode(_mm((1.0, [1, 4, 4])))
    enc.encode(_mm((1.0, [1, 4, 4]), (3.0, [1, 4, 4])))   # same first image, a new second one
    assert len(tower.calls) == 2 and (enc.hits, enc.misses) == (1, 2)


def test_cache_is_bounded_lru():
    tower = FakeTower()
    enc = _QwenHostEncoder(tower, cache_entries=1)
    enc.encode(_mm((1.0, [1, 4, 4])))
    enc.encode(_mm((2.0, [1, 4, 4])))
    enc.encode(_mm((1.0, [1, 4, 4])))   # evicted by the second image -> encoded again
    assert len(tower.calls) == 3


def test_cache_can_be_disabled():
    tower = FakeTower()
    enc = _QwenHostEncoder(tower, cache_entries=0)
    enc.encode(_mm((1.0, [1, 4, 4])))
    enc.encode(_mm((1.0, [1, 4, 4])))
    assert len(tower.calls) == 2
