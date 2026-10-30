"""GGUF expert banks under --pp-size start at the rank's own first layer (``moe/expert_banks.py``).

The banks are read by global block number while ``num_moe_layers`` is rank-local, so the
non-first ranks must offset. Observed on a 2-way split of Qwen3.8-Flash-Next UD-IQ4_XS:
rank 1 loaded blk.0..23 for layers 24..47 and the first MoE call failed with
``ggml weight row is 1360 bytes, iq3_s K=2560 needs 1100`` (layer 0's IQ4_XS bank under
layer 24's IQ3_S type). A regression that drops the offset turns this red without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.distributed.info import PipelineInfo
from freetoken.moe.expert_banks import _gguf_first_layer


def test_single_rank_starts_after_the_dense_prefix():
    assert _gguf_first_layer(SimpleNamespace(first_k_dense_replace=0), pp=None) == 0
    assert _gguf_first_layer(SimpleNamespace(first_k_dense_replace=3), pp=None) == 3


def test_second_rank_starts_at_its_own_first_layer():
    pp0 = PipelineInfo(rank=0, size=2, start=0, end=24, num_layers=48)
    pp1 = PipelineInfo(rank=1, size=2, start=24, end=48, num_layers=48)
    cfg = SimpleNamespace(first_k_dense_replace=0)
    assert _gguf_first_layer(cfg, pp=pp0) == 0
    assert _gguf_first_layer(cfg, pp=pp1) == 24


def test_dense_prefix_and_pipeline_compose():
    # 3 dense layers first: bank index = layer - 3; rank 1 serves layers 20..40 -> blk.20 onward
    pp1 = PipelineInfo(rank=1, size=2, start=20, end=40, num_layers=40)
    assert _gguf_first_layer(SimpleNamespace(first_k_dense_replace=3), pp=pp1) == 20
    # a rank entirely inside the dense prefix has no banks; the window clamps at the prefix
    pp0 = PipelineInfo(rank=0, size=2, start=0, end=20, num_layers=40)
    assert _gguf_first_layer(SimpleNamespace(first_k_dense_replace=3), pp=pp0) == 3
