"""--host-embedding on Qwen3.8-Flash-Next (qwen4_exp).

The flag used to change nothing here: the model built its embedding table in VRAM whatever the
config said (-0.01 GiB on a 12 GB single-card start, guides/23 §8). Now the model reads the mark
like the Qwen3.5-MoE family does: the table is a HostEmbedding, the engine materializes it in
pinned host memory, and --dense-quant fp8 leaves it alone (the host copy keeps the model dtype).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from .common import parsed_config, requires_cuda


def _build(**overrides):
    from freetoken.layers import set_rope_device
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    set_rope_device(torch.device("cpu"))  # rope tables cannot live on meta
    cfg = replace(parsed_config(), moe_strategy="offload", **overrides)
    with torch.device("meta"):
        return Qwen4ExpForCausalLM(cfg)


def test_the_flag_builds_a_host_table():
    from freetoken.layers import HostEmbedding

    model = _build(embed_host=True)
    assert isinstance(model.model.embed_tokens, HostEmbedding)
    assert "model.embed_tokens." in model.host_resident_prefixes
    assert "model.embed_tokens.weight_scale" not in model.state_dict()


def test_host_wins_over_the_fp8_table():
    # --dense-quant fp8 makes the VRAM table fp8; on the host the table stays in the model dtype
    from freetoken.layers import Fp8VocabParallelEmbedding, HostEmbedding

    assert isinstance(_build(embed_quant="fp8_pertensor").model.embed_tokens, Fp8VocabParallelEmbedding)
    assert isinstance(_build(embed_quant="fp8_pertensor", embed_host=True).model.embed_tokens, HostEmbedding)


def test_without_the_flag_nothing_changes():
    from freetoken.layers import HostEmbedding

    model = _build()
    assert not isinstance(model.model.embed_tokens, HostEmbedding)
    assert "model.embed_tokens." not in model.host_resident_prefixes


def test_the_engine_stays_quiet_once_the_table_moved():
    from types import SimpleNamespace

    from freetoken.engine.engine import _host_embedding_ignored

    model = _build(embed_host=True)
    cfg = SimpleNamespace(host_embedding=True, pp_is_first=True, model_config=SimpleNamespace(model_type="qwen4_exp"))
    assert _host_embedding_ignored(cfg, model.host_resident_prefixes) is None


@requires_cuda
def test_rows_gathered_from_the_host_match_the_table():
    # Flash-Next's row width (2560) in bf16, ids clamped the way embed_input_ids clamps image rows
    from freetoken.layers import HostEmbedding

    vocab, width = 5000, 2560
    emb = HostEmbedding(vocab, width)
    emb.weight = torch.randn(vocab, width, dtype=torch.bfloat16).pin_memory()
    ids = torch.tensor([0, 17, vocab - 1, 4242, 17], device="cuda")
    out = emb.forward(ids)
    assert torch.equal(out.cpu(), emb.weight[ids.cpu()])
