"""--host-embedding: the model builds the host-resident table, the engine materializes it on
the host, and the config flag plumbs through. No GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .test_spec_mtp import _build, _parsed, _toy_hf_config


def test_materialize_places_host_prefixes_on_cpu():
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict

    model_state = {
        "model.embed_tokens.weight": torch.empty(2, 2, device="meta", dtype=torch.bfloat16),
        "x.weight": torch.empty(2, 2, device="meta"),
    }
    out = _materialize_loaded_weight_state_dict(
        model_state,
        [("model.embed_tokens.weight", torch.ones(2, 2)), ("x.weight", torch.ones(2, 2))],
        device=torch.device("cpu"),
        host_prefixes=("model.embed_tokens.",),
    )
    assert out["model.embed_tokens.weight"].device.type == "cpu"
    assert out["model.embed_tokens.weight"].dtype == torch.bfloat16
    assert torch.equal(out["model.embed_tokens.weight"].float(), torch.ones(2, 2))


def test_engine_config_flag_reaches_the_model_config(monkeypatch):
    pytest.importorskip("freetoken.engine.config")
    from freetoken.distributed import DistributedInfo
    from freetoken.engine import config as cfg_mod
    from freetoken.engine.config import EngineConfig

    monkeypatch.setattr(cfg_mod, "get_model_spec", lambda arch: SimpleNamespace(module="m", parse_config="p"))
    monkeypatch.setattr(cfg_mod, "_load_attr", lambda module, name: (lambda hf: _parsed(4)))
    monkeypatch.setattr(cfg_mod, "cached_load_hf_config", lambda path: _toy_hf_config(4))
    monkeypatch.setattr(cfg_mod, "checkpoint_quant_config", lambda *a, **k: None)
    kw = dict(model_path="x", dtype=torch.bfloat16, tp_info=DistributedInfo(0, 1))
    assert EngineConfig(**kw).model_config.embed_host is False
    assert EngineConfig(host_embedding=True, **kw).model_config.embed_host is True


def test_model_builds_host_embedding(monkeypatch):
    pytest.importorskip("freetoken.models.qwen3_5_moe.model")
    from dataclasses import replace

    import freetoken.distributed.info as info_mod
    from freetoken.layers import HostEmbedding

    monkeypatch.setattr(info_mod, "_TP_INFO", None)
    info_mod.set_tp_info(0, 1)
    model = _build(replace(_parsed(4), moe_strategy="offload", embed_host=True))
    assert isinstance(model.model.embed_tokens, HostEmbedding)
    assert model.host_resident_prefixes == ("model.embed_tokens.",)
    keys = model.state_dict()
    assert keys["model.embed_tokens.weight"].shape == (1000, 64)
    plain = _build(replace(_parsed(4), moe_strategy="offload"))
    assert plain.host_resident_prefixes == ()
