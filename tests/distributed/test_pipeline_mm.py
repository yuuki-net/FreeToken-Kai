"""Image input under the pipeline engine: the encoder towers live on the first rank only (images
enter the residual stream at the embedding), while every rank serves the model they imply and
admits or refuses an image request alike. Runs without a checkpoint or a GPU."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.distributed.test_pipeline import _parsed_qwen4, _qwen4_hf_config


def _engine_config(monkeypatch, **kw):
    pytest.importorskip("freetoken.engine.config")
    from freetoken.engine import config as cfg_mod
    from freetoken.engine.config import EngineConfig
    from freetoken.models.register import EncoderSpec

    spec = SimpleNamespace(
        module="m", parse_config="p", encoders=(EncoderSpec("vision", "vision_config", ("image",)),)
    )

    def parse(hf):
        config = _parsed_qwen4()
        if getattr(hf, "vision_config", None) is not None:
            config = replace(config, vision_config=SimpleNamespace(deepstack_visual_indexes=()))
        return config

    hf = _qwen4_hf_config()
    hf.vision_config = SimpleNamespace(depth=2)
    monkeypatch.setattr(cfg_mod, "get_model_spec", lambda arch: spec)
    monkeypatch.setattr(cfg_mod, "_load_attr", lambda module, name: parse)
    monkeypatch.setattr(cfg_mod, "cached_load_hf_config", lambda path, overrides: hf)
    monkeypatch.setattr(cfg_mod, "checkpoint_quant_config", lambda *a, **k: None)
    return lambda **more: EngineConfig(model_path="x", dtype=torch.bfloat16, **{**kw, **more})


def test_only_the_first_pipeline_rank_builds_the_tower(monkeypatch):
    from freetoken.distributed import DistributedInfo

    make = _engine_config(monkeypatch)
    first = make(tp_info=DistributedInfo(0, 2), parallel="pp")
    later = make(tp_info=DistributedInfo(1, 2), parallel="pp")
    single = make(tp_info=DistributedInfo(0, 1))

    assert first.encodes_here and single.encodes_here
    assert not later.encodes_here
    # the later rank serves the same multimodal model -- it admits images and ropes them
    assert later.active_encoders == first.active_encoders and later.served_modalities == {"image"}
    assert first.model_config.vision_config is not None
    assert later.model_config.vision_config is None, "a later rank must not build the tower"
    assert later.model_config.rotary_config == first.model_config.rotary_config


def test_a_cpu_tower_gathers_on_the_engine_but_builds_nothing_there(monkeypatch):
    from freetoken.distributed import DistributedInfo
    from freetoken.mm.config import MultimodalConfig

    make = _engine_config(monkeypatch)
    cpu = make(tp_info=DistributedInfo(0, 1), mm=MultimodalConfig(encoder_weights="cpu"))
    gpu = make(tp_info=DistributedInfo(0, 1))
    # the tokenizer worker encodes; the engine still admits images and gathers their rows
    assert cpu.served_modalities == {"image"} and cpu.encodes_here
    assert not cpu.builds_tower and gpu.builds_tower
    assert cpu.model_config.vision_config is None and gpu.model_config.vision_config is not None
    assert cpu.model_config.rotary_config == gpu.model_config.rotary_config
    later = make(tp_info=DistributedInfo(1, 2), parallel="pp", mm=MultimodalConfig(encoder_weights="cpu"))
    assert not later.encodes_here and not later.builds_tower


def test_text_model_only_encodes_nowhere(monkeypatch):
    from freetoken.distributed import DistributedInfo
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig

    make = _engine_config(monkeypatch)
    off = make(tp_info=DistributedInfo(0, 1), mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)))
    assert not off.active_encoders and not off.encodes_here
    assert off.model_config.vision_config is None


def test_every_rank_answers_alike_whether_images_are_served():
    pytest.importorskip("freetoken.scheduler.scheduler")
    from freetoken.scheduler.scheduler import _serves_multimodal

    tower = (object(),)
    later_rank = SimpleNamespace(config=SimpleNamespace(active_encoders=tower), encoder_cache=None)
    first_rank = SimpleNamespace(config=SimpleNamespace(active_encoders=tower), encoder_cache=object())
    text_only = SimpleNamespace(config=SimpleNamespace(active_encoders=()), encoder_cache=None)
    assert _serves_multimodal(later_rank) and _serves_multimodal(first_rank)
    assert not _serves_multimodal(text_only)
    # a stub engine without a config falls back to whether it holds an encoder cache
    assert not _serves_multimodal(SimpleNamespace(encoder_cache=None))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the scheduler pins the staged rows")
def test_a_rank_without_the_encoder_cache_gathers_nothing_but_keeps_the_image_spans():
    """A later pipeline rank has no rows to gather (they arrive in the residual stream), but its
    attention still needs each image span's end (a bidirectional block attends within its span)."""
    pytest.importorskip("freetoken.scheduler.scheduler")
    from freetoken.message import MMItem
    from freetoken.scheduler.scheduler import Scheduler

    item = MMItem(modality="image", hash=3, pad_value=1_000_003, offsets=[[2, 6]],
                  precomputed_embeddings=torch.zeros(4, 8), model_specific_data={"grid_thw": [1, 4, 4]})
    req = SimpleNamespace(uid=1, mm_items=[item], cached_len=0, device_len=8, extend_len=8)
    later = SimpleNamespace(engine=SimpleNamespace(encoder_cache=None), device=torch.device("cpu"),
                            _bidirectional_mm=False, _warned_cut_image=False)
    batch = SimpleNamespace(padded_reqs=[req], mm_gather_plan=None, mm_encoder_jobs=None, mm_rows=None, mm_block_ends=None)
    Scheduler._gather_multimodal(later, batch)
    assert batch.mm_gather_plan is None and batch.mm_rows is None and batch.mm_encoder_jobs is None
    assert batch.mm_block_ends.tolist() == [0, 0, 6, 6, 6, 6, 0, 0]
