"""The console's recommended flags (webui/recommend.py) for the hosts this fork is measured on.

Each case fixes the host (GPUs, RAM, cores) and a checkpoint's config.json plus weight size, and
pins what comes out. A rule change that moves one of these has to be a decision, not an accident."""

from __future__ import annotations

import json

import pytest

from freetoken.webui import recommend as rec

GiB = 1 << 30


def _model(tmp_path, name, *, weight_gib, **config):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    with open(d / "model.safetensors", "wb") as fh:  # sparse: only the size is read
        fh.truncate(int(weight_gib * GiB))
    return str(d)


def _host(monkeypatch, gpus, ram_gib, cores):
    monkeypatch.setattr(rec, "gpus", lambda: [
        {"index": i, "name": name, "total_bytes": int(vram * GiB), "free_bytes": int(vram * GiB), "compute_cap": cap}
        for i, (name, vram, cap) in enumerate(gpus)
    ])
    monkeypatch.setattr(rec, "host_memory", lambda: {"total": int(ram_gib * GiB), "available": int(ram_gib * GiB * 0.6)})
    monkeypatch.setattr(rec, "physical_cores", lambda: cores)


def _flags(result) -> dict:
    return {n["flag"]: n["value"] for n in result["notes"]}


GDN_MOE = dict(
    model_type="qwen3_5_moe", num_hidden_layers=40, num_experts=256, hidden_size=2048, moe_intermediate_size=512,
    num_key_value_heads=2, head_dim=256, max_position_embeddings=262144, vision_config={"depth": 27},
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 10,
    quantization_config={"quant_method": "modelopt"},
)


def test_rtx_2060_small_moe(tmp_path, monkeypatch):
    _host(monkeypatch, [("NVIDIA GeForce RTX 2060", 6, 7.5)], ram_gib=24, cores=12)
    f = _flags(rec.recommend(_model(tmp_path, "Ornith", weight_gib=20, **GDN_MOE)))
    assert f["--dtype"] == "float16"  # Turing has no bf16
    assert f["--moe-strategy"] == "hybrid" and f["--moe-cpu-layers"] == "auto"
    assert f["--moe-cpu-threads"] == "8"
    assert "--moe-cache-auto" in f and "--moe-bank-ram" not in f  # 20 GiB of weights fit in 24 GiB of RAM
    assert f["--kv-cache-dtype"] == "q8_0"
    assert f["--kv-reserve-tokens"] == f["--max-seq-len-override"] == "16384"
    assert "--host-embedding" in f and f["--mm-encoder-weights"] == "cpu"
    assert "--pp-size" not in f and "--gpu" not in f


def test_two_rtx_3060_big_moe_over_ram(tmp_path, monkeypatch):
    _host(monkeypatch, [("NVIDIA GeForce RTX 3060", 12, 8.6)] * 2, ram_gib=64, cores=16)
    big = dict(GDN_MOE, model_type="qwen3_5_moe", num_hidden_layers=48, num_experts=512, hidden_size=4096,
               moe_intermediate_size=1024, vision_config=None)
    f = _flags(rec.recommend(_model(tmp_path, "Flash-Next", weight_gib=70, **big)))
    assert "--dtype" not in f  # Ampere has bf16
    assert "--moe-strategy" not in f  # 12 GiB is not the hybrid tier
    assert f["--moe-bank-ram"] == "44G"  # 70 GiB does not fit 64 GiB: 70% of RAM
    assert f["--kv-reserve-tokens"] == "65536"
    assert "--host-embedding" not in f and "--mm-encoder-weights" not in f


def test_resident_weights_over_one_card_split_across_both(tmp_path, monkeypatch):
    _host(monkeypatch, [("NVIDIA GeForce RTX 3060", 12, 8.6)] * 2, ram_gib=128, cores=16)
    dense_heavy = dict(GDN_MOE, num_experts=8, vision_config=None)  # experts small, the rest large
    f = _flags(rec.recommend(_model(tmp_path, "Heavy", weight_gib=40, **dense_heavy)))
    assert f["--pp-size"] == "2" and f["--gpu"] == "0,1"


def test_small_model_on_two_cards_uses_one(tmp_path, monkeypatch):
    _host(monkeypatch, [("NVIDIA GeForce RTX 3060", 12, 8.6)] * 2, ram_gib=64, cores=16)
    f = _flags(rec.recommend(_model(tmp_path, "Small", weight_gib=4, **GDN_MOE)))
    assert f["--gpu"] == "0" and "--pp-size" not in f


def test_dense_model_on_a_large_card_keeps_its_own_limit(tmp_path, monkeypatch):
    _host(monkeypatch, [("NVIDIA RTX 6000", 48, 8.9)], ram_gib=128, cores=32)
    dense = dict(model_type="llama", num_hidden_layers=32, hidden_size=4096, num_attention_heads=32,
                 num_key_value_heads=8, max_position_embeddings=32768)
    f = _flags(rec.recommend(_model(tmp_path, "Dense", weight_gib=16, **dense)))
    assert f["--kv-reserve-tokens"] == "32768"  # the 131072 tier, capped by the checkpoint
    for moe_flag in ("--moe-cache-auto", "--moe-strategy", "--kv-cache-dtype", "--moe-bank-ram"):
        assert moe_flag not in f


@pytest.mark.parametrize("vram, tier", [(6, "16384"), (8, "16384"), (12, "65536"), (16, "65536"), (24, "131072")])
def test_context_tier_follows_the_smallest_card(tmp_path, monkeypatch, vram, tier):
    _host(monkeypatch, [("GPU", vram, 8.6)], ram_gib=64, cores=8)
    f = _flags(rec.recommend(_model(tmp_path, f"M{vram}", weight_gib=4, **GDN_MOE)))
    assert f["--kv-reserve-tokens"] == tier


def test_no_gpu_still_answers(tmp_path, monkeypatch):
    _host(monkeypatch, [], ram_gib=32, cores=4)
    result = rec.recommend(_model(tmp_path, "NoGpu", weight_gib=4, **GDN_MOE))
    f = _flags(result)
    assert f["--moe-cpu-threads"] == "2" and "--dtype" not in f
    assert result["flags"][:2] == ["--moe-strategy", "hybrid"]


def test_every_note_says_why(tmp_path, monkeypatch):
    _host(monkeypatch, [("NVIDIA GeForce RTX 2060", 6, 7.5)], ram_gib=24, cores=12)
    result = rec.recommend(_model(tmp_path, "Ornith", weight_gib=20, **GDN_MOE))
    assert all(n["why"] and n["why_en"] for n in result["notes"])
    # flags is the same list, spelled for the command line
    spelled = []
    for n in result["notes"]:
        spelled += [n["flag"]] + ([n["value"]] if n["value"] is not None else [])
    assert spelled == result["flags"]


def test_kv_counts_only_full_attention_layers_of_a_gdn_hybrid():
    assert rec.kv_layers(GDN_MOE) == 10
    assert rec.kv_layers({"num_hidden_layers": 32, "num_layers": 32}) == 32


def test_unknown_model_name_is_refused(tmp_path, monkeypatch):
    _host(monkeypatch, [("GPU", 12, 8.6)], ram_gib=64, cores=8)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ValueError):
        rec.recommend("no-such-model-folder")
