"""--spec-mtp on a checkpoint whose draft-head experts are laid out per expert.

The draft head reads its experts as the two stacked bf16 tensors of the RadixArk
Qwen3.8-Flash-Next-NVFP4 release. NVIDIA's release stores them one per expert in 128x128
block-fp8, which the reader skips as routed experts; the load then stopped on an assert saying
the experts were "missing from the checkpoint" -- they are there, in a layout not read yet.
Found by a fork that ran Kai on the NVIDIA release. The start now stops with a message that says
which layout it found and what to do (--spec-mtp 0). CPU-only: nothing is loaded.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from freetoken.engine.engine import Engine


def _checkpoint(tmp_path, keys):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001.safetensors" for k in keys}}), encoding="utf-8"
    )
    return str(tmp_path)


def _start(model_path, raw=None):
    head = SimpleNamespace(_mtp_raw=raw or {}, config=SimpleNamespace(model_path=model_path))
    with pytest.raises(ValueError) as exc:
        Engine._quantize_mtp_experts(head)
    return str(exc.value)


def test_nvidia_per_expert_block_fp8_is_named_as_such(tmp_path):
    keys = ["mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight"]
    for e in range(4):
        for p in ("gate_proj", "up_proj", "down_proj"):
            keys += [f"mtp.layers.0.mlp.experts.{e}.{p}.weight",
                     f"mtp.layers.0.mlp.experts.{e}.{p}.weight_scale_inv"]
    msg = _start(_checkpoint(tmp_path, keys))
    assert "one per expert in block-fp8" in msg
    assert "mtp.layers.0.mlp.experts.0." in msg
    assert "--spec-mtp 0" in msg
    assert "missing" not in msg


def test_no_draft_head_experts_at_all(tmp_path):
    msg = _start(_checkpoint(tmp_path, ["mtp.fc.weight", "model.embed_tokens.weight"]))
    assert "no MTP draft-head experts" in msg and "--spec-mtp 0" in msg


def test_half_a_stacked_pair_is_still_refused(tmp_path):
    msg = _start(_checkpoint(tmp_path, []), raw={"gate_up_proj": object()})
    assert "gate_up_proj" in msg


def test_a_directory_without_an_index_still_gets_a_message(tmp_path):
    msg = _start(str(tmp_path / "nowhere"))
    assert "--spec-mtp 0" in msg
