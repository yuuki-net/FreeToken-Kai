"""--spec-mtp with a GGUF checkpoint while the arguments are parsed (FreeToken-Kai#6).

No GGUF reader yields an MTP head, so the start used to read the config and then die in the
weight load on ``iter_gguf_weights() got an unexpected keyword argument 'include_mtp'``. The
Flash-Next GGUFs ship the head as a separate file, read with --spec-mtp-gguf; without it the
start is refused here, and so is a file that holds no head.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from freetoken.models.gguf.dequant import GGML_F32
from freetoken.models.gguf.rawreader import write_gguf
from freetoken.server.args import parse_args


def _parse(model: str, *argv: str):
    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        return parse_args(["--model", model, *argv])[0]


def _head(path, block=48):
    write_gguf(str(path), [(f"blk.{block}.nextn.eh_proj.weight", (4, 2), GGML_F32, bytes(32))],
               {"general.architecture": "qwen4exp", "qwen4exp.nextn_predict_layers": 1})
    return str(path)


def test_spec_mtp_on_a_gguf_file_without_a_head_file_is_refused(tmp_path, capsys):
    gguf = tmp_path / "Model-UD-IQ4_XS-00001-of-00003.gguf"
    gguf.write_bytes(b"")
    with pytest.raises(SystemExit):
        _parse(str(gguf), "--spec-mtp", "3")
    err = capsys.readouterr().err
    assert "--spec-mtp needs a draft head" in err and "--spec-mtp-gguf" in err


def test_spec_mtp_off_on_a_gguf_file_is_not_refused_here(tmp_path, capsys):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    try:
        _parse(str(gguf))
    except SystemExit:
        pass  # whatever else an empty file trips over is not this check
    assert "--spec-mtp" not in capsys.readouterr().err


def test_spec_mtp_on_a_directory_is_not_refused_here(capsys):
    try:
        _parse("/models/anon", "--spec-mtp", "3")
    except SystemExit:
        pass
    assert "--spec-mtp needs a draft head" not in capsys.readouterr().err


def test_a_head_file_needs_spec_mtp(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse("/models/anon", "--spec-mtp-gguf", _head(tmp_path / "mtp.gguf"))
    assert "add --spec-mtp K" in capsys.readouterr().err


def test_a_file_without_a_head_is_refused(tmp_path, capsys):
    other = tmp_path / "experts.gguf"
    write_gguf(str(other), [("blk.0.ffn_gate_inp.weight", (4, 2), GGML_F32, bytes(32))], {"general.architecture": "qwen4exp"})
    with pytest.raises(SystemExit):
        _parse("/models/anon", "--spec-mtp", "3", "--spec-mtp-gguf", str(other))
    assert "holds no MTP head" in capsys.readouterr().err


def test_a_gguf_model_of_another_family_is_refused(tmp_path, capsys):
    model = tmp_path / "qwen3moe.gguf"
    write_gguf(str(model), [], {"general.architecture": "qwen3moe"})
    with pytest.raises(SystemExit):
        _parse(str(model), "--spec-mtp", "3", "--spec-mtp-gguf", _head(tmp_path / "mtp.gguf"))
    assert "'qwen3moe' GGUF" in capsys.readouterr().err


def test_a_flash_next_gguf_with_its_head_file_passes_this_check(tmp_path, capsys):
    model = tmp_path / "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
    write_gguf(str(model), [], {"general.architecture": "qwen4exp"})
    head = _head(tmp_path / "MTP" / "mtp.gguf") if (tmp_path / "MTP").mkdir() is None else None
    try:
        args = _parse(str(model), "--spec-mtp", "3", "--spec-mtp-gguf", head)
    except SystemExit:
        args = None  # a metadata-only model trips over later checks; not this one
    err = capsys.readouterr().err
    assert "--spec-mtp" not in err
    if args is not None:
        assert args.spec_mtp_gguf == head
