"""--spec-mtp with a GGUF checkpoint is refused while the arguments are parsed (FreeToken-Kai#6).

No GGUF reader yields an MTP head, so the start used to read the config and then die in the
weight load on ``iter_gguf_weights() got an unexpected keyword argument 'include_mtp'``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from freetoken.server.args import parse_args


def _parse(model: str, *argv: str):
    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        return parse_args(["--model", model, *argv])[0]


def test_spec_mtp_on_a_gguf_file_is_refused(tmp_path, capsys):
    gguf = tmp_path / "Model-UD-IQ4_XS-00001-of-00003.gguf"
    gguf.write_bytes(b"")
    with pytest.raises(SystemExit):
        _parse(str(gguf), "--spec-mtp", "3")
    assert "--spec-mtp needs the draft head in the checkpoint" in capsys.readouterr().err


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
    assert "--spec-mtp needs the draft head" not in capsys.readouterr().err
