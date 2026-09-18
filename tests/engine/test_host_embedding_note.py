"""--host-embedding on a model that does not honor it says so.

Only the Qwen3.5-MoE family builds a pinned-RAM embedding for the flag; on Qwen3.8-Flash-Next
it changed nothing (-0.01 GiB on a 12 GB single-card start) and nothing said so.
"""
from __future__ import annotations

from types import SimpleNamespace

from freetoken.engine.engine import _host_embedding_ignored


def _cfg(flag=True, first=True):
    return SimpleNamespace(host_embedding=flag, pp_is_first=first,
                           model_config=SimpleNamespace(model_type="qwen4_exp"))


def test_ignored_flag_is_named():
    note = _host_embedding_ignored(_cfg(), ("mtp.embed_tokens.",))
    assert note and "qwen4_exp" in note and "no effect" in note


def test_honored_flag_is_quiet():
    assert _host_embedding_ignored(_cfg(), ("model.embed_tokens.",)) is None


def test_off_or_not_the_table_rank_is_quiet():
    assert _host_embedding_ignored(_cfg(flag=False), ()) is None
    assert _host_embedding_ignored(_cfg(first=False), ()) is None
