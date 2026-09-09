"""The MTP draft head's captured graphs and the attention metadata they read.

The chain graph (engine/spec_graph) is captured once and replayed per drafted token. Every
per-step input the captured kernels read has to live in a buffer that outlives the capture,
and that includes the qsa_sparse backend's per-forward scatter plan: the head is a single
attention layer that is not the model's first, so the backend's "rebuild the plan at the first
layer" rule never fires for it, and the plan the WARM run built would otherwise be baked into
the graph and then freed by the next step's metadata. The GPU symptom is a delayed
``an illegal memory access was encountered`` inside the replay.
"""

from types import SimpleNamespace

import torch

from freetoken.attention.qsa_sparse import QSASparseAttnBackend, QSASparseMetadata


def _md(kv_lens, *, is_decode=True):
    n = len(kv_lens)
    return QSASparseMetadata(
        is_decode=is_decode,
        last_indices=torch.zeros(n, dtype=torch.int32),
        qo_indptr_cpu=torch.arange(n + 1, dtype=torch.int32),
        kv_len_cpu=torch.tensor(kv_lens, dtype=torch.int32),
        cmp_rows=torch.zeros(n, dtype=torch.int32),
        ring_rows=torch.zeros(n, dtype=torch.int32),
    )


def _backend():
    """A backend instance without __init__ (which wants a CUDA pool); restaging only needs
    the two methods under test plus a recorded prepare_for_replay."""
    backend = object.__new__(QSASparseAttnBackend)
    backend.replayed = []
    backend.prepare_for_replay = lambda batch: backend.replayed.append(batch)
    return backend


def _chain_batch(device_len):
    proxy = SimpleNamespace(table_idx=3, extend_len=1, device_len=device_len, cached_len=device_len - 1)
    batch = SimpleNamespace(padded_reqs=[proxy], reqs=[proxy], attn_metadata=_md([70]), is_prefill=False)
    return batch, proxy


def test_reset_forward_plan_clears_the_scatter_plan():
    batch, _ = _chain_batch(71)
    _backend().reset_forward_plan(batch)
    assert batch.attn_metadata.cmp_rows is None and batch.attn_metadata.ring_rows is None


def test_reset_forward_plan_ignores_a_batch_without_qsa_metadata():
    batch = SimpleNamespace(attn_metadata=None, is_prefill=False)
    _backend().reset_forward_plan(batch)  # no raise


def test_restage_decode_keeps_the_captured_metadata_object():
    backend = _backend()
    batch, proxy = _chain_batch(71)
    md = batch.attn_metadata

    proxy.device_len = 5453
    backend.restage_decode(batch)

    # the same object: everything the captured kernels read hangs off it
    assert batch.attn_metadata is md
    assert int(md.kv_len_cpu[0]) == 5453
    assert md.cmp_rows is None and md.ring_rows is None  # this step's plan, not the capture's
    assert backend.replayed == [batch]


def test_default_restage_decode_rebuilds():
    """Backends that stage everything into their own capture buffers keep the old behaviour."""
    from freetoken.attention.base import BaseAttnBackend

    calls = []

    class Plain(BaseAttnBackend):
        forward = init_capture_graph = prepare_for_capture = None

        def prepare_metadata(self, batch):
            calls.append("prepare")

        def prepare_for_replay(self, batch):
            calls.append("replay")

    batch = SimpleNamespace(attn_metadata=None, is_prefill=False)
    Plain().restage_decode(batch)
    Plain().reset_forward_plan(batch)  # the default caches no plan: nothing to drop
    assert calls == ["prepare", "replay"]


def test_stage_chain_restages_instead_of_rebuilding():
    from freetoken.engine.spec_graph import SpecVerifyGraph

    backend = _backend()
    seen = []
    backend.restage_decode = lambda batch: seen.append(batch)
    backend.prepare_metadata = lambda batch: seen.append(("rebuilt", batch))

    mini, proxy = _chain_batch(71)
    sg = object.__new__(SpecVerifyGraph)
    sg.engine = SimpleNamespace(
        page_table=torch.arange(64, dtype=torch.int32).view(4, 16) * 7, attn_backend=backend
    )
    sg.chain_batch = mini
    for name in ("c_pos", "c_rope", "c_out_loc", "c_table", "c_ids"):
        setattr(sg, name, torch.zeros(1, dtype=torch.int32))

    req = SimpleNamespace(table_idx=2, linear_slot_idx=1)
    sg._stage_chain(req, position=9, token=1234, rope_delta=5)

    assert seen == [mini], "the chain step must restage, never rebuild the metadata object"
    assert proxy.table_idx == 2 and proxy.device_len == 10 and proxy.cached_len == 9
    assert int(sg.c_pos[0]) == 9 and int(sg.c_rope[0]) == 14 and int(sg.c_ids[0]) == 1234
    assert int(sg.c_out_loc[0]) == int(sg.engine.page_table[2, 9])
