"""--pp-prefill-group (engine/prefill_group.py): which chunks the last pipeline rank holds, when it
runs them, that running them layer by layer is the same arithmetic as one chunk per forward, and
that an offloaded layer's prefetched bank serves every chunk of a group."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.prefill_group import DeferredChunk, PrefillGroup, groupable, unusable_reason

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _batch(rows=4096, *, uid=7, table_idx=2, prefill=True, pp_no_tokens=True, size=1, verify=False, mm=None):
    req = SimpleNamespace(uid=uid, table_idx=table_idx)
    return SimpleNamespace(
        is_prefill=prefill, spec_verify=verify, pp_no_tokens=pp_no_tokens, size=size,
        reqs=[req] * size, mm_gather_plan=mm, input_ids=torch.zeros(rows, dtype=torch.int32),
    )


# ---- which chunks ---------------------------------------------------------------------------


def test_only_one_requests_non_final_bank_streaming_chunks_are_held():
    kw = dict(cpu_prefill_max_tokens=256)
    assert groupable(_batch(), **kw)
    assert not groupable(_batch(pp_no_tokens=False), **kw)  # the prompt's last chunk wants tokens
    assert not groupable(_batch(prefill=False), **kw)
    assert not groupable(_batch(verify=True), **kw)
    assert not groupable(_batch(size=2), **kw)
    assert not groupable(_batch(mm=[("item",)]), **kw)
    assert not groupable(_batch(rows=256), **kw)  # takes the CPU executor: no bank to share
    assert groupable(_batch(rows=257), **kw)


def test_the_group_holds_one_request_up_to_its_size():
    group = PrefillGroup(3)
    chunk = lambda b: DeferredChunk(b, torch.zeros(1), 1, None)  # noqa: E731
    assert group.accepts(_batch())
    assert not group.add(chunk(_batch()))
    assert group.accepts(_batch()) and not group.accepts(_batch(uid=8)) and not group.accepts(_batch(table_idx=3))
    assert not group.add(chunk(_batch()))
    assert group.add(chunk(_batch()))
    assert len(group.take()) == 3 and len(group) == 0 and group.accepts(_batch(uid=8))


def test_where_it_cannot_run():
    last = SimpleNamespace(is_first=False, is_last=True, info=SimpleNamespace(size=2))
    model = SimpleNamespace(forward_prefill_group=lambda chunks: None)
    cache = SimpleNamespace(prefill_overlap=True)
    ok = dict(size=3, pp_comm=last, max_running_req=1, model=model, offload_cache=cache)
    assert unusable_reason(**ok) is None
    assert unusable_reason(**{**ok, "size": 1}) == "off"
    assert "last pipeline rank" in unusable_reason(**{**ok, "pp_comm": None})
    assert "last pipeline rank" in unusable_reason(**{**ok, "pp_comm": SimpleNamespace(is_first=True, is_last=False, info=last.info)})
    assert "two pipeline ranks" in unusable_reason(**{**ok, "pp_comm": SimpleNamespace(is_first=False, is_last=True, info=SimpleNamespace(size=3))})
    assert "--max-running-req 1" in unusable_reason(**{**ok, "max_running_req": 4})
    assert "no grouped prefill" in unusable_reason(**{**ok, "model": SimpleNamespace()})
    assert "offloaded" in unusable_reason(**{**ok, "offload_cache": None})
    assert "offloaded" in unusable_reason(**{**ok, "offload_cache": SimpleNamespace(prefill_overlap=False)})


def test_the_flag_needs_two_ranks_and_one_running_request(monkeypatch, capsys):
    from freetoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {"architectures": ["Qwen3_5MoeForCausalLM"], "model_type": "qwen3_5_moe"}

    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda _p: _Config())
    base = ["--model", "/models/anon"]
    with pytest.raises(SystemExit):
        parse_args(base + ["--pp-prefill-group", "3"])
    with pytest.raises(SystemExit):
        parse_args(base + ["--pp-size", "2", "--pp-prefill-group", "3"])  # --max-running-req defaults to 4
    with pytest.raises(SystemExit):
        parse_args(base + ["--pp-prefill-group", "0"])
    args, _ = parse_args(base + ["--pp-size", "2", "--max-running-req", "1", "--pp-prefill-group", "3"])
    assert args.pp_prefill_group == 3
    assert parse_args(base)[0].pp_prefill_group == 1


# ---- when they run ----------------------------------------------------------------------------


class _Model:
    def __init__(self):
        self.groups, self.last = [], []

    def forward_prefill_group(self, chunks):
        self.groups.append([c.batch for c in chunks])
        return [c.hidden + 1 for c in chunks]

    def set_last_hidden(self, hidden):
        self.last.append(hidden)

    def prefill_group_next_token(self, hidden):
        return 900 + int(hidden[0, 0])


def _engine(size=2, spec_k=0):
    from freetoken.engine.engine import Engine

    eng = Engine.__new__(Engine)
    eng._prefill_group = PrefillGroup(size)
    eng.model = _Model()
    eng.spec_k = spec_k
    eng.cpu_moe_executor = None
    eng.device = torch.device("cuda")
    eng.stream = torch.cuda.current_stream()
    eng.ctx = SimpleNamespace(prefill_mixer_pieces=1, attn_backend=None, linear_state_pool=None)
    eng.received = []

    def recv_hidden(rows):
        eng.received.append(rows)
        return torch.full((rows, 3), float(len(eng.received)), device="cuda")

    eng.pp_comm = SimpleNamespace(recv_hidden=recv_hidden)
    eng.drafts = []
    eng._mtp_draft = lambda batch, rows, *, row, next_token, draft: eng.drafts.append((batch, rows, row, next_token, draft))
    return eng


def _chunk_batch(rows, uid=7, first=0):
    b = _batch(rows, uid=uid)
    b.input_ids = torch.full((rows,), first, dtype=torch.int32)
    req = SimpleNamespace(uid=uid, table_idx=2, completed=0, cached_len=0, device_len=rows)

    def complete_one():
        req.completed += 1
        req.cached_len, req.device_len = req.device_len, req.device_len + 1

    req.complete_one = complete_one
    b.reqs = [req]
    b.spec_next_tail = None  # a ChunkedReq's ids end at its chunk: the scheduler never knows it
    return b


@cuda
def test_chunks_are_held_until_the_group_is_full():
    eng = _engine(size=2)
    first, second, third = _chunk_batch(4096), _chunk_batch(4096), _chunk_batch(2048)
    out = eng._maybe_defer_prefill(first)
    assert out is not None and eng.model.groups == [] and eng.received == [4096]
    assert first.reqs[0].completed == 1  # advanced at its step, as a forward would
    assert torch.equal(out.next_tokens_cpu, torch.zeros(1, dtype=torch.int32)) and out.spec is None
    out.copy_done_event.synchronize()
    eng._maybe_defer_prefill(second)
    assert eng.model.groups == [[first, second]] and not eng.holds_deferred_prefill
    eng._maybe_defer_prefill(third)
    assert eng.holds_deferred_prefill and len(eng.model.groups) == 1


@cuda
def test_anything_else_runs_the_held_chunks_first():
    eng = _engine(size=4)
    a = _chunk_batch(4096)
    eng._maybe_defer_prefill(a)
    final = _chunk_batch(1000)
    final.pp_no_tokens = False
    assert eng._maybe_defer_prefill(final) is None  # the prompt's last chunk runs as a forward
    assert eng.model.groups == [[a]] and not eng.holds_deferred_prefill

    eng._maybe_defer_prefill(_chunk_batch(4096))
    other = _chunk_batch(4096, uid=9)
    assert eng._maybe_defer_prefill(other) is not None  # held, but only after the other request ran
    assert len(eng.model.groups) == 2 and eng.holds_deferred_prefill


@cuda
def test_the_draft_head_extends_its_kv_over_every_held_chunk_in_order():
    eng = _engine(size=2, spec_k=5)
    eng.model.mtp = object()  # the rank owns the draft head
    a, b = _chunk_batch(4096, first=11), _chunk_batch(3072, first=12)
    out = eng._maybe_defer_prefill(a)
    assert out.spec.accepted == [0] and out.spec.drafts == []
    eng._maybe_defer_prefill(b)
    # a's successor is b's first prompt token; b's is the target's greedy token (b's hidden is 2 + 1)
    assert [(d[0], d[1], d[2], d[3], d[4]) for d in eng.drafts] == [(a, 4096, 4095, 12, False), (b, 3072, 3071, 903, False)]
    assert [float(h[0, 0]) for h in eng.model.last] == [2.0, 3.0]  # each chunk's own final hidden


@cuda
def test_the_scheduler_runs_held_chunks_before_freeing_or_restoring():
    from freetoken.scheduler.scheduler import Scheduler

    calls = []
    engine = SimpleNamespace(
        holds_deferred_prefill=True, stream=torch.cuda.Stream(),
        flush_deferred_prefill=lambda: calls.append("flush"),
        linear_state_pool=SimpleNamespace(copy_from=lambda src, dst: calls.append(("restore", src, dst))),
    )
    sched = Scheduler.__new__(Scheduler)
    sched.engine = engine
    sched.stream = torch.cuda.current_stream()
    sched.cache_manager = SimpleNamespace(cache_req=lambda req, finished: calls.append(("cache_req", finished)))
    sched.table_manager = SimpleNamespace(free=lambda idx: calls.append(("free", idx)))

    req = SimpleNamespace(table_idx=3)
    sched._free_req_resources(req)
    assert calls == ["flush", ("cache_req", True), ("free", 3)]

    calls.clear()
    hit = SimpleNamespace(mamba_restore_src=5, linear_slot_idx=6)
    sched._restore_linear_states(SimpleNamespace(is_prefill=True, reqs=[hit]))
    assert calls == ["flush", ("restore", 5, 6)]

    calls.clear()
    miss = SimpleNamespace(mamba_restore_src=None, linear_slot_idx=6)
    sched._restore_linear_states(SimpleNamespace(is_prefill=True, reqs=[miss]))
    assert calls == []  # no restore, nothing to protect

    engine.holds_deferred_prefill = False
    sched._free_req_resources(SimpleNamespace(table_idx=4))
    assert calls == [("cache_req", True), ("free", 4)]


# ---- same arithmetic --------------------------------------------------------------------------


@cuda
@pytest.mark.parametrize("pieces", [1, 2])
def test_layer_by_layer_equals_one_chunk_per_forward(pieces):
    """A GDN layer and a QSA layer (hyper-connections, a real fused MoE) over three prefill chunks
    of one prompt: grouped layer by layer, the chunks come out bit for bit as they do one chunk
    per forward, and so do the states the next chunk would resume from."""
    from freetoken.models.prefill_pieces import plan_prefill_pieces
    from freetoken.models.qwen4_exp.model import Qwen4ExpDecoderLayer, Qwen4ExpForCausalLM
    from freetoken.utils.torch_utils import torch_dtype
    from tests.models.qwen4_exp.common import Fixture, fill_weights, parsed_config
    from tests.models.qwen4_exp.test_prefill_pieces import _with_state_pool

    config = parsed_config()
    fixture = Fixture(config, num_pages=512)
    pool = _with_state_pool(fixture, config)
    layers = {}
    for layer_id, seed in ((0, 41), (3, 42)):
        with torch.device(fixture.device), torch_dtype(fixture.dtype):
            layer = Qwen4ExpDecoderLayer(config, layer_id)
        fill_weights(layer, seed, fixture.device)
        if layer._is_linear:
            with torch.no_grad():
                layer.linear_attn.A_log.uniform_(0.0, 2.0)
                layer.linear_attn.dt_bias.uniform_(-1.0, 1.0)
        layers[layer_id] = layer
    cuts = [0, 1024, 2560, 3000]  # the last chunk crosses the 2048 index budget
    width = config.qwen4_args.hc_count * config.hidden_size
    R0 = torch.randn(cuts[-1], width, device=fixture.device, dtype=fixture.dtype) * 0.5

    def chunk_batch(table_idx, start, end):
        req = fixture.req(table_idx, start, end)
        req.linear_slot_idx = None
        req.mamba_ping_pong = None
        req.uid = table_idx
        batch = fixture.batch([req], "prefill")
        batch.input_ids = torch.zeros(end - start, dtype=torch.int32, device=fixture.device)
        batch.fla_metadata = None
        batch.spec_verify = False
        return batch

    def plan(batch):
        return plan_prefill_pieces(batch, pieces, fixture.backend, fixture.device, pool) if pieces > 1 else None

    # one chunk per forward (the reference), request in table row / state slot 1
    ref = []
    for start, end in zip(cuts, cuts[1:]):
        batch = chunk_batch(1, start, end)
        with fixture.ctx.forward_batch(batch):
            p = plan(batch)
            R = R0[start:end]
            for layer_id in (0, 3):
                R = layers[layer_id].forward_pieces(R, batch, p, fixture.ctx) if p else layers[layer_id].forward(R, batch)
        ref.append(R)

    # the same chunks grouped, request in row / slot 2: metadata and pieces built chunk by chunk
    # as the engine defers them, the layers run afterwards
    chunks = []
    for start, end in zip(cuts, cuts[1:]):
        batch = chunk_batch(2, start, end)
        with fixture.ctx.forward_batch(batch):
            p = plan(batch)
        chunks.append(DeferredChunk(batch, R0[start:end], end - start, p))
    stand_in = SimpleNamespace(model=SimpleNamespace(
        pp_first=False, _ple=(), _local_ids=(0, 3), layers=SimpleNamespace(op_list=layers),
    ))
    got = Qwen4ExpForCausalLM.forward_prefill_group(stand_in, chunks)
    assert not fixture.ctx.prefill_group_continuation
    for k, (g, r) in enumerate(zip(got, ref)):
        assert torch.equal(g, r), f"chunk {k}"
    li = pool.local_index(0)
    assert torch.equal(pool.recurrent_states[li][2], pool.recurrent_states[li][1])
    assert torch.equal(pool.conv_states[li][2], pool.conv_states[li][1])


@cuda
def test_a_prefetched_bank_serves_every_chunk_of_a_group():
    """The offload cache's prefill overlap buffers under the grouped choreography: layer 0 fences
    the copy stream once per group, every other call finds its layer already in its buffer, and
    each chunk's experts match the dequantized reference."""
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4
    from freetoken.moe.offload_cache import OffloadMoeCache
    from tests.moe.test_nvfp4_backends import E, H, L, TOPK, _assert_close, _make_native_sources, _ref_moe

    device = torch.device("cuda")
    sources = _make_native_sources(device, seed=51)
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=2 * E, device=device, quant_format="nvfp4",
        prefill_overlap=True,
    )
    cache.set_bank_sources({name: sources[name] for name in cache.bank_schema})
    cache.reset()
    torch.manual_seed(52)
    chunks = [
        (torch.randn(M, H, dtype=torch.bfloat16, device=device) / 4,
         torch.rand(M, TOPK, dtype=torch.float32, device=device),
         torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=device))
        for M in (24, 16, 32)
    ]
    copies = []
    original = cache._invalidate_prefill_buffer
    cache._invalidate_prefill_buffer = lambda buffer_id: (copies.append(buffer_id), original(buffer_id))[1]
    for layer_id in range(L):
        for k, (hidden, weights, ids) in enumerate(chunks):
            if layer_id == 0 and k == 0:
                cache.begin_prefill()
            cache.prefetch_prefill_layer(layer_id)
            cache.prefetch_prefill_layer(layer_id + 1)
            views = cache.wait_prefill_layer(layer_id)
            out = fused_experts_nvfp4(hidden.clone(), *views, weights, ids, E, "silu", False)
            cache.finish_prefill_prefetch()
            cache.release_prefill_layer(layer_id)
            _assert_close(out, _ref_moe(sources, layer_id, hidden, weights, ids))
    assert len(copies) == L  # one copy per layer for the whole group, not one per chunk


@cuda
def test_moe_layer_fences_the_copy_stream_once_per_group(monkeypatch):
    from freetoken.core import get_global_ctx
    from freetoken.layers.moe import OffloadMoELayer

    from tests.models.qwen4_exp.common import fresh_ctx

    ctx = fresh_ctx()
    calls = []
    cache = SimpleNamespace(
        begin_prefill=lambda: calls.append("begin"),
        prefetch_prefill_layer=lambda i: calls.append(("prefetch", i)),
        wait_prefill_layer=lambda i: calls.append(("wait", i)) or (),
    )
    layer = OffloadMoELayer.__new__(OffloadMoELayer)
    layer.layer_id = 0
    layer._wait_prefill_overlap(cache)
    ctx.prefill_group_continuation = True
    layer._wait_prefill_overlap(cache)
    assert calls.count("begin") == 1
    assert get_global_ctx() is ctx
