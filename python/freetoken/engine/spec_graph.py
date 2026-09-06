"""CUDA graphs for the MTP verify window and the draft head (``--spec-mtp``).

A verify step runs the target over ``[t_last, d_1..d_K]`` -- an extend of exactly K+1 rows of
ONE request. Eagerly that costs tens of milliseconds of launch / host overhead (per pipeline
rank) on top of the real work, so the window is captured once as a CUDA graph, the way the
decode step is: every per-step input the captured kernels read lives in a static buffer (token
ids, positions, rope positions, KV slots, the GDN slot / cu_seqlens / continuation flag, the
attention addressing the backend stages, the residual stream a non-first pipeline rank
receives), and a step copies its values in, replays, and reads the static output.

What the window graph contains: this process's decoder layers (prefill-phase kernels: varlen
conv, the per-token GDN kernel that leaves the rollback stashes, paged attention over a cached
prefix, the MoE decode path) and, on the rank that owns the head, the lm_head over every row.
The GDN stashes the captured layers append are graph-pool tensors rewritten by every replay,
so the rollback reads the stashes recorded at capture. What stays eager: sampling and the
rollback.

The draft head (head-owning rank) gets two more graphs: the window pass (the head over the K+1
rows on the target's static final hidden state, the drafting row scored through the shared
lm_head into a static token buffer) and one chain step (a one-token head decode at a staged
position on the head's own previous output). Each drafted token is read back once, as eagerly.

Windows shorter than K+1 rows (the output budget's tail) fall back to the eager path. An
attention backend takes part by implementing ``init_spec_capture(rows)`` / ``stage_spec(md,
table_idx=, kv_len=)`` (the Triton and qsa_sparse backends do); without them the window stays
eager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.attention.linear import FLAMetadata
from freetoken.core import Batch, Req
from freetoken.utils import init_logger, mem_GB

if TYPE_CHECKING:
    from .engine import Engine

logger = init_logger(__name__)


def spec_graph_applicable(batch: Batch, rows: int, graph_rows: int) -> bool:
    """Whether a verify batch can replay the captured K+1-row window."""
    if not getattr(batch, "spec_verify", False) or rows != graph_rows or len(batch.reqs) != 1:
        return False
    req = batch.reqs[0]
    # a continuation of a cached prefix with no chunk-boundary tracking, like the capture
    return req.cached_len > 0 and req.device_len - req.cached_len == rows


class SpecVerifyGraph:
    def __init__(self, engine: "Engine", rows: int) -> None:
        self.engine = engine
        self.rows = rows
        dev = engine.device
        i32 = torch.int32
        self.input_ids = torch.zeros(rows, dtype=i32, device=dev)
        self.positions = torch.zeros(rows, dtype=i32, device=dev)
        self.rope_positions = torch.zeros(rows, dtype=i32, device=dev)  # positions + M-RoPE delta
        self.out_loc = torch.zeros(rows, dtype=i32, device=dev)
        # GDN metadata of a one-request extend: constant indptr, one slot, continuing state
        self.fla_cu = torch.tensor([0, rows], dtype=torch.int64, device=dev)
        self.fla_slot = torch.zeros(1, dtype=i32, device=dev)
        self.fla_init = torch.ones(1, dtype=torch.bool, device=dev)
        pp = engine.pp_comm
        self.pp_in = (
            torch.zeros(rows, pp.hidden_width, dtype=engine.dtype, device=dev)
            if pp is not None and not pp.is_first
            else None
        )
        self.out: torch.Tensor | None = None      # [rows, vocab] logits, or the residual stream to hand on
        self.hidden: torch.Tensor | None = None   # head-owning rank: the final hidden state (draft head input)
        self.stash: list = []                     # the GDN layers' SpecGdnStash objects (graph-pool tensors)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._capture_batch: Batch | None = None
        # draft head (head-owning rank): window graph + one-step chain graph, see capture_mtp
        self.g_window: torch.cuda.CUDAGraph | None = None
        self.g_chain: torch.cuda.CUDAGraph | None = None
        self.chain_batch: Batch | None = None

    # ----- static metadata -------------------------------------------------------------------
    def _fla(self) -> FLAMetadata:
        return FLAMetadata(
            cu_seqlens=self.fla_cu, cache_indices=self.fla_slot, has_initial_state=self.fla_init
        )

    def _bind(self, batch: Batch, *, table_idx: int, slot: int, kv_len: int) -> None:
        """Point ``batch`` at the static buffers and stage the attention addressing."""
        batch.input_ids = self.input_ids
        batch.positions = self.positions
        batch.rope_positions = self.rope_positions
        batch.rope_cos_sin = None
        batch.out_loc = self.out_loc
        self.fla_slot.fill_(slot)
        batch.fla_metadata = self._fla()
        batch.linear_table_idx = self.fla_slot
        attn = self.engine.attn_backend
        if getattr(batch, "attn_metadata", None) is None:
            attn.prepare_metadata(batch)
        attn.stage_spec(batch.attn_metadata, table_idx=table_idx, kv_len=kv_len)

    # ----- capture ----------------------------------------------------------------------------
    def capture(self) -> None:
        eng = self.engine
        model, ctx, dummy = eng.model, eng.ctx, eng.dummy_req
        rows = self.rows
        cached = 64  # any positive length: the window always continues a cached prefix
        req = Req(
            input_ids=torch.zeros(cached + rows, dtype=torch.int32, device="cpu"),
            table_idx=dummy.table_idx,
            cached_len=cached,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=None,  # type: ignore[arg-type]
        )
        req.linear_slot_idx = dummy.linear_slot_idx
        slot = dummy.linear_slot_idx if dummy.linear_slot_idx is not None else dummy.table_idx
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = batch.reqs
        batch.spec_verify = True
        batch.spec_all_rows = True
        # the dummy page-table row points every position at the dummy slot: fine for a capture
        self.positions.copy_(torch.arange(cached, cached + rows, dtype=torch.int32, device=eng.device))
        self.rope_positions.copy_(self.positions)
        self.out_loc.copy_(eng.page_table[dummy.table_idx, cached : cached + rows])
        self._bind(batch, table_idx=dummy.table_idx, slot=slot, kv_len=cached + rows)

        torch.cuda.synchronize(eng.device)
        free_before = torch.cuda.mem_get_info(eng.device)[0]
        graph = torch.cuda.CUDAGraph()
        ctx.spec_stash = []
        try:
            if self.pp_in is not None:
                ctx.pp_hidden_in = self.pp_in
            with ctx.forward_batch(batch):
                warm = model.forward()  # eager warm run (autotune, lazy buffers); shapes the output
                self.out = torch.empty_like(warm)
                self.out.copy_(warm)
                ctx.spec_stash = []
                with torch.cuda.graph(graph, stream=eng.stream):
                    self.out.copy_(model.forward())
                self.stash = list(ctx.spec_stash)
                ctx.spec_stash = []
                pp = eng.pp_comm
                self.hidden = model.last_hidden if pp is None or pp.is_last else None
        finally:
            ctx.pp_hidden_in = None
            if eng.moe_offload_cache is not None:
                eng.moe_offload_cache.reset()
        self.graph = graph
        self._capture_batch = batch
        torch.cuda.synchronize(eng.device)
        free_after = torch.cuda.mem_get_info(eng.device)[0]
        logger.info(
            f"--spec-mtp: captured the {rows}-row verify window as a CUDA graph "
            f"({len(self.stash)} GDN stashes, {mem_GB(free_before - free_after)} of graph memory, "
            f"free {mem_GB(free_after)})"
        )

    # ----- draft head ----------------------------------------------------------------------
    @property
    def mtp_ready(self) -> bool:
        return self.g_window is not None and self.g_chain is not None

    def capture_mtp(self) -> None:
        """Two graphs for the draft head. The window graph runs the head over the K+1 verify
        rows (the target's static final hidden state + the static successor ids), picks the
        drafting row and scores it through the shared lm_head into a static token buffer; the
        chain graph runs one head decode step (static position / KV slot / token) on the head's
        output kept in a static buffer and scores it. Each drafted token is read back once
        (one host sync per draft, as eagerly)."""
        from types import SimpleNamespace

        eng = self.engine
        model, ctx, dummy, attn = eng.model, eng.ctx, eng.dummy_req, eng.attn_backend
        mtp = model.mtp
        assert mtp is not None and self._capture_batch is not None and self.hidden is not None
        dev, rows = eng.device, self.rows
        width = self.hidden.shape[-1]  # the head's input width (hidden, or hc * hidden)
        self.next_ids = torch.zeros(rows, dtype=torch.int32, device=dev)
        self.row_idx = torch.zeros(1, dtype=torch.int64, device=dev)
        self.h_in = torch.zeros(1, width, dtype=self.hidden.dtype, device=dev)
        self.d_buf = torch.zeros(1, dtype=torch.int64, device=dev)
        torch.cuda.synchronize(dev)
        free_before = torch.cuda.mem_get_info(dev)[0]

        # -- window: the verify batch's static metadata (the window graph's capture batch) --
        batch = self._capture_batch

        def window():
            r = mtp.forward(self.hidden, self.next_ids, batch)
            self.h_in.copy_(r.index_select(0, self.row_idx))
            self.d_buf.copy_(model.lm_head.logits(mtp.to_head(self.h_in)).argmax(dim=-1))

        g_window = torch.cuda.CUDAGraph()
        with ctx.forward_batch(batch):
            window()  # warm
            with torch.cuda.graph(g_window, stream=eng.stream):
                window()

        # -- chain step: a one-token decode of the head at a staged position --
        cached = 64
        proxy = SimpleNamespace(
            table_idx=dummy.table_idx, extend_len=1, device_len=cached + rows + 1,
            cached_len=cached + rows, linear_slot_idx=dummy.linear_slot_idx, uid=-1,
            mm_embeds=None, mamba_restore_src=None, mamba_ping_pong=None, decode_batch_idx=0,
            input_ids=torch.zeros(cached + rows + 1, dtype=torch.int32),
        )
        mini = Batch(reqs=[proxy], phase="decode")
        mini.padded_reqs = [proxy]
        self.c_pos = torch.zeros(1, dtype=torch.int32, device=dev)
        self.c_rope = torch.zeros(1, dtype=torch.int32, device=dev)
        self.c_out_loc = torch.zeros(1, dtype=torch.int32, device=dev)
        self.c_ids = torch.zeros(1, dtype=torch.int32, device=dev)
        self.c_table = torch.zeros(1, dtype=torch.int32, device=dev)
        mini.positions, mini.input_ids, mini.out_loc = self.c_pos, self.c_ids, self.c_out_loc
        mini.rope_positions = self.c_rope
        mini.active_table_idx = self.c_table
        self.c_pos.fill_(cached + rows)
        self.c_rope.fill_(cached + rows)
        self.c_out_loc.copy_(eng.page_table[dummy.table_idx, cached + rows : cached + rows + 1])
        self.c_table.fill_(dummy.table_idx)
        attn.prepare_for_capture(mini)  # decode metadata on the backend's static buffers

        def chain():
            r = mtp.forward(self.h_in, self.c_ids, mini)
            self.h_in.copy_(r)
            self.d_buf.copy_(model.lm_head.logits(mtp.to_head(self.h_in)).argmax(dim=-1))

        g_chain = torch.cuda.CUDAGraph()
        with ctx.forward_batch(mini):
            chain()  # warm
            with torch.cuda.graph(g_chain, stream=eng.stream):
                chain()
        if eng.moe_offload_cache is not None:
            eng.moe_offload_cache.reset()
        self.g_window, self.g_chain, self.chain_batch = g_window, g_chain, mini
        torch.cuda.synchronize(dev)
        free_after = torch.cuda.mem_get_info(dev)[0]
        logger.info(
            f"--spec-mtp: captured the draft head (window + chain step) as CUDA graphs "
            f"({mem_GB(free_before - free_after)} of graph memory, free {mem_GB(free_after)})"
        )

    def _stage_chain(self, req: Req, position: int, token: int, rope_delta: int) -> None:
        """Point the chain graph at ``position`` of ``req`` with the previous draft as input."""
        eng = self.engine
        mini = self.chain_batch
        assert mini is not None
        proxy = mini.padded_reqs[0]
        proxy.table_idx = req.table_idx
        proxy.device_len = position + 1
        proxy.cached_len = position
        self.c_pos.fill_(position)
        self.c_rope.fill_(position + rope_delta)
        self.c_out_loc.copy_(eng.page_table[req.table_idx, position : position + 1])
        self.c_table.fill_(req.table_idx)
        self.c_ids.fill_(token)
        # the backend rebuilds this step's decode addressing and copies it into its capture buffers
        eng.attn_backend.prepare_metadata(mini)
        eng.attn_backend.prepare_for_replay(mini)

    def mtp_draft(self, req: Req, *, row: int, next_ids: list, pos_row: int, rope_delta: int = 0, prof=None) -> list:
        """Graph replacement of ``Engine._mtp_draft`` for a replayed window: ``next_ids`` are
        the window rows' successor ids (host), ``row`` the drafting row, ``pos_row`` its
        position; returns the greedy drafts."""
        eng = self.engine
        assert self.mtp_ready
        self.next_ids.copy_(torch.tensor(next_ids, dtype=torch.int32))
        self.row_idx.fill_(row)
        self.g_window.replay()
        d = int(self.d_buf.item())
        drafts = [d]
        if prof is not None:
            prof.mark("mtp_window")
        for j in range(1, eng.spec_k):
            p = pos_row + j
            if p + 1 > (req.spec_alloc_len or 0):
                break  # no reserved KV page for this draft position
            self._stage_chain(req, p, d, rope_delta)
            self.g_chain.replay()
            d = int(self.d_buf.item())
            drafts.append(d)
        if prof is not None:
            prof.mark("mtp_chain")
        return drafts

    # ----- replay -----------------------------------------------------------------------------
    def stage(self, batch: Batch) -> None:
        """Copy a real verify batch's per-step inputs into the static buffers and bind it."""
        req = batch.reqs[0]
        self.input_ids.copy_(batch.input_ids)
        self.positions.copy_(batch.positions)
        rope = getattr(batch, "rope_positions", None)
        self.rope_positions.copy_(batch.positions if rope is None else rope)
        self.out_loc.copy_(batch.out_loc)
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        self._bind(batch, table_idx=req.table_idx, slot=slot, kv_len=req.device_len)

    def replay(self) -> torch.Tensor:
        assert self.graph is not None and self.out is not None
        self.graph.replay()
        return self.out


__all__ = ["SpecVerifyGraph", "spec_graph_applicable"]
