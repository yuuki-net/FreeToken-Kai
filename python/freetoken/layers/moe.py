import os
from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.bank_disk import apply_permutation
from freetoken.moe.fused import fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.utils import decode_sample as _ds


from .base import BaseOP
from .quantization import ExpertView, LayerKind, QuantConfig, quant_method_for

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` with a
# precomputed routing instead of going through the generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

# Hybrid decode overlaps the CPU overflow GEMV behind the GPU PCIe fetch + GEMM by
# default. Set FREETOKEN_HYBRID_OVERLAP=0 to force the serial path (CPU sync before the
# GPU work) -- a measurement-only escape hatch to A/B the overlap benefit.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"


class MoELayer(BaseOP):
    """Resident routed experts.

    The expert format comes from ``quant_method`` (declared by ``create_weights``, run by
    ``apply``); without a ``quant_config`` the experts are plain bf16. The gated activation is
    ``act(clamp(g, limit) * alpha) * (clamp(u) + beta)`` with ``interleaved`` gate|up rows
    for gpt-oss."""

    quant_layer_kind = LayerKind.MOE

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        layer_id: int | None = None,
        strategy: str = "resident",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_rank = tp_info.rank
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.alpha = alpha
        self.beta = beta
        self.limit = limit
        self.interleaved = interleaved
        self.has_bias = has_bias
        self.layer_id = layer_id
        self.strategy = strategy
        self.decode_target = decode_target
        self.prefix = prefix
        # offload layers without a quant config stay on the format-tag banks (GGUF q4_0)
        self.quant_method = None
        if quant_config is not None or allocate_experts:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            if allocate_experts:
                self.quant_method.create_weights(self)

    def finalize(self) -> None:
        if self.quant_method is not None:
            self.quant_method.finalize(self)

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.quant_method is not None
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, self.quant_method.resident_view(self),
            layer=self, is_prefill=get_global_ctx().batch.is_prefill,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._maybe_all_reduce(self._resident_gemm(hidden_states, topk_weights, topk_ids))


class OffloadMoELayer(MoELayer):
    # logical -> physical expert id for this layer, when --moe-bank-ram renumbered the bank
    # so the resident experts occupy rows [0, hot). None = every expert resident, which is
    # the identity map and skips the remap entirely. Set per layer by
    # attach_offload_moe_cache; see moe/bank_disk.py.
    #
    # Declared on the class, not in __init__: the routed entry points are exercised against
    # instances built with __new__ (tests/moe/test_cpu_prefill_short.py drives the dispatch
    # threshold without a real layer), and those never run __init__.
    expert_perm: "torch.Tensor | None" = None

    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
            alpha=alpha,
            beta=beta,
            limit=limit,
            interleaved=interleaved,
            has_bias=has_bias,
            layer_id=layer_id,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.offload_cache: OffloadMoeCache | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        ctx = get_global_ctx()
        # an MTP verify window is an extend for the kernels but must NOT stream whole layers:
        # its few tokens go through the decode (LRU cache / hybrid) path
        if ctx.batch.is_prefill and not getattr(ctx.batch, "spec_verify", False):
            final_hidden_states = self.prefill_forward(hidden_states, router_logits)
        else:
            final_hidden_states = self.decode_forward(hidden_states, router_logits)
        return self._maybe_all_reduce(final_hidden_states)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        ctx = get_global_ctx()
        if ctx.batch.is_prefill and not getattr(ctx.batch, "spec_verify", False):
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._decode_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._decode_routed(hidden_states, topk_weights, topk_ids)

    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._prefill_routed(hidden_states, topk_weights, topk_ids)

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand; prefill streams whole layers, double-buffered when overlap
    # is enabled. The kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here."""
        cache = self.offload_cache
        assert cache is not None
        # Renumbering is applied here, before anything reads an expert id: the slot cache,
        # copy_missing and the CPU executor all address the physical row. The bank keeps its
        # [num_experts, ...] shape either way -- rows past the resident prefix are file-backed
        # pages that fault in -- so nothing downstream can tell. No-op when nothing was moved.
        apply_permutation(topk_ids, self.expert_perm)
        # --moe-collect-stats: event records at the seams of the expert work, captured into the
        # decode graph (utils/decode_sample.py); None outside a timed decode forward
        tl = _ds.timeline()
        layer = self.layer_id
        if tl is not None:
            tl.mark(layer, _ds.START)
        if cache.collect_stats:
            # every decode layer, CPU ones included: physical ids, before a kernel rewrites
            # them to slots (--moe-collect-stats: routing histogram and ring)
            cache.record_routes(layer, topk_ids)
        if cache.is_cpu_layer(layer):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            out = executor.decode(layer, hidden_states, topk_weights, topk_ids)
            if tl is not None:
                tl.layer_kind(layer, "cpu")
                tl.mark(layer, _ds.DONE)
            return out
        if cache.decode_target == "hybrid":
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids, tl)
        cache.ensure_experts(layer, topk_ids)
        if tl is not None:
            tl.mark(layer, _ds.ROUTED)
        cache.copy_missing()
        if tl is not None:
            tl.mark(layer, _ds.FETCHED)
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(layer),
            is_prefill=False,
        )
        if tl is not None:
            tl.layer_kind(layer, "gpu")
            tl.mark(layer, _ds.COMPUTED)
            tl.mark(layer, _ds.DONE)
        return out

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        tl=None,
    ) -> torch.Tensor:
        """Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU
        computes the overflow misses, overlapped, then the partials merge.

        The CPU pool is kicked off (``decode_submit``) before the GPU PCIe fetch + GEMM so
        the CPU overflow GEMV runs concurrently with the GPU work. Capture-safe: the
        routing split is device-side elementwise and the CPU submit/sync are host nodes.
        Each route is computed exactly once -- the GPU weights are zeroed for CPU-assigned
        routes and the CPU ids are -1 for GPU-assigned routes (the C++ kernel skips id<0).
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        raw = topk_ids.clone()  # raw expert ids for the CPU partial
        # --moe-bank-ram: the GPU can only address the registered prefix, so a miss on a row
        # past it must not be fetched. The fetch choice lives in the kernel, so steer it from
        # here instead: hand those positions expert 0 (rank 0 of the renumbering -- the
        # hottest expert, so all but certainly a hit costing no fetch), then overwrite the
        # slot it returns with -1 so the CPU partial takes them. raw still holds the true
        # ids, which is what the CPU executor reads.
        cold = None
        if cache.prefix_pinned_rows is not None:
            cold = raw >= cache.prefix_pinned_rows
            topk_ids.masked_fill_(cold, 0)
        cache.ensure_experts_hybrid(self.layer_id, topk_ids)  # -> slot or -1
        if cold is not None:
            topk_ids.masked_fill_(cold, -1)
        on_gpu = topk_ids >= 0

        cpu_ids = torch.where(on_gpu, raw.new_full((), -1), raw).contiguous()
        if tl is not None:
            tl.mark(self.layer_id, _ds.ROUTED)
        pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        cpu_routed_early = (
            executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
        )

        cache.copy_missing()
        if tl is not None:
            tl.mark(self.layer_id, _ds.FETCHED)
        gpu_slots = topk_ids.clamp_min(0)  # -1 -> slot 0 (zero-weighted below)
        gpu_w = torch.where(on_gpu, topk_weights, topk_weights.new_zeros(())).contiguous()
        gpu_routed = self._expert_gemm(
            cache,
            hidden_states,
            gpu_w,
            gpu_slots,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        if tl is not None:
            tl.mark(self.layer_id, _ds.COMPUTED)
        cpu_routed = cpu_routed_early if not _HYBRID_OVERLAP else executor.decode_sync(pending)
        out = gpu_routed + cpu_routed
        if tl is not None:
            tl.layer_kind(self.layer_id, "gpu")
            tl.mark(self.layer_id, _ds.DONE)
        return out

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement: stream whole layers -- double-buffered behind the
        previous layer's GEMMs when ``prefill_overlap`` is on, else a synchronous
        ``materialize_layer``. In both, position == expert id (physical, after the
        --moe-bank-ram renumbering), so the routing ids pass through unmapped."""
        cache = self.offload_cache
        assert cache is not None
        apply_permutation(topk_ids, self.expert_perm)
        if (
            PREFILL_SPLIT_MIN_TOKENS <= hidden_states.shape[0] <= prefill_split_max_tokens(cache)
            and not self.apply_router_weight_on_input
            and not cache.is_unpinned_layer(self.layer_id)
        ):
            return self._prefill_split(cache, hidden_states, topk_weights, topk_ids)
        # short extends take the CPU executor whether or not the overlap double buffer is on:
        # the decision is per forward (every layer sees the same row count), so a forward that
        # goes this way never touches the overlap machinery
        if (
            getattr(cache, "cpu_executor", None) is not None
            and 0 < hidden_states.shape[0] <= cpu_prefill_max_tokens()
        ):
            return self._prefill_on_cpu(cache, hidden_states, topk_weights, topk_ids)
        if cache.prefill_overlap:
            views = self._wait_prefill_overlap(cache)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            # the GEMMs are issued, so the host is free to run the next layer's bounce copy
            cache.finish_prefill_prefetch()
            cache.release_prefill_layer(self.layer_id)
            return out
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(self.num_experts),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )

    def _prefill_split(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill with the CPU and the GPU at once. The CPU executor's cost grows with the rows
        routed to an expert, moving an expert to the GPU costs the same however many rows use
        it: the experts this forward routes to least go to the CPU, the rest are the only ones
        moved to the GPU (not the whole layer), and the two run concurrently -- the CPU pieces on
        a side stream, the partial copy and the GEMM on this one. ``plan_prefill_split`` picks the
        cut where both sides would finish together. Each route is computed exactly once: the GPU
        sees the CPU's routes with weight 0 on an expert it holds, the CPU sees the GPU's as -1."""
        E = self.num_experts
        counts = torch.bincount(topk_ids.reshape(-1).long(), minlength=E)[:E].tolist()  # syncs
        cpu_experts, gpu_experts = plan_prefill_split(counts, prefill_split_ratio())
        if not gpu_experts:
            return self._prefill_on_cpu(cache, hidden_states, topk_weights, topk_ids)
        device = topk_ids.device
        gpu_idx = torch.tensor(gpu_experts, dtype=torch.int32, device=device)
        main = torch.cuda.current_stream(device)
        pending = None
        if cpu_experts:
            on_cpu = torch.zeros(E, dtype=torch.bool, device=device)
            on_cpu[torch.tensor(cpu_experts, dtype=torch.long, device=device)] = True
            route_cpu = on_cpu[topk_ids.long()]
            cpu_ids = torch.where(route_cpu, topk_ids, topk_ids.new_full((), -1))
            # the GPU GEMM may overwrite hidden_states in place: the CPU reads its own copy
            x_cpu, w_cpu = hidden_states.clone(), topk_weights.clone()
            side = _split_stream(device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                cpu_out = self._prefill_on_cpu(cache, x_cpu, w_cpu, cpu_ids)
            for t in (x_cpu, w_cpu, cpu_ids):
                t.record_stream(side)
            done = torch.cuda.Event()
            done.record(side)
            pending = (cpu_out, done)
            topk_weights = torch.where(route_cpu, topk_weights.new_zeros(()), topk_weights)
            topk_ids = torch.where(route_cpu, topk_ids.new_full((), gpu_experts[0]), topk_ids)
        cache.materialize_experts(self.layer_id, gpu_idx, gpu_experts)
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(E),
            n=E,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )
        if pending is not None:
            cpu_out, done = pending
            main.wait_event(done)
            cpu_out.record_stream(main)
            out = out + cpu_out
        return out

    def _prefill_on_cpu(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """A short extend (a chat turn behind a cached prefix, a few hundred tokens at most)
        skips streaming this layer's whole expert bank to the GPU: the CPU executor computes
        the routed experts instead, in pieces of ``CPU_PREFILL_PIECE`` rows (one buffer shape,
        the tail padded with skipped routes). Streaming a 256-expert NVFP4 layer costs ~0.13 s
        per layer at the ~3.4 GB/s an RTX 2060 sees under WSL2 -- ~5 s per chunk over 40 layers
        whatever the chunk holds -- while the CPU reads each distinct expert once for all the
        rows routed to it. ``FREETOKEN_CPU_PREFILL_MAX_TOKENS`` (default 256; 0 disables) is the
        longest extend that takes this path; longer chunks stream as before."""
        executor = cache.cpu_executor
        assert executor is not None
        total = hidden_states.shape[0]
        piece = max(1, min(CPU_PREFILL_PIECE, int(executor.max_tokens)))
        outs = []
        for start in range(0, total, piece):
            end = min(start + piece, total)
            n = end - start
            x = hidden_states[start:end]
            w = topk_weights[start:end]
            ids = topk_ids[start:end]
            if n < piece:
                pad = piece - n
                x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
                w = torch.cat([w, w.new_zeros(pad, w.shape[1])])
                ids = torch.cat([ids, ids.new_full((pad, ids.shape[1]), -1)])  # -1: skipped
            outs.append(executor.decode(self.layer_id, x, w, ids)[:n])
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        if self.layer_id == 0 and not _prefill_group_continuation():
            # a grouped prefill (--pp-prefill-group) fences the copy stream once per group: a
            # second begin would drop the prefetched buffers and copy layers 0 and 1 again
            cache.begin_prefill()
        cache.prefetch_prefill_layer(self.layer_id)
        cache.prefetch_prefill_layer(self.layer_id + 1)
        return cache.wait_prefill_layer(self.layer_id)

    # ------------------------------------------------------------------
    # Kernel dispatch: ``views`` are the bank tensors the movement step produced (in bank registration order) and ``topk_ids`` already index their rows.
    # GGUF q4_0 experts still dispatch on the cache's format tag until they get a method.
    # ------------------------------------------------------------------

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
    ) -> torch.Tensor:
        if self.quant_method is not None:
            from freetoken.moe.legacy_format import canonical_role  # legacy_format imports this package

            view = ExpertView(
                {canonical_role(name): t for name, t in zip(cache.bank_schema, views)},
                slots=None if n is not None else topk_ids, n=n, alphas=alphas,
            )
            return self.quant_method.apply(
                hidden_states, topk_weights, topk_ids, view, layer=self, is_prefill=is_prefill
            )
        fmt = cache.quant_format
        if fmt == "q4_0":
            # Native GGUF Q4_0 experts: dequant-in-kernel grouped GEMV (MMVQ) over the
            # streamed packed banks; topk_ids already index the cache slots / layer.
            from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

            gate_up, down = views
            return fused_experts_gguf_q4_0(
                hidden_states, gate_up, down, topk_weights, topk_ids, self.activation
            )
        raise AssertionError(f"offload experts without a quant method only serve q4_0 banks, got {fmt!r}")


def _prefill_group_continuation() -> bool:
    """--pp-prefill-group: a grouped prefill is running a layer for a later chunk of its group.
    False without a global context (layer-level callers that never set one)."""
    from freetoken import core

    ctx = core._GLOBAL_CTX
    return ctx is not None and ctx.prefill_group_continuation


# Short-extend prefill on the CPU executor (see OffloadMoELayer._prefill_on_cpu): the piece
# size is the one batch shape the executor sees for it (the engine sizes max_tokens to it).
CPU_PREFILL_PIECE = 64


def cpu_prefill_max_tokens() -> int:
    """Longest prefill extend the CPU executor computes instead of streaming the layer banks
    (``FREETOKEN_CPU_PREFILL_MAX_TOKENS``, default 256; 0 disables)."""
    return int(os.environ.get("FREETOKEN_CPU_PREFILL_MAX_TOKENS", "256") or 0)


# Split prefill (OffloadMoELayer._prefill_split). The ratio is what moving one expert to the
# GPU costs, in CPU-executor routes. Measured on an RTX 2060 host (Ornith, real routing, both
# sides running at once): 4 was fastest from 32 to 1024 rows; the ratio of the two costs taken
# one at a time is ~7, but the CPU kernels and the DMA slow each other down when they overlap.
PREFILL_SPLIT_RATIO = 4.0
# Where it pays, per layer on that host (the production layer at Ornith's geometry, real routing,
# against what ran before -- the CPU short path up to 256 rows, the whole layer above): 64 rows
# 0.78x, 128 0.63x, 256 0.51x, 512 0.64x; 32 and 1024 went either way between runs (the fixed
# cost of the split -- a routing count read back, the side stream, the partial copy -- against
# a CPU that is already quick, or a GPU that needs nearly every expert anyway). Hence the range.
# Layers without a device address (LOCKED/PAGEABLE) never split: their experts can only be staged
# through pinned buffers, and staging scattered experts one run at a time was slower than staging
# the whole layer (1024 rows: 92 ms against 63).
PREFILL_SPLIT_MIN_TOKENS = 64
PREFILL_SPLIT_MAX_TOKENS = 1024


def prefill_split_enabled() -> bool:
    """``FREETOKEN_PREFILL_SPLIT=0`` turns the split prefill off (the CPU short path and the
    whole-layer streaming take over again)."""
    return os.environ.get("FREETOKEN_PREFILL_SPLIT", "1") != "0"


def prefill_split_ratio() -> float:
    return float(os.environ.get("FREETOKEN_PREFILL_SPLIT_RATIO", "") or PREFILL_SPLIT_RATIO)


def prefill_split_limit() -> int:
    """``PREFILL_SPLIT_MAX_TOKENS`` or ``FREETOKEN_PREFILL_SPLIT_MAX_TOKENS``; 0 when the split
    is off. The same with the prefill overlap double buffer on: it streams the next layer before
    that layer's routing is known, so a split layer is not hidden behind the previous one, but on
    two RTX 3060s (Flash-Next, overlap on, 2026-09-27) the first token still came much sooner:
    64 tokens 5.56 -> 2.05 s, 270 8.37 -> 3.01 s, 512 7.99 -> 4.02 s, 1024 7.97 -> 6.76 s."""
    if not prefill_split_enabled():
        return 0
    return int(os.environ.get("FREETOKEN_PREFILL_SPLIT_MAX_TOKENS", "") or PREFILL_SPLIT_MAX_TOKENS)


def prefill_split_max_tokens(cache) -> int:
    """Longest prefill forward of this cache that splits its experts between the CPU and the GPU
    (``prefill_split_limit``); 0 when it cannot: no CPU executor, or a --moe-bank-ram mapped bank
    (its prefill has its own reader)."""
    if getattr(cache, "cpu_executor", None) is None:
        return 0
    if getattr(cache, "prefix_pinned_rows", None) is not None or getattr(cache, "bank_reader", None) is not None:
        return 0
    return prefill_split_limit()


def plan_prefill_split(counts: list[int], ratio: float) -> tuple[list[int], list[int]]:
    """``counts[e]`` = rows routed to expert ``e`` in this forward. Returns the experts for the
    CPU and for the GPU (each ascending; experts nobody routed to are in neither). The CPU
    takes the least-routed experts, as many as keeps ``max(CPU routes, ratio * GPU experts)``
    lowest -- the time both sides need when they run at once, in CPU routes."""
    used = sorted((c, e) for e, c in enumerate(counts) if c > 0)
    n = len(used)
    best, best_k, cum = ratio * n, 0, 0
    for k in range(1, n + 1):
        cum += used[k - 1][0]
        cost = max(cum, ratio * (n - k))
        if cost < best:
            best, best_k = cost, k
    return sorted(e for _, e in used[:best_k]), sorted(e for _, e in used[best_k:])


_SPLIT_STREAMS: dict = {}


def _split_stream(device: torch.device) -> "torch.cuda.Stream":
    """The side stream the split prefill runs its CPU pieces on (their host-node waits would
    otherwise hold back the partial copy and the GEMM on the main stream)."""
    key = device.index if device.index is not None else torch.cuda.current_device()
    s = _SPLIT_STREAMS.get(key)
    if s is None:
        s = _SPLIT_STREAMS[key] = torch.cuda.Stream(device=device)
    return s


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    limit: float | None = None,
    interleaved: bool = False,
    has_bias: bool = False,
    quant_config: QuantConfig | None = None,
    prefix: str = "",
) -> MoELayer:
    """Build the experts layer for ``config.moe_strategy`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_strategy(config.moe_strategy)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        alpha=alpha,
        beta=beta,
        limit=limit,
        interleaved=interleaved,
        has_bias=has_bias,
        quant_config=quant_config,
        prefix=prefix,
    )
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["layer_id"] = layer_id
        kwargs["strategy"] = config.moe_strategy
        kwargs["decode_target"] = config.decode_target
    return layer_cls(**kwargs)
