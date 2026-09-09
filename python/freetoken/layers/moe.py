import os
from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.bank_disk import apply_permutation
from freetoken.moe.fused import fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache


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
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        if cache.decode_target == "hybrid":
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)
        cache.ensure_experts(self.layer_id, topk_ids)
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
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
        # ids, which is what the CPU executor and the routing histogram read.
        cold = None
        if cache.prefix_pinned_rows is not None:
            cold = raw >= cache.prefix_pinned_rows
            topk_ids.masked_fill_(cold, 0)
        cache.ensure_experts_hybrid(self.layer_id, topk_ids, freq_ids=raw)  # -> slot or -1
        if cold is not None:
            topk_ids.masked_fill_(cold, -1)
        if cache.collect_stats:
            cache.record_decode_stats_hybrid(self.layer_id)
        on_gpu = topk_ids >= 0

        cpu_ids = torch.where(on_gpu, raw.new_full((), -1), raw).contiguous()
        pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        cpu_routed_early = (
            executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
        )

        cache.copy_missing()
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
        cpu_routed = cpu_routed_early if not _HYBRID_OVERLAP else executor.decode_sync(pending)
        return gpu_routed + cpu_routed

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
        if self.layer_id == 0:
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


# Short-extend prefill on the CPU executor (see OffloadMoELayer._prefill_on_cpu): the piece
# size is the one batch shape the executor sees for it (the engine sizes max_tokens to it).
CPU_PREFILL_PIECE = 64


def cpu_prefill_max_tokens() -> int:
    """Longest prefill extend the CPU executor computes instead of streaming the layer banks
    (``FREETOKEN_CPU_PREFILL_MAX_TOKENS``, default 256; 0 disables)."""
    return int(os.environ.get("FREETOKEN_CPU_PREFILL_MAX_TOKENS", "256") or 0)


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
