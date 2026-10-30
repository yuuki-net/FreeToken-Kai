from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import (
    causal_conv1d_decode,
    causal_conv1d_varlen,
    causal_conv1d_verify,
    causal_conv1d_window,
)
from freetoken.layers import BaseOP, GatedRMSNorm, LinearColParallelMerged, LinearReplicated
from freetoken.layers.quantization import QuantConfig
from freetoken.utils import init_logger

logger = init_logger(__name__)
# FT_SPEC_CHECK_GDN=1: run the chunk kernel beside the per-token spec kernel on a copy of the
# state for the first verify forwards and log the difference (diagnostics)
_CHECK_GDN = os.environ.get("FT_SPEC_CHECK_GDN") == "1"
_check_left = [48 * 3]
_check_conv_left = [48 * 3]
# FREETOKEN_GDN_WINDOW_CONV=varlen keeps the chunked prefill conv for the verify window (A/B)
_WINDOW_CONV = os.environ.get("FREETOKEN_GDN_WINDOW_CONV", "update")


@dataclass
class SpecGdnStash:
    """One GDN layer's MTP-verify leftovers: the conv state before the window, the window's
    conv inputs and the per-token recurrent states; ``spec_rollback`` restores the accepted
    token's state from these."""

    pool: object
    li: int
    slot: torch.Tensor          # [1] int32 state slot
    prev_conv: torch.Tensor     # [1, conv_dim, K-1]
    conv_in: torch.Tensor       # [T, conv_dim]
    ht: torch.Tensor            # [T, HV, V, K] fp32, state after each token

    def restore(self, accepted: int) -> None:
        from freetoken.engine.spec import rebuild_conv_state

        slot = self.slot.long()
        conv = rebuild_conv_state(self.prev_conv[0], self.conv_in, accepted)
        self.pool.conv_states[self.li].index_copy_(0, slot, conv.unsqueeze(0).to(self.pool.conv_states.dtype))
        rec = self.pool.recurrent_states[self.li]
        rec.index_copy_(0, slot, self.ht[accepted - 1 : accepted].to(rec.dtype).view_as(rec[0:1]))


_SPEC_HT: dict = {}


def spec_window_states(pool, li: int, rows: int) -> torch.Tensor:
    """``[rows, HV, V, K]`` buffer for the per-token recurrent states of an MTP verify window of
    layer ``li``: one per (pool, layer), grown to the longest window seen (the K+1-row warm run
    comes first, before any capture). Every window graph writes the same buffer -- they never
    replay together and the rollback reads it right after -- so the graphs of the shorter
    windows (--spec-min-p) cost temporaries only, not a copy of every layer's states."""
    key = (id(pool), li)
    buf = _SPEC_HT.get(key)
    shape = (rows, *pool.recurrent_states.shape[2:])
    if buf is None or buf.shape[0] < rows:
        buf = _SPEC_HT[key] = torch.empty(shape, dtype=pool.recurrent_states.dtype, device=pool.recurrent_states.device)
    return buf[:rows]


def channels_first_copy(conv_in: torch.Tensor) -> torch.Tensor:
    """``[total, conv_dim]`` -> a fresh ``[conv_dim, total]`` buffer. ``transpose().contiguous()``
    is NOT a copy when ``total == 1`` (a size-1 dim never breaks contiguity), and the varlen conv
    kernel writes its output into the buffer in place -- which then clobbered the raw conv
    inputs an MTP verify window keeps for its rollback (``SpecGdnStash.conv_in``)."""
    return conv_in.t().clone(memory_format=torch.contiguous_format)


def _chunk_prefill(q, k, v, g, beta, *, state, indices, cu_seqlens, scale):
    """Chunked gated-delta-rule prefill (vendored fla): reads and writes ``state[indices]`` in
    place; returns ``(o [total, num_v, V], h)``, h the per-chunk states for the track snapshot."""
    from freetoken.kernel.fla import chunk_gated_delta_rule

    o, _, h = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=state, initial_state_indices=indices.to(torch.int32),
        cu_seqlens=cu_seqlens.to(torch.int64), head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    return o[0], h


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class GatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.

    ``output_gate`` is the gate activation name from ``LinearGatedDeltaGroupConfig``
    ("sigmoid" for Qwen3.8-Flash-Next).
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, output_gate: str = "silu",
        *, quant_config: QuantConfig | None = None, prefix: str = "", out_heads_tiled: bool = False,
    ):
        self.layer_id = layer_id
        self._out_heads_tiled = out_heads_tiled
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6/3.8 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        # quantized checkpoints quantize qkv|z but not b|a, so the fusion splits into a qkvz GEMM and a ba GEMM with their own schemes (matches sglang / vLLM)
        self._split_in_proj = (
            quant_config is not None and quant_config.scheme_for(f"{prefix}.in_proj_qkvz") is not None
        )

        self._in_proj_split = [self.conv_dim, self.value_dim, num_v_heads, num_v_heads]
        if self._split_in_proj:
            self.in_proj_qkvz = LinearColParallelMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_qkvz",
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_ba",
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a.
            self.in_proj = LinearColParallelMerged(
                hidden_size, self._in_proj_split, has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj",
            )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast before the decode kernel. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast.
        self.dt_bias = torch.empty(num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(num_v_heads, dtype=torch.float32)
        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps, activation=output_gate)
        self.out_proj = LinearReplicated(
            self.value_dim, hidden_size, has_bias=False,
            quant_config=quant_config, prefix=f"{prefix}.out_proj",
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state,
                      max_seq_len: int | None = None) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata.
        ``max_seq_len``: the longest extend when the caller knows it (a one-request MTP verify
        window: ``total``), so the Triton fallback needs no device->host read (graph capture)."""
        li = pool.local_index(self.layer_id)
        x = channels_first_copy(conv_in)  # [conv_dim, total]; the kernel writes silu(conv) in place
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state,
                                   max_seq_len=max_seq_len)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_window(self, conv_in: torch.Tensor, cache_indices: torch.Tensor, pool) -> torch.Tensor:
        """The MTP verify window's conv (one request, K+1 tokens) through the update kernel
        (kernel/causal_conv1d.causal_conv1d_window) instead of the varlen prefill conv; returns
        ``[total, conv_dim]`` as a transposed view, like ``_conv_prefill``. FT_SPEC_CHECK_GDN=1
        runs the varlen conv beside it on a copy of the state for the first windows."""
        li = pool.local_index(self.layer_id)
        x = channels_first_copy(conv_in).unsqueeze(0)  # [1, conv_dim, total]
        ref = None
        if _CHECK_GDN and _check_conv_left[0] > 0:
            _check_conv_left[0] -= 1
            ref_state = pool.conv_states[li].clone()
            total = conv_in.shape[0]
            cu = torch.tensor([0, total], dtype=torch.int32, device=conv_in.device)
            ref = causal_conv1d_varlen(
                channels_first_copy(conv_in), self._conv_weight(), ref_state, cu, cache_indices,
                torch.ones(1, dtype=torch.bool, device=conv_in.device), max_seq_len=total,
            )
        out = causal_conv1d_window(x, pool.conv_states[li], self._conv_weight(), cache_indices)
        if ref is not None:
            slot = cache_indices.long()
            d_out = (out[0].float() - ref.float()).abs().max().item()
            d_state = (pool.conv_states[li].index_select(0, slot).float() - ref_state.index_select(0, slot).float()).abs().max().item()
            logger.warning(f"spec conv check layer {self.layer_id} T={conv_in.shape[0]}: |out-ref|max={d_out:.3e} |state-ref|max={d_state:.3e}")
        return out[0].transpose(0, 1)

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _conv_verify(self, conv_in: torch.Tensor, fla, pool) -> torch.Tensor:
        """Causal conv over a verify batch's rows: rolls the live conv state as if every row were accepted and parks each row's window in ``fla.verify_scratch``."""
        li = pool.local_index(self.layer_id)
        total = conv_in.shape[0]
        bs = fla.cu_seqlens.numel() - 1
        # free views both ways: the conv reads any strides and writes token-major output
        x = conv_in.reshape(bs, total // bs, -1).transpose(1, 2)
        out = causal_conv1d_verify(x, pool.conv_states[li], self._conv_weight(), fla.cache_indices, fla.verify_scratch.conv_window[li])
        return out.transpose(1, 2).reshape(total, -1)

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor, fla) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        rec.index_copy_(0, fla.track_dst, h[0, fla.track_h_row].to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device, pool)
            batch.fla_metadata = fla

        if self._split_in_proj:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        if not batch.is_prefill:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            # A verify only reads the live state and stashes each row's inputs; LinearVerifyScratch.commit replays the accepted rows into it.
            from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

            verify = batch.phase == "verify"
            if verify:
                mixed = self._conv_verify(conv_in, fla, pool)
            else:
                mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            core_out = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log, a=a, dt_bias=self.dt_bias,  # already fp32 (stored fp32)
                softplus_beta=1.0, softplus_threshold=20.0,
                q=q, k=k, v=v, b=b,
                initial_state_source=pool.recurrent_states[li],
                initial_state_indices=fla.cache_indices,  # already int32 (built int32 by build_fla_metadata)
                scale=self.head_k_dim ** -0.5, use_qk_l2norm_in_kernel=True, cu_seqlens=fla.cu_seqlens,
                disable_state_update=verify, input_stash=fla.verify_scratch.input_stash(li) if verify else None,
            )[0]  # [total, num_v, V]
        else:
            # MTP verify window (Kai's --spec-mtp, an extend batch): keep what the rollback needs
            # (state before the window, the window's conv inputs) -- the conv kernel updates the
            # slot in place below
            spec = bool(getattr(batch, "spec_verify", False))
            prev_conv = (
                pool.conv_states[li].index_select(0, fla.cache_indices.long()).clone() if spec else None
            )
            if spec and _WINDOW_CONV == "update" and fla.cache_indices.numel() == 1:
                # the few rows of one request: the decode-style update kernel, not the chunked prefill conv
                mixed = self._conv_window(conv_in, fla.cache_indices, pool)
            else:
                mixed = self._conv_prefill(
                    conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state,
                    max_seq_len=total if spec else None)  # a verify window is one request
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            if spec:
                # ``mixed`` is a transposed view (token stride 1, feature stride ``total``) and
                # q/k/v above are views of it; the vendored per-token kernel walks its inputs
                # as contiguous [1, T, heads, dim] with plain pointer arithmetic. For T == 1
                # the two layouts coincide, for a real draft window they do not.
                q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            if spec:
                # Per-token recurrence with a state written out after EVERY token (the vendored
                # vLLM spec-decoding path: inplace_final_state=False); the live slot is left
                # untouched and the model copies the accepted token's state back in
                # spec_rollback. Exact per-token math, so it matches the chunk kernel.
                from freetoken.kernel.fla.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

                ref = None
                if _CHECK_GDN and _check_left[0] > 0:
                    _check_left[0] -= 1
                    ref_state = pool.recurrent_states[li].clone()
                    ref_o, _ = _chunk_prefill(
                        q, k, v, g, beta,
                        state=ref_state, indices=fla.cache_indices,
                        cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                    )
                    ref = (ref_state, ref_o)
                o, ht = fused_recurrent_gated_delta_rule_fwd(
                    q, k, v, g, beta, self.head_k_dim ** -0.5,
                    initial_state=pool.recurrent_states[li],
                    inplace_final_state=False,
                    final_state_out=spec_window_states(pool, li, total),
                    cu_seqlens=fla.cu_seqlens.to(torch.int64),
                    ssm_state_indices=fla.cache_indices,
                    use_qk_l2norm_in_kernel=True,
                )
                core_out = o[0]
                if ref is not None:
                    ref_state, ref_o = ref
                    slot0 = int(fla.cache_indices[0].item())
                    a, b_ = core_out.float().reshape(-1), ref_o.float().reshape(-1)
                    ha, hb = ht[-1].float().reshape(-1), ref_state[slot0].float().reshape(-1)
                    logger.warning(
                        f"spec gdn check layer {self.layer_id} T={total}: "
                        f"|o-ref|max={(a - b_).abs().max().item():.3e} |ref|max={b_.abs().max().item():.3e} "
                        f"|ht-ref|max={(ha - hb).abs().max().item():.3e} |ref|max={hb.abs().max().item():.3e}"
                    )
                get_global_ctx().spec_stash.append(
                    SpecGdnStash(pool=pool, li=li, slot=fla.cache_indices, prev_conv=prev_conv,
                                 conv_in=conv_in, ht=ht)
                )
            else:
                core_out, h = _chunk_prefill(
                    q, k, v, g, beta,
                    state=pool.recurrent_states[li], indices=fla.cache_indices,
                    cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                )  # [total, num_v_heads, head_v_dim]
                if fla.track_dst is not None:
                    self._write_track_snapshot(pool, li, conv_in, h, fla)

        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm.forward(core_out, z).reshape(total, -1)
        if self._out_heads_tiled:
            # grouped value heads (k-head major) -> tiled (k-head minor), the column order of a GGUF ssm_out
            r = self.num_v_heads // self.num_k_heads
            out = out.view(total, self.num_k_heads, r, self.head_v_dim).transpose(1, 2).reshape(total, -1)
        return self.out_proj.forward(out)


__all__ = ["GatedDeltaNet", "SpecGdnStash", "channels_first_copy"]
