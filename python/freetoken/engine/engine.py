from __future__ import annotations

import gc
import math
import os
import re
import time
from datetime import timedelta
from typing import Any, Dict, Iterable, Iterator, NamedTuple, Tuple

import torch
from freetoken.attention import AttnType, attention_backend_info, create_attention_backend
from freetoken.core import Batch, Context, Req, set_global_ctx
from freetoken.distributed import (
    destroy_distributed,
    enable_pynccl_distributed,
    set_pp_info,
    set_tp_info,
)
from freetoken.gpu_select import gpu_identity
from freetoken.layers import set_rope_device
from freetoken.layers.quantization import LayerKind, QuantBackend, finalize_quant, set_quant_backend
from freetoken.moe.offload_cache import iter_offload_moe_layers
from freetoken.models import create_model, load_weight
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.expert_banks import load_expert_banks
from freetoken.moe.host_banks import HostResidency as _HostResidency, PinFailed
from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache
from freetoken.utils import (
    align_ceil,
    init_logger,
    is_pre_ampere,
    is_sm90_family,
    is_sm100_family,
    mem_GB,
    torch_dtype,
)

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory
from .sample import BatchSamplingArgs, Sampler
from .spec import (
    SpecResult,
    accept_drafts,
    pack_spec_message,
    spec_message_len,
    unpack_spec_message,
)
from .spec_graph import spec_graph_applicable
from freetoken.kvcache import create_kv_pool, resolve_pool_class
from freetoken.kvcache.base import CacheRebuildRejected
from freetoken.kvcache.cache_status import _supports_swa_ratio
from freetoken.kvcache.linear_state_pool import (
    _linear_pool_min_slots, _linear_pool_num_slots, state_pool_bytes,
)

logger = init_logger(__name__)


def _require_offload_cache_size(cache_size: int, num_experts: int) -> None:
    """The offload MoE cache needs at least one slot per expert per layer. A too-small size
    (e.g. a bare offload run with moe_cache_size unset and auto disabled) must fail loudly."""
    if cache_size < num_experts:
        raise ValueError(
            f"moe_cache_size={cache_size} is too small: need at least num_experts={num_experts} "
            f"slots. Pass --moe-cache-size/--moe-cache-rate, or use --moe-cache-auto "
            f"(the default for offload/hybrid backends when no cache-sizing flag is given; "
            f"--moe-strategy cpu always sizes its own fixed two-layer buffer and ignores "
            f"cache-sizing flags)."
        )


def _flashinfer_available() -> bool:
    from freetoken.kernel.backend import is_flashinfer_installed

    return is_flashinfer_installed()


def _sgl_flash_attn_available() -> bool:
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache  # noqa: F401
    except Exception as exc:
        detail = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
        logger.warning_rank0(
            "sgl_kernel.flash_attn is unavailable; auto attention backend falls back to fi "
            f"({type(exc).__name__}: {detail})"
        )
        return False
    return True


def _startup_kv_budget(memory_ratio: float, init_free_memory: int, new_free_memory: int) -> int:
    """Bytes available to the KV pool at startup: ratio-scaled pre-load free memory minus
    what the resident model consumed. Kept as a pure function so the composition with the
    pool families' ``solve_num_pages`` stays CPU-testable."""
    return int(memory_ratio * init_free_memory) - (init_free_memory - new_free_memory)


def _page_table_width(max_seq_len: int, page_size: int) -> int:
    """Column count for the page table. ``_write_page_table`` writes WHOLE trailing pages, so the
    highest column touched is ``align_ceil(max_seq_len, page_size) - 1`` -- which the 32-alignment
    alone does not cover once page_size > 32 (an unaligned --max-seq-len-override on DSV4's P=128
    or trtllm's forced 64 would index past the row)."""
    return align_ceil(align_ceil(max_seq_len, page_size), 32)


def _required_attn_types(model_config) -> frozenset[AttnType]:
    """Backend-driving attention types of this model, from the group-spec walk
    (single source shared with the pool factory and the KV cost model). getattr
    fallbacks: duck-typed test configs may not implement the spec walk; for those,
    dsv4_args marks DSV4 (the real config declares a DSV4 attention group)."""
    specs_fn = getattr(model_config, "kv_cache_group_specs", None)
    if specs_fn is None:
        if getattr(model_config, "dsv4_args", None) is not None:
            return frozenset({AttnType.DSV4})
        return frozenset({AttnType.FULL})
    types = frozenset(
        spec.attn_type for spec in specs_fn() if spec.attn_type.backend_driven
    )
    return types or frozenset({AttnType.FULL})


def _backend_parts_serve(name: str, required: frozenset[AttnType]) -> bool:
    return all(
        required <= attention_backend_info(part).supported_types
        for part in name.split(",")
    )


def _backend_requirements_met(name: str) -> bool:
    # flashinfer first across ALL parts: the sgl probe logs a "falls back to fi" warning,
    # which would mislead when the candidate is about to fail on flashinfer anyway.
    infos = [attention_backend_info(part) for part in name.split(",")]
    if any(i.requires_flashinfer for i in infos) and not _flashinfer_available():
        return False
    if any(i.requires_sgl_kernel for i in infos) and not _sgl_flash_attn_available():
        return False
    if any(i.requires_sm100 for i in infos) and not is_sm100_family():
        return False
    return True


def _resolve_auto_attention_backend(required: frozenset[AttnType]) -> str:
    """First candidate (in per-type priority order) whose arch condition holds,
    whose packages are installed, and whose every comma part serves ALL required
    types. Reproduces the historical hardware tree for FULL-only models:
    sm_100 -> trtllm, sm_90+sgl_kernel -> "fa,fi", flashinfer -> fi, else triton."""
    candidates: list[tuple[str, bool]] = []
    if AttnType.DSV4 in required:
        candidates.append(("dsv4_sparse", True))
    if required & {AttnType.MLA, AttnType.DSA}:
        candidates.append(("dsa", True))
    if AttnType.BSA in required:
        candidates.append(("m3_sparse", True))
    if AttnType.QSA in required:
        candidates.append(("qsa_sparse", True))
    if AttnType.SWA in required:
        candidates.append(("triton", True))
    if AttnType.FULL in required:
        candidates += [
            ("trtllm", is_sm100_family()),
            ("fa,fi", is_sm90_family()),
            # flashinfer's JIT attention fails on Turing (sm_75) at head_dim 256 ("unspecified
            # launch failure" in BatchPrefillWithPagedKVCache); the Triton backend serves it.
            # An explicit --attention-backend fi is still accepted.
            ("fi", not is_pre_ampere()),
            ("triton", True),
        ]
    for name, arch_ok in candidates:
        if not arch_ok:
            continue
        if not _backend_parts_serve(name, required):
            continue
        if not _backend_requirements_met(name):
            continue
        return name
    raise RuntimeError(
        "No attention backend can serve attention types "
        f"{sorted(t.value for t in required)} on this machine."
    )


def _validate_attention_backend_choice(config, override, required: frozenset[AttnType]) -> None:
    """Config-time type x backend capability check for the resolved (or explicit)
    backend string: every comma part must serve every required type and have its
    packages/arch available. Replaces the per-model gates; in particular this is
    where a DSV4 or MLA checkpoint rejects a generic backend before weights load,
    and where a generic model rejects dsa/dsv4_sparse."""
    from freetoken.attention import validate_attn_backend

    # Name membership first (ArgumentTypeError listing the supported names): the CLI already
    # ran this, but the programmatic EngineConfig path reaches here unvalidated and would
    # otherwise die on a bare KeyError from the info lookup below.
    validate_attn_backend(config.attention_backend, allow_auto=False)

    model_config = config.model_config
    backend_parts = [p.strip() for p in config.attention_backend.split(",")]
    for part in backend_parts:
        info = attention_backend_info(part)
        missing = required - info.supported_types
        if missing:
            valid = [
                name
                for name in (
                    "fa", "fi", "trtllm", "triton", "dsa", "dsv4_sparse", "m3_sparse",
                    "qsa_sparse",
                )
                if required <= attention_backend_info(name).supported_types
            ]
            missing_names = "/".join(sorted(t.value for t in missing))
            raise ValueError(
                f"{getattr(model_config, 'model_type', 'model')} uses {missing_names} "
                f"attention, which backend {part!r} does not support; valid backends: "
                f"{', '.join(valid)} (or auto), got {config.attention_backend!r}."
            )
        if AttnType.SWA in required and not info.consumes_attn_spec:
            # SWA models drive window/sinks/sm_scale through the per-call AttentionSpec;
            # a backend that drops it would attend with the wrong window silently.
            raise ValueError(
                f"backend {part!r} does not consume the per-call AttentionSpec that "
                f"SWA models require, got {config.attention_backend!r}."
            )

    # --kv-cache-dtype: only these backends dequantize the code slabs. Any other backend
    # would read the uint8 codes as 16-bit floats -- no exception, just wrong numbers --
    # and "auto" resolves to flashinfer on sm_80+, so this is the common case on every card
    # newer than the one it was developed on, not an exotic one.
    #
    # qsa_sparse belongs here and the list is not interchangeable with it: a Flash-Next
    # checkpoint resolves to qsa_sparse and CANNOT run on triton (triton serves FULL/SWA,
    # not QSA), so demanding triton here would make the flag unreachable on exactly the
    # model family it suits best.
    from freetoken.kvcache.kv_quant import resolve as _resolve_kv_quant

    _DEQUANTIZING_BACKENDS = ("triton", "qsa_sparse")
    if _resolve_kv_quant(getattr(config, "kv_cache_dtype", None)) is not None:
        wrong = [p for p in backend_parts if p not in _DEQUANTIZING_BACKENDS]
        if wrong:
            raise ValueError(
                f"--kv-cache-dtype {config.kv_cache_dtype} is only read by the "
                f"{' and '.join(_DEQUANTIZING_BACKENDS)} attention backends; got "
                f"{config.attention_backend!r}. Pass --attention-backend triton (or let a "
                f"Flash-Next checkpoint resolve to qsa_sparse), or drop --kv-cache-dtype."
            )

    # An explicitly-selected backend may require a package that isn't installed. Auto
    # never resolves to one of these when its package is missing, so this only fires for
    # explicit --attention-backend choices.
    for part in backend_parts:
        info = attention_backend_info(part)
        if info.requires_flashinfer and not _flashinfer_available():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires flashinfer, which is "
                "not installed. Install it with `pip install 'freetoken[fi]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sgl_kernel and not _sgl_flash_attn_available():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires sgl_kernel, which is "
                "not installed. Install it with `pip install 'freetoken[sgl]'` (or "
                "'freetoken[accel]'), or use --attention-backend triton."
            )
        if info.requires_sm100 and not is_sm100_family():
            raise RuntimeError(
                f"Attention backend {config.attention_backend!r} requires a compute capability "
                "10.x GPU: flashinfer's trtllm-gen kernels ship sm_100a/103a cubins only. "
                "Use --attention-backend fi (or triton) instead."
            )

    if required & {AttnType.MLA, AttnType.DSA}:
        # Plain MLA/DSA runs on page_size 1; the kpool indexer layout needs 64.
        _kpool_ratio = max(
            (s.index_ratio for s in model_config.kv_cache_group_specs() if s.mla),
            default=1,
        )
        want_page = 64 if _kpool_ratio > 1 else 1
        if config.page_size != want_page:
            logger.warning_rank0(
                f"Page size {config.page_size} is auto-adjusted to {want_page} "
                f"for latent-KV attention."
            )
            override("page_size", want_page)

    for part in backend_parts:
        info = attention_backend_info(part)
        if info.page_sizes is not None and config.page_size not in info.page_sizes:
            override("page_size", info.page_sizes[-1])
            logger.warning_rank0(
                f"Page size is overridden to {info.page_sizes[-1]} for the {part} backend"
            )


def _to_pinned_host(t: torch.Tensor) -> torch.Tensor:
    """A pinned + mapped host copy of ``t`` (FreeToken's cudaHostAlloc: Portable | Mapped, so
    the GPU can dereference it in place); plain host memory when there is no CUDA allocator
    (CPU-only tests)."""
    t = t.to("cpu")
    try:
        from freetoken.kernel.pinned import copy_to_pinned_tensor

        return copy_to_pinned_tensor(t.contiguous())
    except Exception:  # noqa: BLE001
        return t


def _make_dummy_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    *,
    device: torch.device,
    host_prefixes: Tuple[str, ...] = (),
) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    for key, param in model_state.items():
        if host_prefixes and key.startswith(host_prefixes):
            state_dict[key] = _to_pinned_host(torch.randn(param.shape, dtype=param.dtype))
        elif param.dtype in fp8_dtypes:
            # torch.randn is not implemented for fp8; fill via a uint8 view with small
            # codes (avoid NaN/inf fp8 encodings). Lets dummy-weight startup work for
            # block-fp8 models (the dense fp8 linears are fp8 regardless of moe_strategy).
            t = torch.empty(param.shape, dtype=param.dtype, device=device)
            t.view(torch.uint8).random_(0, 16)
            state_dict[key] = t
        elif param.dtype.is_floating_point or param.dtype.is_complex:
            state_dict[key] = torch.randn(param.shape, dtype=param.dtype, device=device)
        elif param.dtype == torch.uint8 and key.endswith("weight_scale_inv"):
            # MXFP8 e8m0 exponent codes: 127 encodes scale 1.0; zeros would collapse
            # every scale to 2^-127 and zero the model. Scoped BY NAME: other uint8
            # buffers are packed payloads whose bytes mean something else entirely
            # (GGUF qweight blocks embed fp16 scales -- 0x7F7F is fp16 NaN), so they
            # keep the benign all-zeros fill below.
            state_dict[key] = torch.full(param.shape, 127, dtype=param.dtype, device=device)
        else:
            state_dict[key] = torch.zeros(param.shape, dtype=param.dtype, device=device)
    return state_dict


def _keep_local_weights(
    weights: Iterable[Tuple[str, torch.Tensor]], model_state: Dict[str, torch.Tensor]
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Pipeline ranks: drop the tensors of layers another rank serves before they are moved
    to this GPU. An unknown key that IS a local layer's still surfaces in load_state_dict."""
    dropped = 0
    for key, weight in weights:
        if key in model_state:
            yield key, weight
        else:
            dropped += 1
            del weight
    logger.info(f"pipeline: skipped {dropped} tensors served by other ranks")


def _remap_weights(weights, remap, model_state: Dict[str, torch.Tensor]):
    for key, weight in weights:
        yield from remap(key, weight, model_state)


def _drop_unknown_mtp(weights, model_state: Dict[str, torch.Tensor]):
    """The reader keeps the checkpoint's mtp.* head; a process that did not build the draft
    head (spec off, or not the last pipeline rank) drops those tensors here."""
    dropped = 0
    for key, weight in weights:
        if key.startswith("mtp.") and key not in model_state:
            dropped += 1
            del weight
            continue
        yield key, weight
    if dropped:
        logger.info_rank0(f"skipped {dropped} MTP head tensors (draft head not built on this rank)")


def _quantize_at_load(
    weights: Iterable[Tuple[str, torch.Tensor]], model_state: Dict[str, torch.Tensor]
) -> Iterator[Tuple[str, torch.Tensor]]:
    """--dense-quant: a bf16 ``X.weight`` whose model buffer is fp8 and that has a sibling
    ``X.weight_scale`` buffer is quantized per output row on the fly (the checkpoint ships
    it unquantized). Anything else, including tensors already fp8, passes through."""
    from freetoken.kernel.triton.fp8_pertensor_linear import FP8, quantize_fp8_per_row

    quantized = 0
    for key, weight in weights:
        expected = model_state.get(key)
        scale_key = key[: -len(".weight")] + ".weight_scale" if key.endswith(".weight") else None
        if (
            expected is not None
            and expected.dtype == FP8
            and weight.dtype != FP8
            and weight.is_floating_point()
            and scale_key in model_state
        ):
            q, s = quantize_fp8_per_row(weight)
            quantized += 1
            yield key, q
            yield scale_key, s
        else:
            yield key, weight
    logger.info_rank0(f"--dense-quant: quantized {quantized} dense projections to per-row fp8 at load")


def _materialize_loaded_weight_state_dict(
    model_state: Dict[str, torch.Tensor],
    weights: Iterable[Tuple[str, torch.Tensor]],
    *,
    device: torch.device,
    host_prefixes: Tuple[str, ...] = (),
) -> Dict[str, torch.Tensor]:
    """Cast each loaded tensor to its model-buffer dtype on ``device``; keys under
    ``host_prefixes`` (a model's ``host_resident_prefixes``: the embedding table under
    --host-embedding, or the draft head's own embedding copy) land in pinned host memory
    instead."""
    state_dict: Dict[str, torch.Tensor] = {}
    for key, weight in weights:
        expected = model_state.get(key)
        dtype = weight.dtype if expected is None else expected.dtype
        if host_prefixes and key.startswith(host_prefixes):
            state_dict[key] = _to_pinned_host(weight.to(dtype=dtype))
            del weight
            if device.type == "cuda":
                torch.cuda.empty_cache()  # the reader had placed it on the device
        else:
            state_dict[key] = weight.to(device=device, dtype=dtype)
    return state_dict


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    # --spec-mtp: the committed tokens of this step and the drafts for the next window
    spec: Any = None


# the stacked bf16 experts of the draft head, as the reader yields them
_MTP_EXPERT_RE = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.(gate_up_proj|down_proj)$")

# FT_SPEC_TRACE=n: log the first n verify windows (ids, samples, drafts, top logits)
_SPEC_TRACE_LEFT = [int(os.environ.get("FT_SPEC_TRACE", "0") or 0)]
# FT_SPEC_CHECK_STEP=n: cross-check the first n one-row verify windows (FT_SPEC_MAX_DRAFTS=0)
# against the plain decode path from the same state: per-layer residual stream, final logits,
# and the GDN state each path leaves behind
_SPEC_CHECK_STEP_LEFT = [int(os.environ.get("FT_SPEC_CHECK_STEP", "0") or 0)]
# FT_SPEC_PROFILE=1: synchronize between the phases of a verify step and log their mean wall
# time every 20 steps (the syncs themselves add a little, so read it as a breakdown, not a total)
_SPEC_PROFILE = os.environ.get("FT_SPEC_PROFILE") == "1"
# FT_STEP_PROFILE=1: the same phase timer for every decode step (and verify window), on every
# pipeline rank -- where a step's time goes between the ranks' forwards and the hand-offs
_STEP_PROFILE = os.environ.get("FT_STEP_PROFILE") == "1"


class _SpecProfiler:
    """Phase timer for the verify step: ``mark(name)`` synchronizes the device and charges
    the time since the previous mark to ``name``; ``step()`` closes a step and logs the
    per-phase means every ``every`` steps."""

    def __init__(self, every: int = 20) -> None:
        self.every = every
        self.totals: dict[str, float] = {}
        self.order: list[str] = []
        self.steps = 0
        self._t = None

    def start(self) -> None:
        torch.cuda.synchronize()
        self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        if self._t is None:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        if name not in self.totals:
            self.totals[name] = 0.0
            self.order.append(name)
        self.totals[name] += now - self._t
        self._t = now

    def step(self) -> None:
        self._t = None
        self.steps += 1
        if self.steps % self.every:
            return
        n = self.steps
        total = sum(self.totals.values())
        parts = " ".join(f"{k}={1000 * v / n:.1f}" for k, v in ((k, self.totals[k]) for k in self.order))
        logger.warning(f"step profile (ms/step over {n} steps, total {1000 * total / n:.1f}): {parts}")
        self.totals = {k: 0.0 for k in self.order}
        self.steps = 0


def _traceback_tail(exc: BaseException, lines: int = 14) -> str:
    """The last frames of an exception's traceback (capture-failure diagnostics)."""
    import traceback

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip().splitlines()
    return "\n".join(tb[-lines:])


def _expand_sampling_args(args: BatchSamplingArgs, rows: int) -> BatchSamplingArgs:
    """Per-row sampling params for a verify window (one request, ``rows`` logits rows)."""
    if args.temperatures is None or args.temperatures.numel() == rows:
        return args

    def ex(t):
        return None if t is None else t.expand(rows).contiguous()

    return BatchSamplingArgs(ex(args.temperatures), top_k=ex(args.top_k), top_p=ex(args.top_p))


def _pinned_empty(shape, dtype: torch.dtype) -> torch.Tensor:
    """An uninitialized pinned host tensor (allocated as bytes: the pinned allocator knows
    uint8 for sure)."""
    from freetoken.kernel.pinned import alloc_pinned_tensor

    numel = 1
    for d in shape:
        numel *= int(d)
    raw = alloc_pinned_tensor(numel * torch.empty((), dtype=dtype).element_size(), dtype=torch.uint8)
    return raw.view(dtype).view(*shape)


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        if config.is_pp:
            # Pipeline: the ranks split the LAYERS, so every layer sees TP=1; the PP info is
            # what the model, the bank loaders and rank-0 logging consult.
            start, end = config.pp_layer_range
            set_pp_info(
                rank=config.tp_info.rank, size=config.tp_info.size,
                start=start, end=end, num_layers=config.full_model_config.num_layers,
            )
            set_tp_info(rank=0, size=1)
        else:
            set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        self.pp_comm = None  # set by _init_communication under --parallel pp
        # --spec-mtp: draft depth; the stacked bf16 MTP experts captured at load, quantized into
        # the offload cache's extra bank layer (see _append_mtp_bank)
        self.spec_k = int(getattr(config, "spec_mtp", 0) or 0)
        self._spec_profiler = None
        self._spec_graph = None  # SpecVerifyGraph once captured (end of __init__)
        self._mtp_raw: Dict[str, torch.Tensor] = {}
        self._mtp_bank_host: Dict[str, torch.Tensor] | None = None
        self._mtp_bank_bytes = 0
        self._mtp_bank_layers = 0
        if self.spec_k > 0:
            from freetoken.kvcache import qsa_pool as _qsa_pool

            _qsa_pool.SPECULATIVE_TOKENS = self.spec_k  # before the pool exists
        set_quant_backend(_adjust_ftw_quant_backend(config.model_path, QuantBackend.parse(config.quant_backend)))
        _ensure_expandable_segments()  # before the first CUDA allocation below

        from freetoken.gpu_select import bind_assigned_gpu

        self.device = bind_assigned_gpu(config.tp_info.rank)
        _adjust_config(config)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.config = config  # retained for runtime cache rebuild (rebuild_runtime_cache)
        # KV pool family fixed at construction from the model config: its classmethods own the
        # page-token geometry and cost arithmetic the engine needs BEFORE the pool exists
        # (num_pages sizing, --moe-cache-auto); the instance owns rebuild/validation after.
        self._pool_cls = resolve_pool_class(config.model_config)
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        self.tp_cpu_group = self._init_communication(config)
        free_min, free_max = self._sync_get_memory()
        init_free_memory = free_max  # startup KV sizing keeps cross-rank MAX (unchanged)
        self._baseline_free = free_min  # rebuild baseline: cross-rank MIN, deterministic across ranks
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        if self.spec_k > 0 and config.pp_is_last and getattr(self.model, "mtp", None) is None:
            raise ValueError(
                f"--spec-mtp: {type(self.model).__name__} builds no MTP draft head (supported: "
                "the Qwen3.5-MoE family and Qwen3.8-Flash-Next, with mtp.* tensors in the checkpoint)"
            )
        if self.pp_comm is not None:
            width = getattr(self.model, "pp_hidden_width", None)
            if width is None:
                raise NotImplementedError(
                    f"--pp-size is not supported for {type(self.model).__name__}: the "
                    "model does not declare pp_hidden_width / a layer-split forward"
                )
            self.pp_comm.configure(int(width), self.dtype)
            logger.info(
                f"pipeline rank {config.tp_info.rank}/{config.tp_info.size}: layers "
                f"[{config.pp_layer_range[0]}, {config.pp_layer_range[1]}) on {self.device}"
            )
        self.model.load_state_dict(self._load_weight_state_dict(config))
        finalize_quant(self.model)
        post_weights_free = self._sync_get_memory()[0]
        self._weights_bytes = self._baseline_free - post_weights_free
        # Pool-budget baseline for the desktop cache sliders: free VRAM after the weights are
        # resident but before ANY runtime cache pool (MoE expert cache below, KV pages, GDN
        # state) is allocated. This is the stable "if all free VRAM went to one pool" budget —
        # unlike a query-time mem_get_info it doesn't drift with allocator caching, CUDA
        # graphs, or other processes. Cross-rank MIN, deterministic across ranks.
        self._post_weights_free = post_weights_free
        self.moe_offload_cache = None
        self.cpu_moe_executor = None
        # Host-side auxiliary stores (qwen4_exp's pinned PLE table): after the weights so a
        # load failure is not masked, before the MoE offload cache so the bank residency
        # planning sees the pin quota the table already spent.
        self._host_tables_bytes = 0
        if hasattr(self.model, "load_host_tables"):
            self._host_tables_bytes = int(self.model.load_host_tables(config) or 0)
        self._host_tables_bytes += self._mtp_bank_bytes  # the draft head's pinned bank layer
        self._host_tables_bytes += getattr(self, "_host_resident_bytes", 0)  # --host-embedding
        if is_offload_moe_strategy(config.moe_strategy):
            self._init_offload_moe_cache(config)
        if hasattr(self.model, "prepare_for_runtime"):
            self.model.prepare_for_runtime()

        # ======================= KV cache initialization ========================
        new_free = self._sync_get_memory()[1]
        # The engine measures the budget and settles the sibling GDN state pool's bytes
        # off it; the KV pool family owns every geometry-specific formula behind the rest.
        available_memory = _startup_kv_budget(config.memory_ratio, init_free_memory, new_free)
        available_memory -= state_pool_bytes(config)
        self.num_pages = self._pool_cls.solve_num_pages(config, available_memory)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kv_pool(
            config, self.num_pages, device=self.device, dtype=self.dtype
        )

        # ======================= Linear (GatedDeltaNet) state initialization ========================
        linear_group = config.model_config.linear_attention_group()
        if linear_group is not None:
            from freetoken.kvcache.linear_state_pool import LinearStatePool

            self.linear_state_pool = LinearStatePool(
                group=linear_group,
                num_slots=_linear_pool_num_slots(config),
                dtype=self.dtype,
                device=self.device,
                tp_size=config.tp_size,
                slot_states=config.model_config.slot_states,
            )
            self.ctx.linear_state_pool = self.linear_state_pool
        else:
            self.linear_state_pool = None

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )
        # Pools routed by the shared table but deriving reads through their own mappings (DSV4)
        # re-point here (and again on any table realloc). The graph-input snapshot that reads
        # through them belongs to the attention backend, built later in init_capture_graph.
        self.kv_cache.attach_page_table(self.page_table)

        # ======================= Attention backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")
        self._preallocate_prefill_scratch(config)

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # padded/dummy rows index the GDN padding slot (0) so gather/scatter hits scratch.
        if self.linear_state_pool is not None:
            self.dummy_req.linear_slot_idx = self.linear_state_pool.padding_slot
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
            pp_hidden=(
                (self.pp_comm.hidden_width, self.dtype)
                if self.pp_comm is not None and not self.pp_comm.is_first
                else None
            ),
        )
        # pre-map before the prefill warmup so its persistent buffers come from the cache
        self._premap_vram()
        # Chunk size decides the prefill transient, so it has to be settled before anything
        # measures or warms at that size. Backend-independent: the buffers are the model's.
        self._autosize_prefill_chunk(config)
        if config.attention_backend.split(",")[0] == "triton":
            # Prefill runs on the first comma part; warm its autotune cache.
            self._warmup_prefill()
        if self.spec_k > 0:
            if self._premap_enabled():
                # the verify-window graphs carve their own pools: hand the pre-mapped memory
                # back for the capture, then take the remainder again below
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
            self._capture_spec_graph()
            self._premap_vram()

    @staticmethod
    def _premap_enabled() -> bool:
        return os.environ.get("FREETOKEN_PREMAP_VRAM") == "1"

    def _preallocate_prefill_scratch(self, config: EngineConfig) -> None:
        """Below Ampere the prefill paths dequantize into persistent scratches (MoE experts in
        chunks, fp8 projections). Take them now, with the pools sized and the GPU idle, instead
        of inside the first prefill: on a full card under WSL2 the allocator's segment growth
        under load failed intermittently with ``CUDA driver error: device not ready``."""
        from freetoken.kernel.triton.fp8_pertensor_linear import _scratch_gemm_preferred
        from freetoken.moe.fused_nvfp4 import _scratch_moe_preferred

        mc = config.model_config
        taken = 0
        torch.cuda.synchronize(self.device)
        if _scratch_moe_preferred() and getattr(mc, "expert_quant", "none") == "nvfp4" and mc.moe_enabled:
            from freetoken.moe.fused_nvfp4 import preallocate_scratch

            taken += preallocate_scratch(mc.hidden_size, mc.moe_intermediate_size, self.dtype, self.device)
        if _scratch_gemm_preferred():
            fp8 = [t.numel() for t in self.model.state_dict().values() if t.dtype == torch.float8_e4m3fn]
            if fp8:
                from freetoken.kernel.triton.fp8_pertensor_linear import preallocate_scratch as prealloc_fp8

                taken += prealloc_fp8(max(fp8), self.dtype, self.device)
        if taken:
            torch.cuda.synchronize(self.device)
            logger.info(
                f"pre-Ampere prefill scratches allocated: {mem_GB(taken)} "
                f"(free {mem_GB(torch.cuda.mem_get_info(self.device)[0])})"
            )

    def _premap_vram(self) -> None:
        """Opt-in (``FREETOKEN_PREMAP_VRAM=1``): once every pool and graph exists, take all but
        ``FREETOKEN_VRAM_HEADROOM_MB`` (default 96) of the remaining free VRAM into the caching
        allocator and release it there, so serving-time allocations are served from cached
        segments and never grow or shrink them through the driver. On WSL2 with a full card
        (RTX 2060 6 GB) segment growth under load intermittently failed with ``CUDA driver
        error: device not ready``; pre-mapping removes those driver calls from the hot path."""
        if not self._premap_enabled():
            return
        headroom = int(os.environ.get("FREETOKEN_VRAM_HEADROOM_MB", "96") or 0) * 2**20
        torch.cuda.synchronize(self.device)
        free = torch.cuda.mem_get_info(self.device)[0]
        size = free - headroom
        if size <= 0:
            logger.info(f"FREETOKEN_PREMAP_VRAM: nothing to pre-map ({mem_GB(free)} free)")
            return
        try:
            block = torch.empty(size, dtype=torch.uint8, device=self.device)
            del block  # stays cached in the allocator
        except RuntimeError as exc:  # noqa: BLE001
            logger.warning(f"FREETOKEN_PREMAP_VRAM: could not pre-map {mem_GB(size)}: {exc!r}")
            return
        logger.info(
            f"FREETOKEN_PREMAP_VRAM: pre-mapped {mem_GB(size)} into the allocator cache "
            f"({mem_GB(headroom)} of headroom left to the driver)"
        )

    def _capture_spec_graph(self) -> None:
        """Capture the K+1-row verify window as a CUDA graph (see engine/spec_graph), then the
        draft head's window pass and chain step. Needs the decode graphs enabled (same
        static-buffer machinery) and an attention backend that stages the window;
        FT_SPEC_NO_GRAPH=1 keeps the eager path for A/B runs, FT_SPEC_NO_MTP_GRAPH=1 keeps the
        head eager. A capture failure logs and falls back to the eager path."""
        from .spec_graph import SpecVerifyGraph

        self._spec_graph = None
        if os.environ.get("FT_SPEC_NO_GRAPH") == "1":
            logger.info("--spec-mtp: FT_SPEC_NO_GRAPH=1, the verify window stays eager")
            return
        if self.graph_runner.max_graph_bs == 0 or not hasattr(self.attn_backend, "stage_spec"):
            logger.info(
                "--spec-mtp: no CUDA graph for the verify window (decode graphs disabled, or the "
                f"attention backend {type(self.attn_backend).__name__} does not stage it); eager"
            )
            return
        rows = self.spec_k + 1
        try:
            self.attn_backend.init_spec_capture(rows)
            sg = SpecVerifyGraph(self, rows)
            sg.capture()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"--spec-mtp: verify-window graph capture failed, staying eager: {exc!r}\n"
                + _traceback_tail(exc)
            )
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.reset()
            return
        self._spec_graph = sg
        if self.model.mtp is not None and os.environ.get("FT_SPEC_NO_MTP_GRAPH") != "1":
            try:
                sg.capture_mtp()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"--spec-mtp: draft-head graph capture failed, the head stays eager: {exc!r}\n"
                    + _traceback_tail(exc)
                )
                sg.g_window = sg.g_chain = None
                if self.moe_offload_cache is not None:
                    self.moe_offload_cache.reset()
        # The window graph keeps the K+1 per-token GDN states of every layer resident (~250 MB
        # for a 30-layer GDN stack at K=3). On a card that is left with almost no free VRAM after
        # that, the eager parts of every step (sampling, staging, the offload cache's fetches)
        # fight the allocator and the step gets slower than eager -- measured on an RTX 2060
        # 6 GB (145 ms eager vs 185 ms graph with 0.1 GiB free). Keep the graphs only with
        # headroom; FT_SPEC_GRAPH_MIN_FREE_MB overrides the floor (0 = always keep).
        min_free_mb = int(os.environ.get("FT_SPEC_GRAPH_MIN_FREE_MB", "256") or 0)
        torch.cuda.synchronize(self.device)
        free_mb = torch.cuda.mem_get_info(self.device)[0] // 2**20
        if free_mb < min_free_mb:
            logger.warning(
                f"--spec-mtp: only {free_mb} MB of VRAM free after the verify-window graphs "
                f"(floor {min_free_mb} MB): dropping them, the window and the head run eagerly "
                "(FT_SPEC_GRAPH_MIN_FREE_MB=0 keeps them)"
            )
            self._spec_graph = None
            del sg
            gc.collect()
            torch.cuda.empty_cache()

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        if config.is_pp:
            # Layer split: the ranks never all-reduce; the residual stream and the sampled
            # tokens go point-to-point over gloo (no NCCL / P2P needed, see distributed/pipeline).
            from freetoken.distributed import get_pp_info
            from freetoken.distributed.pipeline import PipelineComm

            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            self.pp_comm = PipelineComm(get_pp_info(), tp_cpu_group, self.device)
            return tp_cpu_group
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        model_state = self.model.state_dict()
        host_prefixes = tuple(getattr(self.model, "host_resident_prefixes", ()))
        if host_prefixes:
            logger.info(f"host-resident weights (pinned RAM, gathered by the GPU in place): {host_prefixes}")
        if config.use_dummy_weight:
            return _make_dummy_weight_state_dict(model_state, device=self.device, host_prefixes=host_prefixes)
        # _materialize casts each loaded tensor to its model-param dtype (model_state), so
        # models declaring per-tensor dtypes (e.g. DSV4's mixed fp8/fp32/bf16) are preserved;
        # offload models exclude experts (served from the offload cache, not dense weights).
        has_mtp = getattr(self.model, "mtp", None) is not None
        weights = load_weight(
            config.model_path,
            self.device,
            include_moe_experts=not is_offload_moe_strategy(config.moe_strategy),
            include_mtp=has_mtp,
        )
        remap = getattr(self.model, "remap_loaded_weight", None)
        if remap is not None:
            # model-declared renames/splits (e.g. the fp8 GDN's in_proj -> qkvz | ba) come
            # first so the pipeline filter and the quantizer see the model's own keys
            weights = _remap_weights(weights, remap, model_state)
        if has_mtp:
            weights = self._capture_mtp_experts(weights)
        weights = _drop_unknown_mtp(weights, model_state)
        if config.is_pp:
            # The reader yields the whole model; keep only the tensors this rank's layer
            # window declares (other ranks' layers, and the embedding / head it does not own).
            weights = _keep_local_weights(weights, model_state)
        if config.dense_quant != "none":
            weights = _quantize_at_load(weights, model_state)
        state = _materialize_loaded_weight_state_dict(
            model_state, weights, device=self.device, host_prefixes=host_prefixes
        )
        if has_mtp:
            self._quantize_mtp_experts()
        # the pinned tables count against the host pin quota the bank residency planner sees
        self._host_resident_bytes = sum(
            t.numel() * t.element_size() for k, t in state.items() if host_prefixes and k.startswith(host_prefixes)
        )
        return state

    # ------------------------------------------------------------------ MTP expert bank layer
    def _capture_mtp_experts(self, weights):
        """Pull the head's stacked bf16 experts (``mtp.layers.0.mlp.experts.{gate_up,down}_proj``)
        out of the weight stream: they become a bank layer, not model buffers."""
        for key, weight in weights:
            m = _MTP_EXPERT_RE.match(key)
            if m is None:
                yield key, weight
                continue
            self._mtp_raw[m.group(1)] = weight.to("cpu")
            del weight

    def _quantize_mtp_experts(self) -> None:
        """Quantize the captured bf16 experts to the native NVFP4 bank layout, into pinned host
        tensors -- right after the dense weights, so the bf16 copy (1.6 GB for 256 x 512) is gone
        before the expert banks load. ``_append_mtp_bank`` hands the result to the cache."""
        from freetoken.kernel.triton.nvfp4_quant import nvfp4_expert_bank_specs, quantize_nvfp4_experts

        raw = self._mtp_raw
        assert set(raw) == {"gate_up_proj", "down_proj"}, (
            f"--spec-mtp: MTP experts missing from the checkpoint: got {sorted(raw)}"
        )
        gate_up, down = raw["gate_up_proj"], raw["down_proj"]
        if gate_up.shape[-1] != down.shape[-2]:  # [E, H, 2I] / [E, I, H] storage -> [E, 2I, H] / [E, H, I]
            gate_up, down = gate_up.transpose(1, 2).contiguous(), down.transpose(1, 2).contiguous()
        e, two_i, h = gate_up.shape
        host = {
            n: _pinned_empty(shape, dt)
            for n, (shape, dt) in nvfp4_expert_bank_specs(e, h, two_i // 2).items()
        }
        quantize_nvfp4_experts(gate_up, down, chunk=8, device=self.device, out=host)
        del gate_up, down
        self._mtp_raw = {}
        self._mtp_bank_host = host
        self._mtp_bank_bytes = sum(t.numel() * t.element_size() for t in host.values())
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        logger.info(
            f"MTP draft head: {e}-expert layer quantized to NVFP4 "
            f"({self._mtp_bank_bytes / 2**20:.0f} MB pinned)"
        )

    def _append_mtp_bank(self, banks) -> int:
        """Append the head's quantized experts as the offload cache's last layer (the head's
        OffloadMoELayer indexes it by ``mtp_layer_id``). Returns the layers appended (0 or 1)."""
        host = self._mtp_bank_host
        if host is None:
            return 0
        from freetoken.moe.host_banks import HostResidency

        assert banks.quant_format == "nvfp4", (
            f"the MTP expert bank is written in the native nvfp4 layout; this run uses "
            f"{banks.quant_format!r} (use --moe-strategy hybrid/cpu, or --quant-backend moe.nvfp4=triton)"
        )
        for name, per_layer in banks.sources.items():
            t = host[name]
            assert t.shape[1:] == per_layer[0].shape[1:] and t.dtype == per_layer[0].dtype, (
                name, t.shape, per_layer[0].shape, t.dtype, per_layer[0].dtype
            )
            per_layer.append(t)
        if banks.layer_residency is not None:
            banks.layer_residency.append(HostResidency.PINNED.value)
        self._mtp_bank_host = None
        logger.info("MTP draft head: its expert layer appended to the offload cache")
        return 1


    def _resolve_auto_moe_cache_size(self, config: EngineConfig, banks, method=None) -> tuple[int, int, bool]:
        """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

        Pure glue over the Phase-1 budget policy; isolated here so it is unit-testable
        without a GPU. Reused by the Phase-2 runtime rebuild.
        """
        from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto

        cache_per_page, fixed_cache_size, page_tokens, min_reserve = self._pool_cls.kv_cost(config)
        fixed_cache_size += state_pool_bytes(config)  # sibling GDN state pool, engine-summed
        num_experts = config.model_config.num_experts
        total_experts = config.model_config.num_moe_layers * num_experts
        return resolve_moe_cache_auto(
            baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes,
            memory_ratio=config.memory_ratio,
            cache_per_page=cache_per_page,
            fixed_cache_size=fixed_cache_size,
            per_expert_bytes=expert_bytes_per_slot(banks.sources),
            num_experts=num_experts,
            total_experts=total_experts,
            prefill_overlap=config.moe_prefill_overlap,
            kv_reserve_tokens=max(config.kv_reserve_tokens, min_reserve),
            page_size=page_tokens,
            max_slots=method.slot_limit() if method is not None else None,
        )

    def _init_offload_moe_cache(self, config: EngineConfig) -> OffloadMoeCache:
        method = shared_offload_method(self.model)
        num_moe_layers = config.model_config.num_moe_layers
        cpu_layer_ids = _resolve_cpu_layers(config, num_moe_layers, reserved=self._host_tables_bytes, method=method)
        if not config.moe_bank_ram:
            # --moe-bank-ram answers the pin budget its own way: the banks are a file mapping
            # and only the resident prefix is locked, so banks over the budget are the plan
            _check_pin_budget(config, reserved=self._host_tables_bytes, method=method)
        # the kernels were picked for model_config.decode_target; --moe-cpu-layers auto may still find that every bank fits the pin budget
        decode_target = config.model_config.decode_target
        if decode_target == "cpu" and not cpu_layer_ids:
            decode_target = "gpu"
        # split residency: where pinning is quota-capped (_pin_budget_bytes), pin only the GPU layers' banks and mlock the CPU layers'
        # uncapped hosts keep every bank pinned (CPU decode reads them the same; overlap prefill stays on)
        # not applied to plain --moe-strategy cpu; all-locked under a cap = --moe-strategy offload --moe-cpu-layers 1.0
        split_residency = (
            bool(cpu_layer_ids)
            and config.moe_strategy in ("offload", "hybrid")
            and _pin_budget_bytes(self._host_tables_bytes) is not None
        )
        if config.moe_strategy == "cpu" and not split_residency:
            # cpu mode pins every bank for the prefill double buffer; over the pin cap that dies in cudaHostRegister, so lock everything instead
            from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

            budget = _pin_budget_bytes(self._host_tables_bytes)
            bank_bytes = None
            if budget is not None:
                bank_bytes = ftw_bank_bytes(config.model_path) or bank_bytes_estimate(config.model_config, method)
            if bank_bytes and bank_bytes > budget:
                split_residency = True
                logger.info_rank0(
                    f"--moe-strategy cpu: banks {bank_bytes / 2**30:.2f} GiB exceed the "
                    f"pin budget; OS-locking all layers instead of pinning"
                )
        if split_residency and config.moe_prefill_overlap:
            # locked (unregistered) layers cannot feed the async pinned H2D double buffer; their prefill is a synchronous pageable copy via materialize
            logger.info_rank0(
                "--moe-cpu-layers split residency: disabling MoE prefill overlap "
                "(locked layers prefill via synchronous pageable copies)"
            )
            object.__setattr__(config, "moe_prefill_overlap", False)
        # Fast path: an FTW checkpoint loads its repacked banks directly.
        # Slow path: load_expert_banks auto-picks parallel vs serial baseline by
        # expert-tensor granularity. Both pin-after-fill.
        # --expert-load: serial/parallel force the read; auto (None) lets load_expert_banks
        # pick (parallel for scattered experts, with a low-RAM fallback to serial).
        bank_tier = self._build_bank_tier(config)
        expert_parallel = {"serial": False, "parallel": True}.get(config.expert_load, None)
        requested_residency = None
        if split_residency:
            from freetoken.moe.host_banks import HostResidency

            requested_residency = [
                HostResidency.LOCKED.value if i in cpu_layer_ids
                else HostResidency.PINNED.value
                for i in range(config.model_config.num_moe_layers)
            ]
        try:
            banks = load_expert_banks(
                config.model_path,
                config.model_config,
                method=method,
                device=self.device,
                dtype=self.dtype,
                dummy=config.use_dummy_weight,
                parallel=expert_parallel,
                decode_target=("cpu" if decode_target in ("cpu", "hybrid") else "gpu"),
                layer_residency=requested_residency,
                layer_sink=bank_tier.sink if bank_tier else None,
            )
        except PinFailed as exc:
            raise RuntimeError(f"{exc}; {_pin_hint(self._host_tables_bytes)}") from exc
        if bank_tier is not None:
            bank_tier.finish()
            banks.sources.update(bank_tier.sources)
            if config.moe_prefill_overlap and not (
                bank_tier.banks and bank_tier.banks.registered_bytes
            ):
                # Nothing registered: no part of the bank is a legal async DMA source,
                # so the overlap prefetch cannot run at all. With the prefix registered
                # it can -- prefetch_prefill_layer sends those rows on the copy stream
                # and bounces the rest. Decided here rather than in _build_bank_tier so
                # it keys on what actually got registered, and before --moe-cache-auto
                # so the VRAM plan sees the right answer.
                logger.info_rank0(
                    "--moe-bank-ram: disabling MoE prefill overlap (nothing registered, "
                    "so no part of a mapped bank is a legal async DMA source)"
                )
                object.__setattr__(config, "moe_prefill_overlap", False)
        self._mtp_bank_layers = self._append_mtp_bank(banks)
        if config.moe_cache_auto:
            size, pages, overlap = self._resolve_auto_moe_cache_size(config, banks, method)
            object.__setattr__(config, "moe_cache_size", size)
            object.__setattr__(config, "moe_prefill_overlap", overlap)
            if config.num_page_override is None:
                # Honor the plan's KV half too: MoE slots and KV pages were solved
                # against ONE budget (ratio x baseline - weights), so both must come
                # from it. Re-solving pages later from a fresh free-memory reading
                # double-counts everything allocated since the weights measurement
                # (this expert cache, the CPU-executor GPU buffers, allocator
                # slack) and goes negative whenever the expert fill is exact --
                # a greedy fill leaves no headroom for the measurement delta.
                object.__setattr__(config, "num_page_override", pages)
            logger.info_rank0(
                f"--moe-cache-auto resolved moe_cache_size={size} "
                f"num_pages={pages} (prefill_overlap={overlap})"
            )
        _require_offload_cache_size(config.moe_cache_size, config.model_config.num_experts)
        layout = max_slots = None
        if method is not None:
            if banks.kind is not None and (banks.kind, banks.kernel) != (method.kind, method.kernel.name):
                raise ValueError(
                    f"expert banks were packed for {banks.kind} / {banks.kernel} but the model "
                    f"binds {method.kind} / {method.kernel.name}; reconvert or select that kernel"
                )
            layout = method.layout()
            max_slots = method.slot_limit()
        cache = OffloadMoeCache(
            # Models with leading dense layers (GLM-4) only have experts on the MoE
            # layers; num_moe_layers == num_layers when first_k_dense_replace == 0.
            # --spec-mtp appends the draft head's expert layer.
            num_layers=config.model_config.num_moe_layers + self._mtp_bank_layers,
            num_experts=config.model_config.num_experts,
            cache_size=config.moe_cache_size,
            device=self.device,
            cache_policy=config.moe_cache_policy,
            prefill_overlap=config.moe_prefill_overlap,
            prefill_hit_d2d=config.moe_prefill_hit_d2d,
            quant_format=banks.quant_format,
            decode_target=decode_target,
            hybrid_max_fetch=config.moe_hybrid_max_fetch,
            layout=layout,
            max_slots=max_slots,
        )
        # before set_bank_sources: the residency validation and the copy plan's skip of non-pinned layers key on the CPU-layer set
        cache.cpu_layer_ids = cpu_layer_ids
        mapped_pinned = bank_tier is not None and bool(
            bank_tier.banks and bank_tier.banks.registered_bytes
        )
        if bank_tier is not None and not mapped_pinned:
            # Nothing registered: a mapped bank then has no device address at all, so
            # every layer has to decode on the CPU executor -- which is the state the
            # residency label below declares, and set_bank_sources checks the two agree.
            # This costs the VRAM expert cache, so it is the fallback, not the plan.
            cache.cpu_layer_ids = frozenset(range(len(next(iter(banks.sources.values())))))
        cache.set_bank_sources(
            banks.sources,
            # a mapped bank is mlocked, not registered, which is exactly what LOCKED
            # means here: the CPU executor reads it directly and the GPU movement paths
            # must take their pageable branch
            layer_residency=(
                # length must match the bank sources, which --spec-mtp extends by one
                [
                    (
                        _HostResidency.PINNED if mapped_pinned else _HostResidency.LOCKED
                    ).value
                ]
                * len(next(iter(banks.sources.values())))
                if bank_tier is not None
                else banks.layer_residency
            ),
        )
        cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
        if decode_target == "hybrid":
            self._resolve_hybrid_fetch(config, cache)
        # Must be set before CUDA graph capture so the (device-side) accumulation ops are
        # captured and re-run on every decode replay.
        if bank_tier is not None:
            bank_tier.attach(cache, self.device)
        cache.collect_stats = config.moe_collect_stats
        # Per-expert routing histogram: the hot/cold placement table for a disk-backed
        # expert bank is exactly this, ordered. Unlike collect_stats it is a torch-level
        # scatter in ensure_experts rather than an in-kernel accumulate, so it only sees
        # real routing when decode runs eagerly (--disable-cuda-graph).
        cache.collect_decode_freq = bool(config.moe_stats_out)
        self._moe_stats_out = config.moe_stats_out
        self._moe_stats_layer_range = getattr(config, "pp_layer_range", None)
        self._moe_stats_rank = (config.tp_info.rank, config.tp_info.size)
        layers = attach_offload_moe_cache(self.model, cache)
        assert len(layers) == config.model_config.num_moe_layers + self._mtp_bank_layers
        if cache.decode_target in ("cpu", "hybrid"):
            self._init_cpu_moe_executor(config, cache, layers)
        self.ctx.moe_offload_cache = cache
        self.moe_offload_cache = cache
        return cache

    def _resolve_hybrid_fetch(self, config: EngineConfig, cache) -> None:
        """Resolve --moe-hybrid-max-fetch -1 (auto) into a bandwidth-matched fetch fraction.

        Perfect fetch/compute overlap wants fetched : cpu-computed misses = pcie_bw :
        (cpu_bw - pcie_bw), i.e. fetching a pcie_bw / cpu_bw fraction of each decode
        step's misses -- both sides then finish together instead of one idling. The
        achieved bandwidths come from the cached `ft bench bw` profile (the same one the
        auto backend pick reads); without a usable profile the old fixed cap of 1 applies.
        """
        if config.moe_hybrid_max_fetch >= 0:
            return  # explicit fixed cap
        from freetoken.moe.bench_profile import load_hybrid_fetch_fraction

        gpu_name, gpu_uuid = _profile_gpu(self.device.index)
        fraction = load_hybrid_fetch_fraction(
            cache.quant_format, gpu_name=gpu_name, gpu_uuid=gpu_uuid
        )
        if fraction is None:
            cache.hybrid_max_fetch = 1
            logger.warning_rank0(
                "--moe-hybrid-max-fetch auto: no usable `ft bench bw` profile for "
                f"{cache.quant_format!r} experts; using a fixed fetch cap of 1"
            )
            return
        cache.hybrid_max_fetch = cache.num_experts  # inert: the fraction is the cap
        cache.hybrid_fetch_fraction = fraction
        logger.info_rank0(
            f"--moe-hybrid-max-fetch auto: fetching {fraction:.1%} of each decode step's "
            "expert misses over PCIe (benched PCIe/CPU bandwidth ratio), the rest on the CPU"
        )

    def _init_cpu_moe_executor(self, config: EngineConfig, cache, layers) -> None:
        """Build the persistent CPU MoE executor (decode-time expert compute).

        Must run before CUDA graph capture: the worker pool has to be live for the
        eager warmup forward, and the pinned IO buffers / host-func task pointers
        must be stable for the captured nodes. Buffers/tasks themselves are
        allocated lazily on the first (eager) forward at each batch size.
        """
        from freetoken.moe.cpu_executor import CpuMoeExecutor

        sample = layers[0]
        required = ("top_k", "activation", "apply_router_weight_on_input")
        if not all(hasattr(sample, attr) for attr in required):
            raise NotImplementedError(
                "CPU MoE backend is not yet supported for this model architecture "
                f"(MoE layer {type(sample).__name__} is missing {required})."
            )
        # Decode batches never exceed max_running_req, but CUDA-graph padding can
        # round a batch up to the largest captured size; cover both.
        # An MTP verify window ships spec_k + 1 rows through the CPU executor at once, and a
        # short prefill extend goes through it in CPU_PREFILL_PIECE-row pieces.
        from freetoken.layers.moe import CPU_PREFILL_PIECE, cpu_prefill_max_tokens

        max_tokens = max(config.max_running_req, config.cuda_graph_max_bs or 0, 1, self.spec_k + 1)
        if cpu_prefill_max_tokens() > 0:
            max_tokens = max(max_tokens, CPU_PREFILL_PIECE)
        executor = CpuMoeExecutor(
            cache,
            top_k=sample.top_k,
            activation=sample.activation,
            apply_router_weight_on_input=sample.apply_router_weight_on_input,
            num_threads=config.moe_cpu_threads,
            max_tokens=max_tokens,
            device=self.device,
            swiglu_alpha=float(sample.alpha),
            swiglu_limit=sample.limit,
            # FIXME: the None branch serves GGUF q4_0 banks, which have no quant method yet; drop it once GGUF joins the quant path
            fmt=sample.quant_method.cpu_format if sample.quant_method is not None else None,
        )
        cache.set_cpu_executor(executor)
        self.cpu_moe_executor = executor

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        if self.config.is_pp:
            # each pipeline rank budgets its own GPU: the halves hold different weights, so a
            # cross-rank min/max (and the TP imbalance check) would be meaningless here
            return free_memory, free_memory
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def _target_moe_and_expert_bytes(self, moe_cache_size: int | None) -> tuple[int, int]:
        from freetoken.engine.cache_budget import expert_bytes_per_slot

        target_moe = (
            moe_cache_size
            if moe_cache_size is not None
            else (self.moe_offload_cache.cache_size if self.moe_offload_cache else 0)
        )
        per_expert_bytes = (
            expert_bytes_per_slot(self.moe_offload_cache.bank_sources)
            if self.moe_offload_cache is not None else 0
        )
        return target_moe, per_expert_bytes

    def _resize_kv_pool(self, config, num_pages: int, num_swa_pages: int | None) -> None:
        # IN-PLACE, identity-preserving: the CacheManager's swa_pool reference, ctx.kv_cache and
        # the model's per-access pool property all keep pointing at THIS pool, which frees its old
        # buffers before allocating the new ones. mark_for_rebind re-binds per-bind scratch on the
        # next forward (graph re-capture); the prefix tree + page bookkeeping reset is the
        # scheduler's generic cache_manager.rebuild.
        if self.kv_cache.needs_rebind_on_rebuild:
            self.model.mark_for_rebind()
        self.kv_cache.rebuild_from_config(config, num_pages, num_swa_pages=num_swa_pages)
        self.num_pages = num_pages

    def _refresh_seq_state(self, config) -> None:
        num_tokens = self.num_pages * config.page_size
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        if aligned_max_seq_len != self.page_table.shape[1]:
            # max_seq_len changed (e.g. KV grew past the startup token budget); the page table
            # columns must track it or new requests would index out of bounds. The scheduler
            # re-points its managers to engine.page_table on a num_pages rebuild.
            self.ctx.page_table = self.page_table = torch.zeros(
                (config.max_running_req + 1, aligned_max_seq_len),
                dtype=torch.int32,
                device=self.device,
            )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
        self.kv_cache.attach_page_table(self.page_table)

    @torch.inference_mode()
    def rebuild_runtime_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only in-place resize of the MoE slot cache, KV page pool, GDN (mamba) state pool,
        and/or the window pool (num_swa_pages: an absolute pinned window), followed by CUDA-graph
        re-capture. Does NOT reload weights or host expert banks. The caller (scheduler) must
        guarantee no in-flight prefill/decode.
        """
        config = self.config
        if (moe_cache_size is None and num_pages is None and num_mamba_slots is None
                and num_swa_pages is None):
            return

        # 0a. Geometry prevalidation BEFORE any destructive free. An invalid target (moe
        #     slots on a model with no offload cache, moe below num_experts / above the
        #     marlin cap, non-positive pages, or too few GDN slots to run) must reject
        #     recoverably with the old cache intact -- NOT after teardown, which would
        #     leave the server unable to serve. These checks are model-agnostic.
        if moe_cache_size is not None:
            if self.moe_offload_cache is None:
                raise CacheRebuildRejected(
                    "moe_cache_size requested but this model has no MoE offload cache"
                )
            try:
                self.moe_offload_cache.validate_rebuild(moe_cache_size)
            except ValueError as e:
                raise CacheRebuildRejected(str(e)) from e
        if num_pages is not None and num_pages <= 0:
            raise CacheRebuildRejected(f"num_pages must be positive, got {num_pages}")
        if num_mamba_slots is not None:
            if self.linear_state_pool is None:
                raise CacheRebuildRejected(
                    "num_mamba_slots requested but this model has no GDN state pool"
                )
            # num_mamba_slots is the USABLE slot count (what the user sets and the status bar
            # shows); the pool also reserves a padding sink (slot 0), so the physical pool is
            # num_mamba_slots + 1. _linear_pool_min_slots is the physical floor -> usable - 1.
            min_usable = _linear_pool_min_slots(config) - 1
            if num_mamba_slots < min_usable:
                raise CacheRebuildRejected(
                    f"num_mamba_slots {num_mamba_slots} is below the minimum {min_usable} "
                    f"(non-evictable working set for max_running_req={config.max_running_req}) "
                    f"needed to run; admission would deadlock"
                )
        if num_swa_pages is not None:
            # An absolute window pin for the radix-SWA window pool (Gemma) or the DSV4 window tier;
            # meaningless for dense/MHA models and the naive SWA path (concurrency x window).
            if not _supports_swa_ratio(config):
                raise CacheRebuildRejected(
                    "num_swa_pages requested but this model has no window pool "
                    "(needs DSV4 or a sliding-window model with --cache-type radix)"
                )
            if num_swa_pages <= 0:
                raise CacheRebuildRejected(
                    f"num_swa_pages must be positive, got {num_swa_pages}"
                )

        # 0b. Pool-family budget fit-check BEFORE any destructive free: an unfit geometry
        #     must reject (recoverable) so the old caches stay intact and serving continues,
        #     rather than freeing and then OOMing into permanent failure. The engine supplies
        #     the memory account; the pool answers whether its target geometry fits.
        target_moe, per_expert_bytes = self._target_moe_and_expert_bytes(moe_cache_size)
        # Price the sibling GDN state pool at ITS target (physical slots = usable + padding
        # sink) and hand the bytes in -- the KV pool only budgets its own tiers.
        target_mamba = (
            num_mamba_slots + 1
            if num_mamba_slots is not None
            else (self.linear_state_pool.num_slots if self.linear_state_pool is not None else None)
        )
        self.kv_cache.validate_rebuild(
            config, num_pages=num_pages,
            num_swa_pages=num_swa_pages, target_moe=target_moe,
            per_expert_bytes=per_expert_bytes, baseline_free=self._baseline_free,
            weights_bytes=self._weights_bytes, current_num_pages=self.num_pages,
            extra_fixed_bytes=(
                state_pool_bytes(config, target_mamba) if target_mamba is not None else 0
            ),
            extra_note=(
                f", mamba={target_mamba - 1} slots" if target_mamba is not None else ""
            ),
        )

        torch.cuda.synchronize(self.device)
        # Preserve the CUDA-graph batch-size set resolved at startup. The auto heuristic keys
        # off free memory, which is far smaller now that the caches are resident (post-cache
        # free << startup pre-load free), so re-deriving it here would silently drop large
        # batch sizes after the first rebuild. Reusing the already-resolved list keeps the
        # captured coverage identical (the fit-check above guarantees the graph headroom fits).
        prior_graph_bs = self.graph_runner.graph_bs_list
        # Point of no return for the scheduler's rollback logic: from here the live graphs and
        # pools start being freed. A failure BEFORE this flag flips leaves the engine serving
        # untouched (no rollback needed); after it, only a rebuild restores service.
        self.rebuild_teardown_started = True
        # 1. Tear down CUDA graphs + backend capture scratch (free-before-alloc).
        self.attn_backend.reset_capture()
        self.graph_runner.destroy_cuda_graphs()
        # 2. Resize caches in place (each frees its old GPU tensors before allocating).
        # Pin the new window first (validated above) so any KV-pool rebuild below sizes the window
        # to it (_dsv4_pool_sizes / _swa_paged_num_tokens read config.swa_num_pages_override).
        # frozen EngineConfig — mutate in place like the moe_cache_size path; `config.x = y` raises
        # FrozenInstanceError, which here aborts the rebuild after the CUDA graphs are gone (→ 503).
        if num_swa_pages is not None:
            object.__setattr__(config, "swa_num_pages_override", num_swa_pages)
        if moe_cache_size is not None:
            assert self.moe_offload_cache is not None, "no MoE offload cache to resize"
            self.moe_offload_cache.rebuild(moe_cache_size)
        if num_pages is not None:
            # sets self.num_pages (rebuilds KV + window)
            self._resize_kv_pool(config, num_pages, num_swa_pages)
        elif num_swa_pages is not None:
            # Window-only change: no page-count change, but re-derive the window pool at the new
            # pin against the CURRENT page count. This re-allocs the same-size full pool and
            # the resized window, both inside the pool's own rebuild_from_config.
            self._resize_kv_pool(config, self.num_pages, num_swa_pages)
        if num_mamba_slots is not None:
            # Reallocate the GDN state pool (frees old tensors first). Must sit between graph
            # teardown and re-capture so the recaptured graphs bind the new state tensors.
            # +1 for the reserved padding sink: num_mamba_slots is the usable count.
            self.linear_state_pool.rebuild(num_mamba_slots + 1)
        # 3. Refresh max_seq_len (+ generic page table) for the new token budget.
        self._refresh_seq_state(config)
        aligned_max_seq_len = _page_table_width(self.max_seq_len, config.page_size)
        # 4. Re-capture CUDA graphs against the new tensors (reset_capture above re-armed
        #    the backend; _sync_get_memory empties the cache so freed memory is reclaimed).
        gc.collect()
        free_min = self._sync_get_memory()[0]
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=prior_graph_bs,  # reuse the startup-resolved set (see above)
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=free_min,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            moe_offload_cache=self.moe_offload_cache,
        )
        if self.spec_k > 0:
            # the verify-window graph addressed the old pools / page table: capture it again
            self._capture_spec_graph()

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        use_graph = self.graph_runner.can_use_cuda_graph(batch)
        pp = self.pp_comm
        spec = self.spec_k > 0
        mtp = getattr(self.model, "mtp", None) if spec else None
        if spec:
            batch.spec_all_rows = batch.spec_verify  # the verify window needs every row's logits
            self.ctx.spec_stash = []
            if use_graph and batch.size == 1 and mtp is not None:
                # the draft head reads the target's final hidden state, which a graph replay
                # leaves in a buffer the model no longer points at: the (rare) plain
                # single-request decode step runs eagerly on the drafting rank instead
                use_graph = False
        # rows of the residual stream crossing the pipeline: every token of a prefill chunk,
        # one per real request in decode (padding rows never leave the GPU)
        rows = batch.input_ids.numel() if batch.is_prefill else batch.size
        # the captured K+1-row verify window (engine/spec_graph); shorter windows stay eager
        sg = self._spec_graph
        if sg is not None and not (batch.spec_verify and spec_graph_applicable(batch, rows, sg.rows)):
            sg = None
        # diagnostics: cross-check a one-row verify window against the plain decode path
        check = (
            _SPEC_CHECK_STEP_LEFT[0] > 0 and batch.spec_verify and rows == 1
            and not use_graph and sg is None
        )
        snap = self._spec_snapshot(batch.reqs[0]) if check else None
        if check:
            self.ctx.debug_layer_outs = []
        prof = None
        if (_SPEC_PROFILE and batch.spec_verify) or (_STEP_PROFILE and (batch.spec_verify or not batch.is_prefill)):
            prof = self._spec_profiler
            if prof is None:
                prof = self._spec_profiler = _SpecProfiler()
            prof.start()
        if sg is not None:
            sg.stage(batch)
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, use_graph or sg is not None):
            if pp is not None and not pp.is_first:
                hidden_in = pp.recv_hidden(rows)
                if prof is not None:
                    prof.mark("recv_hidden")
                if use_graph:
                    buf = self.graph_runner.buffer.pp_in
                    assert buf is not None
                    buf[:rows].copy_(hidden_in)
                    hidden_in = buf[: batch.padded_size]
                elif sg is not None:
                    sg.pp_in.copy_(hidden_in)
                    hidden_in = sg.pp_in
                self.ctx.pp_hidden_in = hidden_in
            try:
                if sg is not None:
                    logits = sg.replay()
                else:
                    logits = self.graph_runner.replay(batch) if use_graph else self.model.forward()
            finally:
                self.ctx.pp_hidden_in = None
        if sg is not None:
            # the rollback reads the GDN stashes recorded at capture (rewritten by the replay)
            self.ctx.spec_stash = sg.stash
        if prof is not None:
            prof.mark("target_forward")
        if self.cpu_moe_executor is not None:
            # One pinned read: surfaces a fired flag-handshake watchdog (dead coordinator
            # -> stale expert outputs) as a loud error instead of silent corruption.
            self.cpu_moe_executor.raise_if_unhealthy()
        if check:
            v_outs, self.ctx.debug_layer_outs = self.ctx.debug_layer_outs, None
            v_final = logits[:rows].detach().clone()

        if not batch.spec_verify:
            for req in batch.reqs:
                req.complete_one()

        copy_done_event = torch.cuda.Event()
        if pp is not None and not pp.is_last:
            # not the head: hand the residual stream on, then take the tokens the last rank
            # sampled (the scheduler on every rank feeds them into the next step)
            pp.send_hidden(logits[:rows])
            if prof is not None:
                prof.mark("send_hidden")
            spec_res = None
            if batch.pp_no_tokens:
                # a non-final prefill chunk: its sampled token has no reader (the successor is
                # the next prompt token), so do not wait for the last rank -- it is still on
                # the previous chunk, and this rank moves on to the next one meanwhile. The
                # scheduler ignores the tokens of a chunk-only batch.
                next_tokens_cpu = torch.zeros(batch.size, dtype=torch.int32)
            elif spec:
                spec_res = unpack_spec_message(pp.recv_tokens(spec_message_len(self.spec_k)), self.spec_k)
                if prof is not None:
                    prof.mark("wait_tokens")
                if batch.spec_verify:
                    self.model.spec_rollback(batch, len(spec_res.accepted), self.ctx)
                    if prof is not None:
                        prof.mark("rollback")
                next_tokens_cpu = torch.tensor(spec_res.accepted[:1], dtype=torch.int32)
            else:
                next_tokens_cpu = pp.recv_tokens(batch.size)
                if prof is not None:
                    prof.mark("wait_tokens")
            next_tokens_gpu = next_tokens_cpu.to(self.device)
            if prof is not None:
                prof.step()
            if check:
                _SPEC_CHECK_STEP_LEFT[0] -= 1
                self._spec_check_step(batch, snap, v_outs, v_final)
            copy_done_event.record(self.stream)
            return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event, spec_res)

        spec_res = None
        if batch.spec_verify:
            # the target's own sample at every window row decides how many drafts survive
            req = batch.reqs[0]
            sampled = self.sampler.sample(logits[:rows], _expand_sampling_args(args, rows)).to(torch.int32)
            accepted = accept_drafts(sampled.tolist(), req.spec_drafts)
            if prof is not None:
                prof.mark("sample_accept")
            self.model.spec_rollback(batch, len(accepted), self.ctx)
            if prof is not None:
                prof.mark("rollback")
            drafts = self._mtp_draft(
                batch, rows, row=len(accepted) - 1, next_token=accepted[-1], prof=prof, sg=sg,
            )
            spec_res = SpecResult(accepted, drafts)
            if _SPEC_TRACE_LEFT[0] > 0:
                _SPEC_TRACE_LEFT[0] -= 1
                top = logits[:rows].float().topk(3, dim=-1)
                logger.info(
                    f"spec trace: window={batch.input_ids[:rows].tolist()} drafts={req.spec_drafts} "
                    f"sampled={sampled.tolist()} accepted={accepted} next_drafts={drafts} "
                    f"top3={top.indices.tolist()} top3_logits={[[round(x, 2) for x in r] for r in top.values.tolist()]}"
                )
            next_tokens_cpu = torch.tensor(accepted[:1], dtype=torch.int32)
            next_tokens_gpu = next_tokens_cpu.to(self.device)
        else:
            batch_logits = logits[: batch.size]
            next_tokens_gpu = self.sampler.sample(batch_logits, args).to(torch.int32)
            if pp is not None or spec:
                next_tokens_cpu = next_tokens_gpu.cpu()  # synchronous: read on the host below
            else:
                next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            if prof is not None:
                prof.mark("sample")
            if spec:
                token = int(next_tokens_cpu[0])
                drafts = []
                if batch.size == 1 and mtp is not None:
                    tail = batch.spec_next_tail
                    # a non-final prefill chunk only extends the head's KV (no drafting)
                    drafts = self._mtp_draft(
                        batch, rows, row=rows - 1,
                        next_token=tail if tail is not None else token,
                        draft=tail is None,
                    )
                spec_res = SpecResult([token], drafts)
        if pp is not None and not batch.pp_no_tokens:
            # (the first rank did not wait for a chunk-only batch's tokens: nothing to send)
            pp.send_tokens(pack_spec_message(spec_res, self.spec_k) if spec else next_tokens_cpu)
        if prof is not None:
            if pp is not None and not batch.pp_no_tokens:
                prof.mark("send_tokens")
            prof.step()
        if check:
            _SPEC_CHECK_STEP_LEFT[0] -= 1
            self._spec_check_step(batch, snap, v_outs, v_final)
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event, spec_res)

    # ------------------------------------------------------------------ spec diagnostics
    def _spec_slot(self, req: Req) -> int:
        return req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx

    def _spec_snapshot(self, req: Req) -> dict:
        """Copy the per-request GDN slot state (recurrent + conv, plus any slot states) so the
        plain decode path can be replayed from the same point."""
        pool = self.linear_state_pool
        slot = self._spec_slot(req)
        snap: dict = {"slot": slot}
        if pool is not None:
            snap["rec"] = pool.recurrent_states[:, slot].clone()
            snap["conv"] = pool.conv_states[:, slot].clone()
            for name, t in pool.slot_states.items():
                snap["ss:" + name] = t[:, slot].clone()
        return snap

    def _spec_restore(self, snap: dict) -> None:
        pool = self.linear_state_pool
        if pool is None:
            return
        slot = snap["slot"]
        pool.recurrent_states[:, slot] = snap["rec"]
        pool.conv_states[:, slot] = snap["conv"]
        for name, t in pool.slot_states.items():
            t[:, slot] = snap["ss:" + name]

    def _spec_state_digest(self, req: Req, pos: int) -> dict:
        """Every piece of per-request state a one-token step at ``pos`` writes: GDN recurrent +
        conv per GDN layer, the PLE context, and per sparse layer the K/V row at ``pos``, the
        pending-ring row and the compressed-index rows (scratch, and the group row when the
        token closes a group). Clones, keyed by name."""
        d: dict = {}
        pool = self.linear_state_pool
        slot = req.linear_slot_idx if req.linear_slot_idx is not None else req.table_idx
        if pool is not None:
            for li in range(pool.recurrent_states.shape[0]):
                d[f"rec{li}"] = pool.recurrent_states[li, slot].clone()
                d[f"conv{li}"] = pool.conv_states[li, slot].clone()
            for name, t in pool.slot_states.items():
                d[f"ple:{name}"] = t[:, slot].clone()
        kv = self.kv_cache
        idx_slot = getattr(self.attn_backend, "_idx_slot", None)
        if idx_slot:
            loc = int(self.page_table[req.table_idx, pos].item())
            mtp = getattr(self.model, "mtp", None)
            mtp_id = getattr(mtp, "layer_id", None) if mtp is not None else None
            for lid, s in idx_slot.items():
                if lid == mtp_id:
                    continue
                try:
                    d[f"k{lid}"] = kv.k_cache(lid)[loc].clone()
                    d[f"v{lid}"] = kv.v_cache(lid)[loc].clone()
                    # Under --kv-cache-dtype the slabs above are codes; equal codes with
                    # unequal scales are unequal values, so the scales belong in the digest.
                    if kv.kv_quant is not None:
                        d[f"ks{lid}"] = kv.k_scales(lid)[loc].clone()
                        d[f"vs{lid}"] = kv.v_scales(lid)[loc].clone()
                except Exception:  # noqa: BLE001
                    pass
                cap = kv.ring_capacity
                d[f"ring{lid}"] = kv.pending_ring(s)[req.table_idx, pos % cap].clone()
                cmp = kv.cmp_k_cache(s)
                d[f"cmpS{lid}"] = cmp[kv.cmp_scratch_base + req.table_idx].clone()
                if loc % kv.index_ratio == kv.index_ratio - 1:
                    d[f"cmpG{lid}"] = cmp[loc // kv.index_ratio].clone()
        return d

    def _spec_check_step(self, batch: Batch, snap: dict, v_outs: list, v_final: torch.Tensor) -> None:
        """FT_SPEC_CHECK_STEP: after a one-row verify window, rewind the slot state and run the
        same token through the plain decode path (eager, phase 'decode'); log the per-layer
        divergence of the residual streams and of the final output. Both pipeline ranks run
        it in lockstep (the residual crosses the ranks like any forward)."""
        from types import SimpleNamespace

        from freetoken.attention.linear import build_fla_metadata

        req = batch.reqs[0]
        pos = int(batch.positions[0].item())
        token = int(batch.input_ids[0].item())
        state_v = self._spec_state_digest(req, pos)  # what the verify path left behind
        self._spec_restore(snap)
        proxy = SimpleNamespace(
            table_idx=req.table_idx, extend_len=1, device_len=pos + 1, cached_len=pos,
            linear_slot_idx=req.linear_slot_idx, uid=req.uid, mm_embeds=None,
            mamba_restore_src=None, mamba_ping_pong=None, decode_batch_idx=0,
            input_ids=req.input_ids, mm_rope=getattr(req, "mm_rope", None),
        )
        mini = Batch(reqs=[proxy], phase="decode")
        mini.padded_reqs = [proxy]
        mini.positions = torch.tensor([pos], dtype=torch.int32, device=self.device)
        rope = getattr(req, "mm_rope", None)
        if rope is not None and int(rope.delta):
            mini.rope_positions = torch.tensor([pos + int(rope.delta)], dtype=torch.int32, device=self.device)
        mini.input_ids = torch.tensor([token], dtype=torch.int32, device=self.device)
        mini.out_loc = self.page_table[req.table_idx, pos : pos + 1]
        mini.active_table_idx = torch.tensor([req.table_idx], dtype=torch.int32, device=self.device)
        if self.linear_state_pool is not None:
            mini.linear_table_idx = torch.tensor([snap["slot"]], dtype=torch.int32, device=self.device)
            mini.fla_metadata = build_fla_metadata(mini, self.device)
        self.attn_backend.prepare_metadata(mini)
        self.ctx.debug_layer_outs = []
        pp = self.pp_comm
        with self.ctx.forward_batch(mini), self.model.forward_host_ctx(mini, False):
            if pp is not None and not pp.is_first:
                self.ctx.pp_hidden_in = pp.recv_hidden(1)
            try:
                out = self.model.forward()
            finally:
                self.ctx.pp_hidden_in = None
        if pp is not None and not pp.is_last:
            pp.send_hidden(out[:1])
        d_outs, self.ctx.debug_layer_outs = self.ctx.debug_layer_outs, None
        state_d = self._spec_state_digest(req, pos)  # what the decode path leaves behind
        groups: dict[str, list[str]] = {}
        for name, dv in state_v.items():
            dd = state_d.get(name)
            if dd is None:
                continue
            a, b = dv.float().reshape(-1), dd.float().reshape(-1)
            grp = name.rstrip("0123456789")
            groups.setdefault(grp, []).append(
                f"{name[len(grp):]}={(a - b).abs().max().item():.0e}/{b.abs().max().item():.0e}"
            )
        logger.warning(
            f"spec state diff pos={pos} (verify+rollback vs decode, max|diff|/max|decode| per GDN layer): "
            + " | ".join(f"{g}: " + " ".join(v) for g, v in groups.items())
        )
        parts = []
        for (i, a), (_j, b) in zip(v_outs, d_outs):
            a, b = a.float().reshape(-1), b.float().reshape(-1)
            parts.append(f"L{i}:{(a - b).abs().max().item():.1e}/{b.abs().max().item():.1e}")
        a, b = v_final.float().reshape(-1), out[:1].float().reshape(-1)
        msg = f"final:{(a - b).abs().max().item():.1e}/{b.abs().max().item():.1e}"
        if pp is None or pp.is_last:
            msg += (
                f" top3 verify={a.topk(3).indices.tolist()} decode={b.topk(3).indices.tolist()}"
                f" argmax_equal={bool(a.argmax() == b.argmax())}"
            )
        logger.warning(f"spec step check pos={pos} tok={token} {msg} | " + " ".join(parts))

    # ------------------------------------------------------------------ MTP draft head
    def _mtp_draft(
        self, batch: Batch, rows: int, *, row: int, next_token: int, draft: bool = True, prof=None,
        sg=None,
    ) -> list:
        """Run the draft head over this forward's rows (fills its KV for them) and, when
        ``draft``, chain ``spec_k`` greedy draft tokens from ``row`` (the last accepted row) on.
        ``sg`` is the replayed verify-window graph, when the target ran as one: its final hidden
        state lives in the graph's static buffer, and its draft-head graphs take over the whole
        drafting when captured."""
        mtp = self.model.mtp
        if mtp is None:
            return []
        req = batch.reqs[0]
        rope = getattr(req, "mm_rope", None)
        delta = int(rope.delta) if rope is not None else 0
        if sg is not None and draft and sg.mtp_ready and rows == sg.rows:
            # successor ids of the window rows: the drafts, then the target's sample; the
            # drafting row's successor is the sample (see the eager path below)
            next_ids = list(req.spec_drafts)[: rows - 1] + [next_token]
            next_ids[row] = next_token
            return sg.mtp_draft(
                req, row=row, next_ids=next_ids, pos_row=req.cached_len + row, rope_delta=delta, prof=prof,
            )
        # a replayed window leaves its hidden state in the graph's static buffer (the model
        # attribute still points at the last eager forward's tensor)
        hidden = (sg.hidden if sg is not None else self.model.last_hidden)[:rows]
        ids = batch.input_ids[:rows]
        tail = torch.tensor([next_token], dtype=ids.dtype, device=ids.device)
        next_ids = torch.cat([ids[1:], tail]) if rows > 1 else tail
        if row < rows - 1:
            # partial accept: the drafting row's successor is the target's own sample, not the
            # rejected draft that followed it in the window (rows past it are dead weight)
            next_ids = next_ids.clone()
            next_ids[row] = next_token
        with self.ctx.forward_batch(batch):
            h_all = mtp.forward(hidden, next_ids, batch)  # [rows, width]
        if not draft:
            return []
        h = h_all[row : row + 1]
        d = int(self.model.lm_head.logits(mtp.to_head(h)).argmax(dim=-1).item())
        drafts = [d]
        if prof is not None:
            prof.mark("mtp_window")
        pos_row = int(batch.positions[row].item())
        for j in range(1, self.spec_k):
            p = pos_row + j
            if p + 1 > (req.spec_alloc_len or 0):
                break  # no reserved KV page for this draft position
            mini = self._mtp_step_batch(req, p, d, rope_delta=delta)
            with self.ctx.forward_batch(mini):
                h = mtp.forward(h, mini.input_ids, mini)
            d = int(self.model.lm_head.logits(mtp.to_head(h)).argmax(dim=-1).item())
            drafts.append(d)
        if prof is not None:
            prof.mark("mtp_chain")
        return drafts

    def _mtp_step_batch(self, req: Req, position: int, token: int, *, rope_delta: int = 0) -> Batch:
        """A one-token decode batch at ``position`` for a draft step: the same page-table row
        as the request (its KV pages past device_len are reserved by the scheduler)."""
        from types import SimpleNamespace

        proxy = SimpleNamespace(
            table_idx=req.table_idx, extend_len=1, device_len=position + 1, cached_len=position,
            linear_slot_idx=req.linear_slot_idx, uid=req.uid, mm_embeds=None, mamba_restore_src=None,
        )
        mini = Batch(reqs=[proxy], phase="decode")
        mini.padded_reqs = [proxy]
        mini.positions = torch.tensor([position], dtype=torch.int32, device=self.device)
        if rope_delta:  # after an image prompt the rope position runs ahead of the logical one
            mini.rope_positions = torch.tensor([position + rope_delta], dtype=torch.int32, device=self.device)
        mini.input_ids = torch.tensor([token], dtype=torch.int32, device=self.device)
        mini.out_loc = self.page_table[req.table_idx, position : position + 1]
        self.attn_backend.prepare_metadata(mini)
        return mini

    # Default share of the free VRAM one prefill chunk's transient may occupy
    # (--prefill-chunk-budget). The rest absorbs whatever else wants the card: on the machine
    # this was written for, a desktop, free VRAM moved by ~150 MB between runs depending on
    # what was on screen. A box that serves and nothing else can raise it.
    _PREFILL_TRANSIENT_BUDGET = 0.55
    _PREFILL_PROBE_TOKENS = 1024
    _PREFILL_CHUNK_FLOOR = 512

    @torch.inference_mode()
    def _autosize_prefill_chunk(self, config) -> None:
        """Lower ``--max-prefill-length`` until one chunk's transient fits the free VRAM.

        The chunk is the unit of transient allocation: the linear-attention kernels allocate
        buffers proportional to it on every chunk and free them again, so the peak the engine
        has to find room for scales with the chunk and not with the prompt. When that peak is
        the size of the free VRAM the run gets slow first and dies later -- measured on an
        RTX 2060, a 30k prompt took 1081 s at the default 8192 and 232 s at 4096, and at 8192
        it sometimes killed the server outright in the GDN prefill.

        Measure, do not search. Probing the *configured* size downwards would mean allocating
        the very transient that kills the process, at startup, on every boot. Instead this
        runs one small chunk, takes bytes-per-token from it (the buffers are linear in the
        chunk), and solves for the largest chunk that fits the budget.

        Only ever lowers: an explicit smaller ``--max-prefill-length`` is honored as-is, and
        ``--prefill-chunk-budget 0`` turns the whole thing off.
        """
        self._prefill_bytes_per_token = 0.0
        budget_share = float(
            getattr(config, "prefill_chunk_budget", self._PREFILL_TRANSIENT_BUDGET)
        )
        if not 0 < budget_share <= 1:
            return
        self._prefill_budget_share = budget_share
        configured = int(getattr(config, "max_extend_tokens", 0) or 0)
        if configured <= self._PREFILL_CHUNK_FLOOR:
            return
        probe = min(self._PREFILL_PROBE_TOKENS, configured, self.max_seq_len)
        if probe < 64:
            return

        try:
            per_token, free_before = self._measure_prefill_transient(probe)
        except Exception as exc:  # noqa: BLE001 -- a probe that fails must not stop the boot
            logger.info_rank0(
                f"--prefill-chunk-budget: probe failed ({type(exc).__name__}: {exc}); "
                f"keeping --max-prefill-length {configured}"
            )
            return
        if per_token <= 0:
            return
        # Kept for prefill_chunk_now(): the same measurement, re-applied against whatever is
        # free when a prompt actually arrives.
        self._prefill_bytes_per_token = per_token

        budget = free_before * budget_share
        fits = int(budget // per_token)
        # round down to a multiple of 256 so the number in the log is a size people recognize
        fits = max(self._PREFILL_CHUNK_FLOOR, (fits // 256) * 256)
        chosen = min(configured, fits)
        if chosen >= configured:
            logger.info_rank0(
                f"--prefill-chunk-budget: {per_token / 1024:.1f} KiB/token of prefill transient, "
                f"{free_before / 2**30:.2f} GiB free at {budget_share:.0%} -> {configured} fits, keeping it"
            )
            return

        object.__setattr__(config, "max_extend_tokens", chosen)
        logger.info_rank0(
            f"--prefill-chunk-budget: {per_token / 1024:.1f} KiB/token of prefill transient and "
            f"{free_before / 2**30:.2f} GiB free at {budget_share:.0%} -> --max-prefill-length {configured} would need "
            f"{configured * per_token / 2**30:.2f} GiB; using {chosen} instead"
        )

    def _measure_prefill_transient(self, length: int) -> tuple[float, int]:
        """Run one prefill of ``length`` tokens; return (bytes of transient per token, free VRAM).

        Uses the dummy request row the same way ``_warmup_prefill`` does, and restores it. The
        peak is torch's allocator high-water mark over the forward, minus what was already
        held, so it counts the buffers the chunk brings into being and nothing else.
        """
        dummy_row = self.page_table[self.dummy_req.table_idx]
        dummy_slot = int(dummy_row[0].item())
        torch.cuda.synchronize(self.device)
        free_before = int(torch.cuda.mem_get_info(self.device)[0])
        held = int(torch.cuda.memory_allocated(self.device))
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            dummy_row[:length] = torch.arange(length, dtype=torch.int32, device=self.device)
            warm_req = Req(
                input_ids=torch.zeros(length, dtype=torch.int32, device="cpu"),
                table_idx=self.dummy_req.table_idx,
                cached_len=0,
                output_len=1,
                uid=-1,
                sampling_params=None,  # type: ignore[arg-type]
                cache_handle=None,  # type: ignore[arg-type]
            )
            batch = Batch(reqs=[warm_req], phase="prefill")
            batch.padded_reqs = batch.reqs
            batch.input_ids = torch.zeros(length, dtype=torch.int32, device=self.device)
            batch.positions = torch.arange(length, dtype=torch.int32, device=self.device)
            batch.out_loc = dummy_row[:length]
            self.attn_backend.prepare_metadata(batch)
            with self.ctx.forward_batch(batch):
                self.model.forward()
            torch.cuda.synchronize(self.device)
            peak = int(torch.cuda.max_memory_allocated(self.device))
        finally:
            dummy_row.fill_(dummy_slot)
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.reset()
        transient = max(0, peak - held)
        return transient / length, free_before

    def prefill_chunk_now(self, ceiling: int) -> int:
        """The chunk to use for a prompt starting *now*, given what is free *now*.

        The startup sizer settles a chunk against the free VRAM at boot. On a machine that is
        also somebody's desktop that number goes stale within minutes -- free VRAM here moved
        by ~150 MB between runs depending on what was on screen -- and the direction that
        hurts is the one where a chunk sized in a quiet moment is issued into a busy one.

        Re-solving costs a driver query and an integer divide, and changes nothing that is
        allocated: the chunk is a scheduling bound, not a buffer. The caller pays it once per
        prefill batch, and a prefill batch runs for seconds.

        Counts the allocator's cached-but-unused blocks as available, because they are: the
        transient this is sizing will be served out of exactly those.
        """
        per_token = getattr(self, "_prefill_bytes_per_token", 0.0)
        if per_token <= 0 or ceiling <= self._PREFILL_CHUNK_FLOOR:
            return ceiling
        free = int(torch.cuda.mem_get_info(self.device)[0])
        reserved = int(torch.cuda.memory_reserved(self.device))
        allocated = int(torch.cuda.memory_allocated(self.device))
        usable = free + max(0, reserved - allocated)
        share = getattr(self, "_prefill_budget_share", self._PREFILL_TRANSIENT_BUDGET)
        fits = int((usable * share) // per_token)
        fits = max(self._PREFILL_CHUNK_FLOOR, (fits // 256) * 256)
        chosen = min(ceiling, fits)
        # Log only on a real move. This runs before every prefill batch; a line per batch
        # would bury the log, and a line per change is what someone debugging a slow prompt
        # actually wants to see.
        last = getattr(self, "_prefill_chunk_logged", None)
        if last is None or abs(chosen - last) >= 256:
            self._prefill_chunk_logged = chosen
            if last is not None:
                logger.info_rank0(
                    f"prefill chunk {last} -> {chosen} ({usable / 2**30:.2f} GiB usable)"
                )
        return chosen

    @torch.inference_mode()
    def _warmup_prefill(self) -> None:
        """Compile the Triton prefill path before the first real request.

        Decode CUDA graph capture warms the decode path, but the first prefill
        can still pay Triton/cublas setup costs. Use the dummy request row and
        restore it afterwards so padded decode graph replay keeps using the
        dedicated dummy KV slot.
        """
        if self.max_seq_len < 2:
            return

        warmup_lens = [min(80, self.max_seq_len)]
        if self.max_seq_len >= 128:
            warmup_lens.append(128)
        warmup_lens = sorted({length for length in warmup_lens if length >= 2})
        if not warmup_lens:
            return

        dummy_row = self.page_table[self.dummy_req.table_idx]
        dummy_slot = int(dummy_row[0].item())
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record(self.stream)
        try:
            for length in warmup_lens:
                dummy_row[:length] = torch.arange(
                    length, dtype=torch.int32, device=self.device
                )
                warm_req = Req(
                    input_ids=torch.zeros(length, dtype=torch.int32, device="cpu"),
                    table_idx=self.dummy_req.table_idx,
                    cached_len=0,
                    output_len=1,
                    uid=-1,
                    sampling_params=None,  # type: ignore[arg-type]
                    cache_handle=None,  # type: ignore[arg-type]
                )
                batch = Batch(reqs=[warm_req], phase="prefill")
                batch.padded_reqs = batch.reqs
                batch.input_ids = torch.zeros(length, dtype=torch.int32, device=self.device)
                batch.positions = torch.arange(length, dtype=torch.int32, device=self.device)
                batch.out_loc = dummy_row[:length]
                self.attn_backend.prepare_metadata(batch)
                with self.ctx.forward_batch(batch):
                    self.model.forward()
        finally:
            dummy_row.fill_(dummy_slot)
            if self.moe_offload_cache is not None:
                self.moe_offload_cache.reset()
        ended.record(self.stream)
        torch.cuda.synchronize(self.device)
        logger.info_rank0(
            f"Prefill warmup complete for lengths {warmup_lens} "
            f"in {started.elapsed_time(ended) / 1000.0:.3f} s"
        )

    def shutdown(self) -> None:
        self._write_moe_stats()
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()

    def _build_bank_tier(self, config: EngineConfig):
        """``--moe-bank-ram``: the mapped expert banks, or None when off or unnecessary.

        Built before the load, not after: the banks are written per layer as the checkpoint
        streams in, because holding the originals and a second copy at once needs more RAM
        than the host that wants this feature has (moe/mapped_bank.py).
        """
        if not config.moe_bank_ram:
            return None
        from freetoken.moe import bank_disk
        from freetoken.moe.mapped_bank import MappedTier

        # --moe-bank-ram is a whole-host cap, but each rank builds its own banks and every
        # rank of a layer split lives on the same machine -- so the per-rank share is what
        # the placement is solved against. Taking the flag per rank instead would silently
        # double the RAM a two-GPU run uses.
        total = bank_disk.parse_size(config.moe_bank_ram)
        ranks = max(1, config.tp_info.size)
        budget = total // ranks
        layers = list(range(config.model_config.num_moe_layers))
        placement, cell_bytes = bank_disk.plan_from_config(
            config.model_config, budget, layers, config.moe_bank_stats
        )
        bank_gib = (cell_bytes or 0) * len(layers) * config.model_config.num_experts / 2**30
        if placement is None:
            logger.info_rank0(
                f"--moe-bank-ram {config.moe_bank_ram}: {budget / 2**30:.1f} GiB per rank "
                f"({ranks} ranks) covers this rank's {bank_gib:.1f} GiB of banks; no split"
            )
            return None
        logger.info_rank0(
            f"--moe-bank-ram {config.moe_bank_ram}: {budget / 2**30:.1f} GiB per rank "
            f"({ranks} ranks) against {bank_gib:.1f} GiB of banks"
        )
        if not config.moe_bank_stats:
            logger.warning(
                "--moe-bank-ram without --moe-bank-stats: the split ignores routing, so the "
                "resident half is an arbitrary slice. Collect a histogram with "
                "--moe-stats-out --disable-cuda-graph first."
            )
        # Whether the overlap survives is decided after the banks are opened, on what
        # actually got registered -- see the caller.
        return MappedTier(
            placement,
            bank_disk.bank_file_path(
                config.model_path, config.tp_info.rank, config.tp_info.size, config.moe_bank_dir
            ),
            layers,
            log=logger.info_rank0,
        )

    def _write_moe_stats(self) -> None:
        """Dump the decode instrumentation to ``--moe-stats-out`` on an orderly stop.

        Ctrl+C / SIGTERM reach here via uvicorn's lifespan; a hard kill loses the window,
        which is acceptable for an opt-in instrumentation run.
        """
        from freetoken.engine.moe_stats import write_moe_stats

        rank, size = getattr(self, "_moe_stats_rank", (0, 1))
        write_moe_stats(
            getattr(self.ctx, "moe_offload_cache", None),
            getattr(self, "_moe_stats_out", None),
            rank,
            size,
            getattr(self, "_moe_stats_layer_range", None),
        )


def _profile_gpu(index: "int | None" = None) -> Tuple[str | None, str | None]:
    """(name, uuid) of visible device ``index`` (default: the current, i.e. bound, device); (None, None) without CUDA."""
    if not torch.cuda.is_available():
        return None, None
    ident = gpu_identity(torch.cuda.current_device() if index is None else index)
    return ident["name"], ident["uuid"]


def _ensure_expandable_segments() -> None:
    """Default the CUDA allocator to expandable segments.

    The motivating case is the offload prefill, which repeatedly dequantizes
    variable-sized NVFP4 expert blocks to BF16 (a different size per layer as the
    active-expert count varies). Under that alloc/free churn the default caching
    allocator fragments badly -- reserved memory can balloon far past the actual peak
    allocation (observed ~78GiB reserved for a <30GiB working set).
    ``expandable_segments`` lets freed regions of any size be reused, keeping
    reserved ~= allocated, so it is applied to every run, not just offload ones.

    Env vars are parsed once at import and ignored if set afterwards, so we apply the
    setting via the runtime API instead. Must run before the first CUDA allocation (the
    caller guarantees CUDA is not yet initialized). Any user-provided allocator config
    is respected and left untouched.
    """
    if os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        return
    try:
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except Exception as exc:  # pragma: no cover - depends on torch build
        logger.info_rank0(f"Could not enable expandable_segments ({exc}); continuing")
        return
    logger.info_rank0("Enabled expandable_segments (override via PYTORCH_ALLOC_CONF)")


def _resolve_cache_type(has_linear_attention: bool, requested: str) -> str:
    # Hybrid GDN models default to the HybridRadixCache (snapshots GDN state at chunk
    # boundaries -> cross-request prefix reuse). An explicit ``--cache-type naive`` opts out
    # to the old no-reuse path (debugging / parity baseline / lower GDN-state memory).
    if has_linear_attention:
        return "naive" if requested == "naive" else "hybrid_radix"
    return requested


def _adjust_dsv4_config(config: EngineConfig, override) -> None:
    """DSV4 engine-config reconciliation at config-resolution time (before the pool exists).
    Syncs the resolved runtime config into the opaque ``dsv4_args`` payload, sets
    page_size to the window page P, forces single-chunk prefill, and clamps cuda_graph_bs/max_bs to
    the DSV4 decode batch size.
    """
    model_config = config.model_config
    model_config.dsv4_args.max_seq_len = config.max_seq_len
    model_config.dsv4_args.max_batch_size = config.max_running_req + 1  # +1 dummy
    # config.swa_full_tokens_ratio is the DSV4 window/full ratio directly (default sizing);
    # a runtime rebuild pins an absolute window via swa_num_pages_override instead.
    # DSV4's KV page IS the P-token window page (window == radix reuse granularity == lcm of
    # the compress ratios), so max_num_tokens = num_pages * page_size holds like every model.
    P = model_config.dsv4_args.window_size
    override("page_size", P)
    logger.info_rank0(f"DSV4 KV pages are {P}-token window pages; page_size set to {P}")
    # The generic CacheManager materializes DSV4 'radix' as the shared SWARadixCache (is_swa);
    # 'naive' stays naive with the pool's swa currency riding swa_paged.
    if getattr(config, "cache_type", "radix") != "naive":
        override("cache_type", "swa_radix")
    # 'radix' (SWARadixCache on the full-loc currency, carry-aware re-prefill) is the default and is
    # honored, as is an explicit 'naive'. Don't let max_extend_tokens force a second chunk within
    # one prompt (the pool's prefill_chunk_budget still chunks prompts larger than the window
    # pool); prefill batches ragged (bs>=1), each segment resuming from its own cached_len.
    if getattr(config, "max_extend_tokens", 0) < config.max_seq_len:
        override("max_extend_tokens", config.max_seq_len)

    # DSV4 decode batches at most max_running_req rows; its full-loc snapshot is sized to that,
    # so a graph bs above it would exceed the backend's captured snapshot rows. Clamp any
    # oversized explicit list / max_bs here (before GraphRunner ever sees it).
    mr = config.max_running_req
    if config.cuda_graph_max_bs is not None and config.cuda_graph_max_bs > mr:
        logger.warning_rank0(
            f"cuda_graph_max_bs {config.cuda_graph_max_bs} exceeds DSV4 max_running_req {mr}; "
            "clamping to max_running_req (larger decode batches never occur)."
        )
        override("cuda_graph_max_bs", mr)
    if config.cuda_graph_bs is not None:
        kept = [bs for bs in config.cuda_graph_bs if bs <= mr]
        if kept != list(config.cuda_graph_bs):
            dropped = [bs for bs in config.cuda_graph_bs if bs > mr]
            logger.warning_rank0(
                f"dropping cuda_graph_bs entries {dropped} above DSV4 max_running_req {mr} "
                "(larger decode batches never occur)."
            )
            override("cuda_graph_bs", kept)


def _parse_cpu_layers_spec(spec: str, num_moe_layers: int) -> frozenset[int]:
    """Parse ``--moe-cpu-layers``: an explicit MoE-layer id list (``"3,7,11"``), a count
    (``"8"`` -> 8 layers evenly strided across depth), or a fraction (``"0.5"``). Ids are
    indices into the MoE layers, ``[0, num_moe_layers)``."""
    s = spec.strip()
    if not s:
        return frozenset()
    if "," in s:
        ids = {int(x) for x in s.split(",") if x.strip()}
        for i in ids:
            if not 0 <= i < num_moe_layers:
                raise ValueError(
                    f"--moe-cpu-layers id {i} out of range [0, {num_moe_layers})"
                )
        return frozenset(ids)
    if "." in s:
        frac = float(s)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"--moe-cpu-layers fraction {frac} must be in [0, 1]")
        k = round(frac * num_moe_layers)
    else:
        k = int(s)
        if not 0 <= k <= num_moe_layers:
            raise ValueError(f"--moe-cpu-layers count {k} must be in [0, {num_moe_layers}]")
    # k layers spread evenly across depth (frozenset dedups any rounding collisions;
    # k == 0 yields an empty range, hence an empty set).
    return frozenset(round(i * num_moe_layers / k) for i in range(k))


def _resolve_cpu_layers(config: EngineConfig, num_moe_layers: int, *, reserved: int = 0, method=None) -> frozenset[int]:
    """MoE layer ids whose decode runs on the CPU executor.

    ``--moe-strategy cpu`` -> every layer. ``--moe-strategy offload`` + ``--moe-cpu-layers``
    -> the parsed subset, or the pin-budget pick for ``auto`` (the rest stay on the GPU offload/PCIe path). Otherwise none.
    """
    if config.moe_strategy == "cpu":
        return frozenset(range(num_moe_layers))
    spec = config.moe_cpu_layers
    if not spec or not is_offload_moe_strategy(config.moe_strategy):
        return frozenset()
    if spec.strip() == "auto":
        return _auto_cpu_layers(config, num_moe_layers, reserved=reserved, method=method)
    return _parse_cpu_layers_spec(spec, num_moe_layers)


def _decode_target(config: EngineConfig) -> str:
    """Where routed experts decode, from the flags alone: hybrid co-compute, the CPU executor for some or all layers, or the GPU slot cache."""
    if config.moe_strategy == "hybrid":
        return "hybrid"
    if config.moe_strategy == "cpu" or (config.moe_cpu_layers and is_offload_moe_strategy(config.moe_strategy)):
        return "cpu"
    return "gpu"


# expert activations the CPU MoE executor supports (csrc ActKind)
_CPU_MOE_ACTS = (
    "silu", "swish", "gelu", "gelu_tanh", "gelu_pytorch_tanh", "swigluoai",
    "swiglu_clamp",
)


def _cpu_moe_executor_viable(model_config) -> bool:
    """Whether an automatic CPU-decode decision may target the CPU MoE executor.

    A default boot must degrade to GPU offload instead of crashing in CpuMoeExecutor after the whole load; explicit cpu/hybrid/--moe-cpu-layers picks still fail loudly."""
    from freetoken.moe.cpu_executor import _WFMT_IDS, compiled_extension_supports

    try:
        from freetoken.kernel import _cpu_moe  # noqa: F401
    except ImportError:
        return False
    act = getattr(model_config, "hidden_act", "silu")
    moe_wfmt = getattr(model_config, "moe_weight_format", None)
    if act not in _CPU_MOE_ACTS and moe_wfmt != "mxfp4":
        return False
    if moe_wfmt != "mxfp4" and not compiled_extension_supports(act):
        return False
    expert_quant = getattr(model_config, "expert_quant", "none")
    fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
    return fmt == "mxfp4" or fmt in _WFMT_IDS


def _pin_budget_bytes(reserved: int = 0) -> int | None:
    """Bytes this process can still safely cudaHostRegister, or None when the platform does not cap pinning (plain Linux).

    WSL's WDDM-backed CUDA caps pinning near half of RAM, shared across processes -- budget 40%. FREETOKEN_PIN_BUDGET_GB overrides anywhere. ``reserved`` subtracts host bytes already pinned outside the expert banks (qwen4_exp's PLE table)."""
    if env := os.environ.get("FREETOKEN_PIN_BUDGET_GB"):
        cap = int(float(env) * 2**30)
    elif not hasattr(os, "uname") or "microsoft" not in os.uname().release.lower():  # WSL kernel tag
        return None
    else:
        cap = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.4)
    return max(0, cap - reserved)


def _bank_bytes(config: EngineConfig, method=None) -> int | None:
    from freetoken.moe.expert_banks import bank_bytes_estimate, ftw_bank_bytes

    return ftw_bank_bytes(config.model_path) or bank_bytes_estimate(config.model_config, method)


def _pin_hint(reserved: int) -> str:
    if _pin_budget_bytes(reserved) is None:
        return "the expert banks need more page-locked host RAM than this host has; free host RAM or serve a smaller model"
    return (
        "pass --moe-cpu-layers auto to lock the layers over the pin budget for CPU decode, "
        "or --moe-cpu-layers <count|fraction|ids> to choose them yourself"
    )


def _check_pin_budget(config: EngineConfig, *, reserved: int, method=None) -> None:
    """Stop a plain offload boot whose banks exceed a known pin budget before any bank is read."""
    if config.moe_cpu_layers or config.moe_strategy not in ("offload", "hybrid"):
        return
    budget = _pin_budget_bytes(reserved)
    bank_bytes = _bank_bytes(config, method) if budget is not None else None
    if bank_bytes and bank_bytes > budget:
        raise ValueError(
            f"expert banks need {bank_bytes / 2**30:.1f} GiB of pinned host RAM but the pin budget is "
            f"{budget / 2**30:.1f} GiB (WSL caps CUDA pinning; FREETOKEN_PIN_BUDGET_GB overrides); {_pin_hint(reserved)}"
        )


def _auto_cpu_layers(config: EngineConfig, num_moe_layers: int, *, reserved: int = 0, method=None) -> frozenset[int]:
    """Pick CPU (locked) MoE layers for ``--moe-cpu-layers auto``: none while the banks fit the pin budget.

    Locks just enough head+tail layers: per-layer decode miss rates are U-shaped, so the ends are the cheapest to move off the slot cache."""
    bank_bytes = _bank_bytes(config, method)
    if not bank_bytes:
        return frozenset()
    budget = _pin_budget_bytes(reserved)
    if budget is None or bank_bytes <= budget:
        return frozenset()
    if not _cpu_moe_executor_viable(config.model_config):
        raise ValueError(
            f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB exceed the "
            f"pin budget {budget / 2**30:.2f} GiB, but the CPU MoE executor cannot "
            f"serve this model (see --moe-strategy cpu requirements)"
        )
    n = min(num_moe_layers, math.ceil(num_moe_layers * (1 - budget / bank_bytes)))
    head = (n + 1) // 2
    ids = frozenset(range(head)) | frozenset(range(num_moe_layers - (n - head), num_moe_layers))
    logger.info_rank0(
        f"--moe-cpu-layers auto: banks {bank_bytes / 2**30:.2f} GiB > pin budget "
        f"{budget / 2**30:.2f} GiB; locking {n} head+tail MoE layers for CPU decode "
        f"({sorted(ids)})"
    )
    return ids


# MoE-only knobs and the value each resolves to on a dense model. moe_strategy is handled
# separately (its dense value is 'fused', but 'auto' resolves there without a warning).
_DENSE_MOE_SETTINGS = {
    "moe_cache_size": 0,
    "moe_cache_rate": None,
    "moe_cache_auto": False,
    "moe_cpu_layers": None,
    "moe_cpu_threads": 0,
    "moe_hybrid_max_fetch": -1,
    "moe_prefill_overlap": True,
    "moe_prefill_hit_d2d": False,
    "expert_load": "auto",
}


def _adjust_ftw_quant_backend(model_path: str, quant_backend: QuantBackend) -> QuantBackend:
    """--quant-backend with an FTW checkpoint's packed expert kernel filled in where the flag leaves that table automatic.

    An entry that names another kernel is refused here, before any bank is read."""
    from freetoken.checkpoint.ftw import ftw_quant_format
    from freetoken.moe.legacy_format import kind_kernel_for

    fmt = ftw_quant_format(model_path) if model_path else None
    if fmt is None:
        return quant_backend
    try:
        kind, kernel = kind_kernel_for(fmt)
    except KeyError:
        return quant_backend
    requested = quant_backend.select(LayerKind.MOE, kind)
    if requested == kernel:
        return quant_backend
    if requested != "auto":
        raise ValueError(
            f"the FTW checkpoint's expert banks were packed for {kind} / {kernel} but --quant-backend asks for "
            f"{requested}; drop the entry, or reconvert with ft checkpoint --quant-backend moe.{kind}={requested}"
        )
    return QuantBackend(quant_backend.items + (((LayerKind.MOE, kind), kernel),))


def shared_offload_method(model):
    """The expert method every offload MoE layer of ``model`` uses, or None for models whose MoE layers carry none (GGUF).

    The offload cache holds one bank layout, so the layers must agree on (kind, kernel)."""
    layers = [l for l in iter_offload_moe_layers(model) if getattr(l, "quant_method", None) is not None]
    if not layers:
        return None
    keys = {(layer.quant_method.kind, layer.quant_method.kernel.name) for layer in layers}
    if len(keys) != 1:
        found = sorted(f"{kind} / {kernel}" for kind, kernel in keys)
        raise ValueError(f"expert layers disagree on format / kernel: {found}; per-layer mixed expert formats are not supported")
    kind, kernel = keys.pop()
    logger.info_rank0(f"MoE experts: {kind} via {kernel}")
    return layers[0].quant_method


def offload_expert_method(config: EngineConfig):
    """The offload expert method of ``config``'s model, from a meta-device build.

    For tools that load expert banks without an engine (the FTW converter); they pack for GPU decode."""
    from freetoken.models import create_model
    from freetoken.utils.torch_utils import torch_dtype

    set_quant_backend(_adjust_ftw_quant_backend(config.model_path, QuantBackend.parse(config.quant_backend)))
    object.__setattr__(config.model_config, "moe_strategy", config.moe_strategy)
    object.__setattr__(config.model_config, "decode_target", "gpu")
    with torch.device("meta"), torch_dtype(config.dtype):
        model = create_model(config.model_config)
    return shared_offload_method(model)


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    model_config = config.model_config
    single_stream_only = getattr(model_config, "single_stream_only", False)
    is_dsv4 = getattr(model_config, "dsv4_args", None) is not None
    has_swa_attention = getattr(model_config, "has_swa_attention", False)
    has_linear_attention = getattr(model_config, "has_linear_attention", False)
    is_moe = getattr(model_config, "is_moe", False)
    expert_quant = getattr(model_config, "expert_quant", "none")

    if not is_moe:
        # A dense model has no routed experts: the MoE knobs are inert, and the offload family
        # is worse than inert -- engine init would build an expert cache for a model that has
        # none and abort startup (weights already resident) on an unrelated expert-source
        # error. Drop them at this one choke point, which the CLI and the programmatic
        # LLM(...) path both pass through. 'auto'/'fused' is the silent dense resolution;
        # anything else was asked for explicitly, so report what is being ignored.
        dropped = [
            f"{name}={getattr(config, name)!r}"
            for name, dense_value in _DENSE_MOE_SETTINGS.items()
            if getattr(config, name, dense_value) != dense_value
        ]
        if config.moe_strategy not in ("auto", "fused"):
            dropped.insert(0, f"moe_strategy={config.moe_strategy!r}")
        override("moe_strategy", "fused")
        for name, dense_value in _DENSE_MOE_SETTINGS.items():
            override(name, dense_value)
        if dropped:
            logger.warning_rank0(
                f"{getattr(model_config, 'model_type', 'model')} is a dense model (no routed "
                f"experts); ignoring MoE settings: {', '.join(dropped)}"
            )

    if single_stream_only:
        # The model runs one sequence at a time: it collapses the batch to one row and the
        # decode CUDA graph is captured at bs=1. Force the runtime knobs so the KV pool, page
        # table and graph capture all stay bs=1.
        if config.max_running_req != 1:
            override("max_running_req", 1)
        if config.cuda_graph_max_bs is None or config.cuda_graph_max_bs >= 1:
            override("cuda_graph_bs", [1])
            override("cuda_graph_max_bs", 1)

    if config.cuda_graph_max_bs is None:
        override("cuda_graph_max_bs", config.max_running_req)

    if is_dsv4:
        _adjust_dsv4_config(config, override)

    if has_swa_attention:
        # Both SWA cache paths use the global-paged swa pool (page_size==1 only for now).
        if config.page_size != 1:
            raise ValueError(
                f"SWA models currently support only page_size=1, got {config.page_size}."
            )
        # naive keeps cache_type='naive' (NaivePrefixCache, no reuse) on the paged pool (==
        # sglang SWAChunkCache); radix materializes as swa_radix (SWARadixCache, cross-request
        # reuse == sglang SWARadixCache). Both allocate from the same swa pool + free out-of-window.
        if getattr(config, "cache_type", "radix") != "naive":
            if not 0.0 < config.swa_full_tokens_ratio <= 1.0:
                raise ValueError(
                    f"swa_full_tokens_ratio must be in (0, 1], got {config.swa_full_tokens_ratio}"
                )
            override("cache_type", "swa_radix")

    if has_linear_attention:
        override(
            "cache_type",
            _resolve_cache_type(True, getattr(config, "cache_type", "radix")),
        )

    # Type x backend capability matrix: resolve auto from the per-type priority
    # lists, then validate whatever is now selected (explicit or auto) -- every
    # comma part must serve every required type, with packages/arch available.
    required_attn_types = _required_attn_types(model_config)
    _dtype = getattr(config, "dtype", None)  # duck-typed test configs omit it
    if (
        required_attn_types & {AttnType.BSA, AttnType.QSA}
        and _dtype is not None
        and _dtype.itemsize != 2
    ):
        # Reject at config time: the BSA/QSA pool's own assert only fires after the
        # model is resident (and not at all under `python -O`).
        raise ValueError(
            f"--dtype {config.dtype}: block-sparse attention serves 16-bit "
            "compute only (the index slab budgets 2 bytes/token); use bfloat16 "
            "or float16."
        )
    if _dtype == torch.float16 and "mxfp8" in (
        getattr(model_config, "attn_quant", "none"),
        getattr(model_config, "dense_quant", "none"),
    ):
        # The MXFP8 GEMV folds the pow2-descaled fp8 weight into the activation
        # dtype; fp16's narrow exponent can overflow/flush what bf16 represents
        # exactly, and the combination was never numerically validated.
        raise ValueError(
            "--dtype float16 with MXFP8 resident weights is unsupported (the "
            "W8A16 fold is only validated exact in bfloat16); use bfloat16."
        )
    if config.attention_backend == "auto":
        override(
            "attention_backend",
            _resolve_auto_attention_backend(required_attn_types),
        )
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")
    _validate_attention_backend_choice(config, override, required_attn_types)

    if config.moe_cache_rate is not None:
        total_experts = config.model_config.num_moe_layers * config.model_config.num_experts
        override("moe_cache_size", math.ceil(total_experts * config.moe_cache_rate))

    # The CPU MoE executor supports the silu/gelu family plus the clamped
    # swigluoai (csrc ActKind; "gpt_oss_swiglu" rides inside the mxfp4 kernel and
    # swigluoai the generic GEMV epilogue). A model with any other expert
    # activation cannot decode on the CPU: reject an explicit cpu/hybrid pick at
    # config time, and keep auto from upgrading offload -> hybrid off the profile.
    # hidden_act (the dense activation) stands proxy for the expert activation --
    # true for every in-tree model. mxfp4 experts pass regardless: their act runs
    # inside the mxfp4 kernel, not the generic epilogue.
    _cpu_moe_act_ok = getattr(model_config, "hidden_act", "silu") in _CPU_MOE_ACTS or (
        getattr(model_config, "moe_weight_format", None) == "mxfp4"
    )
    if (
        is_moe
        and not _cpu_moe_act_ok
        and (config.moe_strategy in ("cpu", "hybrid") or config.moe_cpu_layers)
    ):
        asked = (
            f"--moe-cpu-layers={config.moe_cpu_layers!r}"
            if config.moe_strategy not in ("cpu", "hybrid")
            else f"--moe-strategy {config.moe_strategy!r}"
        )
        raise ValueError(
            f"{asked}: the CPU MoE executor does not support this model's expert "
            f"activation {getattr(model_config, 'hidden_act', None)!r}; drop the flag "
            "and let every layer decode on the GPU offload path instead."
        )

    if is_moe and config.moe_strategy == "auto":
        # A MoE model always defaults to the offload family: experts stream from pinned host
        # banks into an auto-sized GPU slot cache, which is the only default that serves a model
        # bigger than the GPU. The resident 'fused' path (bf16 / block-fp8 experts, the two
        # formats MoELayer can allocate) is still reachable, but only when asked for explicitly
        # -- auto never picks it, because nothing here knows whether the experts would fit in
        # HBM and a wrong guess is a weight-load OOM rather than a slower-but-working run.
        default_backend = "offload"
        # Hardware-adaptive config: a cached `ft bench bw` profile can upgrade
        # the offload default to hybrid when this machine's CPU MoE bandwidth clears its PCIe
        # gather bandwidth by the bench threshold (default 2x). hybrid is VRAM-equivalent to
        # offload -- same auto-sized GPU slot cache (_resolve_auto_moe_cache_size), plus a
        # host-RAM CPU executor -- so this never raises the OOM risk; with no profile (or one
        # from different hardware) it stays offload. offload remains the always-safe fallback.
        # Key the lookup on the real expert format: mxfp4/q4_0 live in moe_weight_format when
        # expert_quant is "none", and "none" with no weight format means plain bf16 experts.
        moe_wfmt = getattr(model_config, "moe_weight_format", None)
        bench_fmt = expert_quant if expert_quant != "none" else (moe_wfmt or "bf16")
        from freetoken.moe.bench_profile import load_backend_recommendation

        gpu_name, gpu_uuid = _profile_gpu()
        if load_backend_recommendation(bench_fmt, gpu_name=gpu_name, gpu_uuid=gpu_uuid) == "hybrid":
            from freetoken.moe.cpu_executor import compiled_extension_supports

            _act = getattr(model_config, "hidden_act", "silu")
            if not _cpu_moe_act_ok:
                logger.info_rank0(
                    f"benchbw profile recommends hybrid, but the CPU MoE executor does not "
                    f"support this model's expert activation "
                    f"{getattr(model_config, 'hidden_act', None)!r}; staying on offload"
                )
            elif moe_wfmt != "mxfp4" and not compiled_extension_supports(_act):
                # Stale prebuilt _cpu_moe.so: an explicit cpu/hybrid pick still
                # hard-fails in the executor, but a default must not turn into a
                # post-load crash -- degrade to offload.
                logger.info_rank0(
                    f"benchbw profile recommends hybrid, but the compiled _cpu_moe "
                    f"extension predates activation {_act!r} (rebuild with "
                    f"`python setup.py build_ext --inplace`); staying on offload"
                )
            else:
                default_backend = "hybrid"
                logger.info_rank0(
                    f"benchbw profile recommends hybrid for {bench_fmt!r} experts on this GPU"
                )
        override("moe_strategy", default_backend)
        logger.info_rank0(f"Auto-selected MoE strategy: {config.moe_strategy}")

        if (
            is_offload_moe_strategy(config.moe_strategy)
            and config.moe_cache_size <= 0
            and config.moe_cache_rate is None
            and not getattr(config, "moe_cache_auto", False)
        ):
            # args.py's "no sizing flag -> default --moe-cache-auto" only fires when the
            # backend is already offload-family at *parse* time. A bare `ft serve <FTW MoE
            # checkpoint>` (no --moe-strategy, no cache flags) still has moe_strategy=="auto" at
            # parse time -- the auto -> offload/cpu/hybrid resolution above is the first point
            # the concrete backend is known, so mirror the same default here: no sizing flag
            # was given, so let the scheduler resolve the cache size from free VRAM instead of
            # failing the _require_offload_cache_size guard with size=0.
            override("moe_cache_auto", True)
            logger.info_rank0(
                "No MoE cache sizing flag given; defaulting to --moe-cache-auto for "
                f"auto-selected strategy {config.moe_strategy!r}"
            )

    if is_moe and config.moe_strategy == "fused":
        # An explicit 'fused' keeps the experts resident, so there is no slot cache to size. The
        # sizing flags no longer redirect the backend, so ignore them here and say so -- the
        # geometry the user asked for is what runs. Report the flag actually passed: --moe-cache-
        # rate was already folded into moe_cache_size above, and the three are mutually exclusive.
        if config.moe_cache_rate:
            inert = f"--moe-cache-rate={config.moe_cache_rate}"
        elif config.moe_cache_size:
            inert = f"--moe-cache-size={config.moe_cache_size}"
        elif getattr(config, "moe_cache_auto", False):
            inert = "--moe-cache-auto"
        else:
            inert = None
        if inert:
            logger.warning_rank0(
                f"MoE backend 'fused' keeps its experts resident; ignoring {inert} "
                "(use --moe-strategy offload to serve experts from a slot cache)"
            )
            override("moe_cache_size", 0)
            override("moe_cache_rate", None)
            override("moe_cache_auto", False)

    if is_moe and config.moe_strategy == "cpu":
        # CPU-compute decode keeps experts in host RAM and computes them on the CPU;
        # the GPU only holds the two-layer prefill double buffer. So the slot cache is
        # fixed at exactly two expert layers (prefill overlap requires >= 2*num_experts)
        # and --moe-cache-size / --moe-cache-auto / --moe-cache-rate do not apply.
        num_experts = config.model_config.num_experts
        if getattr(config, "moe_cache_auto", False):
            override("moe_cache_auto", False)
        override("moe_cache_size", 2 * num_experts)
        override("moe_prefill_overlap", True)
        logger.info_rank0(
            f"MoE backend 'cpu': decode computes experts on CPU; GPU keeps a "
            f"two-layer prefill buffer (moe_cache_size={2 * num_experts})"
        )

    if (
        is_moe
        and expert_quant not in ("none", "fp8_block")
        and not is_offload_moe_strategy(config.moe_strategy)
    ):
        raise ValueError(
            f"{expert_quant} experts require --moe-strategy offload or cpu, "
            f"got {config.moe_strategy!r}"
        )

    if is_moe and config.moe_cpu_layers and config.moe_strategy not in ("offload", "hybrid"):
        # the layer split needs the offload host banks + slot cache; 'cpu' already runs every layer on CPU, 'fused' keeps experts resident on the GPU (no host banks)
        raise ValueError(
            "--moe-cpu-layers requires --moe-strategy offload or hybrid (got "
            f"{config.moe_strategy!r}); use --moe-strategy cpu to run all layers on CPU"
        )

    if is_moe and config.moe_cpu_layers and config.moe_cpu_layers.strip() != "auto":
        if not _parse_cpu_layers_spec(config.moe_cpu_layers, model_config.num_moe_layers):
            override("moe_cpu_layers", None)

    if is_moe:
        object.__setattr__(model_config, "moe_strategy", config.moe_strategy)
        object.__setattr__(model_config, "decode_target", _decode_target(config))

    # Must stay LAST: page_size is only final here (_adjust_dsv4_config sets P=128, the
    # TRTLLM block sets 64). Also covers the programmatic LLM(...) path that bypasses parse_args.
    if config.num_token_override is not None:
        if config.num_page_override is not None:
            raise ValueError("--num-tokens and --num-pages are mutually exclusive")
        if config.num_token_override % config.page_size != 0:
            raise ValueError(
                f"--num-tokens {config.num_token_override} is not a multiple of the resolved "
                f"page size {config.page_size}; nearest valid values: "
                f"{config.num_token_override // config.page_size * config.page_size} or "
                f"{(config.num_token_override // config.page_size + 1) * config.page_size}"
            )
        override("num_page_override", config.num_token_override // config.page_size)

    # The rope cos/sin table is baked to rotary_config.max_position, and neither rope kernel
    # bounds-checks the position it gathers with -- a longer ceiling reads past the table.
    # DSV4 is exempt: it sizes its own table from the resolved max_seq_len (_adjust_dsv4_config).
    rotary = getattr(model_config, "rotary_config", None)
    seq_override = getattr(config, "max_seq_len_override", None)
    if seq_override is not None and rotary is not None and not is_dsv4:
        if seq_override > rotary.max_position:
            raise ValueError(
                f"--max-seq-len-override {seq_override} exceeds the model's "
                f"rope table ({rotary.max_position} positions). Serving past it would read "
                "out of bounds; extend the checkpoint's rope_scaling / "
                "max_position_embeddings in config.json instead."
            )

    # The startup ServerArgs dump is the *requested* config, printed in the frontend process
    # before any of the resolution above ran -- so "moe_strategy='auto'" is all it can say. This
    # is the one line that reports what actually runs, for every path (explicit backends never
    # hit an "Auto-selected ..." log at all).
    resolved = [
        f"attention_backend={config.attention_backend!r}",
        f"cache_type={getattr(config, 'cache_type', 'radix')!r}",
        f"page_size={config.page_size}",
    ]
    if is_moe:
        resolved.insert(0, f"moe_strategy={config.moe_strategy!r}")
    logger.info_rank0(f"Resolved config: {', '.join(resolved)}")
