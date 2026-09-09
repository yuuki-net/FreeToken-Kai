from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.register import _load_attr, get_model_spec
from freetoken.utils import cached_load_hf_config

if TYPE_CHECKING:
    from freetoken.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    # NVFP4 routed-expert GEMM backend (--nvfp4-backend): auto|marlin|flashinfer|triton.
    nvfp4_backend: str = "triton"
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    # --prefill-chunk-budget: the share of free VRAM one prefill chunk's transient may take.
    # The engine measures that transient per token at startup and sizes max_extend_tokens to
    # fit, then re-solves before each prefill against the VRAM free at that moment. 0 turns
    # the whole thing off and max_extend_tokens is used exactly as given.
    prefill_chunk_budget: float = 0.55
    # --kv-cache-dtype: "auto" (16-bit), "q8_0" or "q4_0". Narrows the paged KV slab so
    # --moe-cache-auto can hand the difference to the expert cache (see kvcache/kv_quant.py).
    kv_cache_dtype: str | None = None
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # --moe-stats-out: write the decode routing histogram (per layer, per expert), the
    # realized miss rates and OffloadMoeCache.decode_routing_stats() to this path at
    # shutdown. Setting it also turns on the per-expert histogram
    # (OffloadMoeCache.collect_decode_freq), which the cache only accumulates outside a
    # captured graph -- pass --disable-cuda-graph for a collection run, or the histogram
    # counts the capture-time warmup routing instead of the real one.
    moe_stats_out: str | None = None
    # --moe-bank-ram: cap on host RAM for the expert banks. Experts beyond the cap are
    # renumbered out of the resident range and read from a cold bank file instead (see
    # moe/bank_disk.py). Unset = every expert resident, which is today's behaviour.
    moe_bank_ram: str | None = None
    # --moe-bank-stats: --moe-stats-out histogram(s) that order the placement. Without one
    # the ordering falls back to logical id, i.e. it ignores routing entirely and the cold
    # half is an arbitrary fifth of the experts.
    moe_bank_stats: list[str] | None = None
    # --moe-bank-dir: where the cold bank file lives. Defaults beside the checkpoint.
    moe_bank_dir: str | None = None
    # CPU MoE backend (--moe-backend cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-backend offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-backend cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-backend hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    # "tp": tp_info ranks shard every layer (tensor parallel). "pp": tp_info ranks each run a
    # contiguous block of decoder layers on their own GPU (pipeline / layer split, see
    # distributed/pipeline.py); layer math then sees TP=1.
    parallel: str = "tp"
    # --pp-layers: the size-1 layer boundaries of the pipeline split; None = even split.
    pp_split: tuple[int, ...] | None = None
    # --dense-quant: quantize the checkpoint's bf16 dense (non-expert) projections at load.
    # "fp8": per-row fp8-e4m3 + fp32 scale, W8A16 (attention, GDN qkv|z / out, shared expert,
    # lm_head, embedding); layers the checkpoint already quantizes keep their own format.
    dense_quant: str = "none"
    # --spec-mtp K: MTP speculative decoding with K drafts per step (0 = off). The draft head
    # (the checkpoint's mtp.* block) is built on the last pipeline rank (the only process when
    # single-GPU) as one extra full-attention layer (id = num_layers) with its own KV slab and
    # one extra expert-bank layer. Single request.
    spec_mtp: int = 0
    # --host-embedding: keep the input embedding table in pinned host memory (the GPU gathers
    # rows in place); ~1 GB of VRAM back for KV pages on a 250k-vocabulary model. Applies to
    # the rank that owns the embedding (rank 0 under the pipeline engine).
    host_embedding: bool = False
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @property
    def is_pp(self) -> bool:
        return self.parallel == "pp" and self.tp_info.size > 1

    @property
    def tp_size(self) -> int:
        """Shard count for the layer / KV / GDN-state math: the pipeline ranks split layers,
        not tensors, so every layer and pool sees 1 there. (tp_info.size stays the world size.)"""
        return 1 if self.is_pp else self.tp_info.size

    @cached_property
    def full_model_config(self) -> ModelConfig:
        """The whole model, before any pipeline windowing."""
        spec = get_model_spec(self.hf_config.architectures[0])
        parse_config = _load_attr(spec.module, spec.parse_config)
        config = parse_config(self.hf_config)
        if self.dense_quant == "fp8":
            from dataclasses import replace

            # only the parts the checkpoint left bf16: a native fp8/nvfp4 layer keeps its format
            fields = {
                f: "fp8_pertensor"
                for f in ("attn_quant", "dense_quant", "lm_head_quant", "embed_quant")
                if getattr(config, f, "none") == "none"
            }
            config = replace(config, **fields)
        elif self.dense_quant != "none":
            raise ValueError(f"--dense-quant {self.dense_quant!r}: supported values are none, fp8")
        return config

    @cached_property
    def pp_layer_range(self) -> tuple[int, int] | None:
        """Decoder layers ``[start, end)`` this rank runs under ``--parallel pp``; None otherwise."""
        if not self.is_pp:
            return None
        from freetoken.distributed import pp_layer_range

        return pp_layer_range(
            self.full_model_config.num_layers, self.pp_split, self.tp_info.rank, self.tp_info.size
        )

    @property
    def pp_is_first(self) -> bool:
        """This process owns the embedding (rank 0 of the pipeline, or the only process)."""
        return (not self.is_pp) or self.tp_info.rank == 0

    @property
    def pp_is_last(self) -> bool:
        """This process owns the head (the last pipeline rank, or the only process)."""
        return (not self.is_pp) or self.tp_info.rank == self.tp_info.size - 1

    @cached_property
    def model_config(self) -> ModelConfig:
        """The model as THIS process serves it: the full config, or its pipeline window (only
        this rank's layers in the attention groups, slot states and MoE layer count, so the
        KV/GDN pools, attention backends and the expert cache size themselves per rank). The
        MTP draft head joins the full-attention group of the head-owning rank as layer
        num_layers; --host-embedding marks the embedding-owning rank's table host-resident."""
        from dataclasses import replace

        config = self.full_model_config
        mtp_layer = config.num_layers if (self.spec_mtp > 0 and self.pp_is_last) else None
        if self.pp_layer_range is not None or mtp_layer is not None:
            from freetoken.models.config import window_model_config

            start, end = self.pp_layer_range or (0, config.num_layers)
            config = window_model_config(config, start, end, extra_full_layer=mtp_layer)
        if self.host_embedding and self.pp_is_first:
            config = replace(config, embed_host=True)
        return config

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
