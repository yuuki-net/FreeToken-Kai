from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, List, Mapping

import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig
    from freetoken.models.config import SlotStateSpec

logger = init_logger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # --hf-overrides: applied to the checkpoint config the model is built from (cached_load_hf_config)
    hf_overrides: Mapping[str, Any] = field(default_factory=dict)
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # --moe-expert-gguf: read the routed experts from this GGUF file (one shard; its block bytes
    # become the expert banks, q2_0 / q4_0) while every other weight stays with --model. The
    # expert format tag is read from the file at config time (models/qwen4_exp/gguf_experts.py).
    moe_expert_gguf: str | None = None
    # --embed-weights: "gpu" (default) keeps the input embedding tables in VRAM; "host" keeps the untied ones in pinned host RAM, read zero-copy by the lookup.
    embed_weights: str = "gpu"
    # DeepSeek-V4.1 Decoder SWA Bounded Replay: "bounded" (default) runs the decoder layers on each
    # prompt's last window; "exact" runs them on every token.
    swa_decoder_replay: str = "bounded"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    # --moe-cache-mem: slot-cache budget in bytes, turned into a slot count once the expert banks are loaded
    moe_cache_mem: int | None = None
    # --moe-cache-scope: unified (one LRU pool per expert quant) or per-layer (one pool per layer)
    moe_cache_scope: str = "unified"
    # --moe-cache-split: how pools share the slots; equal or profile:<path> (by expert use from a profile file)
    moe_cache_split: str = "equal"
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    # --prefill-chunk-budget: the share of free VRAM one prefill chunk's transient may take.
    # The engine measures that transient per token at startup and sizes max_extend_tokens to
    # fit, then re-solves before each prefill against the VRAM free at that moment. 0 turns
    # the whole thing off and max_extend_tokens is used exactly as given.
    prefill_chunk_budget: float = 0.55
    # --prefill-mixer-pieces: run each prefill chunk's sequence mixers (GDN / attention) over
    # this many consecutive pieces and its MoE over the whole chunk (models/prefill_pieces.py).
    # The mixers set the transient that caps the chunk width, so the probe measures less and
    # the solver picks wider chunks: fewer chunks, fewer expert-bank transfers. 1 = off.
    prefill_mixer_pieces: int = 1
    # --pp-prefill-group: the last of two pipeline ranks runs up to this many consecutive prefill
    # chunks of one request layer by layer, streaming each offloaded expert bank once per group
    # (engine/prefill_group.py). 1 = off.
    pp_prefill_group: int = 1
    # --pp-send-ahead: residual streams the first pipeline rank may have in flight to the next
    # rank at once (distributed/pipeline.py). 1 = the send waits for the peer's receive.
    pp_send_ahead: int = 1
    # --prefill-profile: log where each prefill forward's wall time goes, on every rank
    # (utils/prefill_profile.py)
    prefill_profile: bool = False
    # --kv-cache-dtype: "auto" (16-bit), "q8_0" or "q4_0". Narrows the paged KV slab so
    # --moe-cache-auto can hand the difference to the expert cache (see kvcache/kv_quant.py).
    kv_cache_dtype: str | None = None
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 13.0
    # (cudaMemcpyBatchAsync) or HIP (hipMemcpyBatchAsync);
    # no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    # decode miss-rate counters, per-expert routing histogram and routing ring, captured into
    # the cuda graph (OffloadMoeCache.enable_stats); --no-moe-collect-stats turns them off
    moe_collect_stats: bool = True
    # --moe-stats-out: write the decode routing histogram (per layer, per expert), the
    # realized miss rates and OffloadMoeCache.decode_routing_stats() to this path at every
    # idle and at shutdown. Implies moe_collect_stats.
    moe_stats_out: str | None = None
    # --moe-bank-ram: cap on host RAM for the expert banks. Experts beyond the cap are
    # renumbered out of the resident range and read from a cold bank file instead (see
    # moe/bank_disk.py). Unset = every expert resident, which is today's behaviour. "auto" is
    # turned into a size when the arguments are parsed (moe/disk_probe.auto_bank_ram), so every
    # rank splits the same number.
    moe_bank_ram: str | None = None
    # --moe-bank-stats: --moe-stats-out histogram(s) that order the placement. Without one
    # the ordering falls back to logical id, i.e. it ignores routing entirely and the cold
    # half is an arbitrary fifth of the experts.
    moe_bank_stats: list[str] | None = None
    # --moe-bank-dir: the directory of the bank file (bank.ftmb, every MoE layer). Defaults to
    # the one a packed checkpoint names, else ~/.cache/freetoken/bankmap/<model>
    # (bank_pack.bank_path_for).
    moe_bank_dir: str | None = None
    # --moe-bank-readahead: "off" reports the device readahead window against the model's
    # block geometry; "auto" writes the recommended window to sysfs, a number writes that
    # many kB, each rank before it opens its mapping. Device-wide and left set after exit,
    # hence opt-in (MappedTier._apply_readahead).
    moe_bank_readahead: str = "off"
    # --moe-bank-rewarm: seconds of scheduler idle after which a --moe-bank-ram bank's
    # file-backed rows are checked and, if the page cache has lost them, read back in file
    # order until a request arrives (moe/bank_rewarm.py). 0 = off.
    moe_bank_rewarm: float = 0.0
    # --moe-bank-prefetch: the CPU MoE executor advises (mincore + MADV_WILLNEED) the
    # file-backed rows each task routes to before its workers read them, instead of leaving
    # them to 4 KiB faults and readahead windows (kernel/csrc/cpu_moe, PrefetchSpan). Off by
    # default until measured end to end.
    moe_bank_prefetch: bool = False
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    # Hybrid with locked layers (banks the pin quota could not take, OS-locked instead): a locked layer keeps GPU
    # slots of its own pools and fetches this share of each decode step's misses through pinned staging; the rest
    # are computed on the CPU. -1 (default) = auto: per layer from its `ft bench bw` row (bench_profile.locked_fetch_fraction),
    # a fixed cap of 1 without one. 0 = locked layers decode on the CPU only. moe_locked_max_fetch caps the fetches per
    # (layer, step) and sizes the staging (that many rows of the widest expert, per bank).
    moe_locked_fetch: float = -1.0
    moe_locked_max_fetch: int = 8
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Pinned host slots the hybrid tree moves its least recently used GDN snapshots to when the
    # device slots run short, instead of dropping them (0 = off). Counted against the pin budget.
    linear_state_host_slots: int = 0
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
    # --spec-mtp-gguf (Kai): the draft head from a separate MTP GGUF instead of the checkpoint (the
    # Qwen3.8-Flash-Next GGUFs ship it as MTP/mtp-*.gguf). None = the checkpoint's mtp.* block
    spec_mtp_gguf: str | None = None
    # --spec-draft-head-quant: the MTP drafts score through this copy of the lm_head (none/q4_0/q8_0)
    spec_draft_head_quant: str = "none"
    # --spec-min-p: a draft whose head probability is below this ends the window (0 = fixed K)
    spec_min_p: float = 0.0
    # --host-embedding: keep the input embedding table in pinned host memory (the GPU gathers
    # rows in place); ~1 GB of VRAM back for KV pages on a 250k-vocabulary model. Applies to
    # the rank that owns the embedding (rank 0 under the pipeline engine).
    host_embedding: bool = False
    # upstream's speculative decoding (verify batches; no proposer is registered upstream yet).
    # Separate from --spec-mtp above, which runs its windows as extend batches.
    speculative_algorithm: str | None = None
    speculative_num_steps: int = 0
    speculative_draft_model_path: str | None = None
    # the proposer's states on the target's linear-state slots, resolved by _adjust_config
    spec_slot_states: tuple[SlotStateSpec, ...] = ()
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    def __post_init__(self):
        if not (self.moe_locked_fetch == -1 or 0.0 <= self.moe_locked_fetch <= 1.0):
            raise ValueError(f"moe_locked_fetch must be -1 (auto) or in [0, 1], got {self.moe_locked_fetch}")
        if self.moe_locked_fetch > 0 and round(self.moe_locked_fetch * (1 << 16)) == 0:
            # the ensure kernel takes the share in Q16: this one rounds to 0 and the layers would never fetch
            raise ValueError(f"moe_locked_fetch {self.moe_locked_fetch} rounds to 0 in Q16; pass 0 to keep locked layers CPU-only")
        if self.moe_locked_max_fetch < 1:
            raise ValueError(f"moe_locked_max_fetch must be at least 1, got {self.moe_locked_max_fetch}")
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path, self.hf_overrides)

    @property
    def is_pp(self) -> bool:
        return self.parallel == "pp" and self.tp_info.size > 1

    @property
    def tp_size(self) -> int:
        """Shard count for the layer / KV / GDN-state math: the pipeline ranks split layers,
        not tensors, so every layer and pool sees 1 there. (tp_info.size stays the world size.)"""
        return 1 if self.is_pp else self.tp_info.size

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def full_model_config(self) -> ModelConfig:
        """The whole model, before any pipeline windowing."""
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built and getattr(hf_config, key, None) is not None:
                setattr(hf_config, key, None)
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        embed_quant = None
        if self.dense_quant in ("fp8", "q8_0"):
            from freetoken.layers.quantization import LoadTimeFp8Config, LoadTimeQ8Config

            # the layers follow the wrapped config; the embedding table is not a quantized
            # layer kind, so it keeps its own switch (fp8 rows under both: it is a gather)
            quant = (LoadTimeFp8Config if self.dense_quant == "fp8" else LoadTimeQ8Config)(quant)
            embed_quant = "fp8_pertensor"
        elif self.dense_quant != "none":
            raise ValueError(f"--dense-quant {self.dense_quant!r}: supported values are none, fp8, q8_0")
        # The weight readers are handed the model path alone and read their schemes from here,
        # so install what the layers are built from -- the --dense-quant wrapper included, or a
        # reader would quantize against the checkpoint's own schemes and disagree with them.
        set_quant_config(quant)
        config = _load_attr(spec.module, spec.parse_config)(hf_config)
        if embed_quant is not None:
            config = replace(config, embed_quant=embed_quant)
        if self.moe_expert_gguf:
            # the routed experts come packed from a GGUF: the format tag the cache / kernels /
            # CPU executor dispatch on is the file's, and the model's own expert scheme steps
            # aside for them (layers/moe.make_moe_layer)
            from freetoken.models.qwen4_exp.gguf_experts import gguf_expert_format

            fmt = gguf_expert_format(self.moe_expert_gguf)
            logger.info_rank0(f"routed experts from GGUF: {self.moe_expert_gguf} ({fmt} banks)")
            config = replace(config, expert_quant=fmt, moe_weight_format=fmt, moe_expert_gguf=self.moe_expert_gguf)
        return replace(config, quant=quant)

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
    def encodes_here(self) -> bool:
        """This process builds, loads and runs the encoder towers: images enter the residual
        stream at the embedding, so under ``--pp-size`` only the first rank needs them. The
        other ranks still serve the model the towers imply (3-axis rope, ``active_encoders``)."""
        return bool(self.active_encoders) and self.pp_is_first

    @property
    def builds_tower(self) -> bool:
        """This process builds, loads and runs the encoder towers on the GPU: the encoding rank,
        unless ``--mm-encoder-weights cpu`` moved the tower to the tokenizer worker (the items then
        arrive with their embeddings, and this rank only gathers them)."""
        return self.encodes_here and self.mm.encoder_weights != "cpu"

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
        config = self.full_model_config
        mtp_layer = config.num_layers if (self.spec_mtp > 0 and self.pp_is_last) else None
        if self.pp_layer_range is not None or mtp_layer is not None:
            from freetoken.models.config import window_model_config

            start, end = self.pp_layer_range or (0, config.num_layers)
            config = window_model_config(config, start, end, extra_full_layer=mtp_layer)
        if mtp_layer is not None and self.spec_mtp_gguf:
            config = replace(config, mtp_gguf=self.spec_mtp_gguf)
        if self.host_embedding and self.pp_is_first:
            config = replace(config, embed_host=True)
        if config.vision_config is not None and not self.builds_tower:
            # a later pipeline rank, or a CPU tower, builds none; the rope sections parsed from it stay
            config = replace(config, vision_config=None)
        elif config.vision_config is not None and "dtype" in getattr(config.vision_config, "__dataclass_fields__", {}):
            # the Qwen VL tower's compute dtype (--mm-encoder-dtype); other families' towers follow the model
            config = replace(
                config, vision_config=replace(config.vision_config, dtype=self.mm.resolve_encoder_dtype(self.dtype))
            )
        return config

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.table_positions

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def decode_rows_per_req(self) -> int:
        """Query rows each running request takes per decode step; every per-request decode allowance derives from it."""
        return self.speculative_num_steps + 1 if self.speculative_algorithm is not None else 1

    @property
    def slot_states(self) -> tuple[SlotStateSpec, ...]:
        """Every per-request state the linear-state pool holds: the model's own, then the proposer's."""
        return self.model_config.slot_states + self.spec_slot_states

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
