"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import try_get_pp_info
from freetoken.layers import (
    BaseOP,
    Fp8VocabParallelEmbedding,
    HostEmbedding,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.pipeline import RemoteLayer as _RemoteLayer
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .moe import Qwen4ExpMoE
from .ple import PLELayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int, prefix: str) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        quant_config=config.quant,
        prefix=prefix,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(
        self, config: ModelConfig, layer_id: int, moe_layer_offset: int = 0, *, prefix: str = ""
    ) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id, f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        # the offload cache indexes MoE layers rank-locally under the pipeline engine
        self.mlp = Qwen4ExpMoE(config, layer_id - moe_layer_offset, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")
        self.ple = (
            PLELayer(config, layer_id, prefix=f"{prefix}.ple") if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpMTP(BaseOP):
    """The checkpoint's MTP draft head (``mtp.*``): one more decoder layer over the target's
    residual streams. Per token, with ``R [T, hc*hidden]`` the target's final-layer residual
    (BEFORE its stream mixer) and ``t_next`` the token that follows::

        R_i' = fc_hidden(norm_h(R)_i) + fc_embedding(norm_e(embed(t_next)))   # every stream i
        R''  = layer(R')            # full-attention QSA layer + MoE, own KV / index slab
        h    = mixer.mix(R'')       # [T, hidden]  ->  shared lm_head  ->  logits of t_next+1

    (the llama.cpp NextN graph: grouped per-stream rmsnorm on the wide residual, the two fc's
    fused into one block projection, the head's own hyper-connection mixer as output norm).
    Multi-step drafting feeds ``R''`` back as the next step's residual."""

    def __init__(self, config: ModelConfig, layer_id: int, moe_layer_offset: int, own_embedding: bool) -> None:
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        hidden = config.hidden_size
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(self.hc_count * hidden, config.rms_norm_eps, self.hc_count)
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(hidden, config.rms_norm_eps, 1)
        # bf16, or per-row fp8 under --dense-quant fp8 (quantized at load like every projection)
        self.fc_hidden = LinearReplicated(
            hidden, hidden, has_bias=False, quant_config=config.quant, prefix="mtp.fc_hidden"
        )
        self.fc_embedding = LinearReplicated(
            hidden, hidden, has_bias=False, quant_config=config.quant, prefix="mtp.fc_embedding"
        )
        self.layers = OPList([
            Qwen4ExpDecoderLayer(
                config, layer_id, moe_layer_offset=moe_layer_offset, prefix="mtp.layers.0"
            )
        ])
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix="mtp.hyper_connection_mixer"
        )
        # the head shares the target's embedding; a pipeline rank without the embedding table
        # carries its own copy (mtp.embed_tokens, duplicated by the loader) in pinned host
        # memory, gathered by the GPU in place (so the head's graphs can embed too) and the
        # VRAM goes to KV pages instead. The copy stays bf16 under --dense-quant fp8.
        self.embed_tokens = HostEmbedding(config.vocab_size, hidden) if own_embedding else None
        self._embed_ref = None  # the target's embedding when shared (not a state-dict child)

    def forward(self, residual: torch.Tensor, next_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        """``residual [T, hc*hidden]`` + ``next_ids [T]`` -> the head's own residual ``[T, hc*hidden]``
        (its KV / expert routing use ``batch``'s positions and metadata)."""
        emb = self.embed_tokens if self.embed_tokens is not None else self._embed_ref
        assert emb is not None, "MTP head has no embedding table"
        t = residual.shape[0]
        rn = self.pre_fc_norm_hidden.forward(residual)
        fh = self.fc_hidden.forward(rn.reshape(t * self.hc_count, self.hidden_size))
        fh = fh.reshape(t, self.hc_count * self.hidden_size)
        e = self.pre_fc_norm_embedding.forward(emb.forward(next_ids).to(residual.dtype))
        fe = self.fc_embedding.forward(e)
        r = fh + fe.repeat(1, self.hc_count)
        return self.layers.op_list[0].forward(r, batch)

    def mix(self, residual: torch.Tensor) -> torch.Tensor:
        """Collapse the head's residual streams into the lm_head input ``[T, hidden]``."""
        return self.hyper_connection_mixer.mix(residual)[0]

    def to_head(self, h: torch.Tensor) -> torch.Tensor:
        """The engine's common draft-head seam: the head output -> the lm_head input."""
        return self.mix(h)


class Qwen4ExpModel(BaseOP):
    """The decoder stack, or under the pipeline engine (``--parallel pp``) this rank's slice
    of it: layers ``[start, end)`` with the embedding on the first rank and the stream mixer
    on the last; the other layers are ``_RemoteLayer`` placeholders."""

    def __init__(self, config: ModelConfig, *, prefix: str = "model") -> None:
        self.hc_count = config.qwen4_args.hc_count
        self._image_token_id = config.image_token_id
        pp = try_get_pp_info()
        start, end = (0, config.num_layers) if pp is None else (pp.start, pp.end)
        self._pp_first = start == 0
        self._pp_last = end == config.num_layers
        embed_cls = (
            Fp8VocabParallelEmbedding
            if getattr(config, "embed_quant", "none") == "fp8_pertensor"
            else VocabParallelEmbedding
        )
        self.embed_tokens = (
            embed_cls(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
            if self._pp_first
            else None
        )
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(
                    config, layer_id, moe_layer_offset=start,
                    prefix=f"{prefix}.layers.{layer_id}",
                )
                if start <= layer_id < end
                else _RemoteLayer()
                for layer_id in range(config.num_layers)
            ]
        )
        self.hyper_connection_mixer = (
            GatedResidual(
                config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer"
            )
            if self._pp_last
            else None
        )
        # Indices, not layer objects: the offload-cache walk (iter_offload_moe_layers) visits
        # every tuple on the model, so a second reference would attach each MoE layer twice.
        self._local_ids = tuple(range(start, end))
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(
            self.layers.op_list[i].ple for i in self._local_ids if self.layers.op_list[i].ple is not None
        )

    @property
    def pp_first(self) -> bool:
        return self._pp_first

    @property
    def pp_last(self) -> bool:
        return self._pp_last

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(
        self, input_ids: torch.Tensor, batch: Batch, hidden_in: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``hidden_in`` is the residual stream ``[T, hc_count*hidden]`` received from the
        previous pipeline rank (required on non-first ranks); a non-last rank returns the
        stream it hands on instead of the mixed ``[T, hidden]`` head input."""
        if self._pp_first:
            embeds = self.embed_tokens.forward(input_ids)
            mm_embeds = getattr(batch, "mm_embeds", None)
            if mm_embeds is not None and self._image_token_id is not None:
                # image soft tokens (vision tower + merger output, already in the text
                # width) replace the placeholder embeddings; the count was checked at
                # admission. Only the embedding rank sees input embeddings.
                mask = (input_ids == self._image_token_id).unsqueeze(-1)
                embeds = embeds.masked_scatter(mask, mm_embeds.to(embeds.dtype))
            hidden = embeds.repeat(1, self.hc_count)
        else:
            assert hidden_in is not None, "non-first pipeline rank needs the received residual stream"
            hidden = hidden_in
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        layers = self.layers.op_list
        dbg = get_global_ctx().debug_layer_outs
        for i in self._local_ids:
            hidden = layers[i].forward(hidden, batch)
            if dbg is not None:
                dbg.append((i, hidden.detach().clone()))
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        if not self._pp_last:
            return hidden
        # the MTP draft head reads the final residual streams (before the mixer)
        self._last_residual = hidden
        return self.hyper_connection_mixer.mix(hidden)[0]


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        # the residual stream crossing a pipeline boundary is [T, hc_count * hidden] in the
        # model dtype (the engine builds the model under torch_dtype(config.dtype))
        self.pp_hidden_width = config.qwen4_args.hc_count * config.hidden_size
        self._pp_hidden_dtype = torch.get_default_dtype()
        # --spec-mtp: the draft head rides the last pipeline rank as layer mtp_layer_id
        self.mtp = None
        mtp_id = getattr(config, "mtp_layer_id", None)
        if mtp_id is not None and self.model.pp_last:
            pp = try_get_pp_info()
            start = 0 if pp is None else pp.start
            self.mtp = Qwen4ExpMTP(
                config, mtp_id, moe_layer_offset=start,
                own_embedding=self.model.embed_tokens is None,
            )
            if self.model.embed_tokens is not None:
                self.mtp._embed_ref = self.model.embed_tokens
        if not self.model.pp_last:
            self.lm_head = None
        else:
            assert not (config.tie_word_embeddings and self.model.embed_tokens is None), (
                "tied embeddings need the embedding and the head on the same pipeline rank"
            )
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
                quant_config=config.quant,
                prefix="lm_head",
            )
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        if engine_config.ple_backend == "disk":
            from freetoken.utils import download_hf_weight

            from .ple_disk import DiskRowTable, resolve_row_source

            folder = download_hf_weight(engine_config.model_path)
            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
            }
            disk_table = DiskRowTable(
                resolve_row_source(folder),
                constants,
                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),
                max_extend_tokens=engine_config.max_extend_tokens,
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    @property
    def host_resident_prefixes(self) -> tuple[str, ...]:
        """State-dict key prefixes the engine materializes in pinned host memory instead of
        on the device (the draft head's own embedding copy)."""
        mtp = self.mtp
        if mtp is not None and getattr(mtp.embed_tokens, "host_resident", False):
            return ("mtp.embed_tokens.",)
        return ()

    def remap_loaded_weight(
        self, name: str, tensor: torch.Tensor, model_state: dict
    ) -> list[tuple[str, torch.Tensor]]:
        """Loader seam (engine._load_weight_state_dict): the reader fuses the GDN projections
        into one ``in_proj`` (qkv | z | b | a); the fp8 GDN keeps qkv|z quantized and b|a bf16
        as two buffers, so split the rows the way the model declares them."""
        if name.endswith(".linear_attn.in_proj.weight") and name not in model_state:
            base = name[: -len("in_proj.weight")]
            qkvz, ba = base + "in_proj_qkvz.weight", base + "in_proj_ba.weight"
            if qkvz in model_state and ba in model_state:
                n = model_state[qkvz].shape[0]
                assert n + model_state[ba].shape[0] == tensor.shape[0], (name, tensor.shape)
                return [(qkvz, tensor[:n]), (ba, tensor[n:])]
        if name == "model.embed_tokens.weight" and "mtp.embed_tokens.weight" in model_state:
            # a pipeline rank that holds the draft head but not the embedding gets its own copy
            return [(name, tensor), ("mtp.embed_tokens.weight", tensor)]
        return [(name, tensor)]

    @property
    def last_hidden(self) -> torch.Tensor:
        """The draft head's input from the last (eager) forward: the final-layer residual
        streams ``[T, hc*hidden]`` (head-owning rank)."""
        return self.model._last_residual

    def spec_rollback(self, batch: Batch, accepted: int, ctx) -> None:
        """Roll this rank's per-request state back to the first ``accepted`` rows of the verify
        window: GDN recurrent + conv states (from the layers' stashes) and the PLE n-gram
        context (from the host-side ids)."""
        for stash in ctx.spec_stash:
            stash.restore(accepted)
        ctx.spec_stash = []
        if self.model._ple:
            from freetoken.engine.spec import ngram_context_after

            from .ple import _state_slot, rewrite_ngram_context

            req = batch.reqs[0]
            args = self._config.qwen4_args
            ids = ngram_context_after(
                req.input_ids.tolist(), req.spec_drafts, accepted,
                args.ngram_size - 1, args.ngram_boundary_token_id,
            )
            rewrite_ngram_context(_state_slot(req), ids)

    def forward(self) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        hidden_in = None
        if not self.model.pp_first:
            hidden_in = ctx.pp_hidden_in
            if hidden_in is None:
                # warmup / probe forwards run without a peer: feed a zero stream of the right shape
                hidden_in = torch.zeros(
                    (batch.input_ids.numel(), self.pp_hidden_width),
                    dtype=self._pp_hidden_dtype,
                    device=batch.input_ids.device,
                )
        out = self.model.forward(batch.input_ids, batch, hidden_in)
        if not self.model.pp_last:
            return out  # the residual stream for the next pipeline rank
        return self.lm_head.forward(out)


__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM", "Qwen4ExpModel", "build_linear_mixer"]
