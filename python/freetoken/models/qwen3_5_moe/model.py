from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    HostEmbedding,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.pipeline import RemoteLayer, layer_window, received_hidden
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, moe_layer_offset: int = 0):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        # the offload cache indexes MoE layers rank-locally under the pipeline engine
        self.mlp = (
            Qwen3_5MoE(config, layer_id - moe_layer_offset) if config.moe_enabled else Qwen3_5DenseMLP(config)
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5MTP(BaseOP):
    """The checkpoint's MTP draft head (``mtp.*``): one more full-attention decoder layer that
    predicts the token after the next one. Per row, with ``H`` the target's final hidden state
    (after the final norm, the lm_head input) and ``t_next`` the token that follows::

        x  = fc(cat(norm_e(embed(t_next)), norm_h(H)))      # [T, 2*hidden] -> [T, hidden]
        x' = layer(x)                                       # full attention + MoE, own KV slab
        h  = norm(x')                                       # -> shared lm_head -> logits of t_next+1

    (the vLLM/sglang ``Qwen3NextMTP`` graph). Multi-step drafting feeds ``h`` back as the next
    step's ``H``. The head's dense weights are bf16 in the modelopt checkpoints (``mtp*`` is
    excluded from quantization), so its layer is built unquantized; its routed experts become
    one more NVFP4 bank layer (engine._append_mtp_bank), indexed by ``layer_id``."""

    def __init__(
        self, config: ModelConfig, layer_id: int, moe_layer_offset: int = 0, own_embedding: bool = False
    ) -> None:
        from dataclasses import replace

        hidden = config.hidden_size
        self.layer_id = layer_id
        self.pre_fc_norm_embedding = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.fc = LinearReplicated(2 * hidden, hidden, has_bias=False)
        head_config = replace(config, attn_quant="none", dense_quant="none")
        self.layers = OPList([Qwen3_5DecoderLayer(head_config, layer_id, moe_layer_offset=moe_layer_offset)])
        self.norm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        # the head shares the target's embedding; a pipeline rank without the embedding table
        # carries its own copy (mtp.embed_tokens, duplicated by the loader) in pinned host
        # memory, gathered by the GPU in place (so the head's graphs can embed too)
        self.embed_tokens = HostEmbedding(config.vocab_size, hidden) if own_embedding else None
        self._embed_ref = None  # the target's embedding when shared (not a state-dict child)

    def forward(self, hidden: torch.Tensor, next_ids: torch.Tensor, batch=None) -> torch.Tensor:
        """``hidden [T, hidden]`` (the target's final hidden state, or the head's own output
        for a chained draft step) + ``next_ids [T]`` -> the head's normed hidden state
        ``[T, hidden]`` (its KV / expert routing use the active batch's metadata; ``batch`` is
        the engine's common draft-head signature, unused here)."""
        emb = self.embed_tokens if self.embed_tokens is not None else self._embed_ref
        assert emb is not None, "MTP head has no embedding table"
        e = self.pre_fc_norm_embedding.forward(emb.forward(next_ids).to(hidden.dtype))
        h = self.pre_fc_norm_hidden.forward(hidden)
        x = self.fc.forward(torch.cat([e, h], dim=-1))
        x, residual = self.layers.op_list[0].forward(x, None)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x

    @staticmethod
    def to_head(h: torch.Tensor) -> torch.Tensor:
        """The head's output is already the lm_head input."""
        return h


class Qwen3_5Model(BaseOP):
    """The decoder stack, or under the pipeline engine (``--pp-size``) this rank's slice of it:
    the embedding on the first rank, the final norm on the last; the other layers are
    ``RemoteLayer`` placeholders (see ``models/pipeline.py``)."""

    def __init__(self, config: ModelConfig):
        self._image_token_id = config.image_token_id
        win = layer_window(config)
        self._window = win
        self.embed_tokens = None
        if win.first:
            if getattr(config, "embed_host", False):
                # --host-embedding: the table stays in pinned host memory, rows gathered in place
                assert not config.tie_word_embeddings, "host embedding needs an untied lm_head"
                self.embed_tokens = HostEmbedding(config.vocab_size, config.hidden_size)
            else:
                self.embed_tokens = VocabParallelEmbedding(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.hidden_size,
                )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, moe_layer_offset=win.start)
                if win.owns(layer_id)
                else RemoteLayer()
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps) if win.last else None
        # indices, not layer objects: the offload-cache walk visits every tuple on the model
        self._local_ids = win.local_ids

    @property
    def pp_first(self) -> bool:
        return self._window.first

    @property
    def pp_last(self) -> bool:
        return self._window.last

    @property
    def layer_start(self) -> int:
        return self._window.start

    def forward(self, input_ids: torch.Tensor, hidden_in: torch.Tensor | None = None) -> torch.Tensor:
        """``hidden_in`` is the residual stream ``[T, hidden]`` received from the previous
        pipeline rank (required on non-first ranks); a non-last rank returns the stream it hands
        on instead of the normed head input."""
        if self._window.first:
            x = self.embed_tokens.forward(input_ids)
            mm_embeds = getattr(get_global_ctx().batch, "mm_embeds", None)
            if mm_embeds is not None and self._image_token_id is not None:
                # image soft tokens (vision tower + merger output, already in the text width)
                # replace the placeholder embeddings; the count was checked at admission.
                # Only the embedding rank sees input embeddings.
                mask = (input_ids == self._image_token_id).unsqueeze(-1)
                x = x.masked_scatter(mask, mm_embeds.to(x.dtype))
        else:
            assert hidden_in is not None, "non-first pipeline rank needs the received residual stream"
            x = hidden_in
        # residual-stream form: the first local layer folds x into the residual itself
        residual: torch.Tensor | None = None
        layers = self.layers.op_list
        dbg = get_global_ctx().debug_layer_outs
        for i in self._local_ids:
            x, residual = layers[i].forward(x, residual)
            if dbg is not None:  # FT_SPEC_CHECK_STEP: the residual stream after layer i
                dbg.append((i, (residual + x).detach().clone()))
        if not self._window.last:
            return x + residual  # the residual stream for the next pipeline rank
        x, _ = self.norm.forward_add_residual(x, residual)
        self._last_hidden = x  # the MTP draft head reads the final hidden state (lm_head input)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        # the residual stream crossing a pipeline boundary is [T, hidden] in the model dtype
        self.pp_hidden_width = config.hidden_size
        self._pp_hidden_dtype = torch.get_default_dtype()
        # --spec-mtp: the draft head rides the head-owning rank as layer mtp_layer_id
        self.mtp = None
        mtp_id = getattr(config, "mtp_layer_id", None)
        if mtp_id is not None and self.model.pp_last:
            self.mtp = Qwen3_5MTP(
                config, mtp_id, moe_layer_offset=self.model.layer_start,
                own_embedding=self.model.embed_tokens is None,
            )
            if self.model.embed_tokens is not None:
                self.mtp._embed_ref = self.model.embed_tokens
        if not self.model.pp_last:
            self.lm_head = None
        elif getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            if config.tie_word_embeddings and self.model.embed_tokens is None:
                raise NotImplementedError(
                    "tied embeddings need the embedding and the head on the same pipeline rank"
                )
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

    @property
    def host_resident_prefixes(self) -> tuple[str, ...]:
        """State-dict key prefixes the engine materializes in pinned host memory instead of on
        the device: the input embedding table under --host-embedding, and the draft head's own
        embedding copy on a rank without the table."""
        prefixes = []
        if getattr(self.model.embed_tokens, "host_resident", False):
            prefixes.append("model.embed_tokens.")
        mtp = self.mtp
        if mtp is not None and getattr(mtp.embed_tokens, "host_resident", False):
            prefixes.append("mtp.embed_tokens.")
        return tuple(prefixes)

    def remap_loaded_weight(
        self, name: str, tensor: torch.Tensor, model_state: dict
    ) -> list[tuple[str, torch.Tensor]]:
        """Loader seam: a pipeline rank that holds the draft head but not the embedding gets its
        own copy of the table."""
        if name == "model.embed_tokens.weight" and "mtp.embed_tokens.weight" in model_state:
            return [(name, tensor), ("mtp.embed_tokens.weight", tensor)]
        return [(name, tensor)]

    @property
    def last_hidden(self) -> torch.Tensor:
        """The final hidden state ``[T, hidden]`` of the last (eager) forward."""
        return self.model._last_hidden

    def spec_rollback(self, batch: Batch, accepted: int, ctx) -> None:
        """Roll the per-request state back to the first ``accepted`` rows of the verify window:
        the GDN recurrent + conv states, from the stashes the GDN ops recorded."""
        for stash in ctx.spec_stash:
            stash.restore(accepted)
        ctx.spec_stash = []

    def forward(self) -> torch.Tensor:
        input_ids = get_global_ctx().batch.input_ids
        hidden_in = None
        if not self.model.pp_first:
            hidden_in = received_hidden(input_ids, self.pp_hidden_width, self._pp_hidden_dtype)
        out = self.model.forward(input_ids, hidden_in)
        if not self.model.pp_last:
            return out  # the residual stream for the next pipeline rank
        return self.lm_head.forward(out)


__all__ = ["Qwen3_5MoEForCausalLM", "Qwen3_5MTP"]
