from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
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

    def __init__(self, config: ModelConfig, layer_id: int):
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
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
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

    def __init__(self, config: ModelConfig, layer_id: int) -> None:
        from dataclasses import replace

        hidden = config.hidden_size
        self.layer_id = layer_id
        self.pre_fc_norm_embedding = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.fc = LinearReplicated(2 * hidden, hidden, has_bias=False)
        head_config = replace(config, attn_quant="none", dense_quant="none")
        self.layers = OPList([Qwen3_5DecoderLayer(head_config, layer_id)])
        self.norm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self._embed_ref = None  # the target's embedding (shared; not a state-dict child)

    def forward(self, hidden: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        """``hidden [T, hidden]`` (the target's final hidden state, or the head's own output
        for a chained draft step) + ``next_ids [T]`` -> the head's normed hidden state
        ``[T, hidden]`` (its KV / expert routing use the active batch's metadata)."""
        assert self._embed_ref is not None, "MTP head has no embedding table"
        e = self.pre_fc_norm_embedding.forward(self._embed_ref.forward(next_ids).to(hidden.dtype))
        h = self.pre_fc_norm_hidden.forward(hidden)
        x = self.fc.forward(torch.cat([e, h], dim=-1))
        x, residual = self.layers.op_list[0].forward(x, None)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self._image_token_id = config.image_token_id
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        mm_embeds = getattr(get_global_ctx().batch, "mm_embeds", None)
        if mm_embeds is not None and self._image_token_id is not None:
            # image soft tokens (vision tower + merger output, already in the text width)
            # replace the placeholder embeddings; the count was checked at admission
            mask = (input_ids == self._image_token_id).unsqueeze(-1)
            x = x.masked_scatter(mask, mm_embeds.to(x.dtype))
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        self._last_hidden = x  # the MTP draft head reads the final hidden state (lm_head input)
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        # --spec-mtp: the draft head as layer mtp_layer_id (== num_layers), sharing the embedding
        self.mtp = None
        mtp_id = getattr(config, "mtp_layer_id", None)
        if mtp_id is not None:
            self.mtp = Qwen3_5MTP(config, mtp_id)
            self.mtp._embed_ref = self.model.embed_tokens
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()

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
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen3_5MoEForCausalLM", "Qwen3_5MTP"]
