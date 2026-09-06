from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.pipeline import RemoteLayer, layer_window, received_hidden
from freetoken.utils import nvtx_annotate

from .attention import GptOssAttention
from .moe import GptOssMLP

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class GptOssDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, moe_layer_offset: int = 0):
        self.self_attn = GptOssAttention(config, layer_id)
        # the offload cache indexes MoE layers rank-locally under the pipeline engine
        self.mlp = GptOssMLP(config, layer_id - moe_layer_offset)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class GptOssModel(BaseOP):
    """The decoder stack, or under the pipeline engine (``--pp-size``) this rank's slice of it:
    the embedding on the first rank, the final norm on the last; the other layers are
    ``RemoteLayer`` placeholders (see ``models/pipeline.py``)."""

    def __init__(self, config: ModelConfig):
        win = layer_window(config)
        self._window = win
        self.embed_tokens = (
            VocabParallelEmbedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
            if win.first
            else None
        )
        self.layers = OPList(
            [
                GptOssDecoderLayer(config, layer_id, moe_layer_offset=win.start)
                if win.owns(layer_id)
                else RemoteLayer()
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps) if win.last else None
        self._local_ids = win.local_ids

    @property
    def pp_first(self) -> bool:
        return self._window.first

    @property
    def pp_last(self) -> bool:
        return self._window.last

    def forward(self, input_ids: torch.Tensor, hidden_in: torch.Tensor | None = None) -> torch.Tensor:
        """``hidden_in`` is the residual stream ``[T, hidden]`` received from the previous
        pipeline rank (required on non-first ranks); a non-last rank returns the stream it hands
        on instead of the normed head input."""
        if self._window.first:
            x = self.embed_tokens.forward(input_ids)
        else:
            assert hidden_in is not None, "non-first pipeline rank needs the received residual stream"
            x = hidden_in
        residual: torch.Tensor | None = None
        layers = self.layers.op_list
        for i in self._local_ids:
            x, residual = layers[i].forward(x, residual)
        if not self._window.last:
            return x + residual  # the residual stream for the next pipeline rank
        return self.norm.forward(x, residual)[0]

    def prepare_for_runtime(self) -> None:
        layers = self.layers.op_list
        for i in self._local_ids:
            layers[i].mlp.prepare_for_runtime()


class GptOssForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = GptOssModel(config)
        # the residual stream crossing a pipeline boundary is [T, hidden] in the model dtype
        self.pp_hidden_width = config.hidden_size
        self._pp_hidden_dtype = torch.get_default_dtype()
        if not self.model.pp_last:
            self.lm_head = None
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
        self.config = config
        super().__init__()

    def forward(self) -> torch.Tensor:
        input_ids = get_global_ctx().batch.input_ids
        hidden_in = None
        if not self.model.pp_first:
            hidden_in = received_hidden(input_ids, self.pp_hidden_width, self._pp_hidden_dtype)
        out = self.model.forward(input_ids, hidden_in)
        if not self.model.pp_last:
            return out  # the residual stream for the next pipeline rank
        return self.lm_head.forward(out)

    def prepare_for_runtime(self) -> None:
        self.model.prepare_for_runtime()


__all__ = ["GptOssDecoderLayer", "GptOssForCausalLM", "GptOssModel"]
