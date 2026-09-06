from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float | None = None,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # Gemma scales embeddings by sqrt(hidden_size). The scale is materialized in
        # the weight dtype (bf16) to match HF, which downcasts the scalar. The GPU
        # scalar is built lazily (model __init__ runs on the meta device) and cached
        # so it is not reallocated inside a captured CUDA graph.
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import indexing

        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


class HostEmbedding(BaseOP):
    """An input embedding table that lives in pinned + mapped host memory; the GPU gathers the
    looked-up rows in place over PCIe (``kernel/triton/host_embed``), inside CUDA graphs like
    any other kernel. Frees the table's VRAM (1 GB for a 250k x 2048 fp16 vocabulary) for KV
    pages on small cards. The engine materializes the keys under ``host_resident_prefixes`` in
    pinned host memory (see ``_materialize_loaded_weight_state_dict``). TP=1, untied only."""

    host_resident = True

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        assert get_tp_info().size == 1, "host embedding is TP=1 only"
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.empty(num_embeddings, embedding_dim)
        self._table_ptr: int | None = None

    def _ptr(self) -> int:
        if self._table_ptr is None:
            from freetoken.kernel.pinned import device_ptr

            w = self.weight
            assert not w.is_cuda and w.is_pinned(), "host embedding table must be pinned host memory"
            self._table_ptr = device_ptr(w)
        return self._table_ptr

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.host_embed import host_gather_rows

        ids = x.reshape(-1)
        out = torch.empty(ids.numel(), self.embedding_dim, dtype=self.weight.dtype, device=x.device)
        host_gather_rows(self._ptr(), self.num_embeddings, self.embedding_dim, ids, out)
        return out.view(*x.shape, self.embedding_dim)


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill and not getattr(batch, "spec_all_rows", False):
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw head GEMM over the rows given (no batch bookkeeping; TP=1). The MTP draft head
        scores its own hidden states through the shared head with this."""
        module = self.tied_embedding or self
        return F.linear(x, module.weight, self.bias)


class Fp8VocabParallelEmbedding(VocabParallelEmbedding):
    """Embedding table held as fp8-e4m3 rows + a per-row fp32 scale (quantized at load, see
    ``--dense-quant fp8``); rows are gathered and dequantized per lookup. TP=1 only."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        assert self.tp_size == 1, "fp8 embedding is TP=1 only"
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim, dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty(self.num_embeddings_tp, dtype=torch.float32)
        # the model dtype: __init__ runs under the engine's torch_dtype(config.dtype)
        self._out_dtype = torch.get_default_dtype()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x.long()
        # gather on the byte view (index kernels for fp8 dtypes are not universal), then dequant
        rows = self.weight.view(torch.uint8)[idx].view(torch.float8_e4m3fn).to(self._out_dtype)
        return rows * self.weight_scale[idx].to(self._out_dtype)[:, None]


class Fp8ParallelLMHead(ParallelLMHead):
    """W8A16 lm_head: fp8-e4m3 weight + per-row fp32 scale (quantized at load). The full-vocab
    GEMV reads the whole head every decode step, so fp8 halves that traffic. TP=1, untied only."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim, tie_word_embeddings=False)
        assert self.tp_size == 1, "fp8 lm_head is TP=1 only"
        self.weight = torch.empty(num_embeddings, embedding_dim, dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty(num_embeddings, dtype=torch.float32)

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

        batch = get_global_ctx().batch
        if batch.is_prefill and not getattr(batch, "spec_all_rows", False):
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fp8_pertensor_linear(x, self.weight, self.weight_scale)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

        return fp8_pertensor_linear(x, self.weight, self.weight_scale)
