from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool
from .kv_quant import KVQuantSpec


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.

    ``kv_quant`` (``--kv-cache-dtype``) swaps the 16-bit slab for a pair of slabs holding
    block-quantized codes and their fp16 scales (see ``kv_quant.py``). The geometry is
    unchanged except for the last axis, so paging, eviction and the layer remap above do not
    know it happened. ``dtype`` keeps reporting the COMPUTE dtype -- what ``store_kv``
    receives and what a backend sizes its scratch with -- while ``store_dtype`` reports what
    the buffer actually holds. Handing codes back as ``dtype`` is a mistake that surfaces far
    away, as an attention backend compiling its kernels against uint8.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        kv_quant: KVQuantSpec | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        self._kv_quant = kv_quant
        self._compute_dtype = dtype
        self._head_dim = head_dim
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._num_storage_layers = num_storage_layers
        self._local_kv_heads = local_kv_heads
        self._page_size = page_size
        self._device = device
        self._alloc(num_pages)

    # ---- allocation -------------------------------------------------------------------

    def _alloc(self, num_pages: int) -> None:
        shape = (
            2,
            self._num_storage_layers,
            num_pages,
            self._page_size,
            self._local_kv_heads,
        )
        if self._kv_quant is None:
            self._kv_buffer = torch.empty(
                (*shape, self._head_dim), device=self._device, dtype=self._compute_dtype
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._scale_buffer = None
            self._k_scale_buffer = None
            self._v_scale_buffer = None
            width = self._head_dim
        else:
            spec = self._kv_quant
            # Zeroed, not empty: a code slab read before it is written must dequantize to a
            # finite 0 (the attend kernels mask by position, but torch.empty's recycled bit
            # patterns would poison anything that ever slipped past a mask).
            self._kv_buffer = torch.zeros(
                (*shape, spec.code_bytes_per_row(self._head_dim)),
                device=self._device,
                dtype=torch.uint8,
            )
            self._scale_buffer = torch.zeros(
                (*shape, spec.blocks_per_row(self._head_dim)),
                device=self._device,
                dtype=torch.float16,
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._k_scale_buffer = self._scale_buffer[0]
            self._v_scale_buffer = self._scale_buffer[1]
            width = spec.code_bytes_per_row(self._head_dim)
        self._storage_shape = (num_pages * self._page_size, self._local_kv_heads, width)
        self._scale_shape = (
            num_pages * self._page_size,
            self._local_kv_heads,
            0 if self._kv_quant is None else self._kv_quant.blocks_per_row(self._head_dim),
        )

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        self._k_scale_buffer = None
        self._v_scale_buffer = None
        self._scale_buffer = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self._alloc(num_pages)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        total = int(buf.numel() * buf.element_size())
        if self._scale_buffer is not None:
            total += int(self._scale_buffer.numel() * self._scale_buffer.element_size())
        return total // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    # ---- access -----------------------------------------------------------------------

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def k_scales(self, index: int) -> torch.Tensor:
        """fp16 block scales for ``k_cache(index)``; only when ``kv_quant`` is on."""
        assert self._k_scale_buffer is not None, "pool is not quantized"
        return self._k_scale_buffer[self._dense(index)]

    def v_scales(self, index: int) -> torch.Tensor:
        assert self._v_scale_buffer is not None, "pool is not quantized"
        return self._v_scale_buffer[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        dense = self._dense(layer_id)
        if self._kv_quant is None:
            from freetoken.kernel import store_cache

            store_cache(
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                indices=out_loc,
                k=k,
                v=v,
            )
            return

        from freetoken.kernel.triton.kv_quant import quantize_store_kv

        heads, head_dim = self._local_kv_heads, self._head_dim
        quantize_store_kv(
            k.view(-1, heads, head_dim),
            v.view(-1, heads, head_dim),
            out_loc,
            self._k_buffer[dense].view(self._storage_shape),
            self._k_scale_buffer[dense].view(self._scale_shape),
            self._v_buffer[dense].view(self._storage_shape),
            self._v_scale_buffer[dense].view(self._scale_shape),
            self._kv_quant,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The COMPUTE dtype -- what store_kv is handed, not what the slab holds."""
        return self._compute_dtype

    @property
    def store_dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def kv_quant(self) -> KVQuantSpec | None:
        return self._kv_quant

    @property
    def num_layers(self) -> int:
        return self._num_layers
