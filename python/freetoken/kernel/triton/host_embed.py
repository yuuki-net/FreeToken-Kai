"""UVA row gather for a host-resident embedding table (``--host-embedding``).

The table stays in pinned + mapped host memory and the GPU dereferences it in place over
PCIe (at its host VA on Linux/UVA, at the mapped device address on WDDM -- see
``kernel/pinned.device_ptr``). One program per requested row; the launch reads only device
tensors plus the raw table address, so it captures into CUDA graphs like any other kernel. A
decode step gathers one row (~4 KB), a 4096-token prefill chunk 16 MB -- a few milliseconds at
PCIe speed, against ~1 GB of VRAM for a 250k x 2048 fp16 table.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _host_gather_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    num_rows,
    EMB_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < EMB_DIM
    # the table is a host allocation: rebuild the typed pointer from the raw address
    base = table_ptr.to(tl.int64).to(tl.pointer_type(out_ptr.dtype.element_ty))
    values = tl.load(base + idx * EMB_DIM + offsets, mask=mask & in_range, other=0.0)
    tl.store(out_ptr + row * EMB_DIM + offsets, values, mask=mask)


def host_gather_rows(table_ptr: int, num_rows: int, embed_dim: int, row_ids: torch.Tensor,
                     out: torch.Tensor) -> torch.Tensor:
    """Gather ``row_ids`` (flat device int tensor) from the host table at ``table_ptr`` into
    ``out`` (``[n, embed_dim]``, contiguous, the table's dtype). Out-of-range ids store zeros."""
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        _host_gather_kernel[(n,)](
            table_ptr, row_ids, out, num_rows,
            EMB_DIM=embed_dim, BLOCK_D=triton.next_power_of_2(embed_dim), num_warps=4,
        )
    return out


__all__ = ["host_gather_rows"]
