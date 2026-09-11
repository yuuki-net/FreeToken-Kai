from __future__ import annotations

import ctypes
import functools
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def clear_cuda_error() -> None:
    """Drop the error a refused CUDA call left in this thread's error slot.

    Only ``cudaGetLastError`` resets that slot. ``cudaDeviceSynchronize`` -- what
    ``torch.cuda.synchronize()`` calls -- returns success for a non-sticky error and leaves it
    standing, so whatever touches CUDA next reports it instead, with the real culprit nowhere
    in the traceback. Two callers need this and both were found the hard way: a refused
    ``cudaHostRegister`` surfacing as "CUDA error: invalid argument" out of a 48 KB
    ``torch.full`` in ``OffloadMoeCache``, and a tile Triton refuses to load surfacing out of
    a tensor destructor as ``terminate called after throwing c10::AcceleratorError``.

    ``torch._C._cudart`` binds neither call, so go through libcudart (torch dlopens it
    RTLD_GLOBAL, so the symbol resolves by name) and fall back to spending the error on a
    throwaway launch, which clears the slot the same way -- by raising.
    """
    import torch

    if os.name == "posix":
        try:
            ctypes.CDLL(None).cudaGetLastError()
            return
        except (OSError, AttributeError):
            pass
    try:
        torch.empty(1, dtype=torch.int32, device="cuda").fill_(0)
    except RuntimeError:
        pass  # the launch check consumed it, which is the point


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
