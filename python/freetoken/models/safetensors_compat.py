"""Read a checkpoint through the host where safetensors cannot be trusted with the pin budget.

safetensors 0.8.0's ``safe_open(..., device="cuda")`` keeps host memory it took for the
transfer. Measured on gpt-oss-20b (13,123 MB in three shards, one handle per shard held open
the way :mod:`freetoken.models.loader` holds them): the process ends 2,492 MB above the same
read staged through the host, and closing every handle does not give it back -- 3,100 MB
resident against 607 MB.

It is a pool that is kept, not memory that is lost: reading the same checkpoint a second and a
third time in the same process adds nothing (3,100 MB flat), so that figure is a high-water mark
and not a rate. Nothing accumulates per tensor either -- the cost arrives when a shard is opened
and read, which is why the reporter's run died on the third shard's first tensor, at 9,146 MB,
with 4.7 GB of VRAM free (safetensors#858, FreeToken-Kai#2). A bounded pool is still fatal
against a 1 GiB cap, which is the only thing that decides whether a start comes up.

On a host with a large pin budget that is waste. Where the budget is around 1 GiB it is fatal:
the read dies as "CUDA error: out of memory" while the GPU is nearly empty, because what ran
out was page-locked host memory and not VRAM.

Reading on the host and copying each tensor across avoids it, and what that costs is not known:
the 13 GB runs it was timed against (8-9 s staged, 5-11 s direct) put whichever ran second ahead,
so the page cache decided the order and not the path. What the staged path adds is a host-memory
copy of the whole checkpoint -- host memory bandwidth -- and the machines that need it are the
ones least able to afford it (the report came from a 2019 laptop on DDR4-2667; this was measured
on a 13700K on DDR5-4800). So it is not the default. It turns on where this host's pin budget is
too small for a straight read to be safe, and ``FREETOKEN_SAFETENSORS_CPU_LOAD`` forces it either
way.

0.7.0 is reported not to do this (not checked here); either way transformers 5.16 requires
safetensors >= 0.8.0, so holding the older version back is not available to anything that also
imports transformers.
"""

from __future__ import annotations

import contextlib
import glob
import os

from freetoken.utils import init_logger

logger = init_logger(__name__)

# safetensors 0.8.0 retained 2,492 MB of a 13,123 MB read, so a fifth; a quarter of the
# checkpoint is the yardstick a budget is compared against. Deliberately not tuned finer than the
# one model behind it: the retained pool is a high-water mark rather than a rate, and what sets it
# is not known (it did not track shard size across the three shards measured), so this is a proxy
# for "the budget is small enough that a straight read might not fit" and not a law.
RETAINED_SHARE = 0.25


def _forced() -> bool | None:
    """True/False from FREETOKEN_SAFETENSORS_CPU_LOAD, or None when it is unset."""
    raw = os.environ.get("FREETOKEN_SAFETENSORS_CPU_LOAD", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def checkpoint_bytes(model_path: str) -> int:
    """Total size of the checkpoint's safetensors shards; 0 when there are none to size."""
    try:
        return sum(os.path.getsize(p) for p in glob.glob(os.path.join(model_path, "*.safetensors")))
    except OSError:
        return 0


def wanted(model_path: str, *, reserved: int = 0) -> bool:
    """Whether this host should stage the checkpoint through host memory."""
    forced = _forced()
    if forced is not None:
        return forced
    from freetoken.moe.pin_probe import budget

    pin = budget(reserved)
    if pin is None:  # nothing caps page-locking here, so there is no budget to exhaust
        return False
    total = checkpoint_bytes(model_path)
    return bool(total) and pin < total * RETAINED_SHARE


def reason(model_path: str, *, reserved: int = 0) -> str:
    from freetoken.moe.pin_probe import budget, source

    pin, total = budget(reserved), checkpoint_bytes(model_path)
    if _forced() is not None:
        return "FREETOKEN_SAFETENSORS_CPU_LOAD"
    return (f"pin budget {pin / 2**30:.2f} GiB ({source()}) against a {total / 2**30:.1f} GiB checkpoint, "
            f"of which safetensors 0.8.0 keeps about {RETAINED_SHARE:.0%}")


# ------------------------------------------------------------------ the staged handle

class _StagedSlice:
    """``get_slice`` on a host handle, landing each slice on the target device."""

    def __init__(self, inner, device):
        self._inner, self._device = inner, device

    def __getitem__(self, item):
        return self._inner[item].to(self._device)

    def __getattr__(self, name):  # get_shape and anything else safetensors grows
        return getattr(self._inner, name)


class _StagedHandle:
    """A ``safe_open`` handle opened on the host, handing tensors over as if it were not.

    Only the read is moved. Every consumer still gets a tensor on the device it asked for, so
    nothing downstream has to know which way the bytes came."""

    def __init__(self, inner, device):
        self._inner, self._device = inner, device

    def get_tensor(self, name):
        return self._inner.get_tensor(name).to(self._device)

    def get_slice(self, name):
        return _StagedSlice(self._inner.get_slice(name), self._device)

    def __getattr__(self, name):  # keys, metadata, offset_keys, ...
        return getattr(self._inner, name)

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)


def _is_cuda(device) -> bool:
    return str(device).startswith("cuda")


@contextlib.contextmanager
def host_staged_reads(enable: bool = True, *, why: str = ""):
    """While inside, ``safetensors.safe_open(device="cuda...")`` reads on the host and copies.

    Patches the module attribute rather than the call sites: there are twenty of them across
    the model loaders, and the ones that already ask for ``device="cpu"`` must not change."""
    if not enable:
        yield False
        return
    import safetensors

    original = safetensors.safe_open

    def staged(*args, **kwargs):
        device = kwargs.get("device")
        if device is None and len(args) >= 3:  # positional (file, framework, device)
            device = args[2]
        if not _is_cuda(device):
            return original(*args, **kwargs)
        if "device" in kwargs:
            kwargs = {**kwargs, "device": "cpu"}
        else:
            args = (*args[:2], "cpu", *args[3:])
        return _StagedHandle(original(*args, **kwargs), device)

    logger.info_rank0(f"reading the checkpoint through host memory instead of straight onto the GPU: {why}"
                      if why else "reading the checkpoint through host memory instead of straight onto the GPU")
    safetensors.safe_open = staged
    try:
        yield True
    finally:
        safetensors.safe_open = original
