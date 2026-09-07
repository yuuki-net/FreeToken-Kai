"""Which experts are worth keeping in RAM, and the renumbering that makes that expressible.

On a 64GB host the NVFP4 expert banks (63 GiB for Flash-Next) do not fit beside the runtime.
The fix is to keep the frequently routed experts resident and let the rest live on disk; this
module decides which are which, and ``mapped_bank.py`` carries the placement out.

The decision is a per-layer renumbering rather than a lookup table. ``moe_vec.cuh`` reaches
an expert row as ``expert * nrows * (ncols / qk)`` and the CPU executor repeats that
arithmetic, so making residency a property to be looked up per access would put a branch in
the hot loop of two compiled kernels. Instead each layer sorts its experts by measured
routing frequency, so "resident" becomes "physical row < hot_per_layer" -- a contiguous
prefix that can be locked down in one call. The only place that has to translate is the
router's top-k output, one gather per layer per step.

Nothing here needs a GPU or the kernel extensions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------------------
# measured routing
# ---------------------------------------------------------------------------------------
def load_freq(paths) -> dict[int, list[int]]:
    """{global layer id: [count per expert]} from one or more ``--moe-stats-out`` files.

    Counts for the same layer ADD. Files from one run are disjoint by layer (one per pipeline
    rank), so this only matters across runs -- which is the case that needs it. A placement
    fitted on a single session generalizes to that session and little else: measured, an
    ordering taken from one conversation served no more of a different workload's routing
    than a random ordering did (docs/bank-ram.md). Pooling sessions of different work is how you
    widen it, and overwriting would silently keep whichever file was listed last.
    """
    freq: dict[int, list[int]] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        rows = d.get("decode_freq")
        if not rows:
            raise ValueError(
                f"{p}: no decode_freq. Collect with --moe-stats-out and --disable-cuda-graph "
                "(under a captured graph the histogram counts the warmup routing instead)."
            )
        start = (d.get("layer_range") or [0])[0]
        for i, row in enumerate(rows):
            layer = start + i
            have = freq.get(layer)
            if have is None:
                freq[layer] = list(row)
            elif len(have) != len(row):
                raise ValueError(
                    f"{p}: layer {layer} has {len(row)} experts, an earlier file had {len(have)}"
                )
            else:
                freq[layer] = [a + b for a, b in zip(have, row)]
    return freq


# ---------------------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------------------
@dataclass
class BankPlacement:
    """Per-layer expert renumbering: physical rows [0, hot) resident, [hot, E) on disk."""

    num_experts: int
    hot_per_layer: int
    # layer -> logical expert ids, hottest first; the index in this list is the physical row
    order: dict[int, list[int]]

    @property
    def cold_per_layer(self) -> int:
        return self.num_experts - self.hot_per_layer

    def to_physical(self, layer: int) -> list[int]:
        """``logical -> physical`` for one layer: what the router's top-k is mapped through."""
        out = [0] * self.num_experts
        for physical, logical in enumerate(self.order[layer]):
            out[logical] = physical
        return out


def plan_placement(
    layers, num_experts: int, hot_per_layer: int, freq: dict[int, list[int]] | None = None
) -> BankPlacement:
    """Sort each layer's experts by measured frequency, hottest first.

    Ties (and the whole ordering when ``freq`` is missing) fall back to logical id, so the
    same histogram always yields the same file. A restart that reordered experts would leave
    the previous run's bank file describing a placement that no longer holds, with every
    shape still matching.
    """
    layers = list(layers)
    hot_per_layer = max(0, min(hot_per_layer, num_experts))
    order = {}
    for layer in layers:
        counts = (freq or {}).get(layer)
        if counts is not None and len(counts) != num_experts:
            raise ValueError(
                f"layer {layer}: histogram has {len(counts)} experts, model has {num_experts}"
            )
        key = (lambda e: (-counts[e], e)) if counts is not None else (lambda e: e)
        order[layer] = sorted(range(num_experts), key=key)
    return BankPlacement(num_experts=num_experts, hot_per_layer=hot_per_layer, order=order)


def hot_per_layer_for_budget(
    num_layers: int, num_experts: int, cell_bytes: int, ram_budget_bytes: int
) -> int:
    """Largest uniform per-layer resident count that fits ``ram_budget_bytes``.

    Uniform across layers on purpose. A global budget places marginally better -- measured,
    0.12% of accesses reaching disk versus 0.33% -- but gives layers different resident
    counts, and the locked prefix has to be one contiguous range per block.
    """
    if num_layers <= 0 or cell_bytes <= 0:
        return num_experts
    per_layer_budget = max(0, ram_budget_bytes) // num_layers
    return max(0, min(num_experts, per_layer_budget // cell_bytes))


# ---------------------------------------------------------------------------------------
# the router side
# ---------------------------------------------------------------------------------------
def permutation_tensor(placement: BankPlacement, layer: int, device=None, dtype=None):
    """``logical -> physical`` for one layer, as a device tensor for the router remap."""
    import torch

    return torch.as_tensor(
        placement.to_physical(layer), dtype=dtype or torch.int32, device=device
    )


def apply_permutation(topk_ids, perm) -> None:
    """Rewrite a router's top-k expert ids into physical (post-renumbering) ids, in place.

    Called once per MoE layer per step, on a [tokens, top_k] tensor -- ten ids for a decode
    step. Static shapes and no host sync, so it captures into the decode graph like
    everything else on that path; the int64 cast is torch's index dtype and becomes a static
    buffer under capture.

    In place because ``topk_ids`` is already contracted to be mutable (the decode path
    rewrites it into cache slot ids straight after), so an out-of-place remap would only add
    an allocation to the same buffer's lifetime.
    """
    if perm is None:
        return
    flat = topk_ids.reshape(-1)
    flat.copy_(perm.index_select(0, flat.long()).to(topk_ids.dtype))


# ---------------------------------------------------------------------------------------
# sizing, from the config alone
# ---------------------------------------------------------------------------------------
def parse_size(text) -> int | None:
    """``"50G"`` / ``"50GiB"`` / ``"51200M"`` / ``"0.5T"`` -> bytes. None passes through.

    Binary units throughout: the numbers this is compared against (bank bytes, MemTotal) are
    all binary, and mixing the two silently loses 7% at the terabyte end.
    """
    if text is None or text == "":
        return None
    s = str(text).strip().upper().replace("IB", "").replace("B", "")
    mult = 1
    for suffix, m in (("K", 2**10), ("M", 2**20), ("G", 2**30), ("T", 2**40)):
        if s.endswith(suffix):
            s, mult = s[: -len(suffix)], m
            break
    try:
        return int(float(s) * mult)
    except ValueError as exc:
        raise ValueError(f"could not parse size {text!r} (expected e.g. 50G)") from exc


def cell_bytes_from_config(model_config) -> int | None:
    """Bytes one expert occupies across every sub-bank, from the config alone.

    Needed before any layer is loaded, because the resident count has to be decided before
    the first layer reaches the sink. Mirrors ``_BANK_BYTES_PER_EXPERT``, the same table
    ``bank_bytes_estimate`` uses for the pin-budget decisions.
    """
    from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

    quant = getattr(model_config, "expert_quant", "none")
    fmt = quant if quant != "none" else (
        getattr(model_config, "moe_weight_format", None) or "bf16"
    )
    per_expert = _BANK_BYTES_PER_EXPERT.get(fmt)
    hidden = getattr(model_config, "hidden_size", None)
    inter = getattr(model_config, "moe_intermediate_size", None)
    if per_expert is None or not (hidden and inter):
        return None
    return per_expert(hidden, inter)


def plan_from_config(model_config, ram_budget_bytes: int, layers, stats_paths=None):
    """``(placement, cell_bytes)`` for a --moe-bank-ram run, before anything is loaded.

    ``placement`` is None when the budget already covers every expert (nothing to do) or when
    the format's per-expert size is unknown -- the feature declines rather than guessing how a
    checkpoint splits.
    """
    cell_bytes = cell_bytes_from_config(model_config)
    layers = list(layers)
    if not cell_bytes or not layers:
        return None, cell_bytes
    num_experts = int(model_config.num_experts)
    hot = hot_per_layer_for_budget(len(layers), num_experts, cell_bytes, ram_budget_bytes)
    if hot >= num_experts:
        return None, cell_bytes
    if hot == 0:
        raise ValueError(
            f"--moe-bank-ram leaves room for no resident experts "
            f"({cell_bytes / 2**20:.1f} MiB each x {len(layers)} layers). Raise the cap."
        )
    freq = load_freq(stats_paths) if stats_paths else None
    return plan_placement(layers, num_experts, hot, freq), cell_bytes


def bank_file_path(model_path: str, rank: int, size: int, directory=None) -> str:
    """Where this rank's mapped bank lives.

    A cache directory by default, not the checkpoint: the checkpoint may be read-only or
    shared, and this file is derived data that a different --moe-bank-ram invalidates. One
    file per rank, since each rank's placement covers only its own layers.
    """
    if directory is None:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        directory = os.path.join(
            base, "freetoken", "bankmap", os.path.basename(os.path.normpath(model_path))
        )
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"bank.rank{rank}of{size}.ftmb")
