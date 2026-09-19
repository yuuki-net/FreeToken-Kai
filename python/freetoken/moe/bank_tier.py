"""From ``--moe-bank-ram`` and friends to this rank's ``MappedTier`` (or None), before any bank loads.

Kept out of engine.py so it runs without a GPU: the engine hands in its config, the bound expert
method and its pipeline placement, and gets back the tier -- or an error that names what is wrong
with a packed checkpoint before a single weight has been read.
"""

from __future__ import annotations

import os

from . import bank_disk
from .bank_file import layout_from_specs
from .bank_pack import bank_path_for, checkpoint_identity, method_meta, read_pack_manifest
from .mapped_bank import MappedTier, _ranges


def _say_order_without_stats(path, layers, log, warn) -> None:
    """Without --moe-bank-stats a layer keeps the order the bank file holds, and only a layer the file
    lacks gets checkpoint order. Warning on every start was wrong once a histogram had been applied
    (the docs say to leave the flag off after that): seen on the 3060s, whose file held all 48 layers
    in the order of three sessions' histograms."""
    from .bank_file import BankFile

    present: set[int] = set()
    if os.path.exists(path):
        try:
            with BankFile.open(path, writable=False) as bf:
                present = set(bf.present_layers())
        except Exception:  # noqa: BLE001 -- unreadable or foreign: the start reports it properly later
            present = set()
    missing = [layer for layer in layers if layer not in present]
    if not missing:
        log(
            f"--moe-bank-ram without --moe-bank-stats: layers {_ranges(layers)} keep the order already in "
            f"{path} (the histograms it was last written or reordered with)"
        )
        return
    warn(
        f"--moe-bank-ram without --moe-bank-stats: layers {_ranges(missing)} are not in the bank file yet "
        "and will be written in checkpoint order -- an arbitrary resident slice. Collect a histogram with "
        "--moe-stats-out first, or pass one now."
    )


def build_tier(config, method, *, pp=None, log=print, warn=None):
    """``config``: the engine config (``model_path``, ``model_config``, ``full_model_config``,
    ``moe_bank_ram`` / ``_stats`` / ``_dir``, ``tp_info``). ``method``: the bound offload expert
    method, or None for a loader without one (GGUF). ``pp``: the pipeline placement, or None.
    """
    warn = warn or log
    packed = read_pack_manifest(config.model_path)
    if not config.moe_bank_ram:
        if packed is not None:
            raise ValueError(
                f"{config.model_path} was packed by `ft bank pack`: its routed experts are only in "
                f"{bank_path_for(config.model_path, packed, config.moe_bank_dir)}. Serve it with --moe-bank-ram"
            )
        return None
    mc = config.model_config
    full = getattr(config, "full_model_config", None) or mc
    fkd = int(getattr(mc, "first_k_dense_replace", 0) or 0)
    num_experts = int(mc.num_experts)
    total_layers = int(full.num_moe_layers)
    lo, hi = pp.bank_window(fkd) if pp is not None else (0, total_layers)
    layers = list(range(lo, hi))

    # --moe-bank-ram is a whole-host cap, but each rank maps its own layers and every rank of a
    # layer split lives on the same machine -- so the per-rank share is what the resident count
    # is solved against. Taking the flag per rank instead would silently double the RAM a
    # two-GPU run uses.
    ranks = max(1, int(config.tp_info.size))
    first_rank = int(getattr(config.tp_info, "rank", 0) or 0) == 0
    if str(config.moe_bank_ram).strip().lower() == "auto":
        # parse_args resolves it once for ft serve; a config built some other way gets the same
        # arithmetic here, per rank, which only agrees across ranks if they start quietly
        from . import disk_probe

        auto = disk_probe.auto_bank_ram(disk_probe.meminfo(), ranks)
        log(auto.reason())
        object.__setattr__(config, "moe_bank_ram", auto.as_flag())
    total = bank_disk.parse_size(config.moe_bank_ram)
    budget = total // ranks
    resident_specs = method is not None and any(s.resident for s in method.layout().values())
    cell_bytes = (
        bank_disk.cell_bytes_of_layout(method.layout()) if method is not None
        else bank_disk.cell_bytes_from_config(mc)
    )
    if not cell_bytes:
        if packed is not None:
            raise ValueError(f"--moe-bank-ram: the expert size of {config.model_path} is unknown")
        log(f"--moe-bank-ram {config.moe_bank_ram}: the per-expert size of this format is unknown; no split")
        return None
    bank_gib = cell_bytes * len(layers) * num_experts / 2**30
    hot = bank_disk.hot_per_layer_for_budget(len(layers), num_experts, cell_bytes, budget)
    if hot >= num_experts and packed is None:
        log(
            f"--moe-bank-ram {config.moe_bank_ram}: {budget / 2**30:.1f} GiB per rank "
            f"({ranks} ranks) covers this rank's {bank_gib:.1f} GiB of banks; no split"
        )
        return None
    hot = min(hot, num_experts)
    if hot == 0:
        raise ValueError(
            f"--moe-bank-ram leaves room for no resident experts "
            f"({cell_bytes / 2**20:.1f} MiB each x {len(layers)} layers). Raise the cap."
        )
    log(
        f"--moe-bank-ram {config.moe_bank_ram}: {budget / 2**30:.1f} GiB per rank ({ranks} ranks) "
        f"against {bank_gib:.1f} GiB of banks in layers {_ranges(layers)}; {hot}/{num_experts} experts resident"
    )
    if packed is not None and resident_specs:
        raise ValueError(
            f"{config.model_path} was packed for {packed['kind']} / {packed['kernel']} experts, and this "
            f"run binds {method.kind} / {method.kernel.name}, whose GPU-resident values the bank does not "
            f"hold. Serve it with --moe-strategy hybrid"
        )

    wanted = None
    if config.moe_bank_stats:
        freq = bank_disk.load_freq(config.moe_bank_stats, first_k_dense=fkd)
        covered = [l for l in layers if l in freq]
        if not covered:
            warn(
                f"--moe-bank-stats: the histograms cover none of layers {_ranges(layers)}; pass the "
                f"--moe-stats-out file of every rank"
            )
        elif len(covered) < len(layers):
            warn(f"--moe-bank-stats: no histogram for layers {_ranges(set(layers) - set(covered))}; they keep checkpoint order")
        wanted = bank_disk.plan_placement(layers, num_experts, hot, freq).order
    no_stats = wanted is None and packed is None

    layout, meta = None, {}
    if method is not None and not resident_specs:
        meta = method_meta(method)
        if packed is not None:
            meta["fingerprint"] = packed["fingerprint"]
        else:
            try:
                meta["fingerprint"], meta["source_stamp"] = checkpoint_identity(config.model_path, full, method.kind)
            except Exception as exc:  # noqa: BLE001 -- any family, any checkpoint layout
                # The checkpoint's expert tensors cannot be named for this family: the file still
                # works, it just cannot be told apart from another checkpoint of the same shapes,
                # and `ft bank pack` will refuse it.
                log(f"--moe-bank-ram: no expert fingerprint for this checkpoint ({exc})")
                meta["fingerprint"] = None
        layout = layout_from_specs(method.layout(), range(total_layers), num_experts, meta)
    elif resident_specs:
        log(
            f"--moe-bank-ram: {method.kind} / {method.kernel.name} keeps per-expert values on the GPU "
            f"that come from the checkpoint, so its expert tensors are read on every start"
        )

    path = bank_path_for(config.model_path, packed, config.moe_bank_dir)
    if not os.path.exists(path):
        # before anything creates the directory or the file: every rank, since any of them could
        from . import disk_probe

        why = disk_probe.refuses_new_bank(os.path.dirname(os.path.abspath(path)))
        if why:
            raise ValueError(why)
    if first_rank:
        # Before anything is written there or mapped from it: a bank on /mnt/c under WSL2, a
        # network share, tmpfs, a USB or SATA disk all start and serve, just at a fraction of the
        # speed, and nothing else says why. One file for every rank, so one rank says it.
        from . import disk_probe

        for line in disk_probe.storage_warnings(os.path.dirname(os.path.abspath(path))):
            warn(line)
    if no_stats:
        _say_order_without_stats(path, layers, log, warn)
    old = bank_disk.legacy_bank_files(os.path.dirname(path))
    if old:
        gib = sum(os.path.getsize(p) for p in old) / 2**30
        warn(
            f"--moe-bank-ram: {len(old)} per-rank bank files from an older build are no longer read "
            f"({gib:.1f} GiB): {', '.join(old)} -- delete them to get the space back"
        )
    return MappedTier(
        path, layers, all_layers=range(total_layers), num_experts=num_experts, hot_per_layer=hot,
        wanted=wanted, layout=layout, meta=meta, can_write=packed is None, log=log, warn=warn,
        readahead=getattr(config, "moe_bank_readahead", "off") or "off", report_readahead=first_rank,
        first_rank=first_rank,
    )
