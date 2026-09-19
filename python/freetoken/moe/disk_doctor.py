"""``ft doctor disk``: is ``--moe-bank-ram`` usable on this host, and what should be set.

Answers, without a GPU and without root, the questions that otherwise took a session each on
the machines docs/bank-ram.md was measured on: what the bank file sits on (filesystem,
transport, PCIe link, which side of the chipset), whether the device readahead is the one this
model's block geometry wants, how much RAM there is to give and how much of it may be locked,
how fast the disk reads expert rows the way decode does, and -- from all of that plus a routing
histogram if there is one -- roughly what each RAM cap would cost per token.

The prediction is labelled as what it is. The cost model behind it (step = base + disk reads
of the non-resident routes) came within about 2x of what was then measured, and only after two
terms nobody had modelled were found by measuring (docs/bank-ram.md, "What this looked like
while it was wrong"). Every assumption it rests on is printed under the table.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass, field

from freetoken.moe import disk_probe as dp
from freetoken.moe import gpu_probe as gp

GiB = dp.GiB
MiB = 2**20

# Effective share of the static miss that still reaches the disk once the page cache has settled.
# Measured once: Flash-Next, 64 GB, 2360 tokens -- 3% of routes against the placement's 12.5%.
# In that run the page cache left beside 15.4 GiB of non-resident rows came to 0.26 of them by
# this module's own arithmetic (MemAvailable 61 GiB - 48 cap - 9 rest of the server = 4 GiB), so
# that is the share the factor belongs to. Below it the factor is interpolated linearly to 1 at no page cache
# at all -- an assumption, not a second measurement.
PAGE_CACHE_FACTOR = 0.24
PAGE_CACHE_REF_SHARE = 0.26

# Decode step with the banks fully in RAM and one token per step (--spec-mtp 0), where it is
# known. Hardware-specific: printed with where it came from, and --base-step-ms replaces it.
# Flash-Next: 19.33 tok/s (median of two runs, 2026-09-18); it was 59.5 ms (16.8 tok/s) when the
# disk term was fitted, before the prefill, cache and copy work of 2026-09-13..18.
KNOWN_BASE_STEP_MS = {
    "flash-next": (51.7, "2x RTX 3060, --pp-size 2, --spec-mtp 0, banks fully in RAM"),
}


@dataclass
class Shape:
    name: str | None = None
    moe_layers: int | None = None  # the whole model's
    num_experts: int | None = None
    top_k: int | None = None
    cell_bytes: int | None = None
    widest_row_bytes: int | None = None
    sources: list[str] = field(default_factory=list)

    @property
    def bank_bytes(self) -> int | None:
        if self.moe_layers and self.num_experts and self.cell_bytes:
            return self.moe_layers * self.num_experts * self.cell_bytes
        return None


# ---------------------------------------------------------------------------------------
# what model
# ---------------------------------------------------------------------------------------


def find_bank_file(model_path: str | None, bank_dir: str | None) -> tuple[str | None, str]:
    """``(path of bank.ftmb, how it was chosen)`` -- the same resolution ``ft serve`` uses
    (moe/bank_pack.bank_path_for): ``--moe-bank-dir``, else the file a packed checkpoint names
    (normally inside it), else ``~/.cache/freetoken/bankmap/<model>``. The path need not exist."""
    from freetoken.moe.bank_file import BANK_FILE_NAME
    from freetoken.moe.bank_pack import PackError, bank_path_for, bank_root, read_pack_manifest

    if bank_dir:
        return os.path.join(os.path.expanduser(bank_dir), BANK_FILE_NAME), "--moe-bank-dir"
    if model_path:
        try:
            packed = read_pack_manifest(model_path)
        except (PackError, OSError, ValueError):
            packed = None
        how = "named by the packed checkpoint" if packed is not None else "default for --model"
        return bank_path_for(model_path, packed), how
    found = sorted(glob.glob(os.path.join(glob.escape(bank_root()), "*", BANK_FILE_NAME)))
    if len(found) != 1:
        return None, (
            f"{len(found)} models have bank files under {bank_root()}; pass --model or --moe-bank-dir"
            if found else f"no bank file under {bank_root()} and no --model"
        )
    return found[0], "the only bank file under " + bank_root()


@dataclass
class BankState:
    path: str
    layers: list[int] = field(default_factory=list)  # committed, of the file's
    all_layers: int = 0
    canonical_for: str | None = None
    meta: dict = field(default_factory=dict)
    error: str | None = None


def shape_from_bank(path: str, shape: Shape) -> BankState | None:
    """The file's geometry (every MoE layer of the model, whichever ranks wrote them) and which
    layers it has committed. None when there is no file."""
    from freetoken.moe.bank_file import BankFile, MappedBankLayout

    if not os.path.isfile(path):
        return None
    state = BankState(path)
    try:
        lay = MappedBankLayout.read(path)
        with BankFile.open(path, writable=False) as f:
            manifest = f.manifest()
            state.layers = f.present_layers(manifest)
            state.canonical_for = f.canonical_for(manifest)
    except Exception as exc:  # noqa: BLE001 -- a torn or foreign file is a finding, not a crash
        state.error = str(exc)
        return state
    state.all_layers, state.meta = len(lay.layers), dict(lay.meta)
    shape.moe_layers = len(lay.layers)
    shape.num_experts = lay.num_experts
    shape.cell_bytes = sum(rb for _, _, _, rb in lay.banks)
    shape.widest_row_bytes = max(rb for _, _, _, rb in lay.banks)
    shape.sources.append("bank file header")
    return state


def shape_from_model(model_path: str, shape: Shape) -> None:
    """Fill what the bank headers did not from the checkpoint: the parsed model config where it
    parses, the raw config.json where it does not."""
    shape.name = os.path.basename(os.path.normpath(model_path))
    try:
        from dataclasses import replace

        from freetoken.engine.config import checkpoint_quant_config
        from freetoken.layers.quantization import set_quant_config
        from freetoken.models.register import _load_attr, get_model_spec
        from freetoken.moe.bank_disk import cell_bytes_from_config, widest_row_block_bytes_from_config
        from freetoken.utils import cached_load_hf_config

        hf = cached_load_hf_config(model_path)
        spec = get_model_spec(hf.architectures[0])
        quant = checkpoint_quant_config(model_path, hf, spec)
        set_quant_config(quant)
        cfg = replace(_load_attr(spec.module, spec.parse_config)(hf), quant=quant)
        shape.moe_layers = shape.moe_layers or int(cfg.num_moe_layers)
        shape.num_experts = shape.num_experts or int(cfg.num_experts)
        shape.top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0) or None
        shape.cell_bytes = shape.cell_bytes or cell_bytes_from_config(cfg)
        shape.widest_row_bytes = shape.widest_row_bytes or widest_row_block_bytes_from_config(cfg)
        shape.sources.append("model config")
        return
    except Exception as exc:  # an unsupported or partial checkpoint still has a config.json
        why = f"{type(exc).__name__}: {exc}"
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        shape.sources.append(f"{model_path}: no readable config ({exc})")
        return
    t = raw.get("text_config") or raw
    experts = t.get("num_experts") or t.get("num_local_experts") or t.get("n_routed_experts")
    shape.num_experts = shape.num_experts or experts
    shape.top_k = t.get("num_experts_per_tok")
    if t.get("num_hidden_layers"):
        shape.moe_layers = shape.moe_layers or int(t["num_hidden_layers"]) - int(t.get("first_k_dense_replace") or 0)
    shape.sources.append(f"raw config.json (the model config did not parse: {why[:120]})")


# ---------------------------------------------------------------------------------------
# the prediction
# ---------------------------------------------------------------------------------------


def coverage(fit: dict | None, evaluate: dict | None, moe_layers: int, num_experts: int, hot: int) -> float | None:
    """Share of routes that land on resident rows when each layer keeps its ``hot`` most frequent
    experts by ``fit``, counted on ``evaluate`` (``fit`` itself when None). None without data."""
    from freetoken.moe.bank_disk import plan_placement

    counts = evaluate if evaluate is not None else fit
    if not counts:
        return None
    layers = sorted(counts)
    order = plan_placement(layers, num_experts, hot, fit).order
    hits = total = 0
    for layer in layers:
        row = counts[layer]
        total += sum(row)
        hits += sum(row[e] for e in order[layer][:hot])
    return hits / total if total else None


def page_cache_factor(share: float, factor: float = PAGE_CACHE_FACTOR) -> float:
    """How much of the static miss still reaches the disk, given page cache worth ``share`` of the
    non-resident rows. ``factor`` at the measured share and above, linear to 1 at none."""
    return 1.0 - (1.0 - factor) * min(1.0, max(0.0, share) / PAGE_CACHE_REF_SHARE)


@dataclass
class Row:
    cap: int
    hot: int
    residency: float
    coverage: float | None
    coverage_kind: str
    cold_bytes: int
    page_cache: int
    disk_bytes: float | None
    disk_ms: float | None
    step_ms: float | None


def predict(shape: Shape, caps: list[int], ranks: int, mem_available: int, *, fit=None, evaluate=None,
            disk_gbs: float | None = None, base_step_ms: float | None = None,
            factor: float = PAGE_CACHE_FACTOR) -> list[Row]:
    from freetoken.moe.bank_disk import hot_per_layer_for_budget

    L, E, cell = shape.moe_layers, shape.num_experts, shape.cell_bytes
    bank = shape.bank_bytes
    rows = []
    for cap in caps:
        hot = hot_per_layer_for_budget(L, E, cell, cap)
        r = hot / E
        if fit or evaluate:
            cov = coverage(fit, evaluate, L, E, hot)
            kind = "held-out" if evaluate else "in-sample"
        else:
            cov, kind = None, "none"
        static_miss = (1.0 - cov) if cov is not None else (1.0 - r)
        if cov is None:
            kind = "arbitrary"  # no histogram: the server's split ignores routing, so coverage ~ residency
        cold = bank - hot * L * cell
        page = max(0, mem_available - cap - dp.NONBANK_PER_RANK_BYTES * ranks)
        share = page / cold if cold > 0 else 1.0
        disk = disk_ms = step = None
        if shape.top_k:
            disk = L * shape.top_k * static_miss * page_cache_factor(share, factor) * cell if cold > 0 else 0.0
            if disk_gbs:
                disk_ms = disk / (disk_gbs * 1e9) * 1000.0
                if base_step_ms:
                    step = base_step_ms + disk_ms
        rows.append(Row(cap, hot, r, cov if cov is not None else r, kind, max(cold, 0), page, disk, disk_ms, step))
    return rows


def default_caps(bank: int, mem_total: int, auto: int | None) -> list[int]:
    caps = {int(bank * f) for f in (0.5, 0.65, 0.77, 0.9)} | {bank}
    if auto:
        caps.add(auto)
    return sorted(c for c in caps if 0 < c < mem_total)


# ---------------------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------------------


def _gib(n: float) -> str:
    return f"{n / GiB:.1f} GiB"


def _size(n: int | None) -> str:
    if n is None:
        return "unlimited"
    return f"{n / GiB:.2f} GiB" if n >= GiB else f"{n / MiB:.0f} MiB"


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.findings: list[tuple[str, str]] = []  # ("bad"|"warn"|"ok", text)

    def section(self, title: str) -> None:
        self.lines += ["", title]

    def say(self, text: str) -> None:
        self.lines.append("  " + text)

    def find(self, level: str, text: str) -> None:
        self.findings.append((level, text))

    def render(self) -> str:
        out = list(self.lines)
        out += ["", "Findings"]
        if not self.findings:
            out.append("  nothing known to be wrong (which is not the same as measured to be right)")
        mark = {"bad": "!!", "warn": " !", "ok": "  "}
        for level, text in sorted(self.findings, key=lambda f: ("bad", "warn", "ok").index(f[0])):
            out.append(f"  {mark[level]} {text}")
        return "\n".join(out).lstrip("\n")


def _wsl_host(rep: Report, label: str, s: dp.Storage, proc: str, need: int, host=None) -> None:
    """Under WSL2: the Windows drive under the virtual disk, which `df` inside cannot show."""
    host = host if host is not None else dp.wsl_host_disk(proc, attributes=True)
    if host is None:
        rep.say("WSL2: this is a file (ext4.vhdx) on a Windows drive; which one could not be read from the "
                "registry (Windows interop off?). `df /` here is the virtual disk's size, not that drive's free space")
        rep.find("warn", "WSL2: the free space of the Windows drive under the virtual disk is unknown; check it "
                         "before writing a bank file")
        return
    size = f", {host.vhdx_bytes / GiB:.1f} GiB" if host.vhdx_bytes else ""
    sparse = {True: "sparse", False: "not sparse", None: "sparse or not unknown"}[host.sparse]
    rep.say(f"WSL2: the virtual disk is {host.vhdx}{size}, {sparse}")
    if host.free_bytes is None:
        rep.say(f"  {host.drive} is not mounted inside WSL, so its free space is unknown")
    else:
        rep.say(f"  {host.drive} has {host.free_bytes / GiB:.1f} GiB free of {host.total_bytes / GiB:.0f} GiB -- "
                f"that, not `df /` ({s.mount.mountpoint} shows the virtual disk's own size), is what a bank "
                f"write grows into")
    if host.sparse is not True:
        rep.say("  space freed inside the distro is not returned to Windows (the vhdx only grows)")
    if host.free_bytes is not None and need + dp.HOST_FREE_FLOOR_BYTES > host.free_bytes:
        what = f"the {need / GiB:.1f} GiB the bank file still needs" if need else "anything large"
        rep.find("bad" if need > host.free_bytes else "warn",
                 f"{host.drive} has {host.free_bytes / GiB:.1f} GiB free, under {what} plus a "
                 f"{dp.HOST_FREE_FLOOR_BYTES / GiB:.0f} GiB margin: when the Windows drive fills, WSL stops "
                 f"mid-write")


def _storage(rep: Report, label: str, path: str, proc: str, sys: str, gpus, need: int = 0) -> dp.Storage:
    s = dp.probe_storage(path, proc, sys)
    rep.section(f"{label}: {s.path}")
    if s.mount is None:
        rep.say("filesystem: not found in /proc/self/mountinfo")
        return s
    level, why = s.fs_verdict
    rep.say(f"filesystem: {s.mount.fstype} from {s.mount.source}, mounted at {s.mount.mountpoint}")
    if level != "ok":
        rep.find(level, f"{label.lower()} is on {why}")
        if level == "bad":
            return s
    d = s.device
    if d is None:
        rep.say(f"block device: none found for {s.dev}")
        return s
    chain = " -> ".join(d.stacked_on + ([d.partition] if d.partition else []) + [d.name])
    desc = {"nvme": "NVMe", "sata": "SATA", "usb": "USB", "mmc": "SD/eMMC", "scsi": "SCSI/SAS",
            "virtio": "virtio virtual disk", "hyperv": "Hyper-V virtual disk", "xen": "Xen virtual disk",
            "loop": "loop device", "ram": "RAM disk"}.get(d.transport, "unknown transport")
    rep.say(f"device: {chain}, {desc}" + (f" ({d.model})" if d.model else ""))
    if d.backing_file:
        rep.say(f"loop device backed by {d.backing_file}: the drive that matters is the one holding that file")
    if d.virtual:
        rep.say("rotational flag and transport of the drive behind a virtual disk are not visible from here")
        if s.wsl:
            _wsl_host(rep, label, s, proc, need)
            rep.say("  Get-PhysicalDisk | Select-Object FriendlyName,BusType,MediaType   (PowerShell: which drive "
                    "type that is)")
            rep.say("PCIe link and chipset placement cannot be seen from inside WSL2 either")
    else:
        rep.say(f"rotational: {'yes' if d.rotational else 'no' if d.rotational is not None else 'unknown'}")
        if d.transport == "usb":
            rep.find("bad", f"{label.lower()} is on a USB disk ({d.name})")
        elif d.rotational:
            rep.find("bad", f"{label.lower()} is on a rotating disk ({d.name}): random row reads cost seconds per token")
        elif d.transport == "sata":
            rep.find("warn", f"{label.lower()} is on SATA ({d.name}): measured, a SATA SSD adds ~300 ms per decode "
                             f"step even at 64 GB")
    if d.link:
        rep.say(f"link: {d.link.describe()} at {d.link.bdf}")
        gen, top = dp.PciLink.gen(d.link.speed), dp.PciLink.gen(d.link.max_speed)
        if gen is not None and gen <= 3:
            slot = f" (the drive supports Gen{top}: the slot is what limits it)" if top and top > gen else ""
            rep.find("warn", f"{label.lower()}: the NVMe link runs at Gen{gen}{slot}; measured, a Gen3 drive "
                             f"multiplies the disk part of a step by ~1.6 against Gen4")
        kind = dp.upstream_kind(d.pci_chain)
        rep.say(f"upstream (estimated from the PCI topology): {kind}")
        for gpu_bdf, gchain in gpus:
            common = dp.shared_upstream(d.pci_chain, gchain)
            gkind = dp.upstream_kind(gchain)
            if common or (kind == "chipset" and gkind == "chipset"):
                via = f"bridge {common}" if common else "the chipset uplink"
                rep.say(f"shares {via} with GPU {gpu_bdf}")
                rep.find("warn", f"{label.lower()} shares {via} with GPU {gpu_bdf}: under --pp-size 2 the bank "
                                 f"reads and that GPU's traffic compete for one link; a CPU-attached M.2 avoids it")
    return s


def _gpus(rep: Report, ns, shape: Shape, ranks: int, query, rate) -> float | None:
    """Each GPU's PCIe link, idle and under a copy, and the pinned host -> GPU rate.

    A prefill chunk streams every layer's whole expert bank to its rank's GPU, so the link is
    paid once per chunk; decode only pays it for cache misses.
    """
    rep.section("GPUs")
    links = query() if query is not None else None
    if links is None:
        rep.say("not read: nvidia-smi is missing or failed")
        return None
    if not links:
        rep.say("nvidia-smi lists no GPU")
        return None
    rates = []
    per_rank = shape.bank_bytes / ranks if shape.bank_bytes else None
    for link in links:
        rep.say(f"GPU {link.index} ({link.name}, {link.bus_id}): {link.describe()} as reported now")
        measured, under = None, None
        if ns.h2d_seconds > 0:
            try:
                measured, under = rate(link, ns.h2d_seconds)
            except Exception as exc:  # noqa: BLE001 -- no CUDA, a full or busy GPU: say so, go on
                rep.say(f"  pinned host -> GPU rate not measured: {exc}")
        else:
            rep.say("  rate not measured (--h2d-seconds 0); a GPU at idle may report a lower generation "
                    "than it trains to under load")
        if under is not None:
            rep.say(f"  during a copy: {under.describe()}")
        eff = under or link
        if measured:
            rates.append(measured)
            nominal = eff.lane_gbs
            rep.say(f"  pinned host -> GPU: {measured:.1f} GB/s"
                    + (f" ({measured / nominal:.0%} of the link's nominal {nominal:.1f} GB/s)" if nominal else ""))
            if per_rank:
                rep.say(f"  a rank's expert banks ({_gib(per_rank)}) cross it in {per_rank / 1e9 / measured:.1f} s: "
                        f"once per prefill chunk, whatever the chunk holds")
        if under is None and measured is None:
            continue
        top_gen = min(g for g in (eff.gen_max, eff.gen_host) if g) if (eff.gen_max or eff.gen_host) else None
        if eff.width and eff.width_max and eff.width < eff.width_max:
            rep.find("warn", f"GPU {link.index} runs at x{eff.width} of its x{eff.width_max} under load: the slot (or a "
                             f"shared chipset uplink) halves or quarters what every prefill chunk streams to it")
        elif eff.gen and top_gen and eff.gen < top_gen:
            rep.find("warn", f"GPU {link.index} trained to Gen{eff.gen} under load where GPU and slot allow Gen{top_gen}")
    return min(rates) if rates else None


def run(ns, proc: str = "/proc", sys: str = "/sys", *, gpu_query=None, gpu_rate=None) -> str:
    from freetoken.moe.bank_disk import load_freq

    rep = Report()
    rep.lines.append("ft doctor disk -- is --moe-bank-ram usable on this host, and what to set")

    # ---- model
    from freetoken.moe.bank_disk import legacy_bank_files

    bank_path, how = find_bank_file(ns.model, ns.moe_bank_dir)
    bank_dir = os.path.dirname(bank_path) if bank_path else None
    shape = Shape(name=os.path.basename(bank_dir) if bank_dir and not ns.moe_bank_dir else None)
    bank = shape_from_bank(bank_path, shape) if bank_path else None
    if ns.model and os.path.isdir(ns.model):
        shape_from_model(ns.model, shape)
    elif ns.model:
        # a hub id would be downloaded by the config loader; this command reads local disks only
        shape.sources.append(f"--model {ns.model} is not a local directory")
    # one file holds every layer, whichever ranks serve them: nothing on disk says how many
    ranks = ns.pp_size or 1
    rep.section("Model")
    if shape.cell_bytes and shape.num_experts:
        rep.say(
            f"{shape.name or '?'}: {shape.moe_layers or '?'} MoE layers x {shape.num_experts} experts, "
            f"top-{shape.top_k or '?'}, {shape.cell_bytes / MiB:.2f} MiB per expert"
            + (f", widest row block {shape.widest_row_bytes // 1024} kB" if shape.widest_row_bytes else "")
        )
        if shape.bank_bytes:
            rep.say(f"expert banks: {_gib(shape.bank_bytes)} across {ranks} rank{'s' if ranks > 1 else ''}")
        rep.say("from: " + "; ".join(shape.sources))
    else:
        rep.say("unknown: " + ("; ".join(shape.sources) or how) + " -- pass --model for the readahead "
                "recommendation and the prediction")
    if bank_path:
        if bank is None:
            rep.say(f"bank file ({how}): {bank_path} (not written yet)")
        elif bank.error:
            rep.say(f"bank file ({how}): {bank_path} -- unreadable: {bank.error}")
            rep.find("warn", f"the bank file {bank_path} cannot be read ({bank.error}); ft serve will start it over "
                             f"unless it is the only copy of a packed checkpoint")
        else:
            from freetoken.moe.mapped_bank import _ranges

            held = f"layers {_ranges(bank.layers)}" if bank.layers else "no layers"
            rep.say(f"bank file ({how}): {bank_path}, holds {held} of {bank.all_layers}"
                    + (f", {bank.meta.get('kind')} / {bank.meta.get('kernel')}" if bank.meta.get("kind") else ""))
            if bank.canonical_for:
                rep.say(f"it is the only copy of the experts of {bank.canonical_for} (ft bank pack): back it up")
        old = legacy_bank_files(bank_dir) if bank_dir and os.path.isdir(bank_dir) else []
        if old:
            gib = sum(os.path.getsize(p) for p in old) / GiB
            rep.find("warn", f"{len(old)} per-rank bank files from an older build are no longer read ({gib:.1f} GiB): "
                             f"{', '.join(old)}")
    else:
        rep.say(how)

    # ---- storage
    gpus = dp.nvidia_gpus(sys)
    target = bank_dir or (ns.model or os.path.expanduser("~"))
    # what a start would still write into the bank file: all of it without one, the missing layers with one
    if bank is not None and not bank.error and bank.all_layers and shape.bank_bytes:
        need = shape.bank_bytes * (bank.all_layers - len(bank.layers)) // bank.all_layers
    else:
        need = shape.bank_bytes or 0
    s = _storage(rep, "Bank storage", target, proc, sys, gpus, need)
    if ns.model and os.path.exists(ns.model):
        m = dp.probe_storage(ns.model, proc, sys)
        if not (m.mount and s.mount and m.mount.mountpoint == s.mount.mountpoint and m.dev == s.dev):
            _storage(rep, "Checkpoint storage", ns.model, proc, sys, gpus)
            rep.say("its expert tensors are read only while the bank file lacks layers; the rest of it at every "
                    "start, and per token under --ple-backend disk")

    # ---- GPUs (the real host only unless the caller hands in a fake)
    if gpu_query is None and sys == "/sys":
        gpu_query = gp.query_links
    h2d_gbs = _gpus(rep, ns, shape, ranks, gpu_query, gpu_rate or gp.h2d_rate)

    # ---- readahead
    rep.section("Readahead")
    ra = dp.readahead(s.dev, sys) if s.dev else None
    rec = dp.recommend_readahead_kb(shape.widest_row_bytes) if shape.widest_row_bytes else None
    if ra is None:
        rep.say("no read_ahead_kb found for this device")
    else:
        kb, where = ra
        rep.say(f"current: {kb} kB ({where})")
        if rec is not None:
            widest = shape.widest_row_bytes // 1024
            rep.say(f"recommended: {rec} kB for a widest row block of {widest} kB (measured optimum: a quarter "
                    f"to a sixth of it, on two models)")
            if kb != rec:
                rep.say(f"set until reboot, before starting the server (a running one keeps the window its "
                        f"mapping was opened with): {dp.readahead_command(rec, where)}")
                rep.say(f"or: ft serve ... --moe-bank-readahead auto (writes it where the server may)")
            if kb > max(widest, 1):
                rep.find("bad", f"read_ahead_kb {kb} is wider than the widest expert row block ({widest} kB): "
                                f"measured 2.5x slower decode at 8192 than at the optimum. "
                                f"{dp.readahead_command(rec, where)}")
            elif kb < rec // 2:
                rep.find("warn", f"read_ahead_kb {kb} is under half the recommended {rec} kB; "
                                 f"measured 13% slower at half the optimum on gpt-oss-120b")
        if s.device and s.device.virtual and s.wsl:
            rep.say("WSL2: this window is the virtual disk's, and it is the one page faults use")

    # ---- memory
    rep.section("Memory")
    mem = dp.meminfo(proc)
    total, avail = mem.get("MemTotal", 0), mem.get("MemAvailable", 0)
    swap_total, swap_free = mem.get("SwapTotal", 0), mem.get("SwapFree", 0)
    rep.say(f"MemTotal {_gib(total)}, MemAvailable {_gib(avail)}, swap {_gib(swap_total - swap_free)} used of "
            f"{_gib(swap_total)}")
    auto = None
    try:
        auto = dp.auto_bank_ram(mem, ranks, proc)
        rep.say(auto.reason().replace("--moe-bank-ram auto: ", "--moe-bank-ram auto would choose ", 1))
    except ValueError as exc:
        rep.say(str(exc))
        rep.find("bad", "not enough available memory for --moe-bank-ram auto to choose anything")
    lim = dp.memlock_limit(proc)
    if lim is not None:
        soft, hard = lim
        rep.say(f"memlock (ulimit -l): {_size(soft)} soft, {_size(hard)} hard")
        per_rank = (auto.total_bytes // ranks) if auto else None
        if soft is not None and per_rank and soft < per_rank:
            rep.find("warn", f"RLIMIT_MEMLOCK {_size(soft)} is under one rank's share ({_size(per_rank)}). Where "
                             f"the GPU registration covers the resident rows this does not matter; where it does not, "
                             f"the rows past the limit stay evictable. Which it is needs the GPU: the startup line "
                             f"warns when it matters. docs/bank-ram.md#the-memlock-limit")
    if dp.is_wsl(proc):
        rep.say("WSL2: autoMemoryReclaim=gradual hands idle page cache back to Windows, which empties the "
                "non-resident rows between requests; --moe-bank-rewarm reads them back")

    # ---- benchmark
    rep.section("Read benchmark")
    gbs = ns.disk_gbs
    fault_gbs = ns.fault_gbs
    pread_gbs = ns.prefill_read_gbs
    # Not a bank file with layers still to write: its holes read back as zeros at memory speed.
    complete = bank is not None and not bank.error and bank.all_layers and len(bank.layers) == bank.all_layers
    bench_file = bank_path if complete else _largest_file(ns.model)
    if ns.disk_gbs:
        rep.say(f"skipped: --disk-gbs {ns.disk_gbs} given")
    elif ns.bench_seconds <= 0:
        rep.say("skipped: --bench-seconds 0")
    elif not bench_file:
        rep.say("skipped: no bank file or checkpoint file to read")
    elif not shape.cell_bytes:
        rep.say("skipped: the expert row size is unknown (pass --model)")
    else:
        users, unreadable = dp.mapped_by(bench_file, proc)
        if users and not ns.bench_anyway:
            names = ", ".join(f"{pid} ({comm})" for pid, comm in users)
            rep.say(f"skipped: {os.path.basename(bench_file)} is mapped by {names} -- a running server would "
                    f"slow down and the number would be low. --bench-anyway to run it regardless")
        else:
            if unreadable:
                rep.say(f"({unreadable} processes of other users could not be checked for the file)")
            threads = ns.threads or min(32, dp.physical_cores(sys) * ranks)
            row = shape.cell_bytes
            rep.say(f"O_DIRECT (the page cache neither answers nor is touched), whole {row / MiB:.2f} MiB rows at "
                    f"random from {bench_file}, {ns.bench_seconds:g} s each")
            try:
                one = dp.random_row_read(bench_file, row, 1, ns.bench_seconds)
                many = dp.random_row_read(bench_file, row, threads, ns.bench_seconds)
                rep.say(f"1 thread: {one:.2f} GB/s; {threads} threads (the decode workers): {many:.2f} GB/s")
                gbs = many
                if bench_file == bank_path and fault_gbs is None:
                    try:
                        fault_gbs, fault_bytes = dp.fault_read(bench_file, ns.bench_seconds)
                        rep.say(f"through the page cache (a mapping, faults, readahead {ra[0] if ra else '?'} kB, one "
                                f"thread; how a prefill read them with FREETOKEN_BANK_PREAD=0): {fault_gbs:.2f} GB/s "
                                f"over {fault_bytes / GiB:.2f} GiB")
                    except OSError as exc:
                        rep.say(f"page-cache read not measured: {exc.strerror or exc}")
                if bench_file == bank_path and pread_gbs is None:
                    try:
                        pread_gbs, how_read = dp.piece_read(bench_file, ns.bench_seconds)
                        rep.say(f"as a prefill chunk reads them ({how_read}): {pread_gbs:.2f} GB/s")
                    except OSError as exc:
                        rep.say(f"prefill read not measured: {exc.strerror or exc}")
                if bench_file != bank_path:
                    rep.say("(read from the checkpoint, the bank file being absent or incomplete: the same device "
                            "as the bank only if the storage above says so)")
            except OSError as exc:
                rep.say(f"not measured: {exc.strerror or exc} -- a buffered read here would measure RAM, not "
                        f"the disk. Pass --disk-gbs to predict with a known figure")

    # ---- prediction
    rep.section("Prediction per RAM cap (whole host)")
    if not (shape.bank_bytes and total):
        rep.say("needs the model shape (pass --model)")
        return rep.render()
    fit = load_freq(ns.moe_bank_stats) if ns.moe_bank_stats else None
    evaluate = load_freq(ns.eval_stats) if ns.eval_stats else None
    caps = sorted({int(c) for c in ns.ram}) if ns.ram else default_caps(shape.bank_bytes, total, auto.total_bytes if auto else None)
    base = ns.base_step_ms
    base_note = "--base-step-ms"
    if base is None:
        for key, (ms, where) in KNOWN_BASE_STEP_MS.items():
            if key in (shape.name or "").lower():
                base, base_note = ms, f"{where}, measured on another machine"
    rows = predict(shape, caps, ranks, avail, fit=fit, evaluate=evaluate, disk_gbs=gbs, base_step_ms=base,
                   factor=ns.page_cache_factor)
    head = f"  {'cap':>9} {'resident':>9} {'covered':>8} {'page cache':>11} {'disk/token':>11} {'disk':>8} {'step':>8} {'tok/s':>6}"
    rep.lines.append(head)
    for r in rows:
        tag = "auto " if auto and r.cap == auto.total_bytes else ""
        rep.lines.append(
            f"  {tag + f'{r.cap / GiB:.1f}G':>9} {r.residency * 100:8.0f}% {r.coverage * 100:7.1f}% "
            f"{r.page_cache / GiB:10.1f}G "
            + (f"{r.disk_bytes / MiB:8.0f} MiB" if r.disk_bytes is not None else f"{'?':>11}")
            + (f" {r.disk_ms:6.1f}ms" if r.disk_ms is not None else f" {'?':>8}")
            + (f" {r.step_ms:6.1f}ms {1000.0 / r.step_ms:6.1f}" if r.step_ms else f" {'?':>8} {'?':>6}")
        )
    rep.say("")
    rep.say("assumed, and each of these can be off:")
    kind = rows[0].coverage_kind if rows else "none"
    if kind == "arbitrary":
        rep.say("- covered = resident: no histogram, and without --moe-bank-stats the server's split ignores routing too")
    elif kind == "in-sample":
        rep.say("- covered: counted on the same histogram that orders the placement, which overstates it -- measured on "
                "Flash-Next at 77% resident, 99.2% in-sample against 87.5% on a different session. Pass --eval-stats "
                "with a session the placement did not see")
    else:
        rep.say("- covered: placement ordered by --moe-bank-stats, counted on --eval-stats")
    rep.say(f"- page cache = MemAvailable - cap - {dp.NONBANK_PER_RANK_BYTES / GiB:.1f} GiB per rank for the rest of the server")
    rep.say(f"- disk/token = MoE layers x top-k x (1 - covered) x row size x a page cache factor: {ns.page_cache_factor:g} "
            f"when the page cache holds {PAGE_CACHE_REF_SHARE:.0%} of the non-resident rows or more (measured once, "
            f"Flash-Next at 64 GB over 2360 tokens: 3% of routes reached the disk against 12.5% placed there), rising "
            f"linearly to 1 with no page cache. Short runs sit nearer 1: the cache fills over the first ~1500 tokens")
    if not shape.top_k:
        rep.say("- disk/token: needs the router's top-k, which only the model config has (pass --model)")
    if gbs:
        rep.say(f"- disk = disk/token / {gbs:.2f} GB/s, which holds only with the readahead above set right (at 128 kB "
                f"faults capped a Gen4 drive at 1.44 GB/s)")
    else:
        rep.say("- disk: no read rate (benchmark skipped or failed; --disk-gbs gives one)")
    if base:
        rep.say(f"- step = {base:g} ms ({base_note}) + disk; tok/s = 1000 / step, one token per step. "
                "With --spec-mtp a step yields several tokens, so the rate is higher by about the "
                "accepted tokens per step (the decode log's accepted/step)")
    else:
        rep.say("- step: unknown base for this model. Pass --base-step-ms = 1000 / the decode tok/s you get with the "
                "banks fully in RAM (or with a cap that covers them)")
    rep.say("- the VRAM expert cache is not in this: it holds the hottest experts, which are resident anyway")
    _prefill_prediction(rep, rows, ranks, shape, pread_gbs, h2d_gbs, fault_gbs=fault_gbs)
    return rep.render()


# page cache host copy rate assumed when the rows are already in RAM (a single-thread memcpy;
# measured 14-17 GiB/s on the RTX 2060 host, so this is on the low side)
PREFILL_MEMCPY_GBS = 10.0


def _prefill_prediction(rep: Report, rows: list[Row], ranks: int, shape: Shape,
                        read_gbs: float | None, h2d_gbs: float | None, *, fault_gbs: float | None = None) -> None:
    """Data movement per prefill chunk, per rank: what the chunk costs before the GPU computes anything.

    Every chunk longer than the CPU prefill cut-off streams each layer's whole bank to the GPU:
    the resident rows straight from registered memory, the rest copied out of the mapping first.
    When the page cache cannot hold a rank's non-resident rows the copy re-reads nearly all of
    them each chunk -- measured on the RTX 2060 host with a third of them cached, 10.6 of 11 GiB
    came from the disk every chunk, because reading one layer evicts the one before.
    """
    rep.section("Prediction: prefill data movement per chunk, per rank")
    head = f"  {'cap':>9} {'non-resident':>13} {'page cache':>11} {'from disk':>10} {'copy':>8} {'to GPU':>8} {'total':>8}"
    rep.lines.append(head)
    worst = None
    for r in rows:
        cold = r.cold_bytes / ranks
        page = r.page_cache / ranks
        fits = page >= cold
        disk = 0.0 if fits else cold
        copy = None
        if cold == 0:
            copy = 0.0
        elif fits:
            copy = cold / (PREFILL_MEMCPY_GBS * 1e9)
        elif read_gbs:
            copy = cold / (read_gbs * 1e9)
        gpu = shape.bank_bytes / ranks / (h2d_gbs * 1e9) if h2d_gbs else None
        total = copy + gpu if copy is not None and gpu is not None else None
        rep.lines.append(
            f"  {r.cap / GiB:8.1f}G {cold / GiB:12.1f}G {page / GiB:10.1f}G {disk / GiB:9.1f}G "
            + (f"{copy:7.1f}s" if copy is not None else f"{'?':>8}")
            + (f" {gpu:7.1f}s" if gpu is not None else f" {'?':>8}")
            + (f" {total:7.1f}s" if total is not None else f" {'?':>8}")
        )
        if not fits and copy is not None:
            worst = (r, cold, copy)
    rep.say("")
    rep.say("assumed, and each of these can be off:")
    rep.say("- non-resident = the rank's bank rows past the resident prefix; page cache as in the table above, split per rank")
    rep.say("- from disk: all of them each chunk when the page cache is smaller than them (measured once, see above), "
            "none when it is larger (after the first chunk)")
    rep.say("- copy = from disk / the prefill read rate above" + (f" ({read_gbs:.2f} GB/s)" if read_gbs else
            " (not measured: a complete bank file nobody has mapped is needed; --prefill-read-gbs gives one)")
            + f", or / {PREFILL_MEMCPY_GBS:g} GB/s from RAM")
    if fault_gbs and read_gbs:
        rep.say(f"- with FREETOKEN_BANK_PREAD=0 the copy faults the rows in through the mapping instead: x "
                f"{read_gbs / fault_gbs:.1f} the copy time here")
    rep.say("- to GPU = the rank's whole bank / the slowest pinned host -> GPU rate above"
            + ("" if h2d_gbs else " (not measured: --h2d-seconds)"))
    rep.say("- total is data movement only: the GPU's compute comes on top, and a pipeline rank also waits for "
            "the other. Compare with --prefill-profile's per-chunk line on the server")
    if worst is not None:
        r, cold, copy = worst
        rep.find("warn", f"prefill at --moe-bank-ram {r.cap / GiB:.0f}G: the page cache cannot hold a rank's "
                         f"{cold / GiB:.1f} GiB of non-resident rows, so every chunk reads them from the disk again "
                         f"(~{copy:.0f} s per chunk per rank at the prefill read rate)")


def _largest_file(model_path: str | None) -> str | None:
    if not model_path or not os.path.isdir(model_path):
        return None
    best = None
    for name in os.listdir(model_path):
        p = os.path.join(model_path, name)
        if os.path.isfile(p) and (best is None or os.path.getsize(p) > os.path.getsize(best)):
            best = p
    return best


def _cap_list(text: str) -> list[int]:
    from freetoken.moe.bank_disk import parse_size

    try:
        return [parse_size(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser(prog: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="checkpoint directory (model shape, and the default bank directory)")
    p.add_argument("--moe-bank-dir", help="bank directory, as for ft serve (default ~/.cache/freetoken/bankmap/<model>)")
    p.add_argument("--pp-size", type=int, default=None, help="ranks sharing the host (default 1)")
    p.add_argument("--moe-bank-stats", nargs="+", default=None,
                   help="--moe-stats-out histogram(s) that order the placement, as for ft serve")
    p.add_argument("--eval-stats", nargs="+", default=None,
                   help="histogram(s) of a session the placement did not see, to count coverage honestly")
    p.add_argument("--ram", type=_cap_list, default=None,
                   help="whole-host caps to predict, comma-separated (default: auto and 50-100%% of the banks)")
    p.add_argument("--base-step-ms", type=float, default=None,
                   help="decode step with the banks fully in RAM on this machine (1000 / tok/s)")
    p.add_argument("--disk-gbs", type=float, default=None, help="skip the benchmark and assume this read rate")
    p.add_argument("--fault-gbs", type=float, default=None,
                   help="assume this rate for reads through the page cache (FREETOKEN_BANK_PREAD=0) instead of measuring it")
    p.add_argument("--prefill-read-gbs", type=float, default=None,
                   help="assume this rate for a prefill's direct reads of the non-resident rows instead of measuring it")
    p.add_argument("--page-cache-factor", type=float, default=PAGE_CACHE_FACTOR,
                   help=f"share of the placed miss that reaches the disk once the page cache settles (default {PAGE_CACHE_FACTOR})")
    p.add_argument("--bench-seconds", type=float, default=3.0, help="per benchmark pass; 0 skips it")
    p.add_argument("--h2d-seconds", type=float, default=2.0,
                   help="per GPU: pinned host -> GPU copy benchmark, with the PCIe link read during it; 0 skips it "
                        "(needs a GPU that a running server does not already fill)")
    p.add_argument("--threads", type=int, default=0, help="benchmark threads (default: physical cores x ranks)")
    p.add_argument("--bench-anyway", action="store_true",
                   help="benchmark even when another process maps the file")
    p.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    p.add_argument("--sys-root", default="/sys", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None, prog: str = "ft doctor disk") -> int:
    ns = build_parser(prog).parse_args(argv)
    if not os.path.isdir(os.path.join(ns.proc_root, "self")):
        print(f"{prog}: needs Linux (/proc and /sys); on Windows run it inside WSL2")
        return 1
    print(run(ns, ns.proc_root, ns.sys_root))
    return 0
