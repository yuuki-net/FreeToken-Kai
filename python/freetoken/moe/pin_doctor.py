"""``ft doctor pin``: how much host RAM this machine will page-lock, and what to set because of it.

The offload and hybrid MoE paths need the expert banks page-locked before the GPU may read
them, and WSL caps how much of that a host allows. Nothing derives the cap -- see
:mod:`freetoken.moe.pin_probe` -- so this measures it, writes it down, and says what the
figure means for a start: which flag to pass, and what it costs.

Unlike ``ft doctor disk`` this one needs a GPU, because only CUDA can answer. It also holds
the memory it locks while it runs, so it wants a machine with no server on it: the cap is a
quota shared by every process, and a server already holding part of it makes the answer
smaller than the host's.
"""

from __future__ import annotations

import argparse
import os

from freetoken.moe import pin_probe as pp

GiB = pp.GiB


def _fmt(nbytes: int | None) -> str:
    return "-" if nbytes is None else f"{nbytes / GiB:.2f} GiB"


def run(ns) -> str:
    out: list[str] = []
    say = out.append
    total = pp._meminfo_bytes("MemTotal", ns.proc_root)
    available = pp._meminfo_bytes("MemAvailable", ns.proc_root)

    say("host")
    say(f"  guest RAM        {_fmt(total)} total, {_fmt(available)} available")
    capped = pp.is_pin_capped()
    say(f"  pinning capped   {'yes (WSL)' if capped else 'no (this platform does not cap page-locked host memory)'}")
    if not capped:
        say("")
        say("Nothing to measure: on a host that does not cap pinning the banks are pinned in full and")
        say("--moe-cpu-layers is a choice about CPU decode, not about a budget. Done.")
        return "\n".join(out)

    known = pp.remembered(ns.proc_root)
    say(f"  on record        {_fmt(known.cap_bytes) if known else 'nothing has been refused on this host yet'}"
        + (f" ({known.how})" if known else ""))
    say(f"  estimate         {_fmt(pp.estimate(ns.proc_root))} ({pp._ESTIMATE_FRACTION:.0%} of guest RAM -- a guess, "
        "wrong by 10x on some hosts)")
    say(f"  in use now       {pp.source(ns.proc_root)}, {_fmt(pp.budget(proc=ns.proc_root))}")
    say(f"  record kept in   {pp.cache_path()}")

    if ns.no_measure:
        say("")
        say("--no-measure: nothing was locked, so the figures above are only what was already known.")
        return "\n".join(out)

    # The budget is the ceiling, not the guest's free RAM. Nothing will ever plan for more than
    # the budget, so there is nothing above it to learn -- while climbing there costs real,
    # unreclaimable host memory. A ladder run this way pins for a moment what a server pins for
    # hours, and no more.
    budget_now = pp.budget(proc=ns.proc_root) or 0
    ceiling = int(ns.ceiling * GiB) if ns.ceiling else budget_now
    say("")
    if ns.ceiling and ceiling > budget_now:
        say(f"  WARNING: --ceiling {ns.ceiling} GiB is above the {_fmt(budget_now)} a start would use.")
        say("  Nothing above that figure is usable, and pinned pages are never reclaimed: this is")
        say("  how a 128 GB host was taken down. Stop the ladder if the machine starts to struggle.")
    say(f"measuring (locking up to {_fmt(ceiling)} in {pp.STEP_BYTES >> 20} MiB steps; a running server shares this quota)")
    try:
        m = pp.measure(ceiling, proc=ns.proc_root,
                       on_step=None if ns.quiet else lambda n: print(f"  +{n / GiB:5.2f} GiB", flush=True))
    except Exception as exc:  # noqa: BLE001 -- a host without a usable GPU still gets a report
        say(f"  could not measure: {type(exc).__name__}: {exc}")
        say("  (this needs a GPU CUDA can open; inside WSL2 check that nvidia-smi works)")
        return "\n".join(out)

    say(f"  {m.summary()}")
    share = f" ({m.locked_bytes / total:.0%} of guest RAM)" if total else ""
    say(f"  locked {_fmt(m.locked_bytes)}{share}")

    if m.refused:
        wrote = pp.remember(m.locked_bytes, how="ft doctor pin", proc=ns.proc_root)
        say(f"  recorded as this host's cap{'' if wrote else ' -- FAILED to write the record, see the path above'}")
    else:
        # nothing refused, so nothing here is this host's. Deliberately not recorded: "held N GiB
        # once" reads like a budget, and raising one to match took a host down (see the note below)
        say("  nothing recorded: only a refusal measures this host, and the driver did not refuse")
        if not ns.ceiling and m.ceiling_hit:
            say("  and that is the good outcome: the whole budget page-locks, so a start has no reason")
            say("  to move layers to the CPU for want of pinned memory")
        if m.ram_limited:
            say("  the guest ran out of RAM before the budget was reached; free some and run it again")

    say("")
    say("what to do with it")
    budget = m.locked_bytes if m.refused else None
    if budget is None:
        say(f"  Nothing refused up to {_fmt(m.locked_bytes)}, which is what a start plans against anyway,")
        say("  so there is nothing to record and nothing to change. Banks that fit that figure pin in")
        say("  full and need no flag.")
        say("  Going higher is not the next step: the driver may well allow it, but pinned pages are")
        say("  never reclaimed, and on one 128 GB host a budget raised that way pinned a model's whole")
        say("  bank set, served no token in 180 s and took the guest down.")
    elif budget < 2 * GiB:
        say(f"  {_fmt(budget)} is too little to pin expert banks. Serve with --moe-cpu-layers 1.0 (every expert")
        say("  decodes on the CPU), or --moe-strategy cpu, and expect CPU-bound decode. --moe-bank-ram is the")
        say("  other way out: it maps the banks and locks only a resident prefix, so the cap stops deciding.")
    else:
        say(f"  Banks up to {_fmt(budget)} pin in full. Over that, --moe-cpu-layers auto locks the excess")
        say("  head and tail layers for CPU decode; the split is planned against the figure above.")
    if budget is not None:
        say(f"  To override the recorded figure anywhere: FREETOKEN_PIN_BUDGET_GB={budget / GiB:.2f}")
    say("  ft doctor disk covers the other half (the banks on disk, readahead, per-cap cost).")
    return "\n".join(out)


def build_parser(prog: str = "ft doctor pin") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="Measure and record how much host RAM this machine page-locks")
    p.add_argument("--ceiling", type=float, default=None,
                   help="stop after locking this many GiB (default: the budget a start would use; "
                        "above that there is nothing to learn and the pages are never reclaimed)")
    p.add_argument("--no-measure", action="store_true", help="print what is already known and lock nothing")
    p.add_argument("--quiet", action="store_true", help="do not print each step while measuring")
    p.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None, prog: str = "ft doctor pin") -> int:
    ns = build_parser(prog).parse_args(argv)
    if not os.path.isdir(os.path.join(ns.proc_root, "self")):
        print(f"{prog}: needs Linux (/proc); on Windows run it inside WSL2")
        return 1
    print(run(ns))
    return 0
