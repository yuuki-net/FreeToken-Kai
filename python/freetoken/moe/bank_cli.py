"""``ft bank``: the ``--moe-bank-ram`` bank file as the canonical copy of a checkpoint's experts.

    ft bank info    --model-path M [--moe-bank-dir D]
    ft bank pack    --model-path M --out SLIM [--moe-bank-dir D] [--dry-run] [--copy] [--keep-bank]
    ft bank verify  --model-path SLIM [--moe-bank-dir D]
    ft bank unpack  --model-path SLIM --out FULL [--moe-bank-dir D] [--copy]
    ft bank reorder --model-path M --moe-bank-stats A.json [B.json ...] [--moe-bank-dir D]

``pack`` writes a checkpoint without the expert tensors the bank file reproduces, after checking
every one of them byte for byte, and moves the bank into it. It deletes nothing. ``verify``
repeats the check on a packed checkpoint without the original; ``unpack`` writes the original
files back, byte for byte, from a packed checkpoint and its bank. ``reorder`` applies a new
placement to the bank file in place (the server does the same at startup for its own layers).
None of them uses a GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _model_config(model_path: str):
    import torch

    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    # parsing the model config also installs the checkpoint's quant config, which the NVFP4
    # expert naming (dialects) is read from
    return EngineConfig(model_path=model_path, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16).model_config


def _bank_path(args) -> str:
    from freetoken.moe.bank_pack import bank_path_for, read_pack_manifest

    return bank_path_for(args.model_path, read_pack_manifest(args.model_path), args.moe_bank_dir)


def _gib(n: int) -> str:
    return f"{n / 2**30:.1f} GiB"


def _say(msg: str) -> None:
    print(msg, flush=True)


def cmd_info(args) -> int:
    import glob

    from freetoken.moe.bank_file import BankFile, allocated_bytes
    from freetoken.moe.bank_pack import read_pack_manifest
    from freetoken.moe.mapped_bank import _ranges

    path = _bank_path(args)
    packed = read_pack_manifest(args.model_path)
    if packed is not None:
        _say(f"packed checkpoint: {os.path.abspath(args.model_path)} (from {packed['packed_from']} on {packed['date']})")
        _say(f"  removed {packed['removed']['tensors']} expert tensors, {_gib(packed['removed']['bytes'])}; roles {packed['removed']['roles']}")
    if not os.path.isfile(path):
        _say(f"no bank file at {path}")
        return 1
    with BankFile.open(path, writable=False) as bank:
        layout, manifest = bank.layout, bank.manifest()
        present = bank.present_layers(manifest)
        identity = [
            l for l in present if bank.layer_state(l, manifest)[0] == list(range(layout.num_experts))
        ]
        meta = layout.meta
        _say(f"bank file: {path}")
        _say(f"  {_gib(layout.total_bytes())} mapped, {_gib(allocated_bytes(path))} on disk")
        _say(f"  {meta.get('kind', '?')} / {meta.get('kernel', '?')}, {layout.num_experts} experts x {len(layout.layers)} MoE layers, {_gib(layout.layer_bytes())} per layer")
        _say(f"  banks: " + ", ".join(f"{n} {list(s)} {d}" for n, s, d, _ in layout.banks))
        _say(f"  fingerprint: {meta.get('fingerprint')}")
        missing = sorted(set(layout.layers) - set(present))
        _say(f"  committed layers: {_ranges(present)}" + (f"; missing {_ranges(missing)}" if missing else " (complete)"))
        if identity:
            _say(f"  checkpoint order (no placement applied): layers {_ranges(identity)}")
        marked = bank.canonical_for(manifest)
        if marked:
            gone = "" if os.path.exists(marked) else " -- which no longer exists; the next pack drops the mark"
            _say(f"  the only copy of the experts of: {marked}{gone}")
        journals = glob.glob(glob.escape(path) + ".journal.L*")
        if journals:
            _say(f"  unfinished reorder journals: {len(journals)} (resolved on the next open for writing)")
    return 0


def cmd_pack(args) -> int:
    from freetoken.moe.bank_pack import PackError, pack_checkpoint

    started = time.perf_counter()
    bank_path = _bank_path(args)
    try:
        report = pack_checkpoint(
            args.model_path, args.out, config=_model_config(args.model_path), bank_path=bank_path,
            link=not args.copy, keep_bank=args.keep_bank, dry_run=args.dry_run, log=_say,
        )
    except PackError as exc:
        print(f"ft bank pack: {exc}", file=sys.stderr)
        return 1
    if report.get("dry_run"):
        _say(f"dry run: nothing read or written. Without --dry-run, {report['removed_tensors']} tensors "
             f"({_gib(report['removed_bytes'])}) would be checked against the bank's {report['layers']} layers "
             f"and left out of {report['out']}")
        return 0
    shards = report["shards"]
    _say("")
    _say(f"wrote {report['out']} in {time.perf_counter() - started:.0f} s")
    _say(f"  {report['removed_tensors']} expert tensors ({_gib(report['removed_bytes'])}) removed; every one was "
         f"regenerated from the bank and matched byte for byte, and all {report['layers']} layers of the "
         f"bank match what the checkpoint packs to")
    _say(f"  shards: {shards['link']} hard-linked, {shards['copy']} copied, {shards['rewrite']} rewritten "
         f"({_gib(report['rewritten_bytes'])} written), {shards['drop']} left out")
    _say(f"  bank file: {report['bank']} -- now the only copy of those experts; do not delete it")
    _say("")
    _say(f"Nothing was deleted. {os.path.abspath(args.model_path)} is no longer needed to serve "
         f"{report['out']} with --moe-bank-ram and can be deleted"
         + (" (its shards without experts are hard links now, so deleting the original frees only "
            "the expert data)" if shards["link"] else ""))
    return 0


def cmd_verify(args) -> int:
    from freetoken.moe.bank_pack import PackError, verify_packed

    started = time.perf_counter()
    try:
        report = verify_packed(
            args.model_path, config=_model_config(args.model_path),
            bank_path=_bank_path(args) if args.moe_bank_dir else None, log=_say,
        )
    except PackError as exc:
        print(f"ft bank verify: {exc}", file=sys.stderr)
        return 1
    _say(f"ok: {report['layers']} layers of {report['bank']} give back all {report['removed_tensors']} "
         f"removed tensors byte for byte ({time.perf_counter() - started:.0f} s)")
    return 0


def cmd_unpack(args) -> int:
    from freetoken.moe.bank_pack import PackError, unpack_checkpoint

    started = time.perf_counter()
    try:
        report = unpack_checkpoint(
            args.model_path, args.out, config=_model_config(args.model_path),
            bank_path=_bank_path(args) if args.moe_bank_dir else None, link=not args.copy, log=_say,
        )
    except PackError as exc:
        print(f"ft bank unpack: {exc}", file=sys.stderr)
        return 1
    shards = report["shards"]
    _say(f"wrote {report['out']} in {time.perf_counter() - started:.0f} s: "
         f"{shards.get('rewrite', 0) + shards.get('drop', 0)} shards rebuilt and matching the original "
         f"SHA-256, {shards.get('link', 0)} hard-linked, {shards.get('copy', 0)} copied")
    return 0


def cmd_reorder(args) -> int:
    from freetoken.moe import bank_disk
    from freetoken.moe.bank_file import BankFile, BankFileError, free_bytes
    from freetoken.moe.mapped_bank import _ranges

    path = _bank_path(args)
    if not os.path.isfile(path):
        print(f"ft bank reorder: no bank file at {path}", file=sys.stderr)
        return 1
    mc = _model_config(args.model_path)
    fkd = int(getattr(mc, "first_k_dense_replace", 0) or 0)
    freq = bank_disk.load_freq(args.moe_bank_stats, first_k_dense=fkd)
    with BankFile.open(path) as bank:
        bank.recover_journals(log=_say)
        layout = bank.layout
        present = bank.present_layers()
        wanted = bank_disk.plan_placement(present, layout.num_experts, layout.num_experts, freq).order
        todo = [l for l in present if bank.layer_state(l)[0] != wanted[l]]
        uncovered = [l for l in present if l not in freq]
        if uncovered:
            _say(f"no histogram for layers {_ranges(uncovered)}: they go to checkpoint order")
        if not todo:
            _say("every layer is already in that order")
            return 0
        free = free_bytes(path)
        if free is not None and layout.layer_bytes() > free:
            print(f"ft bank reorder: one layer's journal needs {_gib(layout.layer_bytes())}, {_gib(free)} free", file=sys.stderr)
            return 1
        started = time.perf_counter()
        try:
            for i, layer in enumerate(todo, 1):
                bank.reorder_layer(layer, wanted[layer])
                bank.drop_cache(layer)
                _say(f"  layer {layer} reordered ({i}/{len(todo)})")
        except BankFileError as exc:
            print(f"ft bank reorder: {exc}", file=sys.stderr)
            return 1
        _say(f"reordered {len(todo)} layers ({_gib(len(todo) * layout.layer_bytes())}) in {time.perf_counter() - started:.0f} s")
    return 0


def main(argv: list[str] | None = None, prog: str = "ft bank") -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model-path", "--model", dest="model_path", required=True,
                        help="the checkpoint directory (original or packed), as for ft serve")
        sp.add_argument("--moe-bank-dir", default=None,
                        help="directory of the bank file, as for ft serve (default: inside a packed "
                             "checkpoint, else ~/.cache/freetoken/bankmap/<model>)")

    sp = sub.add_parser("info", help="what the bank file holds")
    common(sp)
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("pack", help="write a checkpoint without the expert tensors the bank reproduces")
    common(sp)
    sp.add_argument("--out", required=True, help="new directory for the packed checkpoint")
    sp.add_argument("--copy", action="store_true", help="copy shards without experts instead of hard-linking them")
    sp.add_argument("--dry-run", action="store_true",
                    help="check the bank and the headers and say what would happen, without reading a tensor")
    sp.add_argument("--keep-bank", action="store_true",
                    help="leave the bank file where it is (the packed checkpoint refers to it by path) "
                         "instead of moving it into --out")
    sp.set_defaults(fn=cmd_pack)

    sp = sub.add_parser("verify", help="check a packed checkpoint against its bank, without the original")
    common(sp)
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("unpack", help="write the original checkpoint back from a packed one")
    common(sp)
    sp.add_argument("--out", required=True, help="new directory for the original checkpoint")
    sp.add_argument("--copy", action="store_true", help="copy shards without experts instead of hard-linking them")
    sp.set_defaults(fn=cmd_unpack)

    sp = sub.add_parser("reorder", help="apply a new placement to the bank file in place")
    common(sp)
    sp.add_argument("--moe-bank-stats", nargs="+", required=True, help="--moe-stats-out histograms (every rank's)")
    sp.set_defaults(fn=cmd_reorder)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
