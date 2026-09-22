from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TextIO


def _print_help(file: TextIO) -> None:
    print(
        """usage: ft <command> [args]

Commands:
  serve       Start the FreeToken API server
  shell       Chat with a FreeToken server in the terminal
  ctl         Query and manage a running FreeToken server
  daemon      Run the FreeToken supervisor (persistent engine service)
  mgr         kai: the same supervisor plus the web console at /ui/ (profiles, advice)
  launch      Configure and launch an agent against a FreeToken server
  checkpoint  Convert an HF safetensors checkpoint to FTW
  bank        Inspect, pack, verify or reorder the --moe-bank-ram bank file
  bench       Run a micro-benchmark (e.g. "bench bw" = CPU vs PCIe bandwidth)
  doctor      Check this host for a feature (e.g. "doctor disk" = --moe-bank-ram)

Use "ft <command> --help" for command-specific options.
Use "ft --version" to print the FreeToken version.""",
        file=file,
    )


def _run_serve(argv: list[str]) -> int:
    from freetoken.server import launch_server

    launch_server(argv=argv, prog="ft serve")
    return 0


def _run_shell(argv: list[str]) -> int:
    from freetoken.shell import main

    return main(argv, prog="ft shell")


def _run_launch(argv: list[str]) -> int:
    from freetoken.launch import main

    return main(argv, prog="ft launch")


def _run_checkpoint(argv: list[str]) -> int:
    from freetoken.checkpoint.__main__ import main

    return main(argv, prog="ft checkpoint")


def _run_bank(argv: list[str]) -> int:
    from freetoken.moe.bank_cli import main

    return main(argv, prog="ft bank")


def _run_ctl(argv: list[str]) -> int:
    from freetoken.control_cli import main

    return main(argv, prog="ft ctl")


def _run_mgr(argv: list[str]) -> int:
    from freetoken.daemon import main  # torch-free supervisor; console=True adds kai's web console

    return main(argv, prog="ft mgr", console=True)


def _run_daemon(argv: list[str]) -> int:
    from freetoken.daemon import main  # torch-free supervisor

    return main(argv, prog="ft daemon")


def _print_bench_help(file: TextIO) -> None:
    print(
        """usage: ft bench <subcommand> [args]

Subcommands:
  bw   Benchmark CPU vs PCIe bandwidth and pick the MoE backend (hybrid/offload)

Use "ft bench <subcommand> --help" for subcommand-specific options.""",
        file=file,
    )


def _run_bench(argv: list[str]) -> int:
    if not argv:
        _print_bench_help(sys.stderr)
        return 2
    sub = argv[0]
    if sub in {"-h", "--help"}:
        _print_bench_help(sys.stdout)
        return 0
    if sub == "bw":
        from freetoken.moe.benchbw import main

        return main(argv[1:], prog="ft bench bw")
    print(f"unknown ft bench subcommand: {sub}", file=sys.stderr)
    _print_bench_help(sys.stderr)
    return 2


def _print_doctor_help(file: TextIO) -> None:
    print(
        """usage: ft doctor <subcommand> [args]

Subcommands:
  disk   Whether --moe-bank-ram is usable on this host: storage, readahead, memory,
         a read benchmark and a per-RAM-cap estimate (no GPU, no root)
  pin    How much host RAM this machine will page-lock, measured and recorded, and which
         MoE flags that figure calls for (needs a GPU; run it with no server up)

Use "ft doctor <subcommand> --help" for subcommand-specific options.""",
        file=file,
    )


def _run_doctor(argv: list[str]) -> int:
    if not argv:
        _print_doctor_help(sys.stderr)
        return 2
    sub = argv[0]
    if sub in {"-h", "--help"}:
        _print_doctor_help(sys.stdout)
        return 0
    if sub == "disk":
        from freetoken.moe.disk_doctor import main

        return main(argv[1:], prog="ft doctor disk")
    if sub == "pin":
        from freetoken.moe.pin_doctor import main

        return main(argv[1:], prog="ft doctor pin")
    print(f"unknown ft doctor subcommand: {sub}", file=sys.stderr)
    _print_doctor_help(sys.stderr)
    return 2


COMMANDS = {
    "serve": "_run_serve",
    "shell": "_run_shell",
    "ctl": "_run_ctl",
    "daemon": "_run_daemon",
    "mgr": "_run_mgr",
    "launch": "_run_launch",
    "checkpoint": "_run_checkpoint",
    "bank": "_run_bank",
    "bench": "_run_bench",
    "doctor": "_run_doctor",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_help(sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        _print_help(sys.stdout)
        return 0
    if args[0] in {"-V", "--version"}:
        from freetoken.version import __version__

        print(f"freetoken version {__version__}")
        return 0

    command = args[0]
    runner_name = COMMANDS.get(command)
    if runner_name is None:
        print(f"unknown ft command: {command}", file=sys.stderr)
        _print_help(sys.stderr)
        return 2

    runner = globals()[runner_name]
    return runner(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
