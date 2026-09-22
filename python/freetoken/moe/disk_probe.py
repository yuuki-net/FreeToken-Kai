"""What a ``--moe-bank-ram`` bank sits on, and what the host has to give it. Stdlib only.

Everything the mapped bank's speed depends on outside the code is invisible from inside the
process and was found the hard way (docs/bank-ram.md): the device readahead window (worth
2.5x), the filesystem (a WSL2 ``/mnt/c`` path is a 9p file server), the transport and the
slot the drive is in (a SATA SSD adds ~300 ms a step; a chipset M.2 shares its uplink with a
chipset GPU slot), how much RAM is really available and how much of it may be locked.

This module only reads: ``/proc`` and ``/sys`` (both roots are parameters, so the tests
hand it a fake tree), plus an O_DIRECT read benchmark that writes nothing and leaves the page
cache alone. ``ft doctor disk`` (moe/disk_doctor.py) prints it; the server uses the parts
that decide something at startup: ``auto_bank_ram`` for ``--moe-bank-ram auto``,
``recommend_readahead_kb`` / ``set_readahead`` for ``--moe-bank-readahead``, and
``storage_warnings`` for the one-line complaint about where the bank file is.
"""

from __future__ import annotations

import math
import mmap
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field

GiB = 2**30

# ---------------------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path: str) -> int | None:
    text = _read(path)
    if text is None:
        return None
    try:
        return int(text, 0) if text.startswith("0x") else int(text)
    except ValueError:
        return None


def nearest_existing(path: str) -> str:
    """``path``, or the closest ancestor that exists -- a bank directory is made on first use,
    and the question "what filesystem will it be on" has an answer before that."""
    path = os.path.abspath(os.path.expanduser(path))
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def is_wsl(proc: str = "/proc") -> bool:
    release = _read(os.path.join(proc, "sys/kernel/osrelease")) or ""
    return "microsoft" in release.lower() or "wsl" in release.lower()


# ---------------------------------------------------------------------------------------
# mounts and filesystems
# ---------------------------------------------------------------------------------------


@dataclass
class Mount:
    mountpoint: str
    fstype: str
    source: str
    dev: str  # "major:minor" as mountinfo gives it
    super_options: str = ""


def _unescape(text: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), text)


def parse_mountinfo(text: str) -> list[Mount]:
    """``/proc/self/mountinfo`` -> mounts in the order the kernel lists them (later ones
    stack on earlier ones at the same point)."""
    out = []
    for line in text.splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        f, r = left.split(), right.split()
        if len(f) < 5 or len(r) < 2:
            continue
        out.append(Mount(_unescape(f[4]), r[0], _unescape(r[1]), f[2], r[2] if len(r) > 2 else ""))
    return out


def mount_of(path: str, mounts: list[Mount]) -> Mount | None:
    """The mount ``path`` is on: the longest mount point that contains it, the later one on a tie."""
    best = None
    for m in mounts:
        mp = m.mountpoint.rstrip("/") or "/"
        if mp == "/" or path == mp or path.startswith(mp + "/"):
            if best is None or len(mp) >= len(best.mountpoint.rstrip("/") or "/"):
                best = m
    return best


# (severity, why). "bad" is a place --moe-bank-ram should not be run from at all; "warn" is
# one that works but was never measured or costs something specific.
_FS_VERDICTS = {
    "9p": ("bad", "a 9p file server (on WSL2, a Windows drive under /mnt): every page fault on "
                  "the mapping is a round trip to Windows, orders of magnitude slower than a disk"),
    "drvfs": ("bad", "a Windows drive seen through WSL (drvfs): every page fault on the mapping "
                     "is a round trip to Windows, orders of magnitude slower than a disk"),
    "virtiofs": ("bad", "a virtiofs share from the VM host: page faults go through the host's "
                        "file server"),
    "tmpfs": ("bad", "tmpfs, which is RAM: a bank file there costs the RAM --moe-bank-ram exists to save"),
    "ramfs": ("bad", "ramfs, which is RAM: a bank file there costs the RAM --moe-bank-ram exists to save"),
    "overlay": ("warn", "an overlay (container) layer: 30-60 GiB of bank written into the "
                        "container's upper layer; bind-mount a host directory and point "
                        "--moe-bank-dir at it"),
    "zfs": ("warn", "ZFS, where mapped reads are cached by the ARC and the page cache both, and the device "
                    "readahead window does not govern them; not measured"),
    "fuseblk": ("warn", "a FUSE block filesystem (NTFS/exFAT via FUSE): every fault goes through "
                        "a userspace daemon; not measured"),
    "ntfs3": ("warn", "NTFS (ntfs3), not measured"),
    "exfat": ("warn", "exFAT, not measured"),
    "vfat": ("bad", "FAT, which cannot hold a file over 4 GiB"),
}
_NETWORK_FS = {
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "afs", "ceph", "glusterfs", "lustre", "gpfs",
    "beegfs", "davfs", "fuse.sshfs", "sshfs", "fuse.rclone", "fuse.s3fs", "fuse.glusterfs",
}


def filesystem_verdict(mount: Mount) -> tuple[str, str]:
    """``("ok"|"warn"|"bad", reason)`` for the filesystem a bank file would be mapped from."""
    fstype = mount.fstype
    if fstype == "9p" and "drvfs" in mount.super_options:
        return _FS_VERDICTS["drvfs"]
    if fstype in _FS_VERDICTS:
        return _FS_VERDICTS[fstype]
    if fstype in _NETWORK_FS:
        return "bad", f"{fstype}, a network filesystem: every page fault is a network round trip"
    if fstype.startswith("fuse"):
        return "warn", f"{fstype}, where every fault goes through a userspace daemon; not measured"
    return "ok", fstype


# ---------------------------------------------------------------------------------------
# the block device under a filesystem
# ---------------------------------------------------------------------------------------

_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_GT_TO_GEN = {"2.5": 1, "5.0": 2, "5": 2, "8.0": 3, "8": 3, "16.0": 4, "16": 4, "32.0": 5,
              "32": 5, "64.0": 6, "64": 6}


@dataclass
class PciLink:
    bdf: str
    speed: str | None
    width: int | None
    max_speed: str | None
    max_width: int | None

    @staticmethod
    def gen(speed: str | None) -> int | None:
        if not speed:
            return None
        return _GT_TO_GEN.get(speed.split()[0])

    def describe(self) -> str:
        cur = self.gen(self.speed)
        top = self.gen(self.max_speed)
        text = f"PCIe Gen{cur or '?'} x{self.width or '?'}"
        if (top and cur and top > cur) or (self.max_width and self.width and self.max_width > self.width):
            text += f" (device supports Gen{top or '?'} x{self.max_width or '?'})"
        return text


@dataclass
class BlockDevice:
    name: str  # the whole disk the filesystem is on: nvme0n1, sda
    transport: str  # nvme, sata, usb, mmc, virtio, hyperv, xen, scsi, loop, ram, unknown
    virtual: bool
    rotational: bool | None
    model: str | None = None
    stacked_on: list[str] = field(default_factory=list)  # dm-0 -> nvme0n1p2 -> ...
    partition: str | None = None
    pci_chain: list[tuple[str, str]] = field(default_factory=list)  # (bdf, sysfs dir), root first
    link: PciLink | None = None
    backing_file: str | None = None  # loop devices


def _block_link(dev: str, source: str, sys: str) -> str | None:
    link = os.path.join(sys, "dev/block", dev)
    if os.path.exists(link):
        return link
    # btrfs and friends report an anonymous 0:N device; the mount source still names the disk
    if source.startswith("/dev/"):
        cand = os.path.join(sys, "class/block", os.path.basename(source))
        if os.path.exists(cand):
            return cand
    return None


def _pci_chain(real: str) -> list[tuple[str, str]]:
    parts = real.split("/")
    return [(p, "/".join(parts[: i + 1])) for i, p in enumerate(parts) if _BDF.match(p)]


def block_device(dev: str, source: str = "", sys: str = "/sys") -> BlockDevice | None:
    """The physical disk behind ``dev`` ("major:minor"), through dm/md stacking and partitions."""
    link = _block_link(dev, source, sys)
    if link is None:
        return None
    real = os.path.realpath(link)
    stacked = []
    for _ in range(8):  # dm on md on a partition is as deep as anything sane goes
        slaves = os.path.join(real, "slaves")
        members = sorted(os.listdir(slaves)) if os.path.isdir(slaves) else []
        if not members:
            break
        stacked.append(os.path.basename(real))
        real = os.path.realpath(os.path.join(slaves, members[0]))
    partition = None
    if os.path.exists(os.path.join(real, "partition")):
        partition = os.path.basename(real)
        real = os.path.dirname(real)
    name = os.path.basename(real)
    vendor = (_read(os.path.join(real, "device/vendor")) or "").strip()
    model = (_read(os.path.join(real, "device/model")) or "").strip() or None
    rot = _read_int(os.path.join(real, "queue/rotational"))
    backing = None
    virtual = False
    if name.startswith("nvme") or "/nvme/" in real:
        transport = "nvme"
    elif name.startswith("loop"):
        transport, backing = "loop", _read(os.path.join(real, "loop/backing_file"))
    elif name.startswith(("ram", "zram")):
        transport = "ram"
    elif name.startswith("mmcblk"):
        transport = "mmc"
    elif "/usb" in real:
        transport = "usb"
    elif "/virtio" in real or name.startswith("vd"):
        transport, virtual = "virtio", True
    elif vendor == "Msft" and "Virtual" in (model or ""):
        transport, virtual = "hyperv", True
    elif name.startswith("xvd") or "/vbd-" in real:
        transport, virtual = "xen", True
    elif "/ata" in real:
        transport = "sata"
    elif "/host" in real and "/target" in real:
        transport = "scsi"
    else:
        transport = "unknown"
    chain = _pci_chain(real)
    link_info = None
    if chain and transport == "nvme":
        bdf, d = chain[-1]
        link_info = PciLink(
            bdf,
            _read(os.path.join(d, "current_link_speed")),
            _read_int(os.path.join(d, "current_link_width")),
            _read(os.path.join(d, "max_link_speed")),
            _read_int(os.path.join(d, "max_link_width")),
        )
    return BlockDevice(
        name=name, transport=transport, virtual=virtual,
        rotational=None if rot is None else bool(rot), model=model, stacked_on=stacked,
        partition=partition, pci_chain=chain, link=link_info, backing_file=backing,
    )


# AMD chipset (Promontory) bridge device ids seen on AM4/AM5 boards. Not exhaustive; a board
# whose bridge is not listed reads as "unknown", never as CPU-attached.
_AMD_CHIPSET_BRIDGES = {
    0x43B4, 0x43B5, 0x43B6, 0x43B7, 0x43C6, 0x43C7, 0x43E9, 0x43EA, 0x43EB,
    0x57A3, 0x57A4, 0x57AD, 0x43F4, 0x43F5,
}


def upstream_kind(chain: list[tuple[str, str]]) -> str:
    """``"chipset"``, ``"cpu"`` or ``"unknown"``: which side of the chipset uplink a PCI device is.

    An estimate from the topology, for the one question it matters for: a chipset M.2 shares
    one DMI / Promontory uplink with every other chipset device, a chipset x4 GPU slot
    included, so under ``--pp-size 2`` the bank reads and rank 1's residual stream fight
    over it (docs/bank-ram.md, Limits).
    """
    if not chain:
        return "unknown"
    root_bdf, root_dir = chain[0]
    vendor = _read_int(os.path.join(root_dir, "vendor"))
    bus, slot = root_bdf.split(":")[1], int(root_bdf.split(":")[2].split(".")[0], 16)
    if vendor == 0x8086 and bus == "00" and slot in (0x1B, 0x1C, 0x1D):
        return "chipset"  # Intel PCH root ports: behind DMI
    for _, d in chain[:-1]:
        v, dv = _read_int(os.path.join(d, "vendor")), _read_int(os.path.join(d, "device"))
        if (v == 0x1022 and dv in _AMD_CHIPSET_BRIDGES) or v == 0x1B21:
            return "chipset"  # AMD Promontory, or an ASMedia chipset bridge
    if len(chain) == 2:
        return "cpu"  # an endpoint straight under a root port, and not a PCH one
    return "unknown"


def nvidia_gpus(sys: str = "/sys") -> list[tuple[str, list[tuple[str, str]]]]:
    """``[(bdf, chain)]`` for every NVIDIA display controller sysfs lists."""
    base = os.path.join(sys, "bus/pci/devices")
    out = []
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in names:
        d = os.path.join(base, name)
        cls = _read(os.path.join(d, "class")) or ""
        if _read_int(os.path.join(d, "vendor")) == 0x10DE and cls.startswith(("0x0300", "0x0302")):
            out.append((name, _pci_chain(os.path.realpath(d))))
    return out


def shared_upstream(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> str | None:
    """The deepest bridge two chains share, or None when they meet only at the root complex."""
    common = None
    for (x, _), (y, _) in zip(a[:-1], b[:-1]):
        if x != y:
            break
        common = x
    return common


# ---------------------------------------------------------------------------------------
# readahead
# ---------------------------------------------------------------------------------------


def readahead(dev: str, sys: str = "/sys") -> tuple[int, str] | None:
    """``(kB, file)`` of the readahead window that governs faults on a file on device ``dev``.

    The file is named by where it really is. The partition case goes through ``..``, and
    ``os.path.normpath`` resolves that lexically -- ``/sys/dev/block/259:2/../queue`` becomes
    ``/sys/dev/block/queue``, a path that does not exist, which is what the startup line used to
    tell people to ``echo`` into.
    """
    for rel in (f"dev/block/{dev}/queue/read_ahead_kb",
                f"dev/block/{dev}/../queue/read_ahead_kb",
                f"class/bdi/{dev}/read_ahead_kb"):
        path = os.path.join(sys, rel)
        kb = _read_int(path)
        if kb is not None:
            return kb, _pretty_sysfs(os.path.realpath(path), sys)
    return None


def _pretty_sysfs(real: str, sys: str) -> str:
    """``/sys/devices/.../block/sdf/queue/read_ahead_kb`` -> ``/sys/block/sdf/queue/read_ahead_kb``
    when that shorter spelling reaches the same file."""
    m = re.search(r"/([^/]+)/queue/read_ahead_kb$", real)
    if m:
        short = os.path.join(sys, "block", m.group(1), "queue/read_ahead_kb")
        if os.path.exists(short) and os.path.realpath(short) == real:
            return short
    return real


def device_of(path: str) -> str | None:
    if not hasattr(os, "major"):
        return None
    try:
        st = os.stat(nearest_existing(path))
    except OSError:
        return None
    return f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"


def recommend_readahead_kb(widest_row_bytes: int) -> int:
    """The window to suggest for a bank whose widest per-expert block row is this many bytes.

    Measured optimum, two models: a quarter to a sixth of the widest row -- Flash-Next's is
    1600 kB and 256 won (512 and 128 each 3% slower, 2048 31%, 8192 49%, over 110 tokens);
    gpt-oss-120b's is 8100 kB and 2048 won (4096 5% slower, 1024 13%). This takes the geometric
    middle of that band, widest / sqrt(24), to the nearest power of two, which lands on both
    measured optima. Not a measurement for any other model.
    """
    kb = max(1.0, widest_row_bytes / 1024 / math.sqrt(24))
    return max(16, 2 ** round(math.log2(kb)))


def readahead_command(kb: int, where: str) -> str:
    return f"echo {kb} | sudo tee {where}"


def set_readahead(where: str, kb: int) -> tuple[bool, str]:
    """Write ``kb`` to a ``read_ahead_kb`` file. ``(True, "")`` or ``(False, why)``; never raises."""
    try:
        with open(where, "w", encoding="utf-8") as f:
            f.write(f"{int(kb)}\n")
        return True, ""
    except OSError as exc:
        return False, exc.strerror or str(exc)


# ---------------------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------------------


def meminfo(proc: str = "/proc") -> dict[str, int]:
    """``/proc/meminfo`` in bytes."""
    out = {}
    for line in (_read(os.path.join(proc, "meminfo")) or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return out


def memlock_limit(proc: str = "/proc") -> tuple[int | None, int | None] | None:
    """``(soft, hard)`` RLIMIT_MEMLOCK in bytes (None = unlimited), from ``/proc/self/limits``."""
    for line in (_read(os.path.join(proc, "self/limits")) or "").splitlines():
        if line.startswith("Max locked memory"):
            fields = line[len("Max locked memory"):].split()
            if len(fields) >= 2:
                vals = [None if v == "unlimited" else int(v) for v in fields[:2]]
                return vals[0], vals[1]
    return None


# Flash-Next under --pp-size 2 held 9 GiB of anonymous memory across both rank processes with
# the banks mapped (docs/bank-ram.md: "the rest of the process wants about 9 GiB"). The one
# host-side measurement there is, so it is taken per rank. A model with a large host-resident
# embedding or PLE table wants more; pass an explicit size then.
NONBANK_PER_RANK_BYTES = int(4.5 * GiB)
# What is left over becomes page cache, and the design leans on it: the non-resident rows a
# session keeps routing to settle there (effective disk share 3% instead of the placement's
# 12.5%). The measured 64 GB configuration (48G on a ~62 GiB MemTotal) left about 5% of
# MemTotal after the 9 GiB above, which is where this number comes from.
HEADROOM_FRACTION = 0.05
HEADROOM_MIN_BYTES = 2 * GiB


# The resident rows are cudaHostRegistered, and WSL2's CUDA caps page-locked host memory, shared by
# every process. MemAvailable alone does not see that cap: measured on the 2060 host (23.5 GiB),
# auto took 15.6 GiB, 141 of 240 resident blocks registered and the boot died in a CUDA allocation.
# The cap is not derivable from anything the guest can read -- freetoken.moe.pin_probe has the
# measurements and the reasoning -- so a figure measured on this host is preferred and the fraction
# below is only the fallback for a host that has never measured one. PIN_RESERVE_BYTES stays out of
# the budget for what else the server pins (a host embedding, staging buffers).
PIN_BUDGET_FRACTION = 0.4
PIN_RESERVE_BYTES = 2 * GiB


def _pin_source(proc: str = "/proc") -> str:
    from freetoken.moe.pin_probe import source

    return source(proc)


def pin_budget_bytes(mem: dict[str, int], proc: str = "/proc") -> int | None:
    """Page-locked host bytes the platform allows, or None where it does not cap them (plain Linux)."""
    env = os.environ.get("FREETOKEN_PIN_BUDGET_GB")
    if env:
        return int(float(env) * GiB)
    if not is_wsl(proc) or not mem.get("MemTotal"):
        return None
    from freetoken.moe.pin_probe import remembered

    known = remembered(proc)
    # only a refusal bounds anything (freetoken.moe.pin_probe)
    return known.cap_bytes if known else int(mem["MemTotal"] * PIN_BUDGET_FRACTION)


@dataclass
class AutoBankRam:
    total_bytes: int
    available: int
    mem_total: int
    nonbank: int
    headroom: int
    ranks: int
    pin_cap: int | None = None  # the pin budget less its reserve, when it is what decided

    def as_flag(self) -> str:
        """A ``--moe-bank-ram`` value parse_size reads back to (nearly) the same bytes."""
        return f"{self.total_bytes / GiB:.2f}G"

    def reason(self) -> str:
        text = (
            f"--moe-bank-ram auto: {self.total_bytes / GiB:.1f} GiB for the banks across "
            f"{self.ranks} rank{'s' if self.ranks != 1 else ''}"
        )
        by_ram = self.available - self.nonbank - self.headroom
        if self.pin_cap is not None and self.pin_cap < by_ram:
            text += (
                f" = the CUDA pin budget {(self.pin_cap + PIN_RESERVE_BYTES) / GiB:.1f} GiB "
                f"({_pin_source()}; 'ft doctor pin' measures this host, FREETOKEN_PIN_BUDGET_GB overrides) - "
                f"{PIN_RESERVE_BYTES / GiB:.1f} GiB for other pinned buffers, since the resident rows are "
                f"registered with CUDA; RAM alone would allow {by_ram / GiB:.1f} GiB"
            )
        else:
            text += (
                f" = MemAvailable {self.available / GiB:.1f} GiB - {self.nonbank / GiB:.1f} GiB for the "
                f"rest of the server ({NONBANK_PER_RANK_BYTES / GiB:.1f} per rank) - "
                f"{self.headroom / GiB:.1f} GiB left as page cache for the non-resident rows"
            )
        return text + "; pass a size to override"


def auto_bank_ram(mem: dict[str, int], ranks: int, proc: str = "/proc") -> AutoBankRam:
    """``--moe-bank-ram auto``: MemAvailable, less the rest of the server, less a page cache margin --
    and no more than the platform lets CUDA register, since the resident rows are registered.

    MemAvailable rather than MemTotal so that whatever else is running right now is left
    alone, and read once in the launcher so every rank splits the same number. The pin budget is
    host-wide too (shared by every process), so it caps the total, not each rank.
    """
    avail, total = mem.get("MemAvailable"), mem.get("MemTotal")
    if not avail or not total:
        raise ValueError("--moe-bank-ram auto: MemAvailable/MemTotal not readable; pass a size")
    ranks = max(1, int(ranks))
    nonbank = NONBANK_PER_RANK_BYTES * ranks
    headroom = max(HEADROOM_MIN_BYTES, int(total * HEADROOM_FRACTION))
    by_ram = avail - nonbank - headroom
    pin = pin_budget_bytes(mem, proc)
    # max(0, ...): a *measured* cap can be smaller than the reserve itself (1 GiB on Windows 10,
    # FreeToken-Kai#2), and a negative cap used to reach the message below as a negative size
    pin_cap = None if pin is None else max(0, pin - PIN_RESERVE_BYTES)
    budget = by_ram if pin_cap is None else min(by_ram, pin_cap)
    if budget < GiB:
        if pin_cap is not None and pin_cap < by_ram:
            # the pin cap is what binds, and freeing memory cannot raise it -- the old message
            # said "free some memory" to hosts with 22 GiB of it available
            raise ValueError(
                f"--moe-bank-ram auto: the CUDA pin budget is {pin / GiB:.2f} GiB, leaving "
                f"{pin_cap / GiB:.2f} GiB once {PIN_RESERVE_BYTES / GiB:.1f} GiB is kept for the server's "
                f"other pinned buffers -- too little to size the resident rows from, though RAM alone "
                f"would allow {by_ram / GiB:.1f} GiB. Freeing memory does not raise this cap "
                f"({_pin_source(proc)}); pass an explicit size, which maps the banks and page-locks what "
                f"it can, or serve every expert on the CPU with --moe-cpu-layers 1.0"
            )
        raise ValueError(
            f"--moe-bank-ram auto: MemAvailable is {avail / GiB:.1f} GiB"
            + (f" and the CUDA pin budget {pin / GiB:.1f} GiB" if pin is not None else "")
            + f", which leaves {budget / GiB:.1f} GiB for the banks after {nonbank / GiB:.1f} GiB for "
            f"the rest of the server and {headroom / GiB:.1f} GiB of page cache margin. Free some "
            f"memory or pass a size"
        )
    return AutoBankRam(budget, avail, total, nonbank, headroom, ranks, pin_cap)


# ---------------------------------------------------------------------------------------
# who has the file open
# ---------------------------------------------------------------------------------------


def mapped_by(path: str, proc: str = "/proc") -> tuple[list[tuple[int, str]], int]:
    """``([(pid, comm)], unreadable)``: processes that map ``path``, and how many could not be
    checked (other users' maps are not readable without privilege)."""
    target = os.path.realpath(path)
    found, unreadable = [], 0
    me = os.getpid() if proc == "/proc" else None
    try:
        pids = [int(p) for p in os.listdir(proc) if p.isdigit()]
    except OSError:
        return found, 0
    for pid in pids:
        if pid == me:
            continue
        try:
            with open(os.path.join(proc, str(pid), "maps"), encoding="utf-8", errors="replace") as f:
                hit = any(line.rstrip("\n").endswith(target) or line.rstrip("\n").endswith(target + " (deleted)")
                          for line in f)
        except PermissionError:
            unreadable += 1
            continue
        except OSError:
            continue  # exited meanwhile
        if hit:
            found.append((pid, _read(os.path.join(proc, str(pid), "comm")) or "?"))
    return found, unreadable


# ---------------------------------------------------------------------------------------
# the read benchmark
# ---------------------------------------------------------------------------------------


def physical_cores(sys: str = "/sys") -> int:
    """Distinct (package, core) pairs, which is what the CPU executor runs one thread per."""
    base = os.path.join(sys, "devices/system/cpu")
    cores = set()
    try:
        names = os.listdir(base)
    except OSError:
        names = []
    for name in names:
        if re.fullmatch(r"cpu\d+", name):
            topo = os.path.join(base, name, "topology")
            pkg, core = _read(os.path.join(topo, "physical_package_id")), _read(os.path.join(topo, "core_id"))
            if pkg is not None and core is not None:
                cores.add((pkg, core))
    return len(cores) or max(1, (os.cpu_count() or 2) // 2)


def random_row_read(path: str, row_bytes: int, threads: int, seconds: float) -> float:
    """GB/s (1e9) reading whole expert rows at random offsets from ``threads`` threads, O_DIRECT.

    The shape of a decode step's cold reads: whole rows, anywhere in the file, from every CPU
    worker at once. O_DIRECT so the page cache can neither answer nor be disturbed -- nothing
    is dropped and nothing is written. Raises OSError where the filesystem refuses O_DIRECT
    (tmpfs, 9p), because a buffered number there would be a measurement of RAM.
    """
    o_direct = getattr(os, "O_DIRECT", None)
    if o_direct is None:
        raise OSError("O_DIRECT is not available on this platform")
    size = os.path.getsize(path)
    row = max(4096, -(-int(row_bytes) // 4096) * 4096)
    if size < 2 * row:
        raise OSError(f"{path} is too small to benchmark ({size} bytes)")
    stop = time.perf_counter() + seconds
    done = [0] * threads
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            fd = os.open(path, os.O_RDONLY | o_direct)
        except OSError as exc:
            errors.append(exc)
            return
        buf = mmap.mmap(-1, row)  # page aligned, which O_DIRECT needs
        rnd = random.Random(i)
        try:
            while time.perf_counter() < stop:
                off = rnd.randrange(0, (size - row) // 4096) * 4096
                done[i] += os.preadv(fd, [buf], off)
        except OSError as exc:
            errors.append(exc)
        finally:
            buf.close()
            os.close(fd)

    started = time.perf_counter()
    pool = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    if errors and not sum(done):
        raise errors[0]
    return sum(done) / (time.perf_counter() - started) / 1e9


def fault_read(path: str, seconds: float, nbytes: int = 1 << 30, piece: int = 32 << 20) -> tuple[float, int]:
    """(GB/s, bytes) copying a range of ``path`` out of a mapping after dropping it from the page cache.

    The shape of a prefill chunk's unregistered rows: ``_staged_h2d`` copies each layer's rows
    past the resident prefix out of the bank mapping in file order, 32 MiB at a time, so every
    miss is a page fault that reads one readahead window and waits for it. The rate follows
    ``read_ahead_kb`` -- measured on an RTX 2060 host: 1.2 GiB/s at 8192 kB, 0.11 GiB/s with the
    window off -- which is why it is measured here rather than taken from the O_DIRECT number.

    Takes the range from the second half of the file (the non-resident side) and drops only
    that range, with POSIX_FADV_DONTNEED, which leaves pages another process has mapped alone.
    The caller checks that nobody has the file mapped, or the number would be a cache hit.
    """
    size = os.path.getsize(path)
    span = min(nbytes, size // 2) // mmap.PAGESIZE * mmap.PAGESIZE
    if span < piece:
        raise OSError(f"{path} is too small to benchmark ({size} bytes)")
    start = (size - span) // mmap.PAGESIZE * mmap.PAGESIZE
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, start, span, os.POSIX_FADV_DONTNEED)
        mm = mmap.mmap(fd, span, mmap.MAP_SHARED, mmap.PROT_READ, offset=start)
    finally:
        os.close(fd)
    try:
        out = bytearray(piece)
        view = memoryview(out)
        done = 0
        began = time.perf_counter()
        while done < span and time.perf_counter() - began < seconds:
            m = min(piece, span - done)
            view[:m] = mm[done:done + m]
            done += m
        elapsed = time.perf_counter() - began
    finally:
        mm.close()
    return done / elapsed / 1e9, done


def piece_read(path: str, seconds: float, nbytes: int = 2 << 30) -> tuple[float, str]:
    """(GB/s, how) reading a range of ``path`` the way a prefill chunk reads the non-resident rows:
    ``moe/bank_reader.DirectRangeReader`` with the server's thread count, piece size and mode
    (``FREETOKEN_BANK_PREAD``, buffered by default). A buffered read starts from a range dropped
    from the page cache, as the fault benchmark does, so it measures the disk."""
    from freetoken.moe.bank_reader import DirectRangeReader, _env_int, read_mode

    size = os.path.getsize(path)
    span = min(nbytes, size // 2) // 4096 * 4096
    reader = DirectRangeReader(path, threads=_env_int("FREETOKEN_BANK_READ_THREADS", 8),
                               piece_bytes=_env_int("FREETOKEN_BANK_READ_PIECE_MB", 16) << 20,
                               direct=read_mode(os.environ.get("FREETOKEN_BANK_PREAD")) == "direct")
    try:
        if span < reader.piece:
            raise OSError(f"{path} is too small to benchmark ({size} bytes)")
        start = (size - span) // 4096 * 4096
        if not reader.direct:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, start, span, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        done = 0
        began = time.perf_counter()
        step = reader.piece * reader.threads
        while done < span and time.perf_counter() - began < seconds:
            n = min(step, span - done)
            reader.read(start + done, n, lambda p, host, s: None)
            done += n
        elapsed = time.perf_counter() - began
        how = f"{reader.threads} threads x {reader.piece >> 20} MiB, {'O_DIRECT' if reader.direct else 'buffered'}"
    finally:
        reader.close()
    return done / elapsed / 1e9, how


# ---------------------------------------------------------------------------------------
# the startup complaint
# ---------------------------------------------------------------------------------------


@dataclass
class Storage:
    path: str
    mount: Mount | None
    fs_verdict: tuple[str, str]
    device: BlockDevice | None
    dev: str | None
    wsl: bool


def probe_storage(path: str, proc: str = "/proc", sys: str = "/sys") -> Storage:
    real = os.path.realpath(nearest_existing(path))
    mounts = parse_mountinfo(_read(os.path.join(proc, "self/mountinfo")) or "")
    mount = mount_of(real, mounts)
    verdict = filesystem_verdict(mount) if mount else ("ok", "unknown")
    # mountinfo's major:minor is the superblock's device, the same st_dev a stat of any file
    # on it returns -- and it comes from the proc root, so a fake tree can supply it
    dev = mount.dev if mount else device_of(real)
    device = block_device(dev, mount.source if mount else "", sys) if dev else None
    return Storage(real, mount, verdict, device, dev, is_wsl(proc))


# ---------------------------------------------------------------------------------------
# WSL2: the Windows drive under the virtual disk
# ---------------------------------------------------------------------------------------
#
# Inside WSL2, `df /` reports the free space of the ext4 filesystem in ext4.vhdx -- its declared
# size (1 TB by default), not the Windows drive the vhdx lives on. A dynamically growing vhdx takes
# host space as blocks are first written, so a 16-64 GiB bank can fill the Windows drive while
# `df /` still shows hundreds of GiB free; when it does, the VM stops. Space freed inside is not
# given back to Windows either, unless the vhdx is sparse. Measured once the hard way on the 2060
# host: C: reached zero and WSL went down mid-write.

_REG_LXSS = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Lxss"


@dataclass
class WslHostDisk:
    distro: str
    vhdx: str  # Windows path of the distro's virtual disk
    drive: str  # "C:"
    mount: str | None  # where that drive is mounted inside WSL (/mnt/c), None when it is not
    free_bytes: int | None  # of the Windows drive
    total_bytes: int | None
    vhdx_bytes: int | None = None
    sparse: bool | None = None  # None: not asked (it takes PowerShell)


def parse_reg_lxss(text: str, distro: str) -> tuple[str, str] | None:
    """``(BasePath, VhdFileName)`` of ``distro`` from ``reg.exe query`` of the Lxss key with ``/s``."""
    blocks, cur = [], {}
    for line in text.splitlines():
        if line.startswith("HKEY_"):
            cur = {}
            blocks.append(cur)
            continue
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[1].startswith("REG_"):
            cur[parts[0]] = parts[2].strip()
    for b in blocks:
        if b.get("DistributionName", "").lower() == distro.lower() and b.get("BasePath"):
            base = b["BasePath"]
            if base.startswith("\\\\?\\"):
                base = base[4:]
            return base, b.get("VhdFileName") or "ext4.vhdx"
    return None


def _windows_exe(name: str) -> str | None:
    import shutil

    for cand in (shutil.which(name), f"/mnt/c/Windows/System32/{name}",
                 f"/mnt/c/Windows/System32/WindowsPowerShell/v1.0/{name}"):
        if cand and os.path.exists(cand):
            return cand
    return None


def drive_mount(drive: str, proc: str = "/proc") -> str | None:
    """Where Windows drive ``C:`` is mounted inside WSL, from mountinfo (drvfs shows as 9p)."""
    letter = drive[:1].upper()
    for m in parse_mountinfo(_read(os.path.join(proc, "self/mountinfo")) or ""):
        if m.fstype in ("9p", "drvfs") and m.source.upper().startswith(f"{letter}:"):
            return m.mountpoint
    return None


def wsl_host_disk(proc: str = "/proc", *, attributes: bool = False, timeout: float = 10.0,
                  run=None) -> WslHostDisk | None:
    """The Windows drive holding this distro's ext4.vhdx and its free space, or None outside WSL2
    or when Windows interop cannot say. ``attributes`` also asks PowerShell whether the vhdx is
    sparse (about a second); the startup check does not. ``run``: a subprocess.run stand-in."""
    import subprocess

    if not is_wsl(proc):
        return None
    distro = os.environ.get("WSL_DISTRO_NAME")
    run = run or subprocess.run
    reg = _windows_exe("reg.exe")
    if not distro or not reg:
        return None
    try:
        out = run([reg, "query", _REG_LXSS, "/s"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    found = parse_reg_lxss(getattr(out, "stdout", "") or "", distro)
    if found is None:
        return None
    base, name = found
    vhdx = base.rstrip("\\") + "\\" + name
    drive = vhdx[:2] if vhdx[1:2] == ":" else ""
    mount = drive_mount(drive, proc) if drive else None
    free = total = size = None
    if mount:
        try:
            st = os.statvfs(mount)
            free, total = st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize
        except OSError:
            pass
        try:
            size = os.stat(os.path.join(mount, *vhdx[3:].split("\\"))).st_size
        except OSError:
            pass
    host = WslHostDisk(distro, vhdx, drive, mount, free, total, size)
    ps = _windows_exe("powershell.exe") if attributes else None
    if ps:
        try:
            got = run([ps, "-NoProfile", "-NonInteractive", "-Command",
                       f"(Get-Item -LiteralPath '{vhdx}').Attributes.ToString()"],
                      capture_output=True, text=True, timeout=timeout)
            attrs = (getattr(got, "stdout", "") or "").strip()
            if attrs:
                host.sparse = "SparseFile" in attrs
        except (OSError, subprocess.SubprocessError):
            pass
    return host


def host_space_warning(path: str, need_bytes: int, proc: str = "/proc", host=None) -> str | None:
    """Under WSL2, a warning when writing ``need_bytes`` to ``path`` inside the virtual disk could
    fill the Windows drive under it; None when it fits, outside WSL2, or on a drvfs path (whose
    own free space is already the Windows drive's)."""
    if need_bytes <= 0 or not is_wsl(proc):
        return None
    mounts = parse_mountinfo(_read(os.path.join(proc, "self/mountinfo")) or "")
    m = mount_of(os.path.realpath(nearest_existing(path)), mounts)
    if m is None or m.fstype in ("9p", "drvfs", "tmpfs", "ramfs"):
        return None
    host = host if host is not None else wsl_host_disk(proc)
    if host is None or host.free_bytes is None:
        return None
    if need_bytes + HOST_FREE_FLOOR_BYTES <= host.free_bytes:
        return None
    return (
        f"--moe-bank-ram: writing {need_bytes / GiB:.1f} GiB into {path} can grow this WSL2 distro's "
        f"virtual disk ({host.vhdx}) by that much, and {host.drive} has {host.free_bytes / GiB:.1f} GiB "
        f"free -- `df /` inside WSL shows the virtual disk's size, not the drive's. When the drive "
        f"fills, WSL stops mid-write; space freed inside later is not returned to Windows. Free space "
        f"on {host.drive} or move the bank with --moe-bank-dir"
    )


# A margin under which the Windows drive is treated as full: the vhdx grows in 32 MiB blocks,
# its metadata and Windows itself need room, and swap may live on the same drive.
HOST_FREE_FLOOR_BYTES = 20 * GiB


def storage_warnings(path: str, proc: str = "/proc", sys: str = "/sys") -> list[str]:
    """One line per thing about where ``path`` is that makes --moe-bank-ram slow. Empty = nothing
    known to be wrong (not the same as measured to be right)."""
    if not os.path.isdir(os.path.join(proc, "self")):
        return []
    s = probe_storage(path, proc, sys)
    out = []
    level, why = s.fs_verdict
    if level != "ok":
        where = os.path.abspath(os.path.expanduser(path))
        out.append(f"--moe-bank-ram: the bank directory {where} is on {why}. Move it with --moe-bank-dir")
        return out  # the device under a 9p or network mount is not the one that matters
    d = s.device
    if d is None or d.virtual:
        return out  # WSL2 / VM disks say nothing true about the drive behind them
    if d.transport == "usb":
        out.append(
            f"--moe-bank-ram: the bank file is on a USB disk ({d.name}); decode reads expert rows "
            f"from it at random every token. Put it on an internal NVMe with --moe-bank-dir"
        )
    elif d.rotational:
        out.append(
            f"--moe-bank-ram: the bank file is on a rotating disk ({d.name}); random row reads "
            f"there cost seconds per token. Put it on an NVMe with --moe-bank-dir"
        )
    elif d.transport == "sata":
        out.append(
            f"--moe-bank-ram: the bank file is on a SATA device ({d.name}); measured, a SATA SSD "
            f"adds about 300 ms per decode step even at 64 GB. An NVMe is what this was built for"
        )
    return out


def refuses_new_bank(path: str, proc: str = "/proc", sys: str = "/sys") -> str | None:
    """Why a bank file must not be created under ``path``, or None.

    Only the filesystems the verdict calls bad (9p/drvfs, network, tmpfs, FAT): creating the file
    there costs what the flag exists to save or fills a drive the server cannot see -- a 16-64 GiB
    file on a WSL2 /mnt/c path lands on the Windows drive through a file server, at a speed no
    decode survives. A file somebody already put there is served, with the warning.
    """
    if not os.path.isdir(os.path.join(proc, "self")):
        return None
    s = probe_storage(path, proc, sys)
    level, why = s.fs_verdict
    if level != "bad":
        return None
    return (
        f"--moe-bank-ram: not creating a bank file under {os.path.abspath(os.path.expanduser(path))}: it "
        f"is on {why}. Point --moe-bank-dir at a local Linux filesystem (under WSL2, somewhere in the "
        f"distro, such as the default ~/.cache/freetoken/bankmap)"
    )
