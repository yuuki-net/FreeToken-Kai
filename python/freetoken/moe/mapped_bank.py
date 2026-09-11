"""Expert banks as one file-backed mapping per rank, with the resident rows locked down.

The explicit two-tier arrangement (compacted banks plus a cold file read on demand) ran into
``OffloadMoeCache``: its banks are ``[num_experts, ...]`` by contract, and prefill streams a
whole layer by that count. Shrinking the bank means changing the copy machinery and the
prefill path -- the hot paths -- for every quant format.

This does the same job without touching any of that. Each rank writes its expert banks once
to a file, rows reordered so the frequently routed experts come first, then maps the file and
hands the cache ``[num_experts, ...]`` tensor views. The shape the cache sees never changes.
What changes is which rows are guaranteed to be in RAM:

* rows ``[0, hot)`` of every block are ``mlock``ed, so the prefill sweep -- which touches
  every expert of every layer on each chunk -- cannot evict them, and they are
  ``cudaHostRegister``ed so the PCIe fetch path can still DMA from them;
* rows ``[hot, num_experts)`` are ordinary file-backed pages. They fault in when routing
  reaches them and the kernel drops them again under pressure, with no writeback, because the
  mapping is read-only.

So the RAM the banks hold is the locked prefix, and the renumbering (bank_disk.plan_placement)
is what makes that prefix the experts worth keeping. Measured on Flash-Next: 77% of rows
resident serves 87.5% of routed accesses on unseen work (docs/bank-ram.md).

Cost: the file is a full copy of the banks (31.7 GiB per rank for Flash-Next), written once
and reused for as long as the placement is unchanged.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import struct
import threading
import time
import warnings

from freetoken.utils.torch_utils import clear_cuda_error

MAGIC = b"FTMB"
VERSION = 1


# A block this small per layer is kept whole rather than split hot/cold. The global-scale
# blocks are 2.5 and 5 kB per expert row, so a fault on one reads a whole readahead window
# (256 kB, the measured optimum) to use a few kilobytes -- about 13% of the I/O a cold expert
# costs, for two blocks that fit entirely in 17 MB per rank. Residency is cheaper than
# reading them.
_WHOLE_BLOCK_BYTES = 4 * 2**20

ALIGN = 4096
_PREAMBLE = struct.Struct("<4sIQ")


def _rss_gib() -> float:
    """This process's resident set, for the one place where "did that free?" is the question."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


def _align_up(n: int, a: int = ALIGN) -> int:
    return -(-n // a) * a


def readahead_kb(path: str) -> "tuple[int, str] | None":
    """The device readahead window that applies to ``path``, and where it was read from.

    Worth surfacing because it is invisible, system-wide, and worth 2x: measured on the
    3060 host at 64 GB with MADV_SEQUENTIAL, decode ran 11.5 tok/s at 256 kB and 7.9 at
    2048. Larger is worse, and not by a little. An expert row is six blocks of 1600, 800,
    200, 100, 5 and 2.5 kB, so a window sized for the largest reads mostly other experts'
    rows -- and those are the tail of the frequency order, the least likely to be wanted.
    """
    if not hasattr(os, "major"):  # Windows: no device numbers, no sysfs
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    dev = f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"
    for rel in (f"/sys/dev/block/{dev}/queue/read_ahead_kb",
                f"/sys/dev/block/{dev}/../queue/read_ahead_kb"):
        try:
            with open(rel) as fh:
                return int(fh.read().strip()), os.path.normpath(rel)
        except (OSError, ValueError):
            continue
    return None


def advise_range(start: int, length: int, limit: int) -> tuple[int, int] | None:
    """The page-aligned sub-range of [start, start+length) inside [0, limit), or None.

    madvise(2) rejects an unaligned address outright, and a block's resident prefix is a row
    count times a row size -- 387 rows of 2560 B lands mid-page. Rounding the start up rather
    than down keeps the hint off the neighbouring region, which matters because the two hints
    here are opposites: WILLNEED on the resident prefix, RANDOM on what follows it.
    """
    begin = _align_up(max(0, start))
    end = min(limit, start + length)
    if end <= begin:
        return None
    return begin, end - begin


def _dtype_name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _dtype_of(name: str):
    import torch

    dt = getattr(torch, name, None)
    if dt is None:
        raise ValueError(f"unknown dtype {name!r} in mapped bank header")
    return dt


class MappedBankLayout:
    """Where every (bank, layer) block sits in the file, and which rows are resident."""

    def __init__(self, num_experts: int, hot_per_layer: int, layers, banks, order):
        self.num_experts = int(num_experts)
        self.hot_per_layer = int(hot_per_layer)
        self.layers = [int(x) for x in layers]
        # [(name, row shape, dtype name, bytes per row)]
        self.banks = [(str(n), tuple(int(x) for x in s), str(d), int(b)) for n, s, d, b in banks]
        self.order = {int(k): [int(x) for x in v] for k, v in order.items()}
        self._block = {
            name: _align_up(self.num_experts * row_bytes) for name, _, _, row_bytes in self.banks
        }
        head = _PREAMBLE.size + len(self._header_json())
        self.data_offset = _align_up(head)

    def _header_json(self) -> bytes:
        return json.dumps({
            "version": VERSION,
            "num_experts": self.num_experts,
            "hot_per_layer": self.hot_per_layer,
            "layers": self.layers,
            "banks": [[n, list(s), d, b] for n, s, d, b in self.banks],
            "order": {str(k): v for k, v in self.order.items()},
        }, separators=(",", ":")).encode("utf-8")

    def block_bytes(self, name: str) -> int:
        return self._block[name]

    def offset_of(self, name: str, layer: int) -> int:
        """Byte offset of one (bank, layer) block. Blocks are grouped by bank, then layer."""
        pos = self.data_offset
        for bank_name, _, _, _ in self.banks:
            for other in self.layers:
                if bank_name == name and other == layer:
                    return pos
                pos += self._block[bank_name]
        raise KeyError((name, layer))

    def total_bytes(self) -> int:
        return self.data_offset + sum(
            self._block[name] * len(self.layers) for name, _, _, _ in self.banks
        )

    def header_blob(self) -> bytes:
        body = self._header_json()
        head = _PREAMBLE.pack(MAGIC, VERSION, len(body)) + body
        return head + b"\0" * (self.data_offset - len(head))

    def same_as(self, other: "MappedBankLayout") -> bool:
        """Layout equality including the row order.

        A file built from a different histogram puts different experts in the resident
        prefix while every shape and length still matches, so reusing it would feed the
        model weights it never asked for and look healthy doing it.
        """
        return (
            self.num_experts == other.num_experts
            and self.hot_per_layer == other.hot_per_layer
            and self.layers == other.layers
            and self.banks == other.banks
            and self.order == other.order
        )

    @classmethod
    def from_json(cls, d: dict) -> "MappedBankLayout":
        return cls(
            d["num_experts"], d["hot_per_layer"], d["layers"],
            [(n, tuple(s), t, b) for n, s, t, b in d["banks"]],
            {int(k): v for k, v in d["order"].items()},
        )

    @classmethod
    def read(cls, path: str) -> "MappedBankLayout":
        with open(path, "rb") as f:
            magic, version, body_len = _PREAMBLE.unpack(f.read(_PREAMBLE.size))
            if magic != MAGIC:
                raise ValueError(f"{path}: not a mapped bank file")
            if version != VERSION:
                raise ValueError(f"{path}: mapped bank version {version}, expected {VERSION}")
            return cls.from_json(json.loads(f.read(body_len).decode("utf-8")))


def layout_from(sources, layers, placement) -> MappedBankLayout:
    """Layout for banks a loader is about to produce. ``sources``: {name: [tensor|HostBank]}."""
    banks = []
    for name, per_layer in sources.items():
        t = getattr(per_layer[0], "tensor", per_layer[0])
        row = t[0]
        banks.append((name, tuple(row.shape), _dtype_name(t.dtype), row.numel() * t.element_size()))
    return MappedBankLayout(
        placement.num_experts, placement.hot_per_layer, layers, banks,
        {layer: placement.order[layer] for layer in layers},
    )


class MappedBankWriter:
    """Writes permuted per-layer blocks. Layers may arrive in any order, from any thread."""

    def __init__(self, path: str, layout: MappedBankLayout):
        self.path, self.layout = path, layout
        self._f = open(path, "wb")
        self._f.write(layout.header_blob())
        self._f.truncate(layout.total_bytes())
        self._lock = threading.Lock()
        self._done: set[int] = set()

    def write_layer(self, layer: int, banks) -> None:
        """``banks``: {name: tensor or HostBank} for one layer, rows in LOGICAL order."""
        import torch

        idx = torch.as_tensor(self.layout.order[layer], dtype=torch.long)
        for name, _, _, _ in self.layout.banks:
            src = getattr(banks[name], "tensor", banks[name])
            # permute out of line, then one sequential write per block
            block = src.index_select(0, idx).contiguous()
            raw = block.flatten().view(torch.uint8).numpy()
            with self._lock:
                self._f.seek(self.layout.offset_of(name, layer))
                self._f.write(memoryview(raw))
        with self._lock:
            self._done.add(layer)

    def close(self) -> None:
        missing = [x for x in self.layout.layers if x not in self._done]
        self._f.close()
        if missing:
            # an unwritten block reads as zeros, and a model fed zeroed experts produces
            # fluent nonsense rather than failing, so refuse the file outright
            os.unlink(self.path)
            raise ValueError(f"{self.path}: layers {missing} were never written")


class MappedBanks:
    """An open mapped bank file: tensor views, resident prefix locked and registered."""

    def __init__(self, path: str, register: bool = True):
        import torch

        # FREETOKEN_BANK_REGISTER: prefix (default) registers only the locked rows; "all"
        # registers the whole mapping, which pins every row and gives up the RAM saving --
        # a diagnostic for whether the GPU paths need every row device-addressable; "none"
        # registers nothing, which makes the banks read as unregistered host memory.
        # Default "none". Registering only helped the PCIe fetch path, which a mapped bank
        # cannot use anyway (attach() sends every miss to the CPU), and a range that is
        # registered for part of its length and not the rest is what made the prefill
        # prefetch fail with cudaErrorInvalidValue.
        self.register_mode = os.environ.get("FREETOKEN_BANK_REGISTER", "prefix").strip().lower()
        if self.register_mode not in ("prefix", "all", "none"):
            raise ValueError(
                f"FREETOKEN_BANK_REGISTER={self.register_mode!r}: expected prefix, all or none"
            )
        if self.register_mode == "none":
            register = False
        self.path = path
        self.layout = MappedBankLayout.read(path)
        self._fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        # ACCESS_READ, and the read-only part is load-bearing. ACCESS_COPY maps
        # MAP_PRIVATE *writable*, and mlock populates a writable private mapping with
        # FOLL_WRITE -- which copy-on-writes every locked page into anonymous memory. That
        # turned 24 GiB of shared clean page cache into 24 GiB of private dirty pages per
        # rank, and the host went to swap. Read-only keeps the locked pages as page cache:
        # resident, shared with the file, and droppable without writeback once unlocked.
        self._map = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True) if os.name == "posix" else None
        self._registered: list[int] = []
        self.locked_bytes = 0
        self.requested_bytes = 0  # what the placement wanted locked; locked_bytes is what the OS gave
        self.lock_errno = 0  # last mlock failure, 0 when every lock succeeded
        self.whole_blocks = 0
        self.registered_bytes = 0
        self.sources: dict[str, list] = {}
        # FREETOKEN_BANK_PRELOAD: pull the non-resident rows into the page cache too. They
        # are not locked, so the kernel may still drop them -- this only says how much of
        # the decode cost is faulting them back in, by removing that cost on a host with
        # room to keep the whole file cached.
        #
        # By touching, not by advising. MADV_WILLNEED is a hint: it returns immediately,
        # queues what readahead it feels like, and is free to do nothing under memory
        # pressure. The first attempt used it and the page cache did not move (53 GiB before
        # and after, settle unchanged at 12 s). One byte per page through the mapping is not
        # a hint.
        preload = os.environ.get("FREETOKEN_BANK_PRELOAD", "").strip() not in ("", "0")
        # FREETOKEN_BANK_ADVISE: what the fault path is told about the non-resident rows.
        #
        # Default normal, and that is a considered default rather than a leftover.
        #
        # The fault path already uses the device's read_ahead_kb without being told anything
        # -- measured, by lowering read_ahead_kb from 8192 to 256 with no madvise at all and
        # watching decode go from 5.9 to 15 tok/s at 64 GB. Whatever VM_SEQ_READ would add
        # to the window is on top of a window that already works.
        #
        # And MADV_SEQUENTIAL does not only widen the window: it also drops pages behind the
        # read. That is precisely wrong here. The non-resident rows a session actually routes
        # to accumulate in what page cache is left and stop being disk reads at all -- it is
        # why a long generation settles at 16-17 tok/s where its first 300 tokens ran at 11.
        # Freeing them behind the reader would throw that away.
        advise = os.environ.get("FREETOKEN_BANK_ADVISE", "normal").strip().lower()
        if advise not in ("sequential", "normal", "random"):
            raise ValueError(
                f"FREETOKEN_BANK_ADVISE={advise!r}: expected sequential, normal or random"
            )
        hint = {
            "sequential": getattr(mmap, "MADV_SEQUENTIAL", None),
            "normal": getattr(mmap, "MADV_NORMAL", None),
            "random": getattr(mmap, "MADV_RANDOM", None),
        }[advise]
        self.advise_mode = advise
        started = time.perf_counter()
        with warnings.catch_warnings():
            # torch warns that a read-only buffer cannot be written through. Nothing writes
            # to a bank -- and if anything did, the COW that warning is about is exactly
            # what the read-only mapping exists to prevent.
            warnings.filterwarnings("ignore", message=".*buffer is not writable.*")
            buf = torch.frombuffer(self._map, dtype=torch.uint8)
        # via the tensor, not ctypes.from_buffer: that one insists on a writable buffer,
        # which is the property being avoided here
        self._base = buf.data_ptr()
        for name, row_shape, dtype_name, row_bytes in self.layout.banks:
            dtype = _dtype_of(dtype_name)
            per_layer = []
            for layer in self.layout.layers:
                off = self.layout.offset_of(name, layer)
                span = self.layout.num_experts * row_bytes
                per_layer.append(
                    buf[off:off + span].view(dtype).reshape(self.layout.num_experts, *row_shape)
                )
                lock_rows = (
                    self.layout.num_experts
                    if self.register_mode == "all" or span <= _WHOLE_BLOCK_BYTES
                    else self.layout.hot_per_layer
                )
                self.whole_blocks += span <= _WHOLE_BLOCK_BYTES
                self._settle(off, lock_rows * row_bytes, span, register)
                self._advise(hint, off, span, len(self._map))
                if preload:
                    self._touch(buf, off, span)
            self.sources[name] = per_layer
        self.settle_seconds = time.perf_counter() - started
        self.preloaded = preload

    @staticmethod
    def _touch(buf, offset: int, span: int) -> None:
        """Fault one block in, a byte per page. The sum is only there to make the reads
        happen -- the value is discarded."""
        import torch

        page = 4096
        view = buf[offset:offset + span:page]
        if view.numel():
            int(view.to(torch.int64).sum())

    def _settle(self, offset: int, nbytes: int, block_bytes: int, register: bool) -> None:
        """Fault in and lock the resident prefix of one block.

        WILLNEED before mlock, because mlock faults in every page of its range and would
        otherwise do it one page at a time -- 24 GiB of prefix took eight minutes that way.

        Nothing is advised about the non-resident remainder. MADV_RANDOM was tried there and
        is wrong twice over: prefill streams a whole layer, so that region is read
        sequentially and end to end on every chunk, and even a decode fault wants the rest of
        its 2.6 MiB expert row. Both want readahead; the default heuristic gives it.
        """
        import torch

        limit = len(self._map)
        if nbytes:
            self._advise(getattr(mmap, "MADV_WILLNEED", None), offset, nbytes, limit)
            addr = self._base + offset
            self.requested_bytes += nbytes
            if self._libc is None:
                pass
            elif self._libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)) == 0:
                self.locked_bytes += nbytes
            else:
                # RLIMIT_MEMLOCK, near certainly: systemd's default is MAX(64M, RAM/8), so a
                # 64 GB host stops at 8 GiB of the 24 GiB --moe-bank-ram asks for and the rest
                # is evictable page cache. Silence here cost a bug report: the summary said
                # "7.8 GiB locked resident" against a 24 GiB budget and explained nothing.
                self.lock_errno = ctypes.get_errno()
        if not nbytes:
            return
        addr = self._base + offset
        if not register:
            return
        # Registering the locked prefix is what keeps the GPU paths usable at all: without
        # it every layer is LOCKED, which means every decode miss goes to the CPU executor --
        # and that bypasses the VRAM expert cache entirely, throwing away its ~56% hit rate.
        #
        # flags=0 asks for read-write pinning, which a file mapping cannot give;
        # cudaHostRegisterReadOnly (0x08) is the flag for exactly this case. Ask for it first
        # and keep plain as the fallback for a driver that does not know the flag.
        #
        # Every failed attempt leaves its error in this thread's slot, and the next kernel
        # launch of any size reports it -- a 48 KB tensor in OffloadMoeCache, in practice,
        # with the registration nowhere in the traceback. Clear it after each failure, and
        # clear it with cudaGetLastError: torch.cuda.synchronize() does NOT reset that slot
        # (measured), which is why a host that refuses BOTH flags still died here.
        #
        # A host can refuse both. cudaHostRegisterReadOnly needs
        # cudaDevAttrHostRegisterReadOnlySupported, which is 0 on some driver/platform
        # combinations (an RTX 3060 pair on native Ubuntu with the open kernel module, in the
        # report this was found in), and flags=0 then fails with cudaErrorInvalidValue because
        # read-write pinning is what a read-only file mapping cannot give. Registering nothing
        # is a supported state -- attach() sends every miss to the CPU executor -- so this
        # must return quietly, not poison the context.
        cudart = torch.cuda.cudart()
        for flags in (0x08, 0):
            if int(cudart.cudaHostRegister(addr, nbytes, flags)) == 0:
                self.registered_bytes += nbytes
                self._registered.append(addr)
                return
            clear_cuda_error()

    def _advise(self, option, start: int, length: int, limit: int) -> None:
        if option is None or length <= 0:
            return
        span = advise_range(start, length, limit)
        if span is None:
            return
        try:
            self._map.madvise(option, span[0], span[1])
        except OSError:
            pass  # advisory; a kernel that refuses the hint is not a reason to fail startup

    def close(self) -> None:
        import torch

        for addr in self._registered:
            torch.cuda.cudart().cudaHostUnregister(addr)
        self._registered.clear()
        self.sources.clear()
        os.close(self._fd)


class MappedTier:
    """Assembles a ``--moe-bank-ram`` run: write the mapped bank, open it, hand over views.

    Attaches to the expert loader's per-layer sink. Writing per layer as the checkpoint
    streams in is not an optimization: holding the full banks and a second copy at once needs
    more RAM than the host that wants this feature has, so the overshoot has to stay at one
    layer (~1.3 GiB for Flash-Next).

    When the file on disk already describes exactly this placement the write is skipped --
    it is 31.7 GiB per rank, and a start that only changed the port should not pay it.
    """

    def __init__(self, placement, path: str, layers, log=None, warn=None):
        self.placement = placement
        self.path = path
        self.layers = list(layers)
        self.log = log or (lambda _msg: None)
        self.warn = warn or self.log
        self._writer: MappedBankWriter | None = None
        self._layout: MappedBankLayout | None = None
        self._reuse = False
        self._seen: set[int] = set()
        self._lock = threading.Lock()
        self.banks: MappedBanks | None = None

    def _prepare(self, sample) -> None:
        """Decide reuse-or-write on the first layer, when the bank shapes are finally known."""
        layout = layout_from({k: [v] for k, v in sample.items()}, self.layers, self.placement)
        self._layout = layout
        try:
            self._reuse = MappedBankLayout.read(self.path).same_as(layout)
        except (OSError, ValueError):
            self._reuse = False
        if self._reuse:
            self.log(f"--moe-bank-ram: reusing {self.path} (placement unchanged)")
            return
        self.log(
            f"--moe-bank-ram: writing {layout.total_bytes() / 2**30:.1f} GiB to {self.path} "
            f"({layout.hot_per_layer}/{layout.num_experts} experts resident per layer)"
        )
        self._writer = MappedBankWriter(self.path, layout)

    def sink(self, layer_id: int, banks) -> None:
        """``layer_sink`` for the expert loader."""
        with self._lock:
            if self._layout is None:
                self._prepare(banks)
        if self._writer is not None:
            self._writer.write_layer(layer_id, banks)
        with self._lock:
            self._seen.add(layer_id)
        for bank in banks.values():
            release = getattr(bank, "release", None)
            if release is not None:
                release()  # caps peak RAM at one layer's worth of the original banks

    def finish(self) -> None:
        missing = [x for x in self.layers if x not in self._seen]
        if missing:
            raise RuntimeError(f"--moe-bank-ram: layers {missing} never reached the sink")
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.log(f"--moe-bank-ram: source banks released, RSS now {_rss_gib():.1f} GiB")
        self.log("--moe-bank-ram: faulting in and locking the resident rows")
        self.banks = MappedBanks(self.path)
        b = self.banks
        ra = readahead_kb(self.path)
        if ra is not None:
            kb, where = ra
            # Compare against this model's own geometry, not a constant. A window wider
            # than the widest block row cannot finish inside one expert's row, so the tail
            # of every fault lands in a neighbouring expert -- and the non-resident rows are
            # the tail of the frequency order, the ones least likely to be wanted next.
            #
            # The blocks differ by three orders of magnitude between models: Flash-Next's
            # widest row is 1600 kB (best measured window 256), gpt-oss-120b's is 7.91 MiB.
            # A threshold tuned on one is wrong on the other, and was: 1024 drew this
            # warning on gpt-oss while measuring 17% faster than 256.
            widest_kb = max(rb for _, _, _, rb in b.layout.banks) // 1024
            note = f"--moe-bank-ram: device readahead {kb} kB ({where})"
            if kb > max(widest_kb, 1):
                self.log(
                    f"{note} -- wider than this model's widest expert-row block "
                    f"({widest_kb} kB), so every fault on a non-resident row spills into "
                    f"experts nobody asked for; try {widest_kb} or less: echo N > {where}"
                )
            else:
                self.log(note)
        self.log(
            f"--moe-bank-ram: mapped {b.layout.total_bytes() / 2**30:.1f} GiB, "
            f"{b.locked_bytes / 2**30:.1f} GiB locked resident, "
            f"{b.registered_bytes / 2**30:.1f} GiB registered for PCIe"
            + (f", {b.whole_blocks} small blocks kept whole" if b.whole_blocks else "")
            + f", {b.advise_mode} fault readahead"
            + (", whole file preloaded" if b.preloaded else "")
            + f" ({b.settle_seconds:.0f} s)"
        )
        if b.lock_errno:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            inf = resource.RLIM_INFINITY
            shown = "unlimited" if soft == inf else f"{soft / 2**30:.2f} GiB"
            self.warn(
                f"--moe-bank-ram: only {b.locked_bytes / 2**30:.1f} of the "
                f"{b.requested_bytes / 2**30:.1f} GiB this rank asked to keep resident is "
                f"locked -- mlock: {os.strerror(b.lock_errno)}. RLIMIT_MEMLOCK is {shown} "
                f"per process (`ulimit -l`); the rest stays evictable page cache and comes "
                f"back from disk under pressure. Raise the limit to at least "
                f"{b.requested_bytes / 2**30:.0f} GiB or lower --moe-bank-ram"
            )


    @property
    def sources(self) -> dict:
        assert self.banks is not None, "call finish() first"
        return self.banks.sources

    def attach(self, cache, device) -> None:
        """Hand over the renumbering, and keep the GPU off the rows it cannot address."""
        from freetoken.moe.bank_disk import permutation_tensor

        cache.expert_perm = [
            permutation_tensor(self.placement, layer, device=device) for layer in self.layers
        ]
        if self.banks is None:
            return
        if self.banks.registered_bytes:
            # Only rows [0, hot) are cudaHostRegistered; the rest is host memory the device
            # has no address for, and a GPU fetch of one is an illegal access inside the
            # decode graph (which is how this was found). Telling the cache where the
            # registered prefix ends lets the hybrid path keep its PCIe fetches inside it and
            # send the remaining misses to the CPU -- instead of the whole layer going to the
            # CPU, which is what LOCKED layers do and what costs the VRAM cache.
            cache.prefix_pinned_rows = self.banks.layout.hot_per_layer
            self.log(
                f"--moe-bank-ram: PCIe fetches restricted to the resident "
                f"{self.banks.layout.hot_per_layer}/{self.banks.layout.num_experts} experts "
                f"per layer; the rest decode on the CPU"
            )
            return
        # Nothing registered: no row has a device address, so every miss goes to the CPU.
        # BOTH knobs have to go to zero -- ensure_experts_hybrid fetches ``hybrid_max_fetch``
        # misses, OR ``~hybrid_fetch_fraction * misses`` when the fraction is set, and
        # --moe-hybrid-max-fetch auto sets the fraction.
        had = (getattr(cache, "hybrid_max_fetch", 0), getattr(cache, "hybrid_fetch_fraction", 0.0))
        cache.hybrid_max_fetch = 0
        cache.hybrid_fetch_fraction = 0.0
        if any(had):
            self.log(
                f"--moe-bank-ram: every decode miss goes to the CPU executor "
                f"(was max_fetch={had[0]}, fraction={had[1]:.3f}); the GPU cannot address "
                f"the non-resident rows"
            )
