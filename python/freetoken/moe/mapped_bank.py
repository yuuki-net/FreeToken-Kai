"""Expert banks as a file-backed mapping, with the resident rows locked down.

The explicit two-tier arrangement (compacted banks plus a cold file read on demand) ran into
``OffloadMoeCache``: its banks are ``[num_experts, ...]`` by contract, and prefill streams a
whole layer by that count. Shrinking the bank means changing the copy machinery and the
prefill path -- the hot paths -- for every quant format.

This does the same job without touching any of that. The expert banks live in one file
(moe/bank_file.py), rows reordered so the frequently routed experts come first; each rank maps
the file and hands the cache ``[num_experts, ...]`` tensor views of its own layers. The shape
the cache sees never changes. What changes is which rows are guaranteed to be in RAM:

* rows ``[0, hot)`` of every block are held resident and ``cudaHostRegister``ed, so the
  prefill sweep -- which touches every expert of every layer on each chunk -- cannot evict
  them and the PCIe fetch path can still DMA from them. Which form that takes depends on
  what the device will register: page cache held down with ``mlock`` where a read-only file
  mapping can be registered, private anonymous copies where it cannot (``_pick_map_mode``);
* rows ``[hot, num_experts)`` are ordinary file-backed pages. They fault in when routing
  reaches them and the kernel drops them again under pressure, with no writeback, because the
  mapping is read-only.

So the RAM the banks hold is the locked prefix, and the renumbering (bank_disk.plan_placement)
is what makes that prefix the experts worth keeping. Measured on Flash-Next: 77% of rows
resident serves 87.5% of routed accesses on unseen work (docs/bank-ram.md).

``hot`` is decided per run from ``--moe-bank-ram``; the file does not depend on it, nor on the
layer split. When the file already holds this rank's layers the checkpoint's expert tensors are
not read at all (``MappedTier.prepare``); a checkpoint packed with ``ft bank pack`` has none to
read.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
import time
import warnings

from freetoken.utils.torch_utils import clear_cuda_error

from .bank_file import (  # noqa: F401  (re-exported: the on-disk format lives in bank_file)
    ALIGN,
    BankFile,
    BankFileError,
    MappedBankLayout,
    dtype_of as _dtype_of,
    file_lock,
    free_bytes,
    layout_from_sample,
)


# A block this small per layer is kept whole rather than split hot/cold. The global-scale
# blocks are 2.5 and 5 kB per expert row, so a fault on one reads a whole readahead window
# (256 kB, the measured optimum) to use a few kilobytes -- about 13% of the I/O a cold expert
# costs, for two blocks that fit entirely in 17 MB per rank. Residency is cheaper than
# reading them.
_WHOLE_BLOCK_BYTES = 4 * 2**20


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
    from freetoken.moe import disk_probe

    if not hasattr(os, "major"):  # Windows: no device numbers, no sysfs
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return disk_probe.readahead(f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}")


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


class MappedBanks:
    """An open mapped bank file: tensor views, resident prefix locked and registered.

    ``layers``: the global bank-layer ids to hand out, in order (default: every committed
    layer). Each rank of a layer split maps the whole file and touches only its own blocks.
    ``hot_per_layer``: rows ``[0, hot)`` of each block are the resident prefix (default: all).
    """

    def __init__(self, path: str, register: bool = True, layers=None, hot_per_layer: int | None = None):
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
        with BankFile.open(path, writable=False) as bank_file:
            self.layout = bank_file.layout
            present = set(bank_file.present_layers())
        self.layers = sorted(present) if layers is None else [int(x) for x in layers]
        absent = [x for x in self.layers if x not in present]
        if absent:
            raise BankFileError(f"{path}: layers {absent} are not committed in the file")
        self.hot_per_layer = (
            self.layout.num_experts if hot_per_layer is None
            else max(0, min(int(hot_per_layer), self.layout.num_experts))
        )
        self.mapped_bytes = len(self.layers) * self.layout.layer_bytes()
        self._fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True) if os.name == "posix" else None
        self._buf = None
        # FREETOKEN_BANK_MAP: shared keeps the read-only file mapping, private copies the
        # resident rows into anonymous pages, auto (default) asks the device which of the
        # two it is willing to register. _pick_map_mode has the reasoning.
        self.map_mode = os.environ.get("FREETOKEN_BANK_MAP", "auto").strip().lower()
        if self.map_mode not in ("auto", "shared", "private"):
            raise ValueError(
                f"FREETOKEN_BANK_MAP={self.map_mode!r}: expected auto, shared or private"
            )
        self.private = self._pick_map_mode(register)
        self.probe_refused = self.private and self.map_mode == "auto"
        if not self.layers:
            raise BankFileError(f"{path}: no layers to map")
        # Only this rank's layers, which the file keeps contiguous. A private mapping is charged
        # against the commit limit for its whole length whether or not a page is ever copied, and
        # the whole file is twice one rank's share on a two-rank split.
        first_block, map_end = self.layout.range_of(self.layers)
        # One page more in front, which this process never registers. Every bank view shares the
        # mapping's storage, and Tensor.is_pinned() asks about the STORAGE's first byte, not the
        # view's: a mapping that began at the first block would begin inside a registered
        # resident prefix, every view would read as pinned, and the whole-layer prefill copy
        # (OffloadMoeCache._staged_h2d) would hand the driver an async copy over the
        # unregistered rows -- "CUDA error: invalid argument" on the first prefill (measured on
        # the 2060). The per-rank files of the earlier layout mapped from the header and never
        # met this. The page before any block is a manifest slot or another layer's block.
        self._map_offset = max(0, first_block - ALIGN)
        self._map = mmap.mmap(
            self._fd, map_end - self._map_offset, offset=self._map_offset,
            access=mmap.ACCESS_COPY if self.private else mmap.ACCESS_READ,
        )
        self._registered: list[int] = []
        self.hot_blocks = 0
        self.registered_blocks = 0
        self.locked_bytes = 0
        self.requested_bytes = 0  # what the placement wanted locked; locked_bytes is what the OS gave
        self.lock_errno = 0  # last mlock failure, 0 when every lock succeeded
        self.whole_blocks = 0
        self.registered_bytes = 0
        self.sources: dict[str, list] = {}
        # (file offset, nbytes) of every block's file-backed remainder, in file order: what the
        # page cache may drop and --moe-bank-rewarm reads back (moe/bank_rewarm.py).
        self.cold_spans: list[tuple[int, int]] = []
        # (bank name, position in this rank's layers -- the layer id the cache and the CPU
        # executor use --, address of row 0 in this rank's mapping, bytes per row, first
        # file-backed row) for every block with a file-backed part: what --moe-bank-prefetch
        # hands the CPU executor so it can advise exactly the rows a step routes to.
        self.cold_blocks: list[tuple[str, int, int, int, int]] = []
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
            # torch warns that a read-only buffer cannot be written through -- on the shared
            # form, where nothing writes to a bank anyway. The private form is writable by
            # construction, which is how its resident rows become anonymous, and does not
            # warn; nothing writes to it either beyond the byte per page _cow puts back
            # exactly as it found it.
            warnings.filterwarnings("ignore", message=".*buffer is not writable.*")
            buf = torch.frombuffer(self._map, dtype=torch.uint8)
        # via the tensor, not ctypes.from_buffer: that one insists on a writable buffer,
        # which on the shared form is the property being avoided here
        self._base = buf.data_ptr()
        self._buf = buf
        for name, row_shape, dtype_name, row_bytes in self.layout.banks:
            dtype = _dtype_of(dtype_name)
            per_layer = []
            for pos, layer in enumerate(self.layers):
                off = self.layout.offset_of(name, layer)
                rel = off - self._map_offset  # the mapping starts at this rank's first block
                span = self.layout.num_experts * row_bytes
                per_layer.append(
                    buf[rel:rel + span].view(dtype).reshape(self.layout.num_experts, *row_shape)
                )
                lock_rows = (
                    self.layout.num_experts
                    if self.register_mode == "all" or span <= _WHOLE_BLOCK_BYTES
                    else self.hot_per_layer
                )
                self.whole_blocks += span <= _WHOLE_BLOCK_BYTES
                if span > lock_rows * row_bytes:
                    self.cold_spans.append(
                        (off + lock_rows * row_bytes, span - lock_rows * row_bytes)
                    )
                    self.cold_blocks.append((name, pos, self._base + rel, row_bytes, lock_rows))
                self._settle(rel, lock_rows * row_bytes, span, register)
                self._advise(hint, rel, span, len(self._map))
                if preload:
                    self._touch(buf, rel, span)
            self.sources[name] = per_layer
        self.cold_spans.sort()
        self.settle_seconds = time.perf_counter() - started
        self.preloaded = preload

    def _pick_map_mode(self, register: bool) -> bool:
        """Ask the device which form of resident row it is willing to register.

        ACCESS_READ is the cheaper form and stays the default where it works: the resident
        rows are page cache, shared with the file, droppable without writeback once
        unlocked, and nothing is copied. It needs cudaHostRegisterReadOnly (0x08), because
        flags=0 asks for read-write pinning and a read-only mapping cannot give it.

        Some hosts have no 0x08: cudaDevAttrHostRegisterReadOnlySupported is 0 on an RTX
        3060 pair on native Ubuntu with the open kernel module (the same GA106 has it under
        WSL2 -- platform, not silicon), and there both flags fail. Registering nothing is a
        supported state but an expensive one: attach() then sends every decode miss to the
        CPU executor, which is the whole VRAM expert cache thrown away.

        Anonymous memory does register on that host -- measured, 1 GiB with flags=0. So when
        the read-only form is refused, map ACCESS_COPY instead (MAP_PRIVATE, writable) and
        let _cow turn the resident rows into private anonymous pages. The non-resident rows
        stay file-backed in either form, which is the part that makes 31.7 GiB of banks fit
        in 24 GiB of RAM at all.

        Asked with the bank file itself, one page of it, before the real mapping exists --
        rather than read off cudaDevAttrHostRegisterReadOnlySupported, so that a host which
        advertises the flag and still refuses this file lands on the form that works.
        """
        if not register or self.map_mode == "shared":
            return False
        if self.map_mode == "private":
            return True
        if os.name != "posix":
            return False
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            torch.cuda.init()
            cudart = torch.cuda.cudart()
        except Exception:
            return False
        probe = buf = None
        try:
            probe = mmap.mmap(self._fd, ALIGN, access=mmap.ACCESS_READ)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*buffer is not writable.*")
                buf = torch.frombuffer(probe, dtype=torch.uint8)
            addr = buf.data_ptr()
            for flags in (0x08, 0):
                if int(cudart.cudaHostRegister(addr, ALIGN, flags)) == 0:
                    cudart.cudaHostUnregister(addr)
                    clear_cuda_error()
                    return False
                # a refused call leaves its error in this thread's slot; see _settle
                clear_cuda_error()
            return True
        except Exception:
            return False
        finally:
            del buf
            if probe is not None:
                probe.close()

    def _cow(self, offset: int, nbytes: int) -> None:
        """Make the resident rows of one block private, and give the file's copy back.

        A byte per page, written back as it was read, which is all it takes: MAP_PRIVATE
        shares the page cache until something writes, and that write is what copies the page
        into the anonymous memory flags=0 can pin.

        Deliberately not mlock, though mlock would do the same thing -- it populates a
        writable private mapping with FOLL_WRITE. mlock only reaches as far as
        RLIMIT_MEMLOCK, and the host this exists for stops at 7.83 GiB of the 24 GiB it asks
        to keep resident. A range that is copied for part of its length and still page cache
        for the rest is a range the registration below cannot take.

        Then POSIX_FADV_DONTNEED on the same bytes of the file, per block and not once at
        the end: between the copy and the fadvise the block is resident twice. Per block
        that overshoot is one block (1.3 GiB for Flash-Next); deferred to the end it is the
        whole resident half, 24 GiB, and a 64 GB host goes to swap -- which is exactly how
        this approach was measured into the ground the first time (docs/bank-ram.md).

        MADV_DONTNEED is the wrong call here and would undo the work: on a private mapping
        it discards the private copies, not the file's pages.
        """
        buf = self._buf
        if buf is None or nbytes <= 0:
            return
        view = buf[offset:offset + nbytes:ALIGN]
        if view.numel():
            view |= 0
        # and the last page: hot_rows * row_bytes need not end on a page boundary
        buf[offset + nbytes - 1:offset + nbytes] |= 0
        try:
            os.posix_fadvise(self._fd, self._map_offset + offset, nbytes, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass

    @property
    def fully_registered(self) -> bool:
        """Whether prefix_pinned_rows can be believed.

        It is one number for every layer and every block: rows [0, hot) are device
        addressable, everywhere. A single block that failed to register makes it a lie, and
        the hybrid path then fetches a row the GPU has no address for -- an illegal access
        inside the decode graph, which is how the restriction was found in the first place.

        Registration is all-or-nothing in the cases seen so far: a host either takes the
        flag or does not. It stops being all-or-nothing as soon as a limit is reached
        partway through 24 GiB of blocks, and there is nothing in the summary line to show
        it -- the bytes look registered. Partly registered is not a state the cache can be
        told about; attach() treats it as none.
        """
        return self.hot_blocks > 0 and self.registered_blocks == self.hot_blocks

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
            self.hot_blocks += 1
            if self.private:
                self._cow(offset, nbytes)
            if self._libc is None:
                pass
            elif self._libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)) == 0:
                self.locked_bytes += nbytes
            else:
                # RLIMIT_MEMLOCK, near certainly: systemd's default is MAX(64M, RAM/8), so a
                # 64 GB host stops at 8 GiB of the 24 GiB --moe-bank-ram asks for and the rest
                # is evictable page cache. Silence here cost a bug report: the summary said
                # "7.8 GiB locked resident" against a 24 GiB budget and explained nothing.
                #
                # On the private form this is a second line of defence rather than the
                # mechanism: the rows are copied and resident before mlock is asked, and
                # registering them pins them where mlock could not reach. finish() only
                # warns when the registration did not cover them either.
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
        # flags=0 asks for read-write pinning, which a read-only file mapping cannot give;
        # cudaHostRegisterReadOnly (0x08) is the flag for exactly this case. Ask for it first
        # and keep plain as the fallback for a driver that does not know the flag. On the
        # private form the order reverses: those rows are anonymous now, flags=0 is what
        # they take, and 0x08 is the fallback.
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
        for flags in (0, 0x08) if self.private else (0x08, 0):
            if int(cudart.cudaHostRegister(addr, nbytes, flags)) == 0:
                self.registered_bytes += nbytes
                self.registered_blocks += 1
                self._registered.append(addr)
                return
            clear_cuda_error()

    def cold_residency(self) -> float:
        """Share of the file-backed rows the page cache holds right now (1.0 when there are none).

        mincore(2) on this process's own mapping, over page-aligned spans. The resident prefix
        is left out on purpose: it is held regardless, and counting it would hide a cold side
        that has been emptied behind it.
        """
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        limit = len(self._map)
        pages = present = 0
        for off, nbytes in self.cold_spans:
            span = advise_range(off - self._map_offset, nbytes, limit)
            if span is None:
                continue
            count = (span[1] + ALIGN - 1) // ALIGN
            vec = ctypes.create_string_buffer(count)
            if libc.mincore(ctypes.c_void_p(self._base + span[0]), span[1], vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            pages += count
            present += sum(b & 1 for b in vec.raw)
        return present / pages if pages else 1.0

    def rewarm(self, cancel: threading.Event, step_bytes: int = 64 << 20) -> tuple[int, float, bool]:
        """Read the file-backed rows back into the page cache, in file order, checking
        ``cancel`` between steps. Returns (bytes walked, seconds, finished).

        A byte per page through the mapping, as _touch does for FREETOKEN_BANK_PRELOAD and for
        the same reason: MADV_WILLNEED is a hint the kernel is free to ignore under exactly the
        pressure that emptied the cache. The step bounds how long a request that arrives
        mid-walk waits for this to let go -- 64 MiB is well under a second even from disk.
        """
        started = time.perf_counter()
        walked = 0
        for off, nbytes in self.cold_spans:
            pos = off - self._map_offset
            end = pos + nbytes
            while pos < end:
                if cancel.is_set():
                    return walked, time.perf_counter() - started, False
                step = min(step_bytes, end - pos)
                self._touch(self._buf, pos, step)
                walked += step
                pos += step
        return walked, time.perf_counter() - started, True

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
    """Assembles a ``--moe-bank-ram`` run for one rank: find, finish or write its layers, open them.

    ``layers`` are this rank's global bank-layer ids; the loader's sink numbers them from zero.

    ``prepare`` runs before the loader and answers whether the loader needs to run at all:

    * every layer committed -- in the wanted order, or reordered into it in place -- means the
      checkpoint's expert tensors are never opened;
    * otherwise the loader streams the checkpoint and ``sink`` writes the layers the file lacks.
      Writing per layer as the checkpoint streams in is not an optimization: holding the full
      banks and a second copy at once needs more RAM than the host that wants this feature has,
      so the overshoot has to stay at one layer (~1.3 GiB for Flash-Next).

    ``can_write`` False is a checkpoint without expert tensors (``ft bank pack``): a missing
    layer, a missing file or a file for some other geometry is then an error that says so,
    before anything is loaded.

    ``wanted`` is the placement solved from ``--moe-bank-stats``; without one each layer keeps
    the order already in the file, so a histogram only has to be passed when it changes.
    """

    def __init__(self, path: str, layers, *, num_experts: int, hot_per_layer: int,
                 all_layers=None, wanted: dict | None = None, layout: MappedBankLayout | None = None,
                 meta: dict | None = None, can_write: bool = True, log=None, warn=None,
                 readahead: str = "off", report_readahead: bool = True, first_rank: bool = True):
        self.path = path
        # --moe-bank-readahead: "off" (report only), "auto" (the recommended window) or kB. Every
        # rank maps the same file, so one device: each rank sets it before opening its own
        # mapping, and only the first says anything about it.
        self.readahead = str(readahead or "off").strip().lower()
        self.report_readahead = report_readahead
        # the one rank that speaks for the whole file (the space it still needs, say)
        self.first_rank = first_rank
        self.layers = [int(x) for x in layers]
        # every MoE layer of the model: the file's geometry, whichever of them this rank serves
        self.all_layers = self.layers if all_layers is None else [int(x) for x in all_layers]
        self.num_experts = int(num_experts)
        self.hot_per_layer = int(hot_per_layer)
        self.wanted = {int(k): [int(x) for x in v] for k, v in (wanted or {}).items()}
        self.layout = layout
        self.meta = dict(meta or {})
        self.can_write = can_write
        self.log = log or (lambda _msg: None)
        self.warn = warn or self.log
        self._file: BankFile | None = None
        self._orders: dict[int, list[int]] = {}
        self._committed: dict[int, list[int]] = {}
        self._ready = False
        self._seen: set[int] = set()
        self._lock = threading.Lock()
        self.reordered: list[int] = []
        self.written: list[int] = []
        self.banks: MappedBanks | None = None

    # ----- what the file holds ------------------------------------------------------
    @property
    def placement(self):
        from freetoken.moe.bank_disk import BankPlacement

        return BankPlacement(self.num_experts, self.hot_per_layer, dict(self._orders))

    @property
    def ready(self) -> bool:
        """True once ``prepare`` found every layer: the loader must not run."""
        return self._ready

    def _open(self, layout: MappedBankLayout) -> None:
        """Open the file for this geometry, creating or (when allowed) starting it over."""
        directory = os.path.dirname(os.path.abspath(self.path))
        if self.can_write:
            os.makedirs(directory, exist_ok=True)
        with file_lock(self.path):
            if os.path.exists(self.path):
                try:
                    existing = MappedBankLayout.read(self.path)
                except ValueError as exc:
                    existing, why = None, str(exc)
                else:
                    # a packed checkpoint has no expert shards to stamp: its fingerprint is
                    # what ties it to the file
                    why = existing.mismatch(layout, ignore=() if self.can_write else ("source_stamp",))
                if why is None:
                    # read-only is enough to serve; only a reorder or a journal needs to write
                    self._file = BankFile.open(self.path, writable=os.access(self.path, os.W_OK))
                    return
                canonical = None
                if existing is not None:
                    with BankFile.open(self.path, writable=False) as other:
                        canonical = other.canonical_for()
                if canonical:
                    raise BankFileError(
                        f"--moe-bank-ram: {self.path} is the only copy of the experts of "
                        f"{canonical}, and this run wants a different file ({why}). Serve that "
                        f"checkpoint, or point --moe-bank-dir somewhere else"
                    )
                if not self.can_write:
                    raise BankFileError(
                        f"--moe-bank-ram: {self.path} does not match this run ({why}), and this "
                        f"checkpoint has no expert tensors to write a new one from. The bank was "
                        f"written for {self._describe(existing)}; this run binds "
                        f"{self._describe(layout)}"
                    )
                self.log(f"--moe-bank-ram: {self.path} was written for a different run ({why}); starting it over")
            elif not self.can_write:
                raise BankFileError(
                    f"--moe-bank-ram: no bank file at {self.path}, and this checkpoint has no expert "
                    f"tensors (it was packed by `ft bank pack`); the bank file is the only copy of them"
                )
            self._file = BankFile.create(self.path, layout)

    @staticmethod
    def _describe(layout: MappedBankLayout | None) -> str:
        if layout is None:
            return "an unreadable geometry"
        m = layout.meta
        return f"{m.get('kind', '?')} / {m.get('kernel', '?')} experts"

    def _plan(self, manifest) -> list[int]:
        """Settle each layer's order; returns the layers the file does not hold."""
        present = {}
        for layer in self.layers:
            state = self._file.layer_state(layer, manifest)
            if state is not None:
                present[layer] = state[0]
        for layer in self.layers:
            self._orders[layer] = (
                self.wanted.get(layer) or present.get(layer) or list(range(self.num_experts))
            )
        self._committed = {l: o for l, o in present.items() if o == self._orders[l]}
        return [l for l in self.layers if l not in present]

    def prepare(self) -> bool:
        """Before the loader: True when this rank's layers are all in the file (the loader is skipped).

        Without a known geometry (a loader with no expert method) nothing can be decided before
        the first layer, and this returns False.
        """
        if self.layout is None:
            if not self.can_write:
                raise BankFileError("--moe-bank-ram: the bank geometry is unknown before loading, so a "
                                    "checkpoint without expert tensors cannot be served from it")
            return False
        self._open(self.layout)
        self._file.recover_journals(self.layers, log=self.log)
        manifest = self._file.manifest()
        missing = self._plan(manifest)
        gib = self.layout.layer_bytes() / 2**30
        if missing:
            if not self.can_write or self._file.canonical_for(manifest):
                raise BankFileError(
                    f"--moe-bank-ram: {self.path} has no layers {_ranges(missing)} of this rank's "
                    f"{_ranges(self.layers)}, and {'this checkpoint has no expert tensors to write them from' if not self.can_write else 'it is the only copy of a packed checkpoint and is never written from one'}"
                )
            need = len(missing) * gib * 2**30
            free = free_bytes(self.path)
            if free is not None and need > free:
                raise BankFileError(
                    f"--moe-bank-ram: writing layers {_ranges(missing)} to {self.path} needs "
                    f"{need / 2**30:.1f} GiB and the filesystem has {free / 2**30:.1f} GiB free"
                )
            if self.first_rank:
                # Under WSL2 the check above sees the virtual disk, not the Windows drive it grows
                # on. Counted for the whole file: every rank writes its own missing layers at once.
                from freetoken.moe import disk_probe

                absent = [l for l in self.all_layers if self._file.layer_state(l, manifest) is None]
                line = disk_probe.host_space_warning(self.path, len(absent) * gib * 2**30)
                if line:
                    self.warn(line)
            self.log(
                f"--moe-bank-ram: {self.path} lacks layers {_ranges(missing)}; reading the checkpoint's "
                f"experts for layers {_ranges(self.layers)} and writing {len(missing) * gib:.1f} GiB "
                f"({self.hot_per_layer}/{self.num_experts} experts resident per layer)"
            )
            return False
        todo = [l for l in self.layers if l not in self._committed]
        if todo:
            free = free_bytes(self.path)
            if free is not None and gib * 2**30 > free:
                raise BankFileError(
                    f"--moe-bank-ram: reordering {self.path} in place needs room for one layer's "
                    f"journal ({gib:.1f} GiB) and the filesystem has {free / 2**30:.1f} GiB free"
                )
            self.log(
                f"--moe-bank-ram: placement changed for layers {_ranges(todo)}; reordering "
                f"{len(todo) * gib:.1f} GiB in place (one layer journaled at a time)"
            )
            started = time.perf_counter()
            for layer in todo:
                self._file.reorder_layer(layer, self._orders[layer])
                self._file.drop_cache(layer)
            self.reordered = todo
            self.log(f"--moe-bank-ram: reordered {len(todo)} layers in {time.perf_counter() - started:.0f} s")
        self._ready = True
        self.log(
            f"--moe-bank-ram: {self.path} holds layers {_ranges(self.layers)}; the checkpoint's "
            f"expert tensors are not read"
        )
        return True

    def sink(self, layer_id: int, banks) -> None:
        """``layer_sink`` for the expert loader; ``layer_id`` counts from this rank's first layer.

        A failed write surfaces as BankFileError naming the bank file: the loader around this
        reports anything else as an unreadable checkpoint (WeightLoadError)."""
        try:
            self._sink(layer_id, banks)
        except OSError as exc:
            raise BankFileError(f"--moe-bank-ram: writing {self.path} failed: {exc}") from exc

    def _sink(self, layer_id: int, banks) -> None:
        with self._lock:
            if self._file is None:
                if self.layout is None:
                    self.layout = layout_from_sample(banks, self.all_layers, self.num_experts, self.meta)
                self._open(self.layout)
                self._file.recover_journals(self.layers, log=self.log)
                self._plan(self._file.manifest())
        layer = self.layers[layer_id]
        if self._committed.get(layer) != self._orders[layer]:
            self._file.write_layer(layer, banks, self._orders[layer])
            self._file.drop_cache(layer)
            with self._lock:
                self.written.append(layer)
        with self._lock:
            self._seen.add(layer_id)
        for bank in banks.values():
            release = getattr(bank, "release", None)
            if release is not None:
                release()  # caps peak RAM at one layer's worth of the original banks

    def finish(self) -> None:
        if not self._ready:
            missing = [x for x in range(len(self.layers)) if x not in self._seen]
            if missing:
                raise RuntimeError(f"--moe-bank-ram: layers {missing} never reached the sink")
            if self.written:
                self.log(f"--moe-bank-ram: wrote layers {_ranges(sorted(self.written))}")
            self.log(f"--moe-bank-ram: source banks released, RSS now {_rss_gib():.1f} GiB")
        if self._file is not None:
            self._file.close()
            self._file = None
        # Before the mapping's file is opened: the kernel copies the device window into each
        # open file when it is opened, and a fault reads with that copy -- so the window has to
        # be right before MappedBanks opens the bank, and a change made later reaches only the
        # next start. It also means the settle already reads with the window decode will use.
        self._apply_readahead()
        self.log("--moe-bank-ram: faulting in and locking the resident rows")
        self.banks = MappedBanks(self.path, layers=self.layers, hot_per_layer=self.hot_per_layer)
        b = self.banks
        if b.probe_refused:
            self.log(
                "--moe-bank-ram: this device will not register a read-only file mapping, so "
                "the resident rows are kept as private copies instead -- they cost anonymous "
                "RAM rather than page cache, and the GPU can address them. "
                "FREETOKEN_BANK_MAP=shared forces the read-only form back"
            )
        self.log(
            f"--moe-bank-ram: mapped {b.mapped_bytes / 2**30:.1f} GiB, "
            f"{b.locked_bytes / 2**30:.1f} GiB locked resident, "
            f"{b.registered_bytes / 2**30:.1f} GiB registered for PCIe"
            + (", private pages" if b.private else ", shared page cache")
            + (f", {b.whole_blocks} small blocks kept whole" if b.whole_blocks else "")
            + f", {b.advise_mode} fault readahead"
            + (", whole file preloaded" if b.preloaded else "")
            + f" ({b.settle_seconds:.0f} s)"
        )
        if b.registered_bytes and not b.fully_registered:
            self.warn(
                f"--moe-bank-ram: only {b.registered_blocks} of {b.hot_blocks} resident "
                f"blocks could be registered, and a bank registered in part cannot be told "
                f"apart from an unregistered one by the cache -- every decode miss goes to "
                f"the CPU executor. Lower --moe-bank-ram, or FREETOKEN_BANK_REGISTER=none "
                f"to stop asking"
            )
        if b.lock_errno and not b.fully_registered:
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

    def _apply_readahead(self) -> None:
        """Report the device readahead window against this model's geometry, and set it when asked.

        Compared with this model's own widest block row, not a constant: the blocks differ by
        three orders of magnitude between models -- Flash-Next's widest row is 1600 kB (best
        measured window 256), gpt-oss-120b's is 7.91 MiB (best 2048). A threshold tuned on one
        was wrong on the other: 1024 drew the old warning on gpt-oss while measuring 17% faster
        than 256. A window wider than the widest row cannot finish inside one expert's row, so
        the tail of every fault lands in a neighbouring expert -- and the non-resident rows are
        the tail of the frequency order, the ones least likely to be wanted next.

        Setting it is opt-in (--moe-bank-readahead auto|<kB>) because the window belongs to the
        whole device: every other file on that disk reads with it too, and it outlives the
        server. Nothing is put back at exit, and the log says so.
        """
        from freetoken.moe import disk_probe

        ra = readahead_kb(self.path)
        layout = self.layout
        if ra is None or layout is None or not layout.banks:
            return
        kb, where = ra
        widest = max(rb for _, _, _, rb in layout.banks)
        widest_kb = widest // 1024
        rec = disk_probe.recommend_readahead_kb(widest)
        mode = self.readahead
        log = self.log if self.report_readahead else (lambda _msg: None)
        warn = self.warn if self.report_readahead else (lambda _msg: None)
        want = rec if mode == "auto" else (None if mode in ("off", "") else int(mode))
        if want is not None:
            if want == kb:
                log(f"--moe-bank-readahead {mode}: {where} is already {kb} kB")
                return
            ok, why = disk_probe.set_readahead(where, want)
            if ok:
                log(
                    f"--moe-bank-readahead {mode}: {where} {kb} -> {want} kB (widest expert-row "
                    f"block {widest_kb} kB); device-wide, and left set after exit"
                )
                return
            warn(
                f"--moe-bank-readahead {mode}: could not write {where} ({why}), so it stays "
                f"{kb} kB. Run once as root, then restart the server (an open mapping keeps the "
                f"window it was opened with): {disk_probe.readahead_command(want, where)}"
            )
            return
        note = f"--moe-bank-ram: device readahead {kb} kB ({where})"
        if kb > max(widest_kb, 1):
            warn(
                f"{note} -- wider than this model's widest expert-row block ({widest_kb} kB), "
                f"so every fault on a non-resident row spills into experts nobody asked for "
                f"(measured 2.5x slower decode at 8192). Recommended {rec} kB: "
                f"{disk_probe.readahead_command(rec, where)} and restart, or --moe-bank-readahead auto"
            )
        else:
            log(note + (f", recommended {rec} kB" if kb != rec else ""))

    @property
    def sources(self) -> dict:
        assert self.banks is not None, "call finish() first"
        return self.banks.sources

    def attach(self, cache, device) -> None:
        """Hand over the renumbering, and keep the GPU off the rows it cannot address."""
        from freetoken.moe.bank_disk import permutation_tensor

        placement = self.placement
        perm = [
            permutation_tensor(placement, layer, device=device) for layer in self.layers
        ]
        # --spec-mtp appends the draft head's expert layer to the bank sources after the
        # placement was solved (engine._append_mtp_bank), so the cache can hold one layer
        # more than the placement has ever heard of. That layer is not renumbered -- its
        # rows are the checkpoint's own order -- so identity is the right map for it, which
        # is what None means to attach_offload_moe_cache.
        #
        # Without the padding the head's layer indexes past the end of this list and the
        # boot dies in attach_offload_moe_cache, on whichever rank carries the head (the
        # last one, so rank 0 of a --pp-size 2 run comes up and rank 1 does not).
        banked = getattr(cache, "bank_sources", None)
        if banked:
            perm += [None] * max(len(next(iter(banked.values()))) - len(perm), 0)
        cache.expert_perm = perm
        if self.banks is None:
            return
        from freetoken.moe.bank_reader import read_mode

        mode = read_mode(os.environ.get("FREETOKEN_BANK_PREAD"))
        if self.banks.cold_spans and mode is not None:
            try:
                from freetoken.moe.bank_reader import BankReader

                cache.bank_reader = BankReader(self.banks, direct=mode == "direct")
                self.log(f"--moe-bank-ram: prefill reads the non-resident rows in parallel "
                         f"({cache.bank_reader.describe()})")
            except Exception as exc:  # noqa: BLE001 -- the mapping copy still works
                self.log(f"--moe-bank-ram: parallel prefill reads unavailable ({exc}); page faults as before")
        if self.banks.fully_registered:
            # Only rows [0, hot) are cudaHostRegistered; the rest is host memory the device
            # has no address for, and a GPU fetch of one is an illegal access inside the
            # decode graph (which is how this was found). Telling the cache where the
            # registered prefix ends lets the hybrid path keep its PCIe fetches inside it and
            # send the remaining misses to the CPU -- instead of the whole layer going to the
            # CPU, which is what LOCKED layers do and what costs the VRAM cache.
            cache.prefix_pinned_rows = self.banks.hot_per_layer
            self.log(
                f"--moe-bank-ram: PCIe fetches restricted to the resident "
                f"{self.banks.hot_per_layer}/{self.banks.layout.num_experts} experts "
                f"per layer; the rest decode on the CPU"
            )
            return
        # Nothing registered -- or not every block, which the cache has no way to express:
        # no row can be assumed to have a device address, so every miss goes to the CPU.
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


def _ranges(ids) -> str:
    """``[0, 1, 2, 5, 7, 8]`` -> ``"0-2, 5, 7-8"``: layer lists in messages stay one line."""
    ids = sorted(int(x) for x in ids)
    if not ids:
        return "none"
    out, start, prev = [], ids[0], ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = x
    out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(out)
