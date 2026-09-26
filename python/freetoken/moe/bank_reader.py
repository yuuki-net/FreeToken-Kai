"""Read a prefill chunk's non-resident bank rows with parallel reads, not page faults.

A prefill chunk streams every layer's whole expert bank to the GPU. The rows past the
--moe-bank-ram resident prefix are not registered with the device, so they go through pinned
staging buffers first (``OffloadMoeCache._staged_h2d``), and copying them out of the mapping
turns every row the page cache does not hold into a page fault: one thread, one readahead
window per fault, waiting on each. When the page cache is smaller than a rank's non-resident
rows -- the 64 GB host -- reading one layer evicts the one before, so every chunk reads nearly
all of them from the disk again that way. Measured on an RTX 2060 host (Ornith, 11 GiB of
non-resident rows, the server held to 13 GiB): 10.6 GiB re-read per chunk at 1.2 GiB/s with an
8192 kB readahead window, 9.2 s of a 13 s chunk; 0.11 GiB/s with no window at all. The window
that suits decode (256 kB) sits between the two.

This reads the same bytes from the file with ``pread``, from several threads at once, into the
pinned staging buffers. The rate is the drive's parallel rate rather than one fault at a time,
and does not depend on ``read_ahead_kb``, so the window can stay where decode wants it.

The reads are buffered by default: the rows land in the page cache as the fault path left them,
and the decode that follows finds them there. ``O_DIRECT`` (``FREETOKEN_BANK_PREAD=direct``)
reads a little faster under memory pressure and leaves the page cache alone, but that page
cache is what the next decode reads from. Measured, RTX 2060 host held to 13 GiB (Ornith,
16.9 GiB of rows per chunk on its path): prefill 204 tok/s through faults, 304 buffered, 325
direct. Two RTX 3060s with a 40 GiB balloon (Flash-Next, --moe-bank-ram 42G, readahead 256 kB):
direct cost decode 15.9 -> 13.2 tok/s against the fault path and bought no prefill; with RAM to
spare, direct reads also kept re-reading from the disk every chunk what they never cached.

A piece the page cache already holds (at least ``FREETOKEN_BANK_PREAD_CACHED``, default 0.9 of
its pages, by ``mincore``) is copied out of the mapping on the same threads instead of read:
with a third of the rows cached, deciding per range rather than per piece read every byte again.

``FREETOKEN_BANK_PREAD`` = ``buffered`` (default), ``direct``, or ``0`` for the fault path;
``FREETOKEN_BANK_READ_THREADS`` (default 8) and ``FREETOKEN_BANK_READ_PIECE_MB`` (default 16)
size it. Where the filesystem refuses ``O_DIRECT`` a ``direct`` request reads buffered.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import torch

from freetoken.utils.prefill_profile import major_faults as _major_faults

ALIGN = 4096
READ_SETS = 2


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


class DirectRangeReader:
    """Parallel reads of file ranges into reusable page-aligned host buffers.

    Device-independent (the pinned-ness of the buffers is the caller's ``alloc``), so the read
    path is testable on any host. ``sets`` groups of ``threads`` buffers alternate, so a group
    can be refilled while the device still drains the previous one.
    """

    def __init__(self, path: str, *, threads: int = 8, piece_bytes: int = 16 << 20, sets: int = READ_SETS,
                 alloc: Callable[[int], torch.Tensor] | None = None, direct: bool = True) -> None:
        self.path = path
        self.threads = max(1, threads)
        self.piece = max(ALIGN, piece_bytes // ALIGN * ALIGN)
        self.direct = False
        fd = -1
        o_direct = getattr(os, "O_DIRECT", 0)
        if direct and o_direct:
            try:
                fd = os.open(path, os.O_RDONLY | o_direct)
                self.direct = True
            except OSError:
                fd = -1
        self._fd = fd if fd >= 0 else os.open(path, os.O_RDONLY)
        alloc = alloc or (lambda n: torch.empty(n, dtype=torch.uint8))
        # a buffer holds one piece plus the page on either side the aligned read spills into
        span = self.piece + 2 * ALIGN
        self.buffers: list[list[tuple[torch.Tensor, memoryview]]] = []
        for _ in range(sets):
            group = []
            for _ in range(self.threads):
                raw = alloc(span + ALIGN)
                skew = (-raw.data_ptr()) % ALIGN
                view = raw[skew:skew + span]
                group.append((view, memoryview(view.numpy()).cast("B")))
            self.buffers.append(group)
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="bank-read")
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True)
            os.close(self._fd)

    def _read(self, buf: memoryview, offset: int, nbytes: int) -> tuple[int, int]:
        """Read ``[offset, offset + nbytes)`` into ``buf``; returns (start of the data in buf, bytes)."""
        if self.direct:
            start = offset // ALIGN * ALIGN
            skew = offset - start
            want = -(-(skew + nbytes) // ALIGN) * ALIGN
        else:
            start, skew, want = offset, 0, nbytes
        got = 0
        while got < skew + nbytes:
            n = os.preadv(self._fd, [buf[got:want]], start + got)
            if n <= 0:
                raise OSError(f"{self.path}: short read at {start + got} ({got} of {skew + nbytes} bytes)")
            got += n
        return skew, nbytes

    def _fill(self, s: int, j: int, offset: int, p: int, n: int, source, cached) -> tuple[int, bool]:
        """One piece into buffer ``j`` of set ``s``: copied from ``source`` when ``cached`` says the
        page cache holds it, else read from the file. Returns (start of the data in the buffer, copied)."""
        view, raw = self.buffers[s][j]
        if source is not None and cached is not None and cached(p, n):
            # the buffers are made where the engine runs, under torch.inference_mode(), and that
            # mode is per thread: outside it here an in-place write to them is refused
            with torch.inference_mode():
                view[:n].copy_(source[p:p + n])
            return 0, True
        return self._read(raw, offset + p, n)[0], False

    def read(self, offset: int, nbytes: int, sink: Callable[[int, torch.Tensor, int], None],
             before_group: Callable[[int], None] | None = None, *, source: torch.Tensor | None = None,
             cached: Callable[[int, int], bool] | None = None) -> int:
        """Read ``nbytes`` from ``offset`` in pieces, ``threads`` at a time, and hand each piece to
        ``sink(position, host view, group set)`` in file order. ``before_group(set)`` runs before
        a set's buffers are refilled (the caller waits there for whatever last read them).
        ``source`` is the same bytes as a host tensor (the mapping) and ``cached(position, bytes)``
        picks the pieces to copy from it instead. Returns the bytes copied rather than read."""
        pieces = [(p, min(self.piece, nbytes - p)) for p in range(0, nbytes, self.piece)]
        copied = 0
        for g in range(0, len(pieces), self.threads):
            s = (g // self.threads) % len(self.buffers)
            if before_group is not None:
                before_group(s)
            group = pieces[g:g + self.threads]
            futures = [
                self._pool.submit(self._fill, s, j, offset, p, n, source, cached)
                for j, (p, n) in enumerate(group)
            ]
            for j, ((p, n), fut) in enumerate(zip(group, futures)):
                skew, from_source = fut.result()
                copied += n if from_source else 0
                sink(p, self.buffers[s][j][0][skew:skew + n], s)
        return copied


def read_mode(value: str | None) -> str | None:
    """``FREETOKEN_BANK_PREAD`` -> ``"buffered"``, ``"direct"``, or None for the fault path.
    Anything else reads buffered (``auto`` and ``1`` from earlier builds included)."""
    v = (value or "").strip().lower()
    if v in ("0", "off", "no", "false"):
        return None
    return "direct" if v == "direct" else "buffered"


def pinned_bytes() -> int:
    """Page-locked bytes the BankReader takes as configured, 0 when ``FREETOKEN_BANK_PREAD`` turns it
    off: what the pin budget has to leave for it (mapped_bank.pinned_after_banks)."""
    if read_mode(os.environ.get("FREETOKEN_BANK_PREAD")) is None:
        return 0
    threads = _env_int("FREETOKEN_BANK_READ_THREADS", 8)
    piece = max(ALIGN, (_env_int("FREETOKEN_BANK_READ_PIECE_MB", 16) << 20) // ALIGN * ALIGN)
    return READ_SETS * threads * (piece + 3 * ALIGN)  # DirectRangeReader: a piece + 3 pages each


class BankReader:
    """``OffloadMoeCache.bank_reader``: the prefill copy of a mapped bank's non-resident rows."""

    def __init__(self, banks, *, threads: int | None = None, piece_bytes: int | None = None,
                 cached_share: float | None = None, direct: bool = False, why: str = "") -> None:
        from freetoken.kernel.pinned import alloc_pinned_tensor

        self.banks = banks
        self._lo = banks._base
        self._hi = banks._base + len(banks._map)
        self._file_base = banks._map_offset - banks._base
        self.cached_share = float(os.environ.get("FREETOKEN_BANK_PREAD_CACHED", "") or 0.9) \
            if cached_share is None else cached_share
        self.reader = DirectRangeReader(
            banks.path,
            threads=threads or _env_int("FREETOKEN_BANK_READ_THREADS", 8),
            piece_bytes=piece_bytes or _env_int("FREETOKEN_BANK_READ_PIECE_MB", 16) << 20,
            alloc=lambda n: alloc_pinned_tensor(n, dtype=torch.uint8),
            direct=direct,
        )
        self.why = why
        self._events = [torch.cuda.Event() for _ in self.reader.buffers]
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.bytes_cached = 0

    def describe(self) -> str:
        r = self.reader
        mode = "O_DIRECT" if r.direct else "buffered"
        return (f"{r.threads} threads x {r.piece >> 20} MiB, {mode}"
                + (f": {self.why}" if self.why else "")
                + f"; a piece {self.cached_share:.0%} in the page cache is copied from the mapping")

    def _residency(self, addr: int, nbytes: int) -> tuple[bytes, int] | None:
        """(mincore vector, address of its first page) for ``[addr, addr + nbytes)``; None if it failed."""
        start = addr // mmap.PAGESIZE * mmap.PAGESIZE
        length = addr + nbytes - start
        vec = ctypes.create_string_buffer(-(-length // mmap.PAGESIZE))
        if self._libc.mincore(ctypes.c_void_p(start), length, vec) != 0:
            return None
        return vec.raw, start

    def _share_cached(self, addr: int, nbytes: int) -> float:
        res = self._residency(addr, nbytes)
        if res is None:
            return 1.0
        raw = res[0]
        return (len(raw) - raw.count(0)) / len(raw)

    def h2d(self, dst: torch.Tensor, src: torch.Tensor, prof=None) -> bool:
        """Copy ``src`` (a view of the bank mapping) to ``dst`` on the current stream by direct
        reads, and the pieces the page cache holds by copies out of the mapping. False when ``src``
        is not in the mapping (or mincore fails), and the caller copies it the usual way."""
        addr = src.data_ptr()
        nbytes = src.numel() * src.element_size()
        if nbytes == 0 or addr < self._lo or addr + nbytes > self._hi:
            return False
        res = self._residency(addr, nbytes)
        if res is None:
            return False  # cannot tell what is cached: the mapping copy always works
        raw, first = res
        page = mmap.PAGESIZE
        share = self.cached_share

        def cached(p: int, n: int) -> bool:
            lo = (addr + p - first) // page
            hi = -(-(addr + p + n - first) // page)
            return (hi - lo - raw.count(0, lo, hi)) >= share * (hi - lo)

        d = dst.reshape(-1).view(torch.uint8)
        stream = torch.cuda.current_stream(dst.device)
        events = self._events
        waited = [0.0]

        def before_group(s: int) -> None:
            t = time.perf_counter()
            events[s].synchronize()  # the DMA that last read this set is done
            waited[0] += time.perf_counter() - t

        def sink(p: int, host: torch.Tensor, s: int) -> None:
            d[p:p + host.numel()].copy_(host, non_blocking=True)  # async DMA on the stream
            events[s].record(stream)

        source = src.reshape(-1).view(torch.uint8)
        faults = _major_faults() if prof is not None else 0
        started = time.perf_counter()
        with self._lock:
            copied = self.reader.read(addr + self._file_base, nbytes, sink, before_group,
                                      source=source, cached=cached)
        if prof is not None:
            elapsed = time.perf_counter() - started
            prof.staged_piece(waited[0], elapsed - waited[0], nbytes, _major_faults() - faults)
        self.bytes_cached += copied
        self.bytes_read += nbytes - copied
        return True

    def close(self) -> None:
        self.reader.close()
