"""Prefix cache on disk (``--prefix-disk-cache``): keep what the radix tree lets go, and what a
restart throws away, so a long prompt seen before is read back instead of prefilled again.

Stage 1 covers hybrid GDN models (Qwen3.5-MoE, Qwen3.8-Flash-Next) on one rank. See
``kvcache/prefix_disk_store.py`` for the file format and what makes an entry trustworthy; this
module is the part that touches the pools and the tree.

**What an entry is.** A tree node that owns a live GDN snapshot marks a boundary ``L`` a request
can resume from. An entry is that node's whole path: the ids ``[0, L)``, the KV pages under them
(every storage layer of the paged slab -- the ``--spec-mtp`` draft head's layer included -- and
the ``--kv-cache-dtype`` scales, and Flash-Next's compressed index rows), and the snapshot slot.
Boundaries without a snapshot are not reuse points in memory, so they are not written either.

**When it is written: at idle.** ``persist_idle`` runs where the scheduler is about to block for
the next message, so no forward is in flight and every tree-owned page and slot holds its final
bytes. It copies the not-yet-written snapshot nodes to host memory (bounded per pass) on the
scheduler thread and hands the file writes to a writer thread. Nothing on the eviction or
commit paths changes -- that is where ``guides/25``'s bugs lived.

**When it is read: at admission.** For the request at the head of the prefill queue, the
store is asked for the deepest entry that is a prefix of the prompt. If it goes meaningfully
deeper than the tree's own match, the request waits (the queue behind it too) while a reader
thread loads the file; then ``CacheManager.insert_restored_prefix`` takes pages and a GDN slot
the ordinary way, uploads the entry into them and inserts the node into the tree, and admission
proceeds exactly as for an in-memory hit.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch

from freetoken.kvcache.prefix_disk_store import DiskEntry, PrefixDiskStore, dtype_name, prefix_key
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .cache import CacheManager
    from .utils import PendingReq

logger = init_logger(__name__)

# Prefixes shorter than this are not written: they are quick to prefill again, and every entry
# carries a whole GDN snapshot (tens of MiB) however short its prompt is.
MIN_TOKENS = 1024
# A disk entry is read only when it reaches at least this much deeper than the tree's own hit
# (and at least 1/8 of its length): reading and uploading is not free either.
MIN_GAIN_TOKENS = 512
# A read is abandoned (the request prefills normally) once it has taken longer than recomputing
# the gain could plausibly take: READ_DEADLINE_BASE_S plus the gain at this many tokens per second
# -- faster than any prefill these machines do, so the wait never exceeds the recompute. Seen on
# an RTX 2060 host under memory pressure: an 83 MiB entry that took 7.95 s to read, for a prompt
# that prefills in 3 s.
READ_DEADLINE_BASE_S = 1.0
READ_DEADLINE_TOKENS_PER_S = 2000
# Host memory one idle pass may queue for the writer before it stops and leaves the rest for
# the next idle.
IDLE_PASS_BYTES = 512 << 20
# Largest single device<->host copy (the gather makes a temporary of this size in VRAM).
COPY_CHUNK_BYTES = 32 << 20


class PrefixDiskUnsupported(ValueError):
    pass


# ---------------------------------------------------------------------------------- layout
class PoolLayout:
    """Reads and writes one prefix's pages and one GDN slot, for exactly the pool classes it
    knows. Pool tensors are looked up on every call: a runtime rebuild replaces them."""

    def __init__(self, kv_pool, linear_pool, page_size: int) -> None:
        from freetoken.kvcache.mha_pool import MHAKVCache
        from freetoken.kvcache.qsa_pool import QSAKVCache

        kind = type(kv_pool)
        # exact classes only: a subclass (BSA, DSA, ...) adds tiers this does not copy
        if kind not in (MHAKVCache, QSAKVCache):
            raise PrefixDiskUnsupported(
                f"the KV pool is {kind.__name__}; the prefix disk cache knows how to store the "
                f"plain paged pool (Qwen3.5-MoE) and the QSA pool (Qwen3.8-Flash-Next) only"
            )
        if linear_pool is None:
            raise PrefixDiskUnsupported("the model has no GDN state pool (not a hybrid model)")
        self.kv_pool = kv_pool
        self.linear_pool = linear_pool
        self.page_size = page_size
        self.qsa = kind is QSAKVCache
        if self.qsa:
            ratio = kv_pool.index_ratio
            if page_size % ratio:
                raise PrefixDiskUnsupported(f"page_size {page_size} is not a multiple of index_ratio {ratio}")
            self.rows_per_page = page_size // ratio

    # ---- what is stored, per page / per slot
    def _page_tensors(self) -> List[Tuple[str, torch.Tensor, int]]:
        """(name, buffer, page axis). The page axis of the K/V slabs is 2; the QSA index slab is
        row-flat, one row per index_ratio tokens, on axis 1."""
        p = self.kv_pool
        out = [("kv", p._kv_buffer, 2)]
        if p._scale_buffer is not None:
            out.append(("kv_scale", p._scale_buffer, 2))
        if self.qsa:
            out.append(("cmp_k", p._cmp_k_buffer, 1))
        return out

    def _state_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        lp = self.linear_pool
        out = [("conv", lp.conv_states), ("recurrent", lp.recurrent_states)]
        for name in sorted(lp.slot_states):
            out.append((f"slot.{name}", lp.slot_states[name]))
        return out

    def _item_shape(self, name: str, buf: torch.Tensor, axis: int) -> Tuple[int, ...]:
        shape = list(buf.shape)
        shape[axis] = self.rows_per_page if name == "cmp_k" else 1
        return tuple(shape)

    def describe(self) -> dict:
        """The part of the fingerprint that pins the stored layout (page counts excluded: an
        entry can be restored into any pages)."""
        pages = {}
        for name, buf, axis in self._page_tensors():
            shape = list(buf.shape)
            shape[axis] = None
            pages[name] = [dtype_name(buf.dtype), shape]
        states = {}
        for name, buf in self._state_tensors():
            shape = list(buf.shape)
            shape[1] = None
            states[name] = [dtype_name(buf.dtype), shape]
        p, lp = self.kv_pool, self.linear_pool
        return {
            "kv_pool": type(p).__name__,
            "kv_quant": getattr(p.kv_quant, "name", None),
            "kv_layer_map": p._layer_map,
            "index_ratio": p.index_ratio if self.qsa else None,
            "pages": pages,
            "gdn_layers": sorted(lp._local_index),
            "states": states,
        }

    def entry_bytes(self, length: int) -> int:
        n_pages = length // self.page_size
        total = 4 * length
        for name, buf, axis in self._page_tensors():
            per = buf.element_size() * _numel(self._item_shape(name, buf, axis))
            total += per * n_pages
        for _, buf in self._state_tensors():
            total += buf.element_size() * (buf.numel() // buf.shape[1])
        return total

    def _rows(self, name: str, pages: torch.Tensor) -> torch.Tensor:
        if name != "cmp_k":
            return pages
        r = self.rows_per_page
        return (pages.unsqueeze(1) * r + torch.arange(r, device=pages.device)).reshape(-1)

    # ---- device -> host
    def read_pages(self, pages: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        out = []
        for name, buf, axis in self._page_tensors():
            idx = self._rows(name, pages.to(buf.device, torch.long))
            out.append((name, _gather_to_host(buf, axis, idx)))
        return out

    def read_state(self, slot: int) -> List[Tuple[str, torch.Tensor]]:
        # state_views finds the snapshot on either tier (a host one is waited for first)
        out = []
        for name, view in self.linear_pool.state_views(slot):
            host = torch.empty(view.shape, dtype=view.dtype)
            for layer in range(view.shape[0]):
                host[layer].copy_(view[layer])
            out.append((name, host))
        return out

    # ---- host -> device
    def write_pages(self, pages: torch.Tensor, src: Dict[str, torch.Tensor], first: int) -> None:
        """Upload source pages ``[first, first + len(pages))`` of ``src`` into ``pages``."""
        for name, buf, axis in self._page_tensors():
            host = src[name]
            want = list(buf.shape)
            want[axis] = host.shape[axis]
            if list(host.shape) != want or host.dtype != buf.dtype:
                raise ValueError(f"{name}: entry holds {tuple(host.shape)} {host.dtype}, pool wants {tuple(want)} {buf.dtype}")
            idx = self._rows(name, pages.to(buf.device, torch.long))
            start = first * (self.rows_per_page if name == "cmp_k" else 1)
            _scatter_from_host(buf, axis, idx, host, start)

    def write_state(self, slot: int, src: Dict[str, torch.Tensor]) -> None:
        for name, buf in self._state_tensors():
            host = src[name]
            if tuple(host.shape) != (buf.shape[0], *buf.shape[2:]) or host.dtype != buf.dtype:
                raise ValueError(f"{name}: entry holds {tuple(host.shape)} {host.dtype}, pool wants {(buf.shape[0], *buf.shape[2:])} {buf.dtype}")
            for layer in range(buf.shape[0]):
                buf[layer, slot].copy_(host[layer])


def _numel(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _gather_to_host(buf: torch.Tensor, axis: int, idx: torch.Tensor) -> torch.Tensor:
    shape = list(buf.shape)
    shape[axis] = idx.numel()
    out = torch.empty(shape, dtype=buf.dtype)
    per = buf.element_size() * (buf.numel() // max(buf.shape[axis], 1))
    step = max(1, COPY_CHUNK_BYTES // max(per, 1))
    for i in range(0, idx.numel(), step):
        j = min(idx.numel(), i + step)
        out.narrow(axis, i, j - i).copy_(buf.index_select(axis, idx[i:j]))
    return out


def _scatter_from_host(buf: torch.Tensor, axis: int, idx: torch.Tensor, host: torch.Tensor, start: int) -> None:
    per = buf.element_size() * (buf.numel() // max(buf.shape[axis], 1))
    step = max(1, COPY_CHUNK_BYTES // max(per, 1))
    for i in range(0, idx.numel(), step):
        j = min(idx.numel(), i + step)
        chunk = host.narrow(axis, start + i, j - i).to(buf.device)
        buf.index_copy_(axis, idx[i:j], chunk)


# ---------------------------------------------------------------------------------- cache
class PrefixDiskCache:
    def __init__(
        self,
        store: PrefixDiskStore,
        cache_manager: "CacheManager",
        layout: PoolLayout,
        *,
        min_tokens: int = MIN_TOKENS,
        min_gain: int = MIN_GAIN_TOKENS,
        idle_pass_bytes: int = IDLE_PASS_BYTES,
        read_deadline_base_s: float = READ_DEADLINE_BASE_S,
        read_deadline_tokens_per_s: float = READ_DEADLINE_TOKENS_PER_S,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not cache_manager.is_hybrid:
            raise PrefixDiskUnsupported("the prefix disk cache needs the hybrid radix cache")
        self.store = store
        self.cm = cache_manager
        self.layout = layout
        self.min_tokens = max(int(min_tokens), cache_manager.page_size)
        self.min_gain = int(min_gain)
        self.idle_pass_bytes = int(idle_pass_bytes)
        self.read_deadline_base_s = float(read_deadline_base_s)
        self.read_deadline_tokens_per_s = float(read_deadline_tokens_per_s)
        self._log = log or logger.info
        self._writer = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-disk-write")
        self._reader = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefix-disk-read")
        self._lock = threading.Lock()
        self._queued_bytes = 0
        self._loads: set = set()
        self.stats = {"written": 0, "write_bytes": 0, "write_failed": 0, "restored": 0,
                      "restored_tokens": 0, "restore_skipped": 0, "load_missed": 0, "load_abandoned": 0}

    # ------------------------------------------------------------------ write side
    def persist_idle(self) -> int:
        """Queue every tree snapshot not yet on disk (oldest first, bounded per pass). Must run
        with no forward in flight. Returns how many entries were queued."""
        pc = self.cm.prefix_cache
        nodes = [n for n in pc._snapshot_nodes() if not getattr(n, "disk_done", False)]
        if not nodes:
            return 0
        device = self.cm.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        nodes.sort(key=lambda n: n.timestamp)
        queued, queued_bytes, t0 = 0, 0, time.monotonic()
        for node in nodes:
            length = pc._path_len(node)
            if length < self.min_tokens or length % self.cm.page_size:
                node.disk_done = True
                continue
            ids = self._path_ids(node)
            key = prefix_key(self.store.digest, ids)
            if key in self.store:
                node.disk_done = True
                continue
            est = self.layout.entry_bytes(length)
            if est > self.store.capacity:
                node.disk_done = True
                continue
            with self._lock:
                if self._queued_bytes and self._queued_bytes + est > self.idle_pass_bytes:
                    break  # the rest waits for the next idle
                self._queued_bytes += est
            try:
                tensors = self._read_entry(node, ids)
            except Exception as exc:  # noqa: BLE001 -- a copy that fails loses a cache entry, not the server
                with self._lock:
                    self._queued_bytes -= est
                node.disk_done = True
                logger.warning(f"prefix disk cache: could not copy a {length}-token prefix out: {exc!r}")
                continue
            node.disk_done = True
            self._writer.submit(self._write, key, length, tensors, est)
            queued += 1
            queued_bytes += est
        if queued:
            logger.info(
                f"prefix disk cache: queued {queued} prefix(es), {queued_bytes / (1 << 30):.2f} GiB, "
                f"for disk (copied out in {time.monotonic() - t0:.2f} s)"
            )
        return queued

    def _path_ids(self, node) -> torch.Tensor:
        parts = []
        n = node
        while not n.is_root():
            parts.append(n._key)
            n = n.parent
        parts.reverse()
        return torch.cat(parts).to("cpu", torch.int32)

    def _read_entry(self, node, ids: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
        ps = self.cm.page_size
        kv = self.cm.prefix_cache._collect_kv(node).to("cpu", torch.long)
        if kv.numel() != ids.numel():
            raise RuntimeError(f"path holds {kv.numel()} KV slots for {ids.numel()} ids")
        pages = kv[::ps] // ps
        if ps > 1 and not torch.equal(kv.view(-1, ps), pages.unsqueeze(1) * ps + torch.arange(ps)):
            raise RuntimeError("the path's KV slots are not whole pages")
        return [("ids", ids), *self.layout.read_pages(pages), *self.layout.read_state(node.mamba_value)]

    def _write(self, key: str, length: int, tensors, est: int) -> None:
        t0 = time.monotonic()
        try:
            entry = self.store.save(key, length, tensors)
        except Exception as exc:  # noqa: BLE001
            self.stats["write_failed"] += 1
            logger.warning(f"prefix disk cache: writing a {length}-token prefix failed: {exc!r}")
            entry = None
        finally:
            with self._lock:
                self._queued_bytes -= est
        if entry is not None:
            self.stats["written"] += 1
            self.stats["write_bytes"] += entry.nbytes
            logger.info(
                f"prefix disk cache: wrote {length} tokens ({entry.nbytes / (1 << 20):.0f} MiB) in "
                f"{time.monotonic() - t0:.2f} s; {len(self.store)} entries, "
                f"{self.store.total_bytes / (1 << 30):.2f} of {self.store.capacity / (1 << 30):.2f} GiB"
            )

    # ------------------------------------------------------------------ read side
    def admit_gate(self, req: "PendingReq", *, can_restore: bool = True, reserve_tokens: int = 0) -> bool:
        """False while ``req`` waits for a disk entry; True when it can go on to ordinary
        admission (restored or not).

        Looking the prompt up and starting the read touch no pool, so they happen whenever the
        request is considered. The restore takes pages and a GDN slot, so it only happens when
        ``can_restore`` -- the request is the first of this admission pass, when nothing admitted
        before it holds an unallocated reservation -- and it leaves ``reserve_tokens`` (what the
        running requests' decode still needs) of KV untouched."""
        state = getattr(req, "disk_state", None)
        if state is None:
            req.disk_state = "checked"
            limit = req.input_len - 1      # admission always prefills at least the last token
            if limit < self.min_tokens:
                return True
            ids = req.input_ids[:limit]
            entry = self.store.lookup(ids, limit)
            if entry is None:
                return True
            tree = self.cm.prefix_cache.match_prefix(ids[: entry.length]).cached_len
            gain = entry.length - tree
            if gain < max(self.min_gain, entry.length // 8):
                return True
            req.disk_entry = entry
            req.disk_started = time.monotonic()
            req.disk_deadline = (req.disk_started + self.read_deadline_base_s
                                 + gain / self.read_deadline_tokens_per_s)
            fut = self._reader.submit(self._load, entry, ids[: entry.length].clone())
            with self._lock:
                self._loads.add(fut)
            # an abandoned read (the request was aborted meanwhile) must not stay "loading"
            fut.add_done_callback(self._load_finished)
            req.disk_state = fut
            state = fut
        if isinstance(state, cf.Future):
            if not state.done():
                if time.monotonic() < req.disk_deadline:
                    return False
                # the disk is slower than recomputing would be: stop waiting (the read finishes
                # in the background and its result is dropped)
                req.disk_state = "done"
                self.stats["load_abandoned"] += 1
                logger.warning(
                    f"prefix disk cache: request {req.uid} stopped waiting for its "
                    f"{req.disk_entry.length}-token entry after "
                    f"{time.monotonic() - req.disk_started:.1f} s (the read is slower than "
                    f"prefilling it would be); it prefills instead"
                )
                req.disk_entry = None
                return True
            if not can_restore:
                return False
            req.disk_state = "done"
            loaded = state.result()
            if loaded is not None:
                self._restore(req, req.disk_entry, *loaded, reserve_tokens=reserve_tokens)
            else:
                self.stats["load_missed"] += 1
            req.disk_entry = None
        return True

    def _load_finished(self, fut) -> None:
        with self._lock:
            self._loads.discard(fut)

    def _load(self, entry: DiskEntry, ids: torch.Tensor):
        t0 = time.monotonic()
        try:
            tensors = self.store.load(entry, expect_ids=ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"prefix disk cache: reading {os.path.basename(entry.path)} failed: {exc!r}")
            return None
        if tensors is None:
            return None
        self.store.touch(entry)
        return tensors, time.monotonic() - t0

    def _restore(self, req, entry: DiskEntry, tensors: Dict[str, torch.Tensor], read_s: float,
                 *, reserve_tokens: int = 0) -> None:
        ids = tensors["ids"]
        t0 = time.monotonic()

        def write(pages: torch.Tensor, first_page: int, slot: int) -> None:
            self.layout.write_pages(pages, tensors, first_page)
            self.layout.write_state(slot, tensors)

        try:
            restored, uploaded = self.cm.insert_restored_prefix(ids, write, reserve_tokens=reserve_tokens)
        except Exception as exc:  # noqa: BLE001 -- the request is admitted either way
            logger.warning(f"prefix disk cache: restoring {entry.length} tokens failed: {exc!r}")
            restored, uploaded = 0, 0
        if not restored:
            self.stats["restore_skipped"] += 1
            logger.info(
                f"prefix disk cache: request {req.uid} had a {entry.length}-token entry on disk but "
                f"it was not restored (no room for its pages or a GDN slot, or the tree has it now)"
            )
            return
        node = self.cm.prefix_cache.match_prefix(ids).node
        node.disk_done = True
        self.stats["restored"] += 1
        self.stats["restored_tokens"] += restored
        logger.info(
            f"prefix disk cache: request {req.uid} resumes at {restored} tokens from disk "
            f"(read {entry.nbytes / (1 << 20):.0f} MiB in {read_s:.2f} s, uploaded {uploaded} tokens "
            f"in {time.monotonic() - t0:.2f} s)"
        )

    def wait_for_load(self, timeout: float) -> None:
        """Block briefly on an in-progress load, so a scheduler with nothing else to do does
        not spin while the head of the queue waits for the disk."""
        with self._lock:
            loads = list(self._loads)
        if loads:
            cf.wait(loads, timeout=timeout, return_when=cf.FIRST_COMPLETED)

    @property
    def loading(self) -> bool:
        with self._lock:
            return bool(self._loads)

    def close(self, wait: bool = False) -> None:
        self._reader.shutdown(wait=wait, cancel_futures=True)
        self._writer.shutdown(wait=wait, cancel_futures=True)


# ---------------------------------------------------------------------------------- setup
def model_identity(model_path: str) -> dict:
    """What pins the weights without reading them: config.json's bytes and every top-level
    file's name, size and mtime (a re-download, a re-quantization or a copy changes one)."""
    out: dict = {"path": os.path.abspath(model_path)}
    if not os.path.isdir(model_path):
        return out
    cfg = os.path.join(model_path, "config.json")
    try:
        with open(cfg, "rb") as f:
            out["config_sha256"] = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        out["config_sha256"] = None
    files = []
    for name in sorted(os.listdir(model_path)):
        path = os.path.join(model_path, name)
        if name.startswith(".") or not os.path.isfile(path):
            continue
        st = os.stat(path)
        files.append([name, st.st_size, st.st_mtime_ns])
    out["files"] = files
    return out


def code_identity(src_dir: str | None = None) -> dict:
    """The freetoken version, and when running from a git checkout, its commit plus a digest of
    any uncommitted change to tracked files under the source directory (``python/``) -- a kernel
    fix that moves the numbers must not read entries the old code wrote. (Untracked new files are
    not seen; a change that matters lands in a tracked file.)"""
    from freetoken.version import __version__

    out: dict = {"version": __version__}
    # python/freetoken/scheduler/prefix_disk.py -> python/. The diff is taken over "." from
    # there: a pathspec of "python" would name python/python and always come back empty (the
    # first version of this did exactly that, so uncommitted changes never moved the digest).
    src = src_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        head = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10)
        if head.returncode == 0:
            out["commit"] = head.stdout.strip()
            diff = subprocess.run(["git", "-C", src, "diff", "HEAD", "--", "."],
                                  capture_output=True, timeout=30)
            if diff.returncode == 0 and diff.stdout:
                out["uncommitted"] = hashlib.sha256(diff.stdout).hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def build_fingerprint(config, layout: PoolLayout) -> dict:
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.kvcache.prefix_disk_store import FORMAT_VERSION

    return {
        "format": FORMAT_VERSION,
        "code": code_identity(),
        "model": model_identity(config.model_path),
        "dtype": str(config.dtype),
        "kv_cache_dtype": config.kv_cache_dtype or "auto",
        "dense_quant": config.dense_quant,
        "quant_backend": config.quant_backend,
        "page_size": config.page_size,
        "cache_type": config.cache_type,
        "parallel": config.parallel,
        "world_size": config.tp_info.size,
        "rank": config.tp_info.rank,
        "pp_split": list(config.pp_split) if config.pp_split else None,
        "mtp_layer_id": getattr(config.model_config, "mtp_layer_id", None),
        "gdn_chunk": CHUNK_SIZE,
        "layout": layout.describe(),
    }


def build_prefix_disk_cache(config, engine, cache_manager: "CacheManager") -> PrefixDiskCache:
    """Validate the configuration and open the store. Raises ``PrefixDiskUnsupported`` with the
    reason for anything stage 1 does not cover."""
    from freetoken.moe.bank_disk import parse_size

    flag = "--prefix-disk-cache"
    if config.tp_info.size > 1:
        raise PrefixDiskUnsupported(
            f"{flag} runs on one rank only for now; it is refused with --pp-size / --tp-size > 1 "
            f"(each rank would have to make the same restore decision at the same step)"
        )
    if getattr(config, "offline_mode", False):
        raise PrefixDiskUnsupported(f"{flag} is a server feature (it writes while the scheduler is idle)")
    if config.cache_type != "hybrid_radix" or not cache_manager.is_hybrid:
        raise PrefixDiskUnsupported(
            f"{flag} supports hybrid GDN models (Qwen3.5-MoE, Qwen3.8-Flash-Next) with their "
            f"default radix cache for now; this model runs cache type {config.cache_type!r}"
        )
    layout = PoolLayout(engine.kv_cache, engine.linear_state_pool, config.page_size)
    capacity = parse_size(config.prefix_disk_cache_size)
    store = PrefixDiskStore(
        config.prefix_disk_cache, capacity, build_fingerprint(config, layout), log=logger.warning
    )
    logger.info(
        f"{flag}: {store.dir} holds {len(store)} entries for this configuration; "
        f"{store.total_bytes / (1 << 30):.2f} of {capacity / (1 << 30):.2f} GiB used across "
        f"the directory. Prefixes of {MIN_TOKENS}+ tokens are written while idle."
    )
    return PrefixDiskCache(store, cache_manager, layout)


__all__ = [
    "PoolLayout",
    "PrefixDiskCache",
    "PrefixDiskUnsupported",
    "build_fingerprint",
    "build_prefix_disk_cache",
]
