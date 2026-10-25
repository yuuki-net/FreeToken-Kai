"""The ``--moe-bank-ram`` bank file: one file for every MoE layer, and the only copy it has to be.

Version 1 was one file per pipeline rank, derived from the checkpoint and thrown away whenever
anything about the run changed -- the placement, the resident count, the layer split. That made
it a second copy of the checkpoint's experts (63.4 GiB for Qwen3.8-Flash-Next), and a copy that
could not stand on its own. This version is built so that it can be the canonical copy instead
(``ft bank pack`` then drops the experts from the checkpoint; moe/bank_pack.py):

* **Every MoE layer of the model, indexed by its global bank-layer id.** A rank maps the whole
  file and uses only its own layers, so ``--pp-size`` / ``--pp-layers`` can change without a
  byte being rewritten, and ranks of one run write disjoint blocks of the same file.
* **Nothing run-dependent in the data.** A layer's rows are in frequency order, and that order
  does not depend on how many of them are resident -- ``hot_per_layer`` is decided at startup
  from ``--moe-bank-ram``, not stored. Changing the budget rewrites nothing either.
* **The row order is per layer and mutable**, in a manifest inside the file. A new histogram
  reorders a layer in place (``BankFile.reorder_layer``), behind a journal that holds the
  layer's old bytes until the new order is committed: with no checkpoint to fall back on, a
  crash in the middle must not be able to cost a layer.
* **A layer exists only once it is committed.** Blocks are written and synced first, then the
  manifest names the layer with its order and a SHA-256 per block. A process killed mid-write
  leaves a layer the manifest does not name -- not a file full of zeros that looks reusable,
  which is what version 1 left behind (its header was complete before the first block was).

Layout::

    [0:16]                 b"FTMB", u32 version, u64 header length
    header JSON            geometry: num_experts, layers, banks, meta (kind, kernel, fingerprint)
    slot A, slot B         the manifest, twice; each b"FTMS", seq, length, sha256, JSON
    blocks                 grouped by layer, then bank; each 4096-aligned, rows contiguous

The manifest is written to the older slot and synced, so a torn write leaves the other slot --
the previous state -- intact. The slot size is fixed at creation from the geometry: a
permutation of 0..E-1 renders to the same JSON length whatever the order, so the largest
manifest the file will ever hold is known up front.

The blocks stay contiguous per layer because the CPU executor is handed one base pointer and the
row shape per layer (guides/17 §31); nothing here changes what the kernels see. Grouping them by
layer makes a rank's layers one contiguous range of the file, and a rank maps only that range: a
private (copy-on-write) mapping is charged against the commit limit for its whole length, and the
whole file is twice a rank's share.

Nothing in this module needs a GPU.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
import re
import struct
import threading

MAGIC = b"FTMB"
VERSION = 2
ALIGN = 4096
BANK_FILE_NAME = "bank.ftmb"

_PREAMBLE = struct.Struct("<4sIQ")
_SLOT_MAGIC = b"FTMS"
_SLOT_HEAD = struct.Struct("<4sIQQ32s")  # magic, reserved, seq, body length, sha256(body)
_JOURNAL_MAGIC = b"FTMJ"
_JOURNAL_HEAD = struct.Struct("<4sIQ")  # magic, version, header length
_JOURNAL_RE = re.compile(r"\.journal\.L(\d+)$")
# Room in each slot beyond the largest manifest the geometry can produce: the canonical flag
# and anything a later version adds without changing the data offset.
_SLOT_SPARE = 64 << 10
_PWRITE_MAX = 1 << 30


class BankFileError(RuntimeError):
    """The bank file cannot serve what was asked of it; the message says what is missing."""


def _align_up(n: int, a: int = ALIGN) -> int:
    return -(-n // a) * a


def _dumps(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _dtype_name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def dtype_of(name: str):
    import torch

    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"unknown dtype {name!r} in a mapped bank header")
    return dt


# ---------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------
def _norm_banks(banks) -> list[tuple[str, tuple[int, ...], str, int]]:
    return [(str(n), tuple(int(x) for x in s), str(d), int(b)) for n, s, d, b in banks]


class MappedBankLayout:
    """Where every (bank, layer) block sits. Immutable for the life of the file.

    ``banks`` is the geometry of every layer; ``layer_banks`` overrides it for the layers whose rows
    differ (Kai: a GGUF keeps each layer's own ggml types, so a GSQ-RCO build has three row sizes).
    A file without overrides is the version-2 file as it always was."""

    def __init__(self, num_experts: int, layers, banks, meta: dict | None = None, layer_banks: dict | None = None):
        self.num_experts = int(num_experts)
        self.layers = [int(x) for x in layers]
        if sorted(set(self.layers)) != self.layers:
            raise ValueError(f"bank layers must be strictly increasing, got {self.layers[:8]}...")
        # [(name, row shape, dtype name, bytes per row)]
        self.banks = _norm_banks(banks)
        self.layer_banks = {
            int(layer): _norm_banks(b) for layer, b in (layer_banks or {}).items() if _norm_banks(b) != self.banks
        }
        if any(layer not in self.layers for layer in self.layer_banks):
            raise ValueError(f"per-layer bank geometry for layers outside {self.layers[:8]}...")
        self.meta = json.loads(_dumps(meta or {}))  # normalized: what a read gives back
        self._block = {
            (name, layer): _align_up(self.num_experts * row_bytes)
            for layer in self.layers for name, _, _, row_bytes in self.banks_of(layer)
        }
        head = _PREAMBLE.size + len(self._header_json())
        self.slot_offset = _align_up(head)
        self.slot_bytes = _align_up(_SLOT_HEAD.size + self._largest_manifest() + _SLOT_SPARE)
        self.data_offset = self.slot_offset + 2 * self.slot_bytes
        self._offsets: dict[tuple[str, int], int] = {}
        pos = self.data_offset
        for layer in self.layers:
            for name, _, _, _ in self.banks_of(layer):
                self._offsets[(name, layer)] = pos
                pos += self._block[(name, layer)]
        self._total = pos

    def banks_of(self, layer: int) -> list[tuple[str, tuple[int, ...], str, int]]:
        """One layer's geometry: its own where it has one, else the file's."""
        return self.layer_banks.get(int(layer), self.banks)

    def _header_json(self) -> bytes:
        header = {
            "version": VERSION,
            "num_experts": self.num_experts,
            "layers": self.layers,
            "banks": [[n, list(s), d, b] for n, s, d, b in self.banks],
            "meta": self.meta,
        }
        if self.layer_banks:
            header["layer_banks"] = {str(l): [[n, list(s), d, b] for n, s, d, b in banks] for l, banks in self.layer_banks.items()}
        return _dumps(header)

    def _largest_manifest(self) -> int:
        order = list(range(self.num_experts))
        digest = {name: "0" * 64 for name, _, _, _ in self.banks}
        full = {"layers": {str(layer): {"order": order, "digest": digest} for layer in self.layers}}
        return len(_dumps(full))

    def block_bytes(self, name: str, layer: int | None = None) -> int:
        """Bytes reserved for one (bank, layer) block, alignment padding included (``layer`` None:
        the file's own geometry)."""
        if layer is None:
            return _align_up(self.num_experts * self.row_bytes(name))
        return self._block[(name, int(layer))]

    def row_bytes(self, name: str, layer: int | None = None) -> int:
        for bank_name, _, _, rb in (self.banks if layer is None else self.banks_of(layer)):
            if bank_name == name:
                return rb
        raise KeyError(name)

    def span_bytes(self, name: str, layer: int | None = None) -> int:
        """Bytes of one block that hold rows: ``num_experts * row_bytes``."""
        return self.num_experts * self.row_bytes(name, layer)

    def offset_of(self, name: str, layer: int) -> int:
        return self._offsets[(name, int(layer))]

    def total_bytes(self) -> int:
        return self._total

    def range_of(self, layers) -> tuple[int, int]:
        """``(start, end)`` file offsets covering every block of ``layers`` (contiguous when they are)."""
        layers = [int(x) for x in layers]
        start = min(self.offset_of(self.banks_of(layer)[0][0], layer) for layer in layers)
        end = max(
            self.offset_of(self.banks_of(layer)[-1][0], layer) + self._block[(self.banks_of(layer)[-1][0], layer)]
            for layer in layers
        )
        return start, end

    def layer_bytes(self, layer: int | None = None) -> int:
        """Bytes of rows one layer holds across every bank (``layer`` None: the largest layer)."""
        if layer is None:
            return max(self.layer_bytes(l) for l in self.layers) if self.layer_banks else sum(
                self.span_bytes(name) for name, _, _, _ in self.banks
            )
        return sum(self.span_bytes(name, layer) for name, _, _, _ in self.banks_of(layer))

    def layers_bytes(self, layers) -> int:
        """Bytes of rows ``layers`` hold together."""
        return sum(self.layer_bytes(l) for l in layers)

    def header_blob(self) -> bytes:
        body = self._header_json()
        head = _PREAMBLE.pack(MAGIC, VERSION, len(body)) + body
        return head + b"\0" * (self.slot_offset - len(head))

    def mismatch(self, other: "MappedBankLayout", ignore=()) -> str | None:
        """What differs between two geometries, in words; None when they agree.

        ``ignore``: meta keys not to compare (a packed checkpoint has no shards to stamp).
        """
        if self.num_experts != other.num_experts:
            return f"{self.num_experts} experts per layer against {other.num_experts}"
        if self.layers != other.layers:
            return f"MoE layers {self.layers[0]}..{self.layers[-1]} against {other.layers[0]}..{other.layers[-1]}"
        if self.banks != other.banks:
            mine = {n: (s, d) for n, s, d, _ in self.banks}
            theirs = {n: (s, d) for n, s, d, _ in other.banks}

            def fmt(x):
                return f"{list(x[0])} {x[1]}" if x else "absent"

            differ = [n for n in dict.fromkeys([*mine, *theirs]) if mine.get(n) != theirs.get(n)]
            return "banks " + ", ".join(f"{n} {fmt(mine.get(n))} against {fmt(theirs.get(n))}" for n in differ)
        if self.layer_banks != other.layer_banks:
            differ = sorted(l for l in set(self.layer_banks) | set(other.layer_banks) if self.banks_of(l) != other.banks_of(l))
            return f"the per-layer bank geometry of layers {differ[:8]}"
        for key in sorted(set(self.meta) | set(other.meta)):
            if key not in ignore and self.meta.get(key) != other.meta.get(key):
                return f"{key} {self.meta.get(key)!r} against {other.meta.get(key)!r}"
        return None

    @classmethod
    def from_json(cls, d: dict) -> "MappedBankLayout":
        return cls(
            d["num_experts"], d["layers"],
            [(n, tuple(s), t, b) for n, s, t, b in d["banks"]],
            d.get("meta") or {},
            layer_banks={int(l): [(n, tuple(s), t, b) for n, s, t, b in banks] for l, banks in (d.get("layer_banks") or {}).items()},
        )

    @classmethod
    def read(cls, path: str) -> "MappedBankLayout":
        with open(path, "rb") as f:
            raw = f.read(_PREAMBLE.size)
            if len(raw) < _PREAMBLE.size:
                raise ValueError(f"{path}: too short to be a mapped bank file")
            magic, version, body_len = _PREAMBLE.unpack(raw)
            if magic != MAGIC:
                raise ValueError(f"{path}: not a mapped bank file")
            if version != VERSION:
                raise ValueError(
                    f"{path}: mapped bank version {version}, expected {VERSION}"
                    + (" (a per-rank file from an older build; it is not used any more and can "
                       "be deleted)" if version == 1 else "")
                )
            return cls.from_json(json.loads(f.read(body_len).decode("utf-8")))


def layout_from_specs(specs, layers, num_experts: int, meta: dict | None = None) -> MappedBankLayout:
    """Geometry from a kernel layout (``{role: BankSpec}``), before any bank exists.

    GPU-resident roles (marlin / b12x alphas) are not host banks and have no place in the file.
    """
    import math

    import torch

    banks = []
    for name, spec in specs.items():
        if getattr(spec, "resident", False):
            continue
        itemsize = torch.empty((), dtype=spec.dtype).element_size()
        banks.append((name, tuple(spec.shape), _dtype_name(spec.dtype), math.prod(spec.shape) * itemsize))
    return MappedBankLayout(num_experts, layers, banks, meta)


def layout_from_sample(sample, layers, num_experts: int, meta: dict | None = None) -> MappedBankLayout:
    """Geometry from one layer's banks ``{name: tensor | HostBank}`` (loaders without a method)."""
    banks = []
    for name, bank in sample.items():
        t = getattr(bank, "tensor", bank)
        row = t[0]
        banks.append((name, tuple(row.shape), _dtype_name(t.dtype), row.numel() * t.element_size()))
    return MappedBankLayout(num_experts, layers, banks, meta)


def layout_from_layer_specs(per_layer: dict, layers, num_experts: int, meta: dict | None = None) -> MappedBankLayout:
    """Geometry from each layer's own ``{name: (row shape, dtype)}`` (Kai: GGUF experts, one ggml type
    per layer): the most common geometry is the file's, the others are per-layer overrides."""
    import math
    from collections import Counter

    import torch

    def banks(spec):
        return tuple(
            (name, tuple(shape), _dtype_name(dtype), math.prod(shape) * torch.empty((), dtype=dtype).element_size())
            for name, (shape, dtype) in spec.items()
        )

    geometry = {int(layer): banks(spec) for layer, spec in per_layer.items()}
    common = Counter(geometry.values()).most_common(1)[0][0]
    return MappedBankLayout(num_experts, layers, list(common), meta, layer_banks={l: list(g) for l, g in geometry.items() if g != common})


# ---------------------------------------------------------------------------------------
# locking: ranks of one run share the file
# ---------------------------------------------------------------------------------------
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def file_lock(path: str):
    """Exclusive across processes (flock on ``<path>.lock``) and across threads of this one.

    A separate lock file rather than the bank itself, because creating or replacing the bank
    swaps the inode a lock on it would be held against.
    """
    key = os.path.abspath(path)
    with _THREAD_LOCKS_GUARD:
        tlock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    with tlock:
        try:
            import fcntl
        except ImportError:  # not POSIX: one process, the thread lock is all there is
            yield
            return
        try:
            fd = os.open(key + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
        except OSError:
            # a read-only directory (a packed checkpoint on a shared mount): nobody can be
            # writing the file from another process either
            yield
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _pwrite_all(fd: int, data, offset: int) -> None:
    view = memoryview(data).cast("B")
    done = 0
    while done < len(view):
        n = os.pwrite(fd, view[done:done + _PWRITE_MAX], offset + done)
        if n <= 0:
            raise OSError(f"short write at offset {offset + done}")
        done += n


def _pread_into(fd: int, buf, offset: int) -> None:
    view = memoryview(buf).cast("B")
    done = 0
    while done < len(view):
        n = os.preadv(fd, [view[done:done + _PWRITE_MAX]], offset + done)
        if n <= 0:
            raise BankFileError(f"unexpected end of file at offset {offset + done}")
        done += n


def _drop_cache(fd: int, offset: int, length: int) -> None:
    try:
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------------------
class BankFile:
    """An open bank file: geometry, manifest, and the writes that keep it consistent."""

    def __init__(self, path: str, layout: MappedBankLayout, fd: int):
        self.path, self.layout, self._fd = path, layout, fd

    # ----- open / create ------------------------------------------------------------
    @classmethod
    def open(cls, path: str, writable: bool = True) -> "BankFile":
        layout = MappedBankLayout.read(path)
        size = os.path.getsize(path)
        if size < layout.total_bytes():
            raise BankFileError(
                f"{path}: {size} bytes, the geometry in its header needs {layout.total_bytes()}"
            )
        flags = (os.O_RDWR if writable else os.O_RDONLY) | getattr(os, "O_BINARY", 0)
        return cls(path, layout, os.open(path, flags))

    @classmethod
    def create(cls, path: str, layout: MappedBankLayout) -> "BankFile":
        """Write a new, empty file and move it into place (replacing whatever is there).

        Call under ``file_lock(path)``. Built beside the target and renamed, so a reader never
        sees a header without its slots.
        """
        tmp = f"{path}.tmp{os.getpid()}"
        fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o644)
        try:
            _pwrite_all(fd, layout.header_blob(), 0)
            os.ftruncate(fd, layout.total_bytes())  # sparse: blocks take space as they are written
            bank = cls(path, layout, fd)
            bank._write_slot(0, 1, {"layers": {}})
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        os.replace(tmp, path)
        _fsync_dir(path)
        return bank

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----- manifest -----------------------------------------------------------------
    def _slot_pos(self, slot: int) -> int:
        return self.layout.slot_offset + slot * self.layout.slot_bytes

    def _read_slots(self):
        """``(seq, slot, manifest)`` of the newest intact slot; ``(0, -1, empty)`` when neither is."""
        best = None
        for slot in (0, 1):
            raw = os.pread(self._fd, self.layout.slot_bytes, self._slot_pos(slot))
            if len(raw) < _SLOT_HEAD.size:
                continue
            magic, _, seq, n, digest = _SLOT_HEAD.unpack(raw[:_SLOT_HEAD.size])
            if magic != _SLOT_MAGIC or n > len(raw) - _SLOT_HEAD.size:
                continue
            body = raw[_SLOT_HEAD.size:_SLOT_HEAD.size + n]
            if hashlib.sha256(body).digest() != digest:
                continue
            if best is None or seq > best[0]:
                best = (seq, slot, body)
        if best is None:
            return 0, -1, {"layers": {}}
        return best[0], best[1], json.loads(best[2].decode("utf-8"))

    def _write_slot(self, slot: int, seq: int, manifest: dict) -> None:
        body = _dumps(manifest)
        if _SLOT_HEAD.size + len(body) > self.layout.slot_bytes:
            raise BankFileError(
                f"{self.path}: manifest of {len(body)} bytes does not fit its {self.layout.slot_bytes}-byte slot"
            )
        head = _SLOT_HEAD.pack(_SLOT_MAGIC, 0, seq, len(body), hashlib.sha256(body).digest())
        blob = head + body
        _pwrite_all(self._fd, blob + b"\0" * (self.layout.slot_bytes - len(blob)), self._slot_pos(slot))
        os.fsync(self._fd)

    def manifest(self) -> dict:
        return self._read_slots()[2]

    def _update_manifest(self, change) -> dict:
        """Read-modify-write of the manifest under the file lock; ``change(manifest)`` mutates."""
        with file_lock(self.path):
            seq, slot, manifest = self._read_slots()
            manifest.setdefault("layers", {})
            change(manifest)
            self._write_slot(1 - slot if slot in (0, 1) else 0, seq + 1, manifest)
            return manifest

    def layer_state(self, layer: int, manifest: dict | None = None):
        """``(order, {bank: sha256})`` of a committed layer, or None."""
        entry = (manifest if manifest is not None else self.manifest())["layers"].get(str(int(layer)))
        if entry is None:
            return None
        return [int(x) for x in entry["order"]], dict(entry["digest"])

    def present_layers(self, manifest: dict | None = None) -> list[int]:
        m = manifest if manifest is not None else self.manifest()
        return sorted(int(k) for k in m["layers"])

    def canonical_for(self, manifest: dict | None = None) -> str | None:
        """The slim checkpoint this file is the only copy of the experts for, if any."""
        return (manifest if manifest is not None else self.manifest()).get("canonical_for")

    def set_canonical_for(self, what: str | None) -> None:
        def change(m):
            if what is None:
                m.pop("canonical_for", None)
            else:
                m["canonical_for"] = what

        self._update_manifest(change)

    def _commit_layer(self, layer: int, order, digest) -> None:
        def change(m):
            m["layers"][str(int(layer))] = {"order": [int(x) for x in order], "digest": dict(digest)}

        self._update_manifest(change)

    def _forget_layer(self, layer: int) -> None:
        self._update_manifest(lambda m: m["layers"].pop(str(int(layer)), None))

    # ----- blocks -------------------------------------------------------------------
    def read_block(self, name: str, layer: int):
        """One block's rows as a fresh uint8 tensor ``[num_experts * row_bytes]`` (physical order)."""
        import torch

        n = self.layout.span_bytes(name, layer)
        out = torch.empty(n, dtype=torch.uint8)
        _pread_into(self._fd, out.numpy(), self.layout.offset_of(name, layer))
        return out

    def drop_cache(self, layer: int) -> None:
        """Hand the page cache for one layer's blocks back (large sequential passes)."""
        for name, _, _, _ in self.layout.banks_of(layer):
            _drop_cache(self._fd, self.layout.offset_of(name, layer), self.layout.block_bytes(name, layer))

    def check_digest(self, layer: int, blocks: dict | None = None) -> dict:
        """Recompute a committed layer's block hashes; raises when one disagrees.

        Returns the blocks read, so a caller that needs them does not read twice.
        """
        state = self.layer_state(layer)
        if state is None:
            raise BankFileError(f"{self.path}: layer {layer} is not in the file")
        _, digest = state
        blocks = dict(blocks or {})
        for name, _, _, _ in self.layout.banks_of(layer):
            if name not in blocks:
                blocks[name] = self.read_block(name, layer)
            got = hashlib.sha256(memoryview(blocks[name].numpy())).hexdigest()
            if got != digest.get(name):
                raise BankFileError(
                    f"{self.path}: layer {layer} bank {name!r} does not match the hash it was "
                    f"committed with ({got[:12]} against {str(digest.get(name))[:12]}) -- the file "
                    f"is damaged"
                )
        return blocks

    def write_layer(self, layer: int, banks, order) -> None:
        """Write one layer from its banks in LOGICAL row order, permuted by ``order``, and commit it.

        ``banks``: ``{name: tensor | HostBank}`` shaped ``[num_experts, *row]``. A layer that was
        committed before is forgotten first: overwriting it in place and dying half way must
        not leave a manifest that vouches for the old bytes.
        """
        import torch

        layer = int(layer)
        if self.canonical_for():
            raise BankFileError(
                f"{self.path} is the only copy of the experts of {self.canonical_for()}; it is "
                f"reordered in place, never rewritten from a checkpoint"
            )
        order = [int(x) for x in order]
        if sorted(order) != list(range(self.layout.num_experts)):
            raise ValueError(f"layer {layer}: order is not a permutation of 0..{self.layout.num_experts - 1}")
        if self.layer_state(layer) is not None:
            self._forget_layer(layer)
        idx = torch.as_tensor(order, dtype=torch.long)
        digest = {}
        for name, row_shape, dtype_name, _ in self.layout.banks_of(layer):
            src = getattr(banks[name], "tensor", banks[name])
            if tuple(src.shape) != (self.layout.num_experts, *row_shape) or _dtype_name(src.dtype) != dtype_name:
                raise ValueError(
                    f"layer {layer} bank {name!r}: {tuple(src.shape)} {_dtype_name(src.dtype)}, the file "
                    f"holds {(self.layout.num_experts, *row_shape)} {dtype_name}"
                )
            block = src.index_select(0, idx).contiguous()
            raw = block.flatten().view(torch.uint8).numpy()
            _pwrite_all(self._fd, raw, self.layout.offset_of(name, layer))
            digest[name] = hashlib.sha256(memoryview(raw)).hexdigest()
            del block, raw
        os.fdatasync(self._fd)
        self._commit_layer(layer, order, digest)

    # ----- reorder in place ---------------------------------------------------------
    def journal_path(self, layer: int) -> str:
        return f"{self.path}.journal.L{int(layer)}"

    def reorder_layer(self, layer: int, new_order) -> bool:
        """Put one committed layer's rows into ``new_order`` without a second copy of the file.

        The layer is read and checked against its hashes first -- permuting damaged bytes and
        committing fresh hashes over them would bless the damage. Then its old bytes go to a
        journal, synced, before the first in-place write; the journal goes only after the new
        order is committed. ``recover_journals`` finishes either way after a crash. The extra
        disk this needs is one layer (1.3 GiB for Flash-Next), not the file.

        Returns False when the layer was already in that order.
        """
        import torch

        layer = int(layer)
        state = self.layer_state(layer)
        if state is None:
            raise BankFileError(f"{self.path}: layer {layer} is not in the file")
        old_order, old_digest = state
        new_order = [int(x) for x in new_order]
        if sorted(new_order) != list(range(self.layout.num_experts)):
            raise ValueError(f"layer {layer}: order is not a permutation of 0..{self.layout.num_experts - 1}")
        if new_order == old_order:
            return False
        blocks = self.check_digest(layer)
        position = [0] * self.layout.num_experts
        for physical, logical in enumerate(old_order):
            position[logical] = physical
        idx = torch.as_tensor([position[x] for x in new_order], dtype=torch.long)
        new_blocks, new_digest = {}, {}
        for name, _, _, row_bytes in self.layout.banks_of(layer):
            rows = blocks[name].view(self.layout.num_experts, row_bytes)
            new_blocks[name] = rows.index_select(0, idx).contiguous().flatten()
            new_digest[name] = hashlib.sha256(memoryview(new_blocks[name].numpy())).hexdigest()
        self._write_journal(layer, old_order, new_order, old_digest, new_digest, blocks)
        for name, _, _, _ in self.layout.banks_of(layer):
            _pwrite_all(self._fd, new_blocks[name].numpy(), self.layout.offset_of(name, layer))
        os.fdatasync(self._fd)
        self._commit_layer(layer, new_order, new_digest)
        os.unlink(self.journal_path(layer))
        _fsync_dir(self.path)
        return True

    def _write_journal(self, layer, old_order, new_order, old_digest, new_digest, blocks) -> None:
        header = _dumps({
            "layer": layer, "old_order": old_order, "new_order": new_order,
            "old_digest": old_digest, "new_digest": new_digest,
            "blocks": [[name, self.layout.span_bytes(name, layer)] for name, _, _, _ in self.layout.banks_of(layer)],
        })
        path = self.journal_path(layer)
        h = hashlib.sha256()
        with open(path, "wb") as f:
            head = _JOURNAL_HEAD.pack(_JOURNAL_MAGIC, 1, len(header)) + header
            f.write(head)
            h.update(head)
            for name, _, _, _ in self.layout.banks_of(layer):
                data = memoryview(blocks[name].numpy())
                f.write(data)
                h.update(data)
            f.write(h.digest())
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(path)

    @staticmethod
    def _read_journal(path: str):
        """``(header, {bank: uint8 tensor})`` of an intact journal, or None."""
        import torch

        try:
            with open(path, "rb") as f:
                raw = f.read(_JOURNAL_HEAD.size)
                if len(raw) < _JOURNAL_HEAD.size:
                    return None
                magic, _, n = _JOURNAL_HEAD.unpack(raw)
                if magic != _JOURNAL_MAGIC:
                    return None
                h = hashlib.sha256(raw)
                body = f.read(n)
                h.update(body)
                header = json.loads(body.decode("utf-8"))
                blocks = {}
                for name, nbytes in header["blocks"]:
                    t = torch.empty(int(nbytes), dtype=torch.uint8)
                    got = f.readinto(memoryview(t.numpy()))
                    if got != int(nbytes):
                        return None
                    h.update(memoryview(t.numpy()))
                    blocks[name] = t
                if f.read(32) != h.digest():
                    return None
                return header, blocks
        except (OSError, ValueError, KeyError):
            return None

    def recover_journals(self, layers=None, log=None) -> list[int]:
        """Finish or roll back reorders a crash interrupted. Returns the layers touched.

        A journal that does not check out was never finished, and nothing is written in place
        before it is -- it is discarded. An intact one whose new order is already committed is
        simply removed; otherwise the old bytes go back and the old order stands.
        """
        log = log or (lambda _m: None)
        wanted = None if layers is None else {int(x) for x in layers}
        touched = []
        for path in sorted(glob.glob(glob.escape(self.path) + ".journal.L*")):
            m = _JOURNAL_RE.search(path)
            if m is None:
                continue
            layer = int(m.group(1))
            if wanted is not None and layer not in wanted:
                continue
            got = self._read_journal(path)
            if got is None:
                os.unlink(path)
                log(f"--moe-bank-ram: discarded an unfinished reorder journal for layer {layer} (nothing had been written yet)")
                continue
            header, blocks = got
            state = self.layer_state(layer)
            if state is not None and state[0] == header["new_order"] and state[1] == header["new_digest"]:
                os.unlink(path)
                log(f"--moe-bank-ram: layer {layer}: the interrupted reorder had been committed; journal removed")
            else:
                for name, _ in header["blocks"]:
                    _pwrite_all(self._fd, blocks[name].numpy(), self.layout.offset_of(name, layer))
                os.fdatasync(self._fd)
                self._commit_layer(layer, header["old_order"], header["old_digest"])
                self.check_digest(layer)
                os.unlink(path)
                log(f"--moe-bank-ram: layer {layer}: an interrupted reorder was rolled back from its journal")
            _fsync_dir(path)
            touched.append(layer)
        return touched


def free_bytes(path: str) -> int | None:
    """Space available to this user on the filesystem holding ``path`` (or its directory)."""
    probe = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    try:
        st = os.statvfs(probe)
    except (OSError, AttributeError):
        return None
    return st.f_bavail * st.f_frsize


def allocated_bytes(path: str) -> int:
    """Bytes a (possibly sparse) file actually occupies on disk."""
    try:
        return os.stat(path).st_blocks * 512
    except (OSError, AttributeError):
        return 0
