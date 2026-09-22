"""``ft bank pack``: make the bank file the only copy of a checkpoint's routed experts.

``--moe-bank-ram`` serves the experts out of its bank file (moe/bank_file.py) and, once that file
holds every layer, never opens the checkpoint's expert tensors again. Keeping them anyway is a
second copy of the largest part of the model -- 63.4 GiB of Qwen3.8-Flash-Next's checkpoint.
This writes a checkpoint without them.

What gets dropped is decided by the bytes, not by the names: a checkpoint tensor goes only when
the bank reproduces it exactly. The kernel's ``unpack`` says which piece roles its ``pack`` kept
losslessly; NVFP4's per-tensor global scales went through fp16 on the way in, so they stay (they
are a few hundred kilobytes). Before anything is claimed, every layer is checked two ways:

1. the bank's rows equal what the loader would pack from the original checkpoint -- every bank
   role, the lossy ones included (this is what serving from the bank means); and
2. every tensor that is dropped is regenerated from the bank and compared byte for byte with the
   original, and a SHA-256 over them per (layer, role) is recorded, so ``ft bank verify`` can
   repeat the check later with no original at hand.

The slim checkpoint is a directory beside the original, never the original edited in place:
shards with no expert tensors are hard-linked (no space), shards that mix both are rewritten
without them, shards of nothing but experts are left out, the index is rewritten, and every other
file is copied. The bank file moves into it (a rename, same filesystem), so the checkpoint is one
directory again. The original is not touched and nothing is deleted -- the command ends by saying
that the original can be.

``freetoken_bank_pack.json`` in the slim directory records what was done: the original shard
headers and whole-file hashes of every shard that was rewritten or left out, which is enough to
put the original files back byte for byte from the slim directory and the bank.

Nothing here needs a GPU.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import struct
from collections import defaultdict
from dataclasses import dataclass

from .bank_file import (
    BANK_FILE_NAME,
    BankFile,
    MappedBankLayout,
    dtype_of,
    free_bytes,
    layout_from_specs,
)

PACK_MANIFEST = "freetoken_bank_pack.json"
PACK_FORMAT = "freetoken-bank-pack"
_INDEX = "model.safetensors.index.json"
_COPY_CHUNK = 64 << 20


class PackError(RuntimeError):
    """``ft bank pack`` / ``verify`` refused; the message says why and what to do."""


# ---------------------------------------------------------------------------------------
# a packed checkpoint, from the outside
# ---------------------------------------------------------------------------------------
def read_pack_manifest(model_path: str | None) -> dict | None:
    """The pack manifest of ``model_path``, or None for an ordinary checkpoint."""
    if not model_path or not os.path.isdir(model_path):
        return None
    path = os.path.join(model_path, PACK_MANIFEST)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != PACK_FORMAT:
        raise PackError(f"{path}: not a FreeToken bank pack manifest")
    return manifest


def bank_root() -> str:
    """``~/.cache/freetoken/bankmap`` (or under ``$XDG_CACHE_HOME``): every model's bank directory."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "freetoken", "bankmap")


def default_bank_dir(model_path: str) -> str:
    return os.path.join(bank_root(), os.path.basename(os.path.normpath(model_path)))


def bank_path_for(model_path: str, manifest: dict | None = None, directory: str | None = None) -> str:
    """Where ``model_path``'s bank file is.

    ``--moe-bank-dir`` wins. A packed checkpoint names its own (normally ``bank.ftmb`` inside
    it). Otherwise a cache directory: the checkpoint may be read-only or shared, and for an
    ordinary checkpoint the file can always be written again.
    """
    if directory:
        return os.path.join(directory, BANK_FILE_NAME)
    if manifest is not None:
        ref = manifest.get("bank") or BANK_FILE_NAME
        return ref if os.path.isabs(ref) else os.path.join(model_path, ref)
    return os.path.join(default_bank_dir(model_path), BANK_FILE_NAME)


def check_served_packed(model_path: str, *, moe_bank_ram, offload: bool) -> dict | None:
    """Refuse, before any weight is read, a packed checkpoint run in a way that needs its experts."""
    manifest = read_pack_manifest(model_path)
    if manifest is None:
        return None
    if not offload or not moe_bank_ram:
        raise PackError(
            f"{model_path} has no routed-expert tensors: `ft bank pack` moved them into "
            f"{bank_path_for(model_path, manifest)}, and only --moe-bank-ram reads them from "
            f"there. Serve it with --moe-strategy hybrid (or offload) --moe-bank-ram SIZE"
        )
    return manifest


# ---------------------------------------------------------------------------------------
# safetensors headers
# ---------------------------------------------------------------------------------------
@dataclass
class ShardHeader:
    name: str
    raw: bytes  # the header JSON exactly as stored
    entries: dict  # tensor name -> {"dtype", "shape", "data_offsets"}
    metadata: dict | None
    base: int  # file offset of the data section
    size: int  # file size

    @classmethod
    def parse(cls, name: str, raw: bytes, size: int) -> "ShardHeader":
        entries = json.loads(raw.decode("utf-8"))
        metadata = entries.pop("__metadata__", None)
        return cls(name, raw, entries, metadata, 8 + len(raw), size)

    @classmethod
    def read(cls, path: str) -> "ShardHeader":
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            raw = f.read(n)
        return cls.parse(os.path.basename(path), raw, os.path.getsize(path))


def _weight_map(folder: str) -> tuple[dict[str, str], str | None]:
    """``({tensor: shard}, index text or None)``."""
    index = os.path.join(folder, _INDEX)
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            text = f.read()
        return json.loads(text)["weight_map"], text
    weight_map = {}
    for shard in sorted(p for p in os.listdir(folder) if p.endswith(".safetensors")):
        for name in ShardHeader.read(os.path.join(folder, shard)).entries:
            weight_map[name] = shard
    return weight_map, None


def fingerprint(sources: dict, entry_of) -> str:
    """Identity of the expert tensors a bank is built from: names, placement, dtype, shape.

    Structural, from headers alone, so it is cheap at every start and survives the tensors being
    removed (a packed checkpoint carries it in its manifest). It does not see a re-quantization
    that kept every name and shape -- ``ft bank pack`` compares contents, a start does not.
    """
    h = hashlib.sha256()
    for name in sorted(sources):
        s = sources[name]
        dtype, shape = entry_of(name)
        h.update(f"{name}|{s.bank_layer}|{s.e0}|{s.e1}|{s.role}|{dtype}|{list(shape)}\n".encode())
    return h.hexdigest()


def checkpoint_identity(model_path: str, config, kind) -> tuple[str, str]:
    """``(fingerprint, stamp)`` of an ordinary checkpoint's expert tensors, from headers and stat.

    The stamp adds the size and modification time of every shard that holds expert tensors, so
    a checkpoint downloaded again -- a fixed quantization with the same names and shapes -- does
    not keep being served the experts of the old one out of the bank file. A copy of the
    checkpoint changes it too, and costs one rewrite of the file. A packed checkpoint has no
    expert shards to stat; its start compares the fingerprint alone.
    """
    from freetoken.moe.expert_pieces import expert_sources

    weight_map, _ = _weight_map(model_path)
    sources = expert_sources(model_path, config, kind, weight_map=weight_map)
    headers: dict[str, ShardHeader] = {}

    def entry_of(name):
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = ShardHeader.read(os.path.join(model_path, shard))
        e = headers[shard].entries[name]
        return e["dtype"], e["shape"]

    fp = fingerprint(sources, entry_of)
    h = hashlib.sha256()
    for shard in sorted({weight_map[n] for n in sources}):
        st = os.stat(os.path.join(model_path, shard))
        h.update(f"{shard}|{st.st_size}|{st.st_mtime_ns};".encode())
    return fp, h.hexdigest()


# ---------------------------------------------------------------------------------------
# the kernel a bank was packed for
# ---------------------------------------------------------------------------------------
def method_meta(method) -> dict:
    """What a bank file records about the expert method it was written for."""
    cfg = method.cfg
    return {
        "kind": str(method.kind),
        "kernel": method.kernel.name,
        "kernel_cfg": {
            "num_experts": int(cfg.num_experts), "hidden": int(cfg.hidden),
            "intermediate": int(cfg.intermediate), "tp_rank": int(cfg.tp_rank), "tp_size": int(cfg.tp_size),
        },
    }


def kernel_for(layout: MappedBankLayout):
    """``(kernel, MoEConfig)`` rebuilt from a bank header, checked against the stored geometry."""
    from freetoken.layers.quantization import LayerKind, QuantKind, method_class
    from freetoken.layers.quantization.moe.base import MoEConfig

    meta = layout.meta
    if not meta.get("kind") or not meta.get("kernel") or not meta.get("kernel_cfg"):
        raise PackError(
            f"the bank file does not record the expert kernel it was written for (a loader without an "
            f"expert method wrote it); it cannot be checked against a checkpoint"
        )
    cls = method_class(QuantKind(meta["kind"]), LayerKind.MOE)
    kernels = {k.name: k for k in cls.candidates}
    if meta["kernel"] not in kernels:
        raise PackError(f"no {meta['kind']} expert kernel named {meta['kernel']!r} in this build")
    kc = meta["kernel_cfg"]
    cfg = MoEConfig(
        num_experts=kc["num_experts"], hidden=kc["hidden"], intermediate=kc["intermediate"], top_k=1,
        tp_rank=kc.get("tp_rank", 0), tp_size=kc.get("tp_size", 1),
    )
    kernel = kernels[meta["kernel"]]()
    specs = kernel.layout(cfg)
    if any(spec.resident for spec in specs.values()):
        raise PackError(
            f"{meta['kind']} / {meta['kernel']} keeps per-expert values on the GPU that the bank file "
            f"does not hold; only a CPU-capable kernel's bank can stand in for the checkpoint"
        )
    expected = layout_from_specs(specs, layout.layers, layout.num_experts, meta)
    if expected.banks != layout.banks:
        raise PackError(f"the bank's geometry {layout.banks} is not what {meta['kernel']} lays out: {expected.banks}")
    return kernel, cfg


def lossless_roles(kernel, cfg, layout: MappedBankLayout) -> set[str]:
    """Piece roles the kernel can give back byte for byte (``unpack`` over empty rows)."""
    import torch

    empty = {
        name: torch.empty((0, *shape), dtype=dtype_of(dtype))
        for name, shape, dtype, _ in layout.banks
    }
    return set(kernel.unpack(empty, cfg))


def _logical_rows(bank: BankFile, layer: int, blocks: dict, manifest: dict) -> dict:
    """A layer's bank rows in checkpoint (logical) expert order, typed and shaped."""
    import torch

    layout = bank.layout
    order, _ = bank.layer_state(layer, manifest)
    position = [0] * layout.num_experts
    for physical, logical in enumerate(order):
        position[logical] = physical
    idx = torch.as_tensor(position, dtype=torch.long)
    out = {}
    for name, shape, dtype, row_bytes in layout.banks:
        rows = blocks[name].view(layout.num_experts, row_bytes).index_select(0, idx)
        out[name] = rows.view(dtype_of(dtype)).view(layout.num_experts, *shape)
    return out


def _bytes(t):
    import torch

    return t.contiguous().reshape(-1).view(torch.uint8)


def _first_row_that_differs(a, b, rows: int) -> int:
    import torch

    a, b = _bytes(a).view(rows, -1), _bytes(b).view(rows, -1)
    bad = (a != b).any(dim=1).nonzero()
    return int(bad[0]) if bad.numel() else -1


def _group(sources) -> dict[int, dict[str, list]]:
    """``{layer: {role: [ExpertSource sorted by e0]}}``, with each role's rows covered exactly once."""
    out: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for s in sources.values():
        out[s.bank_layer][s.role].append(s)
    for layer, roles in out.items():
        for role, items in roles.items():
            items.sort(key=lambda s: s.e0)
    return {k: dict(v) for k, v in out.items()}


def _check_coverage(groups, num_experts: int) -> None:
    for layer, roles in groups.items():
        for role, items in roles.items():
            pos = 0
            for s in items:
                if s.e0 != pos:
                    raise PackError(f"layer {layer} role {role}: expert rows {pos}..{s.e0} have no tensor")
                pos = s.e1
            if pos != num_experts:
                raise PackError(f"layer {layer} role {role}: covers {pos} of {num_experts} experts")


def _pieces(items_by_role: dict, tensor_of) -> dict:
    import torch

    pieces = {}
    for role, items in items_by_role.items():
        parts = [s.to_piece(tensor_of(s.name)) for s in items]
        pieces[role] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
    return pieces


def _check_pack(kernel, cfg, layout, pieces, logical, layer: int, what: str) -> None:
    """``kernel.pack(pieces)`` must reproduce every bank role of the layer."""
    import torch

    out = {
        name: torch.empty((layout.num_experts, *shape), dtype=dtype_of(dtype))
        for name, shape, dtype, _ in layout.banks
    }
    resident = kernel.pack(pieces, cfg, out)
    if resident:
        raise PackError(f"{layout.meta.get('kernel')} returned GPU-resident values the bank does not hold")
    for name, _, _, _ in layout.banks:
        if not torch.equal(_bytes(out[name]), _bytes(logical[name])):
            row = _first_row_that_differs(out[name], logical[name], layout.num_experts)
            raise PackError(
                f"layer {layer} bank {name!r}: expert {row} in the bank file is not what {what} packs to"
            )


def _regenerate(kernel, cfg, logical, items_by_role: dict, roles: set[str]):
    """Yield ``(role, ExpertSource, uint8 bytes)`` for every lossless source, from the bank."""
    unpacked = kernel.unpack(logical, cfg)
    for role in sorted(items_by_role):
        if role not in roles:
            continue
        for s in items_by_role[role]:
            yield role, s, _bytes(unpacked[role][s.e0:s.e1])


# ---------------------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------------------
class _SlimShard:
    """Writes a safetensors shard holding a subset of an original's tensors, data streamed in."""

    def __init__(self, path: str, header: ShardHeader, keep: list[str]):
        entries, pos = {}, 0
        for name in keep:  # original data order
            e = header.entries[name]
            n = e["data_offsets"][1] - e["data_offsets"][0]
            entries[name] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [pos, pos + n]}
            pos += n
        if header.metadata is not None:
            entries = {"__metadata__": header.metadata, **entries}
        raw = json.dumps(entries, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)
        self.path, self.data_bytes = path, pos
        self._f = open(path, "wb")
        self._f.write(struct.pack("<Q", len(raw)) + raw)

    def write(self, chunk) -> None:
        self._f.write(chunk)

    def close(self) -> None:
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()


def _link_or_copy(src: str, dst: str, link: bool) -> str:
    real = os.path.realpath(src)
    if link:
        try:
            os.link(real, dst)
            return "link"
        except OSError:
            pass
    shutil.copyfile(real, dst)
    return "copy"


def _write_json_atomic(path: str, obj) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _drop_file_cache(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass
    finally:
        os.close(fd)


def pack_checkpoint(model_path: str, out_dir: str, *, config, bank_path: str, link: bool = True,
                    keep_bank: bool = False, dry_run: bool = False, log=print) -> dict:
    """Write ``out_dir``: ``model_path`` without the tensors ``bank_path`` reproduces. Returns a report.

    ``config`` is the full model config (its quant config set, as ``EngineConfig.model_config``
    leaves it). ``link``: hard-link shards that need no rewrite (falls back to a copy across
    filesystems). ``keep_bank``: leave the bank file where it is and record its absolute path
    instead of moving it into ``out_dir``. ``dry_run``: check the bank and the checkpoint's headers,
    work out what would be removed, linked and rewritten and how much space that takes, and stop
    before reading a tensor or writing a file.
    """
    import torch

    from freetoken.layers.quantization import QuantKind
    from freetoken.models.loader import ShardReader
    from freetoken.moe.expert_pieces import expert_sources

    folder = os.path.abspath(model_path)
    out = os.path.abspath(out_dir)
    if not os.path.isdir(folder):
        raise PackError(f"{model_path}: not a local checkpoint directory")
    if read_pack_manifest(folder) is not None:
        raise PackError(f"{folder} is already packed")
    if out == folder or out.startswith(folder + os.sep):
        raise PackError("--out must be a new directory outside the checkpoint")
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise PackError(f"{out} exists and is not an empty directory")
    bank_path = os.path.abspath(bank_path)
    if not os.path.isfile(bank_path):
        raise PackError(
            f"no bank file at {bank_path}. Serve {folder} once with --moe-bank-ram (on every rank of "
            f"the layer split) so it is written, or pass --moe-bank-dir"
        )

    # ----- the bank: complete, and for this checkpoint --------------------------------------
    bank = BankFile.open(bank_path)
    try:
        bank.recover_journals(log=log)
        layout = bank.layout
        manifest = bank.manifest()
        missing = [l for l in layout.layers if str(l) not in manifest["layers"]]
        if missing:
            from .mapped_bank import _ranges

            raise PackError(
                f"{bank_path} lacks layers {_ranges(missing)}. Every rank of a --pp-size run writes "
                f"its own layers; start the server once with each rank's layers (or --pp-size 1) "
                f"with --moe-bank-ram so the file is complete"
            )
        marked = bank.canonical_for(manifest)
        if marked and os.path.exists(marked):
            raise PackError(f"{bank_path} is already the only copy of the experts of {marked}")
        if marked:
            # the packed checkpoint the mark protects is gone, so the mark protects nothing and a
            # pack would otherwise be refused forever with no way to clear it (measured twice on
            # the same host: guides/39 8.4). One bank backs one packed checkpoint, so a pack that
            # re-points it would retire that one anyway.
            log(f"{bank_path} was the only copy of the experts of {marked}, which no longer "
                f"exists; dropping that mark")
            bank.set_canonical_for(None)
            manifest = bank.manifest()
        kernel, cfg = kernel_for(layout)
        kind = QuantKind(layout.meta["kind"])
        weight_map, index_text = _weight_map(folder)
        shards = sorted(set(weight_map.values()))
        headers = {s: ShardHeader.read(os.path.join(folder, s)) for s in shards}
        sources = expert_sources(folder, config, kind, weight_map=weight_map)
        fp = fingerprint(sources, lambda n: (headers[weight_map[n]].entries[n]["dtype"], headers[weight_map[n]].entries[n]["shape"]))
        if fp != layout.meta.get("fingerprint"):
            raise PackError(
                f"{bank_path} was not written from the expert tensors of {folder} (fingerprint "
                f"{str(layout.meta.get('fingerprint'))[:12]} against {fp[:12]})"
            )
        if sorted({s.bank_layer for s in sources.values()}) != layout.layers:
            raise PackError("the checkpoint's expert layers are not the bank's")
        roles = lossless_roles(kernel, cfg, layout)
        removable = {n for n, s in sources.items() if s.role in roles}
        if not removable:
            raise PackError(f"{layout.meta['kernel']} cannot give any expert tensor back byte for byte")
        groups = _group(sources)
        _check_coverage(groups, layout.num_experts)

        plan = {}
        for shard, h in headers.items():
            rm = set(h.entries) & removable
            plan[shard] = "link" if not rm else ("drop" if rm == set(h.entries) else "rewrite")
        stray = sorted(
            p for p in os.listdir(folder) if p.endswith(".safetensors") and p not in headers
        )

        # ----- room, and the bank's way into the slim directory ---------------------------
        parent = os.path.dirname(out)
        while not os.path.isdir(parent):  # the nearest directory that exists decides the filesystem
            parent = os.path.dirname(parent)
        dev = os.stat(parent).st_dev
        if not keep_bank and os.stat(os.path.dirname(bank_path)).st_dev != dev:
            raise PackError(
                f"{bank_path} is on a different filesystem from {parent}, so it cannot be moved into "
                f"the packed checkpoint without a copy. Pack onto that filesystem, or pass --keep-bank "
                f"to leave it where it is (the packed checkpoint then refers to it by path)"
            )
        same_fs = os.stat(folder).st_dev == dev
        kept_bytes = 0
        for shard, h in headers.items():
            if plan[shard] == "rewrite":
                kept_bytes += sum(
                    e["data_offsets"][1] - e["data_offsets"][0]
                    for n, e in h.entries.items() if n not in removable
                ) + len(h.raw) + 16
            elif plan[shard] == "link" and not (link and same_fs):
                kept_bytes += h.size
        free = free_bytes(parent)
        if free is not None and kept_bytes > free:
            raise PackError(f"writing {out} needs {kept_bytes / 2**30:.1f} GiB and {parent} has {free / 2**30:.1f} GiB free")

        removed_bytes = sum(
            headers[weight_map[n]].entries[n]["data_offsets"][1] - headers[weight_map[n]].entries[n]["data_offsets"][0]
            for n in removable
        )
        log(
            f"ft bank pack: {len(removable)} expert tensors ({removed_bytes / 2**30:.1f} GiB) are "
            f"reproduced by the bank's {layout.meta['kind']} / {layout.meta['kernel']} rows; "
            f"{len(sources) - len(removable)} stay ({', '.join(sorted({s.role for s in sources.values()} - roles)) or 'none'})"
        )
        log(
            f"ft bank pack: shards -- {sum(v == 'link' for v in plan.values())} without experts "
            f"({'hard-linked' if link and same_fs else 'copied'}), {sum(v == 'rewrite' for v in plan.values())} "
            f"rewritten, {sum(v == 'drop' for v in plan.values())} left out; {kept_bytes / 2**30:.1f} GiB to write"
        )
        if dry_run:
            return {
                "out": out, "bank": bank_path, "removed_tensors": len(removable), "removed_bytes": removed_bytes,
                "write_bytes": kept_bytes, "plan": plan, "layers": len(layout.layers), "dry_run": True,
            }
        os.makedirs(out, exist_ok=True)

        # ----- one sequential pass over the shards that carry experts ----------------------
        streamed_names = {n for s, h in headers.items() if plan[s] != "link" for n in h.entries}
        wait = defaultdict(int)  # layer -> sources still to arrive from the stream
        for n, s in sources.items():
            if n in streamed_names:
                wait[s.bank_layer] += 1
        held: dict[int, dict] = defaultdict(dict)
        digests: dict[str, dict[str, str]] = {}
        reader = ShardReader(folder, torch.device("cpu"))

        def verify_layer(layer: int) -> None:
            got = held.pop(layer, {})

            def tensor_of(name):
                return got[name] if name in got else reader.get_tensor(name)

            blocks = bank.check_digest(layer)
            logical = _logical_rows(bank, layer, blocks, manifest)
            del blocks
            _check_pack(kernel, cfg, layout, _pieces(groups[layer], tensor_of), logical, layer, "the original checkpoint")
            per_role: dict[str, hashlib._Hash] = {}
            for role, s, regenerated in _regenerate(kernel, cfg, logical, groups[layer], roles):
                original = _bytes(tensor_of(s.name))
                if regenerated.numel() != original.numel() or not torch.equal(regenerated, original):
                    raise PackError(f"{s.name}: the bank does not give back its bytes")
                per_role.setdefault(role, hashlib.sha256()).update(memoryview(original.numpy()))
            digests[str(layer)] = {role: h.hexdigest() for role, h in per_role.items()}
            bank.drop_cache(layer)
            log(f"ft bank pack: layer {layer} checked")

        from freetoken.models.weight import _ST_DTYPE

        shard_record = {}
        for shard in shards:
            h = headers[shard]
            src_path = os.path.join(folder, shard)
            dst_path = os.path.join(out, shard)
            action = plan[shard]
            if action == "link":
                how = _link_or_copy(src_path, dst_path, link)
                shard_record[shard] = {"action": how, "size": h.size}
                continue
            keep = [n for n, _ in sorted(h.entries.items(), key=lambda kv: kv[1]["data_offsets"][0]) if n not in removable]
            slim = _SlimShard(dst_path, h, keep) if action == "rewrite" else None
            sha = hashlib.sha256()
            with open(os.path.realpath(src_path), "rb") as f:
                head = f.read(h.base)
                sha.update(head)
                pos = h.base
                for name, e in sorted(h.entries.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                    start, end = h.base + e["data_offsets"][0], h.base + e["data_offsets"][1]
                    if start < pos:
                        raise PackError(f"{shard}: tensor {name} overlaps the one before it")
                    if start > pos:
                        sha.update(f.read(start - pos))
                    nbytes = end - start
                    if name in sources:
                        buf = bytearray(nbytes)
                        if f.readinto(buf) != nbytes:
                            raise PackError(f"{shard}: truncated at {name}")
                        sha.update(buf)
                        t = torch.frombuffer(buf, dtype=torch.uint8) if nbytes else torch.empty(0, dtype=torch.uint8)
                        tensor = t.view(_ST_DTYPE[e["dtype"]]).reshape(e["shape"])
                        if name not in removable:
                            slim.write(buf)
                        s = sources[name]
                        held[s.bank_layer][name] = tensor
                        wait[s.bank_layer] -= 1
                        if wait[s.bank_layer] == 0:
                            verify_layer(s.bank_layer)
                    else:
                        left = nbytes
                        while left:
                            chunk = f.read(min(_COPY_CHUNK, left))
                            if not chunk:
                                raise PackError(f"{shard}: truncated at {name}")
                            sha.update(chunk)
                            slim.write(chunk)
                            left -= len(chunk)
                    pos = end
                rest = f.read()
                sha.update(rest)
            if slim is not None:
                slim.close()
            _drop_file_cache(src_path)
            shard_record[shard] = {
                "action": action, "size": h.size, "sha256": sha.hexdigest(), "header": h.raw.decode("utf-8"),
            }
        for layer in [l for l in layout.layers if wait.get(l, 0) == 0 and str(l) not in digests]:
            verify_layer(layer)  # layers whose sources all sit in linked shards (none removable there)
        reader.close()
        unchecked = [l for l in layout.layers if str(l) not in digests]
        if unchecked:
            raise PackError(f"layers {unchecked} were never checked")
        for shard in stray:
            shard_record[shard] = {"action": _link_or_copy(os.path.join(folder, shard), os.path.join(out, shard), link),
                                   "size": os.path.getsize(os.path.join(folder, shard)), "unindexed": True}

        # ----- index, metadata, manifest, bank ------------------------------------------------
        if index_text is not None:
            index = json.loads(index_text)
            index["weight_map"] = {n: s for n, s in weight_map.items() if n not in removable}
            meta = dict(index.get("metadata") or {})
            if "total_size" in meta:
                meta["total_size"] = int(meta["total_size"]) - removed_bytes
            index["metadata"] = meta
            _write_json_atomic(os.path.join(out, _INDEX), index)
        from freetoken.checkpoint.convert import _copy_metadata

        copied = _copy_metadata(folder, out)
        bank_ref = bank_path if keep_bank else BANK_FILE_NAME
        record = {
            "format": PACK_FORMAT,
            "version": 1,
            "date": datetime.date.today().isoformat(),
            "packed_from": folder,
            "bank": bank_ref,
            "bank_origin": bank_path,
            "fingerprint": fp,
            "kind": layout.meta["kind"],
            "kernel": layout.meta["kernel"],
            "num_experts": layout.num_experts,
            "layers": layout.layers,
            "removed": {"tensors": len(removable), "bytes": removed_bytes, "roles": sorted(roles & {s.role for s in sources.values()})},
            "digests": digests,
            "index": index_text,
            "shards": shard_record,
        }
        _write_json_atomic(os.path.join(out, PACK_MANIFEST), record)
        bank.set_canonical_for(out)
    finally:
        bank.close()
    final_bank = bank_path
    if not keep_bank:
        final_bank = os.path.join(out, BANK_FILE_NAME)
        os.rename(bank_path, final_bank)
        for suffix in (".lock",):
            try:
                os.unlink(bank_path + suffix)
            except OSError:
                pass
    slim_bytes = sum(
        os.path.getsize(os.path.join(out, s)) for s, r in shard_record.items()
        if r["action"] in ("rewrite", "copy")
    )
    return {
        "out": out, "bank": final_bank, "removed_tensors": len(removable), "removed_bytes": removed_bytes,
        "rewritten_bytes": slim_bytes, "shards": {a: sum(r["action"] == a for r in shard_record.values()) for a in ("link", "copy", "rewrite", "drop")},
        "copied_files": copied, "layers": len(layout.layers),
    }


# ---------------------------------------------------------------------------------------
# verify (no original needed)
# ---------------------------------------------------------------------------------------
def packed_weight_map(folder: str, manifest: dict) -> tuple[dict[str, str], dict[str, ShardHeader]]:
    """The ORIGINAL checkpoint's ``{tensor: shard}`` and headers, from the manifest and the slim shards."""
    weight_map, headers = {}, {}
    for shard, rec in manifest["shards"].items():
        if "header" in rec:
            h = ShardHeader.parse(shard, rec["header"].encode("utf-8"), rec["size"])
        else:
            h = ShardHeader.read(os.path.join(folder, shard))
        headers[shard] = h
        if rec.get("unindexed"):
            continue
        for name in h.entries:
            weight_map[name] = shard
    return weight_map, headers


def verify_packed(model_path: str, *, config, bank_path: str | None = None, log=print) -> dict:
    """Check a packed checkpoint against its bank without the original. Raises ``PackError``.

    Per layer: the block hashes the bank committed; the removed tensors regenerated from it
    against the per-(layer, role) hashes ``pack`` recorded from the original bytes; and the
    bank's rows against ``pack`` of those regenerated tensors plus the ones the slim checkpoint
    kept (the lossy global scales). And per shard: the slim shards hold exactly the tensors the
    original had minus the removed ones.
    """
    import torch

    from freetoken.layers.quantization import QuantKind
    from freetoken.models.loader import ShardReader
    from freetoken.moe.expert_pieces import expert_sources

    folder = os.path.abspath(model_path)
    manifest = read_pack_manifest(folder)
    if manifest is None:
        raise PackError(f"{folder} is not a packed checkpoint (no {PACK_MANIFEST})")
    bank_path = bank_path or bank_path_for(folder, manifest)
    if not os.path.isfile(bank_path):
        raise PackError(f"no bank file at {bank_path} (it was {manifest.get('bank_origin')} when packed)")
    weight_map, headers = packed_weight_map(folder, manifest)
    kind = QuantKind(manifest["kind"])
    sources = expert_sources(folder, config, kind, weight_map=weight_map)
    fp = fingerprint(sources, lambda n: (headers[weight_map[n]].entries[n]["dtype"], headers[weight_map[n]].entries[n]["shape"]))
    if fp != manifest["fingerprint"]:
        raise PackError("the manifest's shard headers do not give the fingerprint it records")

    # the slim shards hold what they should
    removed_names = set()
    for shard, rec in manifest["shards"].items():
        path = os.path.join(folder, shard)
        if rec["action"] == "drop":
            if os.path.exists(path):
                raise PackError(f"{shard} should have been left out")
            removed_names |= set(headers[shard].entries)
            continue
        if not os.path.isfile(path):
            raise PackError(f"{shard} is missing from {folder}")
        if rec["action"] in ("link", "copy"):
            if os.path.getsize(path) != rec["size"]:
                raise PackError(f"{shard}: {os.path.getsize(path)} bytes, packed as {rec['size']}")
            continue
        slim = ShardHeader.read(path)
        gone = set(headers[shard].entries) - set(slim.entries)
        if set(slim.entries) - set(headers[shard].entries):
            raise PackError(f"{shard} holds tensors the original did not")
        for name in slim.entries:
            a, b = slim.entries[name], headers[shard].entries[name]
            if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
                raise PackError(f"{shard}: {name} changed dtype or shape")
        removed_names |= gone

    bank = BankFile.open(bank_path, writable=os.access(bank_path, os.W_OK))
    try:
        if os.access(bank_path, os.W_OK):
            bank.recover_journals(log=log)
        layout = bank.layout
        if layout.meta.get("fingerprint") != manifest["fingerprint"]:
            raise PackError(f"{bank_path} was written for other expert tensors than this checkpoint's")
        kernel, cfg = kernel_for(layout)
        roles = lossless_roles(kernel, cfg, layout)
        expected_removed = {n for n, s in sources.items() if s.role in roles}
        if removed_names != expected_removed:
            raise PackError(
                f"the slim shards lack {len(removed_names)} tensors, but the bank reproduces "
                f"{len(expected_removed)}: {sorted(removed_names ^ expected_removed)[:4]}"
            )
        groups = _group(sources)
        manifest_bank = bank.manifest()
        reader = ShardReader(folder, torch.device("cpu"))
        try:
            for layer in layout.layers:
                blocks = bank.check_digest(layer)
                logical = _logical_rows(bank, layer, blocks, manifest_bank)
                del blocks
                per_role: dict[str, hashlib._Hash] = {}
                regenerated = {}
                for role, s, data in _regenerate(kernel, cfg, logical, groups[layer], roles):
                    per_role.setdefault(role, hashlib.sha256()).update(memoryview(data.numpy()))
                    regenerated[s.name] = data
                got = {role: h.hexdigest() for role, h in per_role.items()}
                if got != manifest["digests"].get(str(layer)):
                    bad = sorted(r for r in got if got[r] != manifest["digests"].get(str(layer), {}).get(r))
                    raise PackError(f"layer {layer}: the bank no longer gives back the original bytes of {bad}")

                def tensor_of(name):
                    if name in regenerated:
                        shard = weight_map[name]
                        e = headers[shard].entries[name]
                        from freetoken.models.weight import _ST_DTYPE

                        return regenerated[name].view(_ST_DTYPE[e["dtype"]]).reshape(e["shape"])
                    return reader.get_tensor(name)

                _check_pack(kernel, cfg, layout, _pieces(groups[layer], tensor_of), logical, layer, "the packed checkpoint")
                bank.drop_cache(layer)
                log(f"ft bank verify: layer {layer} ok")
        finally:
            reader.close()
    finally:
        bank.close()
    return {"layers": len(layout.layers), "removed_tensors": len(removed_names), "bank": bank_path}


# ---------------------------------------------------------------------------------------
# unpack: the original files back
# ---------------------------------------------------------------------------------------
_NOT_METADATA = (PACK_MANIFEST, BANK_FILE_NAME)


def unpack_checkpoint(model_path: str, out_dir: str, *, config, bank_path: str | None = None,
                      link: bool = True, log=print) -> dict:
    """Write the original checkpoint back from a packed one and its bank. Raises ``PackError``.

    Every shard ``pack`` rewrote or left out is rebuilt from its recorded header, the tensors the
    slim shard kept and the tensors the bank regenerates, and compared with the SHA-256 of the
    original file; the others are hard-linked (or copied) as they are. The index goes back as it
    was, and the other files are copied from the packed checkpoint. Nothing is deleted.
    """
    import torch

    from freetoken.layers.quantization import QuantKind
    from freetoken.moe.expert_pieces import expert_sources

    folder = os.path.abspath(model_path)
    out = os.path.abspath(out_dir)
    manifest = read_pack_manifest(folder)
    if manifest is None:
        raise PackError(f"{folder} is not a packed checkpoint (no {PACK_MANIFEST})")
    if out == folder or out.startswith(folder + os.sep):
        raise PackError("--out must be a new directory outside the packed checkpoint")
    if os.path.exists(out) and (not os.path.isdir(out) or os.listdir(out)):
        raise PackError(f"{out} exists and is not an empty directory")
    bank_path = bank_path or bank_path_for(folder, manifest)
    weight_map, headers = packed_weight_map(folder, manifest)
    kind = QuantKind(manifest["kind"])
    sources = expert_sources(folder, config, kind, weight_map=weight_map)
    os.makedirs(out, exist_ok=True)
    bank = BankFile.open(bank_path, writable=False)
    try:
        layout = bank.layout
        if layout.meta.get("fingerprint") != manifest["fingerprint"]:
            raise PackError(f"{bank_path} was written for other expert tensors than this checkpoint's")
        kernel, cfg = kernel_for(layout)
        roles = lossless_roles(kernel, cfg, layout)
        removable = {n: s for n, s in sources.items() if s.role in roles}
        manifest_bank = bank.manifest()
        cache: dict[int, dict] = {}

        def regenerated(name):
            s = removable[name]
            if s.bank_layer not in cache:
                cache.clear()  # shards run in layer order; one layer at a time is enough
                blocks = bank.check_digest(s.bank_layer)
                cache[s.bank_layer] = kernel.unpack(_logical_rows(bank, s.bank_layer, blocks, manifest_bank), cfg)
            return _bytes(cache[s.bank_layer][s.role][s.e0:s.e1])

        counts = defaultdict(int)
        for shard, rec in sorted(manifest["shards"].items()):
            dst = os.path.join(out, shard)
            if rec["action"] in ("link", "copy"):
                counts[_link_or_copy(os.path.join(folder, shard), dst, link)] += 1
                continue
            original = headers[shard]
            slim = ShardHeader.read(os.path.join(folder, shard)) if rec["action"] == "rewrite" else None
            sha = hashlib.sha256()
            with open(dst, "wb") as f, open(os.path.join(folder, shard), "rb") if slim else _Nothing() as src:
                head = struct.pack("<Q", len(original.raw)) + original.raw
                f.write(head)
                sha.update(head)
                pos = original.base
                for name, e in sorted(original.entries.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                    start = original.base + e["data_offsets"][0]
                    if start > pos:  # never in a file safetensors wrote; the hash will say
                        gap = b"\0" * (start - pos)
                        f.write(gap)
                        sha.update(gap)
                    nbytes = e["data_offsets"][1] - e["data_offsets"][0]
                    if name in removable:
                        data = regenerated(name)
                        if data.numel() != nbytes:
                            raise PackError(f"{name}: the bank gives {data.numel()} bytes, the header says {nbytes}")
                        view = memoryview(data.numpy())
                        f.write(view)
                        sha.update(view)
                    else:
                        if slim is None or name not in slim.entries:
                            raise PackError(f"{shard}: {name} is neither in the packed shard nor in the bank")
                        src.seek(slim.base + slim.entries[name]["data_offsets"][0])
                        left = nbytes
                        while left:
                            chunk = src.read(min(_COPY_CHUNK, left))
                            if not chunk:
                                raise PackError(f"{shard}: packed shard truncated at {name}")
                            f.write(chunk)
                            sha.update(chunk)
                            left -= len(chunk)
                    pos = start + nbytes
                f.flush()
                os.fsync(f.fileno())
            if sha.hexdigest() != rec["sha256"]:
                raise PackError(f"{shard}: rebuilt, but not the original bytes (sha256 {sha.hexdigest()[:12]} against {rec['sha256'][:12]})")
            counts[rec["action"]] += 1
            log(f"ft bank unpack: {shard} rebuilt and matches the original")
        if manifest.get("index") is not None:
            with open(os.path.join(out, _INDEX), "w", encoding="utf-8") as f:
                f.write(manifest["index"])
        copied = []
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d not in (".git", ".cache")]
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), folder)
                if (
                    name.endswith(".safetensors") or rel in _NOT_METADATA or rel == _INDEX
                    or name.startswith(BANK_FILE_NAME + ".")
                ):
                    continue
                dst = os.path.join(out, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(os.path.join(root, name), dst)
                copied.append(rel)
    finally:
        bank.close()
    return {"out": out, "shards": dict(counts), "copied_files": copied}


class _Nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
