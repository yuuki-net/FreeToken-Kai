"""A small GGUF v3 writer for tests that only need a tensor table (routed experts with ggml types).

Holds no tests itself. The bytes are random: the layout is what the callers check."""

from __future__ import annotations

import struct

import numpy as np

from freetoken.layers.quantization.ggml_types import row_bytes
from freetoken.third_party.gguf import GGMLQuantizationType as QType

# unsloth's Qwen3.8-Flash-Next UD-IQ4_XS keeps IQ3_S / IQ4_NL experts in most layers and IQ4_XS in a few;
# layers 0 and 3 share the most common geometry, layers 1 and 2 differ
UD_IQ4_XS_TYPES = [(QType.IQ3_S, QType.IQ4_NL), (QType.IQ4_XS, QType.IQ4_XS), (QType.IQ4_XS, QType.IQ4_NL), (QType.IQ3_S, QType.IQ4_NL)]


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(v) -> tuple[int, bytes]:
    if isinstance(v, str):
        return 8, _string(v)
    if isinstance(v, int):
        return 4, struct.pack("<I", v)
    raise TypeError(f"no GGUF type for {v!r}")


def write_gguf(path: str, metadata: dict, tensors: list[tuple[str, tuple[int, ...], int, bytes]], align: int = 32) -> None:
    """A GGUF v3 file with str / uint32 metadata and ``tensors`` as (name, ggml ne, ggml type, packed bytes)."""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        t, body = _value(value)
        head += _string(key) + struct.pack("<I", t) + body
    offset, blobs = 0, []
    for name, ne, ggml_type, data in tensors:
        head += _string(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne)
        head += struct.pack("<IQ", ggml_type, offset)
        pad = (-len(data)) % align
        blobs.append(data + b"\0" * pad)
        offset += len(data) + pad
    head += b"\0" * ((-len(head)) % align)
    with open(path, "wb") as f:
        f.write(head)
        for blob in blobs:
            f.write(blob)


def write_experts_gguf(path: str, types=UD_IQ4_XS_TYPES, *, experts: int = 4, hidden: int = 256, inter: int = 256,
                       seed: int = 0, first_block: int = 0, metadata: dict | None = None) -> None:
    """A GGUF with nothing but routed experts, ``types[i]`` = (gate/up, down) of block ``first_block + i``."""
    rng = np.random.default_rng(seed)
    tensors = []
    for i, (gate_up, down) in enumerate(types):
        layer = first_block + i
        for name, qtype, k, rows in (("ffn_gate_exps", gate_up, hidden, inter), ("ffn_up_exps", gate_up, hidden, inter),
                                     ("ffn_down_exps", down, inter, hidden)):
            data = rng.integers(0, 256, experts * rows * row_bytes(int(qtype), k), dtype=np.uint8)
            tensors.append((f"blk.{layer}.{name}.weight", (k, rows, experts), int(qtype), data.tobytes()))
    write_gguf(path, metadata or {"general.architecture": "qwen4exp"}, tensors)
