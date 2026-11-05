"""Routed-expert banks straight from a GGUF's ``blk.N.ffn_{gate,up,down}_exps`` tensors, in their packed ggml rows: pinned host banks for the offload cache, or device banks for resident experts.

Each layer keeps its own ggml types, so each layer gets its own bank shapes: ``gate_up`` [E, 2I, row_bytes] (expert e's gate rows, then its up rows) and ``down`` [E, H, row_bytes]; a layer whose gate_up and down are both float (BF16 / F16 / F32) loads and runs bf16 ([E, 2I, H] and [E, H, I], the fused MoE's dtype), while a mixed float/quant layer keeps the float half in the file's own dtype, as main does.
A GGUF that stores ``ffn_gate_up_exps`` (gemma4) already holds the gate_up rows in that order.
When NVFP4 experts carry per-expert ``.scale`` tensors, every layer also gets ``global`` [E, 4] fp32: the (gate, up, down, 1) weight_scale_2, 1 where a tensor has none.
gpt-oss files carry per-expert gate / up / down biases; every layer then also gets ``gate_up_bias`` [E, 2I] and ``down_bias`` [E, H] bf16 rows, so the bias follows the expert through the slot cache like its weights do.

Kai: the shapes, quant tags and MMQ tails are also read off the tensor table alone (``gguf_bank_geometry``),
so ``--moe-bank-ram`` can lay out its bank file before a byte of the experts is read and a run whose layers
are all in that file never opens them; with a ``layer_sink`` each layer streams into the file instead of
being pinned (a DeepSeek-V4.1 GSQ-RCO build is 123 GiB of experts, more than the host's RAM)."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import torch

from freetoken.layers.quantization import QuantKind
from freetoken.layers.quantization.ggml_types import float_dtype, has_expert_scales, mmq_tail_bytes, pad_rows, row_bytes, type_name
from freetoken.utils import init_logger

from .reader import GgufTensorInfo, gguf_tensor_index
from .tensors import float_tensor, weight_scale

logger = init_logger(__name__)


def _expert_rows(info: GgufTensorInfo) -> torch.Tensor:
    """[E, rows, row_bytes] packed ggml rows, or an [E, rows, K] view in the file's float dtype; the
    bank allocation that copies decides bf16 (a whole-layer float layer) or the file's dtype (a mixed one)."""
    k, rows, experts = info.ne
    raw = torch.from_numpy(info.data()).view(experts, rows, info.nbytes // (experts * rows))
    dtype = float_dtype(info.ggml_type)
    return raw if dtype is None else raw.view(dtype)


def _drop_page_cache(info: GgufTensorInfo) -> None:
    try:
        fd = os.open(info.path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, info.offset, info.nbytes, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _device_bank(shape: tuple[int, ...], ggml_type: int, k: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An [E, rows, row] device bank; a zero row follows the last one when ggml's MMQ tiles read past it."""
    experts, rows, row = shape
    flat = torch.empty(experts * rows + pad_rows(ggml_type, k), row, dtype=dtype, device=device)
    flat[experts * rows :].zero_()
    return flat[: experts * rows].view(shape)


def resident_expert_bank(info: GgufTensorInfo, device: torch.device) -> torch.Tensor:
    """One stacked ``[E, rows, K]`` expert tensor as a resident device bank, in its packed ggml rows."""
    rows = _expert_rows(info)
    bank = _device_bank(tuple(rows.shape), info.ggml_type, info.ne[0], rows.dtype, device)
    bank.copy_(rows)
    return bank


def _expert_scales(gate_up: list[GgufTensorInfo], down: GgufTensorInfo, experts: int) -> torch.Tensor:
    """[E, 4] fp32 (gate, up, down, 1) weight_scale_2 of one layer's experts."""
    gate, up = gate_up if len(gate_up) == 2 else (gate_up[0], gate_up[0])
    out = torch.ones(experts, 4, dtype=torch.float32)
    for col, info in enumerate((gate, up, down)):
        s = weight_scale(info)
        if s is None:
            continue
        if s.numel() != experts:
            raise NotImplementedError(f"{info.name}: {s.numel()} scales for {experts} experts")
        out[:, col] = s
    return out


def _layer_tensors(index: dict, blk: str) -> tuple[list[GgufTensorInfo], GgufTensorInfo]:
    """``([gate, up] or [gate_up], down)`` of one layer."""
    down = index[f"{blk}.ffn_down_exps.weight"]
    if f"{blk}.ffn_gate_up_exps.weight" in index:
        return [index[f"{blk}.ffn_gate_up_exps.weight"]], down
    gate, up = index[f"{blk}.ffn_gate_exps.weight"], index[f"{blk}.ffn_up_exps.weight"]
    if gate.ggml_type != up.ggml_type:
        raise NotImplementedError(f"{blk}: gate and up experts have different ggml types")
    return [gate, up], down


def _pure_float(gate: GgufTensorInfo, down: GgufTensorInfo) -> bool:
    """A whole-layer float layer runs the bf16 fused MoE; a mixed float/quant layer keeps the float half's own dtype."""
    return float_dtype(gate.ggml_type) is not None and float_dtype(down.ggml_type) is not None


def _row_spec(info: GgufTensorInfo, rows: int, *, bf16: bool = False) -> tuple[tuple[int, int], torch.dtype]:
    """One expert's ``(rows, row)`` and dtype in a bank of ``info``'s ggml type (``bf16``: a whole-layer
    float layer, which loads as bf16 whatever float type the file stores)."""
    k = info.ne[0]
    dtype = float_dtype(info.ggml_type)
    if dtype is None:
        return (rows, row_bytes(info.ggml_type, k)), torch.uint8
    return (rows, k), torch.bfloat16 if bf16 else dtype


@dataclass(frozen=True)
class GgufBankGeometry:
    """What the banks of a GGUF's MoE layers will look like, from its tensor table alone."""

    specs: list[dict]  # per layer {role: ((rows, row), dtype)} of one expert
    tags: list[tuple]  # per layer ("ggml", gate_up type, down type)
    tails: dict[str, list[int]]  # per role, per layer: MMQ bytes past the last row
    scaled: bool  # NVFP4 per-expert scales: every layer has a "global" bank
    biased: bool = False  # gpt-oss per-expert biases: every layer has "gate_up_bias" and "down_bias" banks

    @property
    def quant_format(self) -> str:
        # the _bias formats are the gguf layouts plus the per-expert bias bank rows (_BANK_SCHEMAS)
        return ("gguf_scaled" if self.scaled else "gguf") + ("_bias" if self.biased else "")


def gguf_bank_geometry(model_path: str, num_layers: int, *, first_layer: int = 0) -> GgufBankGeometry:
    index = gguf_tensor_index(model_path)
    scaled = has_expert_scales({name: info.ggml_type for name, info in index.items()})
    # gpt-oss: a bias tensor per expert and role; other families ship none
    biased = f"blk.{first_layer}.ffn_gate_exps.bias" in index
    roles = ("gate_up", "down", "global") if scaled else ("gate_up", "down")
    roles += ("gate_up_bias", "down_bias") if biased else ()
    specs, tags, tails = [], [], {role: [] for role in roles}
    for layer in range(num_layers):
        parts, down = _layer_tensors(index, f"blk.{first_layer + layer}")
        gate = parts[0]
        bf16 = _pure_float(gate, down)
        gate_up_rows = sum(p.ne[1] for p in parts)
        spec = {"gate_up": _row_spec(gate, gate_up_rows, bf16=bf16), "down": _row_spec(down, down.ne[1], bf16=bf16)}
        tails["gate_up"].append(mmq_tail_bytes(gate.ggml_type, gate.ne[0]))
        tails["down"].append(mmq_tail_bytes(down.ggml_type, down.ne[0]))
        if scaled:
            spec["global"] = ((4,), torch.float32)
            tails["global"].append(0)
        if biased:
            spec["gate_up_bias"] = ((gate_up_rows,), torch.bfloat16)
            spec["down_bias"] = ((down.ne[1],), torch.bfloat16)
            tails["gate_up_bias"].append(0)
            tails["down_bias"].append(0)
        specs.append(spec)
        tags.append(("ggml", type_name(gate.ggml_type), type_name(down.ggml_type)))
    return GgufBankGeometry(specs, tags, tails, scaled, biased)


def gguf_expert_identity(model_path: str) -> tuple[str, str]:
    """``(fingerprint, stamp)`` of a GGUF's routed experts for a ``--moe-bank-ram`` file: the names, types
    and shapes of the expert tensors, and the size and mtime of every split that holds them (a file
    downloaded again is read again)."""
    import hashlib

    from .reader import gguf_split_paths

    index = gguf_tensor_index(model_path)
    names = sorted(n for n in index if "_exps." in n)
    fp = hashlib.sha256()
    for name in names:
        info = index[name]
        fp.update(f"{name}:{info.ggml_type}:{','.join(map(str, info.ne))};".encode())
    stamp = hashlib.sha256()
    for path in sorted({index[n].path for n in names} or set(gguf_split_paths(model_path))):
        st = os.stat(path)
        stamp.update(f"{os.path.basename(path)}:{st.st_size}:{st.st_mtime_ns};".encode())
    return fp.hexdigest(), stamp.hexdigest()


def load_gguf_expert_banks(
    model_path: str, num_layers: int, *, first_layer: int = 0, dummy: bool = False, device: torch.device | None = None, layer_sink=None,
):
    """``ExpertBanks`` with one bank per (layer, role): pinned host banks for the offload cache, filled from the file and pinned while later layers load, or resident banks on ``device``.

    ``layer_sink`` (``--moe-bank-ram``'s ``MappedTier.sink``): each layer's banks go to it as plain host
    tensors instead, one layer at a time, and the returned banks hold no rows (the tier's mapping replaces them)."""
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.host_banks import HostBank, PinPipeline

    index = gguf_tensor_index(model_path)
    geometry = gguf_bank_geometry(model_path, num_layers, first_layer=first_layer)
    scaled, biased = geometry.scaled, geometry.biased
    sources: dict[str, list[torch.Tensor]] = {role: [] for role in geometry.tails}
    streamed = layer_sink is not None and not dummy
    with PinPipeline() if device is None and not streamed else contextlib.nullcontext() as pins:
        for layer in range(num_layers):
            blk = f"blk.{first_layer + layer}"
            parts, down = _layer_tensors(index, blk)
            gate = parts[0]
            rows = [_expert_rows(info) for info in parts]
            d = _expert_rows(down)
            gate_up_shape = (rows[0].shape[0], sum(r.shape[1] for r in rows), rows[0].shape[2])
            experts = rows[0].shape[0]
            spec = geometry.specs[layer]
            gu_dtype, down_dtype = spec["gate_up"][1], spec["down"][1]
            host = None
            if streamed:
                banks = {"gate_up": torch.empty(gate_up_shape, dtype=gu_dtype), "down": torch.empty(tuple(d.shape), dtype=down_dtype)}
                if scaled:
                    banks["global"] = torch.empty(experts, 4, dtype=torch.float32)
            elif device is None:
                host = {"gate_up": HostBank(gate_up_shape, gu_dtype), "down": HostBank(tuple(d.shape), down_dtype)}
                if scaled:
                    host["global"] = HostBank((experts, 4), torch.float32)
                banks = {role: bank.tensor for role, bank in host.items()}
            else:
                banks = {
                    "gate_up": _device_bank(gate_up_shape, gate.ggml_type, gate.ne[0], gu_dtype, device),
                    "down": _device_bank(tuple(d.shape), down.ggml_type, down.ne[0], down_dtype, device),
                }
                if scaled:
                    banks["global"] = torch.empty(experts, 4, dtype=torch.float32, device=device)
            if scaled:
                banks["global"].copy_(_expert_scales(parts, down, experts))
            if biased:
                missing = [f"{blk}.ffn_{p}_exps.bias" for p in ("gate", "up", "down") if f"{blk}.ffn_{p}_exps.bias" not in index]
                if missing:
                    raise NotImplementedError(f"{model_path}: {', '.join(missing)}: the per-expert biases ship on every layer of a gpt-oss file")
                # gpt-oss per-expert biases: gate|up as one [E, 2I] row block, down as [E, H]
                gate_b, up_b, down_b = (
                    float_tensor(index[f"{blk}.ffn_{p}_exps.bias"]).to(torch.bfloat16) for p in ("gate", "up", "down")
                )
                gate_up_bias, down_bias = torch.cat([gate_b, up_b], dim=1), down_b
                for role, rows_bias in (("gate_up_bias", gate_up_bias), ("down_bias", down_bias)):
                    if rows_bias.shape[0] != experts:
                        raise NotImplementedError(f"{blk}.{role}: {tuple(rows_bias.shape)} for {experts} experts")
                    if streamed:
                        banks[role] = torch.empty(tuple(rows_bias.shape), dtype=torch.bfloat16)
                    elif device is None:
                        host[role] = HostBank(tuple(rows_bias.shape), torch.bfloat16)
                        banks[role] = host[role].tensor
                    else:
                        banks[role] = torch.empty_like(rows_bias, device=device)
                    banks[role].copy_(rows_bias)
            if dummy:
                banks["gate_up"].zero_()
                banks["down"].zero_()
                if biased:
                    banks["gate_up_bias"].zero_()
                    banks["down_bias"].zero_()
            else:
                start = 0
                for r in rows:
                    banks["gate_up"][:, start : start + r.shape[1]].copy_(r)
                    start += r.shape[1]
                banks["down"].copy_(d)
                for info in (*parts, down):
                    _drop_page_cache(info)
            if streamed:
                layer_sink(layer, banks)
                del banks  # one layer of copies at a time
                continue
            if device is None:
                pins(layer, host)
            for role, bank in banks.items():
                sources[role].append(bank)
    if streamed:
        logger.info_rank0(f"GGUF expert banks: {num_layers} layers streamed to the bank file")
    else:
        total = sum(t.numel() * t.element_size() for ts in sources.values() for t in ts)
        logger.info_rank0(f"GGUF expert banks: {num_layers} layers, {total / 2**30:.1f} GiB {'on ' + str(device) if device is not None else 'in host RAM'}")
    return ExpertBanks(
        geometry.quant_format, sources, kind=QuantKind.GGUF, kernel="ggml", layout=None, layer_tags=geometry.tags,
        row_tail_bytes=geometry.tails, streamed=streamed,
    )


__all__ = ["GgufBankGeometry", "gguf_bank_geometry", "gguf_expert_identity", "load_gguf_expert_banks", "resident_expert_bank"]
