"""--dense-quant: serve the projections a checkpoint left bf16 as fp8, quantized at load.

Every other QuantConfig answers what the checkpoint *stores*. This one answers what the
deployment wants: on a card too small for the bf16 dense weights, a projection the checkpoint
never quantized is still worth halving. The weights themselves are converted in the loader
(``engine._quantize_at_load``), which sees an fp8 buffer where a bf16 tensor arrived.
"""

from __future__ import annotations

from typing import Any

from .configs.base import NoQuantConfig, QuantConfig
from .linear import LinearConfig
from .names import is_routed_expert, name_set
from .quant_backend import get_quant_backend
from .registry import LayerKind, method_class
from .scheme import QuantKind, QuantScheme, fp8_tensor_scheme

# one fp32 scale per output row: a whole-tensor scale loses too much on a fused projection,
# and the W8A16 kernels read a per-row scale either way
AT_LOAD_FP8 = fp8_tensor_scheme("float", per_row=True)

# Projections that stay bf16: the MoE router, the hyper-connection mixer's skinny GEMMs, the QSA
# indexer, the PLE table's key/value projections and the GatedDeltaNet b/a gates decide which
# experts run, how streams mix, which keys are scored, which n-gram rows are read and how much
# state decays. They are small, so they buy little VRAM, and they are the last place to spend
# accuracy.
KEEP_BF16 = name_set((
    "*.gate", "*.shared_expert_gate", "*hyper_connection*", "*.indexer", "*.ple",
    "*.in_proj_ba", "*.in_proj_b", "*.in_proj_a",
))


class LoadTimeFp8Config(QuantConfig):
    """The checkpoint's config, with every unquantized dense projection reported as per-row fp8."""

    dialect = "dense-quant-fp8"

    def __init__(self, inner: QuantConfig | None):
        inner = inner if inner is not None else NoQuantConfig()
        super().__init__(inner.name_map, ())
        self.inner = inner

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return False  # never selected by a checkpoint; --dense-quant wraps whatever was

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        return self.inner.scheme_for_name(name)

    def _quantize_at_load(self, prefix: str) -> bool:
        if self.inner.scheme_for(prefix) is not None:
            return False  # the checkpoint already quantized this one; keep its format
        names = self.name_map.to_checkpoint(prefix)
        # routed experts are read into banks by their own kernel, never converted here
        return not any(KEEP_BF16(n) or is_routed_expert(n) for n in names)

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        # a model may ask what a projection's storage is before building it (the GatedDeltaNet
        # splits its fused in_proj only when the qkvz half is quantized), so answer as built
        inner = self.inner.scheme_for(prefix)
        if inner is not None:
            return inner
        return AT_LOAD_FP8 if self._quantize_at_load(prefix) else None

    def get_quant_method(self, layer: Any, prefix: str):
        if layer.quant_layer_kind is not LayerKind.LINEAR or not self._quantize_at_load(prefix):
            return self.inner.get_quant_method(layer, prefix)
        cls = method_class(QuantKind.FP8_TENSOR, LayerKind.LINEAR)
        cfg = LinearConfig.from_layer(layer, AT_LOAD_FP8)
        return cls(cfg, get_quant_backend().select(LayerKind.LINEAR, QuantKind.FP8_TENSOR))
