"""The draft head's routed experts share the target's bank, so they share its expert method.

A checkpoint excludes its MTP head from quantization, so the head ships bf16 experts. The
engine does not serve them that way: it quantizes them into one more layer of the *target's*
expert bank (``Engine._append_mtp_bank``), and the offload cache holds a single layout for
every layer it serves. Reading the head's own entry would bind an unquantized method to a
bank of NVFP4 rows, which stops the boot with "expert layers disagree on format / kernel".
"""

from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import LayerKind, NoQuantConfig, QuantConfig, QuantScheme

_UNQUANTIZED = NoQuantConfig()


class DraftHeadQuantConfig(QuantConfig):
    """Routed experts answered at ``expert_prefix``; everything else at its own name.

    ``dense`` is the config the head's own projections follow -- the model's (so
    ``--dense-quant`` reaches them, as on Qwen3.8-Flash-Next) or None to keep them bf16 (the
    Qwen3.5-MoE family, whose head is built unquantized).
    """

    dialect = "mtp-draft-head"

    def __init__(self, experts: QuantConfig, expert_prefix: str, *, dense: QuantConfig | None):
        super().__init__(experts.name_map, ())
        self.experts = experts
        self.expert_prefix = expert_prefix
        self.dense = dense if dense is not None else _UNQUANTIZED

    @classmethod
    def claims(cls, q: dict[str, Any]) -> bool:
        return False  # never selected by a checkpoint; a draft head is wrapped explicitly

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        return self.dense.scheme_for_name(name)

    def scheme_for(self, prefix: str) -> QuantScheme | None:
        return self.dense.scheme_for(prefix)

    def get_quant_method(self, layer: Any, prefix: str):
        if layer.quant_layer_kind is LayerKind.MOE:
            return self.experts.get_quant_method(layer, self.expert_prefix)
        return self.dense.get_quant_method(layer, prefix)


def draft_head_config(config, *, dense_from_model: bool):
    """``config`` for building a draft head's decoder layer, or unchanged when the model
    carries no quant config (dummy weights, unit tests: the head then matches by default)."""
    from dataclasses import replace

    if config.quant is None:
        return config
    first_moe = int(getattr(config, "first_k_dense_replace", 0) or 0)
    return replace(
        config,
        quant=DraftHeadQuantConfig(
            config.quant,
            f"model.layers.{first_moe}.mlp.experts",
            dense=config.quant if dense_from_model else None,
        ),
    )


__all__ = ["DraftHeadQuantConfig", "draft_head_config"]
