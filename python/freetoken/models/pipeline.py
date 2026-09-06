"""Layer-split (pipeline) serving, the model-side helpers (``--pp-size``, see
``distributed/pipeline.py`` for the transport and ``engine/config.py`` for the config window).

Each pipeline rank builds the whole ``layers`` list so the state-dict numbering stays global
(``model.layers.<id>``), but only the layers of its window own weights and run; the others are
``RemoteLayer`` placeholders. Rank 0 owns the embedding, the last rank owns the final norm and
the head; a non-last rank returns the residual stream it hands on (``[T, pp_hidden_width]``)
and a non-first rank starts from the stream it received (``Ctx.pp_hidden_in``). A single
process is the window ``[0, num_layers)``: first and last at once."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import try_get_pp_info
from freetoken.layers import BaseOP


class RemoteLayer(BaseOP):
    """Placeholder for a decoder layer another pipeline rank runs: keeps the state-dict
    numbering global while owning no weights and never executing."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("a remote pipeline layer was invoked on this rank")


@dataclass(frozen=True)
class LayerWindow:
    """Decoder layers ``[start, end)`` of ``num_layers`` this process runs."""

    start: int
    end: int
    num_layers: int

    @property
    def first(self) -> bool:
        return self.start == 0

    @property
    def last(self) -> bool:
        return self.end == self.num_layers

    @property
    def local_ids(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end))

    def owns(self, layer_id: int) -> bool:
        return self.start <= layer_id < self.end


def layer_window(config) -> LayerWindow:
    """This process's window of ``config``'s decoder stack (the whole stack when the pipeline
    engine is not active)."""
    pp = try_get_pp_info()
    if pp is None:
        return LayerWindow(0, config.num_layers, config.num_layers)
    return LayerWindow(pp.start, pp.end, config.num_layers)


def received_hidden(input_ids: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    """The residual stream a non-first rank received for the active forward. Warmup and probe
    forwards run without a peer: they get a zero stream of the right shape."""
    hidden_in = get_global_ctx().pp_hidden_in
    if hidden_in is None:
        hidden_in = torch.zeros((input_ids.numel(), width), dtype=dtype, device=input_ids.device)
    return hidden_in


__all__ = ["LayerWindow", "RemoteLayer", "layer_window", "received_hidden"]
