from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


@dataclass(frozen=True)
class PipelineInfo:
    """Pipeline (layer-split) placement of this process: decoder layers ``[start, end)`` of
    ``num_layers`` run here; rank 0 owns the embedding, the last rank owns the head."""

    rank: int
    size: int
    start: int
    end: int
    num_layers: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size
        assert 0 <= self.start < self.end <= self.num_layers, (self.start, self.end, self.num_layers)

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.size - 1

    def is_primary(self) -> bool:
        return self.rank == 0

    def owns_layer(self, layer_id: int) -> bool:
        return self.start <= layer_id < self.end

    def bank_window(self, first_k_dense: int = 0) -> tuple[int, int]:
        """The MoE-bank layer range ``[s, e)`` this rank serves (bank index = layer - first_k_dense)."""
        return max(0, self.start - first_k_dense), max(0, self.end - first_k_dense)


def pp_layer_range(num_layers: int, split: tuple[int, ...] | None, rank: int, size: int) -> tuple[int, int]:
    """Layer range of ``rank``; ``split`` lists the ``size - 1`` boundaries, None = even split."""
    if split is None:
        bounds = [round(num_layers * i / size) for i in range(size + 1)]
    else:
        if len(split) != size - 1:
            raise ValueError(f"--pp-layers needs {size - 1} boundaries for --pp-size {size}, got {len(split)}")
        bounds = [0, *split, num_layers]
    if any(bounds[i] >= bounds[i + 1] for i in range(size)) or bounds[-1] != num_layers:
        raise ValueError(f"--pp-layers boundaries {bounds[1:-1]} must be strictly increasing within (0, {num_layers})")
    return bounds[rank], bounds[rank + 1]


_PP_INFO: PipelineInfo | None = None


def set_pp_info(rank: int, size: int, start: int, end: int, num_layers: int) -> None:
    global _PP_INFO
    if _PP_INFO is not None:
        raise RuntimeError("PP info has been set")
    _PP_INFO = PipelineInfo(rank, size, start, end, num_layers)


def get_pp_info() -> PipelineInfo:
    if _PP_INFO is None:
        raise RuntimeError("PP info has not been set")
    return _PP_INFO


def try_get_pp_info() -> PipelineInfo | None:
    return _PP_INFO


def try_get_world_info() -> DistributedInfo | PipelineInfo | None:
    """Whatever identifies this process among its peers: the PP placement when the engine runs
    layer-split, else the TP info. For rank-0-only logging and progress bars."""
    return _PP_INFO if _PP_INFO is not None else _TP_INFO


__all__ = [
    "DistributedInfo",
    "PipelineInfo",
    "set_tp_info",
    "get_tp_info",
    "try_get_tp_info",
    "set_pp_info",
    "get_pp_info",
    "try_get_pp_info",
    "try_get_world_info",
    "pp_layer_range",
]
