"""--pp-prefill-group: on the last of two pipeline ranks, run several prefill chunks of one
request layer by layer, so each offloaded MoE layer's expert bank crosses the bus once per group
instead of once per chunk.

Why: an offloaded MoE streams a layer's whole expert bank to the GPU for every prefill chunk,
whatever the chunk holds. Measured on two RTX 3060s (Qwen3.8-Flash-Next, 3072-token chunks), the
second rank sits on a PCIe x4 link: 33 GiB per chunk at 5.9 GiB/s, 5.6 s, against about 3 s of
compute. That rank waited on its banks for half its time, and the first rank waited on it. The
transfer per chunk is fixed, so the remedy is fewer transfers per token.

How: a chunk-only batch (``Batch.pp_no_tokens``) of one request is deferred. The rank takes the
residual stream (the first rank is blocked sending it), plans the chunk's pieces while the request
still describes this chunk, advances the request as a forward would (``complete_one``) and returns
a no-token output; the scheduler reads nothing from a chunk's output. Once ``size`` chunks are
held, or before anything that could observe or free their state, the group runs: every local
layer over every chunk in order, then the draft head over each chunk. A layer's bank is prefetched
for the first chunk and still in its buffer for the rest.

Why the order is sound: within a layer, chunk k+1 reads only chunk k's state of that same layer
(GDN recurrent and conv state, QSA KV and index), and the previous layer's output of chunk k+1.
Each chunk's metadata (attention, GDN, pieces, the snapshot slot it tracks into) was built from
the request as it was when that chunk was scheduled. The next layer is never touched before a
layer is done for every chunk, so no state is read before it is written.

What must never see a held chunk's state unwritten -- the flush points:

* the engine's next batch, unless it is this request's next chunk;
* a COW restore of a GDN snapshot (``Scheduler._restore_linear_states``): the scheduler's drain
  has already donated each chunk's checkpoint to the prefix tree;
* freeing the request (``Scheduler._free_req_resources``: finish or abort);
* a cache rebuild.

It needs ``--max-running-req 1``, so that no other request can be scheduled and match a donated
checkpoint between two chunks without passing a flush point first. The same rule covers the GDN
state pool: a commit locks only the newest checkpoint node, so an older held chunk's snapshot
slot can be evicted and handed out again before the group runs -- but with one running request
the only taker is a later chunk of the same request, which writes that slot after the older
chunk does (the group runs its chunks in order), and the tree no longer points at it. KV pages
of held chunks are either owned by the request or on the locked node's path, never evictable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import torch


@dataclass
class DeferredChunk:
    batch: Any
    hidden: torch.Tensor  # the residual stream received for this chunk
    rows: int
    pieces: list | None  # models/prefill_pieces.py plan, built at defer time


@dataclass
class PrefillGroup:
    size: int
    chunks: List[DeferredChunk] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)

    @staticmethod
    def request_key(batch) -> tuple:
        req = batch.reqs[0]
        return (req.uid, req.table_idx)

    def accepts(self, batch) -> bool:
        """Whether ``batch`` may join the held chunks: nothing held, or the same request."""
        return not self.chunks or self.request_key(self.chunks[0].batch) == self.request_key(batch)

    def add(self, chunk: DeferredChunk) -> bool:
        """Hold ``chunk``; True when the group is full and has to run."""
        self.chunks.append(chunk)
        return len(self.chunks) >= self.size

    def take(self) -> List[DeferredChunk]:
        chunks, self.chunks = self.chunks, []
        return chunks


def groupable(batch, *, cpu_prefill_max_tokens: int) -> bool:
    """A batch the last pipeline rank may defer: one request's non-final prefill chunk that
    streams expert banks. A chunk short enough for the CPU executor copies no bank, a multimodal
    chunk runs the encoder, a verify window is a decode."""
    if not batch.is_prefill or getattr(batch, "spec_verify", False):
        return False
    if not getattr(batch, "pp_no_tokens", False) or batch.size != 1 or len(batch.reqs) != 1:
        return False
    if getattr(batch, "mm_gather_plan", None):
        return False
    return int(batch.input_ids.numel()) > cpu_prefill_max_tokens


def unusable_reason(*, size: int, pp_comm, max_running_req: int, model, offload_cache) -> str | None:
    """Why --pp-prefill-group cannot run on this rank, or None. Checked once at startup."""
    if size <= 1:
        return "off"
    if pp_comm is None or pp_comm.is_first or not pp_comm.is_last:
        return "only the last pipeline rank of a --pp-size run groups chunks"
    if pp_comm.info.size != 2:
        return "only two pipeline ranks are supported"
    if max_running_req != 1:
        return "it needs --max-running-req 1"
    if not hasattr(model, "forward_prefill_group"):
        return f"{type(model).__name__} has no grouped prefill forward"
    if offload_cache is None or not offload_cache.prefill_overlap:
        return "the experts are not offloaded with the prefill overlap buffers"
    return None


__all__ = ["DeferredChunk", "PrefillGroup", "groupable", "unusable_reason"]
