"""Point-to-point transport for the pipeline (layer-split) engine.

One engine process per GPU, each running a contiguous block of decoder layers on the same
lockstep scheduler the TP path uses. Per forward the residual stream flows rank r -> r+1
and the sampled tokens flow last rank -> every other rank. Everything goes through the
gloo CPU group (pinned staging + D2H/H2D), so it works without NCCL and without P2P: the
payload is one hidden vector per token, far below what PCIe staging costs matter for.

Prefill overlap: a batch of non-final prefill chunks (``Batch.pp_no_tokens``) carries no
tokens back, so the first rank sends its residual stream and goes straight on to the next
chunk while the last rank is still computing this one. Hidden and token traffic use distinct
gloo tags so the two directions never match each other's messages.

``--pp-send-ahead N`` is how far ahead the first rank may get. At 1 (the default) the send blocks
until the peer posts its receive, so the rank stays at most one chunk ahead and one staging buffer
is enough. Above 1 the send goes out non-blocking from a ring of N pinned buffers and only blocks
when all N are still in flight -- the rank keeps prefilling chunks while the slower rank works
through what it already has. Measured part by part on two RTX 3060s (Qwen3.8-Flash-Next,
5120-token chunks): the second rank waited 0.22 s per 1k tokens on the first, and the first 0.43 s
on the second. gloo delivers one tag's messages in send order, so the receiver still sees the
chunks in order; each slot costs one chunk's residual stream of pinned host memory (98 MiB at
5120 tokens on that model).

Every send/recv here blocks without a timeout, and a peer that stopped leaves this rank inside
one of them silently; each is bracketed for the rank-wait watchdog (distributed/watchdog), which
logs the wait once it passes FREETOKEN_RANK_WAIT_WARN_SECONDS.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist

from freetoken.utils import init_logger
from freetoken.utils.prefill_profile import active as _prefill_profile

from .info import PipelineInfo
from .watchdog import rank_wait_watchdog

logger = init_logger(__name__)

# gloo has no int16/bf16 send path on every build; ship raw bytes instead
_BYTE_VIEW = torch.uint8
# gloo tags: one per message kind (the scheduler's message-count channel is tag 7, see io.py)
_TAG_HIDDEN = 1
_TAG_TOKENS = 2


class PipelineComm:
    def __init__(
        self,
        info: PipelineInfo,
        group: dist.ProcessGroup,
        device: torch.device,
        send_ahead: int = 1,
    ):
        self.info = info
        self.group = group
        self.device = device
        self.hidden_width: int | None = None
        self.hidden_dtype: torch.dtype | None = None
        self._stage: dict[str, torch.Tensor] = {}
        self._waits = rank_wait_watchdog()
        # --pp-send-ahead: how many residual streams may be in flight to the next rank at once
        self.send_ahead = max(1, int(send_ahead))
        self._in_flight: list[tuple[object, str]] = []  # (work, staging key), oldest first
        self._pin_warned = False
        self._free_keys = [f"send{i}" for i in range(self.send_ahead)]

    @property
    def is_first(self) -> bool:
        return self.info.is_first

    @property
    def is_last(self) -> bool:
        return self.info.is_last

    def configure(self, hidden_width: int, hidden_dtype: torch.dtype) -> None:
        self.hidden_width = hidden_width
        self.hidden_dtype = hidden_dtype

    def _staging(self, key: str, nbytes: int) -> torch.Tensor:
        """Pinned host buffer of at least ``nbytes``, grown on demand (one per direction, plus
        one per --pp-send-ahead slot). Falls back to unpinned when the host cannot pin any more
        (the expert banks take most of that quota under WSL): the copy is slower, not wrong."""
        buf = self._stage.get(key)
        if buf is None or buf.numel() < nbytes:
            cuda = self.device.type == "cuda"
            if cuda:
                # the previous buffer may still be the source of an in-flight H2D copy
                torch.cuda.current_stream(self.device).synchronize()
            try:
                buf = torch.empty(nbytes, dtype=_BYTE_VIEW, pin_memory=cuda)
            except RuntimeError as exc:
                if not cuda:
                    raise
                if not self._pin_warned:
                    self._pin_warned = True
                    logger.warning(
                        f"pipeline staging buffer {key} ({nbytes >> 20} MiB) could not be pinned "
                        f"({exc}); using unpinned host memory for the rank hand-off"
                    )
                buf = torch.empty(nbytes, dtype=_BYTE_VIEW)
            self._stage[key] = buf
        return buf[:nbytes]

    def send_hidden(self, hidden: torch.Tensor) -> None:
        assert not self.is_last
        dst = self.info.rank + 1
        key = self._take_send_slot(dst)
        flat = hidden.contiguous().view(-1).view(_BYTE_VIEW)
        buf = self._staging(key, flat.numel())
        buf.copy_(flat)  # synchronous D2H: the send below reads it on the host
        if self.send_ahead == 1:
            # returns once the peer posts its recv, i.e. after it finishes its previous forward
            self._waits.begin(
                "rank {peer} to take the hidden states ({detail} rows)", dst, hidden.shape[0]
            )
            prof = _prefill_profile()
            try:
                with prof.peer_wait() if prof is not None else nullcontext():
                    dist.send(buf, dst=dst, group=self.group, tag=_TAG_HIDDEN)
            finally:
                self._waits.end()
            self._free_keys.append(key)
            return
        # The buffer stays reserved until the send completes; gloo keeps one tag's messages in
        # send order, so the peer receives the chunks as they were handed over.
        work = dist.isend(buf, dst=dst, group=self.group, tag=_TAG_HIDDEN)
        self._in_flight.append((work, key))

    def _take_send_slot(self, dst: int) -> str:
        """A free staging key, waiting out the oldest send in flight when all of them are busy
        (--pp-send-ahead > 1). That wait is this rank running ``send_ahead`` chunks ahead of the
        next one, so it is billed to the peer like the blocking send is."""
        if self._free_keys:
            return self._free_keys.pop()
        work, key = self._in_flight.pop(0)
        self._waits.begin(
            "rank {peer} to take the hidden states sent {detail} sends ago", dst, len(self._in_flight) + 1
        )
        prof = _prefill_profile()
        try:
            with prof.peer_wait() if prof is not None else nullcontext():
                work.wait()
        finally:
            self._waits.end()
        return key

    def drain_sends(self) -> None:
        """Wait out every residual stream still in flight (--pp-send-ahead > 1)."""
        while self._in_flight:
            work, key = self._in_flight.pop(0)
            work.wait()
            self._free_keys.append(key)

    def recv_hidden(self, rows: int) -> torch.Tensor:
        assert not self.is_first
        assert self.hidden_width is not None and self.hidden_dtype is not None, "configure() first"
        nbytes = rows * self.hidden_width * self.hidden_dtype.itemsize
        buf = self._staging("recv", nbytes)
        src = self.info.rank - 1
        self._waits.begin("the hidden states ({detail} rows) from rank {peer}", src, rows)
        prof = _prefill_profile()
        try:
            with prof.peer_wait() if prof is not None else nullcontext():
                dist.recv(buf, src=src, group=self.group, tag=_TAG_HIDDEN)
        finally:
            self._waits.end()
        # synchronous H2D so the staging buffer can be reused by the next recv
        return buf.to(self.device).view(self.hidden_dtype).view(rows, self.hidden_width)

    def send_tokens(self, tokens_cpu: torch.Tensor) -> None:
        """Last rank: hand the sampled tokens to every other rank."""
        assert self.is_last
        tokens_cpu = tokens_cpu.to(torch.int32).contiguous()
        for dst in range(self.info.size - 1):
            self._waits.begin(
                "rank {peer} to take the sampled tokens ({detail})", dst, tokens_cpu.numel()
            )
            prof = _prefill_profile()
            try:
                with prof.peer_wait() if prof is not None else nullcontext():
                    dist.send(tokens_cpu, dst=dst, group=self.group, tag=_TAG_TOKENS)
            finally:
                self._waits.end()

    def recv_tokens(self, count: int) -> torch.Tensor:
        assert not self.is_last
        buf = torch.empty(count, dtype=torch.int32)
        src = self.info.size - 1
        self._waits.begin("the sampled tokens ({detail}) from rank {peer}", src, count)
        prof = _prefill_profile()
        try:
            with prof.peer_wait() if prof is not None else nullcontext():
                dist.recv(buf, src=src, group=self.group, tag=_TAG_TOKENS)
        finally:
            self._waits.end()
        return buf


__all__ = ["PipelineComm"]
