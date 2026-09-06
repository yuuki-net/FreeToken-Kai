"""Point-to-point transport for the pipeline (layer-split) engine.

One engine process per GPU, each running a contiguous block of decoder layers on the same
lockstep scheduler the TP path uses. Per forward the residual stream flows rank r -> r+1
and the sampled tokens flow last rank -> every other rank. Everything goes through the
gloo CPU group (pinned staging + D2H/H2D), so it works without NCCL and without P2P: the
payload is one hidden vector per token, far below what PCIe staging costs matter for.

Prefill overlap: a batch of non-final prefill chunks (``Batch.pp_no_tokens``) carries no
tokens back, so the first rank sends its residual stream and goes straight on to the next
chunk while the last rank is still computing this one. ``send_hidden`` blocks until the peer
posts its receive, which keeps the first rank at most one chunk ahead (and the single
staging buffer safe). Hidden and token traffic use distinct gloo tags so the two directions
never match each other's messages.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .info import PipelineInfo

# gloo has no int16/bf16 send path on every build; ship raw bytes instead
_BYTE_VIEW = torch.uint8
# gloo tags: one per message kind (the scheduler's message-count channel is tag 7, see io.py)
_TAG_HIDDEN = 1
_TAG_TOKENS = 2


class PipelineComm:
    def __init__(self, info: PipelineInfo, group: dist.ProcessGroup, device: torch.device):
        self.info = info
        self.group = group
        self.device = device
        self.hidden_width: int | None = None
        self.hidden_dtype: torch.dtype | None = None
        self._stage: dict[str, torch.Tensor] = {}

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
        """Pinned host buffer of at least ``nbytes``, grown on demand (one per direction)."""
        buf = self._stage.get(key)
        if buf is None or buf.numel() < nbytes:
            cuda = self.device.type == "cuda"
            if cuda:
                # the previous buffer may still be the source of an in-flight H2D copy
                torch.cuda.current_stream(self.device).synchronize()
            buf = torch.empty(nbytes, dtype=_BYTE_VIEW, pin_memory=cuda)
            self._stage[key] = buf
        return buf[:nbytes]

    def send_hidden(self, hidden: torch.Tensor) -> None:
        assert not self.is_last
        flat = hidden.contiguous().view(-1).view(_BYTE_VIEW)
        buf = self._staging("send", flat.numel())
        buf.copy_(flat)  # synchronous D2H: the send below reads it on the host
        dist.send(buf, dst=self.info.rank + 1, group=self.group, tag=_TAG_HIDDEN)

    def recv_hidden(self, rows: int) -> torch.Tensor:
        assert not self.is_first
        assert self.hidden_width is not None and self.hidden_dtype is not None, "configure() first"
        nbytes = rows * self.hidden_width * self.hidden_dtype.itemsize
        buf = self._staging("recv", nbytes)
        dist.recv(buf, src=self.info.rank - 1, group=self.group, tag=_TAG_HIDDEN)
        # synchronous H2D so the staging buffer can be reused by the next recv
        return buf.to(self.device).view(self.hidden_dtype).view(rows, self.hidden_width)

    def send_tokens(self, tokens_cpu: torch.Tensor) -> None:
        """Last rank: hand the sampled tokens to every other rank."""
        assert self.is_last
        tokens_cpu = tokens_cpu.to(torch.int32).contiguous()
        for dst in range(self.info.size - 1):
            dist.send(tokens_cpu, dst=dst, group=self.group, tag=_TAG_TOKENS)

    def recv_tokens(self, count: int) -> torch.Tensor:
        assert not self.is_last
        buf = torch.empty(count, dtype=torch.int32)
        dist.recv(buf, src=self.info.size - 1, group=self.group, tag=_TAG_TOKENS)
        return buf


__all__ = ["PipelineComm"]
