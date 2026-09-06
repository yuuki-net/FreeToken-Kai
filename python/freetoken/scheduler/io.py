from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from freetoken.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg
from freetoken.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)

# gloo tag of the per-step "how many raw messages follow" note (pipeline hidden/tokens: 1/2)
_MSG_COUNT_TAG = 7


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            if tp_info.is_primary():
                recv = self._recv_msg_multi_rank0
                self._pending_count_sends: list = []
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[BaseTokenizerMsg]) -> None:
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # tell every other rank how many raw messages follow
        self._publish_msg_count(len(pending_raw_msgs))

        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        dst_length = self._await_msg_count()

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _publish_msg_count(self, count: int) -> None:
        """Rank 0: how many raw messages the other ranks must take off the pub socket this
        step. Point-to-point and asynchronous rather than a broadcast: a collective holds
        rank 0 until every rank reaches the same step, and the pipeline engine wants rank 0
        one prefill chunk ahead of the last rank (distributed/pipeline). Each rank consumes
        exactly one count per step, in order, so the sends never pile up; the tensors stay
        referenced until their send has completed."""
        self._pending_count_sends = [
            (t, w) for t, w in self._pending_count_sends if not w.is_completed()
        ]
        count_t = torch.tensor([count], dtype=torch.int64)
        for dst in range(1, self.tp_cpu_group.size()):
            work = self.tp_cpu_group.send([count_t], dst, _MSG_COUNT_TAG)
            self._pending_count_sends.append((count_t, work))

    def _await_msg_count(self) -> int:
        """Other ranks: this step's message count from rank 0 (blocks until rank 0 got there)."""
        buf = torch.tensor([-1], dtype=torch.int64)
        self.tp_cpu_group.recv([buf], 0, _MSG_COUNT_TAG).wait()
        return int(buf.item())

    def _reply_tokenizer_rank0(self, reply: List[BaseTokenizerMsg]) -> None:
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[BaseTokenizerMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks
