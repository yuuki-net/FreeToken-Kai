from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Deque, Final, List, Tuple

import msgpack
import torch
from freetoken.distributed.watchdog import rank_wait_watchdog
from freetoken.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg
from freetoken.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)

# gloo tag of the per-step "how many raw messages follow" note (pipeline hidden/tokens: 1/2)
_MSG_COUNT_TAG = 7
# How many un-retired count sends rank 0 may carry. The channel exists so rank 0 can run
# ahead of the last rank (one prefill chunk); this bounds how far. Retiring is a wait() on
# the oldest, which returns as soon as the peer has taken that message.
_MAX_PENDING_COUNT_SENDS: Final = 64

# The relay join (_join_rank_relay). 0xc1 is the one byte msgpack never emits, so neither frame
# can be taken for a relayed request, which is always a msgpack map.
_RELAY_HELLO: Final = bytes([0xC1]) + b"relay-hello"
_RELAY_JOINED: Final = bytes([0xC1]) + b"relay-joined"
_RELAY_POLL_MS: Final = 50
# How long rank 0 keeps publishing hellos before every rank gives up together. Each round waits
# in a collective for all ranks, so this only counts time they are all here; a subscription lands
# in well under a second (upstream #364 measured 100-200 ms), so a minute means it never will.
_RELAY_JOIN_TIMEOUT_S: Final = 60.0


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
                self._pending_count_sends: Deque[Tuple[torch.Tensor, object]] = deque()
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
        if tp_info.size > 1:
            self._join_rank_relay(tp_info)

    def _join_rank_relay(self, tp_info) -> None:
        """Return only once every other rank's SUB is receiving what rank 0's PUB publishes.

        Rank 0 relays each request to the other ranks over ZeroMQ PUB/SUB, and a PUB drops every
        frame no registered subscription matches. A SUB's connect and SUBSCRIBE reach the PUB on
        ZeroMQ's I/O thread, some time after the socket exists, and nothing waited for it: a
        request published in that window was dropped, rank 1 waited in the SUB for a frame that
        was never coming, rank 0 went on into the forward and waited for rank 1, and the server
        answered /v1/models while serving nothing (upstream #364; under --pp-size the same relay
        carries every request). A client that sends its first request the moment the server says
        ready is exactly the one that lands in that window.

        So rank 0 publishes hellos until every rank has received one, the ranks agreeing each round
        over the gloo group: rank 0 contributes 1, or -1 once it gives up; the others 1 once a hello
        has arrived, else 0. The MIN is 1 when all have heard, and -1 makes every rank raise
        together, rather than rank 0 raising while the others wait on. Then rank 0 publishes one
        joined frame and each rank drains the surplus hellos up to it -- a registered
        subscription delivers in order, so nothing of the join can be read as a request later.
        """
        t0 = time.monotonic()
        primary = tp_info.is_primary()
        heard = 0
        rounds = 0
        while True:
            rounds += 1
            if primary:
                self._send_into_ranks.socket.send(_RELAY_HELLO)
                mine = -1 if time.monotonic() - t0 > _RELAY_JOIN_TIMEOUT_S else 1
            else:
                sub = self._recv_from_rank0.socket
                if not heard and sub.poll(timeout=_RELAY_POLL_MS):
                    self._expect_relay_frame(sub.recv(), _RELAY_HELLO)
                    heard = 1
                mine = heard
            state = torch.tensor([mine], dtype=torch.int64)
            torch.distributed.all_reduce(
                state, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
            )
            agreed = int(state.item())
            if agreed == 1:
                break
            if agreed < 0:
                raise RuntimeError(
                    f"rank relay: after {time.monotonic() - t0:.0f}s and {rounds} hellos not every "
                    "rank's SUB was receiving rank 0's PUB "
                    f"({'rank 0 gave up' if primary else 'this rank heard one' if heard else 'this rank heard none'})"
                )
        if primary:
            self._send_into_ranks.socket.send(_RELAY_JOINED)
            logger.info(
                f"rank relay: {tp_info.size - 1} rank(s) subscribed after {rounds} hello(s) "
                f"in {(time.monotonic() - t0) * 1000:.0f} ms"
            )
            return
        sub = self._recv_from_rank0.socket
        while True:
            # rank 0 sent the joined frame right after the round that agreed, so it is already on
            # its way; a bounded wait turns a lost one into an error rather than a silent stall
            if not sub.poll(timeout=int(_RELAY_JOIN_TIMEOUT_S * 1000)):
                raise RuntimeError("rank relay: rank 0 agreed but its joined frame never arrived")
            frame = sub.recv()
            if frame == _RELAY_JOINED:
                return
            self._expect_relay_frame(frame, _RELAY_HELLO)

    @staticmethod
    def _expect_relay_frame(frame: bytes, expected: bytes) -> None:
        if frame != expected:
            raise RuntimeError(
                f"rank relay: expected {expected!r} while joining, got {bytes(frame[:32])!r}"
            )

    def run_when_idle(self):
        raise NotImplementedError("should be implemented")

    def _rank0_notes(self) -> List[BaseBackendMsg]:
        """Rank 0: messages of its own to relay this step after the tokenizer's (decisions the
        other ranks must apply at the same step). The scheduler overrides this."""
        return []

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
        # rank 0's own notes ride behind the tokenizer's messages, counted with them, and rank 0
        # handles them in the same place of the same step as every other rank does
        notes = self._rank0_notes()
        for note in notes:
            pending_raw_msgs.append(msgpack.packb(note.encoder(), use_bin_type=True))

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

        if dst_length:
            # Rank 0 published these before its count, so each get() should return at once. PUB
            # drops (does not block) for a subscriber past its high-water mark, and a dropped
            # message leaves this rank here for good -- bracket it for the watchdog.
            waits = rank_wait_watchdog()
            for i in range(dst_length):
                waits.begin("relayed request {detail} of this step from rank {peer}", 0, i + 1)
                try:
                    pending_msgs.append(self._recv_from_rank0.get())
                finally:
                    waits.end()
        return pending_msgs

    def _publish_msg_count(self, count: int) -> None:
        """Rank 0: how many raw messages the other ranks must take off the pub socket this
        step. Point-to-point and asynchronous rather than a broadcast: a collective holds
        rank 0 until every rank reaches the same step, and the pipeline engine wants rank 0
        one prefill chunk ahead of the last rank (distributed/pipeline). Each rank consumes
        exactly one count per step, in order; the tensors stay referenced until their send
        has been retired.

        Retiring is a wait() on the OLDEST entry once the backlog exceeds the window, not a
        scan for finished ones: gloo's SendWork marks itself completed inside wait(), so
        ``is_completed()`` stays False for the life of the object even after the peer has
        long since taken the message. Filtering on it retired nothing -- the list grew by one
        entry per scheduler iteration and every iteration re-walked all of it, so each step
        paid a cost proportional to the number of steps served so far (Flash-Next on 2x3060:
        18 -> 2.3 tok/s over a 12-hour session, +0.4 s per decode step; prefill was unaffected
        because a 6 s chunk hides it). The window keeps both the run-ahead and the walk
        bounded."""
        count_t = torch.tensor([count], dtype=torch.int64)
        for dst in range(1, self.tp_cpu_group.size()):
            work = self.tp_cpu_group.send([count_t], dst, _MSG_COUNT_TAG)
            self._pending_count_sends.append((count_t, work))
        while len(self._pending_count_sends) > _MAX_PENDING_COUNT_SENDS:
            _, oldest = self._pending_count_sends.popleft()
            waits = rank_wait_watchdog()
            waits.begin("another rank to take a message count sent {detail} sends ago", None,
                        _MAX_PENDING_COUNT_SENDS)
            try:
                oldest.wait()
            finally:
                waits.end()

    def _await_msg_count(self) -> int:
        """Other ranks: this step's message count from rank 0 (blocks until rank 0 got there)."""
        buf = torch.tensor([-1], dtype=torch.int64)
        waits = rank_wait_watchdog()
        waits.begin("this step's message count from rank {peer}", 0)
        try:
            self.tp_cpu_group.recv([buf], 0, _MSG_COUNT_TAG).wait()
        finally:
            waits.end()
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
