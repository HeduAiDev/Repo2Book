# HOST SEAM: shared-memory broadcast MessageQueue — the ch09 `_msgspec_seam`
# convention (separate file for a dependency outside this chapter's
# subtract-only surface). The real class is
# vllm/distributed/device_communicators/shm_broadcast.py:L465+ (SHM ring buffer
# for small messages + ZMQ XPUB for overflow/remote readers, SpinCondition
# over an in-proc PAIR socket for cancel). This seam keeps the *observable
# control-plane contract* the executor chapter exercises, on real ZMQ:
#
#   * one enqueue  -> every reader receives the same message, FIFO;
#   * dequeue(timeout=...) raises TimeoutError; indefinite=True blocks;
#   * wait_until_ready() is the collective handshake (writer counts N
#     subscriptions, then publishes b"READY"; readers wait for it);
#   * shutdown() wakes a blocked reader with RuntimeError("cancelled")
#     (real: acquire_read raises "cancelled" after the SpinCondition cancel
#     ping, shm_broadcast.py:L797-L798);
#   * export_handle()/create_from_handle() move a pipe-picklable Handle.
#
# Deviations (registered in impl-notes.md §Seam 清单): payloads ride the ZMQ
# message instead of the SHM ring buffer (no writer-side backpressure;
# control-plane volumes are tiny), and ipc:// endpoints become loopback tcp://
# (ch05/ch09 win32 precedent).

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import zmq

from ._host_seams import init_logger, get_open_port

logger = init_logger(__name__)

# HOST SEAM: 进程级单一 zmq Context + LINGER=0——泄漏的 socket 会让
# context.term() 在解释器退出时永久阻塞（pytest unconfigure 的 gc 钩实测
# 卡死）；shutdown() 里显式关 socket，进程退出即回收。
_context = zmq.Context()


# SOURCE: (见 impl-notes.md §Source Map——_shm_broadcast_seam.py)
def _mk_socket(stype) -> "zmq.Socket":  # HOST SEAM
    sock = _context.socket(stype)
    sock.setsockopt(zmq.LINGER, 0)
    return sock


# SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L455-L463 Handle
@dataclass
class Handle:
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L456
    local_reader_ranks: list[int] = field(default_factory=list)
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L460
    local_subscribe_addr: str | None = None
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L462
    remote_subscribe_addr: str | None = None
    # SUBTRACTED: buffer_handle / local_notify_addr / remote_addr_ipv6
    #   （shm_broadcast.py:L458/L461/L463）——SHM 环形缓冲与远端订阅轴不在
    #   本 seam 的实现面内：载荷直接走 ZMQ tcp，缓冲句柄无从谈起。


# SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L465 MessageQueue
class MessageQueue:
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L496-L556 __init__
    def __init__(
        self,
        n_reader,  # number of all readers
        n_local_reader,  # number of local readers through shared memory
        local_reader_ranks: list[int] | None = None,
        # Default of 24MiB chosen to be large enough to accommodate grammar
        # bitmask tensors for large batches (1024 requests).
        max_chunk_bytes: int = 1024 * 1024 * 24,
        max_chunks: int = 10,
        connect_ip: str | None = None,
    ):
        # HOST SEAM: 单节点星形——全部读者都是 local，XPUB 走 loopback tcp。
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader
        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader
        self.shutting_down = False
        context = _context  # HOST SEAM（进程级共享 Context）

        assert n_local_reader > 0, "seam: 远端读者轴已按删除项 2 裁除（单节点）"
        # XPUB is very similar to PUB,
        # except that it can receive subscription messages
        # to confirm the number of subscribers
        self.local_socket = _mk_socket(zmq.XPUB)
        # set the verbose option so that we can receive every subscription
        # message. otherwise, we will only receive the first subscription
        # see http://api.zeromq.org/3-3:zmq-setsockopt for more details
        self.local_socket.setsockopt(zmq.XPUB_VERBOSE, True)
        port = get_open_port()
        local_subscribe_addr = f"tcp://127.0.0.1:{port}"
        logger.debug("Binding to %s", local_subscribe_addr)
        self.local_socket.bind(local_subscribe_addr)
        self.current_idx = -1  # writer side never reads the ring (seam: n/a)

        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1
        # rank does not matter for remote readers
        self._is_remote_reader = False
        self._context = context

        self.handle = Handle(
            local_reader_ranks=local_reader_ranks,
            local_subscribe_addr=local_subscribe_addr,
            remote_subscribe_addr=None,
        )
        logger.debug("vLLM message queue communication handle: %s", self.handle)

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L557-L559 export_handle
    def export_handle(self) -> Handle:
        return self.handle
    @staticmethod
    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L560-L606 create_from_handle
    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False

        context = _context  # HOST SEAM（进程级共享 Context）

        if rank in handle.local_reader_ranks:
            self.local_reader_rank = handle.local_reader_ranks.index(rank)
            self._is_local_reader = True
            self._is_remote_reader = False

            self.local_socket = _mk_socket(zmq.SUB)
            self.local_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket_addr = handle.local_subscribe_addr
            logger.debug("Connecting to %s", socket_addr)
            self.local_socket.connect(socket_addr)

            # SpinCondition seam (shm_broadcast.py:L130-L166): reader polls the
            # subscribe socket AND an in-proc PAIR cancel socket, so a monitor
            # thread in this process can wake a blocked dequeue on shutdown.
            cancel_addr = f"inproc://cancel-{id(self):x}"
            self.write_cancel_socket = _mk_socket(zmq.PAIR)
            self.write_cancel_socket.bind(cancel_addr)
            self.read_cancel_socket = _mk_socket(zmq.PAIR)
            self.read_cancel_socket.connect(cancel_addr)
            self.poller = zmq.Poller()
            self.poller.register(self.read_cancel_socket, zmq.POLLIN)
            self.poller.register(self.local_socket, zmq.POLLIN)
        else:
            # SUBTRACTED: 远端读者分支（shm_broadcast.py:L589-L605）——跨节点
            #   response MQ 装配已按删除项 2 裁除，单节点 rank 必在 local 列表。
            raise AssertionError(
                f"seam: rank {rank} not in local_reader_ranks "
                f"{handle.local_reader_ranks} (multi-node subtracted)"
            )

        self.current_idx = 0
        self.shutting_down = False
        self._context = context
        return self

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L608-L639 wait_until_ready
    def wait_until_ready(self):
        """This is a collective operation. All processes (including the
        readers and the writer) should call this function.
        """
        if self._is_writer:
            # wait for all readers to connect

            # local readers
            for i in range(self.n_local_reader):
                # wait for subscription messages from all local readers
                self.local_socket.recv()
            if self.n_local_reader > 0:
                # send a message to all local readers
                # to make sure the publish channel is working
                self.local_socket.send(b"READY")
        elif self._is_local_reader:
            # wait for the writer to send a message
            recv = self.local_socket.recv()
            assert recv == b"READY"

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L641-L645 shutdown
    def shutdown(self):
        """If this is an idle reader, wakes it up so it can clean up and shut
        down"""
        self.shutting_down = True
        if getattr(self, "read_cancel_socket", None) is not None:
            # SpinCondition.cancel（shm_broadcast.py:L185-L190）：进程内 PAIR
            # 发一声 ping，唤醒阻塞在 poller 上的读者。
            self.write_cancel_socket.send(b"\x00")

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L824-L881 enqueue
    def enqueue(self, obj, timeout: float | None = None):
        """Write to message queue with optional timeout (in seconds)"""
        assert self._is_writer, "Only writers can enqueue"
        # HOST SEAM: pickle 进消息体直发（真实路径：小于 max_chunk_bytes 走 SHM
        # 环形缓冲 acquire_write，超限走 XPUB multipart——宿主无 /dev/shm）。
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self.local_socket.send(payload)

    # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L883-L905 dequeue
    def dequeue(
        self,
        timeout: float | None = None,
        indefinite: bool = False,
    ):
        """Read from message queue with optional timeout (in seconds)"""
        if not self._is_local_reader:
            raise RuntimeError("Only readers can dequeue")
        # Ensure non-negative timeout passed to zmq poll.
        timeout_ms = (
            None if (timeout is None or indefinite) else max(0, int(timeout * 1000))
        )
        events = dict(self.poller.poll(timeout=timeout_ms))
        if not events:
            raise TimeoutError
        if self.read_cancel_socket in events:
            # SOURCE: vllm/distributed/device_communicators/shm_broadcast.py:L797-L798
            #   （shutting_down 后 acquire_read 抛 "cancelled"）
            raise RuntimeError("cancelled")
        recv = self.local_socket.recv(copy=False)
        return pickle.loads(bytes(recv))
