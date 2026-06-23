"""Host bridge for the ch21 subtract-only companion — NOT new vLLM abstractions.

ch21 reads three real vLLM mechanisms whose *control flow* is pure Python but
whose *dependencies* (torch.distributed P2P handles, a DP-group all-reduce, ZMQ
pub/sub sockets, msgspec encode/decode) need CUDA / a live distributed world /
real sockets that the host lacks.

This module supplies **semantic-equivalent minimal stand-ins** so the real
control flow in the companion modules can be driven and value-traced on the
host. It invents no vLLM concept: every stand-in mirrors the exact behaviour the
companion relies on (a `.wait()`-able handle, an int32 SUM/MAX all-reduce over a
process group, an XPUB/PULL/PAIR socket queue, msgpack encode/decode).

Anything touching real NCCL / ZMQ / msgspec belongs in the
`vllm/vllm-openai` container; here we keep just enough to exercise the logic.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable


# --- torch.distributed P2P handle stand-in -------------------------------
# Real vLLM: torch.distributed.irecv / isend return a `Work` handle whose
# `.wait()` blocks until the async op completes. Here a handle just records
# whether it was waited on, and optionally runs an on-wait callback so tests
# can assert "the wait actually happened at first .tensors access".
# SOURCE: vllm/distributed/parallel_state.py:L961 (torch.distributed Work `Handle`)
class Handle:
    """Stand-in for torch.distributed Work handle (the `.wait()`-able object)."""

    # SOURCE: torch.distributed.irecv/isend return value (parallel_state.py:L1010)
    def __init__(self, on_wait: Callable[[], None] | None = None) -> None:
        self._on_wait = on_wait
        self.waited = False

    # SOURCE: handle.wait() in wait_for_comm / execute_model (gpu_worker.py:L92,779)
    def wait(self) -> None:
        self.waited = True
        if self._on_wait is not None:
            self._on_wait()


# --- DP process group + all-reduce stand-in ------------------------------
# Real vLLM: ParallelConfig.sync_dp_state / has_unfinished_dp call
# torch.distributed.all_reduce(tensor, op, group=dp_group). We model a
# `dp_group` as a barrier-synchronized set of ranks that each contribute a
# vector; the group reduces element-wise with SUM or MAX, exactly like the
# integer all-reduce vLLM relies on.
# SOURCE: torch.distributed.ReduceOp (used at parallel.py:L662,687)
class ReduceOp:
    SUM = "sum"
    MAX = "max"


# SOURCE: dp_group (ProcessGroup) passed to sync_dp_state (parallel.py:L667)
class FakeDPGroup:
    """In-process stand-in for a torch.distributed DP process group.

    Each of `world` ranks calls `all_reduce(rank, vec, op)`; the call blocks
    until all ranks have contributed, then every rank gets the reduced vector.
    Mirrors the collective semantics the companion's sync_dp_state depends on.
    """

    # SOURCE: stateless_init_dp_group → self.dp_group (core.py:L1684)
    def __init__(self, world: int) -> None:
        self._world = world
        self._lock = threading.Condition()
        self._contrib: dict[int, list[int]] = {}
        self._result: list[int] | None = None
        self._gen = 0

    # SOURCE: dp_group.size() in sync_dp_state (parallel.py:L688)
    def size(self) -> int:
        return self._world

    # SOURCE: torch.distributed.all_reduce(tensor, op, group=dp_group) (parallel.py:L687)
    def all_reduce(self, rank: int, vec: list[int], op: str) -> list[int]:
        with self._lock:
            gen = self._gen
            self._contrib[rank] = list(vec)
            if len(self._contrib) == self._world:
                acc = [0] * len(vec)
                for v in self._contrib.values():
                    for i, x in enumerate(v):
                        if op == ReduceOp.SUM:
                            acc[i] += x
                        else:  # MAX
                            acc[i] = max(acc[i], x)
                self._result = acc
                self._gen += 1
                self._contrib = {}
                self._lock.notify_all()
            else:
                while self._gen == gen:
                    self._lock.wait()
            return list(self._result)  # type: ignore[arg-type]


# --- ZMQ socket stand-in --------------------------------------------------
# Real vLLM: coordinator/clients exchange msgpack frames over ZMQ XPUB/XSUB/
# PULL/PAIR sockets. We model each link as a thread-safe deque carrying already
# msgpack-decoded Python objects; `send`/`recv` move frames, `send_multipart`
# carries (type, payload). This preserves the message *content* and *ordering*
# the wave state machine reacts to, without a ZMQ runtime.
# SOURCE: zmq.Again (caught at core_client.py:L1270)
class Again(Exception):
    """Raised by recv(block=False) when no frame is available (zmq.Again)."""


# SOURCE: awaited zmq.asyncio.Socket.send return (core_client.py:L1307)
class _Sent:
    """Awaitable returned by FakeSocket.send.

    Real vLLM's zmq.asyncio sockets are awaited (`await socket.send(...)`) on the
    client side but used synchronously on the coordinator side. The frame is
    enqueued eagerly on construction, so both `socket.send(x)` (sync) and
    `await socket.send(x)` work identically.
    """

    # SOURCE: `await self.first_req_send_socket.send(req_msg)` (core_client.py:L1307)
    def __await__(self):
        return iter(())


# SOURCE: zmq XPUB/XSUB/PULL/PAIR sockets (coordinator.py:L214-231, core_client.py:L1190)
class FakeSocket:
    # SOURCE: make_zmq_socket(...) construction (coordinator.py:L214)
    def __init__(self, name: str = "") -> None:
        self.name = name
        self._q: deque[Any] = deque()
        self._lock = threading.Lock()

    # SOURCE: socket.send(msgspec.msgpack.encode(...)) (coordinator.py:L283,447)
    def send(self, frame: Any) -> "_Sent":
        with self._lock:
            self._q.append(frame)
        return _Sent()

    # SOURCE: socket.send_multipart((type, payload)) (coordinator.py:L459)
    def send_multipart(self, parts: tuple[Any, ...]) -> None:
        with self._lock:
            self._q.append(tuple(parts))

    # SOURCE: socket.recv() / recv(flags=zmq.NOBLOCK) (coordinator.py:L291, core_client.py:L1269)
    def recv(self, block: bool = True) -> Any:
        with self._lock:
            if self._q:
                return self._q.popleft()
        if not block:
            raise Again()
        raise Again()

    # SOURCE: poller.poll() readiness check (coordinator.py:L272)
    def pending(self) -> bool:
        with self._lock:
            return bool(self._q)
