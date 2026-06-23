"""DP load-balancing client — subtract-only companion.

Faithful subset of the async DP clients in vllm/v1/engine/core_client.py:

  * `make_async_mp_client`: factory choosing AsyncMPClient / DPAsyncMPClient
    (external LB) / DPLBAsyncMPClient (internal LB) by dp_size + external_lb.
  * `DPAsyncMPClient`: stamps `current_wave` on each request, sends FIRST_REQ to
    wake the coordinator when engines are paused; `run_engine_stats_update_task`
    consumes the coordinator's (counts, wave, running) broadcast.
  * `DPLBAsyncMPClient`: `get_core_engine_for_request` picks the least-loaded
    engine via score = waiting*4 + running, pre-incrementing the local waiting
    count to absorb the ~100ms stats refresh lag.

ZMQ sockets + msgpack are modeled by `_bridge.FakeSocket`/plain tuples; the
elastic-EP (SCALE_ELASTIC_EP) and late-interaction routing branches are
subtracted per the approved delete plan. async methods are kept `async` to match
vLLM's signatures; tests drive them via asyncio.run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from ._bridge import FakeSocket


@dataclass
class EngineCoreRequest:  # SOURCE: vllm/v1/engine/__init__.py (EngineCoreRequest)
    request_id: str
    current_wave: int = 0
    client_index: int = 0
    data_parallel_rank: int | None = None
    pooling_params: Any = None


# EngineIdentity in vLLM is the engine's rank encoded as bytes; an int rank is
# the faithful stand-in for routing/identity here.
EngineIdentity = int


# SOURCE: vllm/v1/engine/core_client.py:L887 (AsyncMPClient — non-DP base subset)
class AsyncMPClient:
    """Single-engine async client (dp_size == 1)."""

    def __init__(  # SOURCE: vllm/v1/engine/core_client.py:L893 (AsyncMPClient.__init__)
        self,
        client_count: int = 1,
        client_index: int = 0,
        num_engines: int = 1,
    ) -> None:
        self.client_count = client_count
        self.client_index = client_index
        self.engines_running = False
        # engine identities in rank order
        self.core_engines: list[EngineIdentity] = list(range(num_engines))
        self.core_engine: EngineIdentity = self.core_engines[0]
        self.engine_ranks_managed: list[int] = list(range(num_engines))
        # in vLLM these go over ZMQ; here the test reads them back
        self._sent_inputs: list[tuple[bytes, EngineCoreRequest, EngineIdentity]] = []

    # SOURCE: vllm/v1/engine/core_client.py:L1179 (_ensure_stats_update_task)
    def _ensure_stats_update_task(self) -> None:
        # SUBTRACTED: lazy asyncio task creation for the stats subscriber — task
        #   lifecycle plumbing · core_client.py:L1179-1294 (logic kept as the
        #   explicit consume_stats_frame below)
        pass

    # SOURCE: vllm/v1/engine/core_client.py:L921 (_ensure_output_queue_task)
    def _ensure_output_queue_task(self) -> None:
        # SUBTRACTED: lazy output-queue drain task · core_client.py:L921
        pass

    # SOURCE: vllm/v1/engine/core_client.py:L1001 (_send_input)
    async def _send_input(
        self, request_type: bytes, request: EngineCoreRequest, engine: EngineIdentity
    ) -> None:
        # SUBTRACTED: msgpack encode + ZMQ send_multipart — wire transport ·
        #   core_client.py:L1001-1011
        self._sent_inputs.append((request_type, request, engine))


# SOURCE: vllm/v1/engine/core_client.py (EngineCoreRequestType.ADD)
class EngineCoreRequestType:
    ADD = b"\x00"


# SOURCE: vllm/v1/engine/core_client.py:L1137 (DPAsyncMPClient — external LB)
class DPAsyncMPClient(AsyncMPClient):
    """External-LB DP client: one client bound to a single DP rank."""

    # SOURCE: vllm/v1/engine/core_client.py:L1141
    def __init__(
        self,
        client_count: int = 1,
        client_index: int = 0,
        num_engines: int = 1,
        first_req_send_socket: FakeSocket | None = None,
    ) -> None:
        self.current_wave = 0
        super().__init__(client_count, client_index, num_engines)
        # List of [waiting, running] per engine; used by DPLB subclass.
        self.lb_engines: list[list[int]] = [[0, 0] for _ in self.core_engines]
        # SUBTRACTED: eep_scaling_cache (elastic-EP) · core_client.py:L1165
        self.first_req_send_socket = first_req_send_socket or FakeSocket("first_req")

    # SOURCE: vllm/v1/engine/core_client.py:L1296
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        await self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request.
            req_msg = ("FIRST_REQ", chosen_engine)
            await self.first_req_send_socket.send(req_msg)

        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1313
    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        return self.core_engine

    # SOURCE: vllm/v1/engine/core_client.py:L1266-1290 (stats subscriber body)
    def consume_stats_frame(self, frame: Any) -> None:
        """Consume one coordinator (counts, wave, running) broadcast frame.

        This is the inner body of `run_engine_stats_update_task`'s drain loop:
        the client trusts only the coordinator's periodic global broadcast and
        slices out the rank range it manages.
        """
        # SUBTRACTED: the surrounding `while True` poller + FIRST_REQ/
        #   SCALE_ELASTIC_EP XSUB up-link handling + NOBLOCK drain-to-latest —
        #   socket polling around the same state update · core_client.py:L1206-1274
        counts, wave, running = frame
        self.current_wave = wave
        self.engines_running = running
        if counts is not None:
            # Global counts from the Coordinator; slice to the cores we manage.
            ranks = self.engine_ranks_managed
            count_slice = slice(ranks[0], ranks[-1] + 1)
            sliced_counts = counts[count_slice]
            self.lb_engines = sliced_counts


# SOURCE: vllm/v1/engine/core_client.py:L1317 (DPLBAsyncMPClient — internal LB)
class DPLBAsyncMPClient(DPAsyncMPClient):
    """Internal-LB DP client: balances across all DP engine processes."""

    # SOURCE: vllm/v1/engine/core_client.py:L1321
    def __init__(
        self,
        client_count: int = 1,
        client_index: int = 0,
        num_engines: int = 2,
        first_req_send_socket: FakeSocket | None = None,
    ) -> None:
        self.client_count = client_count
        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, EngineIdentity] = {}
        super().__init__(client_count, client_index, num_engines, first_req_send_socket)
        assert len(self.core_engines) > 1
        self.eng_start_index = (
            len(self.core_engines) * self.client_index
        ) // client_count

    # SOURCE: vllm/v1/engine/core_client.py:L1350
    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := _get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            # SUBTRACTED: the two walrus branches above stay (explicit-rank /
            #   late-interaction routing skip LB) but late-interaction always
            #   returns None here — pooling-hash routing detail ·
            #   core_client.py:L1352-1356
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help balance when engines are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine


# SOURCE: vllm/v1/engine/utils.py (get_late_interaction_engine_index)
def _get_late_interaction_engine_index(pooling_params: Any, num_engines: int):
    # SUBTRACTED: real pooling-hash routing for late-interaction models — returns
    #   None for ordinary requests, the only path this chapter exercises ·
    #   core_client.py:L1353
    return None


@dataclass
class _ParallelConfig:  # SOURCE: vllm/config/parallel.py (ParallelConfig — DP fields)
    data_parallel_size: int = 1
    data_parallel_external_lb: bool = False


# SOURCE: vllm/v1/engine/core_client.py:L105 (make_async_mp_client factory)
def make_async_mp_client(
    parallel_config: _ParallelConfig,
    client_count: int = 1,
    client_index: int = 0,
    num_engines: int = 1,
) -> AsyncMPClient:
    # SUBTRACTED: vllm_config/executor_class/log_stats/client_addresses args +
    #   client_args tuple packing — construction wiring; the branch logic (the
    #   point of this factory) is kept verbatim · core_client.py:L107-123
    if parallel_config.data_parallel_size > 1:
        if parallel_config.data_parallel_external_lb:
            # External load balancer - client per DP rank.
            return DPAsyncMPClient(client_count, client_index, num_engines)
        # Internal load balancer - client balances to all DP ranks.
        return DPLBAsyncMPClient(client_count, client_index, num_engines)
    return AsyncMPClient(client_count, client_index, num_engines)
