# SOURCE: vllm/v1/engine/core_client.py
# ch34 切面（m14/m15/m16/m17）：make_async_mp_client 三分支工厂 + DPAsyncMPClient
# （外部 LB/统计订阅/FIRST_REQ 转投）+ DPLBAsyncMPClient（内部 LB 打分选路/在飞
# 记账/abort 按引擎路由）。
# 删除（dossier 删除项 5）：elastic EP 面（eep_scaling_cache 与 SCALE_ELASTIC_EP
# 分支 L1332-L1367、_prepared_elastic_ep/commit/prepare_elastic_ep、
# eep_process_engine_core_notification L1541-L1585）、get_core_engine_for_request
# 的 late-interaction pooling 分支（L1470-L1474）。
# 父类 AsyncMPClient 的引擎发射/握手/IO 任务面按 ch05 域收窄（行内标注）。

from __future__ import annotations

import asyncio
import sys
import uuid
from collections import Counter, defaultdict
from typing import Any

import msgspec.msgpack
import zmq
import zmq.asyncio  # noqa: F401  (stats 任务的 asyncio socket 面)

from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_inproc_path, make_zmq_socket
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType, EngineCoreOutputs
from vllm.v1.serial_utils import MsgpackEncoder

# SUBTRACTED: EngineDeadError/PauseMode/SupportedTask/TensorIpcSender/
#   CoreEngineProcManager/launch_core_engines 等 import（L20-L73）——ch05 域与
#   收窄面的依赖；EngineDeadError 以锚位载体保留。

logger = init_logger(__name__)

# SOURCE: vllm/v1/engine/core_client.py:L75 EngineIdentity — 逐字
EngineIdentity = bytes


# SOURCE: vllm/v1/engine/exceptions.py:L12 EngineDeadError —— 锚位载体
#   （VLLMServerError 基类按 ch05 域收窄为 Exception）
class EngineDeadError(Exception):  # HOST/ch05 SEAM
    # SOURCE: vllm/v1/engine/exceptions.py:L12 EngineDeadError（锚点双置）
    pass


# SOURCE: vllm/v1/engine/core_client.py:L78-L303 EngineCoreClient 抽象基类 ——
#   收窄为 make_async_mp_client 工厂的宿主（其余抽象面 L141-L303 归 ch05 域）
class EngineCoreClient:
    # SUBTRACTED: 其余抽象方法族（L141-L160）——ch05 域。

    # SOURCE: vllm/v1/engine/core_client.py:L116-L139 make_async_mp_client
    #   —— 逐字（前端家族三分支）
    @staticmethod
    def make_async_mp_client(
        vllm_config,
        executor_class,
        log_stats,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        # SOURCE: vllm/v1/engine/core_client.py:L116-L139 make_async_mp_client（锚点双置）
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        return AsyncMPClient(*client_args)


# SOURCE: vllm/v1/engine/core_client.py:L974-L1246 AsyncMPClient —— 协作件装配
#   子集。真实类树 AsyncMPClient(MPClient)（MPClient L503-L777：引擎发射/握手/
#   ready 等待，ch05 域）按扁平载体承载：MPClient.__init__ 的 identity 表段
#   （L632-L672：engine_ranks_managed/core_engines/core_engine/utility_results）
#   与 shutdown/_format_exception/ensure_alive/dp_engines_running（L682-L706）
#   原行号并入；AsyncMPClient.__init__（L978-L1014）的 ctx/队列段并入。
class AsyncMPClient(EngineCoreClient):
    # SOURCE: vllm/v1/engine/core_client.py:L978-L1014+L516-L680 __init__ —— 装配子集
    def __init__(
        self,
        vllm_config,
        executor_class,
        log_stats,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        # SOURCE: vllm/v1/engine/core_client.py:L978-L1014+L516-L680（锚点双置）
        self.vllm_config = vllm_config
        self.client_count = client_count
        self.client_index = client_index
        self.engines_running = False
        self.ctx = zmq.Context()
        # SUBTRACTED: 引擎发射（client_addresses 外部管理模式 / launch_core_
        #   engines 内部模式 / ROUTER 绑定与 ready 等待 / 引擎监控线程，
        #   L547-L680）——ch05 域（ch09 的 e2e 承担）；本章测试以替身注入
        #   input_socket / stats_update_address。
        self.input_socket = None  # ch05 SEAM（真实为 ROUTER socket）
        self.stats_update_address = None  # ch05 SEAM
        self.outputs_queue = None  # ch05 SEAM（真实为输出轮询任务的 asyncio 队列）
        self.utility_results: dict[int, Any] = {}

        class _Resources:  # ch05 SEAM：资源登记表（engine_dead/任务/socket 位）
            # SOURCE: vllm/v1/engine/core_client.py:L406-L493 BackgroundResources —— HOST/ch05 SEAM 载体（锚点双置）
            engine_dead = False
            stats_update_task = None
            stats_update_socket = None
            first_req_send_socket = None
            first_req_rcv_socket = None
            engine_manager = None

        self.resources = _Resources()

        parallel_config = vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_index
        dp_local_size = parallel_config.data_parallel_size_local
        offline_mode = parallel_config.data_parallel_rank_local is not None
        # Client manages local+remote EngineCores in pure internal LB case.
        # Client manages local EngineCores in hybrid and external LB case.
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        # SOURCE: vllm/v1/engine/core_client.py:L639-L641 engine_ranks_managed
        self.engine_ranks_managed = (
            [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
        )

        # SOURCE: vllm/v1/engine/core_client.py:L646-L649 core_engines identity
        # ZMQ identity of each engine that this client will talk to.
        self.core_engines: list[EngineIdentity] = [
            rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
        ]
        # SUBTRACTED: ready 等待循环（L651-L669）——ch05 域。

        # SOURCE: vllm/v1/engine/core_client.py:L671 core_engine
        self.core_engine: EngineIdentity = self.core_engines[0]

    # SOURCE: vllm/v1/engine/core_client.py:L682-L693 shutdown（MPClient）—— ch05 域收窄
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        # SUBTRACTED: engine_manager 关停与资源回收（ch05 域）。
        return None

    # SOURCE: vllm/v1/engine/core_client.py:L695-L699 _format_exception（MPClient）—— 逐字
    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    # SOURCE: vllm/v1/engine/core_client.py:L701-L703 ensure_alive（MPClient）—— 逐字
    def ensure_alive(self):
        if self.resources.engine_dead:
            raise EngineDeadError()

    # SOURCE: vllm/v1/engine/core_client.py:L705-L706 dp_engines_running（MPClient）—— 逐字
    def dp_engines_running(self) -> bool:
        return self.engines_running

    # SOURCE: vllm/v1/engine/core_client.py:L1104-L1114 _send_input —— 逐字
    #   （ROUTER 信封首帧 = 目标引擎 identity 的装配位；F4 兑现处）
    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ):
        # SOURCE: vllm/v1/engine/core_client.py:L1104-L1114 _send_input（锚点双置）
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine)

    # SOURCE: vllm/v1/engine/core_client.py:L1116-L1123 _send_input_message ——
    #   逐字（首帧 = 目标引擎 identity 的发送现场）
    def _send_input_message(
        self, message: tuple, engine: EngineIdentity
    ):
        # SOURCE: vllm/v1/engine/core_client.py:L1116-L1123 _send_input_message（锚点双置）
        self.ensure_alive()
        # Any zero-copy tensor/ndarray frames are kept alive by zmq itself
        # until it's finished sending them (there is a ref chain from the underlying
        # memoryview back to the original owning tensor/ndarray).
        return self.input_socket.send_multipart((engine,) + message, copy=False)

    # SUBTRACTED: encoder/decoder 的张量零拷贝通道（MsgpackEncoder 构造位，
    #   L621-L630）——ch05 域；单缓冲编码器 seam：
    @property
    def encoder(self):  # ch05 SEAM
        # SOURCE: vllm/v1/engine/core_client.py:L621-L630 encoder —— HOST/ch05 SEAM（单缓冲）（锚点双置）
        return MsgpackEncoder()

    # SOURCE: vllm/v1/engine/core_client.py:L1145-L1148 add_request_async（基类
    #   AsyncMPClient）—— 逐字
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # SOURCE: vllm/v1/engine/core_client.py:L1145-L1148 add_request_async（基类）（锚点双置）
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1093-L1102 get_output_async —— 逐字
    #   （outputs_queue 的 ch05 IO 任务面以注入承载）
    async def get_output_async(self) -> EngineCoreOutputs:
        # SOURCE: vllm/v1/engine/core_client.py:L1093-L1102 get_output_async（锚点双置）
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L1016-L1091 _ensure_output_queue_task
    #   —— ch05 域收窄（PULL socket 轮询任务）
    def _ensure_output_queue_task(self):
        # SUBTRACTED: 输出 socket 轮询 asyncio 任务（ch05 域）。
        # SOURCE: vllm/v1/engine/core_client.py:L1016-L1091 _ensure_output_queue_task（锚点双置）
        return None


# SOURCE: vllm/v1/engine/core_client.py:L1249-L1428 DPAsyncMPClient — 逐字 minus
#   删除项 5（eep_scaling_cache / SCALE_ELASTIC_EP 分支）
class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""

    # SOURCE: vllm/v1/engine/core_client.py:L1253-L1291 __init__ — 逐字 minus
    #   eep_scaling_cache（L1279，删除项 5）
    def __init__(
        self,
        vllm_config,
        executor_class,
        log_stats,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        # SOURCE: vllm/v1/engine/core_client.py:L1253-L1291 __init__（DPAsyncMPClient）（锚点双置）
        self.current_wave = 0

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        # List of [waiting, running, kv_cache_usage] per engine.
        # Used only by DPLBAsyncMPClient subclass.
        self.lb_engines: list[list[int | float]] = [
            [0, 0, 0.0] for _ in self.core_engines
        ]

        # SUBTRACTED: eep_scaling_cache（L1279）——删除项 5。

        self.first_req_sock_addr = get_open_zmq_inproc_path()
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()
            self._ensure_stats_update_task()
        except RuntimeError:
            pass

    # SOURCE: vllm/v1/engine/core_client.py:L1293-L1408 _ensure_stats_update_task
    #   —— 逐字 minus SCALE_ELASTIC_EP 分支（L1332-L1367，删除项 5）
    def _ensure_stats_update_task(self):
        # SOURCE: vllm/v1/engine/core_client.py:L1293-L1408 _ensure_stats_update_task（锚点双置）
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0

        async def run_engine_stats_update_task():
            # SOURCE: vllm/v1/engine/core_client.py:L1302-L1404 run_engine_stats_update_task（锚点双置）
            with (
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,
                make_zmq_socket(
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0
                ) as first_req_rcv_socket,
            ):
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    if (
                        not self.engines_running
                        and len(events) == 2
                        or (events[0][0] == first_req_rcv_socket)
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()

                        decoded = msgspec.msgpack.decode(buf)
                        # SUBTRACTED: SCALE_ELASTIC_EP 通知分支（L1332-L1367）——
                        #   删除项 5（弹性扩缩容归 ch39）。

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        assert decoded[0] == "FIRST_REQ"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    buf = None
                    while True:
                        # Drain all stats events (we only care about latest).
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)
                        if isinstance(future.exception(), zmq.Again):
                            break
                        buf = future.result()
                    if buf is None:
                        continue

                    # Update local load-balancing state.
                    counts, wave, running = msgspec.msgpack.decode(buf)
                    self.current_wave = wave
                    self.engines_running = running
                    if counts is not None:
                        # Running and waiting counts are global from the
                        # Coordinator including all EngineCores. Slice to get
                        # just the cores managed by this client.
                        ranks = self.engine_ranks_managed
                        count_slice = slice(ranks[0], ranks[-1] + 1)
                        sliced_counts = counts[count_slice]
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )

        resources.stats_update_task = asyncio.create_task(
            run_engine_stats_update_task()
        )

    # SOURCE: vllm/v1/engine/core_client.py:L1410-L1425 add_request_async — 逐字
    #   （盖章+定向+FIRST_REQ 抢先唤醒——F3/F4 回收现场）
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # SOURCE: vllm/v1/engine/core_client.py:L1410-L1425 add_request_async（DPAsync）（锚点双置）
        self._ensure_stats_update_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)

        await to_await

        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1427-L1428 get_core_engine_for_request
    #   —— 逐字（外部 LB：绑定的引擎原样返回）
    def get_core_engine_for_request(self, request: EngineCoreRequest):
        # SOURCE: vllm/v1/engine/core_client.py:L1427-L1428 get_core_engine_for_request（外部 LB）（锚点双置）
        return self.core_engine


# SOURCE: vllm/v1/engine/core_client.py:L1431-L1871 DPLBAsyncMPClient —— 逐字 minus
#   删除项 5（elastic EP 面与 late-interaction pooling 分支）
class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""

    # SOURCE: vllm/v1/engine/core_client.py:L1435-L1466 __init__ — 逐字 minus
    #   _prepared_elastic_ep（L1462，删除项 5）
    def __init__(
        self,
        vllm_config,
        executor_class,
        log_stats,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        # SOURCE: vllm/v1/engine/core_client.py:L1435-L1466 __init__（DPLBAsyncMPClient）（锚点双置）
        self.client_count = client_count

        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, EngineIdentity] = {}

        # Exact per-engine count of this client's unfinished requests.
        self.engine_inflight: Counter[EngineIdentity] = Counter()

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        assert len(self.core_engines) > 1
        # SUBTRACTED: _prepared_elastic_ep（L1462）——删除项 5。

        self.eng_start_index = (
            len(self.core_engines) * self.client_index
        ) // client_count

    # SOURCE: vllm/v1/engine/core_client.py:L1468-L1519 get_core_engine_for_request
    #   —— 逐字 minus late-interaction pooling 分支（L1470-L1474，删除项 5）；
    #   v0.27.1 重写版 score 主式与 KV 斜坡注释原文保留
    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        # SUBTRACTED: late-interaction pooling 引擎选择分支（L1470-L1474）——
        #   删除项 5（专用功能）。
        # SOURCE: vllm/v1/engine/core_client.py:L1468-L1519 get_core_engine_for_request（内部 LB）（锚点双置）
        if (eng_index := request.data_parallel_rank) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score: float = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running, kv_cache_usage = current_counts[idx]
                # Estimate engine load as the greater of the coordinator's
                # latest (waiting + running) snapshot and this client's own
                # in-flight count (scaled by the number of clients). The
                # in-flight floor is exact and can't be erased by a snapshot
                # rebind, so a burst spreads round-robin even when snapshots
                # race with routing decisions; the snapshot raises the score
                # when other clients or stale requests load the engine.
                inflight = self.engine_inflight[self.core_engines[idx]]
                score: float = max(self.client_count * inflight, waiting + running)
                if waiting:
                    # Waiting requests are penalized in proportion to KV cache
                    # pressure: a queue on a KV-bound engine drains slowly, so
                    # new requests should strongly prefer other engines. With
                    # low KV usage the queue is transient (e.g. mid-burst) and
                    # the penalty stays off, preserving exact round-robin.
                    # Ramps from 0 at <=50% usage to 3x waiting at 100%.
                    score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count
            # Rotate the scan start so that ties (equal scores, e.g. right
            # after a coordinator stats reset when engines look equally loaded)
            # don't systematically favor the same engine. This removes the
            # fixed tie-break bias without affecting load-aware decisions when
            # scores actually differ.
            self.eng_start_index = (self.eng_start_index + 1) % num_engines

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        self.engine_inflight[chosen_engine] += 1
        return chosen_engine

    # SOURCE: vllm/v1/engine/core_client.py:L1521-L1530 call_utility_async — 逐字
    async def call_utility_async(self, method: str, *args) -> Any:
        # Only the result from the first engine is returned.
        return (
            await asyncio.gather(
                *[
                    self._call_utility_async(method, *args, engine=engine)
                    for engine in self.core_engines
                ]
            )
        )[0]

    # SOURCE: vllm/v1/engine/core_client.py:L1128-L1140 _call_utility_async — 逐字
    #   （基类携带；DPLB 的 call_utility_async 依赖它）
    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        # SOURCE: vllm/v1/engine/core_client.py:L1128-L1140 _call_utility_async（锚点双置）
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (
            EngineCoreRequestType.UTILITY.value,
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        await self._send_input_message(message, engine)
        self._ensure_output_queue_task()
        return await future

    # SOURCE: vllm/v1/engine/core_client.py:L1533-L1539 process_engine_outputs
    #   —— 逐字（finished 回收 engine_inflight——score 地板的会计闭环）
    @staticmethod
    async def process_engine_outputs(
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        # SOURCE: vllm/v1/engine/core_client.py:L1533-L1539 process_engine_outputs（锚点双置）
        if outputs.finished_requests and self.reqs_in_flight:
            for req_id in outputs.finished_requests:
                if (engine := self.reqs_in_flight.pop(req_id, None)) is not None:
                    self.engine_inflight[engine] -= 1

    # SUBTRACTED: eep_process_engine_core_notification（L1541-L1585）——删除项 5。

    # SOURCE: vllm/v1/engine/core_client.py:L1587-L1602 abort_requests_async
    #   —— 逐字（m16：abort 按引擎路由）
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # SOURCE: vllm/v1/engine/core_client.py:L1587-L1602 abort_requests_async（锚点双置）
        if not request_ids or self.resources.engine_dead:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine[engine].append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    # SOURCE: vllm/v1/engine/core_client.py:L1604-L1607 _abort_requests — 逐字
    async def _abort_requests(
        self, request_ids: list[str], engine: EngineIdentity
    ) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)

    # SUBTRACTED: commit_elastic_ep / prepare_elastic_ep / _commit_scale_up /
    #   _commit_scale_down（L1609-L1870）——删除项 5（弹性扩缩容归 ch39）。
