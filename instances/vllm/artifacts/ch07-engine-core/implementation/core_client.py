# SPDX-License-Identifier: Apache-2.0
# 只做减法的忠实子集 —— 对应 vllm/v1/engine/core_client.py。
# 源码 pin: f3fef123。三层 client（Inproc/Sync/AsyncMP）+ ZMQ socket 装配 + ready 握手
#   + 字节标签多帧发送 + call_utility(_async) 的 call_id+Future RPC + BackgroundResources 终结清理。
#
# 重要简化（subtract-only 边界外的唯一结构性裁剪，已注明）：真实 MPClient 经
#   launch_core_engines 起独立子进程跑 EngineCoreProc；本精简版在同进程内用 daemon 线程承载
#   EngineCoreProc（ZMQ socket 仍是真实跨"端点"通信，控制流/协议 1:1）。这样读者无需 fork
#   子进程即可跑通 ROUTER<->DEALER / PUSH<->PULL 全链路。原 vllm/v1/engine/core_client.py:L535。

import asyncio
import contextlib
import queue
import uuid
import weakref
from abc import ABC
from collections import deque
from collections.abc import Awaitable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import Any, TypeAlias

import msgspec.msgpack
import zmq
import zmq.asyncio

from core import EngineCore, EngineCoreProc
from engine_init import (
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    UtilityOutput,
)
from serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr
from tensor_ipc import TensorIpcSender

AnyFuture: TypeAlias = "asyncio.Future[Any] | Future[Any]"
EngineIdentity = bytes


# SOURCE: vllm/v1/engine/exceptions.py  EngineDeadError
class EngineDeadError(Exception):
    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        # SOURCE: vllm/v1/engine/exceptions.py  EngineDeadError.__init__
        # SUBTRACTED: suppress_context 构造参数（日志旁支）。原 vllm/v1/engine/exceptions.py。
        super().__init__("EngineCore encountered an issue." if not args else args[0])


# SOURCE: vllm/v1/engine/core_client.py:L69  EngineCoreClient
class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: Any = None,
        executor_class: Any = None,
        log_stats: bool = False,
    ) -> "EngineCoreClient":
        # SOURCE: vllm/v1/engine/core_client.py:L80  EngineCoreClient.make_client
        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @staticmethod
    def make_async_mp_client(
        vllm_config: Any = None,
        executor_class: Any = None,
        log_stats: bool = False,
    ) -> "AsyncMPClient":
        # SOURCE: vllm/v1/engine/core_client.py:L105  EngineCoreClient.make_async_mp_client
        # SUBTRACTED: data_parallel_size>1 时返回 DPAsyncMPClient/DPLBAsyncMPClient
        #   （delete 第1项批准）。单 engine 直接走 AsyncMPClient。原 :L124-L130。
        return AsyncMPClient(vllm_config, executor_class, log_stats)


# SOURCE: vllm/v1/engine/core_client.py:L274  InprocClient
class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/v1/engine/core_client.py:L284  InprocClient.__init__
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        # SOURCE: vllm/v1/engine/core_client.py:L287  InprocClient.get_output
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    def get_supported_tasks(self) -> tuple[str, ...]:
        # SOURCE: vllm/v1/engine/core_client.py:L292  InprocClient.get_supported_tasks
        return self.engine_core.get_supported_tasks()

    def add_request(self, request: EngineCoreRequest) -> None:
        # SOURCE: vllm/v1/engine/core_client.py:L295  InprocClient.add_request
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    def abort_requests(self, request_ids: list[str]) -> None:
        # SOURCE: vllm/v1/engine/core_client.py:L299  InprocClient.abort_requests
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self, timeout: float | None = None) -> None:
        # SOURCE: vllm/v1/engine/core_client.py:L303  InprocClient.shutdown
        pass

    # SUBTRACTED: profile / reset_*_cache / sleep / wake_up / add_lora / collective_rpc 等
    #   直通 self.engine_core 的运维方法（delete 第5项：保留机制，逐个直通属冗余）。
    #   原 vllm/v1/engine/core_client.py:L306-L364。


# SOURCE: vllm/v1/engine/core_client.py:L367  BackgroundResources
@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: zmq.Context
    output_socket: zmq.Socket | None = None
    input_socket: zmq.Socket | None = None
    output_queue_task: "asyncio.Task | None" = None
    shutdown_path: str | None = None
    engine_thread: Thread | None = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False

    def __call__(self):
        # SOURCE: vllm/v1/engine/core_client.py:L390  BackgroundResources.__call__
        """Clean up background resources."""
        self.engine_dead = True
        # SUBTRACTED: engine_manager/coordinator.shutdown（子进程管理，本章用线程承载）+
        #   async/sync 分支的 loop.call_soon_threadsafe / PAIR shutdown 信号细节
        #   （delete 第4项：保留最小关闭路径）。原 vllm/v1/engine/core_client.py:L390-L445。
        for sock in (self.output_socket, self.input_socket):
            if sock is not None:
                with contextlib.suppress(Exception):
                    sock.close(linger=0)
        if self.output_queue_task is not None and not self.output_queue_task.done():
            with contextlib.suppress(Exception):
                self.output_queue_task.cancel()

    # SOURCE: vllm/v1/engine/core_client.py:L447  BackgroundResources.validate_alive
    def validate_alive(self, frames: Sequence[zmq.Frame]):
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


# SOURCE: vllm/v1/engine/core_client.py:L460  MPClient
class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: Any = None,
        executor_class: Any = None,
        log_stats: bool = False,
        tensor_queue: Any = None,
        input_address: str | None = None,
        output_address: str | None = None,
    ):
        # SOURCE: vllm/v1/engine/core_client.py:L473  MPClient.__init__
        self.vllm_config = vllm_config

        # ZMQ setup.
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        try:
            # client 侧: input=ROUTER(bind) / output=PULL(bind)。
            # SUBTRACTED: 真实地址来自 get_engine_zmq_addresses（ipc:// 跨进程）；
            #   线程承载用 tcp:// 临时端口，仍是真实跨 ZMQ context 通信。原 :L523。
            self.input_socket = self.resources.input_socket = self.ctx.socket(
                zmq.ROUTER
            )
            self.resources.output_socket = self.ctx.socket(zmq.PULL)
            if input_address is None:
                in_port = self.input_socket.bind_to_random_port("tcp://127.0.0.1")
                input_address = f"tcp://127.0.0.1:{in_port}"
            else:
                self.input_socket.bind(input_address)
            if output_address is None:
                out_port = self.resources.output_socket.bind_to_random_port(
                    "tcp://127.0.0.1"
                )
                output_address = f"tcp://127.0.0.1:{out_port}"
            else:
                self.resources.output_socket.bind(output_address)

            # SUBTRACTED: launch_core_engines 起子进程（delete 第1/3项）。
            #   本章在同进程起 EngineCoreProc daemon 线程承载 engine 侧 IO+busy loop。
            #   原 vllm/v1/engine/core_client.py:L535-L545。
            self._engine = EngineCoreProc(
                input_address,
                output_address,
                zmq.Context(io_threads=2),
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                tensor_queue=tensor_queue,
            )
            busy = Thread(
                target=self._engine.run_busy_loop, daemon=True, name="EngineCoreBusyLoop"
            )
            busy.start()
            self.resources.engine_thread = busy

            # Serialization setup with tensor queues for multimodal tensor IPC.
            tensor_ipc_sender: TensorIpcSender | None = None
            if tensor_queue is not None:
                tensor_ipc_sender = TensorIpcSender(tensor_queue)

            self.encoder = MsgpackEncoder(oob_tensor_consumer=tensor_ipc_sender)
            self.decoder = MsgpackDecoder(EngineCoreOutputs)

            # ZMQ identity of each engine that this client will talk to.
            # SUBTRACTED: DP 多 rank 的 engine_ranks_managed 列表（delete 第1项）。单 engine rank 0。
            #   原 vllm/v1/engine/core_client.py:L558-L575。
            self.core_engines: list[EngineIdentity] = [(0).to_bytes(2, "little")]

            # Wait for ready messages from each engine on the input socket.
            identities = set(self.core_engines)
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            while identities:
                if not sync_input_socket.poll(timeout=30000):
                    raise TimeoutError(
                        "Timed out waiting for engine core processes to start."
                    )
                identity, payload = sync_input_socket.recv_multipart()
                identities.remove(identity)
                self._apply_ready_response(payload)

            self.core_engine: EngineIdentity = self.core_engines[0]
            self.utility_results: dict[int, AnyFuture] = {}

            # Request objects which may contain pytorch-allocated tensors
            # that we need to keep references to until zmq is done with the
            # underlying data.
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            # SUBTRACTED: start_engine_core_monitor 子进程存活监控线程（线程承载无需）。
            #   原 vllm/v1/engine/core_client.py:L606,L641-L665。

            success = True
        finally:
            if not success:
                self._finalizer()

    # SOURCE: vllm/v1/engine/core_client.py:L613  MPClient.shutdown
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        if self._finalizer.detach() is not None:
            self.resources()

    # SOURCE: vllm/v1/engine/core_client.py:L620  MPClient._format_exception
    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    # SOURCE: vllm/v1/engine/core_client.py:L626  MPClient.ensure_alive
    def ensure_alive(self):
        if self.resources.engine_dead:
            raise EngineDeadError()

    # SOURCE: vllm/v1/engine/core_client.py:L630  MPClient.add_pending_message
    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):
        if not tracker.done:
            self.pending_messages.appendleft((tracker, msg))

    # SOURCE: vllm/v1/engine/core_client.py:L634  MPClient.free_pending_messages
    def free_pending_messages(self):
        while self.pending_messages and self.pending_messages[-1][0].done:
            self.pending_messages.pop()

    # SOURCE: vllm/v1/engine/core_client.py:L667  MPClient._apply_ready_response
    def _apply_ready_response(self, payload: bytes) -> None:
        """Decode an EngineCoreReadyResponse and sync any post-initialization
        config changes (e.g. auto-fitted max_model_len) back to the frontend."""
        if not payload:
            return
        # Decode to demonstrate the ready handshake payload round-trips.
        msgspec.msgpack.decode(payload, type=EngineCoreReadyResponse)
        # SUBTRACTED: 把 max_model_len/num_gpu_blocks 写回 vllm_config 的同步逻辑
        #   （依赖完整 VllmConfig，属配置另章）。原 vllm/v1/engine/core_client.py:L672-L691。


# SOURCE: vllm/v1/engine/core_client.py:L694  _process_utility_output
def _process_utility_output(output: UtilityOutput, utility_results: dict[int, AnyFuture]):
    """Set the result from a utility method in the waiting future."""
    future = utility_results.pop(output.call_id)
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        pass


# SOURCE: vllm/v1/engine/core_client.py:L716  SyncMPClient
class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(self, vllm_config=None, executor_class=None, log_stats=False, **kw):
        # SOURCE: vllm/v1/engine/core_client.py:L720  SyncMPClient.__init__
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            **kw,
        )

        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        out_socket = self.resources.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        resources = self.resources

        def process_outputs_socket():
            # SOURCE: vllm/v1/engine/core_client.py:L745  SyncMPClient.process_outputs_socket
            assert isinstance(out_socket, zmq.Socket)
            try:
                while True:
                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            # SUBTRACTED: shutdown PAIR socket + poller 双注册（delete 第4项关闭细节）。
            #   原 vllm/v1/engine/core_client.py:L745-L773。

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None

    # SOURCE: vllm/v1/engine/core_client.py:L786  SyncMPClient.get_output
    def get_output(self) -> EngineCoreOutputs:
        outputs = self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L798  SyncMPClient._send_input
    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        self.free_pending_messages()
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))

        if len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            self.input_socket.send_multipart(msg, copy=False)
            return

        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
        self.add_pending_message(tracker, request)

    # SOURCE: vllm/v1/engine/core_client.py:L812  SyncMPClient.call_utility
    def call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

        return future.result()

    # SOURCE: vllm/v1/engine/core_client.py:L820  SyncMPClient.get_supported_tasks
    def get_supported_tasks(self) -> tuple[str, ...]:
        return self.call_utility("get_supported_tasks")

    # SOURCE: vllm/v1/engine/core_client.py:L823  SyncMPClient.add_request
    def add_request(self, request: EngineCoreRequest) -> None:
        # SUBTRACTED: is_dp engines_running 置位（DP，delete 第1项）。原 :L824-L825。
        self._send_input(EngineCoreRequestType.ADD, request)

    # SOURCE: vllm/v1/engine/core_client.py:L828  SyncMPClient.abort_requests
    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    # SUBTRACTED: profile / reset_*_cache / add_lora / sleep / collective_rpc 等
    #   call_utility 直通方法（delete 第5项：机制讲清后逐个保留属冗余）。
    #   原 vllm/v1/engine/core_client.py:L832-L884。


# SOURCE: vllm/v1/engine/core_client.py:L887  AsyncMPClient
class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(
        self,
        vllm_config=None,
        executor_class=None,
        log_stats=False,
        client_count: int = 1,
        client_index: int = 0,
        **kw,
    ):
        # SOURCE: vllm/v1/engine/core_client.py:L891  AsyncMPClient.__init__
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            **kw,
        )

        self.client_count = client_count
        self.client_index = client_index
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    # SOURCE: vllm/v1/engine/core_client.py:L921  AsyncMPClient._ensure_output_queue_task
    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_socket = resources.output_socket
        assert output_socket is not None

        async def process_outputs_socket():
            # SOURCE: vllm/v1/engine/core_client.py:L942  AsyncMPClient.process_outputs_socket
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        # SUBTRACTED: EEP_NOTIFICATION_CALL_ID 通知回调分支（delete 第2项）。
                        #   原 vllm/v1/engine/core_client.py:L949-L965。
                        _process_utility_output(outputs.utility_output, utility_results)
                        continue

                    if outputs.outputs:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    # SOURCE: vllm/v1/engine/core_client.py:L990  AsyncMPClient.get_output_async
    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L1001  AsyncMPClient._send_input
    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine, request)

    # SOURCE: vllm/v1/engine/core_client.py:L1013  AsyncMPClient._send_input_message
    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity, objects: Any
    ) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        self.ensure_alive()
        self.free_pending_messages()

        msg = (engine,) + message
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)

        future: "asyncio.Future[zmq.MessageTracker]"
        future = self.input_socket.send_multipart(msg, copy=False, track=True)

        def add_pending(f: "asyncio.Future[zmq.MessageTracker]"):
            # SOURCE: vllm/v1/engine/core_client.py:L1031  AsyncMPClient._send_input_message.add_pending
            with contextlib.suppress(BaseException):
                self.add_pending_message(f.result(), objects)

        future.add_done_callback(add_pending)
        return future

    # SOURCE: vllm/v1/engine/core_client.py:L1038  AsyncMPClient.call_utility_async
    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    # SOURCE: vllm/v1/engine/core_client.py:L1041  AsyncMPClient._call_utility_async
    async def _call_utility_async(self, method: str, *args, engine: EngineIdentity) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (
            EngineCoreRequestType.UTILITY.value,
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        await self._send_input_message(message, engine, args)
        self._ensure_output_queue_task()
        return await future

    # SOURCE: vllm/v1/engine/core_client.py:L1055  AsyncMPClient.get_supported_tasks_async
    async def get_supported_tasks_async(self) -> tuple[str, ...]:
        return await self.call_utility_async("get_supported_tasks")

    # SOURCE: vllm/v1/engine/core_client.py:L1058  AsyncMPClient.add_request_async
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1063  AsyncMPClient.abort_requests_async
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    # SUBTRACTED: pause/resume/profile/reset/sleep/add_lora/collective_rpc 等 *_async
    #   call_utility_async 直通方法（delete 第5项）。原 vllm/v1/engine/core_client.py:L1067-L1134。
