# SPDX-License-Identifier: Apache-2.0
# 只做减法的忠实子集 —— 对应 vllm/v1/engine/core.py 的 EngineCore / EngineCoreProc。
# 源码 pin: f3fef123。本章主线 = engine 侧 ZMQ IPC 边界：
#   两个 IO 线程（process_input_sockets / process_output_sockets）+ 两个内部 queue.Queue
#   + run_busy_loop 消费 input_queue / 生产 output_queue + _handle_client_request 字节标签分派。
#
# EngineCore 的 scheduler / executor / KV cache / batch_queue / DP 等内部，按 dossier scope_note
# 不是本章主线（属另章），在此 SUBTRACTED 为最小可注入子集，使 IPC 闭环可跑、可数值追踪，
# 且不杜撰任何"伪 forward"：step 的调度→执行→产出三段结构保留，但调度器/执行器行为由
# 测试注入（与真实代码 self.step_fn = self.step 同一接缝）。

import queue
import threading
from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from inspect import isclass, signature
from typing import Any

import msgspec
import zmq

from engine_init import (
    EngineCoreOutputs,
    EngineCoreReadyResponse,
    EngineCoreRequest,
    EngineCoreRequestType,
    UtilityOutput,
    UtilityResult,
)
from serial_utils import MsgpackDecoder, MsgpackEncoder
from tensor_ipc import TensorIpcReceiver


# SOURCE: vllm/utils/network_utils.py  make_zmq_socket (minimal subset)
def make_zmq_socket(ctx, addr, socket_type, *, identity=None, bind=False, linger=None):
    # SUBTRACTED: 真实 vllm.utils.network_utils.make_zmq_socket 含 buffer/HWM/probe 等调优；
    #   这里保留本章用到的 identity / bind / linger 三个语义。原 vllm/utils/network_utils.py。
    sock = ctx.socket(socket_type)
    if identity is not None:
        sock.setsockopt(zmq.IDENTITY, identity)
    if linger is not None:
        sock.setsockopt(zmq.LINGER, linger)
    if bind:
        sock.bind(addr)
    else:
        sock.connect(addr)
    return sock


# SOURCE: vllm/v1/engine/core.py:L92  EngineCore
class EngineCore:
    """Inner loop of vLLM's Engine.

    本章只保留 IPC 边界依赖的最小接缝：add_request / abort_requests / has_work /
    is_running / step / step_fn / post_step / get_supported_tasks / preprocess_add_request。
    """

    # SOURCE: vllm/v1/engine/core.py:L94  EngineCore.__init__
    def __init__(
        self,
        vllm_config: Any = None,
        executor_class: Any = None,
        log_stats: bool = False,
        executor_fail_callback: Callable | None = None,
        *args,
        **kwargs,
    ):
        self.vllm_config = vllm_config
        # SUBTRACTED: model_executor / scheduler / KV cache / structured output /
        #   batch_queue / spec decode / DP 初始化（数百行，非本章主线）。
        #   原 vllm/v1/engine/core.py:L100-L224。最小请求登记取代 scheduler 等价语义。
        self._requests: dict[str, Request] = {}
        # step_fn 接缝：真实代码 self.step_fn = self.step（或 batch_queue 变体）。
        # 测试可替换 step_fn 注入调度/执行行为，与真实接缝一致。原 :L213。
        self.step_fn = self.step
        self.aborts_queue = queue.Queue[list[str]]()
        self._idle_state_callbacks: list[Callable] = []

    # SOURCE: vllm/v1/engine/core.py:L312  EngineCore.get_supported_tasks
    def get_supported_tasks(self) -> tuple[str, ...]:
        # SUBTRACTED: 真实从 self.model_executor.supported_tasks 取。固定示例值。原 :L313。
        return ("generate",)

    # SOURCE: vllm/v1/engine/core.py:L765  EngineCore.preprocess_add_request
    def preprocess_add_request(
        self, request: EngineCoreRequest
    ) -> tuple["Request", int]:
        # SUBTRACTED: mm hashing / block hasher / Request.from_engine_core_request
        #   的完整字段映射（属输入处理另章）。保留 (Request, request_wave) 二元组返回结构。
        #   原 vllm/v1/engine/core.py:L765-L798。
        req = Request(request.request_id, request.client_index)
        return req, 0

    # SOURCE: vllm/v1/engine/core.py:L315  EngineCore.add_request
    def add_request(self, request: "Request", request_wave: int = 0):
        """Add request to the scheduler."""
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )
        # SUBTRACTED: pooling task 校验 / kv_transfer_params 警告（语义旁支）。原 :L327-L344。
        # self.scheduler.add_request(request) -> 最小登记。
        self._requests[request.request_id] = request

    # SOURCE: vllm/v1/engine/core.py:L348  EngineCore.abort_requests
    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""
        # self.scheduler.finish_requests(request_ids, FINISHED_ABORTED) -> 最小登记。
        for rid in request_ids:
            self._requests.pop(rid, None)

    # SOURCE: vllm/v1/engine/core.py:L402  EngineCore.step
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """
        # SUBTRACTED: scheduler.schedule() / model_executor.execute_model() /
        #   update_from_output() 三段实体（属调度/执行另章，且不可杜撰伪 forward）。
        #   保留"无请求即空返回"的真实短路；有请求时返回空步进，由注入的 step_fn 决定产出。
        #   原 vllm/v1/engine/core.py:L409-L431。
        if not self._requests:
            return {}, False
        return {}, False

    # SOURCE: vllm/v1/engine/core.py:L433  EngineCore.post_step
    def post_step(self, model_executed: bool) -> None:
        # SUBTRACTED: spec decode draft token 更新（非本章主线）。原 :L437-L439。
        pass

    # SOURCE: vllm/v1/engine/core.py:L1152  EngineCore.has_work
    def has_work(self) -> bool:
        # SUBTRACTED: 真实判 scheduler.has_requests() or batch_queue / DP engines_running。
        #   原 vllm/v1/engine/core.py:L1152-L1158。
        return bool(self._requests)

    # SOURCE: vllm/v1/engine/core.py:L1160  EngineCore.is_running
    def is_running(self) -> bool:
        """Returns true if shutdown has not been requested."""
        return self.shutdown_state == EngineShutdownState.RUNNING

    # SOURCE: vllm/v1/engine/core.py:L1164  EngineCore.run_busy_loop
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

        raise SystemExit

    # SOURCE: vllm/v1/engine/core.py:L1174  EngineCore._process_input_queue
    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    # SOURCE: vllm/v1/engine/core.py:L1205  EngineCore._process_engine_step
    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)
        # SUBTRACTED: WAITING_FOR_REMOTE_KVS 让步 sleep（NIXL 旁支）。原 :L1216-L1221。
        return model_executed

    # SOURCE: vllm/v1/engine/core.py:L1225  EngineCore._notify_idle_state_callbacks
    def _notify_idle_state_callbacks(self) -> None:
        while self._idle_state_callbacks:
            callback = self._idle_state_callbacks.pop()
            callback(self)

    # SOURCE: vllm/v1/engine/core.py:L1230  EngineCore._handle_shutdown
    def _handle_shutdown(self) -> bool:
        # SUBTRACTED: EngineShutdownState.REQUESTED 的 draining/timeout 排空状态机
        #   （delete 第4项批准）。保留 RUNNING 直通 + 哨兵驱动的硬退出。
        #   原 vllm/v1/engine/core.py:L1232-L1264。
        return self.shutdown_state == EngineShutdownState.RUNNING

    # SOURCE: vllm/v1/engine/core.py:L1266  EngineCore._handle_client_request
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
            # SUBTRACTED: _reject_add_in_shutdown 拒绝分支（排空状态机，delete 第4项）。原 :L1275。
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            # SUBTRACTED: _reject_utility_in_shutdown（排空状态机，delete 第4项）。原 :L1282。
            output = UtilityOutput(call_id)
            # Lazily look-up utility method so that failure will be handled/returned.
            get_result = lambda: (
                (method := getattr(self, method_name))
                and method(*self._convert_msgspec_args(method, args))
            )
            enqueue_output = lambda out: self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=out))
            )
            self._invoke_utility_method(method_name, get_result, output, enqueue_output)
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            pass

    @staticmethod
    def _invoke_utility_method(
        name: str, get_result: Callable, output: UtilityOutput, enqueue_output: Callable
    ):
        # SOURCE: vllm/v1/engine/core.py:L1322  EngineCore._invoke_utility_method
        try:
            result = get_result()
            # SUBTRACTED: isinstance(result, Future) 的延迟回调分支（async utility）。原 :L1328-L1334。
            output.result = UtilityResult(result)
        except Exception as e:
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        # SOURCE: vllm/v1/engine/core.py:L1341  EngineCore._convert_msgspec_args
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )


# SUBTRACTED: Request 是 vllm/v1/request.py 的完整请求实体（属调度另章）；
#   本章只需 request_id / client_index 两个字段驱动 IPC 登记。原 vllm/v1/request.py。
class Request:
    # SOURCE: vllm/v1/request.py  Request (minimal subset)
    def __init__(self, request_id: str, client_index: int = 0):
        self.request_id = request_id
        self.client_index = client_index


# SOURCE: vllm/v1/engine/core.py:L800  EngineShutdownState
class EngineShutdownState:
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


# SOURCE: vllm/v1/engine/core.py:L806  EngineCoreProc
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    # SOURCE: vllm/v1/engine/core.py:L812  EngineCoreProc.__init__
    def __init__(
        self,
        input_address: str,
        output_address: str,
        ctx: zmq.Context,
        vllm_config: Any = None,
        executor_class: Any = None,
        log_stats: bool = False,
        tensor_queue: Any = None,
        *,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        # SUBTRACTED: executor_fail_callback 经 EXECUTOR_FAILED 投 input_queue（保留概念，
        #   不接真实 executor）。原 vllm/v1/engine/core.py:L827-L829。

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.shutdown_state = EngineShutdownState.RUNNING

        # Receiver for tensor IPC
        self.tensor_ipc_receiver: TensorIpcReceiver | None = None
        if tensor_queue is not None:
            self.tensor_ipc_receiver = TensorIpcReceiver(tensor_queue)

        # SUBTRACTED: _perform_handshakes 的 DP 双握手 / coordinator socket（delete 第3项）。
        #   原 vllm/v1/engine/core.py:L842-L848,L921+。本章单 engine 单次握手在 IO 线程内完成。
        super().__init__(vllm_config, executor_class, log_stats)
        self.process_input_queue_block = True
        self._ctx = ctx

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        ready_event = threading.Event()
        input_thread = threading.Thread(
            target=self.process_input_sockets,
            args=([input_address], None, identity, ready_event),
            daemon=True,
        )
        input_thread.start()

        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=([output_address], None, self.engine_index),
            daemon=True,
        )
        self.output_thread.start()

        # Don't complete construction until the input thread has sent ready.
        while not ready_event.wait(timeout=10):
            if not input_thread.is_alive():
                raise RuntimeError("Input socket thread died during startup")

    # SOURCE: vllm/v1/engine/core.py:L1358  EngineCoreProc._send_engine_dead
    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""
        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)
        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)

    # SOURCE: vllm/v1/engine/core.py:L1372  EngineCoreProc.process_input_sockets
    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding with optional tensor IPC receiver.
        add_request_decoder = MsgpackDecoder(
            EngineCoreRequest, oob_tensor_provider=self.tensor_ipc_receiver
        )
        generic_decoder = MsgpackDecoder(oob_tensor_provider=self.tensor_ipc_receiver)

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            # SUBTRACTED: coordinator XSUB socket + START_DP_WAVE / READY 订阅（delete 第3项）。
            #   原 vllm/v1/engine/core.py:L1396-L1409,L1426-L1429,L1437-L1441。

            # Register sockets with poller.
            poller = zmq.Poller()
            ready_response = EngineCoreReadyResponse(
                max_model_len=getattr(
                    getattr(self.vllm_config, "model_config", None), "max_model_len", 0
                )
                if self.vllm_config
                else 0,
                num_gpu_blocks=0,
                dp_stats_address=None,
            )
            ready_payload = msgspec.msgpack.encode(ready_response)
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(ready_payload)
                poller.register(input_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(req)
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    # SOURCE: vllm/v1/engine/core.py:L1466  EngineCoreProc.process_output_sockets
    def process_output_sockets(
        self, output_paths: list[str], coord_output_path: str | None, engine_index: int
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            # SUBTRACTED: coordinator PUSH socket（client_index==-1 路径，delete 第3项）。
            #   原 vllm/v1/engine/core.py:L1489-L1497,L1510-L1515。
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)
