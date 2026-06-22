# 只做减法的忠实精简版 —— 镜像 vllm/v1/engine/core_client.py（pin f3fef123）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# 本文件保留两端客户端：
#   - InprocClient：进程内对照——get_output 直接调 step_fn()，无忙循环、无 ZMQ。
#   - AsyncMPClient（子集）：output_queue 的另一端——process_outputs_socket 把
#     EngineCoreOutputs 搬进 asyncio outputs_queue，get_output_async 取出；
#     add_request_async 经 _send_input 把 ADD 发进子进程 input_queue。
#     （连回 ch04 AsyncLLM 三段式）
#
# SUBTRACTED: SPDX 头、init_logger，以及与本章无关的客户端方法/资源管理。
import asyncio
from typing import Any

# SUBTRACTED: 从 vllm.* 包导入改为从本精简版同目录模块导入（仅 import 路径调整）。
from interfaces import (
    EngineCoreOutputs,
    EngineCoreRequestType,
    PauseMode,
)
from core import EngineCore


class InprocClient:
    # SOURCE: vllm/v1/engine/core_client.py:L274-L282
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # SOURCE: vllm/v1/engine/core_client.py:L284-L285
    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    # SOURCE: vllm/v1/engine/core_client.py:L287-L290
    def get_output(self) -> EngineCoreOutputs:
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    # SOURCE: vllm/v1/engine/core_client.py:L292-L293
    def get_supported_tasks(self) -> tuple:
        return self.engine_core.get_supported_tasks()

    # SOURCE: vllm/v1/engine/core_client.py:L295-L297
    def add_request(self, request: Any) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    # SOURCE: vllm/v1/engine/core_client.py:L299-L301
    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    # SOURCE: vllm/v1/engine/core_client.py:L303-L304
    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()

    # SOURCE: vllm/v1/engine/core_client.py:L322-L326
    def sleep(self, level: int = 1, mode: "PauseMode" = "abort") -> None:
        if mode == "wait":
            raise ValueError("'wait' pause mode is not supported in inproc-engine mode")
        result = self.engine_core.sleep(level, mode)
        assert result is None

    # SOURCE: vllm/v1/engine/core_client.py:L328-L329
    def wake_up(self, tags: list[str] | None = None) -> None:
        self.engine_core.wake_up(tags)

    # SOURCE: vllm/v1/engine/core_client.py:L331-L332
    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    # SUBTRACTED: profile / reset_mm_cache / reset_prefix_cache / reset_encoder_cache /
    #             execute_dummy_batch / add_lora / remove_lora / list_loras / pin_lora /
    #             save_sharded_state / collective_rpc / dp_engines_running
    #             （vllm/v1/engine/core_client.py:L306-L321, L334-L364）—— 管理 API
    #             转发，与"无忙循环时 step 如何被驱动"主线无关。


class AsyncMPClient:
    # SOURCE: vllm/v1/engine/core_client.py:L887（类定义；这里只保留本章相关子集）
    """异步多进程客户端（子集）：经 ZMQ 把 ADD 发进子进程 input_queue，并以后台 task
    把子进程 output_queue 的 EngineCoreOutputs 搬进 asyncio outputs_queue 供
    get_output_async 消费。是 ch04 AsyncLLM 三段式与本章 step() 之间的桥。

    SUBTRACTED: 真实 __init__（建 DEALER/PULL socket、BackgroundResources、encoder、
                utility_results、client_index 等）——这里以最小注入构造，聚焦
                process_outputs_socket / get_output_async / add_request_async 三个方法。
    """

    # SOURCE: vllm/v1/engine/core_client.py（构造的本章相关字段子集）
    def __init__(self, input_socket: Any, output_socket: Any, encoder: Any):
        self.input_socket = input_socket
        self.output_socket = output_socket
        self.encoder = encoder
        self.client_index = 0
        self.outputs_queue: asyncio.Queue | None = None
        self._output_queue_task: asyncio.Task | None = None

    def _ensure_output_queue_task(self):
        # SOURCE: vllm/v1/engine/core_client.py（_ensure_output_queue_task 启动 task）
        if self._output_queue_task is None:
            self.outputs_queue = asyncio.Queue()
            self._add_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L942-L988
    def _add_output_queue_task(self):
        output_socket = self.output_socket
        decoder = self.encoder  # 精简版用同一对象 decode（真实为独立 MsgpackDecoder）
        outputs_queue = self.outputs_queue

        async def process_outputs_socket():
            # SOURCE: vllm/v1/engine/core_client.py:L942-L984
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    # SUBTRACTED: utility_output 分派（utility_results future）与 EEP
                    #             notification 回调、output_handler（vllm/v1/engine/core_client.py:L948-L977）
                    #             —— utility 结果走 utility_results future，本章只关心正常
                    #             token 输出走 outputs_queue。
                    if outputs.outputs or getattr(outputs, "scheduler_stats", None):
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(RuntimeError("EngineDeadError"))

        self._output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    # SOURCE: vllm/v1/engine/core_client.py:L990-L999
    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise outputs from None
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L1001-L1011
    def _send_input(
        self, request_type: "EngineCoreRequestType", request: Any
    ):
        # SUBTRACTED: engine identity 选择 + _send_input_message 的 tensor backing
        #             buffer 保活（track=True/add_pending）（vllm/v1/engine/core_client.py:L1004-L1036）
        #             —— 多帧零拷贝保活属 ZMQ 细节；保留"打包 (type, request) 发 input_socket"。
        message = (request_type.value, *self.encoder.encode(request))
        return self.input_socket.send_multipart(message, copy=False)

    # SOURCE: vllm/v1/engine/core_client.py:L1058-L1061
    async def add_request_async(self, request: Any) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1063-L1065
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)
