"""Stage2 的进程边界客户端 —— in-process stub（伏笔 f2，ch07 回收）。

真实 vLLM 用 EngineCoreClient.make_async_mp_client(...) 启动**独立进程**的 EngineCore，经
ZMQ/msgpack 跨进程通信，对外暴露 add_request_async / get_output_async / abort_requests_async
三个 async 方法（core_client.py:L1058 / L990 / L1063）。

本章焦点是 AsyncLLM 这层 facade 如何编排三段，**不是** IPC 物理机制。按 dossier 的 subtraction_plan，
这里用 **in-process stub 替代真实 IPC**：持一个 asyncio.Queue 收 EngineCoreRequest，一个背景协程
模拟「产 token」并把 EngineCoreOutput 放入另一队列；对外**仍暴露同名的三个 async 方法**。

这样 AsyncLLM 的 __init__/add_request/_add_request/generate/_run_output_handler 几乎一字不改 ——
三段式结构与生产者-消费者关系原样可见，只是 Stage2 的「独立进程边界」被替换成「同进程队列」。
真实的 ZMQ/msgpack/进程编排留 ch07。
"""

# SUBTRACTED: multiproc/ZMQ/msgpack 的真实 IPC 实现 —— _send_input/MessageTracker/
#   process_outputs_socket/握手、EngineCoreProc 的 socket 线程、独立进程启动。
#   用 in-process stub 替代后，AsyncLLM 看到的接口语义不变；IPC 物理机制是 ch07 的主题。
#   原 vllm/v1/engine/core_client.py:L887(AsyncMPClient) / L942-L1036 / EngineCoreProc。
# SUBTRACTED: DP/数据并行协调（client_addresses/client_count/client_index、DPCoordinator、
#   current_wave/START_DP_WAVE）—— 留 ch21；单引擎下全是默认值，删去不影响主路径。
#   原 vllm/v1/engine/core_client.py:make_async_mp_client 入参。

from __future__ import annotations

import asyncio

from messages import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest


class InProcessEngineCore:
    """In-process 替身：站位真实「独立进程 EngineCore + IPC 客户端」(Stage2)。

    对外接口与 vllm AsyncMPClient 同名同签名（add_request_async / get_output_async /
    abort_requests_async）；内部用两条队列 + 一个背景协程模拟引擎产 token。真实调度+执行
    (continuous batching, ch03) 与进程/ZMQ 机制 (ch07) 都在此替身之外。
    """

    # SOURCE: vllm/v1/engine/core_client.py:L887 (AsyncMPClient) / L107 (make_async_mp_client)
    def __init__(self, tokens_per_request: int = 3):  # SOURCE: vllm/v1/engine/core_client.py:L887
        # 真实版这里是「跨进程的 input_socket / outputs_queue(由背景 socket 任务填充)」。
        # stub 用两条 in-process asyncio.Queue 顶替进程边界两侧。
        self._inputs: asyncio.Queue[EngineCoreRequest] = asyncio.Queue()
        self._outputs: asyncio.Queue[EngineCoreOutputs] = asyncio.Queue()
        self._tokens_per_request = tokens_per_request
        self._aborted: set[str] = set()
        self._engine_loop: asyncio.Task | None = None

    def _ensure_engine_loop(self) -> None:  # SOURCE: vllm/v1/engine/core_client.py:L942 (_ensure_output_queue_task 的 stub 对应)
        # 真实版对应 _ensure_output_queue_task（启动消费 ZMQ output_socket 的背景任务）。
        # 原 vllm/v1/engine/core_client.py:L942。stub 改为启动「模拟引擎」背景协程。
        if self._engine_loop is None:
            self._engine_loop = asyncio.create_task(self._engine_step_loop())

    async def _engine_step_loop(self) -> None:  # SOURCE: vllm/v1/engine/core.py:EngineCoreProc.run_busy_loop (schedule+execute, ch07) 的 in-process stub
        """模拟引擎主循环：取请求 -> 每步逐 token 产出 EngineCoreOutput。

        这站位真实 EngineCore 的 schedule()+execute_model()（ch03/ch07）。stub 行为：对每个收到的
        请求，分多步发出 new_token_ids，最后一步置 finish_reason -> finished。
        """
        inflight: list[tuple[EngineCoreRequest, int]] = []
        while True:
            # 取走当前所有已投递的新请求（非阻塞）。
            while not self._inputs.empty():
                req = self._inputs.get_nowait()
                inflight.append((req, 0))
            if not inflight:
                # 没有在跑的请求：阻塞等下一个，避免空转。
                req = await self._inputs.get()
                inflight.append((req, 0))

            batch: list[EngineCoreOutput] = []
            still: list[tuple[EngineCoreRequest, int]] = []
            for req, step in inflight:
                if req.request_id in self._aborted:
                    continue  # 已 abort，停止产出。
                next_step = step + 1
                finished = next_step >= self._tokens_per_request
                batch.append(
                    EngineCoreOutput(
                        request_id=req.request_id,
                        new_token_ids=[1000 + step],
                        finish_reason="stop" if finished else None,
                    )
                )
                if not finished:
                    still.append((req, next_step))
            inflight = still
            if batch:
                await self._outputs.put(EngineCoreOutputs(outputs=batch))
            # 让出事件循环，模拟一次引擎 step 的边界。
            await asyncio.sleep(0)

    # SOURCE: vllm/v1/engine/core_client.py:L1058 (add_request_async)
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # 真实版：request.client_index = ...; await self._send_input(ADD, request)（ZMQ 投递到
        #   独立进程）+ self._ensure_output_queue_task()。原 core_client.py:L1058-L1061。
        # stub：把请求放进 in-process 输入队列，并确保引擎背景协程已启动。
        self._ensure_engine_loop()
        await self._inputs.put(request)

    # SOURCE: vllm/v1/engine/core_client.py:L990 (get_output_async)
    async def get_output_async(self) -> EngineCoreOutputs:
        # 真实版：self._ensure_output_queue_task(); outputs = await self.outputs_queue.get()
        #   —— outputs_queue 由背景 process_outputs_socket 任务消费 ZMQ output_socket 填充。
        #   原 core_client.py:L990-L999。stub：直接 await 本进程输出队列。
        self._ensure_engine_loop()
        outputs = await self._outputs.get()
        if isinstance(outputs, Exception):
            raise outputs
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L1063 (abort_requests_async)
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # 真实版：经 ZMQ 发 ABORT 控制消息到独立进程 EngineCore。原 core_client.py:L1063。
        # stub：标记 req_id，引擎背景协程下一步停止为其产出。
        for req_id in request_ids:
            self._aborted.add(req_id)
