"""EngineCoreClient 工厂与两个同步实现（精简版，只做减法）。

与真实 vllm/v1/engine/core_client.py 同名、同结构、同控制流：

  * make_client          —— 三分支工厂，按 (multiprocess_mode, asyncio_mode) 选实现。
  * SyncMPClient         —— 【离线默认】后台进程 EngineCore + 后台 daemon 线程喂 outputs_queue +
                            get_output() 阻塞 outputs_queue.get()。
  * InprocClient         —— 回退（VLLM_ENABLE_V1_MULTIPROCESSING=0）：真·进程内、无 ZMQ，
                            get_output() 直接 step_fn()+post_step()。

【关键澄清】离线默认命中 (multiprocess_mode=True, asyncio_mode=False) → SyncMPClient，
**不是** InprocClient。把所有 # SUBTRACTED 分支删回去 ≈ 真实 core_client.py 的这三段。
"""

from __future__ import annotations

import queue
from threading import Thread

from engine_core_stub import StubEngineCoreProc, _StubEngineCore
from messages import EngineCoreOutputs, EngineCoreRequest


# SOURCE: vllm/v1/engine/core_client.py:L69 (class EngineCoreClient)
class EngineCoreClient:
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
        vllm_config=None,
        executor_class=None,
        log_stats: bool = False,
    ) -> "EngineCoreClient":
        # SOURCE: vllm/v1/engine/core_client.py:L80 (make_client)
        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        # SUBTRACTED: (multiprocess_mode and asyncio_mode) 分支 → make_async_mp_client →
        #   AsyncMPClient/DPAsyncMPClient/DPLBAsyncMPClient（ch04 异步侧 + DP）。本章离线侧不命中
        #   asyncio_mode=True。原 vllm/v1/engine/core_client.py:L95-L98, L105-L130。
        if multiprocess_mode and asyncio_mode:
            raise NotImplementedError("async path is ch04's subject")

        # 离线默认命中这里：(mp=True, async=False) → SyncMPClient（后台进程 + ZMQ + 阻塞队列）。
        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        # 仅 VLLM_ENABLE_V1_MULTIPROCESSING=0 回退到这里 → InprocClient（真·进程内、无 ZMQ）。
        return InprocClient(vllm_config, executor_class, log_stats)


# SOURCE: vllm/v1/engine/core_client.py:L716 (class SyncMPClient(MPClient))
class SyncMPClient(EngineCoreClient):
    """Synchronous client for multi-proc EngineCore."""
    # SUBTRACTED: 真实继承 MPClient（持 ZMQ ctx/input_socket/output_socket、序列化 encoder/decoder、
    #   启动独立 EngineCore 进程、存活探测 resources/ensure_alive）。本章用 in-process StubEngineCoreProc
    #   顶替进程边界与 ZMQ；SyncMPClient 自身的'后台线程喂 outputs_queue + 阻塞取'结构原样保留。
    #   原 vllm/v1/engine/core_client.py:MPClient(__init__ 启进程/建 socket)。

    # SOURCE: vllm/v1/engine/core_client.py:L719 (SyncMPClient.__init__)
    def __init__(self, vllm_config=None, executor_class=None, log_stats: bool = False):
        # SUBTRACTED: super().__init__(asyncio_mode=False, ...) 启独立 EngineCore 进程 + 建 ZMQ
        #   input/output socket + msgpack encoder/decoder + 存活探测。原 core_client.py:L723-L728。
        #   stub 改为持一个 in-process StubEngineCoreProc，其 output_socket 是一条 queue.Queue。
        self._engine = StubEngineCoreProc()
        out_socket = self._engine.output_socket  # 站位真实 ZMQ output_socket

        # SOURCE: vllm/v1/engine/core_client.py:L731 (self.outputs_queue = queue.Queue[...]())
        # 主线程与后台收数线程之间的【阻塞队列】：后台线程喂、主线程 get_output() 阻塞取。
        self.outputs_queue: queue.Queue = queue.Queue()
        outputs_queue = self.outputs_queue
        # SUBTRACTED: self.is_dp / decoder / utility_results / shutdown_path 等绑定（DP + ZMQ 收尾），
        #   单引擎 in-process 下无需。原 core_client.py:L730, L735-L743。

        # SOURCE: vllm/v1/engine/core_client.py:L745 (def process_outputs_socket)
        def process_outputs_socket():
            # 真实版：poller 监听 ZMQ output_socket + shutdown_socket；recv_multipart 收帧、
            #   decoder.decode 反序列化 EngineCoreOutputs，再 outputs_queue.put_nowait(outputs)。
            #   原 core_client.py:L745-L773。
            # SUBTRACTED: zmq.Poller/recv_multipart/decoder.decode/validate_alive/utility_output 分流/
            #   shutdown_socket 退出与 socket 关闭。stub 改为从 in-process out_socket(queue) 收
            #   已是 EngineCoreOutputs 的对象，其余'后台线程把输出喂进 outputs_queue'语义不变。
            while True:
                outputs: EngineCoreOutputs = out_socket.get()
                outputs_queue.put_nowait(outputs)

        # SOURCE: vllm/v1/engine/core_client.py:L776 (Thread(target=process_outputs_socket, daemon=True))
        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()
        # SUBTRACTED: self.resources.output_socket = None（把 socket 关闭责任移交线程）。
        #   原 core_client.py:L783-L784。in-process 无 socket 生命周期问题。

    # SOURCE: vllm/v1/engine/core_client.py:L786 (SyncMPClient.get_output)
    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()  # 【同步阻塞】线程安全队列取数

        if isinstance(outputs, Exception):
            raise outputs from None
        # SUBTRACTED: outputs.wave_complete -> self.engines_running = False（DP wave 状态）。
        #   原 core_client.py:L794-L795。单引擎非 DP 不涉及。
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L823 (SyncMPClient.add_request)
    def add_request(self, request: EngineCoreRequest) -> None:
        # SUBTRACTED: if self.is_dp: self.engines_running = True（DP）。原 core_client.py:L824-L825。
        # SUBTRACTED: _send_input(EngineCoreRequestType.ADD, request) 的 ZMQ 多帧/pending message
        #   字节标签协议（ch05/ch06 已细讲）。原 core_client.py:L826 + L798-L810。
        #   stub：经 in-process StubEngineCoreProc 把 ADD 送往（替身的）后台 EngineCore，
        #   其产出会经 output_socket 流回后台线程→outputs_queue。
        self._engine.add_request(request)

    # SOURCE: vllm/v1/engine/core_client.py:L828 (SyncMPClient.abort_requests)
    def abort_requests(self, request_ids: list[str]) -> None:
        # SUBTRACTED: if request_ids and not engine_dead: _send_input(ABORT, ...)（ZMQ）。
        #   原 core_client.py:L828-L830。stub 直接转发给替身后台 EngineCore。
        if request_ids:
            self._engine.abort_requests(request_ids)

    # SUBTRACTED: get_supported_tasks/call_utility/profile/reset_*cache/add_lora/remove_lora 等
    #   运维转发方法（均经 call_utility→ZMQ），与本章 add_request+get_output 主干无关。
    #   原 vllm/v1/engine/core_client.py:L812-L853+。


# SOURCE: vllm/v1/engine/core_client.py:L274 (class InprocClient(EngineCoreClient))
class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # SOURCE: vllm/v1/engine/core_client.py:L284 (InprocClient.__init__)
    def __init__(self, *args, **kwargs):
        # 真实版：self.engine_core = EngineCore(*args, **kwargs)（进程内直接 new，无 ZMQ/无后台进程）。
        # stub 用 _StubEngineCore 顶替真实 EngineCore（schedule+execute 留 ch07）。
        self.engine_core = _StubEngineCore()

    # SOURCE: vllm/v1/engine/core_client.py:L287 (InprocClient.get_output)
    def get_output(self) -> EngineCoreOutputs:
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs and outputs.get(0) or EngineCoreOutputs()

    # SOURCE: vllm/v1/engine/core_client.py:L295 (InprocClient.add_request)
    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    # SOURCE: vllm/v1/engine/core_client.py:L299 (InprocClient.abort_requests)
    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    # SUBTRACTED: shutdown/profile/reset_*cache/sleep/wake_up/execute_dummy_batch/add_lora 等
    #   转发方法（core_client.py:L303-L333+）。InprocClient 在本章只作'回退对照'，
    #   保留 __init__/get_output/add_request/abort_requests 已足以对照 SyncMPClient（无 ZMQ、直接 step_fn）。
