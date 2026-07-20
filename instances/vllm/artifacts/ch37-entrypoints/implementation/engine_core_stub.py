"""EngineCore 替身（in-process stub）——站位真实「独立后台进程 EngineCore」。

真实 vLLM 的 SyncMPClient 背后是一个跑在**独立后台进程**的 EngineCore，主进程经 ZMQ/msgpack
跨进程通信（字节标签协议 ch05/ch06 已细讲）。本章焦点是 **LLM facade + LLMEngine 的同步 step()
驱动**，不是 IPC 物理机制。按 dossier 的 subtraction_plan：

  * SyncMPClient 保留真实形状——后台 daemon 线程从一个'输出源'收 EngineCoreOutputs 喂进
    outputs_queue，主线程 get_output() 阻塞 outputs_queue.get()。我们只把'输出源'从 ZMQ
    output_socket 换成一条 in-process queue.Queue（由本 stub 的模拟引擎线程填充），其余原样。
  * InprocClient 保留真实形状——直接持有 EngineCore，get_output() = step_fn()，无 ZMQ/无后台进程。

这样 SyncMPClient '后台线程喂队列 + 主线程阻塞取' 与 InprocClient '直接 step_fn()' 的对照
原样可见，区别仅在主进程侧驱动方式（与 ch04 异步对照不依赖 in-process）。真实 ZMQ/进程编排留 ch05/ch07。
"""
# SUBTRACTED: 真实 EngineCore 的 schedule()+execute_model()（continuous batching, ch07）、
#   EngineCoreProc 的独立进程启动/busy loop、ZMQ socket 线程、msgpack 序列化。
#   用 in-process 模拟引擎顶替后，两个 client 看到的接口语义不变。
#   原 vllm/v1/engine/core.py:EngineCore / EngineCoreProc。

from __future__ import annotations

import queue
from threading import Thread

from messages import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
)


class _StubEngineCore:
    """模拟引擎：站位真实 EngineCore.schedule()+execute_model()（ch07）。

    对每个收到的请求分多步逐 token 产出，最后一步置 finished。供 InprocClient 直接 step_fn()，
    供 SyncMPClient 的'输出源'线程消费。真实调度/执行与进程/ZMQ 机制都在此替身之外。
    """

    # SOURCE: vllm/v1/engine/core.py:EngineCore (in-process stub)
    def __init__(self) -> None:
        self._inflight: list[tuple[EngineCoreRequest, int]] = []
        self._aborted: set[str] = set()

    def preprocess_add_request(self, request: EngineCoreRequest):
        # SOURCE: vllm/v1/engine/core.py:EngineCore.preprocess_add_request (in-process stub)
        # 真实版做 wave 编号/输入二次加工；stub 直接透传（request_wave 占位 0）。
        return request, 0

    def add_request(self, request: EngineCoreRequest, request_wave: int = 0) -> None:
        # SOURCE: vllm/v1/engine/core.py:EngineCore.add_request (in-process stub)
        self._inflight.append((request, 0))

    def abort_requests(self, request_ids: list[str]) -> None:
        # SOURCE: vllm/v1/engine/core.py:EngineCore.abort_requests (in-process stub)
        for rid in request_ids:
            self._aborted.add(rid)

    def step_fn(self):
        # SOURCE: vllm/v1/engine/core.py:EngineCore.step_fn (in-process stub for schedule+execute, ch07)
        # 真实一拍 = schedule()+execute_model()，本 stub 给每个在跑请求产 1 个 token，
        # 满 max_tokens 即 finished。返回 ({0: EngineCoreOutputs}, model_executed)。
        batch: list[EngineCoreOutput] = []
        still: list[tuple[EngineCoreRequest, int]] = []
        for req, step in self._inflight:
            if req.request_id in self._aborted:
                continue
            next_step = step + 1
            max_tokens = getattr(req.params, "max_tokens", 1)
            finished = next_step >= max_tokens
            batch.append(EngineCoreOutput(request_id=req.request_id,
                                          new_token_ids=[1000 + step],
                                          finished=finished))
            if not finished:
                still.append((req, next_step))
        self._inflight = still
        model_executed = bool(batch)
        return {0: EngineCoreOutputs(outputs=batch)}, model_executed

    def post_step(self, model_executed: bool) -> None:
        # SOURCE: vllm/v1/engine/core.py:EngineCore.post_step (in-process stub)
        # 真实版做 DP wave 推进等收尾；非 DP 单引擎下空实现即可。
        return None

    def has_unfinished(self) -> bool:
        # SOURCE: vllm/v1/engine/core.py:EngineCore (调度状态) — in-process stub
        # stub 辅助：是否还有在跑请求（真实由 EngineCore 内部调度状态决定）。
        return bool(self._inflight)


class StubEngineCoreProc:
    """SyncMPClient 输出源的 in-process 替身：站位'独立后台进程 EngineCore + ZMQ output_socket'。

    真实里 EngineCore 在独立进程跑 busy loop，把 EngineCoreOutputs 经 ZMQ output_socket 推给主进程；
    主进程的后台 daemon 线程从 socket 收。本 stub 用一条 queue.Queue 当'output_socket'：本类在
    background 推进模拟引擎并把 EngineCoreOutputs put 进去，SyncMPClient 的 daemon 线程从中 get。
    """

    # SOURCE: vllm/v1/engine/core.py:EngineCoreProc (in-process stub for ZMQ output_socket)
    def __init__(self) -> None:
        self._core = _StubEngineCore()
        # 站位真实的 ZMQ output_socket：SyncMPClient 后台线程从这里收 EngineCoreOutputs。
        self.output_socket: queue.Queue = queue.Queue()

    def add_request(self, request: EngineCoreRequest) -> None:
        # SOURCE: vllm/v1/engine/core.py:EngineCoreProc (ZMQ ADD → 后台进程 add_request) — stub
        # 站位'经 ZMQ 把 ADD 投递到独立进程并被那边 add_request'。stub 直接入引擎。
        req, wave = self._core.preprocess_add_request(request)
        self._core.add_request(req, wave)
        self._drive()

    def abort_requests(self, request_ids: list[str]) -> None:
        # SOURCE: vllm/v1/engine/core.py:EngineCoreProc (ZMQ ABORT) — stub
        self._core.abort_requests(request_ids)

    def _drive(self) -> None:
        # SOURCE: vllm/v1/engine/core.py:EngineCoreProc.run_busy_loop — in-process stub
        # 站位独立进程的 busy loop：把当前批的所有请求逐拍跑到 finished，
        # 每拍把 EngineCoreOutputs 推上'output_socket'（供后台线程消费）。
        while self._core.has_unfinished():
            outputs, _ = self._core.step_fn()
            self.output_socket.put(outputs.get(0))
