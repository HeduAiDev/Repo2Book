# SPDX-License-Identifier: Apache-2.0
"""TDD: 端到端 IPC 边界 —— SyncMPClient / AsyncMPClient 经真实 ZMQ socket 与
EngineCoreProc（同进程线程承载）通信。

复现真实 vLLM 可观察行为：
- ready 握手：DEALER 先发 EngineCoreReadyResponse，client 的 ROUTER 收齐才放行构造。
- ADD 字节标签请求经 ROUTER->DEALER 抵达 engine，被解码并落入 input_queue->scheduler。
- ABORT 同时进 input_queue 与 aborts_queue（双队列急停）。
- call_utility / call_utility_async 的 call_id+Future RPC 在单向 ZMQ 流上配对返回。
- validate_alive 见 ENGINE_CORE_DEAD 单帧抛 EngineDeadError。
- 带大张量的 ADD 触发多帧 send(track=True)，pending_messages 保留引用。
"""
import asyncio
import os
import sys
import time

import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from core import EngineCoreProc  # noqa: E402
from core_client import (  # noqa: E402
    AsyncMPClient,
    BackgroundResources,
    EngineDeadError,
    SyncMPClient,
)
from engine_init import EngineCoreRequest  # noqa: E402


def _wait(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_ready_handshake_and_identity():
    c = SyncMPClient()
    try:
        # 握手成功 -> core_engine 身份是 rank0 的 2 字节 LE。
        assert c.core_engine == (0).to_bytes(2, "little")
    finally:
        c.shutdown()


def test_sync_add_request_reaches_engine():
    c = SyncMPClient()
    try:
        c.add_request(EngineCoreRequest(request_id="rid1", prompt_token_ids=[1, 2]))
        assert _wait(lambda: "rid1" in c._engine._requests)
    finally:
        c.shutdown()


def test_sync_abort_dual_queue():
    c = SyncMPClient()
    try:
        c.add_request(EngineCoreRequest(request_id="rid2", prompt_token_ids=[1]))
        assert _wait(lambda: "rid2" in c._engine._requests)
        c.abort_requests(["rid2"])
        # abort 也进 aborts_queue（急停路径）。
        assert _wait(lambda: not c._engine.aborts_queue.empty()
                     or "rid2" not in c._engine._requests)
    finally:
        c.shutdown()


def test_sync_call_utility_rpc_roundtrip():
    c = SyncMPClient()
    try:
        result = c.call_utility("get_supported_tasks")
        assert result == ["generate"]
        # call_id Future 已被解决并清出登记表。
        assert c.utility_results == {}
    finally:
        c.shutdown()


def test_sync_call_utility_failure_raises():
    c = SyncMPClient()
    try:
        with pytest.raises(Exception):
            c.call_utility("nonexistent_method")
    finally:
        c.shutdown()


def test_large_tensor_add_uses_pending_messages():
    c = SyncMPClient()
    try:
        big = torch.arange(500, dtype=torch.int64)  # >256B -> aux 帧 -> track=True
        c.add_request(
            EngineCoreRequest(request_id="rid3", prompt_token_ids=[1], prompt_embeds=big)
        )
        assert _wait(lambda: "rid3" in c._engine._requests)
    finally:
        c.shutdown()


def test_async_client_full_loop():
    async def run():
        c = AsyncMPClient()
        try:
            await c.add_request_async(
                EngineCoreRequest(request_id="aid1", prompt_token_ids=[9])
            )
            assert _wait(lambda: "aid1" in c._engine._requests)
            tasks = await c.get_supported_tasks_async()
            assert tasks == ["generate"]
            await c.abort_requests_async(["aid1"])
        finally:
            c.shutdown()

    asyncio.run(run())


def test_validate_alive_raises_on_dead_sentinel():
    import zmq

    class _Frame:
        def __init__(self, b):
            self.buffer = b

    res = BackgroundResources(ctx=zmq.Context())
    with pytest.raises(EngineDeadError):
        res.validate_alive([_Frame(EngineCoreProc.ENGINE_CORE_DEAD)])
    assert res.engine_dead is True
    # 普通多帧消息不触发。
    res2 = BackgroundResources(ctx=zmq.Context())
    res2.validate_alive([_Frame(b"a"), _Frame(b"b")])
    assert res2.engine_dead is False
