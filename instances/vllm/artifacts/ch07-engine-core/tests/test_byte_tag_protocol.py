# SPDX-License-Identifier: Apache-2.0
"""TDD: EngineCoreRequestType 字节标签 + make_client 三层选型 + 字节标签分派。

复现真实 vLLM 行为：
- 6 个请求类型是单字节 hex bytes，可直接作 ZMQ 首帧、一步还原。
- make_client(multiprocess, asyncio) 选 Inproc / Sync / Async；asyncio 但非 mp -> NotImplementedError。
- _handle_client_request 按字节标签分派 ADD/ABORT/UTILITY/WAKEUP/EXECUTOR_FAILED。
"""
import os
import queue
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from core import EngineCore, EngineShutdownState, Request  # noqa: E402
from core_client import (  # noqa: E402
    AsyncMPClient,
    EngineCoreClient,
    InprocClient,
    SyncMPClient,
)
from engine_init import EngineCoreRequestType, UtilityOutput  # noqa: E402


def test_request_type_single_byte_tags():
    assert EngineCoreRequestType.ADD.value == b"\x00"
    assert EngineCoreRequestType.ABORT.value == b"\x01"
    assert EngineCoreRequestType.START_DP_WAVE.value == b"\x02"
    assert EngineCoreRequestType.UTILITY.value == b"\x03"
    assert EngineCoreRequestType.EXECUTOR_FAILED.value == b"\x04"
    assert EngineCoreRequestType.WAKEUP.value == b"\x05"
    # round-trip from raw socket frame bytes (one-step decode).
    for rt in EngineCoreRequestType:
        assert EngineCoreRequestType(bytes(rt.value)) is rt


def test_make_client_inproc():
    c = EngineCoreClient.make_client(multiprocess_mode=False, asyncio_mode=False)
    assert isinstance(c, InprocClient)


def test_make_client_asyncio_without_mp_raises():
    with pytest.raises(NotImplementedError):
        EngineCoreClient.make_client(multiprocess_mode=False, asyncio_mode=True)


def _make_engine_with_queues():
    """EngineCore (无 ZMQ 线程) 直接驱动 _handle_client_request 检验字节标签分派。"""
    eng = EngineCore()
    eng.input_queue = queue.Queue()
    eng.output_queue = queue.Queue()
    eng.shutdown_state = EngineShutdownState.RUNNING
    return eng


def test_dispatch_add_stores_request():
    eng = _make_engine_with_queues()
    req = Request("rA", client_index=0)
    eng._handle_client_request(EngineCoreRequestType.ADD, (req, 0))
    assert "rA" in eng._requests


def test_dispatch_abort_removes_request():
    eng = _make_engine_with_queues()
    eng._requests["rB"] = Request("rB")
    eng._handle_client_request(EngineCoreRequestType.ABORT, ["rB"])
    assert "rB" not in eng._requests


def test_dispatch_utility_enqueues_output():
    eng = _make_engine_with_queues()
    eng._handle_client_request(
        EngineCoreRequestType.UTILITY, (0, 777, "get_supported_tasks", ())
    )
    client_idx, outputs = eng.output_queue.get_nowait()
    assert client_idx == 0
    uo: UtilityOutput = outputs.utility_output
    assert uo.call_id == 777
    assert uo.result.result == ("generate",)


def test_dispatch_utility_failure_sets_message():
    eng = _make_engine_with_queues()
    eng._handle_client_request(
        EngineCoreRequestType.UTILITY, (0, 1, "no_such_method", ())
    )
    _, outputs = eng.output_queue.get_nowait()
    assert outputs.utility_output.failure_message is not None


def test_dispatch_wakeup_is_noop():
    eng = _make_engine_with_queues()
    eng._handle_client_request(EngineCoreRequestType.WAKEUP, None)
    assert eng.output_queue.empty()


def test_dispatch_executor_failed_raises():
    eng = _make_engine_with_queues()
    with pytest.raises(RuntimeError):
        eng._handle_client_request(EngineCoreRequestType.EXECUTOR_FAILED, b"")
