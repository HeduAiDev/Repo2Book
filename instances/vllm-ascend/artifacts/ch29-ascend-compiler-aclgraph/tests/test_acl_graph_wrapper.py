"""ch25 — ACLGraphWrapper.__call__ 的分桶 capture / replay 状态机 + 207008 兜底（真实控制流）。

测可观察行为（对位 vLLM CUDAGraphWrapper）：
  1. runtime_mode 为 NONE 或不匹配 → 直跑 runnable，不建任何 entry。
  2. 首见某 BatchDescriptor → 用 torch.npu.NPUGraph 捕获，建 concrete_aclgraph_entries[descriptor]。
  3. 不同 BatchDescriptor 各捕一张图（按形状分桶）。
  4. 再见同 descriptor → 不重捕、走 entry.aclgraph.replay() 返回缓存 output。
  5. 捕获中报 207008 → 经 _is_stream_resource_capture_error 命中、改写成带指引的 RuntimeError。
"""
import pytest

import _ch25_acl_graph as ag
from vllm.config import CUDAGraphMode
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.forward_context import get_forward_context


def _make_wrapper(runnable, runtime_mode=CUDAGraphMode.PIECEWISE):
    return ag.ACLGraphWrapper(
        runnable=runnable,
        vllm_config=type("Cfg", (), {"compilation_config": object()})(),
        runtime_mode=runtime_mode,
        cudagraph_options=CUDAGraphOptions(),
    )


def _set_ctx(descriptor, mode):
    ctx = get_forward_context()
    ctx.batch_descriptor = descriptor
    ctx.cudagraph_runtime_mode = mode


def test_mode_none_runs_runnable_directly_no_entry():
    calls = []
    w = _make_wrapper(lambda *a, **k: calls.append(a) or "out")
    _set_ctx("d0", CUDAGraphMode.NONE)
    assert w("x") == "out"
    assert calls == [("x",)]
    assert w.concrete_aclgraph_entries == {}


def test_mode_mismatch_runs_runnable_directly():
    w = _make_wrapper(lambda *a, **k: "direct", runtime_mode=CUDAGraphMode.PIECEWISE)
    _set_ctx("d0", CUDAGraphMode.FULL)  # != wrapper mode
    assert w("x") == "direct"
    assert w.concrete_aclgraph_entries == {}


def test_first_seen_descriptor_captures_and_creates_entry():
    sentinel = object()
    w = _make_wrapper(lambda *a, **k: sentinel)
    _set_ctx("dA", CUDAGraphMode.PIECEWISE)
    out = w()
    assert out is sentinel  # capture branch returns the real output
    assert "dA" in w.concrete_aclgraph_entries
    entry = w.concrete_aclgraph_entries["dA"]
    assert isinstance(entry.aclgraph, ag.torch.npu.NPUGraph)
    assert entry.output is sentinel


def test_distinct_descriptors_each_get_their_own_graph():
    w = _make_wrapper(lambda *a, **k: "o")
    _set_ctx("dA", CUDAGraphMode.PIECEWISE)
    w()
    _set_ctx("dB", CUDAGraphMode.PIECEWISE)
    w()
    assert set(w.concrete_aclgraph_entries) == {"dA", "dB"}
    assert w.concrete_aclgraph_entries["dA"].aclgraph is not w.concrete_aclgraph_entries["dB"].aclgraph


def test_second_call_same_descriptor_replays_not_recaptures():
    n = {"count": 0}

    def runnable(*a, **k):
        n["count"] += 1
        return "captured"

    w = _make_wrapper(runnable)
    _set_ctx("dA", CUDAGraphMode.PIECEWISE)
    first = w()
    captured_graph = w.concrete_aclgraph_entries["dA"].aclgraph
    assert n["count"] == 1 and first == "captured"

    # second call: same descriptor -> replay, runnable NOT called again
    second = w()
    assert n["count"] == 1  # runnable not re-run
    assert second == "captured"  # entry.output returned
    assert captured_graph.replay_count == 1  # replay() invoked once
    assert w.concrete_aclgraph_entries["dA"].aclgraph is captured_graph


def test_207008_during_capture_is_rewritten_with_guidance():
    def boom(*a, **k):
        raise RuntimeError("aclnnError 207008: stream resource exhausted during capture")

    w = _make_wrapper(boom)
    _set_ctx("dErr", CUDAGraphMode.PIECEWISE)
    with pytest.raises(RuntimeError) as ei:
        w()
    assert "cudagraph_capture_sizes" in str(ei.value)


def test_non_207008_capture_error_propagates_unchanged():
    def boom(*a, **k):
        raise RuntimeError("plain unrelated capture failure")

    w = _make_wrapper(boom)
    _set_ctx("dErr2", CUDAGraphMode.PIECEWISE)
    with pytest.raises(RuntimeError) as ei:
        w()
    # 非 207008 原样上抛，不被改写
    assert "cudagraph_capture_sizes" not in str(ei.value)
    assert "plain unrelated capture failure" in str(ei.value)
