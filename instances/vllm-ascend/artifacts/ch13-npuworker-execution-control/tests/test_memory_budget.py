"""对位真实行为：determine_available_memory 的 KV 显存预算与 gpu-memory-utilization 回退建议。

显存回退决策是纯 Python 算术（vllm_ascend/worker/worker.py:L334-L462）：
  available_kv_cache_memory_bytes = requested_memory − non_kv_cache_memory − npugraph(若开关启用)
  其中 non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights_memory
  回退建议 suggested_util = min(round(util + npugraph/total, 4), 1.0)

host 无 NPU：把 memory_profiling / torch.npu.memory_stats / CUDAGraphMode / envs_vllm 注入桩，
真正执行 determine_available_memory 验数值（设备路径不真跑，算术真跑）。
"""
import types
from contextlib import contextmanager

import pytest
import torch

import worker

GiB = 1 << 30


def _make_profile_result():
    # memory_profiling 上下文产出的 profile_result（昇腾用其字段算 non_kv_cache_memory）。
    return types.SimpleNamespace(
        before_profile=types.SimpleNamespace(torch_peak=1 * GiB),
        after_profile=types.SimpleNamespace(free_memory=70 * GiB),
        non_torch_increase=5 * GiB,
        weights_memory=10 * GiB,
        torch_peak_increase=None,  # 由方法用 pre-graph 值覆写
        non_kv_cache_memory=None,
    )


def _install_stubs(monkeypatch, profile_result, *, estimate_cudagraphs: bool):
    monkeypatch.setattr(worker, "GiB_bytes", GiB, raising=False)
    monkeypatch.setattr(worker, "CUDAGraphMode", types.SimpleNamespace(NONE="NONE"), raising=False)
    monkeypatch.setattr(
        worker, "envs_vllm",
        types.SimpleNamespace(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=estimate_cudagraphs),
        raising=False,
    )

    @contextmanager
    def fake_memory_profiling(init_snapshot, weights_memory):
        yield profile_result

    monkeypatch.setattr(worker, "memory_profiling", fake_memory_profiling, raising=False)
    # 图捕获前的 torch peak = 8 GiB → torch_peak_increase = 8 − 1 = 7 GiB
    monkeypatch.setattr(
        torch, "npu",
        types.SimpleNamespace(memory_stats=lambda dev: {"allocated_bytes.all.peak": 8 * GiB}),
        raising=False,
    )


def _make_worker(profile_result):
    w = worker.NPUWorker.__new__(worker.NPUWorker)
    w.device = "npu:0"
    w.init_snapshot = types.SimpleNamespace(free_memory=95 * GiB, total_memory=100 * GiB)
    w.requested_memory = 90 * GiB  # = total(100) × util(0.9)
    w.cache_config = types.SimpleNamespace(gpu_memory_utilization=0.9, kv_cache_memory_bytes=None)
    w.vllm_config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(cudagraph_mode="FULL")  # != NONE → 估算 ACLGraph
    )
    w.model_runner = types.SimpleNamespace(
        model_memory_usage=10 * GiB,
        profile_run=lambda: None,
        profile_cudagraph_memory=lambda: 3 * GiB,  # ACLGraph 显存估算 = 3 GiB
    )
    return w


def test_kv_budget_subtracts_aclgraph_when_estimate_enabled(monkeypatch):
    pr = _make_profile_result()
    _install_stubs(monkeypatch, pr, estimate_cudagraphs=True)
    w = _make_worker(pr)

    kv = w.determine_available_memory()

    # non_kv = non_torch(5) + torch_peak_increase(7) + weights(10) = 22 GiB
    # available_kv = requested(90) − non_kv(22) − npugraph(3) = 65 GiB
    assert kv == 65 * GiB
    assert pr.torch_peak_increase == 7 * GiB
    assert pr.non_kv_cache_memory == 22 * GiB
    assert w.npugraph_memory_estimate == 3 * GiB


def test_kv_budget_keeps_aclgraph_when_estimate_disabled(monkeypatch):
    pr = _make_profile_result()
    _install_stubs(monkeypatch, pr, estimate_cudagraphs=False)
    w = _make_worker(pr)

    kv = w.determine_available_memory()

    # 开关关闭 → npugraph 不从预算里扣：available_kv = 90 − 22 = 68 GiB
    assert kv == 68 * GiB
    # 估算值仍被算出并存下（>0 → 触发回退建议日志）
    assert w.npugraph_memory_estimate == 3 * GiB


def test_fallback_util_arithmetic(monkeypatch):
    # 回退建议算式：delta = npugraph/total，suggested = min(util + delta, 1.0)。
    npugraph, total, util = 3 * GiB, 100 * GiB, 0.9
    delta = npugraph / total
    assert round(util + delta, 4) == 0.93
    assert min(round(util + delta, 4), 1.0) == 0.93
    # 封顶 1.0：util 接近满时不超过 1.0
    assert min(round(0.99 + delta, 4), 1.0) == 1.0
