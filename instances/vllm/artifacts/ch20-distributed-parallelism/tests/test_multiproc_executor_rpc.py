"""MultiprocExecutor 控制面测试（进程内线程模型，无 CUDA）：

验证精简版复现真实 vLLM 的 collective_rpc 可观察行为：
  - 一次 collective_rpc enqueue 广播到所有 worker，每个 worker 反射调用对应方法。
  - unique_reply_rank 时只从该 rank 的 response_mq 取结果（PP 末段 + TP0 主线）。
  - 不指定 output_rank 时收齐所有 worker 的结果（list）。
  - non_block 返回 FutureWrapper，.result() 取回。
  - worker 抛异常 -> FAILURE -> collective_rpc 抛 RuntimeError。
  - _get_output_rank = world_size - tp_size*pcp_size。
"""
import threading

import pytest

import multiproc_executor as me


class FakeWorker:
    """注入的 worker 对象，替代真实 Worker（init_device/load_model 已 SUBTRACTED）。"""

    def __init__(self, rank):
        self.rank = rank
        self.calls = []

    def echo_rank(self):
        self.calls.append("echo_rank")
        return self.rank

    def add(self, a, b):
        return a + b + self.rank

    def boom(self):
        raise ValueError(f"worker {self.rank} exploded")


@pytest.fixture
def executor():
    workers = {}

    def factory(rank):
        w = FakeWorker(rank)
        workers[rank] = w
        return w

    ex = me.MultiprocExecutor(world_size=4, worker_factory=factory)
    ex._workers_by_rank = workers
    yield ex
    ex.shutdown()


def test_collective_rpc_collects_all_ranks(executor):
    # 不指定 output_rank -> 收齐所有 rank 的结果（list）。
    res = executor.collective_rpc("echo_rank", timeout=10)
    assert res == [0, 1, 2, 3]


def test_collective_rpc_unique_reply_rank(executor):
    # unique_reply_rank=2 -> 只取 rank2 的结果（单值，非 list）。
    res = executor.collective_rpc("echo_rank", timeout=10, unique_reply_rank=2)
    assert res == 2


def test_collective_rpc_with_args(executor):
    res = executor.collective_rpc("add", args=(10, 20), timeout=10)
    # 每 rank 返回 10+20+rank。
    assert res == [30, 31, 32, 33]


def test_collective_rpc_non_block_returns_future(executor):
    fut = executor.collective_rpc("echo_rank", timeout=10, non_block=True)
    assert isinstance(fut, me.FutureWrapper)
    assert fut.result() == [0, 1, 2, 3]


def test_collective_rpc_worker_exception_raises(executor):
    with pytest.raises(RuntimeError, match="Worker failed"):
        executor.collective_rpc("boom", timeout=10, unique_reply_rank=1)


def test_output_rank_formula():
    def factory(rank):
        return FakeWorker(rank)

    ex = me.MultiprocExecutor(
        world_size=8,
        worker_factory=factory,
        tensor_parallel_size=4,
        prefill_context_parallel_size=1,
    )
    try:
        # world_size - tp*pcp = 8 - 4*1 = 4 (PP 末段第一个 TP worker).
        assert ex.output_rank == 4
        assert ex._get_output_rank() == 4
    finally:
        ex.shutdown()


def test_method_can_be_callable_bytes_path(executor):
    # method 为可调用对象时走 pickle bytes 分支，worker 端反序列化后以 self 调用。
    def top_level_method(worker):
        return worker.rank * 100

    # 必须是模块级可 pickle 的函数；这里用 echo via getattr 的等价：直接测 str 路径
    # 已覆盖，bytes 路径用一个模块级函数验证。
    res = executor.collective_rpc(_mul100, timeout=10, unique_reply_rank=3)
    assert res == 300


def _mul100(worker):
    return worker.rank * 100


def test_futures_drained_in_order(executor):
    # 连发两个 non_block，后者 result() 应先排空前者（FIFO 有序排空）。
    f1 = executor.collective_rpc("echo_rank", timeout=10, non_block=True)
    f2 = executor.collective_rpc("add", args=(1, 1), timeout=10, non_block=True)
    # 先 result f2 -> 内部会先排空 f1。
    assert f2.result() == [2, 3, 4, 5]
    assert f1.result() == [0, 1, 2, 3]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
