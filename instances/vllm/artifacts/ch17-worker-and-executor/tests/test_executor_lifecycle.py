"""Ch17 执行器与 Worker 生命周期 精简版测试 —— 复现真实 vLLM 的可观察行为。

不 import vllm；用桩 VllmConfig / 桩 Worker。覆盖：
  - Executor.get_class 工厂三态分发（type / 'mp' / 'uni' / 自定义 qualname / 非法）；
  - WorkerWrapperBase 延迟初始化（按 worker_cls qualname 解析）+ __getattr__ 透传；
  - run_method 三态派发（str / bytes(cloudpickle) / callable）；
  - UniProcExecutor 同进程直调 + non_block 已完成 Future；
  - FutureWrapper + futures_queue 的 FIFO 顺序排空配对正确性；
  - MultiprocExecutor：拉起子进程、广播 RPC、单 rank 应答(output_rank)、callable 广播、
    方法异常→FAILURE→RuntimeError、进程死亡→sentinel 监控→failure_callback、三级关停。

mp 测试真正 spawn 子进程，较慢；用 pytest.mark 不强制，但默认运行。
"""
import pickle
import time
from collections import deque

import cloudpickle
import conftest  # noqa: F401  (sets sys.path + PYTHONPATH)
import pytest
from conftest import VllmConfig, make_config

from abstract import Executor
from multiproc_executor import FutureWrapper, MultiprocExecutor
from serial_utils import run_method
from uniproc_executor import UniProcExecutor
from worker_base import WorkerWrapperBase


# ----------------------------------------------------------------------------
# Executor.get_class —— 工厂三态分发
# ----------------------------------------------------------------------------

def test_get_class_uni():
    cfg = make_config(backend="uni")
    assert Executor.get_class(cfg) is UniProcExecutor


def test_get_class_mp():
    cfg = make_config(backend="mp")
    assert Executor.get_class(cfg) is MultiprocExecutor


def test_get_class_custom_type():
    class MyExec(UniProcExecutor):
        pass

    cfg = make_config(backend="uni")
    cfg.parallel_config.distributed_executor_backend = MyExec
    assert Executor.get_class(cfg) is MyExec


def test_get_class_type_must_subclass_executor():
    cfg = make_config(backend="uni")
    cfg.parallel_config.distributed_executor_backend = dict  # not an Executor
    with pytest.raises(TypeError):
        Executor.get_class(cfg)


def test_get_class_qualname_string():
    # 自定义 qualname 字符串 → resolve_obj_by_qualname 解析。
    cfg = make_config(backend="uni")
    cfg.parallel_config.distributed_executor_backend = (
        "uniproc_executor.UniProcExecutor"
    )
    assert Executor.get_class(cfg) is UniProcExecutor


def test_get_class_unknown_backend_raises():
    cfg = make_config(backend="uni")
    cfg.parallel_config.distributed_executor_backend = 12345  # not type/str
    with pytest.raises(ValueError):
        Executor.get_class(cfg)


# ----------------------------------------------------------------------------
# WorkerWrapperBase —— 延迟初始化 + __getattr__ 透传
# ----------------------------------------------------------------------------

def test_worker_wrapper_lazy_init_resolves_worker_cls():
    w = WorkerWrapperBase(rpc_rank=0)
    # init_worker 前只记住了 rpc_rank，vllm_config 尚为 None（延迟初始化）。
    assert w.rpc_rank == 0
    assert w.vllm_config is None
    cfg = make_config(backend="uni")
    kwargs = dict(
        vllm_config=cfg, local_rank=0, rank=0,
        distributed_init_method="local", is_driver_worker=True,
        shared_worker_lock=None,
    )
    w.init_worker([kwargs])
    # 解析出的真实类应来自 worker_cls 字符串。
    assert type(w.worker).__name__ == "StubWorker"
    assert w.worker.rank == 0


def test_worker_wrapper_getattr_passthrough():
    w = WorkerWrapperBase(rpc_rank=0)
    cfg = make_config(backend="uni")
    w.init_worker([dict(
        vllm_config=cfg, local_rank=0, rank=0,
        distributed_init_method="local", is_driver_worker=True,
        shared_worker_lock=None,
    )])
    # __getattr__ 把未知属性透传给内部 worker —— collective_rpc 的 getattr 根基。
    w.init_device()
    assert w.device == "stub"          # 透传内部 worker 的属性
    assert w.add(2, 3) == 5            # 透传内部 worker 的方法


def test_worker_wrapper_rejects_non_string_worker_cls():
    w = WorkerWrapperBase(rpc_rank=0)
    cfg = make_config(backend="uni")
    cfg.parallel_config.worker_cls = object  # not a str
    with pytest.raises(ValueError):
        w.init_worker([dict(vllm_config=cfg, shared_worker_lock=None)])


# ----------------------------------------------------------------------------
# run_method —— str / bytes(cloudpickle) / callable 三态派发
# ----------------------------------------------------------------------------

class _Obj:
    def greet(self, name):
        return f"hi {name}"


def test_run_method_str():
    assert run_method(_Obj(), "greet", ("a",), {}) == "hi a"


def test_run_method_str_missing_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        run_method(_Obj(), "nope", (), {})


def test_run_method_callable_receives_obj_as_self():
    obj = _Obj()
    fn = lambda self, x: (self.greet("z"), x)
    assert run_method(obj, fn, (7,), {}) == ("hi z", 7)


def test_run_method_bytes_cloudpickle():
    obj = _Obj()
    fn = lambda self, x: x * 2
    payload = cloudpickle.dumps(fn, protocol=pickle.HIGHEST_PROTOCOL)
    assert run_method(obj, payload, (21,), {}) == 42


# ----------------------------------------------------------------------------
# UniProcExecutor —— 同进程直调（最简对照）
# ----------------------------------------------------------------------------

def test_uniproc_collective_rpc_direct_call():
    ex = UniProcExecutor(make_config(backend="uni"))
    try:
        # 同进程直调 worker 方法，返回 list（每 worker 一个结果，uni 即一个）。
        assert ex.collective_rpc("add", args=(4, 5)) == [9]
        # execute_model 走基类薄封装 → collective_rpc → output[0]。
        assert ex.execute_model("SCHED") == ("echo", 0, "SCHED")
    finally:
        ex.shutdown()


def test_uniproc_non_block_returns_completed_future():
    ex = UniProcExecutor(make_config(backend="uni"))
    try:
        fut = ex.collective_rpc("add", args=(1, 2), non_block=True)
        assert fut.done()
        assert fut.result() == [3]
    finally:
        ex.shutdown()


def test_uniproc_non_block_propagates_exception():
    ex = UniProcExecutor(make_config(backend="uni"))
    try:
        fut = ex.collective_rpc("boom", non_block=True)
        with pytest.raises(ValueError):
            fut.result()
    finally:
        ex.shutdown()


# ----------------------------------------------------------------------------
# FutureWrapper —— FIFO 顺序排空配对正确性
# ----------------------------------------------------------------------------

def test_future_wrapper_fifo_pairing():
    """连发 3 个 future，对最后一个 result() 应先按 FIFO 排空前两个。

    底层回复顺序 = 发出顺序。每个 future 的 get_response 从一个共享 FIFO 取下一条；
    只要 result() 在取自己前先排空更早的 future，第 k 个 future 就拿到第 k 个回复。
    """
    fq: deque[FutureWrapper] = deque()
    responses = deque(["r1", "r2", "r3"])  # FIFO：发出顺序 = 回复顺序
    order = []

    def make_get(tag):
        def get():
            order.append(tag)
            return responses.popleft()
        return get

    f1 = FutureWrapper(fq, get_response=make_get("g1"))
    f2 = FutureWrapper(fq, get_response=make_get("g2"))
    f3 = FutureWrapper(fq, get_response=make_get("g3"))

    # 对最后发出的 f3 调 result()：应先排空 f1、f2，再取 f3。
    assert f3.result() == "r3"
    # 取回复的顺序必须 = 发出顺序（FIFO 配对），否则会取错回复。
    assert order == ["g1", "g2", "g3"]
    # 前面两个 future 已被排空、各自配对到正确回复。
    assert f1.result() == "r1"
    assert f2.result() == "r2"


def test_future_wrapper_appendleft_pop_is_fifo():
    fq: deque[FutureWrapper] = deque()
    f1 = FutureWrapper(fq, get_response=lambda: "a")
    f2 = FutureWrapper(fq, get_response=lambda: "b")
    # appendleft 入队、pop 出队 ⇒ FIFO：最先入队的 f1 最先被 pop 排空。
    assert list(fq) == [f2, f1]
    assert fq.pop() is f1


# ----------------------------------------------------------------------------
# MultiprocExecutor —— 真正拉起子进程的控制平面闭环
# ----------------------------------------------------------------------------

def _mp_executor(world_size=2):
    return MultiprocExecutor(make_config(backend="mp", world_size=world_size))


def test_mp_broadcast_to_all_workers():
    ex = _mp_executor(world_size=2)
    try:
        # 不指定 output_rank → 收齐所有 worker 的回复（list 长度 = world_size）。
        results = ex.collective_rpc("add", args=(10, 1))
        assert results == [11, 11]
    finally:
        ex.shutdown()


def test_mp_single_rank_reply_via_execute_model():
    ex = _mp_executor(world_size=2)
    try:
        # execute_model 用 unique_reply_rank=output_rank，只收一个 rank 的结果。
        # output_rank = world_size - tp*pcp = 2 - 2*1 = 0 → rank0 回 ('echo', 0, 'X')。
        assert ex.output_rank == 0
        out = ex.execute_model("X")
        assert out == ("echo", 0, "X")
    finally:
        ex.shutdown()


def test_mp_callable_broadcast_cloudpickled():
    ex = _mp_executor(world_size=2)
    try:
        # method 是 callable → cloudpickle.dumps 序列化整函数发给所有 worker，
        # worker 侧 cloudpickle.loads 后以 self=worker 调用。
        fn = lambda self, k: self.rank * 100 + k
        assert ex.collective_rpc(fn, args=(7,)) == [7, 107]
    finally:
        ex.shutdown()


def test_mp_method_exception_becomes_runtime_error():
    ex = _mp_executor(world_size=2)
    try:
        # worker 方法抛异常 → busy_loop 捕获 → enqueue (FAILURE, msg) →
        # collective_rpc 的 get_response 抛 RuntimeError。
        with pytest.raises(RuntimeError):
            ex.collective_rpc("boom", unique_reply_rank=0)
    finally:
        ex.shutdown()


def test_mp_process_death_triggers_failure_callback():
    ex = _mp_executor(world_size=2)
    fired = {"v": False}

    def cb():
        fired["v"] = True

    try:
        ex.register_failure_callback(cb)
        # 让 rank0 子进程直接退出（模拟 OOM/segfault，根本到不了 except）。
        # 不收回复（output_rank=None 但进程会死），用 non_block 避免主进程阻塞在死队列上。
        ex.collective_rpc("crash_process", unique_reply_rank=0, non_block=True)
        # sentinel 监控线程应检测到进程死亡 → is_failed=True → shutdown → 回调。
        deadline = time.time() + 10
        while time.time() < deadline and not fired["v"]:
            time.sleep(0.05)
        assert ex.is_failed is True
        assert fired["v"] is True
    finally:
        ex.shutdown()


def test_mp_register_callback_after_failure_fires_immediately():
    ex = _mp_executor(world_size=1)
    try:
        ex.is_failed = True
        fired = {"v": False}
        ex.register_failure_callback(lambda: fired.__setitem__("v", True))
        assert fired["v"] is True
    finally:
        ex.shutdown()
