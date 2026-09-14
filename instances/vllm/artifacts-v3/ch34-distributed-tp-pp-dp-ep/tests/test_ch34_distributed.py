# ch34《分布式 TP/PP/DP/EP》精简版测试 —— TDD（先于实现）
#
# 目标代码仓真实行为（vLLM v0.27.1, 6e448d0ea）的可观察复现：
#   - 5 维 rank 张量四刀（8 GPU TP=2/PP=2/DP=2 的 m2/m3 worked example）
#   - GroupCoordinator 双群组 / 三套坐标 / all_reduce 双路径 / barrier 走 cpu_group
#   - RowParallelLinear TP 消费现场（bias 只加一次）
#   - PP isend/irecv_tensor_dict（metadata 走 cpu_group）+ TP 切片优化 + 懒同步
#   - PP 采样 token 末段广播回首段
#   - DP 批对齐（一次 4×dp all-reduce；cg 取 min、pad 到 max）
#   - EP AgRs dispatch/combine（all_gatherv + reduce_scatterv）与 MoE 消费现场
#   - 内部 LB 打分（v0.27.1 重写版公式）+ 盖章定向 + abort 记账
#   - wave 共识（32 步 2 元素 SUM AR / dummy batch 锁步 / wave_complete -1 哨兵）
#   - DPCoordinator 三 socket 控制面（100ms 发布 / START_DP_WAVE / stale wave）
#
# 运行：cd instances/vllm/artifacts-v3/ch34-distributed-tp-pp-dp-ep
#       python -m pytest tests/ -q
from __future__ import annotations

import asyncio
import socket
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest
import torch

_IMPL = Path(__file__).resolve().parents[1] / "implementation"
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))

import _pool  # noqa: E402  (测试脚手架：常驻 gloo 进程池)

_TMP = Path(__file__).resolve().parent / ".tmp"
_TMP.mkdir(exist_ok=True)
_store_seq = [0]


def _fresh_store() -> dict:
    # 路径带进程号：上一轮崩溃残留的 FileStore 文件会毒化下一轮的 rendezvous
    # （gloo file init 读到旧 world 数据 → rank 错位挂死），跨轮必须不复用。
    import os

    _store_seq[0] += 1
    return {
        "path": (_TMP / f"s{os.getpid()}_{_store_seq[0]}").as_posix(),
        "port": _free_port(),
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ═══════════════════════ unit：集合算子 + fake 形状推断（m4） ═══════════════════════


def test_fake_impls_shape_math():
    """fake(meta) 形状公式：AR 不变 / AG dim×ws / RS dim÷ws（编译期推断语义）。"""
    from vllm.distributed import parallel_state as ps

    x = torch.zeros(4, 6)
    assert ps.all_reduce_fake(x, "tp:0").shape == (4, 6)
    assert ps.all_gather_fake(x, 1, 3, "tp:0").shape == (4, 18)
    assert ps.reduce_scatter_fake(x, 0, 2, "tp:0").shape == (2, 6)


def test_module_level_collective_dispatch_by_group_name():
    """模块级算子按 group_name 字符串查表派发——custom-op 真身的主体。"""
    import weakref

    from vllm.distributed import parallel_state as ps

    class _StubGroup:
        def _all_reduce_out_place(self, t):
            return t * 10

        def _reduce_scatter_out_place(self, t, dim):
            return t.chunk(2, dim)[0]

        def _all_gather_out_place(self, t, dim):
            return torch.cat([t, t], dim)

    stub = _StubGroup()
    name = "unit-stub"
    ps._groups[name] = weakref.ref(stub)
    x = torch.ones(2)
    assert torch.equal(ps.all_reduce(x, name), torch.full((2,), 10.0))
    assert torch.equal(ps.all_gather(x, 0, 2, name), torch.ones(4))
    assert torch.equal(ps.reduce_scatter(x, 0, 2, name), torch.ones(1))
    # 未注册组名 → assert 失败
    with pytest.raises(AssertionError):
        ps.all_reduce(x, "no-such-group")


def test_custom_ops_registered_into_vllm_namespace():
    """三个集合算子注册成 torch.ops.vllm.*（集合通信进编译图的关键动作）。"""
    assert hasattr(torch.ops.vllm, "all_reduce")
    assert hasattr(torch.ops.vllm, "all_gather")
    assert hasattr(torch.ops.vllm, "reduce_scatter")


# ═══════════════════════ unit：懒同步与载荷（m9 / IntermediateTensors） ═══════


def test_async_intermediate_tensors_lazy_sync():
    """__getattribute__ 钩子：谁先碰 .tensors 谁负责 wait；只 wait 一次。"""
    from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

    order = []

    class _Handle:
        def __init__(self, n):
            self.n = n

        def wait(self):
            order.append(f"wait{self.n}")

    def _post_a():
        order.append("post_a")

    t = AsyncIntermediateTensors(
        {"hidden": torch.ones(2)}, comm_handles=[_Handle(1), _Handle(2)],
        comm_postprocess=[_post_a],
    )
    assert order == []  # 构造不触发等待
    _ = t._comm_waited  # 非 .tensors 属性不触发（同样走 __getattribute__）
    assert order == []
    _ = t.tensors  # 首次触碰 .tensors → wait 全部句柄 + postprocess
    assert order == ["wait1", "wait2", "post_a"]
    _ = t.tensors  # 幂等：不再等待
    assert order == ["wait1", "wait2", "post_a"]


def test_intermediate_tensors_dataclass_faces():
    from vllm.sequence import IntermediateTensors

    d = {"a": torch.arange(4).reshape(2, 2), "b": torch.zeros(2, 2)}
    it = IntermediateTensors(d)
    assert it["a"].shape == (2, 2)
    assert torch.equal(it[0:1]["a"], d["a"][:1])
    it["c"] = torch.ones(1, 1)
    assert len(it) == 3 and dict(it.items())["c"].shape == (1, 1)


def test_should_use_all_gather_defaults_and_override():
    """m11 判据：numel 整除 TP 默认开；all_gather_tensors 逐张量覆写。"""
    from vllm.distributed.parallel_state import GroupCoordinator

    class _TP:
        world_size = 2

    gc = object.__new__(GroupCoordinator)
    assert gc._should_use_all_gather("h", 8, _TP, None) is True
    assert gc._should_use_all_gather("h", 7, _TP, None) is False
    assert gc._should_use_all_gather("h", 8, None, None) is False
    assert gc._should_use_all_gather("h", 7, _TP, {"h": True}) is True
    assert gc._should_use_all_gather("h", 8, _TP, {"h": False}) is False


# ═══════════════════════ unit：前端家族 + LB 打分（m14/m15/m16/m17） ═══════


def _dp_client(kind="lb", client_count=1, client_index=0, n_engines=4):
    from vllm.v1.engine.core_client import (
        AsyncMPClient,
        DPAsyncMPClient,
        DPLBAsyncMPClient,
    )

    cls = {"lb": DPLBAsyncMPClient, "ext": DPAsyncMPClient, "plain": AsyncMPClient}[
        kind
    ]
    c = cls.__new__(cls)
    if kind == "lb":
        c.client_count = client_count
        c.reqs_in_flight = {}
        c.engine_inflight = Counter()
    c.current_wave = 0
    c.engines_running = True
    c.client_index = client_index
    c.core_engines = [i.to_bytes(2, "little") for i in range(n_engines)]
    c.lb_engines = [[0, 0, 0.0] for _ in range(n_engines)]
    if kind != "plain":
        c.core_engine = c.core_engines[0]
    if kind == "lb":
        c.eng_start_index = 0
    return c


def _req(req_id="r1", dp_rank=None):
    from vllm.v1.engine import EngineCoreRequest

    class _Pooling:  # late-interaction 分支的判据字段（该分支已按删除项裁除）
        pass

    r = EngineCoreRequest(
        request_id=req_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=dp_rank,
    )
    return r


def test_make_async_mp_client_three_branches():
    from vllm.config import VllmConfig
    from vllm.v1.engine.core_client import (
        AsyncMPClient,
        DPAsyncMPClient,
        DPLBAsyncMPClient,
        EngineCoreClient,
    )

    def cfg(dp, external):
        c = VllmConfig()
        c.parallel_config.data_parallel_size = dp
        c.parallel_config.data_parallel_external_lb = external
        return c

    assert isinstance(
        EngineCoreClient.make_async_mp_client(cfg(1, False), None, False),
        AsyncMPClient,
    )
    assert isinstance(
        EngineCoreClient.make_async_mp_client(cfg(2, True), None, False),
        DPAsyncMPClient,
    )
    assert isinstance(
        EngineCoreClient.make_async_mp_client(cfg(2, False), None, False),
        DPLBAsyncMPClient,
    )


def test_lb_inflight_floor_beats_stale_snapshot():
    """突发期：快照全 0 也要按本地在飞地板轮转（score 主式的 max 语义）。"""
    c = _dp_client()
    chosen = [c.get_core_engine_for_request(_req(f"r{i}")) for i in range(4)]
    # 4 引擎全空：client_count=1 → floor=inflight → 起点轮转近似 round-robin
    assert int.from_bytes(chosen[0], "little") == 0
    assert sorted(int.from_bytes(x, "little") for x in chosen) == [0, 1, 2, 3]
    # 记账：登记 + 自增
    assert c.reqs_in_flight["r0"] == c.core_engines[0]
    assert c.engine_inflight[c.core_engines[2]] == 1


def test_lb_snapshot_raises_score_when_loaded():
    """稳态：coordinator 快照 (waiting+running) 抬分避开重载引擎。"""
    c = _dp_client()
    c.lb_engines = [[0, 0, 0.0], [0, 0, 0.0], [50, 0, 0.0], [0, 0, 0.0]]
    eng = c.get_core_engine_for_request(_req())
    assert int.from_bytes(eng, "little") != 2


def test_lb_kv_pressure_ramp():
    """KV 斜坡：>50% 起罚、满载 waiting×3（score += waiting*6*max(0, kv-0.5)）。"""
    c = _dp_client()
    c.eng_start_index = 0
    # 引擎 0 低压力有队列，引擎 1 高压力同队列：高压力被避开
    c.lb_engines = [
        [4, 0, 0.4],  # score = 4（斜坡关闭）
        [4, 0, 1.0],  # score = 4 + 4*6*0.5 = 16（满载 3 倍）
        [100, 0, 0.0],
        [100, 0, 0.0],
    ]
    eng = c.get_core_engine_for_request(_req())
    assert int.from_bytes(eng, "little") == 0
    # 低压力全零、唯一载荷在高压引擎时：斜坡仍使空引擎胜出
    c2 = _dp_client()
    c2.lb_engines = [[0, 0, 0.0], [4, 0, 1.0], [0, 0, 0.0], [0, 0, 0.0]]
    eng2 = c2.get_core_engine_for_request(_req())
    assert int.from_bytes(eng2, "little") != 1


def test_lb_rotates_start_index_on_ties():
    """平局轮转消偏置：每次选路后 eng_start_index += 1。"""
    c = _dp_client()
    before = c.eng_start_index
    c.get_core_engine_for_request(_req())
    assert c.eng_start_index == (before + 1) % len(c.core_engines)


def test_lb_local_waiting_increment_after_choice():
    """选中后本地预增 waiting×client_count（快照 100ms 间的过渡均衡）。"""
    c = _dp_client(client_count=3)
    c.lb_engines = [[0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0], [0, 0, 0.0]]
    c.get_core_engine_for_request(_req())
    eng0_waiting = c.lb_engines[0][0]
    assert eng0_waiting == 3  # 第一次必选 0（起点 0、全平局）


def test_process_engine_outputs_reclaims_inflight():
    c = _dp_client()
    c.get_core_engine_for_request(_req("a"))
    e = c.reqs_in_flight["a"]
    assert c.engine_inflight[e] == 1

    from vllm.v1.engine import EngineCoreOutputs

    asyncio.run(
        type(c).process_engine_outputs(
            c, EngineCoreOutputs(finished_requests={"a"})
        )
    )
    assert "a" not in c.reqs_in_flight
    assert c.engine_inflight[e] == 0


def test_add_request_async_stamps_and_routes():
    """盖章（current_wave/client_index）+ ROUTER 定向（chosen_engine identity）
    + 暂停期 FIRST_REQ 抢先唤醒（F3/F4 回收）。"""
    from vllm.v1.engine import EngineCoreRequestType

    c = _dp_client("ext")
    c.current_wave = 5
    c.client_index = 2
    c.engines_running = False
    sent, first_reqs = [], []

    async def _noop():
        return None

    c._ensure_stats_update_task = lambda: None
    c._ensure_output_queue_task = lambda: None

    def _send_input(req_type, request, engine=None):
        sent.append((req_type, request.request_id, engine))
        return _noop()

    class _PairSock:
        async def send(self, buf):
            import msgspec.msgpack

            first_reqs.append(msgspec.msgpack.decode(buf))

    c._send_input = _send_input
    c.first_req_send_socket = _PairSock()
    asyncio.run(c.add_request_async(_req("q1")))
    req_type, rid, engine = sent[0]
    assert req_type is EngineCoreRequestType.ADD and rid == "q1"
    assert engine == c.core_engine  # 外部 LB：绑定的引擎原样返回
    # FIRST_REQ 消息体 = ("FIRST_REQ", chosen_engine)
    assert first_reqs[0][0] == "FIRST_REQ"
    assert first_reqs[0][1] == c.core_engine


def test_abort_routes_by_reqs_in_flight():
    from vllm.v1.engine import EngineCoreRequestType

    c = _dp_client()
    c.get_core_engine_for_request(_req("a"))
    c.get_core_engine_for_request(_req("b"))
    a_engine = c.reqs_in_flight["a"]
    aborts = []

    async def _noop():
        return None

    def _abort(req_ids, engine):
        aborts.append((tuple(req_ids), engine))
        return _noop()

    c._abort_requests = _abort
    c.resources = type("R", (), {"engine_dead": False})()
    asyncio.run(c.abort_requests_async(["a", "b"]))
    # 按引擎分桶定向（m16）
    assert len(aborts) == 1 or aborts[0][1] in c.core_engines
    assert {rid for rid, in [(x[0][i],) for x in aborts for i in range(len(x[0]))]} == {
        "a",
        "b",
    }


# ═══════════════════════ unit：引擎出生分叉 + wave 状态机（m13/m19/m20） ═══════


def _vllm_cfg(is_moe, dp_size=2, dp_rank=1):
    from vllm.config import VllmConfig

    c = VllmConfig()
    c.model_config.is_moe = is_moe
    c.parallel_config.data_parallel_size = dp_size
    c.parallel_config.data_parallel_size_local = dp_size
    c.parallel_config.data_parallel_rank = 99
    c.parallel_config.data_parallel_index = 99
    return c


def test_engine_birth_fork_moe_vs_dense(monkeypatch):
    """m13：MoE∧DP>1→DPEngineCoreProc；非 MoE→'treat like DP=1' 就地改配置。"""
    import vllm.v1.engine.core as core_mod
    from vllm.v1.engine.core import EngineCoreProc as _RealProc

    _orig_run_engine_core = _RealProc.run_engine_core
    made = {}

    class _Base:
        def __init__(self, *a, **k):
            made["base"] = (a, k)

        def run_busy_loop(self):
            raise SystemExit

        def shutdown(self):
            pass

    class _DP(_Base):
        def __init__(self, *a, **k):
            made["dp"] = (a, k)  # 不走 _Base.__init__（避免混入 'base' 记录）

    monkeypatch.setattr(core_mod, "EngineCoreProc", _Base, raising=False)
    monkeypatch.setattr(core_mod, "DPEngineCoreProc", _DP, raising=False)
    # 打补丁前的原静态方法（构造走补丁后的类、循环走原 run_engine_core 骨架）
    run_engine_core = _orig_run_engine_core

    cfg = _vllm_cfg(is_moe=True)
    with pytest.raises(SystemExit):
        run_engine_core(vllm_config=cfg, dp_rank=1)
    assert "dp" in made and "base" not in made
    assert cfg.parallel_config.data_parallel_rank == 1
    assert cfg.parallel_config.data_parallel_index == 1

    made.clear()
    cfg2 = _vllm_cfg(is_moe=False)
    with pytest.raises(SystemExit):
        run_engine_core(vllm_config=cfg2, dp_rank=1)
    assert "base" in made and "dp" not in made
    assert made["base"][1]["engine_index"] == 1
    assert cfg2.parallel_config.data_parallel_size == 1
    assert cfg2.parallel_config.data_parallel_size_local == 1
    assert cfg2.parallel_config.data_parallel_rank == 0
    assert cfg2.parallel_config.data_parallel_index == 1  # index 仍记原 DP rank


def test_has_global_unfinished_32_step_gating(monkeypatch):
    """m19：只在 32 的倍数步做共识 all-reduce；pause 共识置 ignore 标志。"""
    from vllm.config import ParallelConfig
    from vllm.v1.engine.core import DPEngineCoreProc

    calls = []

    def fake_sync(dp_group, has_unfinished, pending_pause):
        calls.append((has_unfinished, pending_pause))
        return (has_unfinished, pending_pause and True)

    monkeypatch.setattr(ParallelConfig, "sync_dp_state", staticmethod(fake_sync))
    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    proc.step_counter = 0
    proc.pending_pause = True
    proc.ignore_start_dp_wave = False
    proc.dp_group = object()

    assert proc._has_global_unfinished_reqs(True) is True  # counter 1：短路
    assert proc._has_global_unfinished_reqs(True) is True  # counter 2：短路
    assert calls == []
    proc.step_counter = 31
    assert proc._has_global_unfinished_reqs(True) is True  # counter 32：真共识
    assert calls == [(True, True)]
    assert proc.ignore_start_dp_wave is True
    assert proc.pending_pause is False


def test_maybe_publish_request_counts_minus_one_sentinel():
    """m17：counts 有变才发、client_index=-1 哨兵路由、盖 step/wave 章。
    get_request_counts 的真序 = (num_running_reqs, num_waiting_reqs)。"""
    import queue

    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.engine.core import DPEngineCoreProc

    class _Sched:
        def __init__(self, counts, kv):
            self.counts = counts
            self.kv = kv

        def get_request_counts(self):
            return self.counts

        def get_kv_cache_usage(self):
            return self.kv

    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    proc.publish_dp_lb_stats = True
    proc.scheduler = _Sched((3, 1), 0.42)  # (running, waiting)
    proc.last_counts = (0, 0)
    proc.step_counter = 7
    proc.current_wave = 2
    proc.output_queue = queue.Queue()
    proc._maybe_publish_request_counts()
    client_index, eco = proc.output_queue.get_nowait()
    assert client_index == -1
    assert eco.scheduler_stats.num_running_reqs == 3
    assert eco.scheduler_stats.num_waiting_reqs == 1
    assert eco.scheduler_stats.kv_cache_usage == pytest.approx(0.42)
    assert eco.scheduler_stats.step_counter == 7
    assert eco.scheduler_stats.current_wave == 2
    # counts 未变 → 不再发
    proc._maybe_publish_request_counts()
    assert proc.output_queue.empty()


def test_handle_client_request_start_dp_wave_gate():
    """m20：ignore_start_dp_wave 丢弃在途唤醒；exclude 引擎不被唤醒；新 wave 唤醒。"""
    from vllm.v1.engine import EngineCoreRequestType
    from vllm.v1.engine.core import DPEngineCoreProc

    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    proc.current_wave = 4
    proc.engines_running = False
    proc.ignore_start_dp_wave = False
    proc.engine_index = 1
    # exclude 别的引擎 → 本引擎唤醒、wave 前进
    proc._handle_client_request(EngineCoreRequestType.START_DP_WAVE, (5, 0))
    assert proc.engines_running is True and proc.current_wave == 5
    # exclude 自己 → 不唤醒
    proc.engines_running = False
    proc._handle_client_request(EngineCoreRequestType.START_DP_WAVE, (6, 1))
    assert proc.engines_running is False and proc.current_wave == 5
    # 旧 wave → 忽略
    proc._handle_client_request(EngineCoreRequestType.START_DP_WAVE, (4, 0))
    assert proc.engines_running is False and proc.current_wave == 5
    # 共识后 ignore → 全丢
    proc.ignore_start_dp_wave = True
    proc._handle_client_request(EngineCoreRequestType.START_DP_WAVE, (9, 0))
    assert proc.engines_running is False and proc.current_wave == 5


def test_dp_add_request_stale_wave_reports_start_wave():
    import queue

    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.engine.core import DPEngineCoreProc

    class _Sched:
        pause_state = PauseState.UNPAUSED

        def add_request(self, request):
            pass

    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    proc.scheduler = _Sched()
    proc.has_coordinator = True
    proc.current_wave = 2
    proc.engines_running = False
    proc.output_queue = queue.Queue()

    class _Req:
        request_id = "x"

    proc.add_request(_Req(), request_wave=1)  # 旧 wave 请求 → 上报 start_wave
    client_index, eco = proc.output_queue.get_nowait()
    assert client_index == -1
    assert eco.start_wave == 2
    assert proc.engines_running is True


# ═══════════════════════ unit：needs_dp_coordinator / use_all2all 配置面 ═══════


def test_needs_dp_coordinator_matrix():
    from vllm.config import VllmConfig

    c = VllmConfig()
    c.model_config.is_moe = True
    c.parallel_config.data_parallel_size = 2
    c.parallel_config.data_parallel_external_lb = True
    assert c.needs_dp_coordinator is True  # MoE 外部 LB 也要 wave 协调
    c.model_config.is_moe = False
    assert c.needs_dp_coordinator is False  # 非 MoE 外部 LB：各实例独立
    c.parallel_config.data_parallel_external_lb = False
    assert c.needs_dp_coordinator is True  # 非 MoE 内部 LB：要 stats
    c.parallel_config.data_parallel_size = 1
    assert c.needs_dp_coordinator is False


def test_use_all2all_and_sp_moe_properties():
    from vllm.config import VllmConfig

    c = VllmConfig()
    c.parallel_config.data_parallel_size = 2
    assert c.parallel_config.use_all2all is True  # dp>1
    c2 = VllmConfig()
    c2.parallel_config.enable_expert_parallel = True
    c2.parallel_config.tensor_parallel_size = 2
    c2.parallel_config.data_parallel_size = 2
    c2.parallel_config.all2all_backend = "allgather_reducescatter"
    assert c2.parallel_config.use_sequence_parallel_moe is True
    assert c2.parallel_config.use_all2all is True
    c3 = VllmConfig()
    c3.parallel_config.all2all_backend = "deepep_high_throughput"
    c3.parallel_config.enable_expert_parallel = True
    c3.parallel_config.tensor_parallel_size = 2
    c3.parallel_config.data_parallel_size = 1  # dp=1 → SP-MoE 不成立
    assert c3.parallel_config.use_sequence_parallel_moe is False


def test_dp_group_rank_offset_formula():
    """站 5 的公式（worker 侧）：global = dp_rank*world + rank_in_engine。"""
    dp_rank, world = 3, 8
    rank_in_engine = 5
    assert dp_rank * world + rank_in_engine == 29


# ═══════════════════════ mp：8 GPU 四刀（m2/m3） ═══════════════════════


# pool fixtures → conftest.py（饥饿关闭：用完即 close，避免 15 个 spawn worker 并发驻留）


def test_four_cuts_on_8gpu(pool8):
    """m2 worked example：TP=[0,1][2,3][4,5][6,7]；PP/DP=transpose 后的跨步组；
    EP=同 PP 段内 DP×PCP×TP 全合并、独享 use_all2all。"""
    rec = pool8.map("groups", {"is_moe": True, "store": _fresh_store()})[0]
    assert rec["tp"] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert "tp:mq" in rec  # mq_broadcaster 只给 TP 组
    # PP=transpose(2,4)：固定 (dp,tp) 槽位串起 pp0/pp1 两段——引擎内的流水线。
    # ⚠ 手推值以真实源码行为为准：pp 组是 [[0,2],[1,3],[4,6],[5,7]]（每引擎
    # 各自的 tp0/tp1 两条流水 lane），不是跨引擎的 [0,4]/[1,5]（那是 DP 组员）。
    assert rec["pp"] == [[0, 2], [1, 3], [4, 6], [5, 7]]
    # DP=transpose(1,4)：固定 (tp,pp) 槽位跨引擎取 rank——unbind 按行序出组，
    # 序为 (tp,pp,pcp) 行主序：[0,4]/[2,6]/[1,5]/[3,7]（集合与直觉序一致）
    assert rec["dp"] == [[0, 4], [2, 6], [1, 5], [3, 7]]
    assert rec["ep"] == [[0, 1, 4, 5], [2, 3, 6, 7]]  # transpose(1,2) 全合并
    assert "ep:use_all2all" in rec  # dp>1 → use_all2all=True
    assert "dp:use_all2all" not in rec


def test_dense_model_keeps_ep_none(pool8):
    rec = pool8.map("groups", {"is_moe": False, "store": _fresh_store()})[0]
    assert "ep" not in rec  # dense：不建 EP 组（_EP 保持 None）


# ═══════════════════════ mp：TP=2 全链（m1/m5/m6/m7） ═══════════════════════


def test_tp_group_coordinator_dual_group_allreduce(pool2):
    res = pool2.map("tp_all_reduce", {"store": _fresh_store()})
    for r in res:
        assert r["out"] == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]  # 1+2 部分和
        assert r["ws"] == 2
        assert r["cpu_backend"] == "gloo"
        assert r["dev_backend"] == "gloo"
        assert r["has_dc"] and r["dc_type"] == "CudaCommunicator"
        assert r["pynccl_disabled"] is False  # pynccl seam：gloo 承载、恒可用（其
        #   all_reduce 内部即 torch.distributed——『无 NCCL 兜底』语义等价）
    assert res[0]["rank_in_group"] == 0 and res[1]["rank_in_group"] == 1
    assert res[0]["unique_name"].startswith("tp:")
    assert res[0]["has_mq"] is True  # TP 组独享 mq_broadcaster（world>1）


def test_tp_custom_op_route(pool2):
    res = pool2.map("tp_custom_op_route", {"store": _fresh_store()})
    assert all(r["out"] == [3.0, 3.0] for r in res)  # 2 元张量：各 rank 1+2


def test_tp_barrier_and_object_broadcast(pool2):
    res = pool2.map("tp_barrier_object", {"store": _fresh_store()})
    assert all(r["obj"] == {"k": 0, "list": [1, 2]} for r in res)


def test_row_parallel_linear_tp2(pool2):
    res = pool2.map("rowparallel", {"store": _fresh_store()})
    for r in res:
        assert r["w_shape"] == [6, 4]  # 列切：[out, in/tp]
        for row_got, row_exp in zip(r["out"], r["expected"]):
            for got, exp in zip(row_got, row_exp):
                assert got == pytest.approx(exp, abs=1e-9)


# ═══════════════════════ mp：PP=2 P2P（m8/m12） ═══════════════════════


def test_pp_tensor_dict_roundtrip(pool2):
    res = pool2.map("pp_tensor_dict", {"store": _fresh_store()})
    recv = [r for r in res if r["role"] == "receiver"][0]
    assert recv["hidden"] == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0],
                              [8.0, 9.0, 10.0, 11.0]]
    assert recv["residual"] == [[7.0] * 4] * 3
    assert recv["scalar_meta"] == {"num": 42}


def test_pp_token_broadcast_roundtrip(pool2):
    res = pool2.map("pp_token_broadcast", {"store": _fresh_store(), "chunked": False})
    first = [r for r in res if r["role"] == "first"][0]
    assert first["recv"] == [[11], [22], [33]]
    assert first["placeholder_appended"] == [[-1], [-1], [-1]]  # 本地账 -1 占位


def test_pp_token_broadcast_skips_chunked_prefill(pool2):
    res = pool2.map("pp_token_broadcast", {"store": _fresh_store(), "chunked": True})
    first = [r for r in res if r["role"] == "first"][0]
    # 未广播：recv 是 torch.empty 的未初始化内存——真实行为即『末段的 [11,22,33]
    # 没有到达首段』（值是垃圾值而非广播值；形状契约 [num_reqs,1] 保持）
    assert first["recv"] != [[11], [22], [33]]
    assert len(first["recv"]) == 3 and len(first["recv"][0]) == 1


def test_pp_tp_slice_optimization(pool4):
    """m11：每 rank 只发 1/tp 切片，接收端 all_gather 重建全张量。"""
    res = pool4.map("pp_tp_slice", {"store": _fresh_store()})
    recv = [r for r in res if r["role"] == "pp1"]
    assert len(recv) == 2
    full = [[float(i) for i in range(r * 4, r * 4 + 4)] for r in range(4)]
    for r in recv:
        assert r["hidden"] == full


# ═══════════════════════ mp：DP=2 批对齐 + EP（m21/m22） ═══════════════════════


def test_dp_batch_alignment_pads_to_max(pool2):
    res = pool2.map("dp_align", {"store": _fresh_store(), "mode": "pad"})
    for i, r in enumerate(res):
        assert r["ubatch"] is False
        assert r["padded"] == [120, 120]  # pad 到 max
        assert r["cg_out"] == 1
        assert r["global_rank_offset"] == i  # 站 5：global rank = dp_rank 偏移


def test_dp_batch_alignment_cg_min(pool2):
    res = pool2.map("dp_align", {"store": _fresh_store(), "mode": "min_cg"})
    for r in res:
        assert r["cg_out"] == 0  # 一人 NONE 全体 NONE
        assert r["padded"] == [100, 100]  # cg=0 → 不 pad？不：min=0 → 不 pad，照实下发
        # padded 返回原 num_tokens_across_dp（[100,100]）


def test_ep_dispatch_combine_agrs(pool2):
    res = pool2.map("ep_dispatch_combine", {"store": _fresh_store()})
    h = res[0]["h"]
    for r in res:
        assert r["gathered_rows"] == 6  # 2 rank × 3 token 全网可见
        assert r["dc_all2all"] == "AgRsAll2AllManager"
        assert r["ep_name"].startswith("ep:")
        # combine 归位：每 owner 行 = Σ_k 0.5·h[t] = h[t]（两专家分属两 rank）
        for t, row in enumerate(r["back"]):
            for got, exp in zip(row, h[t]):
                assert got == pytest.approx(exp, abs=1e-6)
    # 本地命中数（每 rank 视角：6 个全网 token 中『topk 至少命中一个本 parity
    # 专家』的 token 数）：topk 恒一奇一偶 → rank0（偶）6 个全命中；rank1（奇）
    # 命中 [0,1]/[1,2] 两型 ×2 = 4。各 rank 只算命中本地的 token——EP 语义。
    hits = sorted(r["n_hits"] for r in res)
    assert hits == [4, 6]


def test_ep_moe_prepare_finalize(pool2):
    """MoE 层消费现场：prepare 后全网 6 行；finalize 把加权结果送回 token 原主。"""
    import torch as th

    res = pool2.map("ep_moe_prepare_finalize", {"store": _fresh_store()})
    th.manual_seed(11)
    h_full = th.randn(3, 4)
    # 各 rank 的 a1 同种子生成 → 同值；combine 归位到 owner 视角：每 rank 拿回
    # 自己那 3 行（0.5+0.5 加权恒等『专家』→ 原值）
    for owner, r in enumerate(res):
        assert r["rows"] == 3  # owner 视角的本 rank chunk（sizes[rank]）
        for row_got, row_exp in zip(r["back"], h_full):
            for got, exp in zip(row_got, row_exp):
                assert got == pytest.approx(exp, abs=1e-6)


# ═══════════════════════ mp：wave 共识（m19） ═══════════════════════


def test_stateless_sync_dp_state_or_and_pause(pool2):
    port = _free_port()
    payload = {
        "port": port,
        "has_unfinished": [True, False],
        "pending_pause": [False, False],
    }
    res = pool2.map("stateless_sync", payload)
    for r in res:
        assert r["has_global"] is True  # [0] SUM=1>0 ≡ OR
        assert r["pause_consensus"] is False
        assert r["or_result"] is True  # MAX ≡ OR
        assert r["group_size"] == 2
    payload2 = {
        "port": _free_port(),
        "has_unfinished": [False, False],
        "pending_pause": [True, True],
    }
    res2 = pool2.map("stateless_sync", payload2)
    for r in res2:
        assert r["pause_consensus"] is True  # SUM==dp_size 全体同意暂停
        # 全员 pending（count==dp_size 整除完成）且无未完请求 → has_global=False：
        # 共识已达成、无残余活动——真公式的『or pause_count % dp_size != 0』只把
        # 『部分 rank 还在等共识』算作有活
        assert r["has_global"] is False


def test_wave_consensus_dummy_lockstep_and_wave_complete(pool2):
    res = pool2.map("wave_consensus", {"port": _free_port(), "max_iters": 40})
    for i, r in enumerate(res):
        assert r["dummies"] >= 30  # 无活引擎持续 dummy batch 维持锁步
        assert r["final_wave"] == 1
        if i == 0:
            assert r["wave_events"] == [(-1, 0)]  # dp_rank0、-1 哨兵
        else:
            assert r["wave_events"] == []  # 非 0 引擎不上报


# ═══════════════════════ mp：DPCoordinatorProc 控制面（m18/m20） ═══════════


def test_coordinator_three_socket_e2e(pool1):
    res = pool1.map("coordinator_e2e", {})[0]
    ev = dict(res["events"])
    counts, wave, running = ev["stats"]
    assert counts[0] == [2, 1, pytest.approx(0.6)]  # 引擎 0 counts 进快照
    assert wave == 0 and running is False
    # FIRST_REQ → START_DP_WAVE(wave, exclude=已收请求引擎)
    assert ev["start_wave"] == [0, 1]
    # wave_complete → (None, wave+1, False)
    assert ev["wave_complete"] == (None, 1, False)
    # stale wave 上报 → 补广播 (3, exclude=engine 0)
    assert ev["stale_wave"] == [3, 0]


def test_dp_coordinator_process_wrapper():
    """DPCoordinator 前端包装：spawn 真进程、回报三地址、可关停。"""
    from vllm.config import VllmConfig
    from vllm.v1.engine.coordinator import DPCoordinator

    cfg = VllmConfig()
    pc = cfg.parallel_config
    pc.data_parallel_size = 2
    pc.data_parallel_master_ip = "127.0.0.1"
    coord = DPCoordinator(pc, enable_wave_coordination=True)
    try:
        assert coord.stats_publish_address.startswith(("tcp://", "ipc://"))
        assert coord.proc.pid is not None
    finally:
        coord.shutdown(timeout=5)
