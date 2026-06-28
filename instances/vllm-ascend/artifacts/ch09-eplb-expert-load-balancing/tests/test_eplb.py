"""ch09 — EPLB 在线热迁移流水线：可读控制流测试。

测的是「复现昇腾真实源码的可观察行为」（dossier 记录的节拍时序 / 队列解耦 / 策略多态 /
三态机），而非精简版自洽。host 无 NPU/CANN/分布式后端：真实 P2P 权重搬运与子进程绑 NPU 不真跑
（由 eplb_runtime_stub 的 record-only dist 接住），只验四块主线的纯 Python 控制流。
"""
import sys
import threading
import types
from pathlib import Path

import torch

IMPL = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

from eplb_device_transfer_loader import D2DExpertWeightLoader, ExpertWeightUpdateState  # noqa: E402
from eplb_updator import EplbUpdator  # noqa: E402
from eplb_worker import EplbProcess, EplbWorker  # noqa: E402
from policy_default_eplb import DefaultEplb  # noqa: E402
from policy_factory import PolicyFactory  # noqa: E402
from policy_other import FlashLB, RandomLoadBalance, SwiftBalanceEplb  # noqa: E402


# ---------------------------------------------------------------------------
# 测试替身（fake 模型适配层 / 配置 / 子进程句柄）—— 仅喂控制流，不模拟真实计算。
# ---------------------------------------------------------------------------
def _make_eplb_config(interval=4, algo=2, policy_type=1):
    return types.SimpleNamespace(
        eplb_policy_type=policy_type,
        expert_map_path=None,
        expert_map_record_path=None,
        expert_heat_collection_interval=interval,
        algorithm_execution_interval=algo,
    )


class _FakeModel:
    def __init__(self):
        self.cleared = 0

    def clear_all_moe_loads(self):
        self.cleared += 1


class _FakeAdaptor:
    def __init__(self, num_moe_layers=3, num_dense_layers=0):
        self.num_moe_layers = num_moe_layers
        self.num_dense_layers = num_dense_layers
        self.model = _FakeModel()
        self._load = torch.arange(num_moe_layers * 2, dtype=torch.float32).reshape(num_moe_layers, 2)

    def get_rank_expert_workload(self):
        return self._load

    def get_global_expert_map(self):
        return self._load.clone()


class _FakeProcess:
    pid = 4242

    def is_alive(self):
        return False


def _make_updator(interval=4, algo=2, num_moe_layers=3):
    cfg = _make_eplb_config(interval=interval, algo=algo)
    loader = D2DExpertWeightLoader()
    proc = EplbProcess(shared_dict={}, policy_type=1)
    up = EplbUpdator(cfg, loader, proc, _FakeProcess())
    up.set_adaptor(_FakeAdaptor(num_moe_layers=num_moe_layers))
    return up


# ---------------------------------------------------------------------------
# 主线①：节拍状态机 —— cur_iterations + 三 flag 全靠对 interval 常量的算术比较。
# ---------------------------------------------------------------------------
def test_cadence_flags_fire_on_exact_iterations():
    interval, algo, L = 4, 2, 3
    up = _make_updator(interval, algo, L)

    wakeup_hits, getinfo_hits, weight_hits = [], [], []
    total = interval + algo + L  # 9
    for cur in range(total):
        up.cur_iterations = cur
        if up.wakeup_eplb_worker_flag():
            wakeup_hits.append(cur)
        if up.get_update_info_flag():
            getinfo_hits.append(cur)
        if up.update_expert_weight_flag():
            weight_hits.append(cur)

    # gather+唤醒 恰在 interval-1；取规划 恰在 interval+algo-1；逐层搬权重窗口 [interval+algo, +L)
    assert wakeup_hits == [interval - 1]  # [3]
    assert getinfo_hits == [interval + algo - 1]  # [5]
    assert weight_hits == [interval + algo + i for i in range(L)]  # [6, 7, 8]


def test_update_iteration_resets_full_cycle():
    interval, algo, L = 4, 2, 3
    up = _make_updator(interval, algo, L)
    total = interval + algo + L  # 9

    # 走到满一整轮前一拍，再推进一拍触发归零 + clear_all_moe_loads
    up.cur_iterations = total - 1  # 8
    up.update_iteration()
    assert up.cur_iterations == 0
    assert up.adaptor.model.cleared == 1

    # 未满一轮不归零
    up.cur_iterations = 2
    up.update_iteration()
    assert up.cur_iterations == 3
    assert up.adaptor.model.cleared == 1


def test_wakeup_puts_signal_and_moe_load_written():
    up = _make_updator()
    up.wakeup_eplb_worker()
    assert up.eplb_process.planner_q.get() == 1  # planner_q 是「点火」信号

    up.compute_and_set_moe_load()
    assert up.shared_dict["moe_load"] is not None  # all_gather 全局负载写入 shared_dict


# ---------------------------------------------------------------------------
# 主线②：两条队列解耦 —— planner_q 无界点火，block_update_q maxsize=1 背压（只留最新一份）。
# ---------------------------------------------------------------------------
def test_block_update_queue_is_bounded_to_one():
    proc = EplbProcess(shared_dict={}, policy_type=1)
    assert proc.block_update_q._maxsize == 1
    # planner_q 无界（Queue() 默认上限为 SEM_VALUE_MAX 量级，远大于背压队列的 1）
    assert proc.planner_q._maxsize > 1


def test_worker_process_roundtrip_and_backpressure():
    proc = EplbProcess(shared_dict={}, policy_type=1)

    # 用桩替换真正的规划（do_update 依赖 NPU 上的 moe_load，不在 host 跑）；只验队列编排。
    # 第 3 次点火主动抛错，借 worker_process 的 except→break 干净结束线程（消除拆机噪声）。
    counter = {"n": 0}

    def fake_do_update():
        counter["n"] += 1
        if counter["n"] > 2:
            raise RuntimeError("stop worker thread")
        return f"plan-{counter['n']}"

    proc.worker.do_update = fake_do_update

    t = threading.Thread(target=proc.worker_process, args=(proc.planner_q, proc.block_update_q), daemon=True)
    t.start()

    # 第一次点火 → 子进程算 → 结果进 block_update_q
    proc.planner_q.put(1)
    assert proc.block_update_q.get(timeout=5) == "plan-1"

    # 背压：再点火，block_update_q 已空 → 新结果可入队（消费后才 put 的 while 自旋逻辑）
    proc.planner_q.put(1)
    assert proc.block_update_q.get(timeout=5) == "plan-2"

    proc.planner_q.put(1)  # 第 3 次：do_update 抛错 → worker_process break
    t.join(timeout=5)
    assert not t.is_alive()


# ---------------------------------------------------------------------------
# 主线②：EplbWorker.compose_expert_update_info_greedy —— 用 expert_map 的 -1 差集算 send/recv。
# ---------------------------------------------------------------------------
def test_compose_send_recv_from_map_diff():
    worker = EplbWorker.__new__(EplbWorker)  # 跳过 __init__（避免起策略/分布式），仅测纯逻辑方法
    worker.rank_id = 0

    # 1 层 / 2 rank / 2 个 global expert。expert_map[rank][expert] = 本地槽 或 -1（该 rank 不持有）。
    # current：rank0 持 expert0，rank1 持 expert1。
    current = torch.tensor([[[0, -1], [-1, 0]]])
    # updated：把 expert0 也铺到 rank1（rank1 需 recv expert0；无人 send-out → 从持有者 rank0 复制）。
    updated = torch.tensor([[[0, -1], [0, 0]]])

    (send_info, recv_info, _new_map, layer_id) = next(
        worker.compose_expert_update_info_greedy(updated, current)
    )

    assert layer_id == 0
    # rank0 把 expert0 发给 rank1
    assert send_info == {0: [(1, 0)]}
    # rank1 从 rank0 收 expert0
    assert recv_info == {1: [(0, 0)]}


# ---------------------------------------------------------------------------
# 主线④：策略多态 —— PolicyFactory int→策略类，未知回退 RandomLoadBalance。
# ---------------------------------------------------------------------------
def test_policy_factory_dispatch():
    assert isinstance(PolicyFactory.generate_policy(0), RandomLoadBalance)
    assert isinstance(PolicyFactory.generate_policy(1), DefaultEplb)
    assert isinstance(PolicyFactory.generate_policy(2), SwiftBalanceEplb)
    assert isinstance(PolicyFactory.generate_policy(3), FlashLB)
    # 未知 policy_type → 回退 RandomLoadBalance
    assert isinstance(PolicyFactory.generate_policy(99), RandomLoadBalance)


# ---------------------------------------------------------------------------
# 主线④：DefaultEplb.rebalance_experts —— 冗余复制 + 贪心装箱 + 0.95 收益门槛。
# ---------------------------------------------------------------------------
def test_default_eplb_redundancy_count():
    # placement 行 [[0,0],[1,2]] → unique {0:2,1:1,2:1} → 冗余数 = sum(counts-1) = 1
    import numpy as np

    _, counts = np.unique(np.array([0, 0, 1, 2]), return_counts=True)
    assert DefaultEplb.get_redundant_num(2, counts) == 1


def test_default_eplb_rebalance_balanced_input_no_change():
    policy = DefaultEplb()
    # 1 层 / 2 npu / 2 expert/npu，专家 0..3 各一份（无冗余），负载完全均衡 → 无法再降 5% → change=0
    current = [[[0, 1], [2, 3]]]
    workload = [[[10, 10], [10, 10]]]
    change, priority, deployment = policy.rebalance_experts(current, workload)
    assert change == 0
    assert len(deployment) == 1  # 层数保持
    assert len(deployment[0]) == 2  # npu 数保持
    assert len(priority) == 1


# ---------------------------------------------------------------------------
# 主线③：D2DExpertWeightLoader 三态机 WAITING→READY→TRANSFERRING→WAITING。
# ---------------------------------------------------------------------------
class _FakeLoaderAdaptor:
    def __init__(self):
        # global expert id → local slot（per layer）
        self.expert_map_per_layer_cpu = {0: torch.tensor([0, 1, 2, 3])}
        # per layer, per local expert：一份权重张量列表（这里每 expert 1 块，喻「多块张量」）
        self.expert_param_per_layer = {0: [[torch.zeros(2)] for _ in range(4)]}
        # 收方预分配 buffer（每 buffer 一份等形张量列表）
        self.buffer_tensor_list = [[torch.ones(2)]]
        self.updated_map_calls = []
        self.updated_log2phy_calls = []
        self.updated_weight_calls = []

    def do_update_expert_map(self, layer_id, updated_expert_map):
        self.updated_map_calls.append(layer_id)

    def do_update_log2phy_map(self, layer_id, updated_log2phy_map):
        self.updated_log2phy_calls.append(layer_id)

    def do_update_expert_weight(self, layer_id, local_expert_to_replace, buffer_tensor_id):
        self.updated_weight_calls.append((layer_id, local_expert_to_replace, buffer_tensor_id))


def test_d2d_three_state_transitions():
    loader = D2DExpertWeightLoader()
    loader.set_adator(_FakeLoaderAdaptor())
    assert loader.state == ExpertWeightUpdateState.WAITING

    # rank 视角：发 expert0 给 rank1；收 expert2（buffer 0）；updated_map 给出落地的本地槽。
    expert_send_info = [(1, 0)]
    expert_recv_info = [(1, 2)]
    updated_expert_map = torch.tensor([0, 1, 2, 3])

    loader.set_log2phy_map(torch.tensor([0, 1, 2, 3]))
    loader.generate_expert_d2d_transfer_task(expert_send_info, expert_recv_info, updated_expert_map, layer_id=0)
    assert loader.state == ExpertWeightUpdateState.READY
    # 1 个 isend（送方权重）+ 1 个 irecv（收进 buffer）= 2 个 P2POp
    assert len(loader.comm_op_list) == 2
    assert loader.recv_expert_list == [(2, 0)]  # (local_expert_to_replace, buffer_tensor_id)

    reqs = []
    loader.asyn_expert_weight_transfer(reqs)
    assert loader.state == ExpertWeightUpdateState.TRANSFERRING
    assert len(reqs) == 2  # batch_isend_irecv 为每个 P2POp 返回一个句柄

    loader.update_expert_map_and_weight(reqs)
    assert loader.state == ExpertWeightUpdateState.WAITING  # 闭环回 WAITING
    assert all(r.waited for r in reqs)  # 先 req.wait 收车
    assert loader.eplb_adaptor.updated_map_calls == [0]
    assert loader.eplb_adaptor.updated_weight_calls == [(0, 2, 0)]  # 再 copy_ 落地


def test_d2d_rejects_new_task_while_busy():
    loader = D2DExpertWeightLoader()
    loader.set_adator(_FakeLoaderAdaptor())
    loader.generate_expert_d2d_transfer_task([(1, 0)], [(1, 2)], torch.tensor([0, 1, 2, 3]), layer_id=0)
    assert loader.state == ExpertWeightUpdateState.READY

    # 上一轮未落地（非 WAITING）→ 新任务被拒，状态不变
    before = loader.comm_op_list
    loader.generate_expert_d2d_transfer_task([(0, 1)], [(0, 3)], torch.tensor([0, 1, 2, 3]), layer_id=1)
    assert loader.state == ExpertWeightUpdateState.READY
    assert loader.comm_op_list is before  # 未被改写
