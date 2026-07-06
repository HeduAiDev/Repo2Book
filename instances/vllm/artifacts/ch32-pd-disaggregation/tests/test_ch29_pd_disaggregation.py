"""TDD tests for ch29《PD 分离的抽象与调度器集成》subtract-only companion.

测的是精简版复现 **真实 vLLM 的可观察行为**（不是测自洽）：
  - KVConnectorRole 二分；factory 按 role 各构造一份实例（进程隔离契约）。
  - 决策侧 get_num_new_matched_tokens / update_state_after_alloc / build_connector_meta
    的真实落地（ExampleConnector：靠文件夹存在判命中、block 粒度对齐）。
  - 调度集成 f12 闭环：远程命中 + load_kv_async → WAITING_FOR_REMOTE_KVS 隔离进
    skipped_waiting（避队头阻塞）→ worker 报 finished_recving 入
    finished_recving_kv_req_ids → 下步 _try_promote_blocked_waiting_request 提升回
    WAITING/PREEMPTED；整 prompt 命中回退一个 token；finished_sending 释放块。

纯单元测试，不 import vllm → host pytest 即可。
"""
import os
import sys
import types

import pytest
import torch

# 让 `import implementation.xxx` 与精简版里 `from implementation...` 解析一致。
_HERE = os.path.dirname(os.path.abspath(__file__))
_CH = os.path.dirname(_HERE)
if _CH not in sys.path:
    sys.path.insert(0, _CH)

from implementation.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from implementation.example_connector import (  # noqa: E402
    ExampleConnector,
    ExampleConnectorMetadata,
    align_to_block_size,
)
from implementation.factory import KVConnectorFactory  # noqa: E402
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.request_queue import SchedulingPolicy  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402


# --------------------------------------------------------------------------
# 测试桩
# --------------------------------------------------------------------------
def make_vllm_config(tmp_path, block_size=4, connector="ExampleConnector"):
    extra = {"shared_storage_path": str(tmp_path)}
    kv_transfer_config = types.SimpleNamespace(
        kv_connector=connector,
        get_from_extra_config=lambda k, d: extra.get(k, d),
    )
    cache_config = types.SimpleNamespace(block_size=block_size)
    return types.SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        cache_config=cache_config,
    )


class StubKVCacheManager:
    """桩缓存管理器：只记录被调用过哪些方法，并返回调度循环需要的形状。"""

    empty_kv_cache_blocks = object()

    def __init__(self):
        self.cached = []
        self.freed = []

    def get_computed_blocks(self, request):
        return (object(), 0)  # 本地命中 0 token

    def allocate_slots(self, request, num_new_tokens, **kwargs):
        return object()  # 分配成功

    def get_blocks(self, request_id):
        return [[0]]

    def get_block_ids(self, request_id):
        return [[0]]

    def remove_skipped_blocks(self, **kwargs):
        pass

    def cache_blocks(self, request, num_tokens):
        self.cached.append((request.request_id, num_tokens))

    def free(self, request):
        self.freed.append(request.request_id)


def make_scheduler(vllm_config, kvm=None, policy=SchedulingPolicy.FCFS):
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_manager=kvm or StubKVCacheManager(),
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[object()]),
        policy=policy,
    )


# --------------------------------------------------------------------------
# role-split 契约
# --------------------------------------------------------------------------
def test_role_enum_split():
    assert KVConnectorRole.SCHEDULER.value == 0
    assert KVConnectorRole.WORKER.value == 1


def test_factory_builds_one_instance_per_role(tmp_path):
    cfg = make_vllm_config(tmp_path)
    sched = KVConnectorFactory.create_connector(cfg, KVConnectorRole.SCHEDULER)
    worker = KVConnectorFactory.create_connector(cfg, KVConnectorRole.WORKER)
    # 两份独立实例，各持自己的 role（进程隔离契约）。
    assert sched is not worker
    assert sched.role == KVConnectorRole.SCHEDULER
    assert worker.role == KVConnectorRole.WORKER


def test_factory_unknown_connector_raises(tmp_path):
    cfg = make_vllm_config(tmp_path, connector="NoSuchConnector")
    with pytest.raises(ValueError):
        KVConnectorFactory.create_connector(cfg, KVConnectorRole.SCHEDULER)


def test_example_connector_is_base_v1(tmp_path):
    cfg = make_vllm_config(tmp_path)
    conn = KVConnectorFactory.create_connector(cfg, KVConnectorRole.SCHEDULER)
    assert isinstance(conn, KVConnectorBase_V1)


# --------------------------------------------------------------------------
# 决策侧 ExampleConnector 真实行为
# --------------------------------------------------------------------------
def test_align_to_block_size():
    # (n-1)//block*block —— 与 vLLM ExampleConnector.align_to_block_size 一致
    assert align_to_block_size(9, 4) == 8
    assert align_to_block_size(8, 4) == 4
    assert align_to_block_size(1, 4) == 0


def test_get_num_new_matched_tokens_miss_returns_zero(tmp_path):
    cfg = make_vllm_config(tmp_path, block_size=4)
    conn = ExampleConnector(cfg, KVConnectorRole.SCHEDULER)
    req = Request("r1", prompt_token_ids=list(range(9)))
    # 文件夹不存在 → 未命中 → (0, False)
    ext, load_async = conn.get_num_new_matched_tokens(req, 0)
    assert ext == 0
    assert load_async is False


def test_update_state_after_alloc_registers_load(tmp_path):
    cfg = make_vllm_config(tmp_path)
    conn = ExampleConnector(cfg, KVConnectorRole.SCHEDULER)
    req = Request("r1", prompt_token_ids=list(range(9)))
    conn.update_state_after_alloc(req, blocks=None, num_external_tokens=8)
    assert "r1" in conn._requests_need_load
    # num_external_tokens == 0 时不登记
    req2 = Request("r2", prompt_token_ids=list(range(9)))
    conn.update_state_after_alloc(req2, blocks=None, num_external_tokens=0)
    assert "r2" not in conn._requests_need_load


def test_build_connector_meta_load_and_reset(tmp_path):
    cfg = make_vllm_config(tmp_path)
    conn = ExampleConnector(cfg, KVConnectorRole.SCHEDULER)
    req = Request("r1", prompt_token_ids=list(range(9)))
    conn.update_state_after_alloc(req, blocks=None, num_external_tokens=8)

    new_req = types.SimpleNamespace(req_id="r1", prompt_token_ids=list(range(9)),
                                    block_ids=[[0, 1]])
    sched_out = types.SimpleNamespace(scheduled_new_reqs=[new_req])
    meta = conn.build_connector_meta(sched_out)
    assert isinstance(meta, ExampleConnectorMetadata)
    assert isinstance(meta, KVConnectorMetadata)
    # 已登记 load 的请求 → is_store=False（要 load）
    assert len(meta.requests) == 1
    assert meta.requests[0].is_store is False
    # build_connector_meta 顺带 reset 内部状态
    assert conn._requests_need_load == {}


# --------------------------------------------------------------------------
# 调度集成：f12 闭环
# --------------------------------------------------------------------------
class _AsyncHitConnector(KVConnectorBase_V1):
    """决策侧桩：声称远程命中 8 个 token 且异步加载（load_kv_async=True）。"""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.meta_built = 0

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 8, True  # 命中 8 token，异步

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        self.meta_built += 1
        return ExampleConnectorMetadata()

    # worker-side abstractmethods（本测试不触发）
    def start_load_kv(self, forward_context, **kw):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kw):
        pass

    def wait_for_save(self):
        pass


def _connector_factory_with(monkeypatch, conn_cls):
    orig = KVConnectorFactory.create_connector.__func__

    def fake(cls, config, role, kv_cache_config=None):
        return conn_cls(config, role, kv_cache_config)

    monkeypatch.setattr(KVConnectorFactory, "create_connector",
                        classmethod(fake))
    return orig


def test_async_remote_hit_isolated_to_skipped_waiting(tmp_path, monkeypatch):
    _connector_factory_with(monkeypatch, _AsyncHitConnector)
    cfg = make_vllm_config(tmp_path, block_size=4)
    sched = make_scheduler(cfg)

    req = Request("r1", prompt_token_ids=list(range(16)))
    sched.requests["r1"] = req
    sched.waiting.add_request(req)

    sched.schedule()

    # 异步远程命中 → 请求进 WAITING_FOR_REMOTE_KVS，被隔离到 skipped_waiting，
    # 不进 running（避队头阻塞）。
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req in list(sched.skipped_waiting)
    assert req not in sched.running
    # num_computed_tokens 已被置为 local+external
    assert req.num_computed_tokens == 8


def test_promotion_after_finished_recving(tmp_path, monkeypatch):
    _connector_factory_with(monkeypatch, _AsyncHitConnector)
    cfg = make_vllm_config(tmp_path, block_size=4)
    kvm = StubKVCacheManager()
    sched = make_scheduler(cfg, kvm=kvm)

    req = Request("r1", prompt_token_ids=list(range(16)))
    sched.requests["r1"] = req
    sched.waiting.add_request(req)
    sched.schedule()  # → WAITING_FOR_REMOTE_KVS, in skipped_waiting
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # worker 回传 finished_recving → 入 finished_recving_kv_req_ids
    kv_out = types.SimpleNamespace(finished_recving={"r1"}, finished_sending=None)
    sched._update_from_kv_xfer_finished(kv_out)
    assert "r1" in sched.finished_recving_kv_req_ids

    # 下一步 schedule 遍历 skipped_waiting → 提升回 WAITING（num_preemptions==0）
    sched.schedule()
    assert "r1" not in sched.finished_recving_kv_req_ids
    # block 已被缓存（_update_waiting_for_remote_kv 调 cache_blocks）
    assert any(c[0] == "r1" for c in kvm.cached)
    # 提升后正常调度进 running，开始 decode
    assert req in sched.running
    assert req.status == RequestStatus.RUNNING


def test_promotion_to_preempted_when_preempted_before(tmp_path, monkeypatch):
    _connector_factory_with(monkeypatch, _AsyncHitConnector)
    cfg = make_vllm_config(tmp_path, block_size=4)
    sched = make_scheduler(cfg)

    req = Request("r1", prompt_token_ids=list(range(16)))
    req.num_preemptions = 1  # 曾被抢占
    sched.requests["r1"] = req
    sched.waiting.add_request(req)
    sched.schedule()
    sched._update_from_kv_xfer_finished(
        types.SimpleNamespace(finished_recving={"r1"}, finished_sending=None)
    )
    # 提升判定本身：被抢占过 → PREEMPTED
    promoted = sched._try_promote_blocked_waiting_request(req)
    assert promoted is True
    assert req.status == RequestStatus.PREEMPTED


def test_full_prompt_hit_backs_off_one_token(tmp_path, monkeypatch):
    _connector_factory_with(monkeypatch, _AsyncHitConnector)
    cfg = make_vllm_config(tmp_path, block_size=4)
    sched = make_scheduler(cfg)

    # 整 prompt 命中：num_computed_tokens == num_tokens
    req = Request("r1", prompt_token_ids=list(range(8)))
    req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    req.num_computed_tokens = 8  # == num_tokens
    sched.requests["r1"] = req
    sched.finished_recving_kv_req_ids.add("r1")

    sched._update_waiting_for_remote_kv(req)
    # 回退一个 token 以便采样下一个
    assert req.num_computed_tokens == 7


def test_finished_recving_for_finished_req_frees_blocks(tmp_path, monkeypatch):
    _connector_factory_with(monkeypatch, _AsyncHitConnector)
    cfg = make_vllm_config(tmp_path, block_size=4)
    kvm = StubKVCacheManager()
    sched = make_scheduler(cfg, kvm=kvm)

    # 请求已 finished：finished_recving 走释放块而非加入提升集
    req = Request("r1", prompt_token_ids=list(range(8)))
    req.status = RequestStatus.FINISHED_STOPPED
    sched.requests["r1"] = req
    sched._update_from_kv_xfer_finished(
        types.SimpleNamespace(finished_recving={"r1"}, finished_sending=None)
    )
    assert "r1" not in sched.finished_recving_kv_req_ids
    assert "r1" in kvm.freed


def test_finished_sending_frees_blocks(tmp_path, monkeypatch):
    _connector_factory_with(monkeypatch, _AsyncHitConnector)
    cfg = make_vllm_config(tmp_path, block_size=4)
    kvm = StubKVCacheManager()
    sched = make_scheduler(cfg, kvm=kvm)

    req = Request("r1", prompt_token_ids=list(range(8)))
    req.status = RequestStatus.FINISHED_STOPPED
    sched.requests["r1"] = req
    sched._update_from_kv_xfer_finished(
        types.SimpleNamespace(finished_recving=None, finished_sending={"r1"})
    )
    assert "r1" in kvm.freed


# --------------------------------------------------------------------------
# 双队列选取与阻塞态判定
# --------------------------------------------------------------------------
def test_is_blocked_waiting_status(tmp_path):
    cfg = make_vllm_config(tmp_path)
    sched = make_scheduler(cfg)
    assert sched._is_blocked_waiting_status(RequestStatus.WAITING_FOR_REMOTE_KVS)
    assert not sched._is_blocked_waiting_status(RequestStatus.WAITING)
    assert not sched._is_blocked_waiting_status(RequestStatus.RUNNING)


def test_select_waiting_queue_fcfs_prefers_skipped(tmp_path):
    cfg = make_vllm_config(tmp_path)
    sched = make_scheduler(cfg, policy=SchedulingPolicy.FCFS)
    a = Request("a", prompt_token_ids=[1, 2])
    b = Request("b", prompt_token_ids=[3, 4])
    sched.waiting.add_request(a)
    sched.skipped_waiting.add_request(b)
    # FCFS：skipped_waiting 优先
    q = sched._select_waiting_queue_for_scheduling()
    assert q is sched.skipped_waiting


def test_blocked_request_not_promotable_is_reisolated(tmp_path, monkeypatch):
    _connector_factory_with(monkeypatch, _AsyncHitConnector)
    cfg = make_vllm_config(tmp_path)
    sched = make_scheduler(cfg)
    # 处于 WAITING_FOR_REMOTE_KVS 但 KV 还没到（不在 finished_recving_kv_req_ids）
    req = Request("r1", prompt_token_ids=list(range(8)))
    req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    sched.requests["r1"] = req
    sched.skipped_waiting.add_request(req)

    sched.schedule()
    # 不可提升 → 仍被隔离回 skipped_waiting，主 waiting 不受阻
    assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert req in list(sched.skipped_waiting)


# --------------------------------------------------------------------------
# 请求结束：connector 接管异步释放
# --------------------------------------------------------------------------
def test_connector_finished_delegates_to_request_finished(tmp_path, monkeypatch):
    class _DeferFreeConnector(_AsyncHitConnector):
        def request_finished(self, request, block_ids):
            return True, {"foo": "bar"}  # 接管异步释放

    _connector_factory_with(monkeypatch, _DeferFreeConnector)
    cfg = make_vllm_config(tmp_path)
    sched = make_scheduler(cfg)
    req = Request("r1", prompt_token_ids=list(range(8)))
    delay_free, params = sched._connector_finished(req)
    assert delay_free is True
    assert params == {"foo": "bar"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
