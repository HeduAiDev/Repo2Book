"""测试弹性 EP 扩缩状态机的真实 vLLM 可观察行为。

不 import vllm —— 用与真实 dp_group/dp_store/model_executor 接口一致的替身
驱动 ElasticEPScalingState，验证状态推进顺序、空转语义、握手、barrier、
scale_down 的 SHUTDOWN_COMPLETE 等与 vllm/distributed/elastic_ep/elastic_state.py
一致的行为。
"""
import weakref

import pytest

from elastic_state import (
    EEPNotificationType,
    ElasticEPScalingState,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    ScaleDownRemainingEngineState,
    ScaleDownRemovingEngineState,
    ScaleUpExistingEngineState,
    ScaleUpNewEngineState,
)


# ---------------------------------------------------------------------------
# 替身：模拟真实 DP group / TCPStore / executor 的最小可观察接口
# ---------------------------------------------------------------------------
class FakeDPGroup:
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size
        self.barriered = 0
        self.destroyed = False

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def barrier(self):
        self.barriered += 1

    def destroy(self):
        self.destroyed = True

    def all_reduce_max(self, values):
        # 单进程语义：MAX 等于自身（真实是跨组 all_reduce MAX）
        return list(values)


class FakeStore:
    """模拟 TCPStore：set/get/check/add/delete_key/compare_set。"""

    def __init__(self):
        self.kv = {}

    def set(self, k, v):
        self.kv[k] = v

    def get(self, k):
        return self.kv[k]

    def check(self, keys):
        return all(k in self.kv for k in keys)

    def add(self, k, n):
        self.kv[k] = int(self.kv.get(k, 0)) + n
        return self.kv[k]

    def delete_key(self, k):
        self.kv.pop(k, None)

    def compare_set(self, k, expected, desired):
        self.kv[k] = desired


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def collective_rpc(self, method, args=()):
        self.calls.append((method, args))


class FakeParallelConfig:
    def __init__(self, dp_size=1):
        self.data_parallel_size = dp_size
        self.data_parallel_rank = 0
        self.data_parallel_rank_local = 0
        self.data_parallel_master_ip = ""
        self.data_parallel_master_port = 0
        self._data_parallel_master_port_list = None
        self._coord_store_port = 0
        self._new_group = None

    def stateless_init_dp_group(self, return_store=False):
        grp = self._new_group or FakeDPGroup(rank=0, size=1)
        store = FakeStore()
        return (grp, store)

    def sync_kv_cache_memory_size(self, group, val):
        # 真实是从全 DP 组同步统一显存额度；单进程返回一个正数
        return 100 if val == -1 else val


class FakeVllmConfig:
    def __init__(self, dp_size=1):
        self.parallel_config = FakeParallelConfig(dp_size)


class FakeEngineCore:
    """模拟 DPEngineCoreProc 上被状态机读写的字段。"""

    def __init__(self, dp_group, dp_store):
        self.dp_group = dp_group
        self.dp_store = dp_store
        self.dp_rank = dp_group.rank()
        self.available_gpu_memory_for_kv_cache = -1
        self.engines_running = False
        self.current_wave = 0
        self.step_counter = 0
        self.sent_notifications = []

    def _eep_send_engine_core_notification(self, ntype, vllm_config=None):
        self.sent_notifications.append(ntype)


def make_existing_scale_up(rank=0, size=1):
    dp_group = FakeDPGroup(rank=rank, size=size)
    dp_store = FakeStore()
    engine_core = FakeEngineCore(dp_group, dp_store)
    # 存在引擎是"已在运行"的引擎：其 available_gpu_memory_for_kv_cache 在原始
    # 启动 _initialize_kv_caches(core.py:L251) 时已被设为正值。SYNC_KV_CACHE_MEMORY_SIZE
    # 阶段的 assert >0（elastic_state.py:L496）正是依赖这一点把额度同步给新引擎。
    engine_core.available_gpu_memory_for_kv_cache = 100
    executor = FakeExecutor()
    vllm_config = FakeVllmConfig(dp_size=size)
    new_pc = FakeParallelConfig(dp_size=size + 1)
    # 单进程测试：把"本引擎在新 DP 组中"建模为新组里唯一在跑的 rank0（size=1），
    # 这样 SWITCH_AND_PREPARE 后的新组 EPLB barrier（等 new_dp_group.size() 到齐）
    # 由本引擎自身满足；真实多进程下其余新 rank 会各自 +count/set arrival。
    new_pc._new_group = FakeDPGroup(rank=0, size=1)
    reconfig = ReconfigureDistributedRequest(new_data_parallel_size=size + 1)
    state = ElasticEPScalingState(
        model_executor=executor,
        engine_core=engine_core,
        vllm_config=vllm_config,
        new_parallel_config=new_pc,
        worker_type="existing",
        scale_type="scale_up",
        reconfig_request=reconfig,
    )
    return state, engine_core, executor, dp_store


# ---------------------------------------------------------------------------
# __init__: 初始状态按 worker_type × scale_type 选取
# ---------------------------------------------------------------------------
def test_init_existing_scaleup_starts_at_wait_new_core_engines_init():
    state, *_ = make_existing_scale_up()
    assert state.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
    # existing/new dp group/store 各就位
    assert state.old_dp_group is not None and state.new_dp_group is None


def test_init_new_scaleup_starts_at_pre_kv_init():
    dp_group = FakeDPGroup(rank=1, size=2)
    dp_store = FakeStore()
    engine_core = FakeEngineCore(dp_group, dp_store)
    state = ElasticEPScalingState(
        model_executor=FakeExecutor(),
        engine_core=engine_core,
        vllm_config=FakeVllmConfig(),
        new_parallel_config=FakeParallelConfig(),
        worker_type="new",
        scale_type="scale_up",
    )
    assert state.state == ScaleUpNewEngineState.PRE_KV_INIT
    assert state.new_dp_group is not None and state.old_dp_group is None


def test_init_removing_scaledown_starts_at_prepare():
    dp_group = FakeDPGroup()
    state = ElasticEPScalingState(
        model_executor=FakeExecutor(),
        engine_core=FakeEngineCore(dp_group, FakeStore()),
        vllm_config=FakeVllmConfig(),
        new_parallel_config=FakeParallelConfig(),
        worker_type="removing",
        scale_type="scale_down",
    )
    assert state.state == ScaleDownRemovingEngineState.PREPARE


# ---------------------------------------------------------------------------
# WAIT_* 是被动空转：progress() 返回 False 不推进；靠 handle_notification 推进
# ---------------------------------------------------------------------------
def test_wait_state_is_passive_spin():
    state, *_ = make_existing_scale_up()
    assert state.progress() is False
    assert state.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT


def test_handle_notification_init_ready_advances_and_counts():
    state, engine_core, _, dp_store = make_existing_scale_up()
    state.handle_notification(EEPNotificationType.NEW_CORE_ENGINES_INIT_READY)
    assert state.state == ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS
    assert int(dp_store.get("eep_barrier_engine_count")) == 1


def test_handle_notification_new_must_not_be_new_worker():
    dp_group = FakeDPGroup(rank=1, size=2)
    state = ElasticEPScalingState(
        model_executor=FakeExecutor(),
        engine_core=FakeEngineCore(dp_group, FakeStore()),
        vllm_config=FakeVllmConfig(),
        new_parallel_config=FakeParallelConfig(),
        worker_type="new",
        scale_type="scale_up",
    )
    with pytest.raises(AssertionError):
        state.handle_notification(EEPNotificationType.NEW_CORE_ENGINES_INIT_READY)


# ---------------------------------------------------------------------------
# scale_up existing 引擎 9 状态全程推进（size=1 单进程，barrier 立即满足）
# ---------------------------------------------------------------------------
def test_existing_engine_full_progress_order():
    state, engine_core, executor, dp_store = make_existing_scale_up(rank=0, size=1)
    # 模拟收到 INIT_READY 通知
    state.handle_notification(EEPNotificationType.NEW_CORE_ENGINES_INIT_READY)

    seen = [state.state]
    # 逐轮 progress 直到 COMPLETE；空转/未满足返回 False 时注入推进条件
    for _ in range(50):
        if state.is_complete():
            break
        # WAIT_WEIGHTS 需要 weights-init 通知才推进；通知会把状态推进到
        # TRANSFER_WEIGHTS（由 handle_notification 而非 progress），记录之以免
        # 紧接的 progress() 在同一轮把它消费成 SYNC_* 而漏记。
        if state.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT:
            state.handle_notification(
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
            )
            if state.state != seen[-1]:
                seen.append(state.state)
        state.progress()
        if state.state != seen[-1]:
            seen.append(state.state)

    assert state.is_complete()
    # 验证经过的关键状态有序覆盖 9 态骨架
    assert ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS in seen
    assert ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING in seen
    assert ScaleUpExistingEngineState.TRANSFER_WEIGHTS in seen
    assert ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE in seen
    assert ScaleUpExistingEngineState.SWITCH_AND_PREPARE in seen
    assert ScaleUpExistingEngineState.EPLB_RESHUFFLE in seen
    assert seen[-1] == ScaleUpExistingEngineState.COMPLETE


def test_switch_and_prepare_destroys_old_group_and_switches_engine_core():
    state, engine_core, executor, dp_store = make_existing_scale_up(rank=0, size=1)
    old_group = engine_core.dp_group
    # 跳到 SWITCH_AND_PREPARE
    state.handle_notification(EEPNotificationType.NEW_CORE_ENGINES_INIT_READY)
    for _ in range(50):
        if state.state == ScaleUpExistingEngineState.SWITCH_AND_PREPARE:
            break
        if state.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT:
            state.handle_notification(
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
            )
        state.progress()
    assert state.state == ScaleUpExistingEngineState.SWITCH_AND_PREPARE
    state.progress()  # 执行 _switch_and_prepare
    # 旧组被销毁，engine_core.dp_group 切到新组，发 RECONFIGURE_FINISHED
    assert old_group.destroyed
    assert engine_core.dp_group is state.new_dp_group
    assert EEPNotificationType.RECONFIGURE_FINISHED in engine_core.sent_notifications
    assert ("switch_and_prepare",) == executor.calls[-1][1]


# ---------------------------------------------------------------------------
# scale_up new 引擎: PRE_KV_INIT 拿统一显存额度 + run_pre_kv_init_states
# ---------------------------------------------------------------------------
def make_new_scale_up(rank=1, size=2):
    dp_group = FakeDPGroup(rank=rank, size=size)
    dp_store = FakeStore()
    engine_core = FakeEngineCore(dp_group, dp_store)
    executor = FakeExecutor()
    state = ElasticEPScalingState(
        model_executor=executor,
        engine_core=engine_core,
        vllm_config=FakeVllmConfig(),
        new_parallel_config=FakeParallelConfig(),
        worker_type="new",
        scale_type="scale_up",
    )
    return state, engine_core, executor, dp_store


def test_new_engine_pre_kv_init_syncs_memory_and_sends_weights_ready():
    state, engine_core, executor, dp_store = make_new_scale_up()
    assert engine_core.available_gpu_memory_for_kv_cache == -1
    ok = state.progress()  # PRE_KV_INIT -> PREPARE
    assert ok is True
    assert state.state == ScaleUpNewEngineState.PREPARE
    # 发了 WEIGHTS_INIT_READY、调了 receive_weights/prepare_new_worker、显存被同步
    assert (
        EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
        in engine_core.sent_notifications
    )
    methods = [c[1][0] for c in executor.calls]
    assert "receive_weights" in methods
    assert "prepare_new_worker" in methods
    assert engine_core.available_gpu_memory_for_kv_cache == 100


def test_run_pre_kv_init_states_lands_on_prepare():
    state, engine_core, *_ = make_new_scale_up()
    state.run_pre_kv_init_states()
    assert state.state == ScaleUpNewEngineState.PREPARE


def test_new_engine_prepare_pulls_wave_state_via_all_reduce_max():
    state, engine_core, executor, dp_store = make_new_scale_up()
    state.progress()  # -> PREPARE
    # 在新组里预置 wave 状态由 all_reduce_max 拿（替身回传输入）
    state.new_dp_group.all_reduce_max = lambda v: [1, 7, 3]
    state.progress()  # PREPARE -> EPLB_RESHUFFLE
    assert engine_core.engines_running is True
    assert engine_core.current_wave == 7
    assert engine_core.step_counter == 3
    assert state.state == ScaleUpNewEngineState.EPLB_RESHUFFLE


# ---------------------------------------------------------------------------
# scale_down 余留 / 移除引擎
# ---------------------------------------------------------------------------
def make_scale_down(worker_type, rank=0, size=2):
    dp_group = FakeDPGroup(rank=rank, size=size)
    dp_store = FakeStore()
    engine_core = FakeEngineCore(dp_group, dp_store)
    executor = FakeExecutor()
    vllm_config = FakeVllmConfig(dp_size=size)
    new_pc = FakeParallelConfig(dp_size=size - 1)
    new_pc._new_group = FakeDPGroup(rank=rank, size=size - 1)
    reconfig = ReconfigureDistributedRequest(new_data_parallel_size=size - 1)
    state = ElasticEPScalingState(
        model_executor=executor,
        engine_core=engine_core,
        vllm_config=vllm_config,
        new_parallel_config=new_pc,
        worker_type=worker_type,
        scale_type="scale_down",
        reconfig_request=reconfig,
    )
    return state, engine_core, executor, dp_store


def test_remaining_engine_completes_in_compact_sequence():
    state, engine_core, executor, _ = make_scale_down("existing", rank=0, size=1)
    assert state.state == ScaleDownRemainingEngineState.PREPARE
    for _ in range(20):
        if state.is_complete():
            break
        state.progress()
    assert state.is_complete()
    assert state.state == ScaleDownRemainingEngineState.COMPLETE
    methods = [c[1][0] for c in executor.calls]
    # 余留引擎在 EPLB_RESHUFFLE 一态里连做 reshuffle→建组→switch
    assert "perform_scale_down_eplb_reshuffle" in methods
    assert "switch_and_prepare" in methods


def _seed_other_ranks_arrived(store, barrier_name, group_size, self_rank):
    """模拟"DP 组中其它 rank 已抵达该 barrier"：在 TCPStore 预置它们的
    arrival key。_progress_removing_engine / _progress_new_engine 的 EPLB_RESHUFFLE
    断言 rank>0 且等待全组到齐，单进程测试需让其余 rank 先到，barrier 方可完成
    （与真实多进程下其它 EngineCore set arrival key 行为一致）。"""
    barrier_id = f"eep_barrier_{barrier_name}"
    for i in range(group_size):
        if i != self_rank:
            store.set(f"arrival_{barrier_id}_{i}", b"1")


def test_removing_engine_emits_shutdown_complete():
    # 被移除引擎是 rank>0（_progress_removing_engine 断言要求），size=2 组里
    # 另一 rank(0) 是余留引擎；预置其 barrier 到达使单进程测试可推进。
    state, engine_core, executor, dp_store = make_scale_down("removing", rank=1, size=2)
    # eep_barrier_engine_count 也需达到 group size：另一 rank 在 PREPARE 时 +1
    dp_store.add("eep_barrier_engine_count", 1)
    _seed_other_ranks_arrived(dp_store, "eplb_reshuffle", group_size=2, self_rank=1)
    assert state.state == ScaleDownRemovingEngineState.PREPARE
    for _ in range(20):
        if state.is_complete():
            break
        state.progress()
    assert state.is_complete()
    assert EEPNotificationType.SHUTDOWN_COMPLETE in engine_core.sent_notifications
    methods = [c[1][0] for c in executor.calls]
    assert "switch_and_remove" in methods


# ---------------------------------------------------------------------------
# _staged_barrier 两阶段：首次超时 -> compare_set sync_key 并返回 False
# ---------------------------------------------------------------------------
def test_staged_barrier_first_timeout_sets_sync_key_returns_false():
    state, engine_core, _, dp_store = make_existing_scale_up(rank=0, size=2)
    # group size=2 但只有 rank0 到达 -> 首次 barrier 超时
    # 把 timeout 缩短以快速触发
    import elastic_state as es

    orig = es.timedelta

    def fast_timedelta(seconds=0):
        return orig(seconds=0.05)

    es.timedelta = fast_timedelta
    try:
        ok = state._staged_barrier(use_new_group=False, barrier_name="t")
    finally:
        es.timedelta = orig
    assert ok is False
    # sync_key 被置位，下次 barrier 将用 timeout=None
    assert dp_store.check(["eep_barrier_t_sync"])


def test_update_parallel_config_writes_new_dp_size():
    state, engine_core, _, _ = make_existing_scale_up(rank=0, size=1)
    state._update_parallel_config()
    assert state.vllm_config.parallel_config.data_parallel_size == 2


# ---------------------------------------------------------------------------
# weakref: engine_core 被回收后访问应抛 RuntimeError
# ---------------------------------------------------------------------------
def test_engine_core_garbage_collected_raises():
    dp_group = FakeDPGroup()
    ec = FakeEngineCore(dp_group, FakeStore())
    state = ElasticEPScalingState(
        model_executor=FakeExecutor(),
        engine_core=ec,
        vllm_config=FakeVllmConfig(),
        new_parallel_config=FakeParallelConfig(),
        worker_type="existing",
        scale_type="scale_up",
        reconfig_request=ReconfigureDistributedRequest(new_data_parallel_size=2),
    )
    ref = weakref.ref(ec)
    del ec
    import gc

    gc.collect()
    if ref() is None:
        with pytest.raises(RuntimeError):
            _ = state.engine_core
