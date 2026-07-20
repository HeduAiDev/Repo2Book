"""测试 EngineCore/DPEngineCoreProc 的弹性 EP 钩子行为。

验证：KV-init 前扩（VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 下 _initialize_kv_caches
用同步得到的统一显存额度而非各自 profiling）、reinitialize_distributed 据
new_dp_size 判 scale_up/scale_down + shutdown 并转非阻塞、run_busy_loop 的
eep 钩子（complete 后清空状态、removing 引擎 SystemExit）。
"""
import pytest

import elastic_state as es
import engine_core_eep as ece
from elastic_state import EEPNotificationType, ReconfigureDistributedRequest, ReconfigureRankType


class FakeParallelConfig:
    def __init__(self, dp_size=1):
        self.data_parallel_size = dp_size
        self.data_parallel_rank = 0
        self.data_parallel_rank_local = 0
        self.data_parallel_master_ip = ""
        self.data_parallel_master_port = 0
        self._data_parallel_master_port_list = None
        self._coord_store_port = 0

    def stateless_init_dp_group(self, return_store=False):
        return (FakeDPGroup(), FakeStore())

    def sync_kv_cache_memory_size(self, group, val):
        return 100 if val == -1 else val


class FakeVllmConfig:
    def __init__(self, dp_size=1):
        self.parallel_config = FakeParallelConfig(dp_size)


class FakeDPGroup:
    def __init__(self, rank=1, size=1):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def barrier(self):
        pass

    def all_reduce_max(self, v):
        return list(v)


class FakeStore:
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

    def delete_key(self, k):
        self.kv.pop(k, None)

    def compare_set(self, k, e, d):
        self.kv[k] = d


class FakeExecutor:
    def __init__(self, kv_specs=None, avail=None):
        self.calls = []
        self._kv_specs = kv_specs if kv_specs is not None else [{"a": 1}]
        self._avail = avail if avail is not None else [42]

    def collective_rpc(self, method, args=()):
        self.calls.append((method, args))

    def get_kv_cache_specs(self):
        return self._kv_specs

    def determine_available_memory(self):
        return self._avail


# ---------------------------------------------------------------------------
# _initialize_kv_caches: 非 eep -> 自己 profiling; eep -> 用同步显存额度
# ---------------------------------------------------------------------------
def test_kv_init_without_eep_uses_profiling(monkeypatch):
    monkeypatch.setattr(ece.envs, "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", False)
    executor = FakeExecutor(avail=[42])
    core = ece.EngineCore(FakeVllmConfig(), executor)
    assert core.available_gpu_memory_for_kv_cache == 42
    assert core.kv_cache_config == [42]


def test_kv_init_with_eep_uses_synced_memory(monkeypatch):
    monkeypatch.setattr(ece.envs, "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", True)
    executor = FakeExecutor(kv_specs=[{"a": 1}], avail=[999])

    # 注入 new 引擎所需的 dp_group/store/字段到将被 __init__ 创建的 core
    # 通过子类挂上 PRE_KV_INIT 所需上下文
    class TestCore(ece.EngineCore):
        def __init__(self, vllm_config, model_executor):
            self.dp_group = FakeDPGroup(rank=1, size=1)
            self.dp_store = FakeStore()
            self.engines_running = False
            self.current_wave = 0
            self.step_counter = 0
            self.sent_notifications = []
            super().__init__(vllm_config, model_executor)

        def _eep_send_engine_core_notification(self, ntype, vllm_config=None):
            self.sent_notifications.append(ntype)

    core = TestCore(FakeVllmConfig(), executor)
    # eep 路径：KV 用 _eep_scale_up_before_kv_init 同步得到的 100，而非 999
    assert core.available_gpu_memory_for_kv_cache == 100
    assert core.kv_cache_config == [100]
    # determine_available_memory 不应被走（available 来自同步）
    assert all(m[0] != "determine_available_memory" for m in [] )


# ---------------------------------------------------------------------------
# reinitialize_distributed: scale_up / scale_down / shutdown 分支
# ---------------------------------------------------------------------------
def make_dp_proc(dp_size=2):
    proc = ece.DPEngineCoreProc.__new__(ece.DPEngineCoreProc)
    proc.vllm_config = FakeVllmConfig(dp_size)
    proc.model_executor = FakeExecutor()
    proc.dp_group = FakeDPGroup(rank=0, size=dp_size)
    proc.dp_store = FakeStore()
    proc.engines_running = False
    proc.current_wave = 0
    proc.step_counter = 0
    proc.available_gpu_memory_for_kv_cache = -1
    proc.eep_scaling_state = None
    proc.process_input_queue_block = True
    proc.sent_notifications = []
    return proc


def test_reinit_scale_up_sets_existing_state_and_nonblocking():
    proc = make_dp_proc(dp_size=2)
    req = ReconfigureDistributedRequest(new_data_parallel_size=3)
    proc.reinitialize_distributed(req)
    assert proc.eep_scaling_state.worker_type == "existing"
    assert proc.eep_scaling_state.scale_type == "scale_up"
    assert proc.process_input_queue_block is False


def test_reinit_scale_down_sets_scale_down():
    proc = make_dp_proc(dp_size=4)
    req = ReconfigureDistributedRequest(new_data_parallel_size=2)
    proc.reinitialize_distributed(req)
    assert proc.eep_scaling_state.scale_type == "scale_down"
    assert proc.eep_scaling_state.worker_type == "existing"


def test_reinit_shutdown_sets_removing():
    proc = make_dp_proc(dp_size=4)
    req = ReconfigureDistributedRequest(
        new_data_parallel_size=2,
        new_data_parallel_rank=ReconfigureRankType.SHUTDOWN_CURRENT_RANK,
    )
    proc.reinitialize_distributed(req)
    assert proc.eep_scaling_state.worker_type == "removing"
    assert proc.eep_scaling_state.scale_type == "scale_down"


# ---------------------------------------------------------------------------
# run_busy_loop eep 钩子: complete 后清状态/恢复阻塞; removing -> SystemExit
# ---------------------------------------------------------------------------
class FakeState:
    def __init__(self, worker_type="existing", complete_after=1):
        self.worker_type = worker_type
        self._calls = 0
        self._complete_after = complete_after

    def progress(self):
        self._calls += 1
        return True

    def is_complete(self):
        return self._calls >= self._complete_after


def test_busy_loop_clears_state_on_complete():
    proc = make_dp_proc()
    proc.eep_scaling_state = FakeState("existing", complete_after=1)
    proc.process_input_queue_block = False
    # 用一次性 shutdown 钩子让 loop 跑一轮后退出
    runs = {"n": 0}
    proc._handle_shutdown = lambda: runs["n"] == 0 and (runs.__setitem__("n", 1) or True)
    proc._process_input_queue = lambda: None
    proc._process_engine_step = lambda: None
    with pytest.raises(SystemExit):
        proc.run_busy_loop()
    # 完成后状态被清空且恢复阻塞
    assert proc.eep_scaling_state is None
    assert proc.process_input_queue_block is True


def test_busy_loop_removing_engine_raises_systemexit():
    proc = make_dp_proc()
    proc.eep_scaling_state = FakeState("removing", complete_after=1)
    proc._handle_shutdown = lambda: True
    proc._process_input_queue = lambda: None
    proc._process_engine_step = lambda: None
    with pytest.raises(SystemExit):
        proc.run_busy_loop()


# ---------------------------------------------------------------------------
# eep_handle_engine_core_notification: 字符串转枚举后转发给状态机
# ---------------------------------------------------------------------------
def test_handle_notification_forwards_to_state():
    proc = make_dp_proc()
    received = []

    class S:
        def handle_notification(self, nt):
            received.append(nt)

    proc.eep_scaling_state = S()
    proc.eep_handle_engine_core_notification("NEW_CORE_ENGINES_INIT_READY")
    assert received == [EEPNotificationType.NEW_CORE_ENGINES_INIT_READY]
