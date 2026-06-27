# 精简版（只做减法）—— 弹性 EP 扩缩状态机
#
# 对应真实源码：vllm/distributed/elastic_ep/elastic_state.py
# 源码 pin：f3fef123
#
# 本文件是 ElasticEPScalingState 确定性状态机的忠实子集：与真实 vLLM
# 同名、同结构、同控制流，只删不增。删除处均以 `# SUBTRACTED:` 标注。
#
# 控制流（progress/4 个 _progress_*/_staged_barrier/handle_notification/
# is_complete/_switch_and_prepare 等）逐字保留；被裁掉的仅是真实代码里
# 委派给 worker 侧（elastic_ep_execute）与 torch.distributed 集合通信的
# 实现体——它们在真实源码里也是经 model_executor.collective_rpc / dp_group
# 这些**外部对象**完成的，本精简版照样经这些对象调用，由测试注入可观察的
# 替身来驱动状态机推进，因此控制流与真实源码一致。
#
# SUBTRACTED: `import torch.distributed`、`from vllm.config import ParallelConfig`、
#   `from vllm.distributed import sched_yield, stateless_destroy_torch_distributed_process_group`、
#   `from vllm.logger import init_logger`、`from vllm.v1.engine import (...)`、
#   `from vllm.v1.engine.core import DPEngineCoreProc`、TYPE_CHECKING 块
#   ——真实导入触 CUDA/torch.distributed/vllm，本章只读状态机控制流；
#   原 vllm/distributed/elastic_ep/elastic_state.py:L1-L28。
#   下面用纯枚举/占位重建被引用到的少量符号（名称/取值与真实一致），
#   不杜撰任何 vLLM 没有的抽象。
import enum
import time
import weakref
from datetime import timedelta
from typing import Literal


# SOURCE: vllm/distributed (sched_yield) — 经 elastic_state.py:L13, L180 调用
# SUBTRACTED: 真实 sched_yield 让出 GIL/调度（vllm/distributed）；精简版用
#   time.sleep(0) 等价让步，不改变 barrier 轮询语义。原引用 elastic_state.py:L13。
def sched_yield():
    # SOURCE: vllm/distributed (sched_yield) — 经 elastic_state.py:L13, L180 调用
    time.sleep(0)


# SOURCE: vllm/distributed (stateless_destroy_torch_distributed_process_group) — 经 elastic_state.py:L510 调用
# SUBTRACTED: 真实 stateless_destroy_torch_distributed_process_group 销毁
#   torch 进程组（vllm/distributed）；精简版调用其在 dp_group 上的等价钩子，
#   保留"销毁旧组"这一控制流步骤。原引用 elastic_state.py:L14, L510。
def stateless_destroy_torch_distributed_process_group(group):
    # SOURCE: vllm/distributed (stateless_destroy_torch_distributed_process_group) — 经 elastic_state.py:L510 调用
    if group is not None and hasattr(group, "destroy"):
        group.destroy()


# SUBTRACTED: 真实 EEPNotificationType 定义在 vllm/v1/engine/__init__.py，
#   此处复刻同名同取值枚举（四个成员），供状态机握手使用。
#   原引用 elastic_state.py:L17-L21。
class EEPNotificationType(enum.Enum):
    # SOURCE: vllm/v1/engine/__init__.py (EEPNotificationType)
    NEW_CORE_ENGINES_INIT_READY = "NEW_CORE_ENGINES_INIT_READY"
    NEW_CORE_ENGINES_WEIGHTS_INIT_READY = "NEW_CORE_ENGINES_WEIGHTS_INIT_READY"
    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"


# SUBTRACTED: 真实 ReconfigureRankType 定义在 vllm/v1/engine/__init__.py；
#   此处复刻 reinitialize_distributed / _update_parallel_config 用到的两个哨兵。
#   原引用 elastic_state.py:L17-L21。
class ReconfigureRankType:
    # SOURCE: vllm/v1/engine/__init__.py (ReconfigureRankType)
    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2


# SUBTRACTED: 真实 ReconfigureDistributedRequest 是 msgspec.Struct
#   （vllm/v1/engine/__init__.py），携带新 DP 维度与 master ip/port 等。
#   精简版用等价的轻量数据类，只保留状态机读取到的字段。
class ReconfigureDistributedRequest:
    # SOURCE: vllm/v1/engine/__init__.py (ReconfigureDistributedRequest)
    def __init__(
        self,
        new_data_parallel_size,
        new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_master_ip="",
        new_data_parallel_master_port=0,
        new_data_parallel_master_port_list=None,
        coord_store_port=0,
    ):
        self.new_data_parallel_size = new_data_parallel_size
        self.new_data_parallel_rank = new_data_parallel_rank
        self.new_data_parallel_rank_local = new_data_parallel_rank_local
        self.new_data_parallel_master_ip = new_data_parallel_master_ip
        self.new_data_parallel_master_port = new_data_parallel_master_port
        self.new_data_parallel_master_port_list = new_data_parallel_master_port_list
        self.coord_store_port = coord_store_port


WorkerType = Literal["existing", "new", "removing"]


# SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L33-L42
class ScaleUpExistingEngineState(enum.IntEnum):
    WAIT_NEW_CORE_ENGINES_INIT = 0
    CREATE_STANDBY_GROUPS = 1
    TRANSFER_EXPERT_MAPPING = 2
    WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT = 3
    TRANSFER_WEIGHTS = 4
    SYNC_KV_CACHE_MEMORY_SIZE = 5
    SWITCH_AND_PREPARE = 6
    EPLB_RESHUFFLE = 7
    COMPLETE = 8


# SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L45-L49
class ScaleUpNewEngineState(enum.IntEnum):
    PRE_KV_INIT = 0
    PREPARE = 1
    EPLB_RESHUFFLE = 2
    COMPLETE = 3


# SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L52-L56
class ScaleDownRemainingEngineState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    SWITCH_AND_PREPARE = 2
    COMPLETE = 3


# SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L59-L62
class ScaleDownRemovingEngineState(enum.IntEnum):
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    COMPLETE = 2


# SUBTRACTED: `EngineState: TypeAlias = (四枚举的联合)` —— 仅类型别名，
#   不影响运行时控制流。原 vllm/.../elastic_state.py:L65-L70。


# SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L73-L79
class _BarrierTimeoutError(RuntimeError):
    """
    Exception raised for timeout
    in the first stage of our two-staged
    TCPStore based barrier to synchronize the
    execution of all engines in the DP group.
    """


# SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L82
class ElasticEPScalingState:
    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L83-L117
    def __init__(
        self,
        model_executor,
        engine_core,
        vllm_config,
        new_parallel_config,
        worker_type: WorkerType,
        scale_type: Literal["scale_up", "scale_down"],
        reconfig_request=None,
    ):
        self.model_executor_ref = weakref.ref(model_executor)
        self.engine_core_ref = weakref.ref(engine_core)
        self.vllm_config = vllm_config
        self.old_dp_group = self.engine_core.dp_group if worker_type != "new" else None
        self.old_dp_store = self.engine_core.dp_store if worker_type != "new" else None
        self.new_parallel_config = new_parallel_config
        self.new_dp_group = self.engine_core.dp_group if worker_type == "new" else None
        self.new_dp_store = self.engine_core.dp_store if worker_type == "new" else None
        self.worker_type = worker_type
        self.scale_type = scale_type
        self.reconfig_request = reconfig_request

        self.state: enum.IntEnum
        if scale_type == "scale_up":
            self.state = (
                ScaleUpNewEngineState.PRE_KV_INIT
                if worker_type == "new"
                else ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
            )
        else:
            self.state = (
                ScaleDownRemovingEngineState.PREPARE
                if worker_type == "removing"
                else ScaleDownRemainingEngineState.PREPARE
            )

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L119-L124
    @property
    def model_executor(self):
        # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L119-L124
        model_executor = self.model_executor_ref()
        if model_executor is None:
            raise RuntimeError("Model executor has been garbage collected")
        return model_executor

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L126-L131
    @property
    def engine_core(self):
        # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L126-L131
        engine_core = self.engine_core_ref()
        if engine_core is None:
            raise RuntimeError("Engine core has been garbage collected")
        return engine_core

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L133-L144
    def progress(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self._progress_new_engine()
                if self.worker_type == "new"
                else self._progress_existing_engine()
            )
        return (
            self._progress_removing_engine()
            if self.worker_type == "removing"
            else self._progress_remaining_engine()
        )

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L146-L150
    def run_pre_kv_init_states(self) -> None:
        assert self.scale_type == "scale_up" and self.worker_type == "new"
        assert self.state == ScaleUpNewEngineState.PRE_KV_INIT
        assert self.progress()
        assert self.state == ScaleUpNewEngineState.PREPARE

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L152-L180
    def _execute_tcp_store_barrier(
        self, dp_store, group_rank, group_size, barrier_id, timeout=None
    ):
        arrival_key = f"arrival_{barrier_id}_{group_rank}"
        dp_store.set(arrival_key, b"1")

        start_time = time.time()
        processes_arrived: set[int] = set()

        while len(processes_arrived) < group_size:
            if (
                timeout is not None
                and time.time() - start_time > timeout.total_seconds()
            ):
                raise _BarrierTimeoutError(
                    f"Barrier timed out after {timeout.total_seconds()} seconds"
                )

            for i in range(group_size):
                if i in processes_arrived:
                    continue

                key = f"arrival_{barrier_id}_{i}"
                present = dp_store.check([key])
                if present:
                    processes_arrived.add(i)

            if len(processes_arrived) < group_size:
                sched_yield()

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L182-L225
    def _staged_barrier(self, use_new_group: bool, barrier_name: str) -> bool:
        """
        Execute a two-staged barrier to synchronize all engines in the DP group.

        Some DP EngineCores may receive the reconfiguration notifications
        later than others, and already proceed to engine step (model forward)
        in the busy loop.
        In this case, EngineCores that already proceed to reconfiguration
        should skip reconfiguration and execute model forward for one more
        step, so in the next step, all EngineCores will be synchronized.
        We use a two-staged barrier to achieve this. The first time each
        EngineCore executes the barrier, if a timeout is reached before the
        barrier completes, that means some EngineCores have already entered
        engine step. The EngineCores that timed out will then proceed to
        engine step, and will synchronize with the other EngineCores in the
        next step with a barrier without timeout.
        """
        dp_group = self.new_dp_group if use_new_group else self.old_dp_group
        dp_store = self.new_dp_store if use_new_group else self.old_dp_store
        assert dp_group is not None and dp_store is not None

        group_rank = dp_group.rank()
        group_size = dp_group.size()
        barrier_id = f"eep_barrier_{barrier_name}"
        sync_key = f"{barrier_id}_sync"

        # TODO(yongji): figure out appropriate timeout for the barrier
        timeout = None if dp_store.check([sync_key]) else timedelta(seconds=5)

        try:
            self._execute_tcp_store_barrier(
                dp_store, group_rank, group_size, barrier_id, timeout=timeout
            )
            # SUBTRACTED: torch.distributed.barrier(dp_group) —— TCPStore barrier 通过后
            #   的进程组栅栏；委派给 dp_group.barrier()，保留"再做一次组栅栏"控制流。
            #   原 elastic_state.py:L215。
            dp_group.barrier()
            if group_rank == 0:
                dp_store.delete_key(sync_key)
                for i in range(group_size):
                    dp_store.delete_key(f"arrival_{barrier_id}_{i}")
            return True
        except _BarrierTimeoutError as e:
            if timeout is None:
                raise RuntimeError("Unexpected timeout encountered") from e
            dp_store.compare_set(sync_key, "", b"1")
            return False

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L227-L307
    def _progress_existing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT:
            return False

        elif state == ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS:
            # NOTE(yongji): wait for all existing workers to receive the request
            if (
                int(self.old_dp_store.get("eep_barrier_engine_count"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=False, barrier_name="create_standby_groups"
            ):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("eep_barrier_engine_count")
            self._create_standby_groups()
            self.state = ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING
            return True

        elif state == ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING:
            self._transfer_expert_mapping()
            self.state = ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
            return True

        elif state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT:
            return False

        elif state == ScaleUpExistingEngineState.TRANSFER_WEIGHTS:
            if (
                int(self.old_dp_store.get("eep_barrier_engine_count"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=False, barrier_name="transfer_weights"
            ):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("eep_barrier_engine_count")
            self._transfer_weights()
            self.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
            return True

        elif state == ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE:
            self._sync_kv_cache_memory_size()
            self.state = ScaleUpExistingEngineState.SWITCH_AND_PREPARE
            return True

        elif state == ScaleUpExistingEngineState.SWITCH_AND_PREPARE:
            self._switch_and_prepare()
            self.state = ScaleUpExistingEngineState.EPLB_RESHUFFLE
            assert self.new_dp_store is not None
            self.new_dp_store.add("eep_barrier_engine_count", 1)
            return True

        elif state == ScaleUpExistingEngineState.EPLB_RESHUFFLE:
            assert self.new_dp_group is not None and self.new_dp_store is not None
            if (
                int(self.new_dp_store.get("eep_barrier_engine_count"))
                < self.new_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=True, barrier_name="eplb_reshuffle"
            ):
                return False
            if self.new_dp_group.rank() == 0:
                self.new_dp_store.delete_key("eep_barrier_engine_count")
            self._eplb_reshuffle()
            self.state = ScaleUpExistingEngineState.COMPLETE
            self._update_parallel_config()
            return True

        else:
            assert self.state == ScaleUpExistingEngineState.COMPLETE
            return True

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L309-L361
    def _progress_new_engine(self) -> bool:
        state = self.state
        assert self.new_dp_group is not None and self.new_dp_store is not None

        if state == ScaleUpNewEngineState.PRE_KV_INIT:
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
            )
            self.model_executor.collective_rpc(
                "elastic_ep_execute", args=("receive_weights",)
            )
            self.engine_core.available_gpu_memory_for_kv_cache = (
                self.new_parallel_config.sync_kv_cache_memory_size(self.new_dp_group, -1)
            )
            self.model_executor.collective_rpc(
                "elastic_ep_execute", args=("prepare_new_worker",)
            )
            self.state = ScaleUpNewEngineState.PREPARE
            return True

        elif state == ScaleUpNewEngineState.PREPARE:
            # SUBTRACTED: torch.tensor([0,0,0]) + torch.distributed.all_reduce(MAX)
            #   从全 DP 组取 [engines_running, current_wave, step_counter] 的最大值；
            #   委派给 new_dp_group.all_reduce_max(...)，保留"all_reduce MAX 拿统一
            #   wave 状态"控制流（与 ch21 DP wave 接缝）。原 elastic_state.py:L330-336。
            data = self.new_dp_group.all_reduce_max([0, 0, 0])
            self.engine_core.engines_running = bool(data[0])
            self.engine_core.current_wave = int(data[1])
            self.engine_core.step_counter = int(data[2])
            self.state = ScaleUpNewEngineState.EPLB_RESHUFFLE
            self.new_dp_store.add("eep_barrier_engine_count", 1)
            return True

        elif state == ScaleUpNewEngineState.EPLB_RESHUFFLE:
            if (
                int(self.new_dp_store.get("eep_barrier_engine_count"))
                < self.new_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=True, barrier_name="eplb_reshuffle"
            ):
                return False
            assert self.new_dp_group.rank() > 0
            self._eplb_reshuffle()
            self.state = ScaleUpNewEngineState.COMPLETE
            return True

        else:
            assert self.state == ScaleUpNewEngineState.COMPLETE
            return True

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L363-L401
    def _progress_remaining_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleDownRemainingEngineState.PREPARE:
            self.state = ScaleDownRemainingEngineState.EPLB_RESHUFFLE
            self.old_dp_store.add("eep_barrier_engine_count", 1)
            return True

        elif state == ScaleDownRemainingEngineState.EPLB_RESHUFFLE:
            if (
                int(self.old_dp_store.get("eep_barrier_engine_count"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=False, barrier_name="eplb_reshuffle"
            ):
                return False
            if self.old_dp_group.rank() == 0:
                self.old_dp_store.delete_key("eep_barrier_engine_count")
            self._eplb_reshuffle_before_scale_down()
            self.state = ScaleDownRemainingEngineState.SWITCH_AND_PREPARE
            # NOTE(yongji): currently, after EPLB reshuffle
            # that redistributes experts to remaining workers, workers
            # to be removed will immediately initiate shutdown;
            # existing workers can no longer execute forward steps using
            # the old setup. In the future, we may keep
            # the removing workers alive a bit longer,
            # e.g., to drain in-batch requests.
            self._create_standby_groups()
            self._switch_and_prepare()
            self._update_parallel_config()
            self.state = ScaleDownRemainingEngineState.COMPLETE
            return True

        else:
            assert self.state == ScaleDownRemainingEngineState.COMPLETE
            return True

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L403-L433
    def _progress_removing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None and self.old_dp_store is not None

        if state == ScaleDownRemovingEngineState.PREPARE:
            self.state = ScaleDownRemovingEngineState.EPLB_RESHUFFLE
            self.old_dp_store.add("eep_barrier_engine_count", 1)
            return True

        if state == ScaleDownRemovingEngineState.EPLB_RESHUFFLE:
            if (
                int(self.old_dp_store.get("eep_barrier_engine_count"))
                < self.old_dp_group.size()
            ):
                return False
            if not self._staged_barrier(
                use_new_group=False, barrier_name="eplb_reshuffle"
            ):
                return False
            assert self.old_dp_group.rank() > 0
            self._eplb_reshuffle_before_scale_down()
            self._switch_and_remove()
            self.state = ScaleDownRemovingEngineState.COMPLETE
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.SHUTDOWN_COMPLETE
            )
            return True

        else:
            assert self.state == ScaleDownRemovingEngineState.COMPLETE
            return True

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L435-L450
    def handle_notification(self, notification_type: EEPNotificationType):
        assert self.worker_type != "new"
        assert self.old_dp_store is not None
        if (
            notification_type == EEPNotificationType.NEW_CORE_ENGINES_INIT_READY
            and self.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
        ):
            self.old_dp_store.add("eep_barrier_engine_count", 1)
            self.state = ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS
        elif (
            notification_type == EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY
            and self.state
            == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
        ):
            self.old_dp_store.add("eep_barrier_engine_count", 1)
            self.state = ScaleUpExistingEngineState.TRANSFER_WEIGHTS

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L452-L463
    def is_complete(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self.state == ScaleUpNewEngineState.COMPLETE
                if self.worker_type == "new"
                else self.state == ScaleUpExistingEngineState.COMPLETE
            )
        return (
            self.state == ScaleDownRemovingEngineState.COMPLETE
            if self.worker_type == "removing"
            else self.state == ScaleDownRemainingEngineState.COMPLETE
        )

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L465-L474
    def _create_standby_groups(self):
        assert self.old_dp_group is not None
        self.new_dp_group, self.new_dp_store = (
            self.new_parallel_config.stateless_init_dp_group(return_store=True)
        )
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("create_standby_groups", self.reconfig_request)
        )
        # SUBTRACTED: rank0 logger.info("[Elastic EP] Created standby ...")
        #   纯日志，不影响控制流。原 elastic_state.py:L473-474。

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L476-L485
    def _transfer_weights(self):
        assert self.reconfig_request is not None and self.old_dp_group is not None
        old_dp_size = self.old_dp_group.size()
        new_dp_size = self.reconfig_request.new_data_parallel_size

        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("transfer_weights", old_dp_size, new_dp_size)
        )
        # SUBTRACTED: rank0 logger.info(...) 纯日志。原 elastic_state.py:L484-485。

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L487-L493
    def _transfer_expert_mapping(self):
        assert self.old_dp_group is not None
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("broadcast_expert_mapping",)
        )
        # SUBTRACTED: rank0 logger.info(...) 纯日志。原 elastic_state.py:L492-493。

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L495-L503
    def _sync_kv_cache_memory_size(self):
        assert self.engine_core.available_gpu_memory_for_kv_cache > 0
        assert self.new_dp_group is not None and self.old_dp_group is not None
        self.new_parallel_config.sync_kv_cache_memory_size(
            self.new_dp_group,
            self.engine_core.available_gpu_memory_for_kv_cache,
        )
        # SUBTRACTED: rank0 logger.info(...) 纯日志。原 elastic_state.py:L502-503。

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L505-L535
    def _switch_and_prepare(self):
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("switch_and_prepare",)
        )
        old_dp_group = self.old_dp_group
        stateless_destroy_torch_distributed_process_group(old_dp_group)
        assert self.new_dp_group is not None
        new_dp_group = self.new_dp_group
        self.engine_core.dp_group = new_dp_group
        self.engine_core.dp_rank = new_dp_group.rank()
        self.engine_core.dp_store = self.new_dp_store
        engines_running = int(self.engine_core.engines_running)
        current_wave = self.engine_core.current_wave
        step_counter = self.engine_core.step_counter
        # SUBTRACTED: torch.tensor([...]) + torch.distributed.all_reduce(MAX) 跨新组
        #   对齐 [engines_running, current_wave, step_counter]；委派给
        #   new_dp_group.all_reduce_max(...)，保留"切到新组后 all_reduce MAX 对齐
        #   DP wave"控制流（'不停机切换'核心）。原 elastic_state.py:L519-526。
        data = new_dp_group.all_reduce_max([engines_running, current_wave, step_counter])
        self.engine_core.engines_running = bool(data[0])
        self.engine_core.current_wave = int(data[1])
        self.engine_core.step_counter = int(data[2])
        if new_dp_group.rank() == 0:
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.RECONFIGURE_FINISHED
            )
            # SUBTRACTED: logger.info("[Elastic EP] Switched to new setup")
            #   纯日志。原 elastic_state.py:L535。

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L537-L548
    def _eplb_reshuffle(self):
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("perform_eplb_reshuffle",)
        )
        assert self.new_dp_group is not None
        # SUBTRACTED: rank0 logger.info(...) 纯日志。原 elastic_state.py:L542-543。

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L550-L560
    def _eplb_reshuffle_before_scale_down(self):
        assert self.reconfig_request is not None and self.old_dp_group is not None
        self.model_executor.collective_rpc(
            "elastic_ep_execute",
            args=(
                "perform_scale_down_eplb_reshuffle",
                self.reconfig_request.new_data_parallel_size,
            ),
        )
        # SUBTRACTED: rank0 logger.info(...) 纯日志。原 elastic_state.py:L554-555。

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L562-L565
    def _switch_and_remove(self):
        self.model_executor.collective_rpc(
            "elastic_ep_execute", args=("switch_and_remove",)
        )

    # SOURCE: vllm/distributed/elastic_ep/elastic_state.py:L567-L593
    def _update_parallel_config(self):
        assert self.reconfig_request is not None
        reconfig_request = self.reconfig_request
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        parallel_config._coord_store_port = reconfig_request.coord_store_port
