# 精简版（只做减法）—— EngineCore / DPEngineCoreProc 的弹性 EP 钩子
#
# 对应真实源码：vllm/v1/engine/core.py
# 源码 pin：f3fef123
#
# 本文件抽取 EngineCore / DPEngineCoreProc 上与弹性 EP 扩缩相关的方法，
# 与真实源码同名、同控制流，只删不增。真实 EngineCore 体量巨大，本章只读
# 扩缩接缝（KV-init 前扩、busy loop 里 progress() 挂钩、reinitialize_distributed
# 触发、_initialize_kv_caches 的 eep 分支），故 EngineCore 的其余构造/调度逻辑
# 整体以 `# SUBTRACTED:` 标注省略，仅保留这些方法附着所需的最小载体。
#
# SUBTRACTED: 真实 EngineCore.__init__ 的执行器构造、StructuredOutputManager、
#   scheduler、batch_queue、profiling 等数十行（vllm/v1/engine/core.py:L80-L200+），
#   以及 DPEngineCoreProc 的 ZMQ/coordinator/输入队列全部构造——本章只读
#   "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH → KV init 前扩"与 busy loop 的 eep 钩子。
#   原 vllm/v1/engine/core.py。

from elastic_state import EEPNotificationType, ReconfigureRankType


# SUBTRACTED: 真实 envs 是 vllm.envs 全局环境变量模块；此处仅复刻本章读取到的
#   一个开关 VLLM_ELASTIC_EP_SCALE_UP_LAUNCH（默认 False），不杜撰其余。
#   原引用 vllm/v1/engine/core.py:L124, L240。
class _Envs:
    # SOURCE: vllm/envs.py (VLLM_ELASTIC_EP_SCALE_UP_LAUNCH) — 经 vllm/v1/engine/core.py:L124, L240 读取
    VLLM_ELASTIC_EP_SCALE_UP_LAUNCH = False


envs = _Envs()


class EngineCore:
    """EngineCore 的弹性 EP 相关切片（其余构造逻辑见上方 SUBTRACTED 说明）。"""

    # SOURCE: vllm/v1/engine/core.py:L80 (EngineCore.__init__, 仅 eep 相关行)
    def __init__(self, vllm_config, model_executor):
        # SUBTRACTED: executor_class(vllm_config) 构造 + register_failure_callback
        #   等。原 vllm/v1/engine/core.py:L118-L121。本章直接接收已构造的
        #   model_executor，保留 self.model_executor 字段语义。
        self.vllm_config = vllm_config
        self.model_executor = model_executor

        # SOURCE: vllm/v1/engine/core.py:L122
        self.available_gpu_memory_for_kv_cache = -1

        # SOURCE: vllm/v1/engine/core.py:L124-L125
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self._eep_scale_up_before_kv_init()

        # SOURCE: vllm/v1/engine/core.py:L127-L128
        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        self.kv_cache_config = kv_cache_config
        # SUBTRACTED: StructuredOutputManager / scheduler / batch_queue / ...
        #   构造。原 vllm/v1/engine/core.py:L129+。

    # SOURCE: vllm/v1/engine/core.py:L235-L255 (_initialize_kv_caches, eep 分支)
    def _initialize_kv_caches(self, vllm_config):
        # SUBTRACTED: 真实方法前段 get_kv_cache_specs / has_kv_cache 判定
        #   与后段 bind_kv_cache / initialize_from_config 等。本章只读
        #   "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH 下用同步得到的统一显存额度"分支。
        #   原 vllm/v1/engine/core.py:L226-L290。
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
                # NOTE(yongji): should already be set
                # during _eep_scale_up_before_kv_init
                assert self.available_gpu_memory_for_kv_cache > 0
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)
        return available_gpu_memory

    # SOURCE: vllm/v1/engine/core.py:L1965-L1978
    def _eep_scale_up_before_kv_init(self):
        from elastic_state import ElasticEPScalingState

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=self.vllm_config.parallel_config,
            worker_type="new",
            scale_type="scale_up",
            reconfig_request=None,
        )
        self.eep_scaling_state.run_pre_kv_init_states()
        self.process_input_queue_block = False


class DPEngineCoreProc(EngineCore):
    """数据并行 EngineCore 进程的弹性 EP 切片（ch07/ch21 详述其 busy loop）。"""

    # SUBTRACTED: 真实 DPEngineCoreProc.__init__ 建立 dp_group/dp_store/
    #   coordinator/输入输出队列等（vllm/v1/engine/core.py，DP 段）。本章
    #   只读其 busy loop 的 eep 钩子与 reinitialize_distributed，故省略构造，
    #   由测试注入 dp_group/dp_store/engines_running/current_wave/step_counter。

    # SOURCE: vllm/v1/engine/core.py:L1790-L1844 (run_busy_loop, eep 钩子段)
    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    if self.eep_scaling_state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None

            # SUBTRACTED: 此后是 ch21 已讲的 DP wave 逻辑——_process_engine_step /
            #   _maybe_publish_request_counts / execute_dummy_batch /
            #   _has_global_unfinished_reqs / current_wave++ / step_counter 重置。
            #   本章只截到 eep 钩子，wave 部分交叉引用 ch21。
            #   原 vllm/v1/engine/core.py:L1806-L1842。
            self._process_engine_step()

        # SOURCE: vllm/v1/engine/core.py:L1844
        raise SystemExit

    # SOURCE: vllm/v1/engine/core.py:L1865-L1911
    def reinitialize_distributed(self, reconfig_request) -> None:
        from copy import deepcopy

        from elastic_state import ElasticEPScalingState

        new_parallel_config = deepcopy(self.vllm_config.parallel_config)
        old_dp_size = new_parallel_config.data_parallel_size
        new_parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            new_parallel_config.data_parallel_rank = (
                reconfig_request.new_data_parallel_rank
            )
        new_parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        new_parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        new_parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        new_parallel_config._coord_store_port = reconfig_request.coord_store_port

        is_scale_down = reconfig_request.new_data_parallel_size < old_dp_size
        is_shutdown = (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        )

        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.process_input_queue_block = False
        # SUBTRACTED: logger.info("[Elastic EP] Received reconfiguration ...")
        #   纯日志。原 vllm/v1/engine/core.py:L1909-1911。

    # SOURCE: vllm/v1/engine/core.py:L1953-L1963 (eep_handle_engine_core_notification)
    def eep_handle_engine_core_notification(self, notification_type) -> None:
        """
        Handle notification received from EngineCoreClient
        (forwarded from new core engines).
        """
        assert self.eep_scaling_state is not None
        if isinstance(notification_type, str):
            notification_type = EEPNotificationType(notification_type)
        self.eep_scaling_state.handle_notification(notification_type)

    # SOURCE: vllm/v1/engine/core.py:L1913 (_eep_send_engine_core_notification)
    def _eep_send_engine_core_notification(self, notification_type, vllm_config=None):
        # SUBTRACTED: 真实方法把通知经 output_queue/ZMQ 发回 EngineCoreClient
        #   再转发给其它引擎（vllm/v1/engine/core.py:L1913-L1951）。本章只读
        #   状态机调用此发送点的时机，故委派给可观察的注入回调，保留调用语义。
        if getattr(self, "_eep_notification_sink", None) is not None:
            self._eep_notification_sink(notification_type)
