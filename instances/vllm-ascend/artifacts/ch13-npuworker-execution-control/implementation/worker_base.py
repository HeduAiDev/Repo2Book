"""对照基座：硬件无关的 Worker 抽象接口（subtract-only 忠实摘录）。

本章 thesis 的支点：NPUWorker 与 GPU 的 Worker 是『同一个抽象 WorkerBase 的两个
平级实现』，而不是 NPUWorker 继承 GPU Worker。这里只保留两样东西证明这一点：
  (1) 四步生命周期方法在抽象层全是 `raise NotImplementedError`（谁派生谁就得自己实现）；
  (2) 公共 __init__ 把 vllm_config 摊开成各 config 字段——super().__init__ 复用的就是它。

可在 host 上 import/运行（纯 Python，抽象方法只抛 NotImplementedError）。
"""
# SUBTRACTED: 文件头 SPDX 许可证 + 大量 import（原 vllm/v1/worker/worker_base.py:L1-L17）。
# SUBTRACTED: from vllm.platforms import current_platform 等运行时依赖——host 无 vllm 平台层。
from typing import NamedTuple


# SOURCE: vllm/v1/worker/worker_base.py:L33-L35
class CompilationTimes(NamedTuple):
    language_model: float
    encoder: float


# SOURCE: vllm/v1/worker/worker_base.py:L38
class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    # SOURCE: vllm/v1/worker/worker_base.py:L44-L88
    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """Initialize common worker components."""
        # 把整份 vllm_config 摊开成各 config 字段——NPUWorker / GPU Worker 都靠
        # super().__init__ 复用这一段公共逻辑，无需各自重写。
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        # SUBTRACTED: from vllm.platforms import current_platform; self.current_platform = current_platform
        #   （原 L76-L78）——平台层句柄，host 无 vllm 平台，删后不影响四步生命周期的对位。

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device = None
        self.model_runner = None

    # ---- 四步生命周期：抽象层全是 raise NotImplementedError ----
    # 谁从 WorkerBase 派生（GPU Worker / NPUWorker），谁就必须自己实现这几个方法。

    # SOURCE: vllm/v1/worker/worker_base.py:L94-L100
    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Prepare model for execution through compilation/warmup.

        Returns:
            Compilation times (language_model, encoder) in seconds.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L106-L110
    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L134-L143
    def execute_model(self, scheduler_output):
        """If this method returns None, sample_tokens should be called immediately
        after to obtain the ModelRunnerOutput.
        """
        raise NotImplementedError

    # SUBTRACTED: get_kv_cache_spec / check_health / reset_mm_cache / get_model /
    #   apply_model / load_model / sample_tokens / add_lora 等其余接口声明（原 L90-L176，
    #   多为同样 raise NotImplementedError 的接口）——四步生命周期对位无需展开。

# SUBTRACTED: class WorkerWrapperBase（原 L179-L345）——进程级 worker 懒初始化/生命周期管理，
#   是 executor 侧的外壳，不在『Worker 重写 vs 继承』主线上。
