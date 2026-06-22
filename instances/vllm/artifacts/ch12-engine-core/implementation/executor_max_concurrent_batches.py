# ch12 精简版：Executor.max_concurrent_batches 的三处真实定义。
#
# 这是 batch_queue_size 的来源，也是「async_scheduling 标志间接决定 step_fn
# 绑哪个 step」这条链的枢纽：EngineCore.__init__ 读 model_executor.max_concurrent_batches
# → batch_queue_size；>1 才建 batch_queue、step_fn 绑 step_with_batch_queue。
#
# 只做减法：保留三个 max_concurrent_batches 的完整控制流，删去与之无关的 executor
# 方法体（collective_rpc / check_health / _get_output_rank 等，dossier delete[4]）。
# 用最小的 config 占位替身承接 pipeline_parallel_size / async_scheduling 字段。

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass
class _ParallelConfig:  # SOURCE: vllm/config/parallel.py:ParallelConfig (占位字段)
    pipeline_parallel_size: int = 1


@dataclass
class _SchedulerConfig:  # SOURCE: vllm/config/scheduler.py:SchedulerConfig (占位字段)
    async_scheduling: bool = False


class AbstractExecutor:
    """真实基类 vllm/v1/executor/abstract.py:Executor（已精简到只剩本章相关属性）。
    # SUBTRACTED: collective_rpc / take_draft_token_ids / profile 等执行接口
    #             (abstract.py 其余，dossier delete[4]，与 max_concurrent_batches 正交)。"""

    @property
    def max_concurrent_batches(self) -> int:
        # SOURCE: vllm/v1/executor/abstract.py:L256-L258
        return 1


class MultiprocExecutor(AbstractExecutor):
    """真实类 vllm/v1/executor/multiproc_executor.py:MultiprocExecutor（精简）。
    # SUBTRACTED: 进程池 / collective_rpc / check_health / _get_output_rank 等
    #             (multiproc_executor.py 其余，dossier delete[4])。"""

    # SOURCE: vllm/v1/executor/multiproc_executor.py (MultiprocExecutor 构造，测试替身)
    def __init__(self, parallel_config: _ParallelConfig, scheduler_config: _SchedulerConfig):
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config

    @cached_property
    def max_concurrent_batches(self) -> int:
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L474-L478
        # PP requires PP-size concurrent batches to fill the pipeline.
        pp_size = self.parallel_config.pipeline_parallel_size
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size


class UniProcExecutor(AbstractExecutor):
    """真实类 vllm/v1/executor/uniproc_executor.py:UniProcExecutor（精简）。
    # SUBTRACTED: collective_rpc / 单进程 worker 初始化等
    #             (uniproc_executor.py 其余，dossier delete[4])。"""

    # SOURCE: vllm/v1/executor/uniproc_executor.py (UniProcExecutor 构造，测试替身)
    def __init__(self, scheduler_config: _SchedulerConfig):
        self.scheduler_config = scheduler_config

    @cached_property
    def max_concurrent_batches(self) -> int:
        # SOURCE: vllm/v1/executor/uniproc_executor.py:L63-L65
        return 2 if self.scheduler_config.async_scheduling else 1
