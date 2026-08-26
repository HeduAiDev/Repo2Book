# SOURCE: vllm/v1/executor/executor_factory.py
# get_executor_class —— 按后端字符串选 executor（vllm_config 仲裁链的输入：
# executor_supports_async_sched = executor_class.supports_async_scheduling()，
# vllm/config/vllm.py:L1057）。multiproc/ray 的生命周期全文归 ch17；本章只保
# 各自的 supports_async_scheduling 一票（multiproc L526-L527 覆写 True；
# ray 未覆写 → 基类默认 False → 默认禁 async）。
from __future__ import annotations

from .uniproc_executor import Executor, UniProcExecutor


# SOURCE: vllm/v1/executor/multiproc_executor.py:L526-L527 supports_async_
# scheduling（覆写 True；类的其余——进程池/消息队列全文归 ch17）
class MultiprocExecutor(Executor):
    # SOURCE: vllm/v1/executor/multiproc_executor.py:L526-L527
    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # SOURCE: vllm/v1/executor/multiproc_executor.py:L526-L527
        return True

    # SUBTRACTED: 多进程生命周期/worker 进程管理（ch17 全文）。


# SOURCE: vllm/v1/executor/ray_executor.py RayExecutor（ray 后端——未覆写
# supports_async_scheduling，继承基类默认 False → 默认禁 async）
class RayExecutor(Executor):
    # SUBTRACTED: ray actor 生命周期（ch17）。
    # SOURCE: vllm/v1/executor/ray_executor.py RayExecutor——未覆写
    #   supports_async_scheduling，继承基类默认 False
    #   （vllm/v1/executor/abstract.py:L364-L368 → 默认禁 async）。
    pass


# SOURCE: vllm/v1/executor/executor_factory.py get_executor_class
def get_executor_class(vllm_config):
    # SOURCE: vllm/v1/executor/executor_factory.py get_executor_class
    # （真实按 distributed_executor_backend/env 推断——本方法切面保仲裁链
    # 消费的三后端映射）
    backend = vllm_config.executor_backend
    # SUBTRACTED: external_launcher/后续后端分支。
    if backend == "uniproc":
        return UniProcExecutor
    if backend == "mp" or backend == "multiproc":
        return MultiprocExecutor
    if backend == "ray":
        return RayExecutor
    raise ValueError(f"Unknown distributed executor backend: {backend}")
