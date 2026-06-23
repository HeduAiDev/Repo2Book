"""Shared fixtures / path setup for ch17 executor & worker-lifecycle tests.

精简版用 bare import（from multiproc_executor import ...），需把 implementation/ 加进
sys.path。这里提供桩 VllmConfig/ParallelConfig/SchedulerConfig（只含精简版控制流读到的字段）
与一个桩 Worker 类（按 worker_cls qualname 解析），复现真实 vLLM 的可观察行为——不 import vllm。

注意：MultiprocExecutor 会真正 spawn 子进程，子进程需能 import 到本目录的桩 Worker 和
implementation 模块，故把两路径都注册进 PYTHONPATH（通过 sys.path + 环境变量传给子进程）。
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPL = HERE.parent / "implementation"
for p in (str(IMPL), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 让 spawn 出的子进程也能找到 implementation/ 与 tests/（桩 Worker 在 tests 包里）。
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(IMPL), str(HERE), os.environ.get("PYTHONPATH", "")]
)


@dataclass
class ParallelConfig:
    """精简版执行器/worker 控制流读到的并行配置字段子集。"""

    distributed_executor_backend: object = "uni"
    # worker_cls 是 qualname 字符串 —— init_worker 按它解析真实 Worker 类。
    worker_cls: str = "stub_worker.StubWorker"
    worker_extension_cls: str = ""
    world_size: int = 1
    local_world_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    nnodes_within_dp: int = 1
    node_rank_within_dp: int = 0
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    rank: int = 0


@dataclass
class SchedulerConfig:
    async_scheduling: bool = False


@dataclass
class VllmConfig:
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)


def make_config(backend="mp", world_size=1, worker_cls="stub_worker.StubWorker",
                async_scheduling=False):
    pc = ParallelConfig(
        distributed_executor_backend=backend,
        worker_cls=worker_cls,
        world_size=world_size,
        local_world_size=world_size,
        tensor_parallel_size=world_size,
    )
    return VllmConfig(
        parallel_config=pc,
        scheduler_config=SchedulerConfig(async_scheduling=async_scheduling),
    )
