"""桩 Worker —— 被 init_worker 按 worker_cls qualname 解析（复现真实 vLLM 用字符串类名
延迟实例化的行为）。必须是模块级类，才能被 spawn 子进程按 'stub_worker.StubWorker' 重新 import。

它继承精简版的 WorkerBase，但把 init_device/load_model 实现为无副作用，execute_model 实现为
一个确定性纯函数（不触 CUDA/torch、不杜撰前向）——只为让控制平面闭环可跑、可数值追踪：
collective_rpc('execute_model', (x,)) 应在每个 worker 上得到 ('echo', rank, x)。
"""
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))

from worker_base import WorkerBase  # noqa: E402


class StubWorker(WorkerBase):
    def init_device(self) -> None:
        self.device = "stub"

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        self.loaded = True

    def execute_model(self, scheduler_output):
        # 确定性回声：用于验证广播到达 + 单 rank 应答 + 结果回传。
        return ("echo", self.rank, scheduler_output)

    def add(self, a, b):
        return a + b

    def boom(self):
        raise ValueError("intentional worker failure")

    def crash_process(self):
        # 让 worker 子进程直接退出（模拟 OOM/segfault），触发 sentinel 监控路径。
        import os

        os._exit(1)
