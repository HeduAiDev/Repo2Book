# Subtract-only companion for v3 ch19 — vllm/v1/worker/worker_base.py
# (pin v0.27.1 / 6e448d0ea). Kept surface: CompilationTimes (the NamedTuple
# gpu_worker.compile_or_warm_up_model returns — 启动耗时回传链的载体，m15).
# SUBTRACTED: WorkerBase 抽象类与其余内容（L43 起——worker 控制面/通信协议，
# ch17 域；ch19 的 Worker 只锚定 compile_or_warm_up_model 编排本体）。
from __future__ import annotations

from typing import NamedTuple


# SOURCE: vllm/v1/worker/worker_base.py:L34-L36 CompilationTimes —— 启动期
#   编译耗时回传（worker → executor → 主进程取 max 落 compilation_config）
class CompilationTimes(NamedTuple):  # SOURCE: vllm/v1/worker/worker_base.py:L34-L36
    language_model: float
    encoder: float
