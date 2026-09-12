# SOURCE: vllm/v1/worker/worker_base.py
# v3 ch31 脊柱⑦：WorkerBase 的两方法契约（L142-L157，全硬件后端统一）——
# execute_model 返回 None ⇒ 必须紧跟 sample_tokens；docstring 自注技术债
# （L147-L149 'may be changed in future if/when structured outputs parallelism
# is re-architected'——两段式 API 形态就是为结构化输出位掩码的并行而生的物证）。
# SUBTRACTED：WorkerBase 其余装配面（init_worker/load_model/health/LoRA/…
# L60-L141、L159 起——ch17 worker 域全文已立）。
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


# SOURCE: vllm/v1/worker/worker_base.py WorkerBase —— 本章切面（两方法契约位）
class WorkerBase:
    # SOURCE: vllm/v1/worker/worker_base.py:L142-L151 execute_model —— 逐字
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput | None":
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    # SOURCE: vllm/v1/worker/worker_base.py:L153-L157 sample_tokens —— 逐字
    def sample_tokens(
        self, grammar_output: "GrammarOutput"
    ) -> "ModelRunnerOutput | AsyncModelRunnerOutput":
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    # SUBTRACTED: vllm/v1/worker/worker_base.py:L60-L141/L159-L360 装配面
    #   （compile_or_warm_up/check_health/init_device/load_model/LoRA/…
    #   + WorkerWrapperBase 的 RPC 壳）——ch17 全文已立。
