# SOURCE: vllm/v1/core/sched/output.py
# ch34 切面：SchedulerOutput 载体面（Worker.execute_model 触碰的
# total_num_scheduled_tokens 与 num_scheduled_tokens 字段）。差量协议全貌归
# ch18 域。

from __future__ import annotations

from dataclasses import dataclass, field


# SOURCE: vllm/v1/core/sched/output.py:L193-L283 SchedulerOutput —— 字段子集载体
@dataclass
class SchedulerOutput:  # HOST/ch18 SEAM
    # SOURCE: vllm/v1/core/sched/output.py:L193-L283（锚点双置）
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    # SUBTRACTED: scheduled_new_reqs/cached_reqs/spec/encoder/prefix 差量字段族
    #   ——ch18 域（差量协议本体）。
