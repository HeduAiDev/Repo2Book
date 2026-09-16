# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""调度输出载体：回执信封随 `EngineCoreOutput` 出引擎。

# SOURCE: vllm/v1/core/sched/output.py:L40-L330
# SUBTRACTED: 结构化输出/投机解码草稿/多模态输入/EC 传输等载荷字段——
#   本章只保留 `kv_transfer_params`（回执出引擎的通道）与 connector 消费的
#   `num_scheduled_tokens`。
"""

from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/v1/core/sched/output.py:L193-L283（SchedulerOutput）
@dataclass
class SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L193-L283（SchedulerOutput）
    """调度器每步产出的批描述（本章只用它的形状，不驱动真前向）。"""

    scheduled_new_reqs: list[Any] = field(default_factory=list)
    scheduled_cached_reqs: Any = None
    num_scheduled_tokens: dict[str, int] | None = None
    total_num_scheduled_tokens: int = 0
    finished_req_ids: set[str] = field(default_factory=set)
    kv_connector_metadata: Any | None = None


__all__ = ["SchedulerOutput"]
