# SOURCE: vllm/config/__init__.py（真实为 config/ 包各类的 re-export 门面）
# HOST SEAM：配置面最小承载。本章消费面 = build_logitsprocs 读
# vllm_config.speculative_config（logits_processor/__init__.py:L202）与
# MinP 构造器读 vllm_config.scheduler_config.max_num_seqs（builtin.py:L27）。
# 真实 VllmConfig/SpeculativeConfig/SchedulerConfig 是 ch03 域的装配链
# （数千行）——此处以同名字段载体镜像（ch23 同款做法）。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/config/scheduler.py SchedulerConfig —— HOST SEAM 字段面
#   （本章消费：max_num_seqs——MinP 构造器的槽位预分配宽度）
@dataclass
class SchedulerConfig:
    # SOURCE: vllm/config/scheduler.py max_num_seqs 字段 —— HOST SEAM
    max_num_seqs: int = 8


# SOURCE: vllm/config/vllm.py:L331 起 VllmConfig —— HOST SEAM 字段面
#   （本章消费：speculative_config——build_logitsprocs 的 spec 分支判据）
@dataclass
class VllmConfig:
    # SOURCE: vllm/config/vllm.py speculative_config 字段 —— HOST SEAM
    speculative_config: Any = None
    # SOURCE: vllm/config/vllm.py scheduler_config 字段 —— HOST SEAM
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
