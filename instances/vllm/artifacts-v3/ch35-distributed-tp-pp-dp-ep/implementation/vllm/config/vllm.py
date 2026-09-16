# SOURCE: vllm/config/vllm.py —— 本章切面：VllmConfig 载体（field 子集，装配线归
# ch03）+ needs_dp_coordinator 属性逐字（L660-L681，站 7 的判定式）+ current-config
# holder（L2369-L2448 的 set/get 子集）。真实文件的其余面（编译检查/分层解析）不携带。

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.config.parallel import ParallelConfig

if TYPE_CHECKING:
    pass


# SOURCE: vllm/config/model.py ModelConfig —— HOST SEAM 载体（ch03 域）：
#   本章只消费 is_moe / multimodal_config 两个面。
@dataclass
class ModelConfig:  # HOST SEAM
    # SOURCE: vllm/config/model.py:L122-L2026（锚点双置）
    is_moe: bool = False
    multimodal_config = None


# SOURCE: vllm/config/scheduler.py SchedulerConfig —— HOST SEAM 载体（ch09 域）：
#   本章只消费 prefill_schedule_interval。
@dataclass
class SchedulerConfig:  # HOST SEAM
    # SOURCE: vllm/config/scheduler.py:L26-L285（锚点双置）
    prefill_schedule_interval: int = 1


# SOURCE: vllm/config/vllm.py:L331-L2366 VllmConfig —— 载体 field 子集（装配
#   线/校验族归 ch03 域；本章消费面只读 model_config/parallel_config）
@dataclass
class VllmConfig:
    # SOURCE: vllm/config/vllm.py:L331 VllmConfig 类头（锚点双置）
    model_config: ModelConfig = field(default_factory=ModelConfig)
    parallel_config: "ParallelConfig" = field(default_factory=lambda: ParallelConfig())
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)

    # SUBTRACTED: 其余 config 字段（cache/scheduler/deep面）——ch03 域；
    #   本章消费面只有 model_config/parallel_config/scheduler_config。

    # SOURCE: vllm/config/vllm.py:L661-L681 needs_dp_coordinator — 逐字
    @property
    def needs_dp_coordinator(self) -> bool:
        """
        Determine if the DPCoordinator process is needed.

        The DPCoordinator is needed in two cases:
        1. For MoE models with DP > 1: to handle wave coordination
           (even in external LB mode, since wave coordination runs in the coordinator)
        2. For non-MoE models in internal/hybrid LB mode: to collect and publish
           queue stats for load balancing across DP ranks

        Returns:
            True if DPCoordinator process is needed, False otherwise.
        """
        # SOURCE: vllm/config/vllm.py:L661-L681（锚点双置）

        # For non-MoE models, only need coordinator in internal/hybrid LB mode
        # (for stats collection).
        return self.parallel_config.data_parallel_size > 1 and (
            self.model_config is None
            or self.model_config.is_moe
            or not self.parallel_config.data_parallel_external_lb
        )


# SOURCE: vllm/config/vllm.py:L2369 _current_vllm_config 模块级 holder — 逐字
_current_vllm_config: VllmConfig | None = None


# SOURCE: vllm/config/vllm.py:L2374-L2425 set_current_vllm_config —— 语义子集（锚点双置）
#   （真实版还做编译计数/check_compile 告警——L2380-L2382/L2402-L2419，ch19 域，
#   按 SUBTRACTED 裁除；进出栈与恢复语义逐字保留）。
@contextlib.contextmanager
def set_current_vllm_config(
    vllm_config: VllmConfig, check_compile=False, prefix: str | None = None
):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    # SOURCE: vllm/config/vllm.py:L2374-L2425（锚点双置）
    global _current_vllm_config
    old_vllm_config = _current_vllm_config
    try:
        _current_vllm_config = vllm_config
        yield
    finally:
        _current_vllm_config = old_vllm_config


# SOURCE: vllm/config/vllm.py:L2434-L2444 get_current_vllm_config — 逐字
def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
        )
    return _current_vllm_config


# SOURCE: vllm/config/vllm.py:L2447-L2448 get_current_vllm_config_or_none — 逐字
def get_current_vllm_config_or_none() -> VllmConfig | None:
    return _current_vllm_config
