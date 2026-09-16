# SOURCE: vllm/sampling_params.py
# HOST SEAM：接口面最小承载。本章精简版里 SamplingParams 只被
# logits_processor/interface.py:L11 `from vllm import SamplingParams`
# （validate_params 的参数注解）与已删除的 update_state 家族按名引用——
# 采样参数到快照的批量化归 gpu_input_batch（ch18 域），本章以构造好的
# SamplingMetadata 为输入。此处只镜像三个内置处理器读过的字段面
# （真实定义在 vllm/sampling_params.py 的 SamplingParams 类）。
from __future__ import annotations

from dataclasses import dataclass, field


# SOURCE: vllm/sampling_params.py SamplingParams —— HOST SEAM 字段面
#   （本章消费位：MinP/MinTokens/LogitBias 的 update_state 家族——已按
#   subtraction_plan delete[5] 删除归 ch18；字段定义原样镜像）
@dataclass
class SamplingParams:
    # SOURCE: vllm/sampling_params.py min_p 字段（默认 0.0）—— HOST SEAM
    min_p: float = 0.0
    # SOURCE: vllm/sampling_params.py logit_bias 字段（默认 None）—— HOST SEAM
    logit_bias: dict[int, float] | None = None
    # SOURCE: vllm/sampling_params.py min_tokens 字段（默认 0）—— HOST SEAM
    min_tokens: int = 0
    # SOURCE: vllm/sampling_params.py all_stop_token_ids（真实为组合
    #   stop_token_ids ∪ eos 的 @property；HOST SEAM 直存集合）—— HOST SEAM
    all_stop_token_ids: set[int] = field(default_factory=set)
