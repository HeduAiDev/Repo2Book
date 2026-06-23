# 只做减法的忠实精简版 —— 镜像 vllm/v1/outputs.py 中 Sampler 用到的两个载体（pin f3fef123）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: SPDX 版权头；本文件其余结构体（LogprobsLists、ModelRunnerOutput 等）与
# LogprobsTensors 的 tolists/to_cpu_nonblocking/filter/empty_cpu 辅助方法、
# cu_num_generated_tokens 字段（outputs.py:L28-49, L59-109）—— 不在 9 步采样主路径上。
from dataclasses import dataclass
from typing import NamedTuple

import torch


class LogprobsTensors(NamedTuple):
    # SOURCE: vllm/v1/outputs.py:L51-57
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: torch.Tensor


@dataclass
class SamplerOutput:
    # SOURCE: vllm/v1/outputs.py:L117-124
    # [num_reqs, max_num_generated_tokens]
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
