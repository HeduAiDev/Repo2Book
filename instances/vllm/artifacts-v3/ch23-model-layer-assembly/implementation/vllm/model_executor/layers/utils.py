# SOURCE: vllm/model_executor/layers/utils.py
# ch23 消费面：dispatch_unquantized_gemm（UnquantizedLinearMethod.apply /
# UnquantizedEmbeddingMethod.apply 的 GEMM 派发）。
# SUBTRACTED：rocm/aiter/CPU linear 派发族与 quant_utils 面——平台 GEMM 域。
from __future__ import annotations

from typing import Callable

import torch

from vllm.platforms import current_platform


# SOURCE: vllm/model_executor/layers/utils.py:L92-L98 default_unquantized_gemm
#   （逐字——F.linear 主路径）
def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    # SOURCE: vllm/model_executor/layers/utils.py:L92-L98 default_unquantized_gemm
    return torch.nn.functional.linear(x, weight, bias)


# SUBTRACTED: rocm_unquantized_gemm / cpu_unquantized_gemm
#   （layers/utils.py:L101-L345）——ROCm aiter 与 CPU 平台 GEMM 派发域


# SOURCE: vllm/model_executor/layers/utils.py:L348-L355 dispatch_unquantized_gemm
#   —— 减法子集（派发主干逐字；rocm/cpu 两支的派发函数已随平台 GEMM 族
#   删除——非 ROCm/CPU 平台同型地落到 default_unquantized_gemm）
def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    # SUBTRACTED: rocm/cpu 平台支（layers/utils.py:L349-L352：rocm_
    #   unquantized_gemm / cpu_unquantized_gemm）——平台 GEMM 派发域
    # SOURCE: vllm/model_executor/layers/utils.py:L348-L355 dispatch_unquantized_gemm
    return default_unquantized_gemm
