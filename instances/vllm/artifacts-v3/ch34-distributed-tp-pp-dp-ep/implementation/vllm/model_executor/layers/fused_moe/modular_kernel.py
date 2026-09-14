# SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py
# ch34 切面：naive_dp_ep 消费的契约面——FusedMoEActivationFormat / TopKWeightAndReduce
# ABC / PrepareResultType 别名 / FusedMoEPrepareAndFinalizeModular 基类头。
# SUBTRACTED：FusedMoEExpertsModular 族与 batched/deepep 各 PrepareAndFinalize
# 实现（ch26 域）、ActivationFormat 转换工具族。

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch


# SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py:L83-L92
#   FusedMoEActivationFormat —— 枚举面子集
class FusedMoEActivationFormat(Enum):
    # SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py:L83-L92（锚点双置）
    Standard = 1
    # SUBTRACTED: Batched/HiddenVectors 等其余成员——ch26 域。


# SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py:L117-L136
#   TopKWeightAndReduce ABC —— 逐字
class TopKWeightAndReduce(ABC):
    """
    An abstract base class for weight application and reduction implementations.
    """
    # SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py:L117-L136（锚点双置）

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        """
        Apply topk_weights to the fused_experts_outputs and/or reduce.
        If an output tensor is not passed, it will be created in the
        function.
        """
        # SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py:L123-L136（锚点双置）
        raise NotImplementedError


# SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py:L140-L157
#   PrepareResultType 别名与注释 —— 逐字 minus ExpertTokensMetadata（L96-
#   L114，ch26 域）→ object 承载类型位
#
# PrepareResultType is a tuple of:
# - quantized + dispatched a.
# - quantized + dispatched a1_scales.
# - Optional ExpertTokensMetadata containing gpu/cpu tensors
#   as big as the number of local experts with the information about the
#   number of tokens assigned to each local expert.
# - Optional dispatched expert topk IDs
# - Optional dispatched expert topk weight
#
# See `prepare` method below.
#
PrepareResultType = tuple[
    torch.Tensor,
    torch.Tensor | None,
    object | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


# SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py:L257-L418
#   FusedMoEPrepareAndFinalizeModular —— 基类头（prepare/finalize 的抽象契约；
#   prepare L264-L299/finalize L354-L376 归 ch26 域）
class FusedMoEPrepareAndFinalizeModular:
    # SOURCE: vllm/model_executor/layers/fused_moe/modular_kernel.py:L257-L418（锚点双置）
    """
    Base class for modular fused MoE preparation and finalization.
    """

    # SUBTRACTED: FusedMoEPrepareAndFinalize 基类与 activation-format 转换族
    #   （L60-L256）——ch26 域；本章只要 Modular 版 prepare/finalize 契约位。
