#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/methods/base.py

三类 scheme 的 ABC 契约（linear / attention / moe）。契约固定 get_weight/apply 等抽象方法，
所以 method_adapters 里的 wrapper 才能「盲转交」——接口固定，scheme 各自填。
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

from vllm_ascend.quantization.quant_type import QuantType


# SOURCE: vllm_ascend/quantization/methods/base.py:L28
def get_moe_num_logical_experts(
    layer: torch.nn.Module,
    num_experts: int,
    global_redundant_expert_num: int = 0,
    num_shared_experts: int = 0,
) -> int:
    moe_config = getattr(layer, "moe_config", None)
    num_logical_experts = getattr(moe_config, "num_logical_experts", None)
    if num_logical_experts is not None:
        return int(num_logical_experts)

    return int(num_experts - global_redundant_expert_num - num_shared_experts)


# SOURCE: vllm_ascend/quantization/methods/base.py:L42
class AscendLinearScheme(ABC):
    """Base class for all linear quantization schemes.

    Subclasses must implement get_weight() and apply() methods.
    Other methods have default implementations that return empty dicts
    or do nothing.
    """

    @abstractmethod
    # SOURCE: vllm_ascend/quantization/methods/base.py:L50
    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return weight tensor specifications."""
        ...

    # SOURCE: vllm_ascend/quantization/methods/base.py:L65
    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        """Return per-tensor parameter specifications (e.g., input_scale)."""
        return {}

    # SOURCE: vllm_ascend/quantization/methods/base.py:L77
    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return per-channel parameter specifications (e.g., weight_scale)."""
        return {}

    # SOURCE: vllm_ascend/quantization/methods/base.py:L89
    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        """Return per-group parameter specifications."""
        return {}

    @abstractmethod
    # SOURCE: vllm_ascend/quantization/methods/base.py:L105
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None, tp_rank: int | None = 0
    ) -> torch.Tensor:
        """Forward computation."""
        ...

    # SOURCE: vllm_ascend/quantization/methods/base.py:L122
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing (transpose, format conversion, etc.)."""
        return


# SOURCE: vllm_ascend/quantization/methods/base.py:L131
class AscendAttentionScheme(ABC):
    """Base class for all attention quantization schemes.

    Subclasses must implement apply() method.
    Other methods have default implementations.
    """

    # SOURCE: vllm_ascend/quantization/methods/base.py:L138
    def create_weights(self, layer: torch.nn.Module) -> None:
        """Create weights for attention quantization."""
        return

    # SOURCE: vllm_ascend/quantization/methods/base.py:L146
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing for attention layer."""
        return

    @abstractmethod
    # SOURCE: vllm_ascend/quantization/methods/base.py:L154
    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        """Forward computation for attention layer."""
        ...


# SOURCE: vllm_ascend/quantization/methods/base.py:L186
class AscendMoEScheme(ABC):
    """Base class for all MoE quantization schemes.

    Subclasses must implement get_weight(), get_dynamic_quant_param(),
    and apply() methods.
    """

    # Default quant type - subclasses should override this
    quant_type: QuantType = QuantType.NONE

    @abstractmethod
    # SOURCE: vllm_ascend/quantization/methods/base.py:L200
    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Return weight tensor specifications for MoE layer."""
        ...

    @abstractmethod
    # SOURCE: vllm_ascend/quantization/methods/base.py:L217
    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Return dynamic quantization parameters for MoE layer."""
        ...

    @abstractmethod
    # SOURCE: vllm_ascend/quantization/methods/base.py:L234
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        """Forward computation for MoE layer."""
        ...

    # SOURCE: vllm_ascend/quantization/methods/base.py:L292
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading weight processing for MoE layer."""
        return
