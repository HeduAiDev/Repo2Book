#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/method_adapters.py

三个适配器 wrapper，各继承 vLLM 一个方法基类、自身只「搬运」给内部 scheme：
- AscendLinearMethod(LinearMethodBase)：create_weights 把 scheme 返回的参数 dict 逐个 register_parameter；
  apply 转交 scheme.apply（量化 matmul）。
- AscendKVCacheMethod(BaseKVCacheMethod)：KV cache 量化，create_weights/apply 转交 attention scheme。
- AscendFusedMoEMethod(FusedMoEMethodBase)：量化版 MoE（承接 ch26 FusedMoE）。
"""

from collections.abc import Callable

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.utils import enable_dsa_cp_with_layer_shard

from .methods import AscendAttentionScheme, AscendLinearScheme, AscendMoEScheme, is_mx_quant_type


# SOURCE: vllm_ascend/quantization/method_adapters.py:L37
class AscendLinearMethod(LinearMethodBase):
    """Linear method for Ascend quantization.

    This wrapper class delegates to the actual quantization scheme implementation.
    The scheme is determined by the Config class and passed directly to this wrapper.
    """

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L47
    def __init__(self, scheme: AscendLinearScheme) -> None:
        self.quant_method = scheme
        self._enable_dsa_cp_with_layer_shard = enable_dsa_cp_with_layer_shard()

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L51
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(input_size_per_partition, output_size_per_partition, params_dtype)

        # Extract packing information (if present)
        packed_dim = weight_dict.pop("_packed_dim", None)
        packed_factor = weight_dict.pop("_packed_factor", None)

        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})

            # Set packing attributes if the weight is packed
            if packed_dim is not None and packed_factor is not None:
                set_weight_attrs(param, {"packed_dim": packed_dim, "packed_factor": packed_factor})

            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        # NOTE: In flatquant quantization implementation,
        # the shape of pertensor_param requires introducing layer_type
        layer_type = "row" if isinstance(layer, RowParallelLinear) else "others"

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype, layer_type=layer_type)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(data=pertensor_param, weight_loader=weight_loader)
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)
            param.weight_loader = extra_weight_attrs.get("weight_loader")

        perchannel_dict = self.quant_method.get_perchannel_param(output_size_per_partition, params_dtype)
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        # NOTE: In w4a8 quantization implementation,
        # for down_proj and o_proj scale_bias shape is [output_size, 16],
        # others are [output_size, 1]
        layer_type = "row" if isinstance(layer, RowParallelLinear) else "others"

        pergroup_dict = self.quant_method.get_pergroup_param(
            input_size_per_partition, output_size_per_partition, params_dtype, layer_type=layer_type
        )
        scale_packed_dim = pergroup_dict.pop("_packed_dim", None)
        scale_packed_factor = pergroup_dict.pop("_packed_factor", None)
        for pergroup_name, pergroup_param in pergroup_dict.items():
            param = torch.nn.Parameter(pergroup_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(pergroup_name, param)
            set_weight_attrs(param, extra_weight_attrs)
            if scale_packed_dim is not None and scale_packed_factor is not None:
                set_weight_attrs(param, {"packed_dim": scale_packed_dim, "packed_factor": scale_packed_factor})
            if (
                "weight_scale_second" in pergroup_name
                or "weight_offset_second" in pergroup_name
                or is_mx_quant_type(self.quant_method)
            ):
                param.input_dim = 1

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L124
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L128
    def get_computed_params(self) -> set[str]:
        """Return parameter name patterns that are computed, not loaded."""
        return {"weight_offset", "quant_bias", "deq_scale", "weight_scale"}

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L140
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SUBTRACTED: RowParallelLinear 下 o_proj/down_proj/out_proj × oproj_tp/mlp_tp/flashcomm2/dsa_cp
        #   的 5 分支 tp_rank 选路（并行通信场景，原 L146-L159）——保留「非 row 则 tp_rank=0」主路径。
        tp_rank = 0
        return self.quant_method.apply(layer, x, bias, tp_rank)


# SOURCE: vllm_ascend/quantization/method_adapters.py:L165
class AscendKVCacheMethod(BaseKVCacheMethod):
    """KVCache method for Ascend quantization.

    This wrapper class delegates to the actual attention quantization scheme.
    """

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L174
    def __init__(self, scheme: AscendAttentionScheme) -> None:
        self.quant_method = scheme

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L177
    def create_weights(self, layer: torch.nn.Module) -> None:
        # Different from linear method, there are no weight processing/slicing
        # steps for attention in vllm. So the whole process of create weights
        # is hidden into the specific quant method.
        self.quant_method.create_weights(layer)

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L183
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.quant_method.process_weights_after_loading(layer)

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L186
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
        return self.quant_method.apply(layer, query, key, value, kv_cache, attn_metadata, attn_type, scale, output)


# SOURCE: vllm_ascend/quantization/method_adapters.py:L201
class AscendFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for Ascend quantization.

    This wrapper class delegates to the actual MoE quantization scheme.
    """

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L211
    def __init__(self, scheme: AscendMoEScheme, moe_config: FusedMoEConfig, tid2eid=None) -> None:
        super().__init__(moe_config)
        self.quant_method = scheme
        self.tid2eid = tid2eid

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L216
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.quant_method.get_weight(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        per_group_param = ["weight_scale_second", "weight_offset_second", "scale_bias"] + (
            ["weight_scale", "weight_offset"]
            if hasattr(self.quant_method, "group_size") and self.quant_method.group_size > 0
            else []
        )
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size, params_dtype
        )
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(fields in param_key for fields in per_group_param):
                param.quant_method = FusedMoeWeightScaleSupported.GROUP.value

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L249
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
        global_redundant_expert_num=0,
        pertoken_scale: torch.Tensor | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_experts=num_experts,
            expert_map=expert_map,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            pertoken_scale=pertoken_scale,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            mc2_mask=mc2_mask,
            tid2eid=self.tid2eid,
        )

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L300
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L304
    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        pass

    @property
    # SOURCE: vllm_ascend/quantization/method_adapters.py:L307
    def supports_eplb(self):
        supports_eplb = getattr(self.quant_method, "supports_eplb", False)
        return supports_eplb


# SOURCE: vllm_ascend/quantization/method_adapters.py:L313
class AscendEmbeddingMethod(AscendLinearMethod):
    """Embedding method for Ascend quantization.

    This is essentially the same as AscendLinearMethod, just with a different name
    for clarity when used with VocabParallelEmbedding layers.
    """

    # SOURCE: vllm_ascend/quantization/method_adapters.py:L323
    def __init__(self, scheme: AscendLinearScheme) -> None:
        self.quant_method = scheme
