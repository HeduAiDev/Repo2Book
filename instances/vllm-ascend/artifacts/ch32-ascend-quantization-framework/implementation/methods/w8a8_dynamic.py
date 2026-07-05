#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/quantization/methods/w8a8_dynamic.py

走通全链的样本 scheme：
- AscendW8A8DynamicLinearMethod @register_scheme("W8A8_DYNAMIC","linear")：int8 权重 + per-channel
  weight_scale；apply 走 npu_dynamic_quant(x) → npu_quant_matmul（NPU 硬特化，host 不真跑）。
- AscendW8A8DynamicFusedMoEMethod @register_scheme("W8A8_DYNAMIC","moe")：量化版 MoE（承接 ch26）。
"""

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme

# SUBTRACTED: scale_from_float_to_int64（float32 scale → int64，原 w8a8_dynamic.py:L38-L45）——
#   仅被下方 enable_fused_mc2 分支使用，该分支已减去，故此辅助函数一并删去。


@register_scheme("W8A8_DYNAMIC", "linear")
# SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L48
class AscendW8A8DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_DYNAMIC.

    This scheme uses dynamic per-token quantization for activations
    and per-channel quantization for weights.
    """

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L56
    def __init__(self):
        pass

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L59
    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}
        return params_dict

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L63
    def get_perchannel_param(
        self,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L73
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        need_unsqz = False
        if pertoken_scale.dim() == 2:
            need_unsqz = True
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)

        # SUBTRACTED: _chunk_size>0 时把 >=65536 维大权重切两半各跑一次 npu_quant_matmul 再 cat 的
        #   workaround（原 L87-L111）——非主路径，保留 else 单次量化 matmul 即讲清全链。
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=x.dtype,
        )
        if need_unsqz:
            output = output.unsqueeze(dim=1)
        return output

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L125
    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # SUBTRACTED: wq_b 大权重(>=65536) + enable_dsa_cp() 的切两半 workaround（原 L127-L143）。
        # cast quantized weight tensors in NZ format for higher inference speed
        layer.weight.data = maybe_trans_nz(layer.weight.data)
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()


@register_scheme("W8A8_DYNAMIC", "moe")
# SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L152
class AscendW8A8DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W8A8_DYNAMIC."""

    # Declare the quantization type for this scheme
    quant_type: QuantType = QuantType.W8A8

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L159
    def __init__(self):
        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate

        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
        self.in_dtype = vllm_config.model_config.dtype
        self.supports_eplb = True

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            logger.warning_once(
                "[vllm-ascend/W8A8_DYNAMIC] MC2 group metadata unavailable, "
                "falling back to empty moe_all_to_all_group_name."
            )
            self.moe_all_to_all_group_name = ""

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L185
    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.int8
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition, dtype=torch.int8
        )
        return param_dict

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L197
    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=params_dtype
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=params_dtype)
        return param_dict

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L211
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
        tid2eid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_shared_experts = getattr(layer, "n_shared_experts", 0)
        mix_placement = getattr(layer, "mix_placement", False)
        if n_shared_experts is None:
            n_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=n_shared_experts,
        )
        # SUBTRACTED: zero_expert_num/zero_expert_type 取值 + 全局专家数量一致性 assert（原 L237-L257）。
        # SUBTRACTED: multistream_overlap_gate 分支——从 flash_common3 context 直接取 topk（原 L259-L266）。
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            mix_placement=mix_placement,
            num_logical_experts=router_logits.shape[1],
            num_shared_experts=n_shared_experts,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )
        assert topk_ids is not None
        assert topk_weights is not None
        # SUBTRACTED: zero_experts_compute（零专家计算，原 L287-L294）。
        # SUBTRACTED: enable_force_load_balance 的随机均衡 topk_ids 重排（profile run 专用，原 L298-L300）。
        assert topk_weights is not None
        topk_weights = topk_weights.to(self.in_dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # SUBTRACTED: fused_scale_flag(FUSED_MC2) + dynamic_eplb 下的多权重/多 scale 列表准备
        #   （w13_weight_list / fused_w*_scale_list 等，原 L306-L321）；保留单组 w/scale 主路径。
        w1 = [layer.w13_weight]
        w1_scale = [layer.w13_weight_scale_fp32]
        w2 = [layer.w2_weight]
        w2_scale = [layer.w2_weight_scale]
        w1_scale_bias = None
        w2_scale_bias = None

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                swiglu_limit=layer.swiglu_limit,
            )
        )
        # SUBTRACTED: zero_expert_result 累加回 final_hidden_states（原 L346-L347）。
        return final_hidden_states

    # SOURCE: vllm_ascend/quantization/methods/w8a8_dynamic.py:L350
    def process_weights_after_loading(self, layer):
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()
        # TODO(zzzzwwjj): Currently, `torch_npu.npu_grouped_matmul_swiglu_quant`
        # can only support weight nz.
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(layer.w13_weight_scale.data.shape[0], -1)
        layer.w13_weight_scale_fp32 = layer.w13_weight_scale.data.to(torch.float32)
        layer.w13_weight_offset.data = layer.w13_weight_offset.data.view(layer.w13_weight_offset.data.shape[0], -1)
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.view(layer.w2_weight_scale.data.shape[0], -1)
        layer.w2_weight_offset.data = layer.w2_weight_offset.data.view(layer.w2_weight_offset.data.shape[0], -1)
        # SUBTRACTED: enable_fused_mc2 的 scale→int64 转换 + dynamic_eplb 的逐专家 weight/scale list 化
        #   与对应张量释放（原 L363-L391）——高级特性，非「转置+NZ+scale view」主干。
