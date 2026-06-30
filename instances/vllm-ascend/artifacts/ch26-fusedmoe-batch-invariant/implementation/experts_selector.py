"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/ops/fused_moe/experts_selector.py

路由器选 topk 专家——token 重分发的输入来源。AscendUnquantizedFusedMoEMethod.apply 调它，
拿到 (topk_weights, topk_ids) 后才能 build_fused_experts_input → moe_comm_method.fused_experts。
"""
from collections.abc import Callable

import torch


# SOURCE: vllm_ascend/ops/fused_moe/experts_selector.py:L30
def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor=1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    indices_type: torch.dtype | None = None,
    mix_placement: bool = False,
    num_logical_experts: int = -1,
    num_shared_experts: int = 0,
    num_experts: int = -1,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
):
    """Select top-k experts.

    Returns:
        topk_weights: router weights of shape (num_tokens, top_k).
        topk_ids: selected expert IDs of shape (num_tokens, top_k).
    """
    # SUBTRACTED: weight_prefetch（gate_up 权重预取）（原 experts_selector.py:L71-L73）——性能优化。
    is_support_npu_moe_gating_top_k = check_npu_moe_gating_top_k(
        hidden_states=hidden_states,
        top_k=top_k,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        scoring_func=scoring_func,
        custom_routing_function=custom_routing_function,
    )

    if is_support_npu_moe_gating_top_k:
        # NPU 融合 gating 算子路径（npu_moe_gating_top_k）
        topk_weights, topk_ids = _select_experts_with_fusion_ops(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            tid2eid=tid2eid,
            input_ids=input_ids,
        )
    else:
        # 原子算子回退路径（softmax/sigmoid + grouped/topk）
        topk_weights, topk_ids = _native_select_experts(
            hidden_states=hidden_states,
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
            num_experts=num_experts,
            tid2eid=None,
            input_ids=None,
        )
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor
    # SUBTRACTED: mix_placement 时把 shared expert id 拼进 topk（原 L120-L135）——共享专家旁路，
    #            本章主线是 routed 专家。
    return topk_weights, topk_ids


# SUBTRACTED: check_npu_moe_gating_top_k / _select_experts_with_fusion_ops / _native_select_experts /
#            _select_expert_use_group_topk / _native_grouped_topk / _renormalize_topk_weights /
#            zero_experts_compute（原 experts_selector.py:L140-L420）——融合 gating 算子与原子回退的
#            具体打分/分组/归一实现，属路由算法内部；本章聚焦『选好专家后如何按专家重分发 token』，
#            故只保留 select_experts 的分流骨架。测试以 stub 提供这两个 helper 的返回。
def check_npu_moe_gating_top_k(**kwargs):
    # SOURCE: vllm_ascend/ops/fused_moe/experts_selector.py:L140
    raise NotImplementedError("NPU 能力探测，host 由 stub 注入")


def _select_experts_with_fusion_ops(**kwargs):
    # SOURCE: vllm_ascend/ops/fused_moe/experts_selector.py:L236
    raise NotImplementedError("npu_moe_gating_top_k 融合算子，host 不真跑")


def _native_select_experts(**kwargs):
    # SOURCE: vllm_ascend/ops/fused_moe/experts_selector.py:L313
    raise NotImplementedError("原子算子打分/分组/topk，host 由 stub 注入")
