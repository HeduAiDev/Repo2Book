"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/ops/fused_moe/moe_comm_method.py

f10 回收的落地：_MoECommMethods 注册表把 ch15 选定的 MoECommType 枚举映射到真正建好的
*CommImpl 实例；每种通信方式 = 一对 (TokenDispatcherWith*, PrepareAndFinalizeWith*)，
骨架方法 fused_experts(dispatch→mlp→combine) 不变。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType

_MoECommMethods: dict[MoECommType | None, "MoECommMethod"] = {}


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L51
def get_moe_comm_method(moe_comm_type: MoECommType | None) -> "MoECommMethod | None":
    return _MoECommMethods.get(moe_comm_type)


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L55
def setup_moe_comm_method(moe_config):
    if moe_config.ep_size > 1:
        _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
        _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
        _MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)
    else:
        # EP=1：无专家并行，不需要跨卡重分发，只注册稳妥的 AllGather。
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L65
def set_gmmswigluquant_method():
    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


@dataclass
class FusedExpertsResult:
    # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L72
    routed_out: torch.Tensor
    # SUBTRACTED: before_dispatch_evt / before_gmm2_evt / before_combine_evt 三个 torch.npu.Event
    #             字段——shared-expert 并流计时用（原 moe_comm_method.py:L78-L80）；本精简版不走并流，
    #             保留为 None 占位即可。
    before_dispatch_evt = None
    before_gmm2_evt = None
    before_combine_evt = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None
    swiglu_limit: float = 0.0


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L87
class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L90
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()

    # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L97
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )

    # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L113
    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape=None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L122
    def fused_experts(self, fused_experts_input: MoEFusedExpertsInput):
        # SUBTRACTED: hidden_states.dtype 的 assert 白名单（原 moe_comm_method.py:L127-L133）——
        #             host 无 NPU dtype 约束，删后不改三段流水线骨架。
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        # SUBTRACTED: before_dispatch_evt/before_combine_evt = record_event() 计时（原 L138,L157）——
        #             shared-expert 并流用，本精简版不走并流。
        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        # ① 按专家把 token 重分发到对应专家所在卡
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
        )
        # ② 每个专家算自己那摞 token 的 gmm1→swiglu→gmm2
        mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)

        # ③ 把结果按原顺序聚回来
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )

        return FusedExpertsResult(
            routed_out=routed_out,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

    # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L173
    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L176
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L180
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L185
class AllGatherCommImpl(MoECommMethod):
    """Default implementation, compatible with all scenarios (and EP=1).

    Uses `npu_moe_init_routing_v2` for pre-processing and `npu_moe_token_unpermute`
    for post-processing. NOTE: `npu_moe_finalize_routing` would lead to accuracy
    issues, so `npu_moe_token_unpermute` is used instead (a workaround).
    """

    def _get_token_dispatcher(self):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L204
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L211
        return PrepareAndFinalizeWithAllGather(self.moe_config)


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L215
class MC2CommImpl(MoECommMethod):
    """Uses the MC2 communication method, optimized for Communication and
    Computation parallelism on Ascend devices (requires enable_expert_parallel)."""

    # SUBTRACTED: pad_and_split_input_ids 透传到 prepare_finalize（原 L225-226）——input_ids 旁路，
    #             与 token 重分发主线无关。
    def _get_token_dispatcher(self):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L228
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L231
        return PrepareAndFinalizeWithMC2(self.moe_config)


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L235
class AlltoAllCommImpl(MoECommMethod):
    """Uses all-to-all communication to exchange tokens between data parallel
    ranks before/after the MLP. Better than AllGather when DP size > 1.（回收 f3 的策略层）"""

    def _get_token_dispatcher(self):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L248
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L255
        return PrepareAndFinalizeWithAll2All(self.moe_config)


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L259
class FusedMC2CommImpl(MoECommMethod):
    """Like MC2 but fuses dispatch + ffn + combine into a single C++ operator,
    overriding the base three-step fused_experts."""

    def __init__(self, moe_config):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L269
        super().__init__(moe_config)
        if get_ascend_config().enable_fused_mc2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

    def _get_token_dispatcher(self):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L279
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L282
        return PrepareAndFinalizeWithMC2(self.moe_config)

    # SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L285
    def fused_experts(self, fused_experts_input: MoEFusedExpertsInput):
        # SUBTRACTED: w1_scale/w2_scale/scale_bias 的非空 assert（原 L289-L299）——量化前提校验，
        #             本章聚焦控制流。
        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        expert_tokens = None
        if get_ascend_config().enable_fused_mc2 == 1:
            out = torch.empty_like(fused_experts_input.hidden_states)
            # 把 dispatch + 两段 ffn(gmm1/swiglu/gmm2) + combine 融成单个 C++ 算子一次调完，
            # 从而覆写基类 fused_experts 的 dispatch→mlp→combine 三步分解。
            torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                x=fused_experts_input.hidden_states,
                weight1=fused_experts_input.weights.w1,
                weight2=fused_experts_input.weights.w2,
                expert_idx=topk_ids,
                # SUBTRACTED: scale1/scale2/bias1/bias2/probs/max_output_size/swiglu_limit/x_active_mask
                #             等完整入参（原 L314-L324）——融合算子的量化/掩码细节，host 无 NPU 不真跑。
                group=self.token_dispatcher.moe_all_to_all_group_name,
                out=out,
                expert_token_nums=self.expert_token_nums,
            )
            expert_tokens = self.expert_token_nums
        elif get_ascend_config().enable_fused_mc2 == 2:
            # SUBTRACTED: dispatch_gmm_combine_decode 的完整入参列表（原 L329-L343）——decode 专用融合算子。
            out, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(  # type: ignore
                x=fused_experts_input.hidden_states,
                expert_ids=topk_ids,
                group_ep=self.token_dispatcher.moe_all_to_all_group_name,
            )
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out, expert_tokens=expert_tokens, swiglu_limit=fused_experts_input.swiglu_limit
        )
