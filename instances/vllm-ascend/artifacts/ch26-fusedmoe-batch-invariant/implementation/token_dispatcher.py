"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/ops/fused_moe/token_dispatcher.py

token 按专家重分发的三种落地：
  MC2     —— npu_moe_distribute_dispatch/combine 融合算子
  All2AllV —— async_all_to_all(dist.all_to_all_single, all2all-v 不等长 split)，回收 f3 核心
  AllGather —— 假设已全局 all-gather，仅本地 npu_moe_init_routing 排序 + npu_moe_token_unpermute 还原
"""
from abc import ABC, abstractmethod
from typing import Generic

import torch
import torch_npu
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.comm_utils import async_all_to_all
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEAllToAllCombineMetadata,
    MoEMC2CombineMetadata,
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    TMoECombineMetadata,
)


# SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L63
class MoETokenDispatcher(ABC, Generic[TMoECombineMetadata]):
    def __init__(self, **kwargs) -> None:
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L64
        """Initialize the MoE Token Dispatcher."""
        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)

    @property
    def ep_group(self):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L71
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_size(self):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L80
        return get_ep_group().world_size

    @abstractmethod
    def token_dispatch(self, token_dispatch_input: MoETokenDispatchInput):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L84
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(self, hidden_states, combine_metadata, bias=None):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L91
        raise NotImplementedError("Combine function not implemented.")


# SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L101
class TokenDispatcherWithMC2(MoETokenDispatcher[MoEMC2CombineMetadata]):
    def __init__(self, **kwargs):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L102
        super().__init__(**kwargs)
        # SUBTRACTED: mc2 group/hccl comm name 解析、enable_dispatch_v2 探测、need_extra_args(A3/A5)、
        #             global_bs = max_bs_per_rank*ep_world_size 计算、need_comm_alg(hierarchy) 等
        #             硬件代际拼装（原 token_dispatcher.py:L102-L149）——host 无 NPU 无法触发。
        self.global_bs = 0

    # SUBTRACTED: get_dispatch_mc2_kwargs（原 L151-L224）——quant_mode/A3-A5 extra_args/hierarchy/
    #             global_bs 的入参拼装细节；保留 token_dispatch『调融合算子 → 拆 expand_x/expert_token_nums
    #             /combine metadata』的形状契约。

    # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L226
    def token_dispatch(self, token_dispatch_input: MoETokenDispatchInput):
        kwargs_mc2 = self.get_dispatch_mc2_kwargs(token_dispatch_input)
        # MC2 把『按专家 all-to-all 重分发 token』整个塞进一个 NPU 融合算子。
        output = torch_npu.npu_moe_distribute_dispatch_v2(**kwargs_mc2)
        (
            expand_x,            # 本卡专家收到的 token
            dynamic_scale,
            assist_info_for_combine,
            expert_token_nums,   # 每个专家几条
            ep_recv_counts,
            tp_recv_counts,
            expand_scales,
        ) = output[0:7]

        group_list_type = kwargs_mc2["expert_token_nums_type"]
        return MoETokenDispatchOutput(
            hidden_states=expand_x,
            dynamic_scale=dynamic_scale,
            group_list=expert_token_nums,
            group_list_type=group_list_type,
            combine_metadata=MoEMC2CombineMetadata(
                topk_ids=token_dispatch_input.topk_ids,
                topk_weights=token_dispatch_input.topk_weights,
                expert_map=token_dispatch_input.routing.expert_map,
                ep_recv_counts=ep_recv_counts,
                tp_recv_counts=tp_recv_counts,
                assist_info_for_combine=assist_info_for_combine,
                expand_scales=expand_scales,
                quant=token_dispatch_input.quant,
                mc2_mask=token_dispatch_input.routing.mc2_mask if self.global_bs == 0 else None,
            ),
        )

    def get_dispatch_mc2_kwargs(self, token_dispatch_input):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L151
        # SUBTRACTED: 见上——入参拼装细节整体省略，测试以 stub 注入返回 dict。
        raise NotImplementedError("kwargs 拼装含 A3/A5/quant 细节，host 不真跑")

    # SUBTRACTED: get_combine_mc_kwargs（原 L266-L328）——对称的 combine 入参拼装。

    # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L330
    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."
        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states, combine_metadata)
        combined_output = torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2)
        return combined_output

    def get_combine_mc_kwargs(self, hidden_states, combine_metadata):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L266
        # SUBTRACTED: 见上。
        raise NotImplementedError("kwargs 拼装含 quant 细节，host 不真跑")


# SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L343
class TokenDispatcherWithAllGather(MoETokenDispatcher[MoEAllGatherCombineMetadata]):
    def __init__(self, **kwargs):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L344
        super().__init__(**kwargs)
        self.max_num_tokens = kwargs.get("max_num_tokens")
        num_experts_local = kwargs.get("num_local_experts", 0)
        self.num_experts_local = (
            num_experts_local.item() if torch.is_tensor(num_experts_local) else int(num_experts_local)
        )

    # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L352
    def token_dispatch(self, token_dispatch_input: MoETokenDispatchInput):
        # SUBTRACTED: with_quant/MXFP4 的 quant_mode 选择、apply_router_weight_on_input 分支
        #             （原 L356-L382）——量化与权重前置是正交维度。
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        expert_map = token_dispatch_input.routing.expert_map
        dynamic_scale = token_dispatch_input.routing.pertoken_scale
        global_redundant_expert_num = token_dispatch_input.routing.global_redundant_expert_num
        restore_shape = hidden_states.shape
        quant_mode = -1

        num_tokens = hidden_states.shape[:-1].numel()
        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            mask = expert_map[topk_ids] != -1
            topk_weights = topk_weights * mask
            # active_expert_range 圈出本卡负责的专家区间
            first_expert_idx = get_ep_group().rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local
        # AllGather 路径不跨卡 all_to_all——hidden_states 已被 prepare 阶段 all-gather 成全局张量，
        # 这里只在本地用 npu_moe_init_routing 把 token 按 expert 排序聚拢。
        sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,
            scale=dynamic_scale,
            active_num=num_tokens * self.top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=quant_mode,
        )
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode

        return MoETokenDispatchOutput(
            hidden_states=sorted_hidden_states,
            dynamic_scale=None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                restore_shape=restore_shape,
            ),
        )

    # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L419
    def token_combine(self, hidden_states, combine_metadata, bias=None):
        # NOTE: 用 npu_moe_token_unpermute 而非 npu_moe_finalize_routing（后者有精度问题，workaround）。
        final_hidden_states = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=torch.abs(combine_metadata.expanded_row_idx),
            probs=combine_metadata.topk_weights,
        )
        if len(combine_metadata.restore_shape) == 3:
            final_hidden_states = final_hidden_states.view(combine_metadata.restore_shape)
        return final_hidden_states


# SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L432
class TokenDispatcherWithAll2AllV(MoETokenDispatcher[MoEAllToAllCombineMetadata]):
    """AlltoAll-based token dispatcher：各卡 dispatch 整条序列、hidden state 被切分。
    回收 f3：ch06 NPUCommunicator.all_to_all 的形状代数，在这里第一次有了『按专家路由』的用武之地。
    """

    def __init__(self, **kwargs):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L439
        super().__init__(**kwargs)
        self.num_local_experts = kwargs.get("num_local_experts", 0)
        assert self.num_local_experts > 0, "Expected at least one expert"
        # SUBTRACTED: expert_ids_per_ep_rank 张量、local_expert_indices 连续性 assert、
        #             mc2 group/hccl comm name 解析（原 L444-L463）——多本地专家/分布式句柄细节，
        #             host 无 NPU 不真跑。

    # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L465
    def token_dispatch(self, token_dispatch_input: MoETokenDispatchInput):
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids

        # dispatch 阶段先按 topk_ids 把本卡 token permute 排好（含各专家收到的 token 数 input/output_splits）
        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)

        # SUBTRACTED: with_quant 的 npu_dynamic_quant + 第二趟 all_to_all 传 scale 的分支
        #             （原 L485-L492）——量化正交维度。

        # 调 async_all_to_all 按 input_splits/output_splits 把 token 真正跨 EP 卡重分发。
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits, self.ep_group
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # SUBTRACTED: _dispatch_postprocess（多本地专家时按 local-expert-index 再 permute 一次，
        #             原 L500-L508/L621-L642）——保留单一重分发主线。
        reversed_global_input_permutation_mapping = None

        return MoETokenDispatchOutput(
            hidden_states=global_input_tokens,
            dynamic_scale=None,
            group_list=tokens_per_expert,
            group_list_type=1,
            combine_metadata=MoEAllToAllCombineMetadata(
                input_splits=input_splits,
                output_splits=output_splits,
                topk_weights=topk_weights,
                reversed_local_input_permutation_mapping=reversed_local_input_permutation_mapping,
                reversed_global_input_permutation_mapping=reversed_global_input_permutation_mapping,
                hidden_shape=hidden_shape,
                hidden_shape_before_permute=hidden_shape_before_permute,
            ),
        )

    # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L526
    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."
        # 1. Preprocess
        hidden_states = self._combine_preprocess(hidden_states, combine_metadata)
        # 2. AllToAll —— 对称地再 all_to_all 一次把专家输出送回原卡（注意 input/output_splits 互换）
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            combine_metadata.input_splits,
            combine_metadata.output_splits,
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)
        # 3. Postprocess —— unpermute 回原 token 顺序、按 topk_weights 加权
        output = self._combine_postprocess(permutated_local_input_tokens, combine_metadata)
        return output

    # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L547
    def _dispatch_preprocess(self, hidden_states, topk_ids):
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        (
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            num_out_tokens,
        ) = self._preprocess(topk_ids)
        hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=num_out_tokens,
        )
        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            global_input_tokens_local_experts_indices,
            hidden_shape,
            hidden_shape_before_permute,
        )

    def _preprocess(self, topk_ids):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L576
        # SUBTRACTED: histc 统计各专家 token 数 → reshape(ep_size, num_local_experts).sum → input/output_splits
        #             + gather_from_sequence_parallel_region 求全局计数（原 L576-L619）。这是 all2all-v
        #             不等长 split 的真正来源；host 无 NPU/分布式不真跑，测试以 stub 返回固定 splits。
        raise NotImplementedError("histc/all-gather 统计 splits，host 不真跑")

    def _combine_preprocess(self, hidden_states, combine_metadata):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L644
        # SUBTRACTED: 多本地专家时 npu_moe_token_unpermute(rev_global) 还原（原 L644-L651）。
        return hidden_states

    def _combine_postprocess(self, permutated_local_input_tokens, combine_metadata):
        # SOURCE: vllm_ascend/ops/fused_moe/token_dispatcher.py:L653
        # Unpermutation 1: AlltoAll output → output，按 topk_weights 加权
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=combine_metadata.reversed_local_input_permutation_mapping.to(torch.int32),
            probs=combine_metadata.topk_weights,
            restore_shape=combine_metadata.hidden_shape_before_permute,
        )
        output = output.view(combine_metadata.hidden_shape)
        return output
