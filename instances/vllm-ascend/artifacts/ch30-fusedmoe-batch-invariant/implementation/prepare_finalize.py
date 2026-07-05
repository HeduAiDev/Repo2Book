"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/ops/fused_moe/prepare_finalize.py

通信前置(prepare)/后置(finalize)。AllGather 做 DP all-gather + DP reduce-scatter；
MC2/All2All 做 pad 到 TP 边界 + TP 切片，finalize 时 all-gather 回拼并 unpad。
"""
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEPrepareOutput
from vllm_ascend.quantization.quant_type import QuantType


# SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L40
class PrepareAndFinalize(ABC):
    """Abstract base for MoE tensor preparation/finalization in distributed envs.
    Subclasses implement AllGather / All2All / MC2 communication strategies."""

    def __init__(self, moe_config):
        # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L54
        self.moe_config = moe_config
        # SUBTRACTED: multistream_overlap_gate / quant_stream（原 L52-L59）——gate 并流性能优化，
        #             不改数值，本精简版不走并流。
        get_ascend_config()

    @abstractmethod
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L61
        """Prepare tensors before MoE: pad to comm boundary / slice across TP /
        broadcast across DP."""
        raise NotImplementedError("Prepare not implemented.")

    # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L93
    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape=None,
    ) -> torch.Tensor:
        """Finalize MoE output: gather TP slices / reduce-scatter across DP / unpad."""
        raise NotImplementedError("Finalize function not implemented.")


# SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L116
class PrepareAndFinalizeWithAll2All(PrepareAndFinalize):
    """All-to-All style slicing：pad 到 TP size、TP>1 时切片取本 rank 份。"""

    def __init__(self, moe_config):
        # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L123
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L127
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L132
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp

        padded_hidden_states_shape = hidden_states.shape
        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            self.num_tokens, _ = hidden_states.shape
            pad_size = self.tp_size - self.num_tokens  # Pad to TP size (cyclic)

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)
                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )

    # SUBTRACTED: pad_and_split_input_ids（原 L179-L191）——input_ids 旁路，与隐藏态主线无关。

    # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L193
    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape=None,
    ) -> torch.Tensor:
        if not (self.enable_shared_expert_dp or self.replace_allreduce):
            if self.tp_size > 1:
                assert padded_hidden_states_shape is not None
                gathered_hidden_states = torch.empty(
                    padded_hidden_states_shape, device=hidden_states.device, dtype=hidden_states.dtype
                )
                split_hidden_states = torch.tensor_split(gathered_hidden_states, self.tp_size, dim=0)
                dist.all_gather(list(split_hidden_states), hidden_states, self.moe_config.tp_group.device_group)
                hidden_states = gathered_hidden_states

            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[: self.num_tokens]  # unpad
        return hidden_states


# SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L228
class PrepareAndFinalizeWithMC2(PrepareAndFinalizeWithAll2All):
    """MC2 基于 All2All，故继承它、共用 finalize；prepare 额外用 mc2_mask 与 padded_num_tokens 对齐。"""

    def __init__(self, moe_config):
        # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L123
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L127
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L249
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp
        mc2_mask = _EXTRA_CTX.mc2_mask
        if self.tp_size > 1:
            split_mc2_mask = torch.tensor_split(mc2_mask, self.tp_size, dim=0)
            mc2_mask = split_mc2_mask[self.tp_rank]

        padded_hidden_states_shape = hidden_states.shape
        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            target_pad_length = _EXTRA_CTX.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            if pad_size > 0 and not self.enable_shared_expert_dp:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
                padded_hidden_states_shape = hidden_states.shape

            if self.tp_size > 1 and not self.enable_shared_expert_dp:
                split_hidden_states = torch.tensor_split(hidden_states, self.tp_size, dim=0)
                split_router_logits = torch.tensor_split(router_logits, self.tp_size, dim=0)
                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=mc2_mask,
            padded_hidden_states_shape=padded_hidden_states_shape,
            pertoken_scale=None,
        )


# SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L321
class PrepareAndFinalizeWithAllGather(PrepareAndFinalize):
    """All-Gather (DP) + Reduce-Scatter (DP) on the DP group.
    无 SP 时：Attn → TP AR → DP AG → MoE → DP RS → TP AR。"""

    # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L343
    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        # SUBTRACTED: enable_sp()/enable_sp_by_pass() 时走 _prepare_with_ep_group（SP/flashcomm 叠加，
        #             原 L358-L401）；本精简版只演示基本 DP 路径。
        return self._prepare_with_dp_group(hidden_states, router_logits, enable_shared_expert_dp, replace_allreduce)

    # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L403
    def _prepare_with_dp_group(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type=QuantType.NONE,
    ) -> MoEPrepareOutput:
        self.enable_shared_expert_dp = enable_shared_expert_dp
        if self.moe_config.dp_size > 1:
            max_tokens_across_dp = _EXTRA_CTX.max_tokens_across_dp
            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits, (0, 0, 0, pad_size))
            # All-gather across DP group → 全局输入张量
            hidden_states = self.moe_config.dp_group.all_gather(hidden_states, 0)
            router_logits = self.moe_config.dp_group.all_gather(router_logits, 0)

        # SUBTRACTED: pcp_size>1 的 PCP all-gather 分支（原 L434-L450）——更上层并行模式叠加。
        return MoEPrepareOutput(
            hidden_states=hidden_states,
            router_logits=router_logits,
            mc2_mask=None,
            padded_hidden_states_shape=None,
            pertoken_scale=None,
        )

    # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L470
    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape=None,
    ) -> torch.Tensor:
        # SUBTRACTED: enable_sp 时走 _finalize_with_ep_group（原 L483-L500）。
        return self._finalize_with_dp_group(hidden_states, reduce_results)

    # SOURCE: vllm_ascend/ops/fused_moe/prepare_finalize.py:L502
    def _finalize_with_dp_group(self, hidden_states: torch.Tensor, reduce_results: bool) -> torch.Tensor:
        from vllm.distributed.parallel_state import get_dp_group

        if self.moe_config.dp_size > 1 and not self.enable_shared_expert_dp:
            hidden_states = get_dp_group().reduce_scatter(hidden_states, 0)
            hidden_states = hidden_states[: self.num_tokens]  # slice to local count
        # SUBTRACTED: pcp_size>1 的 PCP reduce_scatter（原 L516-L517）。
        return hidden_states
