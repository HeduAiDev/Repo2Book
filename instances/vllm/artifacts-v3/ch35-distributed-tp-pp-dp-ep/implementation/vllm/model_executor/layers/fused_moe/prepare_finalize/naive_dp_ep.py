# SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
# ch34 切面（m22 消费现场）：MoEPrepareAndFinalizeNaiveDPEPModular 的 prepare
# 尾段（get_ep_group().dispatch）与 finalize 尾段（combine + output.copy_）。
# 删除（dossier 删除项 9）：MoEPrepareAndFinalizeNaiveDPEPMonolithic 整类（L212-L283）、
# _quantize_and_setup_dispatch/_unwrap_scale 的量化细节压成两行注释（归 ch27）、
# LoRA 随行映射段。

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
# SUBTRACTED: moe_kernel_quantize_input / nvfp4_block_scale_interleave import
#   （L12-L13）——量化细节归 ch27（删除项 9）。

# SUBTRACTED: _quantize_and_setup_dispatch（L16-L52）与 _unwrap_scale_and_
#   prepare_for_moe（L55-L68）——量化细节压成两行注释：prepare 路径上量化后
#   的输入 a1q（与可选 scale）随 dispatch 全网重排、combine 后再按 kernel 需要
#   解包/交织；细节归 ch27。


# SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
#   :L71-L209 MoEPrepareAndFinalizeNaiveDPEPModular —— 逐字 minus 删除项 9
class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the topk weights and ids.
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L79-L91（锚点双置）
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers
        # Set by FusedMoEWithLoRA.set_mapping() when LoRA is active. When
        # present, prepare() dispatches the per-token LoRA mapping alongside
        # hidden_states and writes the gathered result back to the context so
        # experts can use the per-rank-local mapping.
        self._lora_context = None

    def set_lora_context(self, ctx) -> None:
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L93-L94（锚点双置）
        self._lora_context = ctx

    @property
    # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/
    #   naive_dp_ep.py:L97-L98 activation_format
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L97-L98（锚点双置）
        return mk.FusedMoEActivationFormat.Standard

    @property
    # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/
    #   naive_dp_ep.py:L100-L101 max_num_tokens_per_rank
    def max_num_tokens_per_rank(self) -> int | None:
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L100-L101（锚点双置）
        return None

    @property
    # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/
    #   naive_dp_ep.py:L103-L104 topk_indices_dtype
    def topk_indices_dtype(self) -> torch.dtype | None:
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L103-L104（锚点双置）
        return None

    @property
    # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/
    #   naive_dp_ep.py:L106-L107 num_dispatchers
    def num_dispatchers(self) -> int:
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L106-L107（锚点双置）
        return self._num_dispatchers

    @property
    # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/
    #   naive_dp_ep.py:L109-L110 output_is_reduced
    def output_is_reduced(self) -> bool:
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L109-L110（锚点双置）
        return False

    # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/
    #   naive_dp_ep.py:L112-L185 prepare —— 主干逐字：apply_router_weight_on_input 的
    #   topk=1 防御、量化/LoRA 的 extra_tensors 装配（删除项 9 压缩）、dispatch
    #   调用位与三元组解包逐字
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        """Quantize and Dispatch Topk Weights and Topk Ids."""
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L112-L185（锚点双置）

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # SUBTRACTED: _quantize_and_setup_dispatch 的量化装配（L132-L134）——
        #   删除项 9：a1q=a1、scales=None（无量化路径）。
        a1q, scales, a1q_scale_orig = a1, None, None

        # SUBTRACTED: LoRA per-token 映射随行段（L136-L156）——LoRA 域。

        extra_tensors: list[torch.Tensor] | None = None
        if scales is not None:
            extra_tensors = list(scales)

        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

        if extra_tensors is None:
            assert len(res) == 3
            a1q, topk_weights, topk_ids = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 4
            a1q, topk_weights, topk_ids, gathered_extras = res
            gathered_extras = list(gathered_extras)
            # SUBTRACTED: dispatched_lora_mapping 写回与 scale 解包（L174-L183）——
            #   LoRA/量化域（删除项 9）。
            a1q_scale = a1q_scale_orig

        return a1q, a1q_scale, None, topk_ids, topk_weights

    # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/
    #   naive_dp_ep.py:L187-L209 finalize —— 逐字（weight_and_reduce 施加 + combine
    #   归位 + output.copy_ 覆写）
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L187-L209（锚点双置）
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        output.copy_(
            get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
        )


# SUBTRACTED: MoEPrepareAndFinalizeNaiveDPEPMonolithic（L212-L283）——删除项 9
#   （router 内置变体）；make_moe_prepare_and_finalize_naive_dp_ep 工厂
#   （L286-L301）随之裁除。
