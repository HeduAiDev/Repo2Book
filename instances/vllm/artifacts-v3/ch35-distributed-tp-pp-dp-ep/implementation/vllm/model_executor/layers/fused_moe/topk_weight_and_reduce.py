# SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py
# ch34 切面：TopKWeightAndReduceDelegate 与 TopKWeightAndReduceContiguous
# （naive_dp_ep.finalize 的权重施加与归约实现）。Contiguous 近逐字——唯
# ops.moe_sum（fused kernel）以等义 torch.sum 承载（行内标注）。

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk


# SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py:L11-L41
#   TopKWeightAndReduceDelegate —— docstring 逐字 + __eq__ 逐字
class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):
    """
    Useful in the case when some FusedMoEExpertsModular
    implementation does not perform weight application and reduction
    but cannot address the needs of all the compatible PrepareAndFinalize
    implementations.
    For example, BatchedTritonExperts is compatible with both batched
    PrepareAndFinalize implementations like DeepEPLLPrepareAndFinalize and
    BatchedPrepareAndFinalize. Some PrepareAndFinalize implementations do
    the weight-application + reduction as part of the combine kernel, while
    BatchedPrepareAndFinalize needs an explicit implementation. To facilitate
    this case, the BatchedTritonExperts could use TopKWeightAndReduceDelegate
    so the PrepareAndFinalize implementations could choose how to
    weight + reduce.
    """

    def __eq__(self, other):
        # SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py:L27-L28（锚点双置）
        return isinstance(other, TopKWeightAndReduceDelegate)

    # SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py
    #   Delegate.apply —— 委派（真实版转投 DefaultImpl；本章载体按 ABC 契约
    #   直接以 Contiguous 语义实现——HOST/ch26 SEAM）
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py:L30-L41（锚点双置）
        return TopKWeightAndReduceContiguous().apply(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
        )


# SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py:L80-L121
#   TopKWeightAndReduceContiguous —— 逐字 minus ops.moe_sum
class TopKWeightAndReduceContiguous(mk.TopKWeightAndReduce):
    """
    TopKWeightAndReduce implementation for a fused_experts output
    of shape (m, topk, K)
    """
    # SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py:L80-L121（锚点双置）

    def __eq__(self, other):
        # SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py:L50-L51（锚点双置）
        return isinstance(other, TopKWeightAndReduceContiguous)

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py:L53-L77（锚点双置）
        m, num_topk = topk_ids.size()
        k = fused_expert_output.size(-1)
        if fused_expert_output.ndim == 2:
            fused_expert_output = fused_expert_output.view(m, num_topk, k)

        assert fused_expert_output.size() == (m, num_topk, k), (
            f"Expected fused_expert_output size {(m, num_topk, k)}. But got "
            f"{fused_expert_output.size()}"
        )

        if not apply_router_weight_on_input:
            fused_expert_output.mul_(topk_weights.view(m, -1, 1))

        if output is None:
            output = torch.empty(
                (m, k),
                device=fused_expert_output.device,
                dtype=fused_expert_output.dtype,
            )
        assert output.size() == (m, k), (
            f"Expected output size {(m, k)}. But got {output.size()}"
        )

        # SUBTRACTED: ops.moe_sum(fused_expert_output, output)（真实版为 fused
        #   kernel，ch26 域）——等义 torch.sum 承载（沿 topk 维求和）。
        torch.sum(fused_expert_output, dim=1, out=output)
        return output
