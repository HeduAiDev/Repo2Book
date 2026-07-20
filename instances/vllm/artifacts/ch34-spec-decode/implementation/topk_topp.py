# SUBTRACTED: SPDX 版权头与 flashinfer/aiter/triton 相关 import —— 本章只需
#             apply_top_k_top_p 的 PyTorch sort 实现，作为草稿位置目标 logits 的
#             top-k/top-p 约束。完整 TopKTopPSampler 见 ch27。
import torch


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L305-L315 (apply_top_k_top_p)
# SUBTRACTED: `if HAS_TRITON and logits.shape[0] >= 8: apply_top_k_top_p_triton`
#             二级分流（L263-L264）—— 纯性能；精简版固定走 PyTorch sort 实现，
#             对相同输入产生相同 mask。
def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L305-L315
    if p is None and k is None:
        return logits
    return apply_top_k_top_p_pytorch(logits, k, p)


# SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L318-L359
# SUBTRACTED: allow_cpu_sync 形参与 apply_top_k_only 免排序快路（L274, L287-L289）——
#             纯性能；精简版始终走排序路径，行为等价。
def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits (may update in-place)."""
    # SOURCE: vllm/v1/sample/ops/topk_topp_sampler.py:L318-L359
    if p is None:
        if k is None:
            return logits

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
