# ch28 penalties.py —— subtract-only 精简版
#
# 薄壳「同接口换内核」的实证：apply_all_penalties 与 vLLM 基座
# vllm/v1/sample/ops/penalties.py 的 apply_all_penalties 同名同签名，仅把内核从纯
# torch 换成昇腾 Triton kernel apply_penalties_triton。本文件几乎逐字保留——penalties
# 子系统没有可删的正交分支。
#
# SOURCE: vllm_ascend/sample/penalties.py
import torch
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad

from vllm_ascend.ops.triton.penalty import apply_penalties_triton


# SOURCE: vllm_ascend/sample/penalties.py:L13
def _convert_to_tensors(output_token_ids: list[list[int]], vocab_size: int, device: torch.device) -> torch.Tensor:
    """Convert output_token_ids (list of lists) to padded tensor."""
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)


# SOURCE: vllm_ascend/sample/penalties.py:L25
def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    """Apply penalties to logits via Triton-Ascend."""
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties_triton(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )
