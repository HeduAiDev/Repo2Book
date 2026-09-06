# SOURCE: vllm/_custom_ops.py
# HOST SEAM：本章消费面一件——apply_repetition_penalties（惩罚 step6 的
# repetition 自定义 op，layers/utils.py:L81 惰性 import）。真实分派
# （_custom_ops.py:L336-L358）：CUDA 连续张量走 torch.ops._C 编译核
# （L325-L333），否则走 torch 参考算式（L309-L323）——二者数学同构。
# HOST SEAM 保留 torch 参考算式（逐字），减去 torch.ops._C 分派臂
# （精简版无 vLLM 编译扩展；正除负乘的真算式与数值不变）。
from __future__ import annotations

import torch


# SOURCE: vllm/_custom_ops.py:L309-L323 apply_repetition_penalties_torch —— 逐字（正 logit 除以 penalty、负 logit 乘以 penalty 的参考算式）
def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


# SUBTRACTED: vllm/_custom_ops.py:L325-L333 apply_repetition_penalties_cuda
#   （torch.ops._C.apply_repetition_penalties_ 编译核分派臂）与 L336-L358
#   apply_penalties 分派器里的 CUDA 判断——精简版无 vLLM 编译扩展注册的
#   torch.ops._C；上方 torch 参考算式即真实 CUDA 核的逐元素等价实现
#   （正除负乘、未出现位 no-op），数值不变。保留同名入口供
#   layers/utils.py:L81 的惰性 import 消费。


# SOURCE: vllm/_custom_ops.py:L336-L358 apply_repetition_penalties —— HOST SEAM：签名与 docstring 逐字，函数体按上方 SUBTRACTED 说明 直调 torch 参考算式
def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    apply_repetition_penalties_torch(
        logits, prompt_mask, output_mask, repetition_penalties
    )
