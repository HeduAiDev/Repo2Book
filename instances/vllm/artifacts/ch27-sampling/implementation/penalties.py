# 只做减法的忠实精简版 —— 镜像三处真实源码（pin f3fef123）：
#   - vllm/v1/sample/ops/penalties.py     （apply_all_penalties 张量化 wrapper）
#   - vllm/model_executor/layers/utils.py （apply_penalties 三种惩罚的真正算式）
#   - vllm/_custom_ops.py                 （apply_repetition_penalties 派发 + torch 实现）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: 三份文件各自的 SPDX 版权头与无关 import —— 仅许可证/无关依赖，不影响行为。
import numpy as np
import torch


# ===== vllm/utils 内嵌的两个小工具（精简版自包含，避免 import vllm）=====
# SUBTRACTED: 这两个辅助原本从 vllm.utils.torch_utils / vllm.utils.platform_utils 导入。
# 为让精简版在无 vLLM 环境可跑，原样内嵌；它们只做"列表补齐成张量"与"是否有锁页内存"，
# 不属本章采样逻辑。

def is_pin_memory_available() -> bool:
    # SOURCE: vllm/utils/platform_utils.py:L43（host 无加速器 → False，行为等价）
    return False


def make_tensor_with_pad(
    x, pad, dtype, *, max_len=None, device=None, pin_memory=False
):
    # SOURCE: vllm/utils/torch_utils.py:L644-L666（+ 内联 make_ndarray_with_pad:L587-609）
    if max_len is None:
        max_len = max(map(len, x), default=0)
    np_dtype = {torch.int64: np.int64, torch.int32: np.int32}[dtype]
    padded_x = np.full((len(x), max_len), pad, dtype=np_dtype)
    for ind, row in enumerate(x):
        assert len(row) <= max_len
        padded_x[ind, : len(row)] = row
    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()
    return tensor


# ===== vllm/_custom_ops.py: 重复惩罚的真正算式 =====
# SUBTRACTED: apply_repetition_penalties_cuda（_custom_ops.py:L487-495）——
# 仅 CUDA contiguous 时调 torch.ops._C 的融合内核；与下面的 torch 版数值等价，
# host 无 CUDA，本精简版只保留可运行的 torch 路径，派发器据此分流。


def apply_repetition_penalties_torch(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    # SOURCE: vllm/_custom_ops.py:L472-L485
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    # SOURCE: vllm/_custom_ops.py:L499-L520
    """Apply repetition penalties to logits in-place."""
    if logits.is_cuda and logits.is_contiguous():
        # SUBTRACTED: apply_repetition_penalties_cuda 分支（_custom_ops.py:L512-515）——
        # CUDA 融合内核，与 torch 版数值等价；host 无 CUDA 不走此路。
        apply_repetition_penalties_torch(
            logits, prompt_mask, output_mask, repetition_penalties
        )
    else:
        apply_repetition_penalties_torch(
            logits, prompt_mask, output_mask, repetition_penalties
        )


# ===== vllm/model_executor/layers/utils.py: 三种惩罚的落地算式 =====
def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SOURCE: vllm/model_executor/layers/utils.py:L34-48
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    # SOURCE: vllm/model_executor/layers/utils.py:L51-89
    """Applies penalties in place to the logits tensor."""
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    # SUBTRACTED: 原是 `from vllm._custom_ops import apply_repetition_penalties`；
    # 此处直接调用本文件上方内嵌的同名函数（vllm/_custom_ops.py 的派发器）。
    apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


# ===== vllm/v1/sample/ops/penalties.py: 张量化 wrapper =====
def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/penalties.py:L11-39
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(output_token_ids, vocab_size, logits.device)

    # In the async scheduling case, rows that won't have penalties applied may contain
    # -1 placeholder token ids. We must replace these with valid token ids so that the
    # scatter done in apply_penalties is valid.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _convert_to_tensors(
    output_token_ids: list[list[int]], vocab_size: int, device: torch.device
) -> torch.Tensor:
    # SOURCE: vllm/v1/sample/ops/penalties.py:L42-57
    """
    Convert the different list data structures to tensors.
    """
    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=True)
