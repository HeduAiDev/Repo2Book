# Subtract-only companion for v3 ch17 — vllm/v1/structured_output/utils.py
# (pin v0.27.1 / 6e448d0ea)。本章只取 apply_grammar_bitmask（sample_tokens
# 的 bitmask 施加点，m15/ch9 接点）；ch09 的精简版持有同一函数的逐字版
# （同款裁剪：GPU async H2D 分支删除），本档沿用同一形态。
#
# SUBTRACTED: utils.py 的其余部分（wrap_compile / OutlinesVocabulary /
#   StructuredOutputLogitsProcessor 等——ch30 结构化输出域）。
# xgrammar 宿主缺席时由 CPU 内核替身代行 apply_token_bitmask_inplace 的
# 文档语义（禁用位清零→-inf），容器内有真内核时优先真内核（ch09 同款）。

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._host_seams import init_logger

if TYPE_CHECKING:
    from .._host_seams import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)

# SOURCE: vllm/utils/torch_utils.py PIN_MEMORY — HOST SEAM（无 CUDA 宿主）
PIN_MEMORY = False  # HOST SEAM


# SOURCE: vllm/v1/structured_output/utils.py xgr 惰性加载（LazyLoader
#   xgrammar）— HOST SEAM：优先真 xgrammar，缺席时 CPU 内核替身。
try:  # HOST SEAM
    import xgrammar as xgr  # type: ignore
except ImportError:  # HOST SEAM
    class _XgrammarSeam:  # HOST SEAM
        """apply_token_bitmask_inplace 的 CPU 替身：xgrammar 内核的文档语义——
        bitmask[i, j] 是行 i 第 j 个 32 位字，第 t 个 token 的允许位是
        (bitmask[i, t//32] >> (t%32)) & 1；允许位为 0 的 token 写成 -inf。"""
        @staticmethod
        # SOURCE: vllm/v1/structured_output/utils.py apply_token_bitmask_inplace 调用面
        def apply_token_bitmask_inplace(logits, bitmask, indices=None):  # HOST SEAM
            import torch as _torch

            rows = (
                _torch.arange(logits.shape[0])
                if indices is None
                else _torch.as_tensor(indices, dtype=_torch.long)
            )
            words = bitmask.to(_torch.int64)[rows]  # [n_rows, n_words]
            vocab = logits.shape[1]
            token_idx = _torch.arange(vocab)
            allowed = (
                words[:, token_idx // 32] >> (token_idx % 32)
            ) & 1  # [n_rows, vocab]
            allowed = allowed.to(dtype=_torch.bool, device=logits.device)
            logits[rows] = _torch.where(
                allowed, logits[rows], _torch.tensor(float("-inf"))
            )

    xgr = _XgrammarSeam  # type: ignore[assignment]


# SOURCE: vllm/v1/structured_output/utils.py:L86-L175 apply_grammar_bitmask —
# 逐字 minus GPU async H2D 分支
# SOURCE: (见 impl-notes.md §Source Map——structured_output/utils.py)
def apply_grammar_bitmask(
    scheduler_output: "SchedulerOutput",
    grammar_output: "GrammarOutput",
    input_batch,
    logits: torch.Tensor,
) -> None:
    """
    Apply grammar bitmask to output logits of the model with xgrammar function.

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        input_batch (InputBatch): The input of model runner.
        logits (torch.Tensor): The output logits of model forward.
    """
    # Serialization of np.ndarray is much more efficient than a tensor,
    # so we receive it in that format.
    grammar_bitmask = grammar_output.grammar_bitmask

    # We receive the structured output bitmask from the scheduler,
    # compacted to contain bitmasks only for structured output requests.
    # The order of the requests in the bitmask is not guaranteed to be the
    # same as the order of the requests in the gpu runner's batch. We need
    # to sort the bitmask to match the order of the requests used here.

    # Get the batch indices of the structured output requests.
    # Keep track of the number of speculative tokens scheduled for every
    # request in the batch, as the logit indices are offset by this amount.
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices = []

    # Reorder the bitmask to match the order of the requests in the batch.
    sorted_bitmask_tensor = torch.full(
        (logits.shape[0], grammar_bitmask.shape[1]),
        -1,
        dtype=torch.from_numpy(grammar_bitmask[:0]).dtype,
        pin_memory=PIN_MEMORY,
    )
    sorted_bitmask = sorted_bitmask_tensor.numpy()
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens

    # SUBTRACTED: GPU async H2D 分支（utils.py:L137-L149——`if not
    #   logits.is_cpu:` 的 index_tensor 异步上传与 xgr GPU 内核路径；本档
    #   锚定 CPU logits 直算路径）。
    # Copy async to device.
    grammar_bitmask = sorted_bitmask_tensor.to(logits.device, non_blocking=True)

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    # CPU case, use list for indices.
    indices = None if skip_out_indices else out_indices
    # Handle dtype conversion for CPU (older xgrammar CPU kernels require float32)
    # See: https://github.com/vllm-project/vllm/issues/31901
    if logits.dtype != torch.float32:
        # Convert to float32, apply bitmask, then convert back
        logits_fp32 = logits.to(torch.float32)
        xgr.apply_token_bitmask_inplace(logits_fp32, grammar_bitmask, indices=indices)
        # Copy the modified values back to the original tensor
        logits.copy_(logits_fp32.to(logits.dtype))
    else:
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=indices)
