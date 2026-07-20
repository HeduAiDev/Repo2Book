# SOURCE: vllm/v1/structured_output/utils.py
# 只做减法的忠实精简版。这是本章 m11「两条并存路径」里默认生效的那一条——
# VLLM_USE_V2_MODEL_RUNNER 默认 False 时，gpu_model_runner.py 调的就是这个
# apply_grammar_bitmask，而不是 worker/gpu/structured_outputs.py 里 vLLM 自写的
# Triton kernel。两者做同一件事（把非法 token 的 logit 打成 -inf），但这一条直接
# 调 xgrammar 库自带的 xgr.apply_token_bitmask_inplace。
#
# SUBTRACTED: SPDX 版权头、CPU 兜底分支（`if not logits.is_cpu: ... return` 之后的
# float32 转换与手工回写，utils.py 原函数末尾）——只在 CPU 后端且老版本 xgrammar
# 内核下触发，与本章的 GPU 主路径正确性无关（subtraction_plan 批准项3）。
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from output import GrammarOutput, SchedulerOutput
    from input_batch import InputBatch

try:
    import xgrammar as xgr
except ImportError:  # host 未装 xgrammar，见 dossier.analyst_notes_on_plan
    xgr = None


def apply_grammar_bitmask(
    scheduler_output: "SchedulerOutput",
    grammar_output: "GrammarOutput",
    input_batch: "InputBatch",
    logits: "torch.Tensor",
) -> None:
    # SOURCE: vllm/v1/structured_output/utils.py:L44-105（GPU 主路径；CPU 分支已删）
    """
    Apply grammar bitmask to output logits of the model with xgrammar function.
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
    sorted_bitmask = np.full(
        shape=(logits.shape[0], grammar_bitmask.shape[1]),
        fill_value=-1,
        dtype=grammar_bitmask.dtype,
    )
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens

    # Copy async to device as tensor.
    grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
        logits.device, non_blocking=True
    )

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    index_tensor = None
    if not skip_out_indices:
        # xgrammar expects a python list of indices but it will actually work with
        # a tensor. If we copy the tensor ourselves here we can do it in a
        # non_blocking manner and there should be no cpu sync within xgrammar.
        index_tensor = torch.tensor(
            out_indices, dtype=torch.int32, device="cpu", pin_memory=False
        )
        index_tensor = index_tensor.to(logits.device, non_blocking=True)

    xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
