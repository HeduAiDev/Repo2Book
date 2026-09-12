# SOURCE: vllm/v1/structured_output/utils.py
# v3 ch31 脊柱②：apply_grammar_bitmask（L86-L175，V1 正典落地，本章 payoff）——
# 紧凑掩码按 worker 批序+spec 偏移重排进 pinned sorted_bitmask → 非阻塞 H2D →
# xgr.apply_token_bitmask_inplace 原地 -inf。
# SUBTRACTED（delete[1]）：compile_regex_with_timeout（L48-L83，ReDoS 超时）与
#   OutlinesVocabulary/OutlinesDiskCache/get_outlines_cache/词表归约/lark↔EBNF
#   转换/choice_as_grammar（L178-L561）——全部归 ch30（m23/m24 编译侧）。
# SUBTRACTED（delete[4]）：apply_grammar_bitmask 的 CPU 后端兜底分支
#   （L164-L175：indices 用 python list + 老版 xgrammar CPU 核的 fp32 转换
#   回写，#31901）——只在 CPU 设备+老版本 xgrammar 核触发，与 GPU 主路径
#   正确性无关（正文以对照形式一句话带过）。
# SUBTRACTED（delete[7]）：LazyLoader 样板中除 xgr 外的全部（oc/file_utils/
#   convert_slow_tokenizer——其消费方已删）；xgr 的 LazyLoader 保留（host 无
#   xgrammar，eager import 会炸 import 期——LazyLoader 延迟到首次属性访问，
#   与真实运行时语义一致）。
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.utils.import_utils import LazyLoader
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.v1.worker.gpu_input_batch import InputBatch
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
# SUBTRACTED: 真实文件 L5-L17 的 regex/cachetools/sqlite3/tempfile 等 import
#   与 L24-L38 的 oc/file_utils/convert_slow_tokenizer LazyLoader 位
#   （消费方全在已删的 outlines 编译侧）；InputBatch 的导入真实在
#   vllm/v1/worker/gpu_input_batch（本章镜像路径 v1/worker/gpu/ 下无此文件，
#   以 TYPE_CHECKING 注解面承载——本函数只消费 input_batch.req_ids）。


#   （仅删 L164-L175 CPU 兜底分支，delete[4]）
# SOURCE: vllm/v1/structured_output/utils.py:L86-L175 apply_grammar_bitmask —— 逐字
def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
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

    # Copy async to device.
    grammar_bitmask = sorted_bitmask_tensor.to(logits.device, non_blocking=True)

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    if not logits.is_cpu:
        index_tensor = None
        if not skip_out_indices:
            # xgrammar expects a python list of indices but it will actually work with
            # a tensor. If we copy the tensor ourselves here we can do it in a
            # non_blocking manner and there should be no cpu sync within xgrammar.
            index_tensor = async_tensor_h2d(
                out_indices, dtype=torch.int32, device=logits.device
            )

        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
        return

    # SUBTRACTED: vllm/v1/structured_output/utils.py:L164-L175 CPU 后端兜底分支
    #   （indices 用 python list + 老版 xgrammar CPU 核 fp32 转换回写，#31901）
    #   ——delete[4]：只在 CPU 设备+老版本 xgrammar 核触发，与 GPU 主路径
    #   正确性无关。删后 CPU 张量原样通过（本精简版固定演示 GPU 路径）。


# SUBTRACTED: vllm/v1/structured_output/utils.py:L48-L83 compile_regex_with_timeout
#   （ReDoS 超时防护）与 L178-L561 OutlinesVocabulary/OutlinesDiskCache/
#   get_outlines_cache/re_llama_byte_token 正则/_reduced_vocabulary/
#   get_outlines_vocabulary/grammar_is_likely_lark/convert_lark_to_ebnf/
#   choice_as_grammar —— delete[1]，全部归 ch30 编译侧（m23/m24）。
