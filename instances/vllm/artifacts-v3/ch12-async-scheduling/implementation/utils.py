# SOURCE: vllm/v1/core/sched/utils.py
# 判停与摘除的工具函数——update_from_output 收尾路径的消费面。与真实同名同
# 结构（五连判顺序即优先级；remove_all 单元素快路径）。
from __future__ import annotations

import contextlib
from collections.abc import Sequence

from .request import Request, RequestStatus


# SOURCE: vllm/v1/core/sched/utils.py:L94 check_stop —— 五连判（顺序即优先级）
def check_stop(request: Request, max_model_len: int) -> bool:
    # SOURCE: vllm/v1/core/sched/utils.py:L95-L98
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    # SOURCE: vllm/v1/core/sched/utils.py:L100-L101 第一判 min_tokens 门槛
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    # SOURCE: vllm/v1/core/sched/utils.py:L103-L106 第二判 EOS
    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    # SOURCE: vllm/v1/core/sched/utils.py:L108-L111 第三判 stop_token_ids
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    # SOURCE: vllm/v1/core/sched/utils.py:L112-L117 第四判 长度封顶
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # SOURCE: vllm/v1/core/sched/utils.py:L119-L128 第五判 重复检测
    # SUBTRACTED: 重复检测（RepetitionDetectionParams 面——ch11 全文已立；
    #   本章请求不带 repetition_detection，谓词恒 False 不再保留调用链）。

    return False


# SOURCE: vllm/v1/core/sched/utils.py:L62 remove_all —— 单元素快路径
def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set."""
    # SOURCE: vllm/v1/core/sched/utils.py:L81-L83
    if not items_to_remove:
        return lst

    # SOURCE: vllm/v1/core/sched/utils.py:L84-L89 单元素快路径
    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    # SOURCE: vllm/v1/core/sched/utils.py:L90-L91
    return [item for item in lst if item not in items_to_remove]


# SUBTRACTED: _has_repeating_pattern / check_sequence_repetition（L10-L70——
#   第五判谓词，随重复检测面归 ch11）。
