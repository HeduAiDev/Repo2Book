# SOURCE: vllm/v1/core/sched/utils.py
# 判停与摘除的工具函数——本章四个全在路径上：check_stop 五连判（m13）、
# remove_all 批量摘除（m20 单元素快路径）、check_sequence_repetition 重复
# 检测（五连判第五判的谓词）。与真实 vllm/v1/core/sched/utils.py 同名同结构，
# 无删除（130 行全保留；RepetitionDetectionParams 从 request.py 模块引入）。
from __future__ import annotations

import contextlib
from collections.abc import Sequence

from .request import RepetitionDetectionParams, Request, RequestStatus


# SOURCE: vllm/v1/core/sched/utils.py:L10 _has_repeating_pattern
def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last pattern_len tokens against the preceding
    (repetition_min_count - 1) repetitions of the same length.
    """
    # SOURCE: vllm/v1/core/sched/utils.py:L20-L25
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


# SOURCE: vllm/v1/core/sched/utils.py:L28 check_sequence_repetition
def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.
    Args:
        token_ids: List of token IDs
        params: Repetition detection parameters.
    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    # SOURCE: vllm/v1/core/sched/utils.py:L39-L41
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    # SOURCE: vllm/v1/core/sched/utils.py:L43-L45
    if min_pattern_size <= 0:
        min_pattern_size = 1

    # SOURCE: vllm/v1/core/sched/utils.py:L46-L47
    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    # SOURCE: vllm/v1/core/sched/utils.py:L49-L58
    for pattern_len in range(
        min_pattern_size,
        max_pattern_size + 1,
    ):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


# SOURCE: vllm/v1/core/sched/utils.py:L62 remove_all —— 单元素快路径（m20）
def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from.
        items_to_remove: Set of items to remove.

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
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
    # （整型 id 级——stop string 的子串匹配在前端 detokenizer，不在此处）
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
    repetition_detection = sampling_params.repetition_detection
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False
