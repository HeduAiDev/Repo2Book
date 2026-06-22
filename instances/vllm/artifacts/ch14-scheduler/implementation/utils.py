# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版。验收：删掉 # SUBTRACTED 分支 ≈ 真实 vLLM。
import contextlib

from request import Request, RequestStatus


# SOURCE: vllm/v1/core/sched/utils.py:L62 (remove_all)
def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


# SOURCE: vllm/v1/core/sched/utils.py:L94 (check_stop)
def check_stop(request: Request, max_model_len: int) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    # SUBTRACTED: repetition_detection 分支（check_sequence_repetition）——
    # 重复检测的模式匹配算法（原 vllm/v1/core/sched/utils.py:L10-L59,L119-L128）属
    # 停止判定细节；精简版 repetition_detection 恒为 None，停止主线（EOS/stop_token/
    # length）完整。dossier embed_excerpts 已将此标为可 elide。

    return False
