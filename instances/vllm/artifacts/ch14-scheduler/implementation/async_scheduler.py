# SPDX-License-Identifier: Apache-2.0
# 只做减法的精简版。验收：删掉 # SUBTRACTED 分支 ≈ 真实 vLLM。
from request import Request, RequestStatus
from scheduler import Scheduler


# SOURCE: vllm/v1/core/sched/async_scheduler.py:L12 (class AsyncScheduler)
class AsyncScheduler(Scheduler):
    # SUBTRACTED: __init__ 的 _spec_token_placeholders / _update_after_schedule 的
    # num_output_placeholders 预增 + spec 占位 —— 异步调度的占位簿记前半段，由调度
    # 阶段填充；本章只需覆写 _update_request_with_output 的回扣后半段。
    # 原 vllm/v1/core/sched/async_scheduler.py:L13-L35。

    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L37 (_update_request_with_output)
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> "tuple[list[int], bool]":
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
