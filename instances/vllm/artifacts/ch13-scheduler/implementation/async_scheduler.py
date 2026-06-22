# SOURCE: vllm/v1/core/sched/async_scheduler.py
# AsyncScheduler 继承 Scheduler，只覆写 _update_after_schedule /
# _update_request_with_output，用 num_output_placeholders 占位机制解决‘调度本拍时
# 上一拍 token 还没算完’，让 schedule(N) 与 forward(N-1) 重叠（async_scheduling=True
# 时 EngineCore 实例化它）—— 回收 f6。与真实文件 1:1，无删除。
from __future__ import annotations

from .output import SchedulerOutput
from .request import RequestStatus
from .scheduler import Scheduler


# SOURCE: vllm/v1/core/sched/async_scheduler.py:L12 AsyncScheduler
class AsyncScheduler(Scheduler):
    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L13 __init__
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens

    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L18 _update_after_schedule
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L37 _update_request_with_output
    def _update_request_with_output(
        self, request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # SUBTRACTED: discard_latest_async_tokens 分支（reset_prefix_cache 强制抢占下
        #   丢弃最新 async token，dossier.delete 批准的 KVConnector/缓存重置边角）。
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
