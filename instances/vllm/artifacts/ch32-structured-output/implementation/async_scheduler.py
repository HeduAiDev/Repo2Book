# SOURCE: vllm/v1/core/sched/async_scheduler.py
# 只做减法的忠实精简版。AsyncScheduler 是异步调度模式下的 Scheduler 子类；本章只
# 关心它在 _update_after_schedule 里额外置位的 pending_structured_output_tokens——
# 告诉 engine core 本步的输出 token 还没落定，必须延后再算掩码（m15，见 engine_core.py
# 的 step_with_batch_queue）。
#
# SUBTRACTED: SPDX 版权头、__init__ 的其余部分（本章只需要 _spec_token_placeholders
# 这一个只读占位列表，用来标记「本步新产出的草稿位置」）。
from typing import TYPE_CHECKING

from scheduler import Scheduler

if TYPE_CHECKING:
    from output import SchedulerOutput


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L13-16
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens

    def _update_after_schedule(self, scheduler_output: "SchedulerOutput") -> None:
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L18-35
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
