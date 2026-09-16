# SOURCE: vllm/v1/core/sched/async_scheduler.py
# v3 ch31 脊柱⑤：AsyncScheduler._update_after_schedule（L19-L49——
# pending_structured_output_tokens 置位 + 占位记账 + spec_token_ids 的 -1
# 占位数组，延后采样的信号源 m16）。V2+PP 的 next_decode_eligible_step
# 微批步距删（L46-L49，dossier 摘录 elide 注明归 PP 微批调度）。
# _update_request_with_output（L51-L70）删——占位回扣/块缓存归 ch12 m15。
from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)
# SUBTRACTED: from vllm.v1.request import Request, RequestStatus（L4-L7——
#   Request 本体归 ch02，requests 值以测试替身承载；注解面字符串引用）。


# SOURCE: vllm/v1/core/sched/async_scheduler.py:L12 AsyncScheduler —— 本章切面
class AsyncScheduler(Scheduler):
    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L13-L17 __init__ —— 逐字
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    #   —— 逐字（仅删 L46-L49 use_v2 PP 微批步距）
    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L19-L49 _update_after_schedule
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            # SUBTRACTED: vllm/v1/core/sched/async_scheduler.py:L46-L49
            #   use_v2_model_runner 的 next_decode_eligible_step 步距——
            #   V2+PP 微批调度面（dossier 摘录 elide 注明），归 ch12 展望。

    # SUBTRACTED: vllm/v1/core/sched/async_scheduler.py:L51-L70
    #   _update_request_with_output 覆写（占位回扣 + 块缓存推进——ch12 m15
    #   全文已立；本章 update_from_output 切面不含其调用链）。
