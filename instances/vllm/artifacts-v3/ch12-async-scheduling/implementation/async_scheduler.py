# SOURCE: vllm/v1/core/sched/async_scheduler.py
# AsyncScheduler —— 盲调度的账本引擎（全文件 70 行，本章两个主角方法的宿主）。
# 只覆写两个方法：_update_after_schedule（占位 +1）与 _update_request_with_
# output（占位 -1 + 块转正）。删除项仅 V2 runner 分支（dossier.delete 第 4 条）。
from __future__ import annotations

from .logger import init_logger
from .output import SchedulerOutput
from .request import Request, RequestStatus
from .scheduler import Scheduler

logger = init_logger(__name__)


# SOURCE: vllm/v1/core/sched/async_scheduler.py:L12 AsyncScheduler
class AsyncScheduler(Scheduler):
    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L13-L17 __init__
    def __init__(self, *args, **kwargs) -> None:
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L14
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L16
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        # SUBTRACTED: self.pp_size = parallel_config.pipeline_parallel_size
        #   （L17——仅 V2+PP 的 next_decode_eligible_step 步距用，随第 4 条删）。

    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L19 _update_after_schedule
    # （占位 +1——m6）
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L20 先跑基类乐观推进
        super()._update_after_schedule(scheduler_output)
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L21-L25 spec 占位列表
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L26-L44 逐请求占位
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            # SOURCE: vllm/v1/core/sched/async_scheduler.py:L31-L33 pending
            # 置位（deferred sampling 的开关）
            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) — only the canvas
            # (spec) tokens.
            # SOURCE: vllm/v1/core/sched/async_scheduler.py:L38-L41 占位 +1
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            # SOURCE: vllm/v1/core/sched/async_scheduler.py:L44 spec_token_ids
            # 换 -1 占位列表（真 token 由 worker 原地替换）
            request.spec_token_ids = self._spec_token_placeholders

            # SUBTRACTED: use_v2_model_runner 的 next_decode_eligible_step
            #   （L46-L49——V2+PP 步距，dossier.delete 第 4 条批准）。

    # SOURCE: vllm/v1/core/sched/async_scheduler.py:L51 _update_request_with_
    # output（占位 -1 + 块转正——m7）
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int], is_stale: bool = False
    ) -> tuple[list[int], bool]:
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L54 先跑基类逐 token 收账
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Placeholders were zeroed at preemption; a stale delivery must not
        # decrement them (it would underflow).
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L59-L63 占位扣减
        # （stale 不扣——#48245 类 underflow 的防线）
        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L65-L69 块转正（参数
        # computed − placeholders = 真实已算——不变式的化身）
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        # SOURCE: vllm/v1/core/sched/async_scheduler.py:L70
        return new_token_ids, stopped
