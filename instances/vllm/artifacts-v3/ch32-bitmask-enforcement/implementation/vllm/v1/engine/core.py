# SOURCE: vllm/v1/engine/core.py
# v3 ch31 脊柱⑥：EngineCore 的两段式窗口编排面——step()（L584-L614，四段排布）
# 与 step_with_batch_queue（L625-L739，v0.27 服务默认心跳：延后分流 + deferred
# 兑现链 L719-L737）+ post_step（L616-L623）+ _should_throttle_prefills
# （L579-L582）。
# SUBTRACTED（delete[2] 掩码无关分支）：EC consumer 记账（is_ec_consumer）、
#   pooling 快路、批队列长度调度（队未满即早退 L682-L687）、可观测性外壳
#   （capture_iteration_details/log_error_detail with 块/_attach_iteration_details）、
#   abort 队列处理（_process_aborts_queue 及其调用位）；忙循环/ZMQ/IO 线程/
#   启动握手/DGCO/DP 面归 ch9/ch12 全文已立。
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from typing import Any, cast

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/core.py:L65 EngineCore —— 本章切面
class EngineCore:
    #   裁剪；真实装配 L74-L235 的剖析/握手/统计面删，ch9/ch12 已立）
    # SOURCE: vllm/v1/engine/core.py:L74-L100 __init__ 签名 —— 保留（消费面
    def __init__(
        self,
        vllm_config,
        structured_output_manager,
        scheduler,
        model_executor,
    ) -> None:
        # SUBTRACTED: usage_context/deadlock detection/统计/executor 工厂装配
        #   （L74-L143——ch9 启动域）。ENGINE SEAM：scheduler / model_executor /
        #   structured_output_manager 由外部注入（测试替身或真实件直挂）。
        self.structured_output_manager = structured_output_manager
        self.scheduler = scheduler
        self.model_executor = model_executor
        # SOURCE: vllm/v1/engine/core.py:L169-L172 spec 探测位
        self.use_spec_decode = vllm_config.speculative_config is not None
        self.check_for_draft_tokens = (
            self.use_spec_decode or vllm_config.model_config.is_diffusion
        )

        # SOURCE: vllm/v1/engine/core.py:L204-L212 批队列装配（deferred 链的
        #   物质基础——m16 挂起/兑现/重新入队都发生在它上面）
        self.batch_queue_size = vllm_config.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        # SUBTRACTED: is_ec_consumer（L214-L217——delete[2] EC consumer 记账；
        #   常规部署恒 True，等价于无条件记 model_executed）、is_pooling_model
        #   （L218——pooling 快路，delete[2]；本章固定生成模型）、
        #   request_block_hasher 前缀缓存装配（L220-L229——ch15）、step_fn 静态
        #   绑定（L231-L234——ch12 m3 三级间接，本章两心跳函数直呼）。

        # SOURCE: vllm/v1/engine/core.py:L234
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

    # SOURCE: vllm/v1/engine/core.py:L579-L582 _should_throttle_prefills —— 逐字
    def _should_throttle_prefills(self) -> bool:
        """Whether to defer new prefills this step (DP prefill balancing).
        Overridden by the DP engine core; never throttles otherwise."""
        return False

    #   （仅删可观测性 with 块与 abort 处理两调用位，delete[2]）
    # SOURCE: vllm/v1/engine/core.py:L584-L614 step —— 逐字
    def step(self) -> tuple[dict[int, "EngineCoreOutputs"], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        # SUBTRACTED: L598-L601 capture_iteration_details/log_error_detail
        #   可观测性 with 块（delete[2]）。
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)

        # SUBTRACTED: L606-L608 _process_aborts_queue（delete[2] abort 队列）。
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        # SUBTRACTED: L612 _attach_iteration_details（可观测性外壳，delete[2]）。

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    # SOURCE: vllm/v1/engine/core.py:L616-L623 post_step —— 逐字
    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    #   （保留 1/2/3 段骨架 + deferred 兑现链逐字；删 delete[2] 的四类分支）
    # SOURCE: vllm/v1/engine/core.py:L625-L739 step_with_batch_queue —— 切面
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, "EngineCoreOutputs"] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """

        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            # SUBTRACTED: L654 log_error_detail with 块（delete[2] 可观测性）。
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            # SUBTRACTED: L658-L659 is_ec_consumer 条件（delete[2] EC consumer
            #   记账——常规部署恒 True，等价于无条件执行）。
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

            # SUBTRACTED: L661 is_pooling_model 快路（delete[2]）；保留
            #   not model_executed 空拍分支（无请求被排 → 无需采样）。
            if not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

                if not deferred_scheduler_output:
                    # Add this step's future to the queue.
                    batch_queue.appendleft((future, scheduler_output, exec_future))
                    # SUBTRACTED: L682-L687 队未满即早退的批队列长度调度
                    #   （delete[2]——吞吐优化；删后每次调用都走到 pop 收输出，
                    #   deferred 链时序不受影响）。

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        # SUBTRACTED: L697-L700 可观测性 with 块（delete[2]）。
        model_output = future.result()
        if model_output is None:
            # None from sample_tokens() implies that the original execute_model()
            # call failed - raise that exception.
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

        # SUBTRACTED: L710 _process_aborts_queue（delete[2] abort 队列）。
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        # SUBTRACTED: L714 _attach_iteration_details（可观测性外壳，delete[2]）。

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # When draft tokens are used with structured output, validate them
            # before computing the grammar bitmask for the deferred request.
            if self.check_for_draft_tokens:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                if draft_token_ids is not None:
                    # Update the draft token ids in the scheduler output to
                    # filter out the invalid spec tokens, which will be padded
                    # with -1 and skipped by the grammar bitmask computation.
                    self.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        return engine_core_outputs, model_executed

    # SUBTRACTED: vllm/v1/engine/core.py:L741-L749 _process_aborts_queue 及
    #   abort_requests 面（delete[2] abort 队列处理——ch9 生命周期域）。
