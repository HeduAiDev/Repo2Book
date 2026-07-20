# SOURCE: vllm/v1/engine/core.py
# 只做减法的忠实精简版（pin ad7125a4 / v0.21.0）。EngineCore 是真实引擎的主循环
# 驱动者，本章只保留两条与掩码装配时机直接相关的方法：
#   - step()：m07，execute_model 非阻塞发车之后才算掩码，让 CPU 填掩码与 GPU 前向
#     重叠。
#   - step_with_batch_queue()：m15/9b，异步调度 + 投机解码 + 结构化输出三者交汇的
#     延后采样链——pending_structured_output_tokens 为真时，本步的掩码必须等到
#     "上一步的模型输出已经落定、草稿 token 已经 D2H 回传" 之后才能算。骨架保留
#     dossier 钉死的六步：pending 判定 → 延后 scheduler_output → 下一次调用里
#     take_draft_token_ids → update_draft_token_ids_in_output → get_grammar_bitmask
#     → sample_tokens。批队列的"已满才阻塞"早退分支同样保留——它不是无关的吞吐量
#     旁支，而是让 deferred_scheduler_output 真的跨调用延迟一轮的机制本身：没有
#     它，本轮刚 append 的 future 会在同一次调用里被立刻 pop 掉，队列永远深度为
#     0，"延后到下一轮"就无从谈起。
#
# SUBTRACTED: SPDX 版权头、is_ec_consumer / is_pooling_model 分支（EC connector 与
# pooling 模型不产出需要掩码的 logits，与本章无关）、log_error_detail /
# log_iteration_details 包裹的可观测性外壳、_process_aborts_queue（请求中止处理，
# 与本章无关）、post_step/update_draft_token_ids（同步调度路径下的草稿回传，走的是
# scheduler.update_draft_token_ids 而非本章聚焦的 _in_output 变体，非本章 must_keep）。
import collections
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from output import GrammarOutput, SchedulerOutput


class EngineCore:
    def __init__(
        self,
        scheduler,
        model_executor,
        async_scheduling: bool = False,
        use_spec_decode: bool = False,
        batch_queue_size: int = 2,
    ) -> None:
        # SOURCE: vllm/v1/engine/core.py EngineCore.__init__（精简到本章相关的
        # 五个属性：scheduler/model_executor/async_scheduling/use_spec_decode/
        # batch_queue 本身就是真实 __init__ 里挂的同名属性）
        #
        # SUBTRACTED: 真实 __init__ 从 vllm_config 构造整个执行器/连接器/日志
        # 对象图（模型加载、KV cache 初始化、EC/KV connector、profiler 等，属
        # ch01/ch13 范围）——本章用依赖注入直接接受已构造好的 scheduler/
        # model_executor，只保留与掩码装配时机相关的控制流。
        self.scheduler = scheduler
        self.model_executor = model_executor
        self.async_scheduling = async_scheduling
        self.use_spec_decode = use_spec_decode
        self.batch_queue_size = batch_queue_size
        self.batch_queue: "collections.deque | None" = (
            collections.deque() if batch_queue_size > 0 else None
        )

    def step(self):
        # SOURCE: vllm/v1/engine/core.py:L406-433（精简：删 log_error_detail/
        # log_iteration_details 包裹、_process_aborts_queue）
        """Schedule, execute, and make output."""
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def step_with_batch_queue(self):
        # SOURCE: vllm/v1/engine/core.py:L447-561（精简：删 EC consumer/pooling
        # 分支、log_error_detail/log_iteration_details 包裹）
        """Schedule and execute batches with the batch queue.

        1. Try to schedule a new batch if the batch queue is not full. If a
           new batch is scheduled, and we don't need to block on it yet,
           return immediately -- fulfilling the batch queue has priority over
           getting model outputs.
        2. If pending_structured_output_tokens is set, defer sampling until
           the prior step's output (and draft tokens) have landed.
        3. Block on the oldest queued future, update the scheduler from its
           output, then (if this round deferred) compute the bitmask now that
           we have the draft tokens and submit the deferred sample_tokens.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if not model_executed:
                # No sampling required (no requests scheduled).
                future = exec_future
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
                    # We need to defer sampling until we have processed the model
                    # output from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is
                    # full or there are no more requests to schedule. This is
                    # exactly what lets the deferred branch below see a
                    # *different* (older) scheduler_output than the one that
                    # just set pending_structured_output_tokens.
                    return None, True
        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        model_output = future.result()
        if model_output is None:
            # None from sample_tokens() implies that the original execute_model()
            # call failed - raise that exception.
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
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
