# SOURCE: vllm/v1/engine/core.py
# EngineCore 的本章切面——装配（batch_queue/step_fn 静态绑定 L206-L234）、
# 同步 step()（L584-L614，对照用）、post_step（L616-L623）、本章绝对主角
# step_with_batch_queue（L625-L739 两态循环）、_process_aborts_queue、
# has_work（L1365-L1371）。
# SUBTRACTED（装配面）：忙循环/ZMQ/IO 线程/启动握手（ch9 全文已立）、
#   KV cache 剖析初始化（ch13/ch17）、DP/EC/connector/mm 装配——见各就地注释。
from __future__ import annotations

import queue
from collections import deque
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, cast

from .engine import EngineCoreOutput, EngineCoreOutputs
from .logger import init_logger
from .outputs import ModelRunnerOutput
from .request import RequestStatus

if TYPE_CHECKING:
    from .output import SchedulerOutput

logger = init_logger(__name__)


# SOURCE: vllm/v1/structured_output/__init__.py StructuredOutputManager —
# ENGINE SEAM（ch30 边界）：真实管理器持有 grammar 后端与异步编译流水线
# （init 依赖 tokenizer/model_config）；本章只消费 grammar_bitmask(requests,
# request_ids, spec_tokens) 一个面（scheduler.py:L1663-L1667 调用位）。HOST 侧
# 位掩码恒为全 1（= 无约束：允许全部 token），数值上等价于『掩码还没算好』之外
# 的一切情形——掩码怎么算归 ch30。
class StructuredOutputManager:
    # SOURCE: vllm/v1/structured_output/__init__.py StructuredOutputManager.__init__ — ENGINE SEAM
    def __init__(self, vllm_config: Any):
        # SOURCE: vllm/v1/structured_output/__init__.py StructuredOutputManager.__init__ — ENGINE SEAM
        self.vocab_size = 16
        self.grammar_bitmask_calls: list[tuple] = []  # ENGINE SEAM observation

    # SOURCE: vllm/v1/structured_output/__init__.py grammar_bitmask — ENGINE SEAM
    # （ch30 边界：全 1 位掩码站真实编译产物）
    def grammar_bitmask(self, requests, request_ids, spec_decode_tokens):
        # SOURCE: vllm/v1/structured_output/__init__.py grammar_bitmask — ENGINE SEAM
        import numpy as np

        self.grammar_bitmask_calls.append((tuple(request_ids),))
        num_cols = (self.vocab_size + 31) // 32
        return np.ones((len(request_ids), num_cols), dtype=np.int32)


# SOURCE: vllm/v1/engine/core.py:L65 EngineCore —— 本章切面
class EngineCore:
    # SOURCE: vllm/v1/engine/core.py:L74-L100 __init__ 签名（保留装配路径）
    def __init__(
        self,
        vllm_config,
        executor_class=None,
        log_stats: bool = False,
        model_executor: Any | None = None,
        num_gpu_blocks: int = 1 << 20,
    ) -> None:
        # SUBTRACTED: usage_context/deadlock detection/统计/weight version 等
        #   装配（L74-L130）——观测与启动面（ch9）。
        # SOURCE: vllm/v1/engine/core.py:L132 Setup Model.
        if model_executor is not None:
            # ENGINE SEAM（测试位）：外部给定 executor（脚本化 forward/失败注入）。
            self.model_executor = model_executor
        else:
            # SOURCE: vllm/v1/engine/core.py:L131-L132 executor 工厂
            if executor_class is None:
                from .executor_factory import get_executor_class

                executor_class = get_executor_class(vllm_config)
            self.model_executor = executor_class(vllm_config)

        # SUBTRACTED: _initialize_kv_caches 的显存剖析与 KVCacheConfig 推导
        #   （L143/L250-L410——ch13 显存域）。ENGINE SEAM：块数给定值直供。
        # SOURCE: vllm/v1/engine/core.py:L144 Setup structured output manager.
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # SOURCE: vllm/v1/engine/core.py:L146-L147 Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        # SUBTRACTED: 非 KV 模型的 chunked 关闭/块尺寸仲裁（L149-L158——ch13）。
        # SOURCE: vllm/v1/engine/core.py:L160-L168 Scheduler 装配
        self.scheduler = Scheduler(
            vllm_config=vllm_config,
            structured_output_manager=self.structured_output_manager,
            log_stats=log_stats,
            num_gpu_blocks=num_gpu_blocks,
        )
        # SOURCE: vllm/v1/engine/core.py:L169-L172 spec 探测位
        self.use_spec_decode = vllm_config.speculative_config is not None
        self.check_for_draft_tokens = (
            self.use_spec_decode or vllm_config.model_config.is_diffusion
        )
        # SUBTRACTED: scheduler.connector 的 init_kv_output_aggregator（L173-L174
        #   ——connector）、mm registry 与 kv connector 握手（L176-L201——mm/EP）、
        #   request_block_hasher 前缀缓存装配（L220-L229——ch15）。

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        # SOURCE: vllm/v1/engine/core.py:L206-L212 批队列装配（m2 物质基础）
        self.batch_queue_size = vllm_config.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        # SUBTRACTED: is_ec_consumer（L214-L217——分布式 encoder cache 非消费端，
        #   dossier.delete 第 1 条批准：常规部署恒 True，分支删除）、
        #   is_pooling_model（L218——第 2 条批准：pooling 快路删除）。

        # SOURCE: vllm/v1/engine/core.py:L231-L234 step_fn 静态绑定（m3 三级
        # 间接的最后一环——不看 async 标志，看队列建没建）+ async_scheduling
        # 落地实例字段（post_step 短路用）
        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # SOURCE: vllm/v1/engine/core.py:L236 aborts 队列
        self.aborts_queue = queue.Queue[list[str]]()

        # SOURCE: vllm/v1/engine/core.py:L1035 engines_running 初值
        self.engines_running = False

        # SUBTRACTED: 忙循环/IO 线程/启动握手/信号路径（ch9 全文已立）、
        #   _idle_state_callbacks、GC freeze/env cache（L238-L247）。

    # ------------------------------------------------------------------ #
    # 同步 step()——ch9 主角，本章只作重叠的背景板与对照（m20）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/engine/core.py:L584 step
    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """
        # SOURCE: vllm/v1/engine/core.py:L591-L594
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        # SUBTRACTED: throttle_prefills 实参（DP prefill balancing——dossier.delete
        #   第 5 条批准：_should_throttle_prefills 与 DP 覆写一并删除）。
        # SOURCE: vllm/v1/engine/core.py:L595 串行脊柱：schedule→execute→bitmask
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        # SUBTRACTED: capture_iteration_details/log_error_detail 诊断上下文
        #   （L598-L601——dossier.delete 第 3 条批准，纯观测）。
        # SOURCE: vllm/v1/engine/core.py:L602-L604 future.result + 条件采样
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        # SOURCE: vllm/v1/engine/core.py:L606-L611
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        # SUBTRACTED: _attach_iteration_details（L612——第 3 条，观测）。

        # SOURCE: vllm/v1/engine/core.py:L614
        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    # SOURCE: vllm/v1/engine/core.py:L616 post_step —— spec 分叉（m17）
    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    # ------------------------------------------------------------------ #
    # 本章绝对主角：两态循环（上半段填管道 / 下半段收最老批）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/engine/core.py:L625 step_with_batch_queue
    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
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
            # SUBTRACTED: throttle_prefills 实参（第 5 条——DP 面）。
            # SOURCE: vllm/v1/engine/core.py:L653 盲调度（此刻上一批还在 GPU 上）
            scheduler_output = self.scheduler.schedule()
            # SUBTRACTED: log_error_detail 上下文（L654——第 3 条，观测）。
            # SOURCE: vllm/v1/engine/core.py:L655-L657 发起前向（不等 GPU）
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            # SUBTRACTED: is_ec_consumer 分支（L658-L659——dossier.delete 第 1 条
            #   批准：非消费端小众场景；常规部署直接以调度数为准）。
            # SOURCE: vllm/v1/engine/core.py:L659
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

            # SUBTRACTED: is_pooling_model 条件（L661——dossier.delete 第 2 条
            #   批准：pooling 无采样快路删除；保留 not model_executed 支——
            #   没调度到 token 就无需采样）。
            # SOURCE: vllm/v1/engine/core.py:L661-L663
            if not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                # SOURCE: vllm/v1/engine/core.py:L665-L673 立即采样（bitmask 先算）
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
                    # SOURCE: vllm/v1/engine/core.py:L674-L677 deferred sampling
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                # SOURCE: vllm/v1/engine/core.py:L681 三元组 appendleft（m21）
                batch_queue.appendleft((future, scheduler_output, exec_future))
                # SOURCE: vllm/v1/engine/core.py:L682-L687 填管道优先判定
                # （v0.27.1 判定条件：len<size and (model_executed or
                # has_requests())——v0.21 是 not batch_queue[-1][0].done()，已改）
                if len(batch_queue) < self.batch_queue_size and (
                    model_executed or self.scheduler.has_requests()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, model_executed

        # SOURCE: vllm/v1/engine/core.py:L689-L693 空队列防御
        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        # SOURCE: vllm/v1/engine/core.py:L696 pop 最老批（appendleft/pop = FIFO）
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        # SUBTRACTED: capture_iteration_details/log_error_detail 上下文与
        #   _attach_iteration_details（L697-L699/L714——第 3 条，观测）。
        # SOURCE: vllm/v1/engine/core.py:L701-L706 None ⇒ execute 失败，重抛真异常
        model_output = future.result()
        if model_output is None:
            # None from sample_tokens() implies that the original execute_model()
            # call failed - raise that exception.
            exec_model_fut.result()
            raise RuntimeError("unexpected error")

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        # SOURCE: vllm/v1/engine/core.py:L708-L713
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        # SOURCE: vllm/v1/engine/core.py:L719-L737 deferred 补采（m14）
        if deferred_scheduler_output:
            # SUBTRACTED: check_for_draft_tokens 的内层块（L722-L730——
            #   dossier.delete 第 8 条批准压缩为存证注释）：真实代码在此
            #   take_draft_token_ids() → 若非 None →
            #   scheduler.update_draft_token_ids_in_output(draft_token_ids,
            #   deferred_scheduler_output) 过滤无效草稿（-1 填充被 bitmask
            #   计算跳过）；spec+structured 小众叠加归 ch33。
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        # SOURCE: vllm/v1/engine/core.py:L739
        return engine_core_outputs, model_executed

    # SOURCE: vllm/v1/engine/core.py:L741 _process_aborts_queue
    def _process_aborts_queue(self):
        # SOURCE: vllm/v1/engine/core.py:L742-L748 攒批 abort
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            # SUBTRACTED: self.abort_requests 的外部分流（L749——ch9 abort 路径），
            #   就地 finish_requests 落地。
            self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # SOURCE: vllm/v1/engine/core.py:L1365 has_work —— 在飞批保活（m20）
    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        # SOURCE: vllm/v1/engine/core.py:L1367-L1371
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )

    # SUBTRACTED: run_busy_loop/_process_input_queue/IO 线程/启动握手/
    #   shutdown 体系（L1377-L1500s——ch9 全文已立）、DP 面（L1994-L2150）、
    #   preprocess_add_request/mm 接收（Part II）、kv 初始化（ch13）。
