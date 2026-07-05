# vllm_ascend/core/scheduler_profiling_chunk.py —— subtract-only 精简版
#
# 三个 Scheduler 子类之一。ProfilingChunkScheduler 在启动期 profiling 多个 chunk size 的
# prefill 延迟、拟合二次模型；调度时用预测的最优 chunk size 收窄每请求 num_new_tokens，
# 并按 predict_time 扣 time_budget。
#
# schedule() 同样是 vLLM Scheduler.schedule 的整段 override，仅 3 处以 >>> PROFILING CHUNK
# 注释标出改动；逐字复刻段以 # SUBTRACTED 标注。
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. (Apache-2.0)
import time

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.core.profiling_chunk_predictor import ProfilingChunkManager


# SOURCE: vllm_ascend/core/scheduler_profiling_chunk.py:L46
class ProfilingChunkScheduler(Scheduler):
    """Scheduler with profiling-based dynamic chunk sizing."""

    # SOURCE: vllm_ascend/core/scheduler_profiling_chunk.py:L56
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        # `hash_block_size` was added in vLLM #40946; keep it optional so the
        # subclass works on both pinned vllm and main.
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config

        init_ascend_config(vllm_config)
        profiling_cfg = get_ascend_config().profiling_chunk_config
        base_chunk = self.max_num_scheduled_tokens

        self.profiling_chunk_manager = ProfilingChunkManager(
            base_chunk_size=base_chunk,
            page_size=self.cache_config.block_size,
            smooth_factor=profiling_cfg.smooth_factor,
            min_chunk=profiling_cfg.min_chunk,
        )
        self._profiling_initialized = False

        logger.info(
            "[ProfilingChunk] Scheduler initialized. base_chunk=%d, page_size=%d, smooth_factor=%.2f, min_chunk=%d",
            base_chunk,
            self.cache_config.block_size,
            profiling_cfg.smooth_factor,
            profiling_cfg.min_chunk,
        )

    # SOURCE: vllm_ascend/core/scheduler_profiling_chunk.py:L106
    def run_profiling_chunk_init(self, model_executor) -> None:
        """Profile prefill latency using real model forward passes, then fit
        the quadratic model.  Called by EngineCore after model_executor is ready."""
        if self._profiling_initialized:
            return
        self._profiling_initialized = True

        if model_executor is None:
            logger.warning("[ProfilingChunk] No model_executor provided, skipping profiling")
            return

        seq_lens: list[int] = []
        latencies: list[float] = []

        base_chunk_size = self.profiling_chunk_manager.base_chunk_size

        # SUBTRACTED: collective_rpc('profile_prefill_latency') 采样循环（PP rank/warm-up/
        #   日志/异常细节，L122-L188）与 _build_rpc_kwargs/_extract_latency 辅助
        #   （L200-L227）。启动期一次性采样，与调度控制流解耦；保留「采样→fit→
        #   set_target_latency→is_ready」骨架即可。原 scheduler_profiling_chunk.py:L122-L227

        if len(seq_lens) < 8:
            logger.warning("[ProfilingChunk] Profiling failed: only %d samples collected", len(seq_lens))
            return

        predictor = self.profiling_chunk_manager.predictor
        if not predictor.fit(seq_lens, latencies):
            return

        predictor.set_target_latency(base_chunk_size)
        predictor.is_ready = True
        self.profiling_chunk_manager._profiling_done = True

        logger.info("[ProfilingChunk] Profiling completed successfully")

    # SOURCE: vllm_ascend/core/scheduler_profiling_chunk.py:L238
    def schedule(self) -> SchedulerOutput:  # noqa: C901
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        # >>> PROFILING CHUNK >>>
        # NOTE(gjc): FIA operator has abnormal performance when processing
        # multiple request groups in a batch, so the time_budget feature is
        # temporarily disabled (fixed to a small non-zero guard). It will be
        # enabled again after the FIA operator issues are resolved.
        # time_budget = self.profiling_chunk_manager.predictor.target_latency
        time_budget = 0.01
        # <<< PROFILING CHUNK <<<
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        # For logging.
        scheduled_timestamp = time.monotonic()  # noqa: F841

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests.
        req_index = 0
        # >>> PROFILING CHUNK: time_budget guards the loop condition >>>
        while req_index < len(self.running) and token_budget > 0 and time_budget > 0:
            # <<< PROFILING CHUNK <<<
            request = self.running[req_index]

            # SUBTRACTED: num_new_tokens 计算（async 早停 / long_prefill 截断 /
            #   max_model_len 夹取 / encoder 输入）逐字同 vllm Scheduler.schedule。
            #   原 scheduler_profiling_chunk.py:L277-L314
            num_new_tokens = 0  # placeholder for readability of the change below

            # >>> PROFILING CHUNK: dynamic chunk sizing for RUNNING >>>
            if (
                self.profiling_chunk_manager is not None
                and self.profiling_chunk_manager.is_ready
                and num_new_tokens > 1
                and request.num_computed_tokens > 0
            ):
                predicted_chunk = self.profiling_chunk_manager.predict_chunk_size(
                    num_computed_tokens=request.num_computed_tokens,
                    target_time=time_budget,
                )
                if predicted_chunk is not None and predicted_chunk > 0:
                    num_new_tokens = min(predicted_chunk, num_new_tokens)
            # <<< PROFILING CHUNK <<<

            if num_new_tokens == 0:
                req_index += 1
                continue

            # SUBTRACTED: allocate_slots + PRIORITY/FCFS preempt while-loop 逐字同
            #   vllm Scheduler.schedule。原 scheduler_profiling_chunk.py:L338-L378
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            # >>> PROFILING CHUNK: charge time_budget for prefill chunks (decode
            # requests with num_new_tokens==1 have negligible latency, skipped). >>>
            if num_new_tokens > 1:
                time_budget -= self.profiling_chunk_manager.predict_time(num_new_tokens, request.num_computed_tokens)
            # <<< PROFILING CHUNK <<<
            req_index += 1
            # SUBTRACTED: spec-decode 记账 + encoder 缓存分配逐字同 vllm。
            #   原 scheduler_profiling_chunk.py:L392-L417

        # SUBTRACTED: WAITING 循环——同样 time_budget>0 守卫入口（L434）、WAITING 侧
        #   dynamic chunk sizing（L523-536）、predict_time 记账（L632-635），其余逐字
        #   同 vllm Scheduler.schedule。原 scheduler_profiling_chunk.py:L429-L653

        # SUBTRACTED: 约束断言 / num_common_prefix_blocks / cached_request_data 组装
        #   逐字同 vllm Scheduler.schedule。原 scheduler_profiling_chunk.py:L655-L698

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=self._make_cached_request_data(
                scheduled_running_reqs, [], num_scheduled_tokens, {}, req_to_new_blocks
            ),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
            preempted_req_ids={req.request_id for req in preempted_reqs},
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        # SUBTRACTED: connector / ec_connector meta 构建逐字同 vllm。
        #   原 scheduler_profiling_chunk.py:L718-L724
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
