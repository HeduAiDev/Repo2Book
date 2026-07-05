# vllm_ascend/core/recompute_scheduler.py —— subtract-only 精简版
#
# 三个 Scheduler 子类之一，PD 分离侧。核心特化：block 不够时 kv_consumer 不做常规
# preempt，而是「丢弃请求 → 记入 recomputed_reqs → update_from_output 以
# stop_reason='recomputed' 回吐给 PD proxy 让上游改投他节点重算」。
#
# schedule() / update_from_output() 都是 vLLM Scheduler 同名方法的整段 override——
# 改动点（recompute 分叉、recomputed_reqs 回吐）夹在父方法逐字复刻的循环体内部，
# Python 无法热补片段，只能整段重写。本精简版保留改动点 + 骨架 + 签名，逐字复刻段
# 以 # SUBTRACTED 标注「同 vllm Scheduler.*」。
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. (Apache-2.0)
# Adapted from vllm-project/vllm/vllm/v1/core/sched/scheduler.py

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, fields

from vllm.config import SchedulerConfig, VllmConfig
from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.utils import ConstantList, record_function_or_nullcontext


# `spec_manager_map` in single_type_kv_cache_manager is a module-level dict
# whose keys are class objects bound at import time. When the async recompute
# scheduler is enabled, recompute_scheduler.py is imported *before* this patch
# runs a second time (e.g. unpickling an AscendMLAAttentionSpec in the
# EngineCoreProc subprocess), so a later lookup with type(instance) raises
# KeyError. Fix: register AscendMLAAttentionSpec as an additional key whenever
# this patch is applied.
# SOURCE: vllm_ascend/core/recompute_scheduler.py:L65
def register_ascend_mla_spec_in_manager():
    import sys as _sys

    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
    from vllm.v1.kv_cache_interface import MLAAttentionSpec as AscendMLAAttentionSpec

    _stm = _sys.modules.get("vllm.v1.core.single_type_kv_cache_manager")
    if _stm is not None and AscendMLAAttentionSpec not in _stm.spec_manager_map:
        _stm.spec_manager_map[AscendMLAAttentionSpec] = FullAttentionManager


@dataclass
# SOURCE: vllm_ascend/core/recompute_scheduler.py:L77
class RecomputeSchedulerConfig(SchedulerConfig):
    scheduler_cls: str | type[object] = "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"

    @classmethod
    # SOURCE: vllm_ascend/core/recompute_scheduler.py:L80
    def initialize_from_config(cls, vllm_config: VllmConfig):
        vllm_scheduler_config = vllm_config.scheduler_config
        scheduler_config = {
            field.name: getattr(vllm_scheduler_config, field.name)
            for field in fields(vllm_scheduler_config)
            if field.init
        }
        if vllm_scheduler_config.async_scheduling:
            scheduler_config["scheduler_cls"] = "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler"
        else:
            scheduler_config["scheduler_cls"] = "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"
        scheduler_config["max_model_len"] = vllm_config.model_config.max_model_len
        scheduler_config["is_encoder_decoder"] = vllm_config.model_config.is_encoder_decoder
        return cls(**scheduler_config)


@dataclass
# SOURCE: vllm_ascend/core/recompute_scheduler.py:L97
class RecomputeReqInfo:
    request_id: str
    output_token_ids: ConstantList
    client_index: int = 0


@dataclass
# SOURCE: vllm_ascend/core/recompute_scheduler.py:L104
class RecomputeSchedulerOutput(SchedulerOutput):
    recomputed_reqs: list[RecomputeReqInfo] | None = None


# SOURCE: vllm_ascend/core/recompute_scheduler.py:L109
class RecomputeScheduler(Scheduler):
    running: list[Request]

    # SOURCE: vllm_ascend/core/recompute_scheduler.py:L112
    def __init__(self, *args, **kwargs):
        register_ascend_mla_spec_in_manager()

        super().__init__(*args, **kwargs)
        # When is_mtp_kv_consumer is true, we fill request.spec_token_ids with
        # placeholder tokens to enable full graph when decode nodes pull the KV
        # cache of one request from prefill nodes.
        self.is_mtp_kv_consumer = (
            self.vllm_config.speculative_config
            and self.vllm_config.kv_transfer_config
            and self.vllm_config.kv_transfer_config.is_kv_consumer
        )
        self.is_kv_producer = self.vllm_config.kv_transfer_config and self.vllm_config.kv_transfer_config.is_kv_producer
        self.is_hybrid_model = (
            "qwen3_next" in self.vllm_config.model_config.hf_text_config.model_type
            or "qwen3_5" in self.vllm_config.model_config.hf_text_config.model_type
        )

    # SOURCE: vllm_ascend/core/recompute_scheduler.py:L130
    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            # Fill in placeholder tokens to enable full graph compatibility. Without
            # placeholders, graph matching may fail, forcing eager mode execution.
            if self.is_kv_producer and self.is_hybrid_model and request.num_tokens > 1:
                request.prompt_token_ids.pop()
                request._all_token_ids.pop()
                request.num_prompt_tokens -= 1
            if self.is_mtp_kv_consumer and (self.max_model_len >= (request.num_tokens + self.num_spec_tokens)):
                request.spec_token_ids = [PLACEHOLDER_TOKEN_ID] * self.num_spec_tokens
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    # SOURCE: vllm_ascend/core/recompute_scheduler.py:L160
    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """KV Connector: update request state after async recv is finished."""
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                self.kv_cache_manager.free(request)
            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Use Ascend-specific block_ids logic to handle multi-group KV cache
            # configurations (e.g. MLA) where len(block_ids) > 1.
            block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
            if len(block_ids) == 1:
                num_computed_tokens = len(block_ids[0]) * self.block_size
                num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            else:
                num_computed_tokens = request.num_tokens
            # on a full prompt hit, re-compute the last token to be able to
            # sample the next token.
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
            request.num_computed_tokens = num_computed_tokens

        self.finished_recving_kv_req_ids.remove(request.request_id)

    # SOURCE: vllm_ascend/core/recompute_scheduler.py:L214
    def schedule(self) -> RecomputeSchedulerOutput:
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
        recomputed_reqs: list[RecomputeReqInfo] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # SUBTRACTED: num_new_tokens 计算（async placeholder 早停、long_prefill
            #   截断、token/max_model_len 夹取、encoder 输入、mamba 对齐、num==0 跳过）
            #   逐字同 vllm Scheduler.schedule。原 recompute_scheduler.py:L255-L319
            num_new_tokens = 0  # placeholder for readability of the fork below

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # >>> ASCEND CHANGE: kv_consumer drops the request to PD proxy
                    # for recompute instead of the usual local preempt. >>>
                    transfer_config = self.vllm_config.kv_transfer_config
                    if transfer_config is not None and not transfer_config.is_kv_producer:
                        recomputed_req = self.running.pop()
                        self.kv_cache_manager.free(recomputed_req)
                        recomputed_reqs.append(
                            RecomputeReqInfo(
                                recomputed_req.request_id, recomputed_req.output_token_ids, recomputed_req.client_index
                            )
                        )
                        if recomputed_req == request:
                            break
                    # <<< ASCEND CHANGE <<<
                    else:
                        # SUBTRACTED: 常规 PRIORITY/FCFS preempt 分支（kv_producer 侧或
                        #   非 PD 场景）逐字同 vllm Scheduler.schedule。
                        #   原 recompute_scheduler.py:L350-L378
                        if self.policy == SchedulingPolicy.PRIORITY:
                            preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
                            self.running.remove(preempted_req)
                        else:
                            preempted_req = self.running.pop()
                        self._preempt_request(preempted_req, scheduled_timestamp)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
            # SUBTRACTED: spec-decode token 记账 + encoder 缓存分配，逐字同
            #   vllm Scheduler.schedule。原 recompute_scheduler.py:L392-L420

        # SUBTRACTED: WAITING 循环（remote-KV promote / get_computed_blocks /
        #   allocate_slots / mtp_kv_consumer spec token / running.append）逐字同
        #   vllm Scheduler.schedule，仅以 recomputed_reqs 守卫入口循环条件。
        #   原 recompute_scheduler.py:L422-L703

        # SUBTRACTED: 约束断言、num_common_prefix_blocks、cached_request_data 组装
        #   逐字同 vllm Scheduler.schedule。原 recompute_scheduler.py:L705-L753

        # Construct the scheduler output — carries the extra `recomputed_reqs`.
        scheduler_output = RecomputeSchedulerOutput(
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
            recomputed_reqs=recomputed_reqs,
        )

        # SUBTRACTED: connector / ec_connector meta 构建逐字同 vllm Scheduler.schedule。
        #   原 recompute_scheduler.py:L778-L789
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    # SOURCE: vllm_ascend/core/recompute_scheduler.py:L795
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        # >>> ASCEND CHANGE: return recomputed requests as EngineCoreOutput >>>
        if scheduler_output.recomputed_reqs is not None:
            for req_info in scheduler_output.recomputed_reqs:
                logger.warning("Recompute triggered for request %s.", req_info.request_id)
                outputs[req_info.client_index].append(
                    EngineCoreOutput(
                        request_id=req_info.request_id,
                        finish_reason=FinishReason.STOP,
                        new_token_ids=[],
                        stop_reason="recomputed",
                    )
                )
        # <<< ASCEND CHANGE <<<

        # SUBTRACTED: update_from_output 其余 ~280 行——perf_stats / 失效 KV block 处理 /
        #   routed_experts 持久化（L843-862,L927-955）/ 逐请求 stop 检测与 EngineCoreOutput
        #   组装（per-req 主循环）/ logprobs / structured-output / kv events / finished 收尾——
        #   逐字复刻 vllm Scheduler.update_from_output，非昇腾特化。上面 recomputed_reqs
        #   回吐是唯一新增。原 recompute_scheduler.py:L800-L828,L843-L1075
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0  # 主循环骨架占位，细节见上 SUBTRACTED

        # Create EngineCoreOutputs for all clients that have outputs this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        # SUBTRACTED: finished_req_ids 透传 + make_stats 注入逐字同 vllm。
        #   原 recompute_scheduler.py:L1065-L1083
        return engine_core_outputs


# SOURCE: vllm_ascend/core/recompute_scheduler.py:L1088
class AsyncRecomputeScheduler(AsyncScheduler, RecomputeScheduler):
    # MRO: AsyncRecomputeScheduler → AsyncScheduler → RecomputeScheduler → Scheduler.
    # schedule() 走 RecomputeScheduler.schedule；_update_after_schedule /
    # _update_request_with_output 走 AsyncScheduler。
    # SOURCE: vllm_ascend/core/recompute_scheduler.py:L1089
    def __init__(self, *args, **kwargs):
        register_ascend_mla_spec_in_manager()

        super().__init__(*args, **kwargs)
