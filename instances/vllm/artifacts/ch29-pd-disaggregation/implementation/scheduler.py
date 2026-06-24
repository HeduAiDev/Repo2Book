# SPDX-License-Identifier: Apache-2.0
# Subtract-only companion for ch29《PD 分离的抽象与调度器集成》.
# 只做减法：与 vLLM 同名/同结构/同控制流，只删不增。
#
# 本文件是 vllm/v1/core/sched/scheduler.py 的子集：只保留 KV-connector 集成所需
# 的部分 —— 构造 SCHEDULER-role connector、waiting/skipped_waiting 双队列、
# WAITING 调度循环中查远程命中→隔离避队头阻塞→KV 到位提升回 WAITING/PREEMPTED
# 的完整路径（f12 回收闭环）。真实 Scheduler 还做 RUNNING 调度、抢占、encoder、
# spec-decode、stats、mamba 等，与本章主线无关者全部 SUBTRACTED。
from typing import TYPE_CHECKING, Any

from .base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole
from .factory import KVConnectorFactory
from .request import Request, RequestStatus
from .request_queue import SchedulingPolicy, create_request_queue

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput

import logging

logger = logging.getLogger(__name__)


class Scheduler:
    """vLLM v1 Scheduler 的 KV-connector 集成子集。"""

    # SOURCE: vllm/v1/core/sched/scheduler.py (Scheduler.__init__ — 仅 KV-connector 部分)
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_manager: Any,
        kv_cache_config: Any = None,
        policy: SchedulingPolicy = SchedulingPolicy.FCFS,
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_manager = kv_cache_manager
        self.kv_cache_config = kv_cache_config
        self.policy = policy

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L118
        self.connector: KVConnectorBase_V1 | None = None
        if self.vllm_config.kv_transfer_config is not None:
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
        # SUBTRACTED: connector_prefix_cache_stats / recompute_kv_load_failures /
        # is_encoder_decoder 断言 / ec_connector（编码器缓存 connector，与 KV 主线
        # 平行的另一套）均与 role-split 调度集成无关（原 L122-148）。

        # req_id -> Request
        self.requests: dict[str, Request] = {}

        # Priority queues for requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L167
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # KV Connector: requests in process of async KV loading or recving
        # SOURCE: vllm/v1/core/sched/scheduler.py:L183
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # SUBTRACTED: max_num_running_reqs / token_budget / 各类调度上限、
        # finished_req_ids、encoder budget 等正常调度配额（原 L150-200+）。

    # SOURCE: vllm/v1/core/sched/scheduler.py (schedule — 仅 WAITING 循环)
    def schedule(self) -> "SchedulerOutput":
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []

        # Next, schedule the WAITING requests.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L568
        step_skipped_waiting = create_request_queue(self.policy)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L571
        while self.waiting or self.skipped_waiting:
            request_queue = self._select_waiting_queue_for_scheduling()
            assert request_queue is not None

            request = request_queue.peek_request()
            request_id = request.request_id

            # try to promote blocked statuses while traversing skipped queue.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L581
            if self._is_blocked_waiting_status(
                request.status
            ) and not self._try_promote_blocked_waiting_request(request):
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.", request_id
                    )
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue
            # SUBTRACTED: lora max_loras 约束分支（原 L594-607），与 KV 主线正交。

            num_external_computed_tokens = 0
            load_kv_async = False

            # Get already-cached tokens.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L614
            if request.num_computed_tokens == 0:
                # Get locally-cached tokens.
                new_computed_blocks, num_new_local_computed_tokens = (
                    self.kv_cache_manager.get_computed_blocks(request)
                )

                # Get externally-cached tokens if using a KVConnector.
                if self.connector is not None:
                    ext_tokens, load_kv_async = (
                        self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens
                        )
                    )

                    if ext_tokens is None:
                        # The request cannot be scheduled because
                        # the KVConnector couldn't determine
                        # the number of matched tokens.
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

                    num_external_computed_tokens = ext_tokens

                # Total computed tokens (local + external).
                num_computed_tokens = (
                    num_new_local_computed_tokens + num_external_computed_tokens
                )
                assert num_computed_tokens <= request.num_tokens
                # SUBTRACTED: prefill_stats.set 记录块、connector_prefix_cache_*
                # 统计（原 L638-656），纯指标。
            else:
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L657
                new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

            # SOURCE: vllm/v1/core/sched/scheduler.py:L668
            if load_kv_async:
                # KVTransfer: loading remote KV, do not allocate for new work.
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
            else:
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                assert num_new_tokens > 0
                # SUBTRACTED: long_prefill_token_threshold 截断、chunked prefill
                # token_budget 约束、encoder inputs 调度、mamba_block_aligned_split
                # 都是正常 prefill 计算路径的边角（原 L672-721）。

            # Handles an edge case when P/D Disaggregation
            # is used with Spec Decoding where an extra block gets allocated.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L728
            effective_lookahead_tokens = (
                0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
            )

            # SOURCE: vllm/v1/core/sched/scheduler.py:L744
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_local_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=effective_lookahead_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
                delay_cache_blocks=load_kv_async,
            )

            if new_blocks is None:
                # The request cannot be scheduled.
                break

            # KVTransfer: the connector uses this info to determine
            # if a load is needed.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L769
            if self.connector is not None:
                self.connector.update_state_after_alloc(
                    request,
                    self.kv_cache_manager.get_blocks(request_id),
                    num_external_computed_tokens,
                )
                # SUBTRACTED: connector_prefix_cache_stats.record（原 L775-783），指标。

            # SOURCE: vllm/v1/core/sched/scheduler.py:L785
            request = request_queue.pop_request()
            if load_kv_async:
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                step_skipped_waiting.prepend_request(request)
                # Set num_computed_tokens even though KVs are not yet loaded.
                # request.num_computed_tokens will not be used anywhere until
                # the request finished the KV transfer.
                # _update_waiting_for_remote_kv will then cache
                # only the successfully loaded tokens.
                request.num_computed_tokens = num_computed_tokens
                continue

            self.running.append(request)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            # SUBTRACTED: record_event(SCHEDULED)、lora 记账、req_to_new_blocks/
            # num_scheduled_tokens/token_budget 记账、encoder cache allocate
            # 都是正常调度记账，非 KV 主线（原 L808-842）。

        # re-queue requests skipped in this pass ahead of older skipped items.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L844
        if step_skipped_waiting:
            self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # SUBTRACTED: 构造完整 SchedulerOutput（scheduled_*_reqs/num_scheduled_tokens/
        # finished_req_ids 等）的大段装配（原 L860-926）；精简版用一个轻量壳承载本步
        # scheduled_new_reqs 即可演示 build_connector_meta。
        scheduler_output = _SchedulerOutput(scheduled_new_reqs)

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        # SOURCE: vllm/v1/core/sched/scheduler.py:L928
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        return scheduler_output

    # SOURCE: vllm/v1/core/sched/scheduler.py:L947
    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    @staticmethod
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1553
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1567
    def _select_waiting_queue_for_scheduling(self):
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1996
    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_computed_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        # SUBTRACTED: SupportsHMA 分支（request_finished_all_groups，多 kv_cache_group）
        # 与 PD 分离正交；精简版只走单 group 的 request_finished（原 L2017-2025）。
        return self.connector.request_finished(request, block_ids[0])

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2027
    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2061
    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        # SUBTRACTED: WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR / WAITING_FOR_STREAMING_REQ
        # 两个非本章阻塞态的提升分支（原 L2078-2092）。
        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status} for request {request.request_id}"
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2094
    def _update_from_kv_xfer_finished(self, kv_connector_output: "KVConnectorOutput"):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    # 真实 _free_blocks 还做 connector 块归还/encoder cache 清理；本子集委托给
    # kv_cache_manager.free 以保持可运行。
    # SOURCE: vllm/v1/core/sched/scheduler.py (_free_blocks — 通用块释放工具)
    def _free_blocks(self, request: Request) -> None:
        self.kv_cache_manager.free(request)

    # num_lookahead_tokens 在真实 Scheduler 由 spec-decode 配置推出；本子集默认 0。
    num_lookahead_tokens = 0


# SOURCE: vllm/v1/core/sched/output.py (SchedulerOutput — 仅 build_connector_meta 读到的字段)
# 一个承载 scheduled_new_reqs / kv_connector_metadata 的轻量壳，对应真实
# SchedulerOutput 的相关字段。
# SUBTRACTED: 真实 SchedulerOutput 有 scheduled_cached_reqs/num_scheduled_tokens/
# finished_req_ids 等数十字段，本子集只保留 build_connector_meta 触碰的两个。
class _SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py (SchedulerOutput.__init__ 相关字段)
    def __init__(self, scheduled_new_reqs: list[Request]) -> None:
        # build_connector_meta 读 scheduled_new_reqs[*].req_id / prompt_token_ids /
        # block_ids；这里把 Request 适配成它期望的形状。
        self.scheduled_new_reqs = [
            _NewReqData(r.request_id, r.prompt_token_ids) for r in scheduled_new_reqs
        ]
        self.kv_connector_metadata: KVConnectorMetadata | None = None


# SOURCE: vllm/v1/core/sched/output.py (NewRequestData — 仅 connector 读到的字段)
class _NewReqData:
    # SOURCE: vllm/v1/core/sched/output.py (NewRequestData.__init__ 相关字段)
    def __init__(self, req_id: str, prompt_token_ids: list[int]) -> None:
        self.req_id = req_id
        self.prompt_token_ids = prompt_token_ids
        # block_ids 在真实链路由 allocate_slots 填充；测试里由桩 kv_cache_manager 提供。
        self.block_ids: list[list[int]] = [[0]]
