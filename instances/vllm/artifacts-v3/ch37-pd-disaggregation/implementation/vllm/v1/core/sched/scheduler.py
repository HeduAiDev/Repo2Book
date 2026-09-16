# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""调度器的「P/D 切面」：connector 挂钩点、块所有权转移、回执出引擎、判活。

# SOURCE: vllm/v1/core/sched/scheduler.py:L200-L2440（逐挂钩点行号见各方法上方）
# SUBTRACTED: 调度主循环的批装填/抢占/chunked prefill/前缀缓存哈希/结构化输出/
#   spec-decode/encoder-decoder/LoRA/流式/暂停状态机（L600-L1900 大部）——
#   它们是 ch09-ch16 的面；本章需要的是请求生命周期上那四处 connector 挂钩，
#   以及「回执怎么随 EngineCoreOutput 出引擎」。
"""

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


# SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200
# SEAM: 真源码里 Scheduler 持一个 KVCacheManager，负责块分配/前缀缓存哈希/回收，
#   `request_finished` 拿到的块表由它产出（scheduler.py:L2300-L2327）。
#   本章的块是测试注入的（P 的块号来自 kv_transfer_params 的搬运，不是本地分配），
#   故这里只留「按请求登记块表 / 归还」两个面，语义与真块池一致。
# SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200（块池 SEAM）
class _StubBlockPool:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200（块池 SEAM）
    """按请求登记块表；`get_block_ids` 返回该请求各组的块号。"""

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200（块池 SEAM）
    def __init__(self) -> None:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200（块池 SEAM）
        self._blocks: dict[str, tuple[list[int], ...]] = {}
        self.freed: set[str] = set()

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200（块池 SEAM）
    def set_blocks(self, request_id: str, groups: tuple[list[int], ...]) -> None:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200（块池 SEAM）
        self._blocks[request_id] = groups

    # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200（块池 SEAM）
    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        # SOURCE: vllm/v1/core/kv_cache_manager.py:L117-L200（块池 SEAM）
        return self._blocks.get(request_id, ())


# SOURCE: vllm/v1/core/sched/scheduler.py:L200-L340
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py:L200-L340
    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: Any,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.log_stats = log_stats

        self.requests: dict[str, Request] = {}
        self.finished_req_ids: set[str] = set()
        self.waiting: list[Request] = []

        # SOURCE: vllm/v1/core/sched/scheduler.py:L200-L340
        self.connector: KVConnectorBase_V1 | None = None
        if vllm_config.kv_transfer_config is not None:
            from vllm.distributed.kv_transfer.kv_connector.factory import (
                KVConnectorFactory,
            )
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                KVConnectorRole,
            )

            self.connector = KVConnectorFactory.create_connector(
                config=vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=kv_cache_config,
            )

        # SOURCE: vllm/v1/core/sched/scheduler.py:L200-L340
        # SUBTRACTED: ECConnector（编码器缓存传输）——本章不涉多模态。
        self.ec_connector = None

        # 异步加载完成、等待提升回 WAITING 的请求（ch16 站 9 的账本）。
        self.finished_recving_kv_req_ids: set[str] = set()

        # 块池（见 _StubBlockPool 的 SEAM 说明）与「已放块」账。
        self.kv_cache_manager = _StubBlockPool()
        self.kv_blocks_freed: set[str] = set()

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2568-L2570
    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2571-L2573
    def get_ec_connector(self):
        return self.ec_connector

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2213-L2235
    def add_request(self, request: Request) -> None:
        self.requests[request.request_id] = request
        self.waiting.append(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2237-L2280
    def finish_requests(
        self, request_ids: str | list[str], finished_status: RequestStatus
    ) -> None:
        for req_id in [request_ids] if isinstance(request_ids, str) else request_ids:
            request = self.requests.get(req_id)
            if request is None:
                continue
            request.status = finished_status
            self._free_request(request, delay_free_blocks=False)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2300-L2327
    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        # EC Connector: mirror the KV hook. The contract requires firing
        # before the encoder cache is freed so the connector can inspect
        # per-request state (e.g. which mm_hashes it recorded during
        # save_caches()) and emit ec_transfer_params for the response body.
        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        request_id = request.request_id
        self.finished_req_ids.add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2329-L2332
    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2341-L2354
    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks（defer 栅栏归 ch16，本章直还块池）。"""
        # SUBTRACTED: defer_block_free / sched_step_seq 在途写栅栏——异步调度
        #   （ch12）与 connector 的连乘场景才需要；本章的块池是测试替身。
        self.kv_blocks_freed.add(request.request_id)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2577-L2600
    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """调用 connector 的 request_finished 钩子（P/D 的交接点）。"""
        if self.connector is None:
            return False, None
        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished_all_groups(request, block_ids)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2714-L2741
    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
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

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2406-L2420
    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or (
                self.ec_connector is not None
                and self.ec_connector.has_pending_push_work()
            )
        )

    # SOURCE: vllm/v1/core/sched/interface.py:L173-L180（接口默认实现）
    def has_unfinished_requests(self) -> bool:
        return any(not r.is_finished() for r in self.requests.values())

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2394-L2404
    def has_finished_requests(self) -> bool:
        return bool(self.finished_req_ids)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L439-L600
    def schedule(self) -> SchedulerOutput:
        """本章不装填批：产出空批以驱动 connector 的 build_connector_meta。"""
        return SchedulerOutput()

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1670-L1690
    def update_from_output(self, kv_connector_output: KVConnectorOutput | None) -> None:
        if kv_connector_output is not None:
            self._update_from_kv_xfer_finished(kv_connector_output)


__all__ = ["Scheduler"]
