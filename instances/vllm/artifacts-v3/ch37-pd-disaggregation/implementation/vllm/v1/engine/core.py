# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore：本章只保留两处 P/D 装配点。

站 2（待命）：从全部 worker 收集握手元数据 → 注入调度器侧 connector → 起 side channel；
拒绝回执：`abort_immediately` 的占位请求 add 完立即 abort，只为触发 request_finished 钩子。

# SOURCE: vllm/v1/engine/core.py:L103-L1000（EngineCore 收窄载体）
# SUBTRACTED: 引擎主循环（步进/输出分发/输入队列 IO 双线程/暂停恢复/显存分析）、
#   PP 微批、结构化输出管理、多模态接收器——它们是 ch05/ch09/ch17 的面。
"""

from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/core.py:L103-L200
class EngineCore:
    # SOURCE: vllm/v1/engine/core.py:L103-L200
    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: Any,
        model_executor: Any = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.model_executor = model_executor

        # SOURCE: vllm/v1/engine/core.py:L103-L200
        self.scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
        )

        # SOURCE: vllm/v1/engine/core.py:L172-L174（聚合器装配）
        if self.scheduler.connector is not None:
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)

        # SOURCE: vllm/v1/engine/core.py:L181-L200（握手元数据聚合注入）
        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {(pp_rank, tp_rank): metadata}
                # Merge all worker dicts into a single dict
                content: dict[tuple[int, int], Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata_pp_aware(content)

    # SOURCE: vllm/v1/engine/core.py:L969-L990（preprocess_add_request）
    # SUBTRACTED: mm_receiver_cache 与 structured_output_manager.grammar_init——
    #   多模态与结构化输出（ch31/ch32）的面；本章的请求只有 token id 与回执信封。
    # SOURCE: vllm/v1/engine/core.py:L969-L990
    def preprocess_add_request(self, request: EngineCoreRequest) -> Request:
        # SOURCE: vllm/v1/engine/core.py:L969-L990
        return Request.from_engine_core_request(request, None)

    # SOURCE: vllm/v1/engine/core.py:L438-L483
    def add_request(self, request: Request, request_wave: int = 0) -> None:
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        # SUBTRACTED: 池化任务白名单校验（L445-L457）与 EC 传输参数告警
        #   （L468-L475）——本章的 P/D 请求都是生成式、不带 encoder cache。
        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        self.scheduler.add_request(request)
        # SOURCE: vllm/v1/engine/core.py:L438-L483
        if request.abort_immediately:
            # Immediately abort so the connector's request_finished hook runs
            # to free any pre-admission KV-transfer resources.
            self.abort_requests([request.request_id])

    # SOURCE: vllm/v1/engine/core.py:L485-L495
    def abort_requests(self, request_ids: list[str]) -> None:
        """Abort requests from the scheduler."""
        from vllm.v1.request import RequestStatus

        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    # SOURCE: vllm/v1/engine/core.py:L1404-L1433 + L1514（ADD 消息入口）
    #   把 EngineCoreRequest 送进 core；core 侧在输入队列里 preprocess 后 add）
    # SOURCE: vllm/v1/engine/core.py:L1404-L1433 + L1514（ADD 入口）
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # SOURCE: vllm/v1/engine/core.py:L1404-L1433 + L1514（ADD 入口）
        self.add_request(self.preprocess_add_request(request))


__all__ = ["EngineCore"]
