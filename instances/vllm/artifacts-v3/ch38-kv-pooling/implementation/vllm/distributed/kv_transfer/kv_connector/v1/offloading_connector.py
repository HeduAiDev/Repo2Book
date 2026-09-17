# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ch38 池化 facade：best-effort 三件 + build_offloading_config→spec→按 role 建半边。
#
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L1-L230
# SUBTRACTED（按减法计划）：
#   · 删除项 5：register_cross_layers_kv_cache 的转发方法（worker 侧实现删除）。
#   · 删除项 7：get_kv_connector_stats / build_kv_connector_stats / build_prom_metrics
#     的遥测体——契约签名保留默认 None 返回。
from collections.abc import Iterable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.forward_context import ForwardContext
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L49-L229
class OffloadingConnector(KVConnectorBase_V1, SupportsHMA):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L50-L52
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L50-L52
        return True

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L54-L58
    @property
    def requires_kv_delivery(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L54-L58
        # Runs as kv_both, but is a best-effort cache: a dropped save is just a
        # future cache miss, so opt out of the producer-role default.
        return False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L60-L80
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L66-L80
        super().__init__(vllm_config, role, kv_cache_config)

        offloading_config = build_offloading_config(vllm_config, kv_cache_config)
        spec = OffloadingSpecFactory.create_spec(offloading_config)

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(
                spec, vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(
                spec, vllm_config, kv_cache_config
            )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L82-L86
    def shutdown(self) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L82-L86
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L88-L90
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L88-L90
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    # SUBTRACTED: register_cross_layers_kv_cache 转发（L92-L96）——删除项 5
    #   （cross-layer 布局族；prefer_cross_layer_blocks 旗标仍真，通用逐层路径承接）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L98-L101
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L98-L101
        assert self.connector_worker is not None
        assert isinstance(kv_connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.handle_preemptions(kv_connector_metadata)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L103-L106
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L103-L106
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_kv_transfers(self._connector_metadata)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L108-L109
    def wait_for_layer_load(self, layer_name: str) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L108-L109
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L111-L118
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L111-L118
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L120-L123
    def wait_for_save(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L120-L123
        # Store deferral is handled in get_finished(), which always runs even
        # when wait_for_save() is skipped (e.g. kv_connector_no_forward).
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L125-L134
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L125-L134
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)

        # Defer store jobs to the next step's start_kv_transfers. Done here
        # (rather than wait_for_save) so stores are queued even on steps where
        # wait_for_save is skipped.
        self.connector_worker.prepare_store_kv(self._connector_metadata)

        return self.connector_worker.get_finished(finished_req_ids)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L136-L139
    def build_connector_worker_meta(self) -> OffloadingWorkerMetadata | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L136-L139
        if self.connector_worker is not None:
            return self.connector_worker.build_connector_worker_meta()
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L141-L143
    def on_new_request(self, request: "Request") -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L141-L143
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L145-L151
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L145-L151
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L153-L159
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L153-L159
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L161-L165
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L161-L165
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L167-L169
    def has_pending_push_work(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L167-L169
        assert self.connector_scheduler is not None
        return self.connector_scheduler.has_pending_push_work()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L171-L173
    def update_connector_output(self, connector_output: KVConnectorOutput):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L171-L173
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L175-L181
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L175-L181
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L183-L189
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L183-L189
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L191-L193
    def take_events(self) -> Iterable[KVCacheEvent]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L191-L193
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L195-L197
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L195-L197
        return "HND"

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L199-L202
    def reset_cache(self) -> bool | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L199-L202
        assert self.connector_scheduler is not None
        self.connector_scheduler.reset_cache()
        return True

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L204-L207
    def get_kv_connector_stats(self) -> Any | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L204-L207
        # SUBTRACTED: 遥测链——删除项 7（契约签名保留默认 None 返回）。
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L209-L217
    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> Any | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L209-L217
        # SUBTRACTED: OffloadingConnectorStats 构建——删除项 7。
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L219-L229
    @classmethod
    def build_prom_metrics(cls, *args: Any, **kwargs: Any) -> Any:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L219-L229
        # SUBTRACTED: OffloadPromMetrics 构建——删除项 7。
        return None
