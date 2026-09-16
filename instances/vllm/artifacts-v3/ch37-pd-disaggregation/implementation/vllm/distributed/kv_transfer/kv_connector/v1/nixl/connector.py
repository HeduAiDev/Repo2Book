# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL connector facade：按 role 只建半边子对象，其余全部转发。

This module hosts the thin facade classes that vLLM's KV-connector layer
instantiates. Almost all the real work lives in the per-mode scheduler
and worker classes; the connector classes here only forward calls.

* :class:`NixlBaseConnector` – common logic shared by pull and push.
* :class:`NixlPullConnector` – pull-based (READ) KV transfer.
* :class:`NixlPushConnector` – push-based (WRITE) KV transfer.
* ``NixlConnector`` – backward-compatible alias for :class:`NixlPullConnector`.

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L1-L395
# SUBTRACTED: stats/Prometheus 转发（get_kv_connector_stats / build_kv_connector_stats
#   / build_prom_metrics，L250-L275）——减法计划删除项 6；prefer_cross_layer_blocks
#   属性体（L82-L110）——删除项 4，保留签名返回 False。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
        NixlBaseConnectorScheduler,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L79-L320
class NixlBaseConnector(KVConnectorBase_V1, SupportsHMA):
    """Base connector with common logic shared by pull and push modes."""

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L82-L110
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # SUBTRACTED: [4] 跨层单张量布局的探测（Mamba 组/后端支持/HND 判定与
        #   enable_cross_layers_blocks extra config）——本章走逐层注册，恒 False。
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L82-L110
        return False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L112-L135
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None

        if vllm_config.kv_transfer_config.kv_role == "kv_both":
            logger.warning_once(
                "Using kv_role='kv_both' with NixlConnector is deprecated "
                "and will be removed in a future release. Please set "
                "kv_role='kv_producer' for prefill instances and "
                "kv_role='kv_consumer' for decode instances. "
            )

        self.kv_cache_config = kv_cache_config
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config
        # Subclasses must set self.connector_scheduler and self.connector_worker
        self.connector_scheduler: NixlBaseConnectorScheduler | None = None
        self.connector_worker: NixlBaseConnectorWorker | None = None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L140-L157
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L140-L157
        if vllm_config.model_config is None:
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        use_mla = vllm_config.model_config.use_mla
        if use_mla:
            # return None when we have mla
            # as the layout should not matter in that case,
            # which fallback to the default behavior.
            return None
        logger.info_once(
            "NixlConnector setting KV cache layout to HND for better xfer performance."
        )
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L163-L169
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L171-L177
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L179-L184
    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L186-L188
    def on_new_request(self, request: "Request") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.on_new_request(request)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L190-L192
    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L194-L200
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L202-L208
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L210-L221
    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set handshake metadata keyed by (pp_rank, tp_rank) so the side
        channel can serve every PP stage's agent metadata.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)

    ############################################################
    # Worker Side Methods
    ############################################################
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L226-L228
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L230-L234
    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_cross_layers_kv_caches(kv_cache)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L236-L238
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        assert self.connector_worker is not None
        self.connector_worker.set_host_xfer_buffer_ops(copy_operation)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L240-L243
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L245-L248
    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get block IDs that failed to load via NIXL."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L250-L253
    def get_kv_connector_stats(self):
        # SUBTRACTED: [6] NixlKVConnectorStats 转发——遥测旁路（契约签名保留）。
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L277-L279
    def wait_for_layer_load(self, layer_name: str) -> None:
        """NixlConnector does not do layerwise saving."""
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L281-L289
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """NixlConnector does not save explicitly."""
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L291-L295
    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        if self.connector_worker.use_host_buffer and self.connector_worker.copy_blocks:
            self.connector_worker.save_kv_to_host(self._connector_metadata)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L297-L300
    def has_pending_push_work(self) -> bool:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.has_pending_push_work()
        return False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L302-L306
    def shutdown(self):
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L308-L319
    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L322-L347
class NixlPullConnector(NixlBaseConnector):
    """Pull-based (READ) NIXL KV transfer connector."""

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L325-L341
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPullConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlPullConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L343-L347
    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self.connector_worker, NixlPullConnectorWorker)
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L350-L383
class NixlPushConnector(NixlBaseConnector):
    """Push-based (WRITE) NIXL KV transfer connector."""

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L353-L371
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.connector_scheduler: NixlPushConnectorScheduler | None = None
        self.connector_worker: NixlPushConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPushConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = NixlPushConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L373-L383
    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        """Drive push processing on the worker.

        The worker enqueues registrations / finished blocks for the
        background ``nixl-push-writer`` thread; the writer issues the
        WRITE transfers and polls NIXL notifs without further
        engine-thread involvement.
        """
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:L386-L387
# Backward compatibility: NixlConnector is the pull-based connector.
NixlConnector = NixlPullConnector


__all__ = [
    "NixlBaseConnector",
    "NixlConnector",
    "NixlPullConnector",
    "NixlPushConnector",
]
