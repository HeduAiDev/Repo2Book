# SPDX-License-Identifier: Apache-2.0
# Subtract-only companion for ch29《PD 分离的抽象与调度器集成》.
# 只做减法：与 vLLM 同名/同结构/同控制流，只删不增。
#
# 本文件是 vllm/distributed/kv_transfer/kv_connector/v1/base.py 的子集：
# 保留 role-split 契约的骨架 —— KVConnectorRole 二分、KVConnectorMetadata
# 单向信使、以及把方法显式切成『决策侧(Scheduler-side)/搬运侧(Worker-side)』
# 的 KVConnectorBase_V1。
"""
KVConnectorBase_V1 Class for Distributed KV Cache & Hidden State
communication in vLLM v1

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to load/save KV cache.
        get_num_new_matched_tokens() - get number of new tokens
            that exist in the remote KV cache.
        update_state_after_alloc() - update KVConnector state after
            temporary buffer alloc by the CacheManager.
        update_connector_output() - update KVConnector state after
            output is received from worker-side connectors.
        request_finished() - called once when a request is finished,
            returns whether KV cache should be freed now or if the
            connector now assumes responsibility for freeing the
            blocks asynchronously.

    Worker-side: runs in each worker, loads/saves KV cache to/from
    the Connector based on the metadata.
        start_load_kv() - starts loading all KVs (maybe async)
        wait_for_layer_load() - blocks until layer i load is done
        save_kv_layer() - starts saving KV for layer i (maybe async)
        wait_for_save() - blocks until all saves are done
        get_finished() - returns ids of requests that have completed
            async sending/recving.
"""
import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

# SUBTRACTED: SupportsHMA / supports_hma（HMA 多 kv_cache_group 路径）与 PD 分离
# 正交，精简版只走单 group 的 request_finished（原 base.py:L84-120）。

# SUBTRACTED: CopyBlocksOp 类型别名仅供 NIXL host-buffer 路径用（原 L69-79）。


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L123
class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L140
class KVConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate
    Scheduler KVConnector -> Worker KVConnector.
    """

    pass

# SUBTRACTED: KVConnectorHandshakeMetadata（P/D worker 间带外握手）与
# KVConnectorWorkerMetadata（worker→scheduler 回传聚合）是特定 connector 的进阶
# 信使，本章 role-split 骨架不需要（原 L131-167）。


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L170
class KVConnectorBase_V1(ABC):
    """
    Base class for KV connectors.
    """

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L183
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        self._connector_metadata: KVConnectorMetadata | None = None
        self._vllm_config = vllm_config
        if vllm_config.kv_transfer_config is not None:
            self._kv_transfer_config = vllm_config.kv_transfer_config
        else:
            raise ValueError("kv_transfer_config must be set for KVConnectorBase_V1")
        self._kv_cache_config = kv_cache_config
        self._role = role
        # SUBTRACTED: 实验性 API 警告 logger、kv_cache_config 缺失的弃用警告
        # 仅是提示，不影响契约（原 L189-206）。

    @property
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L209
    def role(self) -> KVConnectorRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L217
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.
        """
        self._connector_metadata = connector_metadata

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L229
    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata after model execution."""
        self._connector_metadata = None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L237
    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata. Should only be called inside the connector."""
        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    # SUBTRACTED: register_kv_caches / register_cross_layers_kv_cache /
    # set_host_xfer_buffer_ops / handle_preemptions 是 NIXL/offloading/cudagraph
    # 的可选钩子，基类均为 no-op 默认，本章 role-split 骨架不展开（原 L257-296）。

    @abstractmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L298
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.
        """
        pass

    @abstractmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L316
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        """
        pass

    @abstractmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L330
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.
        """
        pass

    @abstractmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L352
    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L363
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
        """
        return None, None

    # SUBTRACTED: get_block_ids_with_load_errors / shutdown / get_kv_connector_stats
    # / get_kv_connector_kv_cache_events / get_handshake_metadata /
    # build_connector_worker_meta 都是特定 connector 的可选扩展点，基类 no-op，
    # 与本章 role-split 骨架无关（原 L381-443）。

    # ==============================
    # Scheduler-side methods
    # ==============================

    @abstractmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L449
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.
        """
        # SUBTRACTED: 完整 Args 文档与 Notes 段（largest prefix / eviction 注意事项）
        # 正文用散文转述（原 L459-481）。
        pass

    @abstractmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L484
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.
        """
        pass

    @abstractmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L505
    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        """
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L520
    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        """
        Update KVConnector state from worker-side connectors output.
        """
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L530
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs.
        """
        return False, None

    # SUBTRACTED: take_events / get_required_kvcache_layout /
    # requires_piecewise_for_cudagraph / get_finished_count / build_kv_connector_stats
    # / set_xfer_handshake_metadata / build_prom_metrics / reset_cache 均为 metrics /
    # cudagraph / handshake 的可选扩展点，基类默认 no-op（原 L551-662）。
