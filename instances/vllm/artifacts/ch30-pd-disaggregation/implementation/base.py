# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Worker-side abstract contract every connector fills.

SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py
本章三类后端（P2P NCCL / NIXL RDMA / Offloading）都实现这同一套 worker 契约：
start_load_kv → wait_for_layer_load → save_kv_layer → wait_for_save → get_finished。
"""

from abc import ABC, abstractmethod


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorRole
class KVConnectorRole:
    # SUBTRACTED: 真实是 enum.Enum，精简版用类常量保留 SCHEDULER/WORKER 两值。
    SCHEDULER = "scheduler"
    WORKER = "worker"


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorMetadata
class KVConnectorMetadata:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorMetadata
    # ch29 决策侧 build_connector_meta 打包、随 SchedulerOutput 下发、worker 侧
    # bind_connector_metadata 吃进来的不透明载体。各后端有自己的子类。
    pass


class KVConnectorBase_V1(ABC):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorBase_V1
    def __init__(self, vllm_config, role, kv_cache_config=None):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorBase_V1.__init__
        # SUBTRACTED: _vllm_config/_kv_transfer_config 等字段的完整赋值（vllm base.py
        #             __init__）按各后端需要在子类里补，精简版基类只保留 metadata 槽位。
        self._connector_metadata = None
        self._vllm_config = vllm_config
        self._role = role

    @property
    def role(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorBase_V1.role
        return self._role

    # ==============================
    # Worker-side metadata binding
    # ==============================

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L217-L227
        """Set the connector metadata from the scheduler.

        Called by the model runner every time before model execution. The
        metadata will be used for runtime KV cache loading and saving.
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L229-L235
        """Clear the connector metadata. Called after model execution."""
        self._connector_metadata = None

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L237-L247
        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L249-L255
        return self._connector_metadata is not None

    # ==============================
    # Worker-side lifecycle contract
    # ==============================

    @abstractmethod
    def start_load_kv(self, forward_context, **kwargs) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L298-L314
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.
        """
        pass

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L316-L328
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        """
        pass

    @abstractmethod
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L330-L350
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.
        """
        pass

    @abstractmethod
    def wait_for_save(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L352-L361
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.
        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

    def get_finished(self, finished_req_ids):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L363-L379
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker. The scheduler process
        (via the Executors) uses this to track which workers are done.

        Returns tuple of (sending/saving ids, recving/loading ids).
        """
        return None, None

    def get_block_ids_with_load_errors(self) -> set:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L381-L399
        return set()

    # ==============================
    # Optional worker-side hooks (基类默认 no-op / None)
    # ==============================

    def get_kv_connector_stats(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:get_kv_connector_stats
        # SUBTRACTED: 各后端 stats/metrics/prom 扩展点（dossier delete 项）整组删去，
        #             基类默认 None，不影响 load/save/finished 传输闭环。
        return None

    def get_kv_connector_kv_cache_events(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:get_kv_connector_kv_cache_events
        return None

    def build_connector_worker_meta(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:build_connector_worker_meta
        return None
