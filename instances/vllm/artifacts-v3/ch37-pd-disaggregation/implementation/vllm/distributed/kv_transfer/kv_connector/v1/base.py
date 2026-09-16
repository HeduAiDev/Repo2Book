# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVConnectorBase_V1：ch16 立下的双面契约（本章给它装上第一条真实的腿）。

调度器侧五原语（get_num_new_matched_tokens / update_state_after_alloc /
build_connector_meta / update_connector_output / request_finished）与 worker 侧
收发族（start_load_kv / wait_for_layer_load / save_kv_layer / wait_for_save /
get_finished）。**本章只保留契约骨架与 NIXL 实际覆写的那些方法**。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L7-L720
# SUBTRACTED: 观测面（take_events / build_kv_connector_stats / build_prom_metrics /
#   reset_cache）、CoW 与 block-pool 绑定（bind_gpu_block_pool / handle_preemptions
#   的实现体）、EC 传输族——它们的实现归 ch16/ch36，本章的 P/D 主线不触。
"""

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L70-L82
# s_tensor_list, d_tensor_list, s_indices, d_indices, direction
CopyBlocksOp = Callable[
    [
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        list[int],
        list[int],
        Literal["h2d", "d2h"],
    ],
    None,
]


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L85-L114
class SupportsHMA(ABC):
    """
    The class that indicates the corresponding connector supports hybrid memory
    allocator (HMA).
    This is required to use the connector together with hybrid memory allocator.
    """

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L93-L114
    @abstractmethod
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L93-L114
        """Called exactly once when a request has finished for all kv cache groups,
        before its blocks are freed for each group.

        The connector may assume responsibility for freeing the blocks
        asynchronously by returning True.
        """
        raise NotImplementedError


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L117-L121
def supports_hma(connector: Any) -> bool:
    if isinstance(connector, type):
        return issubclass(connector, SupportsHMA)
    else:
        return isinstance(connector, SupportsHMA)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L124-L130
class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L132-L139
class KVConnectorHandshakeMetadata(ABC):  # noqa: B024
    """Metadata used for out of band connector handshake between
    P/D workers. This needs to be serializable."""


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L141-L148
class KVConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract base class for KVConnectorMetadata.
    """


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L171-L720
class KVConnectorBase_V1(ABC):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L196-L213
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L196-L213
        self._connector_metadata: KVConnectorMetadata | None = None
        self._vllm_config = vllm_config
        if vllm_config.kv_transfer_config is not None:
            self._kv_transfer_config = vllm_config.kv_transfer_config
        else:
            raise ValueError("kv_transfer_config must be set for KVConnectorBase_V1")
        self._kv_cache_config = kv_cache_config
        self._role = role

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L215-L217
    @property
    def role(self) -> KVConnectorRole:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L215-L217
        return self._role

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L176-L182
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L176-L182
        return False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L223-L235
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler."""
        self._connector_metadata = connector_metadata

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L235-L243
    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata."""
        self._connector_metadata = None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L263-L271
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Initialize with the KV caches."""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L290-L296
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L297-L302
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        """Handle preempted requests or evicted blocks BEFORE they are overwritten."""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L304-L320
    @abstractmethod
    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L304-L320
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L322-L334
    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L322-L334
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L336-L357
    @abstractmethod
    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs
    ) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L336-L357
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L359-L367
    @abstractmethod
    def wait_for_save(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L359-L367
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L369-L385
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        return None, None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L387-L395
    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get block ids of blocks that failed loading."""
        return set()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L407-L413
    def shutdown(self):
        """Shutdown the connector."""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L415-L427
    def get_kv_connector_stats(self) -> Any | None:
        """Get the KV connection stats for the connector.

        # SUBTRACTED: 遥测对象（KVConnectorStats）——减法计划删除项 6。
        """
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L429-L439
    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """Get the KVConnector handshake metadata for this connector."""
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L465-L499
    @abstractmethod
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L465-L499
        """Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens."""
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L501-L525
    @abstractmethod
    def update_state_after_alloc(
        self, request: "Request", blocks: Any, num_external_tokens: int
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L501-L525
        """Update KVConnector state after block allocation."""
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L527-L539
    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L527-L539
        """Build the connector metadata for this step."""
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L541-L547
    def on_new_request(self, request: "Request") -> None:
        """Called when a new request is added to the scheduler."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L549-L557
    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        """Update the connector state based on step output."""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L559-L578
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called exactly once when a request has finished. Returns True if the
        request is being saved/sent asynchronously and blocks should not be
        freed until the request_id is returned from get_finished()."""
        return False, None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L589-L599
    def has_pending_push_work(self) -> bool:
        """Return True if the connector has pending push work.

        Used by the engine to stay alive while a connector still has in-flight
        transfers that outlive all 'live' requests.
        """
        return False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L601-L618
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L601-L618
        """Get the required KV cache layout for this connector.
        Default implementation returns None (no requirement)."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L642-L652
    def get_finished_count(self) -> int | None:
        """Get the count of requests expected to complete send/receive operations
        via this connector. This method is used to initialize the
        KVOutputAggregator, overwriting the default world_size."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L665-L674
    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """Set the KV connector handshake metadata for this connector."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L676-L691
    def set_xfer_handshake_metadata_pp_aware(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """Set handshake metadata keyed by (pp_rank, tp_rank)."""
        if any(pp_rank != 0 for pp_rank, _ in metadata):
            raise ValueError(
                f"{type(self).__name__} received pp_rank > 0 handshake metadata "
                "but does not support PP-disaggregated KV transfer."
            )
        self.set_xfer_handshake_metadata(
            {tp_rank: meta for (_, tp_rank), meta in metadata.items()}
        )


__all__ = [
    "CopyBlocksOp",
    "KVConnectorBase_V1",
    "KVConnectorHandshakeMetadata",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "SupportsHMA",
    "supports_hma",
]
