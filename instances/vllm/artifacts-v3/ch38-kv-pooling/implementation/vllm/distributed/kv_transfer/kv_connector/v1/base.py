# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVConnectorBase_V1：ch16 立下的双面契约（本章给它装上 offload 池化的腿）。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L7-L720
# SUBTRACTED: NIXL/EC 传输族与 xPU host buffer 旁路（set_host_xfer_buffer_ops）——
#   ch37 的面；观测面保留契约签名（get_kv_connector_stats → None）。
"""

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent
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
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L85-L92
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
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L117-L121
    if isinstance(connector, type):
        return issubclass(connector, SupportsHMA)
    else:
        return isinstance(connector, SupportsHMA)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L124-L130
class KVConnectorRole(enum.Enum):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L124-L130
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L132-L139
class KVConnectorHandshakeMetadata(ABC):  # noqa: B024
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L132-L139
    """Metadata used for out of band connector handshake between
    P/D workers. This needs to be serializable."""


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L141-L148
class KVConnectorMetadata(ABC):  # noqa: B024
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L141-L148
    """
    Abstract base class for KVConnectorMetadata.
    """


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L150-L171
class KVConnectorWorkerMetadata(ABC):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L150-L160
    """
    Abstract Metadata used to communicate back
    Worker KVConnector -> Scheduler KVConnector.

    Each worker can output its own metadata.
    For a single engine step, all metadata objects returned by workers
    will be aggregated using the `aggregate` method below, before
    being passed to the Scheduler KVConnector.
    """

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L162-L171
    @abstractmethod
    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L162-L171
        """
        Aggregate metadata with another `KVConnectorWorkerMetadata` object.
        """
        pass


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

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L219-L237
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L219-L224
        """Set the connector metadata from the scheduler."""
        self._connector_metadata = connector_metadata

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L235-L243
    def clear_connector_metadata(self) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L235-L243
        """Clear the connector metadata."""
        self._connector_metadata = None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L245-L261
    def has_connector_metadata(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L245-L261
        return self._connector_metadata is not None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L263-L271
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L263-L271
        """Initialize with the KV caches."""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L273-L288
    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L273-L288
        """Initialize with a single KV cache tensor used by all layers."""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L297-L302
    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L297-L302
        """Handle preempted requests or evicted blocks BEFORE they are overwritten."""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L304-L320
    @abstractmethod
    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L304-L320
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L322-L334
    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L322-L334
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L336-L357
    @abstractmethod
    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs
    ) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L336-L357
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L359-L367
    @abstractmethod
    def wait_for_save(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L359-L367
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L369-L385
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L369-L385
        """Get the finished recving and sending requests."""
        return None, None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L387-L405
    def get_block_ids_with_load_errors(self) -> set[int]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L387-L405
        """Get the set of block IDs that failed to load."""
        return set()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L407-L413
    def shutdown(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L407-L413
        """Shutdown the connector."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L415-L419
    def get_kv_connector_stats(self) -> "Any | None":
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L415-L419
        """Get the KV connector stats collected during the last interval."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L429-L439
    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L429-L439
        """Get the KVConnector handshake metadata for this connector."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L441-L453
    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L441-L453
        """Get the KVConnectorWorkerMetadata of this connector."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L465-L499
    @abstractmethod
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L465-L499
        """Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens."""
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L501-L525
    @abstractmethod
    def update_state_after_alloc(
        self, request: "Request", blocks: Any, num_external_tokens: int
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L501-L525
        """Update KVConnector state after block allocation."""
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L527-L539
    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L527-L539
        """Build the connector metadata for this step."""
        pass

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L541-L547
    def on_new_request(self, request: "Request") -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L541-L547
        """Called when a new request is added to the scheduler."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L549-L557
    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L549-L557
        """Update the connector state based on step output."""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L559-L578
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L559-L578
        """Called exactly once when a request has finished. Returns True if the
        request is being saved/sent asynchronously and blocks should not be
        freed until the request_id is returned from get_finished()."""
        return False, None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L580-L587
    def take_events(self) -> Iterable["KVCacheEvent"]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L580-L587
        """Take the KV cache events from the connector."""
        return ()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L589-L599
    def has_pending_push_work(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L589-L599
        """Return True if the connector has pending push work."""
        return False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L601-L618
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L601-L618
        """Get the required KV cache layout for this connector.
        Default implementation returns None (no requirement)."""
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L708-L720
    def reset_cache(self) -> bool | None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L708-L720
        """Reset the cache of the connector, if supported."""
        return None


__all__ = [
    "CopyBlocksOp",
    "KVConnectorBase_V1",
    "KVConnectorHandshakeMetadata",
    "KVConnectorMetadata",
    "KVConnectorWorkerMetadata",
    "KVConnectorRole",
    "SupportsHMA",
    "supports_hma",
]
