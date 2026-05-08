# Ch12 KV Cache Offload — package init
# REFERENCE: vllm/v1/kv_offload/__init__.py at commit 98661fe
"""Re-implementation of vLLM's KV cache offload subsystem.

Mirrors the production v1 layout:
    OffloadingSpec (ABC) + OffloadingSpecFactory   ← factory
    OffloadingManager (ABC) + CPUOffloadingManager  ← scheduler-side keyspace
    CachePolicy (ABC) + LRU + ARC                   ← eviction
    OffloadingHandler (ABC) + CpuGpuOffloadingHandlers  ← worker-side async copy
    OffloadingConnectorScheduler                     ← reactive prefetch
"""

from .offload_spec import (
    OffloadKey,
    LoadStoreSpec,
    BlockIDsLoadStoreSpec,
    GPULoadStoreSpec,
    CPULoadStoreSpec,
    PrepareStoreOutput,
    OffloadingEvent,
    ReqContext,
    OffloadingSpec,
    CPUOffloadingSpec,
    make_offload_key,
    get_offload_block_hash,
    get_offload_group_idx,
)
from .offload_manager import OffloadingManager, CPUOffloadingManager
from .reuse_manager import FilterReusedOffloadingManager
from .policies import (
    BlockStatus,
    CachePolicy,
    LRUCachePolicy,
    ARCCachePolicy,
)
from .factory import OffloadingSpecFactory
from .cpu_gpu_worker import (
    OffloadingHandler,
    OffloadingWorker,
    SingleDirectionOffloadingHandler,
    CpuGpuOffloadingHandlers,
    TransferResult,
    Transfer,
)
from .simple_offload_manager import SimpleCPUOffloadScheduler
from .connector_taxonomy import (
    KVConnectorRole,
    KVConnectorBase_V1,
    SupportsHMA,
    CONNECTOR_TAXONOMY,
)
from .offloading_scheduler import (
    OffloadingConnectorScheduler,
    SchedulerOffloadConfig,
    GroupOffloadConfig,
)

__all__ = [
    "OffloadKey",
    "LoadStoreSpec",
    "BlockIDsLoadStoreSpec",
    "GPULoadStoreSpec",
    "CPULoadStoreSpec",
    "PrepareStoreOutput",
    "OffloadingEvent",
    "ReqContext",
    "OffloadingSpec",
    "CPUOffloadingSpec",
    "make_offload_key",
    "get_offload_block_hash",
    "get_offload_group_idx",
    "OffloadingManager",
    "CPUOffloadingManager",
    "FilterReusedOffloadingManager",
    "BlockStatus",
    "CachePolicy",
    "LRUCachePolicy",
    "ARCCachePolicy",
    "OffloadingSpecFactory",
    "OffloadingHandler",
    "OffloadingWorker",
    "SingleDirectionOffloadingHandler",
    "CpuGpuOffloadingHandlers",
    "TransferResult",
    "Transfer",
    "SimpleCPUOffloadScheduler",
    "KVConnectorRole",
    "KVConnectorBase_V1",
    "SupportsHMA",
    "CONNECTOR_TAXONOMY",
    "OffloadingConnectorScheduler",
    "SchedulerOffloadConfig",
    "GroupOffloadConfig",
]
