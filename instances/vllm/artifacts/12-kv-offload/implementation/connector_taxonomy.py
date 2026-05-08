# SPDX-License-Identifier: Apache-2.0
"""
Connector taxonomy + KVConnectorBase_V1 mirror.

REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py (662 LOC)
REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/ — 18 connector .py files

This module pedagogically mirrors:
  * `KVConnectorBase_V1`     — abstract template with 30+ methods
  * `KVConnectorRole`        — enum (SCHEDULER / WORKER / both)
  * `KVConnectorMetadata`    — base for per-step metadata payload
  * `SupportsHMA`            — Hybrid Memory Allocation marker mixin
  * The TAXONOMY of all 18 connector implementations at this commit.

Why a taxonomy, not a single class (HARD GATE design decision 10):
  The 18 connectors at 98661fe target wildly different transports
  (Python dict, RDMA, DMA, file-system) and use cases (CPU offload,
  PD-disagg KV transfer, layerwise streaming, semantic cache). They
  share an abstract API but have different latency / capacity /
  dependency profiles. Trap D: "all connectors are interchangeable" is
  WRONG — selection is dictated by deployment topology + workload.

Outline reframe (Trap D variant):
  The chapter must walk this taxonomy honestly rather than pretend
  "offload = single class". OffloadingConnector is the canonical CPU
  offload; LMCache / Mooncake / Nixl are forward pointers (Ch22, Ch24,
  Ch29 hypothetical). See impl-notes O03 + O11 + O12.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L123-L130
class KVConnectorRole(Enum):
    """Connector role — same connector class can be instantiated as
    SCHEDULER (lifecycle, lookup, prepare_load), WORKER (transfer_async,
    register_kv_caches), or both depending on the deployment.

    PD-disagg connectors (Mooncake, Nixl) typically have one role per
    process: prefill node = WORKER + SCHEDULER, decode node = WORKER.
    """

    SCHEDULER = "scheduler"
    WORKER = "worker"


# REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L42-L83
@dataclass
class KVConnectorMetadata:
    """Base class for per-step connector metadata.

    Each connector subclass extends this with its own payload — e.g.
    OffloadingConnectorMetadata wraps load/store TransferJob lists;
    LMCacheConnectorMetadata wraps cache-line warmup hints.
    `bind_connector_metadata` swaps it into the worker each scheduler step.
    """

    # subclasses extend


# REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L84-L115
class SupportsHMA(ABC):
    """Marker mixin: this connector supports Hybrid Memory Allocation.

    HMA = the scheduler can allocate "virtual" KV blocks that physically
    live in CPU DRAM (or another tier) but logically count as part of
    the request's KV state. Without HMA, offload is a back-channel
    bypass; with HMA, the scheduler reasons about offloaded capacity
    as a first-class resource.

    Connectors that implement HMA: OffloadingConnector, MultiConnector,
    SimpleCPUOffloadConnector. Connectors that DO NOT: LMCacheConnectorV1
    (relies on its own internal allocator), the example/debug connectors.
    """

    # marker


# REFERENCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:L170-L660
class KVConnectorBase_V1(ABC):
    """Abstract template for v1 KV connectors.

    SIMPLIFIED: production has 30+ methods. We list the ~12 LOAD-BEARING
    ones here for the educational walkthrough. The rest are
    optional / role-conditional / for forward-compat.

    Lifecycle (per step):
      1. SCHED: get_num_new_matched_tokens(req, num_computed_tokens)
            → tells scheduler how many additional tokens are cache-hit
      2. SCHED: update_state_after_alloc(req, blocks, num_external_tokens)
            → scheduler allocated GPU dst slots; bind them to the load
      3. SCHED: build_connector_meta(scheduler_output)
            → produce the metadata payload for the worker
      4. WRKR: bind_connector_metadata(metadata)
            → install metadata for the upcoming step
      5. WRKR: start_load_kv(forward_context)
            → kick off async H2D transfers; blocks consumed by attn
      6. WRKR: wait_for_layer_load(layer_name)
            → blocks the model forward when the next layer's KV is needed
      7. WRKR: save_kv_layer(layer_name, kv_layer, attn_meta)
            → optional per-layer streaming save (Ch24)
      8. WRKR: wait_for_save() / get_finished(finished_req_ids)
            → fence the step's stores

    Key insight: lookup/prepare_load run on SCHEDULER; transfer runs on
    WORKER; they communicate via LoadStoreSpec inside KVConnectorMetadata.
    REFERENCE: same range L298-L362 (worker lifecycle), L449-L506 (sched lifecycle).
    """

    def __init__(
        self,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
    ) -> None:
        self.role = role
        self.kv_cache_config = kv_cache_config
        self._connector_metadata: Optional[KVConnectorMetadata] = None

    # --- worker lifecycle (subclass overrides) ---

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Worker: bind GPU KV-cache tensors. May also pin DRAM mirrors."""

    def bind_connector_metadata(
        self, metadata: KVConnectorMetadata
    ) -> None:
        """Worker: install per-step metadata (cleared each step)."""
        self._connector_metadata = metadata

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Worker: kick off all H2D transfers for this step (async)."""

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Worker: block until the named layer's KV is loaded.
        Typically a no-op for layer 0 (already loaded), then blocks
        on layer N's H2D transfer if it hasn't completed."""

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """Worker: optional per-layer save (used by layerwise connectors)."""

    def wait_for_save(self) -> None:
        """Worker: block until all in-flight stores have been kicked off."""

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str], set[str]]:
        """Worker: drain (finished_load_reqs, finished_save_reqs)."""
        return set(), set()

    # --- scheduler lifecycle (subclass overrides) ---

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        """Scheduler: how many additional tokens beyond num_computed_tokens
        can be loaded from offload?

        Returns (num_tokens, is_async).
        - num_tokens=None means defer (LMCache async lookup not ready yet).
        - is_async=True means tokens will arrive between scheduler steps.
        """
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Scheduler: bind newly allocated GPU dst blocks to the load."""

    def build_connector_meta(
        self, scheduler_output: Any
    ) -> KVConnectorMetadata:
        """Scheduler: produce the per-step metadata payload."""
        return KVConnectorMetadata()

    def take_events(self) -> list[Any]:
        """Scheduler: drain emitted KV cache events for prom-metrics."""
        return []

    def shutdown(self) -> None:
        """Release all resources (GPU streams, pinned mem, RDMA queues)."""


# REFERENCE: directory listing of vllm/distributed/kv_transfer/kv_connector/v1/
@dataclass
class ConnectorEntry:
    """One row in the taxonomy: name, transport, tier, scope, status."""

    name: str
    file: str
    transport: str
    tier: str
    scope: str  # "ch12" | "ch22-ch25" | "research" | "debug"
    status: str  # "production" | "research" | "reference" | "debug"
    notes: str


# Trap D anchor: the 18 connectors at 98661fe are NOT interchangeable.
CONNECTOR_TAXONOMY: list[ConnectorEntry] = [
    # CPU-offload family — Ch12 IN-SCOPE
    ConnectorEntry(
        name="OffloadingConnector",
        file="offloading_connector.py",
        transport="DMA + cuMemcpyBatchAsync",
        tier="CPU DRAM",
        scope="ch12",
        status="production",
        notes="Canonical CPU offload; composes with vllm/v1/kv_offload/. "
              "192 LOC. Implements SupportsHMA. Walks LRU/ARC, ghost lists, "
              "shared mmap region.",
    ),
    ConnectorEntry(
        name="SimpleCPUOffloadConnector",
        file="simple_cpu_offload_connector.py",
        transport="DMA + cuMemcpyBatchAsync",
        tier="CPU DRAM",
        scope="ch12",
        status="reference",
        notes="247 LOC. Pedagogical baseline; pairs with vllm/v1/simple_kv_offload/. "
              "Single-rank, LRU only, no ghost lists. Used as the §3 anchor.",
    ),
    # Composite
    ConnectorEntry(
        name="MultiConnector",
        file="multi_connector.py",
        transport="composed",
        tier="multi-tier",
        scope="ch12",
        status="production",
        notes="629 LOC. Composes multiple connectors (e.g. LMCache + "
              "OffloadingConnector). Implements SupportsHMA + KVConnectorBase_V1.",
    ),
    # Production semantic cache (forward pointer to a hypothetical Ch29)
    ConnectorEntry(
        name="LMCacheConnectorV1",
        file="lmcache_connector.py",
        transport="LMCache RPC + disk tier",
        tier="CPU + DISK",
        scope="ch12",
        status="production",
        notes="354 LOC. Flagship production semantic-cache library; disk-backed "
              "prefix cache. Out-of-scope deep dive — forward pointer only.",
    ),
    ConnectorEntry(
        name="LMCacheMpConnector",
        file="lmcache_mp_connector.py",
        transport="multi-process LMCache",
        tier="CPU + DISK",
        scope="ch12",
        status="production",
        notes="Multi-process variant of LMCacheConnectorV1.",
    ),
    # PD-disagg family — punt to Ch22+
    ConnectorEntry(
        name="MooncakeConnector",
        file="mooncake/mooncake_connector.py",
        transport="RDMA (MooncakeStore)",
        tier="remote DRAM via RDMA",
        scope="ch22-ch25",
        status="production",
        notes="Moonshot's RDMA-based KV-disaggregation backend. Ch22 territory.",
    ),
    ConnectorEntry(
        name="NixlConnector",
        file="nixl/connector.py",
        transport="NVIDIA NIXL (RDMA + GPU-direct)",
        tier="remote HBM",
        scope="ch22-ch25",
        status="production",
        notes="NVIDIA Inference Xfer Library. GPU-to-GPU over RDMA. Ch24.",
    ),
    ConnectorEntry(
        name="HF3FSConnector",
        file="hf3fs/hf3fs_connector.py",
        transport="HuggingFace 3FS distributed FS",
        tier="distributed file system",
        scope="ch22-ch25",
        status="production",
        notes="HuggingFace 3FS-backed KV transfer; remote scale-out.",
    ),
    # Layerwise / experimental
    ConnectorEntry(
        name="FlexKVConnector",
        file="flexkv_connector.py",
        transport="experimental",
        tier="varies",
        scope="research",
        status="research",
        notes="Research/experimental flex-KV; layout-aware connector.",
    ),
    # P2P sub-family
    ConnectorEntry(
        name="P2P_Connector_NCCL",
        file="p2p/p2p_nccl_connector.py",
        transport="NCCL P2P send/recv",
        tier="GPU↔GPU intra-node",
        scope="ch22-ch25",
        status="production",
        notes="P2P KV transfer over NCCL. PD-disagg fast path.",
    ),
    # MORI-IO transport
    ConnectorEntry(
        name="MoriIO_Connector",
        file="moriio/connector.py",
        transport="MoriIO transport",
        tier="varies",
        scope="ch22-ch25",
        status="production",
        notes="MoriIO transport adapter (KV transfer over fabric).",
    ),
    # Sub-package: offloading/* (the production CPU offload internals)
    ConnectorEntry(
        name="OffloadingConnectorScheduler",
        file="offloading/scheduler.py",
        transport="(scheduler internals)",
        tier="CPU DRAM",
        scope="ch12",
        status="production",
        notes="881 LOC. Scheduler-side prefix lookup + sliding-window lookup "
              "+ store job builder. Pure logic, not a connector itself.",
    ),
    ConnectorEntry(
        name="OffloadingConnectorWorker",
        file="offloading/worker.py",
        transport="(worker internals)",
        tier="CPU DRAM",
        scope="ch12",
        status="production",
        notes="370 LOC. Worker-side handle_preemptions + start_kv_transfers.",
    ),
    # LMCache integration helpers
    ConnectorEntry(
        name="LMCache_Integration",
        file="lmcache_integration/...",
        transport="LMCache adapters",
        tier="varies",
        scope="ch22-ch25",
        status="reference",
        notes="LMCache integration helpers and metric adapters.",
    ),
    # Reference / debug / examples (NOT for production traffic)
    ConnectorEntry(
        name="ExampleConnector",
        file="example_connector.py",
        transport="(reference)",
        tier="(none)",
        scope="debug",
        status="debug",
        notes="Reference impl skeleton; how to write a new connector.",
    ),
    ConnectorEntry(
        name="ExampleHiddenStatesConnector",
        file="example_hidden_states_connector.py",
        transport="(reference)",
        tier="(none)",
        scope="debug",
        status="debug",
        notes="Variant showing hidden-states transfer pattern.",
    ),
    ConnectorEntry(
        name="DecodeBenchConnector",
        file="decode_bench_connector.py",
        transport="(synthetic)",
        tier="(none)",
        scope="debug",
        status="debug",
        notes="Benchmark connector: synthesizes loads to stress-test decode path.",
    ),
    ConnectorEntry(
        name="SsmConvTransfer",
        file="ssm_conv_transfer_utils.py",
        transport="(SSM-specific helpers)",
        tier="(none)",
        scope="research",
        status="reference",
        notes="State-space-model conv-state transfer helpers (Mamba hybrid).",
    ),
]


def connectors_by_scope(scope: str) -> list[ConnectorEntry]:
    """Filter taxonomy by scope (used by demos to slice the table)."""
    return [c for c in CONNECTOR_TAXONOMY if c.scope == scope]


def count_by_status() -> dict[str, int]:
    """Demo helper — counts of production / research / reference / debug."""
    out: dict[str, int] = {}
    for c in CONNECTOR_TAXONOMY:
        out[c.status] = out.get(c.status, 0) + 1
    return out
