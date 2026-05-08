"""Tests for connector_taxonomy — 18-connector enumeration + KVConnectorBase_V1.

Trap D anchor: connectors are NOT interchangeable.
"""

from __future__ import annotations

import pytest

from implementation.connector_taxonomy import (
    CONNECTOR_TAXONOMY,
    ConnectorEntry,
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
    connectors_by_scope,
    count_by_status,
)


# ---------------------------------------------------------------------------
# KVConnectorRole enum
# ---------------------------------------------------------------------------
class TestRoleEnum:
    def test_scheduler_role(self):
        assert KVConnectorRole.SCHEDULER.value == "scheduler"

    def test_worker_role(self):
        assert KVConnectorRole.WORKER.value == "worker"

    def test_two_roles_only(self):
        """Exactly TWO roles in the enum (some connectors instantiate as both)."""
        assert len(list(KVConnectorRole)) == 2


# ---------------------------------------------------------------------------
# KVConnectorBase_V1 abstract template
# ---------------------------------------------------------------------------
class TestConnectorBase:
    def test_has_load_bearing_methods(self):
        """The 12 load-bearing lifecycle methods are all present."""
        for m in (
            "register_kv_caches",
            "bind_connector_metadata",
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
            "wait_for_save",
            "get_finished",
            "get_num_new_matched_tokens",
            "update_state_after_alloc",
            "build_connector_meta",
            "take_events",
            "shutdown",
        ):
            assert hasattr(KVConnectorBase_V1, m)

    def test_role_stored(self):
        c = KVConnectorBase_V1(role=KVConnectorRole.WORKER)
        assert c.role == KVConnectorRole.WORKER

    def test_default_get_num_new_matched_tokens(self):
        """Default returns (0, False) — no offload tokens, not async."""
        c = KVConnectorBase_V1(role=KVConnectorRole.SCHEDULER)
        assert c.get_num_new_matched_tokens(None, 0) == (0, False)

    def test_default_get_finished_returns_empty_tuples(self):
        c = KVConnectorBase_V1(role=KVConnectorRole.WORKER)
        loaded, saved = c.get_finished(set())
        assert loaded == set()
        assert saved == set()

    def test_default_take_events(self):
        c = KVConnectorBase_V1(role=KVConnectorRole.SCHEDULER)
        assert c.take_events() == []

    def test_bind_connector_metadata(self):
        """bind stores metadata; available via _connector_metadata."""
        c = KVConnectorBase_V1(role=KVConnectorRole.WORKER)
        meta = KVConnectorMetadata()
        c.bind_connector_metadata(meta)
        assert c._connector_metadata is meta


# ---------------------------------------------------------------------------
# SupportsHMA marker
# ---------------------------------------------------------------------------
class TestSupportsHMA:
    def test_is_abstract_base(self):
        """SupportsHMA is an ABC mixin (marker, no abstract methods)."""
        # Should be subclassable without overrides (marker mixin).
        class ConcreteHMA(SupportsHMA):
            pass
        # SupportsHMA has no @abstractmethod, so subclass is instantiable
        # via cooperative multiple inheritance — but as an ABC alone it
        # isn't directly instantiable in some pythons; check it's an ABC.
        from abc import ABC
        assert issubclass(SupportsHMA, ABC)


# ---------------------------------------------------------------------------
# Taxonomy structure
# ---------------------------------------------------------------------------
class TestTaxonomyStructure:
    def test_total_count_18(self):
        """Demo verbatim: total connectors = 18 at vLLM 98661fe."""
        assert len(CONNECTOR_TAXONOMY) == 18

    def test_each_entry_is_connector_entry(self):
        """All rows have the ConnectorEntry shape."""
        for c in CONNECTOR_TAXONOMY:
            assert isinstance(c, ConnectorEntry)

    def test_each_entry_has_required_fields(self):
        """name / file / transport / tier / scope / status / notes."""
        for c in CONNECTOR_TAXONOMY:
            assert c.name
            assert c.file
            assert c.transport
            assert c.tier
            assert c.scope
            assert c.status
            assert c.notes

    def test_unique_names(self):
        """Each connector name appears exactly once."""
        names = [c.name for c in CONNECTOR_TAXONOMY]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Status counts — verbatim from demo-output
# ---------------------------------------------------------------------------
class TestStatusCounts:
    def test_demo_status_counts(self):
        """Demo verbatim: debug=3, production=11, reference=3, research=1."""
        counts = count_by_status()
        assert counts["debug"] == 3
        assert counts["production"] == 11
        assert counts["reference"] == 3
        assert counts["research"] == 1

    def test_status_sums_to_18(self):
        counts = count_by_status()
        assert sum(counts.values()) == 18


# ---------------------------------------------------------------------------
# Scope filters — verbatim from demo-output
# ---------------------------------------------------------------------------
class TestScopeFilters:
    def test_in_scope_ch12_count_7(self):
        """Demo verbatim: 'in scope (ch12) : 7'."""
        assert len(connectors_by_scope("ch12")) == 7

    def test_punted_ch22_25_count_6(self):
        """Demo verbatim: 'punted to ch22-ch25 : 6'."""
        assert len(connectors_by_scope("ch22-ch25")) == 6

    def test_research_plus_debug_5(self):
        """Demo verbatim: 'research / debug : 5'."""
        n = len(connectors_by_scope("research")) + len(connectors_by_scope("debug"))
        assert n == 5


# ---------------------------------------------------------------------------
# Trap D HONESTY: connectors are NOT interchangeable
# ---------------------------------------------------------------------------
class TestTrapDNotInterchangeable:
    """Surface protocol differences across connectors — they target different
    transports and use cases. The chapter must avoid 'all connectors equivalent'
    framing."""

    def test_offloading_connector_is_DMA(self):
        """OffloadingConnector targets DMA + cuMemcpyBatchAsync (CPU DRAM)."""
        c = next(x for x in CONNECTOR_TAXONOMY if x.name == "OffloadingConnector")
        assert "DMA" in c.transport
        assert c.tier == "CPU DRAM"

    def test_mooncake_is_RDMA(self):
        """MooncakeConnector uses RDMA — not interchangeable with CPU offload."""
        c = next(x for x in CONNECTOR_TAXONOMY if x.name == "MooncakeConnector")
        assert "RDMA" in c.transport

    def test_nixl_is_GPU_direct(self):
        """NixlConnector uses GPU-Direct over RDMA — different tier."""
        c = next(x for x in CONNECTOR_TAXONOMY if x.name == "NixlConnector")
        assert "GPU-direct" in c.transport.lower() or "rdma" in c.transport.lower()

    def test_lmcache_uses_disk(self):
        """LMCacheConnectorV1 has a disk tier — DMA connectors do NOT."""
        c = next(x for x in CONNECTOR_TAXONOMY if x.name == "LMCacheConnectorV1")
        assert "DISK" in c.tier or "disk" in c.tier.lower()

    def test_distinct_transport_protocols(self):
        """At least 5 distinct transport substrings appear — non-trivial diversity."""
        transports = {c.transport for c in CONNECTOR_TAXONOMY}
        assert len(transports) >= 5

    def test_only_offloading_connector_has_dma_to_cpu_dram(self):
        """OffloadingConnector + SimpleCPUOffloadConnector are the canonical
        CPU-offload pair; PD-disagg connectors target REMOTE memory tiers."""
        cpu_dram_only = [c for c in CONNECTOR_TAXONOMY if c.tier == "CPU DRAM"]
        names = {c.name for c in cpu_dram_only}
        assert "OffloadingConnector" in names
        assert "SimpleCPUOffloadConnector" in names
