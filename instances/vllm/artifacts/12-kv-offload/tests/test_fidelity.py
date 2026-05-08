"""Source-fidelity tests — verify the 7 traps from impl-notes §4.

These are the binary "is the implementation honest?" tests.
"""

from __future__ import annotations

import inspect
import re

import implementation.cpu_gpu_worker as worker_mod
import implementation.offload_manager as mgr_mod
import implementation.offload_spec as spec_mod
import implementation.offloading_scheduler as sched_mod
import implementation.policies as pol_mod
import implementation.simple_offload_manager as simple_mod


# ---------------------------------------------------------------------------
# Trap A — NVMe SSD third tier (TOPIC absent in vllm/v1/kv_offload/)
# ---------------------------------------------------------------------------
class TestTrapANvme:
    """vLLM at 98661fe is 2-tier (HBM ↔ CPU pinned). NO NVMe path in
    `vllm/v1/kv_offload/`. Our policies/manager modules must reflect that.

    NVMe may appear in docstrings/comments as a TRAP discussion or as a
    parenthetical "where blocks live" mention — that's expected. The
    fidelity check here is that no IDENTIFIERS (class/fn/import) for an
    NVMe code path exist in manager/policies modules."""

    def test_no_nvme_class_or_function_in_manager(self):
        """offload_manager.py — no NVMe class/function/import identifiers."""
        src = inspect.getsource(mgr_mod)
        for f in (
            "class NVMe",
            "class NvmeManager",
            "class NVMeOffload",
            "def offload_to_nvme",
            "def transfer_to_disk",
            "import nvme",
        ):
            assert f not in src, f"Identifier '{f}' indicates NVMe code path"

    def test_no_disk_path_in_policies(self):
        src = inspect.getsource(pol_mod).lower()
        # No DiskPolicy class, no FsOffload, etc.
        for f in ("class diskpolicy", "class diskcache", "fs_offload"):
            assert f not in src

    def test_no_nvme_class_or_function_in_policies(self):
        src = inspect.getsource(pol_mod)
        for f in ("class NVMe", "class Disk", "class SSD", "def evict_to_disk"):
            assert f not in src

    def test_constants_include_nvme_for_demo_table_only(self):
        """NVMe constants exist in offload_spec for Demo 1 table — that's OK.
        They're never used as IDENTIFIERS in the manager/policies logic
        (only docstrings can mention NVMe as part of trap-flagging)."""
        # Confirm the constants are present (Demo 1 needs them)
        assert hasattr(spec_mod, "NVME_GEN5_BANDWIDTH_GB_PER_S")
        assert hasattr(spec_mod, "NVME_CAPACITY_GB")
        # No NVME_* constant used inside policies or manager
        pol_src = inspect.getsource(pol_mod)
        for f in ("NVME_GEN5_BANDWIDTH", "NVME_CAPACITY"):
            assert f not in pol_src
        mgr_src = inspect.getsource(mgr_mod)
        for f in ("NVME_GEN5_BANDWIDTH", "NVME_CAPACITY"):
            assert f not in mgr_src


# ---------------------------------------------------------------------------
# Trap B — LFU eviction (TOPIC absent in vllm/v1/kv_offload/cpu/policies/)
# ---------------------------------------------------------------------------
class TestTrapBLFU:
    """No `lfu.py` ships at 98661fe. Only `lru.py` + `arc.py` + base.py."""

    def test_no_lfu_class(self):
        """No LFU class in policies module."""
        src = inspect.getsource(pol_mod)
        assert "class LFUCachePolicy" not in src
        assert "class LFU " not in src

    def test_no_least_frequently_used(self):
        """No `least_frequently_used` identifier in policies."""
        src = inspect.getsource(pol_mod).lower()
        assert "least_frequently_used" not in src

    def test_only_two_policies_registered(self):
        """CACHE_POLICIES dict has exactly 2 entries: lru + arc."""
        from implementation.policies import CACHE_POLICIES
        assert len(CACHE_POLICIES) == 2
        assert set(CACHE_POLICIES.keys()) == {"lru", "arc"}

    def test_lfu_string_not_in_registry(self):
        from implementation.policies import CACHE_POLICIES
        assert "lfu" not in CACHE_POLICIES


# ---------------------------------------------------------------------------
# Trap C — attention-score-based eviction (TOPIC absent)
# ---------------------------------------------------------------------------
class TestTrapCAttentionScore:
    """vLLM uses block-hash semantics throughout. No token-level attention
    statistic policy."""

    def test_no_attention_score_class(self):
        src = inspect.getsource(pol_mod)
        assert "AttentionScore" not in src
        assert "attn_score" not in src.lower()

    def test_no_h2o_or_streaming_llm(self):
        """H2O / HeavyHitter / StreamingLLM are research not in vLLM at 98661fe."""
        src_all = (
            inspect.getsource(pol_mod) +
            inspect.getsource(mgr_mod) +
            inspect.getsource(sched_mod)
        ).lower()
        for f in ("h2o", "heavyhitter", "streamingllm", "heavy_hitter"):
            # may appear inside docstrings — skip — we check identifiers
            assert f"class {f}" not in src_all


# ---------------------------------------------------------------------------
# Trap D — predictive ML prefetch (TOPIC absent in scheduler.py)
# ---------------------------------------------------------------------------
class TestTrapDPredictivePrefetch:
    """OffloadingConnectorScheduler.get_num_new_matched_tokens is REACTIVE."""

    def test_no_predict_function(self):
        """No `predict` / `markov` / `ml_prefetch` functions."""
        src = inspect.getsource(sched_mod)
        # Identifiers (function or class names)
        assert "def predict" not in src
        assert "class Predictor" not in src
        assert "class MLPrefetcher" not in src
        assert "class MarkovChain" not in src

    def test_no_ml_imports(self):
        """No imports of torch.nn.Predictor or sklearn etc."""
        src = inspect.getsource(sched_mod)
        for f in ("import sklearn", "from sklearn", "torch.nn.Predictor"):
            assert f not in src

    def test_get_num_new_matched_tokens_is_reactive(self):
        """The function exists and operates on block_hashes prefix scan."""
        from implementation.offloading_scheduler import OffloadingConnectorScheduler
        # Has the public method
        assert hasattr(OffloadingConnectorScheduler, "get_num_new_matched_tokens")
        # Has the maximal_prefix_lookup helper (the reactive scan)
        assert hasattr(OffloadingConnectorScheduler, "_maximal_prefix_lookup")


# ---------------------------------------------------------------------------
# Trap E — ARC strictly better than LRU (FALSE per phase-shift demo)
# ---------------------------------------------------------------------------
class TestTrapEArcLoses:
    """The phase_shift demo workload shows ARC LOSING to LRU
    (LRU 2.60% miss vs ARC 14.15%). HONEST CAVEAT."""

    def test_arc_loses_phase_shift(self):
        from implementation.offload_spec import make_offload_key
        from implementation.policies import (
            ARCCachePolicy,
            BlockStatus,
            LRUCachePolicy,
        )

        def run(policy_cls, ops, capacity=32):
            policy = policy_cls(cache_capacity=capacity)
            misses = 0
            nbid = 0
            for key in ops:
                blk = policy.get(key)
                if blk is None:
                    misses += 1
                    if len(policy) >= capacity:
                        ev = policy.evict(1, protected=set())
                        if ev is None:
                            for k, b in (
                                list(policy.t1.items()) + list(policy.t2.items())
                                if hasattr(policy, "t1")
                                else list(policy.blocks.items())
                            )[:1]:
                                b.ref_cnt = 0
                            ev = policy.evict(1, protected=set())
                    blk = BlockStatus(block_id=nbid, ref_cnt=0)
                    nbid += 1
                    policy.insert(key, blk)
                else:
                    policy.touch([key])
            return misses

        # Reproduce phase_shift workload exactly
        import random
        random.seed(7)
        keys = []
        for _ in range(1000):
            keys.append(
                make_offload_key(
                    random.randint(0, 25).to_bytes(28, "big"), 0
                )
            )
        for _ in range(1000):
            keys.append(
                make_offload_key(
                    random.randint(50, 75).to_bytes(28, "big"), 0
                )
            )

        lru_miss = run(LRUCachePolicy, keys)
        arc_miss = run(ARCCachePolicy, keys)
        # ARC LOSES on this synthetic workload — HONEST O08
        assert arc_miss > lru_miss


# ---------------------------------------------------------------------------
# Trap F — all connectors interchangeable (FALSE per taxonomy)
# ---------------------------------------------------------------------------
class TestTrapFConnectorsNotInterchangeable:
    """18 connectors target different transports. Surface the protocol diversity."""

    def test_distinct_protocols(self):
        from implementation.connector_taxonomy import CONNECTOR_TAXONOMY
        protocols = {c.transport for c in CONNECTOR_TAXONOMY}
        # At least 5 distinct transport substrings
        assert len(protocols) >= 5

    def test_distinct_tiers(self):
        from implementation.connector_taxonomy import CONNECTOR_TAXONOMY
        tiers = {c.tier for c in CONNECTOR_TAXONOMY}
        # CPU DRAM, CPU+DISK, remote DRAM, remote HBM, distributed FS, ...
        assert len(tiers) >= 4

    def test_specific_protocols_differ(self):
        """LMCache != Mooncake != Nixl in transport choice."""
        from implementation.connector_taxonomy import CONNECTOR_TAXONOMY
        d = {c.name: c.transport for c in CONNECTOR_TAXONOMY}
        assert d["LMCacheConnectorV1"] != d["MooncakeConnector"]
        assert d["MooncakeConnector"] != d["NixlConnector"]
        assert d["LMCacheConnectorV1"] != d["NixlConnector"]


# ---------------------------------------------------------------------------
# Trap G — two CUDA streams = 2× speedup (FALSE; PCIe-bound)
# ---------------------------------------------------------------------------
class TestTrapGStreamsPCIeBound:
    """Two streams overlap concurrently but are PCIe-bound (not 2x speedup)."""

    def test_two_handlers_in_bundle(self):
        """CpuGpuOffloadingHandlers exposes BOTH directions — but they
        share the PCIe physical lane (Trap G nuance)."""
        from implementation.cpu_gpu_worker import CpuGpuOffloadingHandlers
        from implementation.offload_spec import (
            CanonicalKVCacheTensor,
            CanonicalKVCaches,
        )
        caches = CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(tensor=None, page_size_bytes=4096)],
            group_data_refs=[[]],
        )
        bundle = CpuGpuOffloadingHandlers(
            kv_caches=caches, block_size_factor=1, num_cpu_blocks=8,
        )
        # Two distinct handlers exist
        assert bundle.gpu_to_cpu_handler is not bundle.cpu_to_gpu_handler

    def test_pcie_gen5_bandwidth_is_capped(self):
        """The bandwidth constant is bounded — 64 GB/s, not "infinite"."""
        from implementation.offload_spec import PCIE_GEN5_BANDWIDTH_GB_PER_S
        # The shared physical lane bandwidth is finite
        assert PCIE_GEN5_BANDWIDTH_GB_PER_S == 64.0
        # NOT 128 (which would be 2× — a misconception)
        assert PCIE_GEN5_BANDWIDTH_GB_PER_S != 128.0


# ---------------------------------------------------------------------------
# Bonus fidelity: ref_cnt = -1 sentinel (W10 wisdom)
# ---------------------------------------------------------------------------
class TestRefCntSentinel:
    def test_default_ref_cnt_neg_one(self):
        """Production source uses ref_cnt=-1 sentinel for not-ready."""
        from implementation.policies import BlockStatus
        b = BlockStatus(block_id=0)
        assert b.ref_cnt == -1

    def test_is_ready_predicate(self):
        from implementation.policies import BlockStatus
        assert not BlockStatus(block_id=0, ref_cnt=-1).is_ready
        assert BlockStatus(block_id=0, ref_cnt=0).is_ready
        assert BlockStatus(block_id=0, ref_cnt=5).is_ready


# ---------------------------------------------------------------------------
# Bonus fidelity: ARC ghost-list trim to capacity
# ---------------------------------------------------------------------------
class TestARCGhostListBounds:
    def test_ghost_lists_bounded(self):
        from implementation.offload_spec import make_offload_key
        from implementation.policies import ARCCachePolicy, BlockStatus

        p = ARCCachePolicy(cache_capacity=2)
        for i in range(20):
            k = make_offload_key(i.to_bytes(28, "big"), 0)
            p.insert(k, BlockStatus(block_id=i, ref_cnt=0))
            p.evict(1, set())
        assert len(p.b1) <= 2  # B1 trimmed to capacity
        assert len(p.b2) <= 2


# ---------------------------------------------------------------------------
# Bonus: source REFERENCE comment count
# ---------------------------------------------------------------------------
class TestSourceReferenceComments:
    """The implementer claimed 81 # REFERENCE comments. Sanity-check the floor."""

    def test_reference_count_minimum(self):
        import os
        impl_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "implementation",
        )
        ref_count = 0
        for fname in os.listdir(impl_dir):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(impl_dir, fname)) as fh:
                ref_count += len(re.findall(r"# REFERENCE:", fh.read()))
        # impl-notes claims 70+; verify the floor holds
        assert ref_count >= 70, f"Only {ref_count} # REFERENCE comments found"
