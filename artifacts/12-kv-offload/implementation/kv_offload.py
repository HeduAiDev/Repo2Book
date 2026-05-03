"""
KV Cache Offload — Our Reimplementation.

REFERENCE sources:
    CPUOffloadingManager:   vllm/v1/kv_offload/cpu/manager.py
    LRU policy:             vllm/v1/kv_offload/cpu/policies/lru.py
    ARC policy:             vllm/v1/kv_offload/cpu/policies/arc.py
    swap_blocks_batch:      vllm/csrc/cache_kernels.cu:L78-L150
    OffloadingConnector:    vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
    SharedOffloadRegion:    vllm/v1/kv_offload/cpu/shared_offload_region.py

Storage tier hierarchy:
    GPU HBM (800 GB/s BW, 80 GB)  ← fastest, most expensive
        ↕ PCIe Gen5 x16 (~64 GB/s)
    CPU DRAM (50 GB/s BW, 512 GB+) ← slower, cheaper
        NOT SUPPORTED:
    NVMe SSD (7 GB/s BW, 2 TB+)   ← slowest, cheapest

Key design decisions:
    - Stores are DEFERRED to next step (don't delay current token generation)
    - Dedicated low-priority CUDA streams for DMA (don't compete with compute)
    - Pinned CPU memory for GPU Direct DMA (no CPU-side copy)
    - cuMemcpyBatchAsync for batched transfers (CUDA 12.8+)
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Set


# ═══════════════════════════════════════════════════════════════════════════
# BlockStatus & LRU Policy
# REFERENCE: vllm/v1/kv_offload/cpu/policies/lru.py
#            vllm/v1/kv_offload/cpu/policies/base.py:L10-L33
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OffloadBlock:
    """A block in the CPU offload pool."""
    block_id: int
    ref_cnt: int = -1    # -1 = not ready (data still being transferred)
    key: bytes = b''      # block hash for prefix cache lookup

    @property
    def is_ready(self) -> bool:
        return self.ref_cnt >= 0


class LRUPolicy:
    """
    Least-Recently-Used eviction for offloaded blocks.

    REFERENCE: vllm/v1/kv_offload/cpu/policies/lru.py:L1-L47

    Uses OrderedDict: iteration order = insertion order.
    Move recently touched items to end. Evict from beginning.

    Key insight: LRU is chosen over more sophisticated policies (ARC)
    because offload workloads are dominated by system prompt prefixes
    — blocks are either used once (streaming) or many times (prefix).
    LRU naturally keeps frequently-reused blocks (they keep getting
    touched to the end) and evicts one-shot blocks.
    """

    def __init__(self):
        self._cache: OrderedDict[bytes, OffloadBlock] = OrderedDict()

    def get(self, key: bytes) -> Optional[OffloadBlock]:
        blk = self._cache.get(key)
        if blk is not None and blk.is_ready:
            return blk
        return None

    def insert(self, key: bytes, block: OffloadBlock):
        block.key = key
        self._cache[key] = block

    def _get_any(self, key: bytes) -> Optional[OffloadBlock]:
        """Get block regardless of ready state (for complete_store)."""
        return self._cache.get(key)

    def touch(self, keys: List[bytes]):
        """Move keys to MRU end. REFERENCE: lru.py:touch()"""
        for key in reversed(keys):
            if key in self._cache:
                self._cache.move_to_end(key)

    def evict(self, n: int, protected: Set[bytes]) -> List[OffloadBlock]:
        """
        Evict n LRU blocks that are ready and not protected.

        REFERENCE: lru.py:L28-L42 — evict()
        """
        evicted = []
        iter_keys = list(self._cache.keys())
        for key in iter_keys:
            if len(evicted) >= n:
                break
            if key in protected:
                continue
            blk = self._cache[key]
            if blk.ref_cnt == 0:  # Not in use
                evicted.append(blk)
                del self._cache[key]
        return evicted

    def remove(self, key: bytes):
        self._cache.pop(key, None)

    def __len__(self):
        return len(self._cache)


# ═══════════════════════════════════════════════════════════════════════════
# CPUOffloadingManager
# REFERENCE: vllm/v1/kv_offload/cpu/manager.py
# ═══════════════════════════════════════════════════════════════════════════

class CPUOffloadingManager:
    """
    Manages the CPU offload block pool.

    REFERENCE: vllm/v1/kv_offload/cpu/manager.py:L1-L201

    Lifecycle:
        1. prepare_store(keys) → evict if needed, allocate blocks, mark ref_cnt=-1
        2. [DMA transfer GPU→CPU happens asynchronously]
        3. complete_store(keys) → mark ref_cnt=0 (now readable)
        4. lookup(key) → check if key is cached and ready
        5. prepare_load(keys) → increment ref_cnt to prevent eviction during load
        6. [DMA transfer CPU→GPU happens asynchronously]
        7. complete_load(keys) → decrement ref_cnt
    """

    def __init__(self, num_blocks: int, policy: str = "lru"):
        self.num_blocks = num_blocks
        self.blocks: List[OffloadBlock] = [
            OffloadBlock(block_id=i) for i in range(num_blocks)
        ]
        self._free_list: List[int] = list(range(num_blocks))
        self._policy = LRUPolicy() if policy == "lru" else None

    @property
    def num_free(self) -> int:
        return len(self._free_list)

    def lookup(self, key: bytes) -> bool:
        """Check if block is cached AND ready."""
        blk = self._policy.get(key)
        return blk is not None and blk.is_ready

    def prepare_store(self, keys: List[bytes]) -> dict:
        """
        Prepare to store new blocks. Evict if needed.

        REFERENCE: manager.py:prepare_store()
        """
        new_keys = [k for k in keys if self._policy._get_any(k) is None]
        if not new_keys:
            return {"stored": 0, "evicted": []}

        # Evict if not enough free blocks
        evicted = []
        num_to_evict = len(new_keys) - self.num_free
        if num_to_evict > 0:
            evicted = self._policy.evict(num_to_evict, protected=set())
            for blk in evicted:
                self._free_list.append(blk.block_id)

        # Allocate blocks (mark as not ready)
        for key in new_keys[:self.num_free]:
            blk_id = self._free_list.pop()
            blk = self.blocks[blk_id]
            blk.ref_cnt = -1    # Not ready
            blk.key = key
            self._policy.insert(key, blk)

        return {"stored": min(len(new_keys), len(new_keys) - num_to_evict),
                "evicted": [b.key for b in evicted]}

    def complete_store(self, keys: List[bytes], success: bool = True):
        """Mark blocks as ready (or free on failure). REFERENCE: manager.py:complete_store()"""
        for key in keys:
            blk = self._policy._get_any(key)  # Must find even non-ready blocks
            if blk is None:
                continue
            if success:
                blk.ref_cnt = 0  # Ready to read
            else:
                self._policy.remove(key)
                self._free_list.append(blk.block_id)
                blk.ref_cnt = -1

    def prepare_load(self, keys: List[bytes]):
        """Protect blocks during load. REFERENCE: manager.py:prepare_load()"""
        for key in keys:
            blk = self._policy._get_any(key)  # Use _get_any — block is ready (ref_cnt>=0)
            if blk is not None:
                blk.ref_cnt += 1

    def complete_load(self, keys: List[bytes]):
        """Release protection after load. REFERENCE: manager.py:complete_load()"""
        for key in keys:
            blk = self._policy._get_any(key)
            if blk is not None:
                blk.ref_cnt -= 1

    def touch(self, keys: List[bytes]):
        """Mark keys as recently used. REFERENCE: manager.py:touch()"""
        self._policy.touch(keys)


# ═══════════════════════════════════════════════════════════════════════════
# PCIe Bandwidth Analysis
# ═══════════════════════════════════════════════════════════════════════════

def pcie_bandwidth_analysis(
    seq_len: int, num_kv_heads: int, head_dim: int,
    num_layers: int, dtype_bytes: int = 2,
    pcie_generation: int = 5,
) -> dict:
    """
    Quantify how much PCIe bandwidth KV cache offloading needs.

    PCIe Generation reference:
        Gen4 x16: ~32 GB/s (bidirectional)
        Gen5 x16: ~64 GB/s (bidirectional)
        NVLink (H100): ~900 GB/s (bidirectional) — for comparison
    """
    pcie_bw = {4: 32, 5: 64}[pcie_generation]  # GB/s

    # KV cache bytes per layer per token
    kv_per_token = 2 * num_kv_heads * head_dim * dtype_bytes

    # One full sequence offload
    total_offload_bytes = kv_per_token * seq_len * num_layers
    offload_time_ms = total_offload_bytes / (pcie_bw * 1e9) * 1000

    # Per-step decode: store 1 new token's KV, maybe load 0-16 blocks
    per_step_store = kv_per_token * num_layers  # 1 new token
    per_step_load = kv_per_token * 16 * num_layers  # Up to 16 blocks prefetch
    per_step_total = per_step_store + per_step_load
    per_step_time_us = per_step_total / (pcie_bw * 1e9) * 1e6

    return {
        "pcie_bw_gbs": pcie_bw,
        "per_token_kv_bytes": kv_per_token,
        "full_seq_offload_gb": round(total_offload_bytes / (1024**3), 3),
        "full_seq_offload_ms": round(offload_time_ms, 1),
        "per_decode_step_kb": round(per_step_total / 1024, 1),
        "per_decode_step_us": round(per_step_time_us, 1),
        "overlap_feasibility": (
            "Per-step offload is <50µs for typical models — easily hidden "
            "behind GPU compute (2-5ms per decode step). Offload is asynchronous "
            "on separate CUDA streams → compute and DMA overlap."
        ),
    }


def demonstrate():
    print("KV Cache Offload: Policy & PCIe Analysis")
    print("=" * 60)

    # LRU eviction demo
    mgr = CPUOffloadingManager(num_blocks=5, policy="lru")
    result = mgr.prepare_store([b'hash_a', b'hash_b', b'hash_c'])
    mgr.complete_store([b'hash_a', b'hash_b', b'hash_c'])
    print(f"Stored: {result['stored']}, Evicted: {len(result['evicted'])}")
    print(f"Free blocks: {mgr.num_free}")

    # Lookup
    print(f"hash_a cached: {mgr.lookup(b'hash_a')}")
    print(f"hash_z cached: {mgr.lookup(b'hash_z')}")

    # Eviction
    mgr.prepare_store([b'hash_d', b'hash_e', b'hash_f'])  # Need 3, only 2 free
    print(f"\nAfter allocate: free={mgr.num_free}")

    # PCIe analysis
    print(f"\nPCIe Gen5 ×16 Bandwidth Analysis (DeepSeek V3, seq=32K):")
    r = pcie_bandwidth_analysis(32768, 1, 128, 64)
    print(f"  Full seq offload: {r['full_seq_offload_gb']} GB in {r['full_seq_offload_ms']} ms")
    print(f"  Per decode: {r['per_decode_step_kb']} KB in {r['per_decode_step_us']} µs")
    print(f"  {r['overlap_feasibility']}")


if __name__ == "__main__":
    demonstrate()
