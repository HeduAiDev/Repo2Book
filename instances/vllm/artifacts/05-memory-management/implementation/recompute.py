# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972
"""Preemption strategy: recompute vs. swap — and why vLLM v1 chose recompute.

This module is *pedagogical* — vLLM v1 has only one preemption path,
and that's recompute. The legacy v0 also had a swap-to-CPU path
(`SwapInBlocks`/`SwapOutBlocks`), which v1 removed for simplicity. The
trade-off below is what informs the design decision.

The "Swap" concept survives in v1 only as part of `kv_offload/` for *prefix
cache* offloading (CPU/disk caching of finished prefixes), NOT as a preemption
mechanism. See `instances/vllm/source/vllm/v1/kv_offload/cpu/gpu_worker.py`.

Trade-off matrix:

                       Recompute              Swap-to-CPU
    GPU ↔ CPU PCIe     0 bytes               O(KV bytes)
    Compute redo       O(prompt_len)          0
    Bandwidth need     0                     PCIe ~32 GB/s
    Latency cost       prefill time          KV / PCIe bandwidth
    Code complexity    1 path                 2 paths + cudaMemcpyAsync
    Determinism        same numerical result  bit-identical replay
    OOM safety         always works           fails if CPU also full

For an 8K prompt at fp16 with 32 layers, 8 KV-heads, head_size=128:
    KV size  = 8 * 128 * 2 * 32 * 8192 * 2 = ~1 GiB per request
    Swap cost   = 1 GiB / 32 GB/s = 31 ms (round-trip 62 ms)
    Recompute  = 8192 tokens / typical_prefill_throughput

Modern GPUs hit > 50K tok/s prefill, so recompute is ~160 ms — slower than
the *one-way* swap, but vLLM's authors decided the simplicity, OOM-safety,
and lack of CPU-memory dependency outweigh the latency penalty.

This module exposes both as analytical models so a reader can plug their own
numbers in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreemptionScenario:
    """Inputs to the recompute-vs-swap calculation."""

    # Sequence shape.
    prompt_tokens: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    dtype_bytes: int = 2

    # System throughput.
    pcie_bandwidth_bytes_per_sec: float = 32 * 1024**3  # 32 GB/s for PCIe Gen4 x16
    prefill_throughput_tokens_per_sec: float = 50_000.0

    # ────────────────────────────────────────────────────────────────────
    # Quantities
    # ────────────────────────────────────────────────────────────────────

    @property
    def kv_bytes(self) -> int:
        """Total KV cache bytes for this request's prompt.

        Formula: 2 * num_layers * num_kv_heads * head_size * prompt_tokens * dtype
        REFERENCE: same shape as `AttentionSpec.real_page_size_bytes` summed
        over `prompt_tokens` tokens (kv_cache_interface.py:L153-L170).
        """
        return (
            2  # K and V
            * self.num_layers
            * self.num_kv_heads
            * self.head_size
            * self.prompt_tokens
            * self.dtype_bytes
        )

    # ────────────────────────────────────────────────────────────────────
    # Recompute path: zero data movement, redo prefill.
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L964
    # `request.num_computed_tokens = 0` — full re-prefill.
    # ────────────────────────────────────────────────────────────────────

    @property
    def recompute_bytes_moved(self) -> int:
        return 0

    @property
    def recompute_seconds(self) -> float:
        # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972
        # The preempted request re-enters the waiting queue with
        # num_computed_tokens=0, so prefill is redone end-to-end.
        return self.prompt_tokens / self.prefill_throughput_tokens_per_sec

    # ────────────────────────────────────────────────────────────────────
    # Swap path: GPU→CPU on preempt, CPU→GPU on resume.
    # NOT IMPLEMENTED IN VLLM V1 (was in v0). Listed for the trade-off.
    # ────────────────────────────────────────────────────────────────────

    @property
    def swap_bytes_moved(self) -> int:
        # Out + In = 2 * KV
        return 2 * self.kv_bytes

    @property
    def swap_seconds(self) -> float:
        # REFERENCE: instances/vllm/source/vllm/v1/kv_offload/cpu/gpu_worker.py:L319
        # In v1, the only place GPU↔CPU KV transfer happens is the kv_offload
        # subsystem (prefix-cache offload), NOT preemption.
        return self.swap_bytes_moved / self.pcie_bandwidth_bytes_per_sec

    # ────────────────────────────────────────────────────────────────────
    # Crossover analysis.
    # ────────────────────────────────────────────────────────────────────

    @property
    def recompute_is_faster(self) -> bool:
        return self.recompute_seconds < self.swap_seconds

    def report(self) -> str:
        kv_gib = self.kv_bytes / (1024**3)
        return (
            f"  prompt_tokens     : {self.prompt_tokens}\n"
            f"  KV bytes/request  : {kv_gib:.3f} GiB\n"
            f"  Recompute (no IO) : {self.recompute_seconds * 1000:.1f} ms\n"
            f"  Swap round-trip   : {self.swap_seconds * 1000:.1f} ms\n"
            f"  → Recompute wins  : {self.recompute_is_faster}\n"
        )
