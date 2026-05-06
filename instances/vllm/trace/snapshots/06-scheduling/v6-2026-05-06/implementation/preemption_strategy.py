# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972
"""Three preemption strategies — recompute (v1), swap (v0), abort.

Ch04 covers the MECHANICS of `_preempt_request` — block-free + reset
`num_computed_tokens=0` + `waiting.prepend_request()`. THIS file is the
*algorithmic comparison* that motivates "why recompute, why not swap or
abort". Each strategy is exposed as an analytical model so a reader can
plug their own numbers in.

Strategies:

    RECOMPUTE  — drop KV, redo prefill on resume. v1's only path.
                 (vLLM v1 `_preempt_request`: scheduler.py:L961-L964)
    SWAP       — copy KV to CPU, copy back on resume. v0 had this.
                 (NOT in vLLM v1; only `kv_offload` for prefix cache)
    ABORT      — terminate the request entirely. Used only when no
                 preemption candidate exists or by the API on disconnect.
                 (vLLM v1 `finish_requests` with FINISHED_ABORTED:
                  scheduler.py:L1750-L1811)

Trade-off matrix (per request, 8K-token prompt at fp16, 32 layers, 8 KV-heads,
head_size=128, PCIe Gen4 ~32 GB/s, prefill ~50 K tok/s):

                    | RECOMPUTE | SWAP      | ABORT
    ----------------+-----------+-----------+-------------
    GPU↔CPU bytes   | 0         | 2 × KV    | 0
    Compute redo    | full      | 0         | n/a
    Latency cost    | 164 ms    | 62 ms RT  | 0 ms (lost)
    User-visible    | retry-ok  | retry-ok  | error / drop
    Code complexity | 1 path    | 2 paths   | 1 path
    OOM safety      | always ok | needs CPU | always ok
    Determinism     | bit-equal | bit-equal | n/a

vLLM v1's choice (recompute) trades a 2-3x latency penalty for simplicity,
no CPU memory dependency, and a single code path. Ch05 already laid out the
recompute-vs-swap trade-off for memory-management context; this file extends
to the abort case and to multi-request scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972
# (recompute), instances/vllm/source/vllm/v1/kv_offload/cpu/gpu_worker.py:L319
# (swap, used only for prefix-cache offload), and
# instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1750-L1811 (abort).
class PreemptionStrategy(Enum):
    RECOMPUTE = "recompute"
    SWAP = "swap"
    ABORT = "abort"


@dataclass
class PreemptionScenario:
    """All inputs the analytical model needs."""

    prompt_tokens: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    dtype_bytes: int = 2

    pcie_bandwidth_bytes_per_sec: float = 32 * 1024**3   # PCIe Gen4 x16
    prefill_throughput_tokens_per_sec: float = 50_000.0  # H100 at fp16
    abort_user_penalty_seconds: float = 5.0              # SLA penalty for dropped req

    @property
    def kv_bytes(self) -> int:
        # Same shape as Ch05's `PreemptionScenario.kv_bytes`. K and V both.
        # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L153-L170
        # REFERENCE: instances/vllm/source/vllm/v1/kv_cache_interface.py:L196-L204
        # (FullAttentionSpec.max_memory_usage_bytes uses the same per-block
        #  formula multiplied by ceil(L / block_size) blocks)
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_size
            * self.prompt_tokens
            * self.dtype_bytes
        )

    # Recompute: zero data movement, full re-prefill.
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L961-L964
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L967
    # (num_preemptions counter incremented; used by prefill_stats)
    def recompute_seconds(self) -> float:
        return self.prompt_tokens / self.prefill_throughput_tokens_per_sec

    # Swap: round-trip KV over PCIe.
    # REFERENCE: instances/vllm/source/vllm/v1/kv_offload/cpu/gpu_worker.py:L319
    # REFERENCE: instances/vllm/source/vllm/v1/kv_offload/base.py
    # (BlockIDsLoadStoreSpec describes the swap interface used by kv_offload)
    def swap_seconds(self) -> float:
        return (2 * self.kv_bytes) / self.pcie_bandwidth_bytes_per_sec

    # Abort: latency is zero TO THE ENGINE, but the user sees a dropped
    # request — we charge an SLA penalty.
    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L1750-L1811
    def abort_seconds(self) -> float:
        return self.abort_user_penalty_seconds

    def latency_for(self, strategy: PreemptionStrategy) -> float:
        if strategy is PreemptionStrategy.RECOMPUTE:
            return self.recompute_seconds()
        if strategy is PreemptionStrategy.SWAP:
            return self.swap_seconds()
        return self.abort_seconds()

    def winner(self) -> PreemptionStrategy:
        """Lowest-latency choice for THIS scenario.

        Note: vLLM v1 picks RECOMPUTE regardless. This function returns the
        latency-optimal pick to highlight WHY v1's universal choice is a
        conscious trade-off, not a latency optimum.
        """
        latencies = {s: self.latency_for(s) for s in PreemptionStrategy}
        return min(latencies, key=latencies.get)


def crossover_prompt_length(
    base: PreemptionScenario,
    *,
    pcie_bandwidth_bytes_per_sec: float | None = None,
    prefill_throughput_tokens_per_sec: float | None = None,
) -> int:
    """At what prompt length does swap become slower than recompute?

    Setting `recompute_seconds == swap_seconds`:
        L / TP        = 2 * (2 * NL * NH * D * dt) * L / BW
        1 / TP        = 4 * NL * NH * D * dt / BW
        BW / (4 NL NH D dt) = TP

    The L (prompt_length) cancels out — both scale linearly with L! So the
    crossover is a property of (BW, TP, model shape), independent of prompt
    size. If TP / BW > some threshold, recompute always wins. Otherwise
    swap always wins.

    Returns -1 if recompute always wins (TP > threshold), the threshold
    prompt length if exact equality, or 0 if swap always wins.
    """
    bw = pcie_bandwidth_bytes_per_sec or base.pcie_bandwidth_bytes_per_sec
    tp = prefill_throughput_tokens_per_sec or base.prefill_throughput_tokens_per_sec
    bytes_per_token = (
        4 * base.num_layers * base.num_kv_heads * base.head_size * base.dtype_bytes
    )
    threshold_tp = bw / bytes_per_token
    if tp >= threshold_tp:
        return -1  # recompute always faster regardless of L
    return 0       # swap always faster regardless of L


def expected_latency_under_oom_rate(
    scenario: PreemptionScenario,
    strategy: PreemptionStrategy,
    oom_probability_per_step: float,
    avg_steps_per_request: int,
) -> float:
    """E[latency] = base prefill time + p_oom * preempt_cost summed over steps.

    A simple model of throughput-weighted preemption cost. `p_oom` is the
    per-step probability of any KV-cache OOM that triggers preemption, and
    `avg_steps_per_request` is how many scheduler steps a typical request
    survives before completion.
    """
    base = scenario.recompute_seconds()  # always pay this once at admit
    expected_preempts = oom_probability_per_step * avg_steps_per_request
    return base + expected_preempts * scenario.latency_for(strategy)
