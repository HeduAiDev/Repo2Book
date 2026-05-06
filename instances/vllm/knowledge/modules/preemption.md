# Preemption Knowledge

---

## P01: vLLM v1 has only ONE preemption strategy — recompute. Swap and abort are NOT preemption paths.

**Module**: preemption
**Chapter**: 06-scheduling
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: preemption, v1-design, code-paths

vllm/v1/core/sched/scheduler.py:L952-L972 (_preempt_request) is the sole preemption code path: free blocks, set num_computed_tokens=0, increment num_preemptions, prepend to waiting. The kv_offload subsystem (vllm/v1/kv_offload/cpu/gpu_worker.py:L319 swap_blocks_batch) is for PREFIX CACHE OFFLOAD only — never invoked by the scheduler on preemption. The abort path (`finish_requests` at L1750-L1811 with FINISHED_ABORTED) is triggered by API client disconnect or admin command, not by KV-cache OOM. Three paths exist in the codebase, only ONE is reachable via preemption.

---

## P02: Recompute-vs-swap crossover is prompt-length INDEPENDENT — depends only on (TP, BW, model shape)

**Module**: preemption
**Chapter**: 06-scheduling
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: preemption, analytical, trade-off

Setting recompute_seconds == swap_seconds: L/TP = 4*NL*NH*D*dt*L/BW. The L cancels. So the crossover is purely a function of (prefill_throughput, PCIe_bandwidth, model_shape). For a typical 32-layer / 8-KV-heads / head_size=128 / fp16 model: bytes_per_token = 4*32*8*128*2 = 256 KiB; threshold_TP = 32 GiB/s / 256 KiB = 131072 tok/s. Modern H100 prefill is ~50K tok/s, well below threshold → swap is faster regardless of prompt length. vLLM still chose recompute for non-latency reasons (simplicity, OOM-safety, code paths).

---

## P03: long_prefill_token_threshold gives 4x p95 TTFT at constant throughput

**Module**: preemption
**Chapter**: 06-scheduling
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: scheduling-config, fairness, pareto

vllm/v1/core/sched/scheduler.py:L413-L415 (Phase 1) and L678-L680 (Phase 2 — same threshold applied to waiting requests). Setting threshold=B/4 caps any single request at B/4 tokens/step, leaving 3B/4 for short requests. The Pareto sweep in Ch06 demo §5 confirms ~4x p95 TTFT improvement (200ms → 50ms in the 32-seq/2048-budget config) at modest throughput cost. Default is 0 (disabled).

---

## P04: TESTING the crossover length-independence — assert dispatch equality at extreme L

**Module**: preemption
**Chapter**: 06-scheduling
**Discovered by**: tester (Ch06 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: testing, crossover, p02-application

P02 says "the L cancels" in the recompute-vs-swap crossover equation. The natural temptation is to test "swap is faster than recompute at default config" — but that test would still pass if a future refactor re-introduced an L-dependence (e.g. `swap_seconds = (2 * KV / BW) + L * constant`). The cleanest fail-loudly test is:

```python
short = PreemptionScenario(prompt_tokens=128, ...)
long = PreemptionScenario(prompt_tokens=131072, ...)
assert crossover_prompt_length(short) == crossover_prompt_length(long)
```

`crossover_prompt_length` returns `-1 / 0` based on the dispatch decision, so equality at extreme L's catches any L-dependence in the formula. Used in `test_length_independent` and `test_P02_crossover_independent_of_prompt_length`. For testers writing assertions on derived analytical quantities: prefer "the dispatch is INVARIANT" over "the dispatch is X at default" — invariance tests catch a wider class of regressions.

---

## P05: 16x p95 TTFT improvement claim — sweep-pair sizing matters

**Module**: preemption
**Chapter**: 06-scheduling
**Discovered by**: tester (Ch06 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: pareto, p95-ttft, sweep-design

The implementer's "16x p95 TTFT improvement" claim isn't from a single threshold change; it's the ratio of the WORST sweep config (max_seqs=8, B=512 → 800ms) to the BEST (max_seqs=128, B=8192 → 50ms). The threshold rows (max_seqs=32/B=2048/threshold=512 and max_seqs=64/B=4096/threshold=1024) ALSO hit 50ms — that's the headline narrative point: you can match the best p95 TTFT at smaller engine sizes by enabling the threshold.

For testers: don't assert "16x improvement from threshold" at a single config — the sweep design is the proof. Test `estimate_p95_ttft(worst) / estimate_p95_ttft(best) == 16.0`, AND test that threshold rows hit 50ms at smaller configs. Both together prove the writer's point. Used in `test_sixteen_x_p95_ttft_improvement` + `test_with_threshold_short_fits_in_leftover`.
