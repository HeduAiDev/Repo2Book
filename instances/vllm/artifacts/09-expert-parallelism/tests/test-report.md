# Test Report — Ch09 Expert Parallelism

**Tester**: tester@book-factory
**Date**: 2026-05-07
**Source commit**: `98661fe`
**Verdict**: APPROVED → handoff to Writer

## Summary

```
204 passed in 2.24s
```

| Module | Tests | Status |
|---|---|---|
| `test_routing.py` | 28 | PASS |
| `test_expert_map.py` | 24 | PASS |
| `test_ep_groups.py` | 27 | PASS |
| `test_all2all_baseline.py` | 20 | PASS |
| `test_fused_moe_block.py` | 30 | PASS |
| `test_eplb.py` | 32 | PASS |
| `test_mixtral_vs_deepseek.py` | 16 | PASS |
| `test_integration.py` | 18 | PASS |
| `test_smoke.py` | 9 | PASS |

Run from chapter dir:

```bash
cd instances/vllm/artifacts/09-expert-parallelism
/home/zjq/.conda/envs/mujoco/bin/python -m pytest tests/ --ignore=tests/_legacy -q
```

Floor target was ≥80; achieved **204** (Ch08 baseline was 144). Breadth comes
from parametrising ep_size ∈ {1, 2, 4, 8} in routing/expert_map/integration,
two routing-path families (Mixtral fused_topk + DeepSeek grouped_topk) at
each test level, and exhaustive demo-numerics pins across §3.1-§3.5.

## Demo numerics — every headline number reproduced (writer quotes verbatim)

### §3.1 Top-K routing distributions

```
Mixtral (E=8, K=2):
  per_expert_count = [250, 285, 277, 243, 253, 272, 247, 221]
  max=285  min=221  mean=256.00  coverage=1.000
  per-token weight sum: min=1.0000  max=1.0000  mean=1.0000

DeepSeek-V2 grouped (E=64, K=6, n_group=8, topk_group=3):
  max=131  min=78  mean=96.00  coverage=1.000
  per-token weight sum: min=1.0000  max=1.0000  mean=1.0000

Renormalize on/off (Mixtral, K=2):
  renormalize=True  → sum range [1.0000, 1.0000]  mean 1.0000
  renormalize=False → sum range [0.2730, 0.6171]  mean 0.3899
```

Pinning tests: `test_section_31_mixtral_distribution_verbatim`,
`test_section_31_deepseek_distribution_verbatim`,
`test_section_31_renormalize_off_range_verbatim`.

### §3.2 All-to-all alpha-beta (NVLink, P=8)

```
       128 tokens →  T_AR=  16.09μs   T_A2A=   8.05μs   ratio=2.000
      1024 tokens →  T_AR=  67.47μs   T_A2A=  33.74μs   ratio=2.000
      8192 tokens →  T_AR= 478.51μs   T_A2A= 239.26μs   ratio=2.000
     65536 tokens →  T_AR=3766.85μs   T_A2A=1883.42μs   ratio=2.000

IB headlines: 128 tok → 50.70μs;  65536 tok → 18804.48μs
```

Pinning tests: `test_section_32_nvlink_alpha_beta_table_verbatim`,
`test_section_32_ib_alpha_beta_headlines`.

### §3.3 Per-rank load — placement × ep_size, hot 20% gets 60% of routes

```
hot 20% of experts received 4915/8192 routed pairs (0.600)

placement      ep_size     rank loads                           max/mean
linear         1           [8192]                                1.000
linear         4           [5175, 980, 1017, 1020]               2.527
linear         8           [3329, 1846, 483, 497, 458, 559,
                            515, 505]                            3.251
round_robin    4           [2350, 2420, 1695, 1727]              1.182
round_robin    8           [1199, 1195, 1186, 1205, 1151,
                            1225, 509, 522]                      1.196
```

Pinning tests: `test_section_33_placement_table_verbatim`,
`test_section_33_hot_total_verbatim`,
`test_section_33_max_mean_ratios_verbatim`.

**Trap A evidence**: linear placement at ep=8 gives **max/mean=3.251** — one
rank does 3.25× the work of the mean rank. Round-robin reduces this to
**1.196** (a 2.7× improvement). Memory shards perfectly with EP; throughput
does NOT, because the routing distribution is skewed.

### §3.4 EP×TP weight memory (E=64 DeepSeek-V2-Lite block)

```
Per-expert params: 3·1408·2048 = 8,650,752
Total params:       E·3·intermediate·hidden = 553,648,128 (1056 MiB at bf16)

  ep  tp    mem/rank (MiB)    reduction
   1   1            1056.00         1.00x
   4   1             264.00         4.00x
   4   2             132.00         8.00x
   8   2              66.00        16.00x
  16   1              66.00        16.00x
   8   4              33.00        32.00x
```

Pinning tests: `test_section_34_memory_table_verbatim`,
`test_section_34_total_params_verbatim`,
`test_memory_inverse_proportional_to_ep_times_tp`.

Cross-cell invariant: `mem ∝ 1 / (ep × tp)`. Verified that (4,2), (8,1),
(2,4) all give the same per-rank MiB.

### §3.5 EPLB rebalance timeline (100 steps, E_logical=32, redundant=4, ep=4)

```
step    placement       per-rank load                              max/mean
   0    linear          [1292, 246, 257, 253]                       2.523
  25    linear          [1295, 230, 261, 262]                       2.529
  50    round_robin     [591, 616, 423, 418]                        1.203  ← EPLB triggered
  51    round_robin     [575, 593, 463, 417]                        1.158
  75    round_robin     [594, 629, 391, 434]                        1.229
  99    round_robin     [582, 611, 434, 421]                        1.193

EplbState: num_logical=32, num_redundant=4, num_physical=36
physical_to_logical[0:8]=[0, 1, 2, 3, 4, 5, 6, 7]
physical_to_logical[-4:]=[5, 2, 0, 4]   (after rearrangement)
```

Pinning tests: `test_section_35_step0_imbalance_pin`,
`test_section_35_step25_imbalance_pin`,
`test_section_35_step50_post_rebalance_pin`,
`test_section_35_eplb_layout_after_100_steps_verbatim`.

Imbalance ratio drops 2.523 → 1.203 (a **2.1× improvement**) at step 50 when
EPLB rearranges the layout; the redundant slots `[5, 2, 0, 4]` show the four
hottest logical experts duplicated.

## 7-trap fidelity verification — all pinned

### Trap A: "EP=N gives N× capacity for the same compute" — WRONG

**Method**: `test_section_33_max_mean_ratios_verbatim` and
`test_section_33_placement_table_verbatim`.

EP shards the parameter store by N (each rank holds E/N experts), but per-token
compute is `K × per_expert_FLOPs` regardless of N. Under skewed routing, linear
placement at ep=8 produces **max/mean=3.251**: rank 0 does 3.25× the mean
rank's work. Throughput is gated by the slowest rank. Memory scales with EP;
throughput does not, except via EPLB shuffling hot experts.

### Trap B: All-to-all is symmetric / "cost = all-reduce / 2" — WRONG IN PRACTICE

**Method**: `test_section_32_nvlink_alpha_beta_table_verbatim` plus
`test_dispatch_combine_round_trip` and `test_all_gatherv_concatenates`.

The α-β model's clean 2× ratio (`T_AR / T_A2A = 2.000` exactly across all
payloads) is the textbook claim. But vLLM's AgRs path is `all_gatherv +
reduce_scatterv` (with per-rank `sizes`) — not a true symmetric all-to-all.
Under skewed routing, per-rank chunks differ; the `_v` variants exist exactly
for that reason. Production DeepEP backends use fused IBGDA/NVLink kernels
that have different cost profiles than the α-β baseline.

### Trap C: Experts are independent so EP scaling is free — WRONG

**Method**: `test_rearrange_hot_experts_get_redundant_slots` (test_eplb.py),
`test_deepseek_config_matches_source` (test_mixtral_vs_deepseek.py),
`test_trap_C_shared_experts_DO_NOT_appear_in_FusedMoEBlock_memory`
(test_integration.py).

Two coupling effects:
1. **Routing concentration** — hot experts produce per-rank load skew
   (verified by §3.3 placement table).
2. **Shared experts** — DeepSeek's `n_shared_experts` are constructed
   OUTSIDE FusedMoE and replicated on every rank regardless of ep_size.
   Test `test_deepseek_config_matches_source` asserts `has_shared_expert=True`
   and `test_trap_C_shared_experts_DO_NOT_appear_in_FusedMoEBlock_memory`
   demonstrates the memory-model omission.

### Trap D: EPLB is a free runtime bolt-on — WRONG

**Method**: `test_eplb_group_is_distinct_object_from_ep`,
`test_eplb_not_created_when_disabled`,
`test_trap_D_eplb_separate_group_prevents_aliasing`.

Three side doors verified:
1. EPLB requires a SEPARATE `_EPLB` process group (object-identity check
   `ep is not eplb`) — `parallel_state.py:L1700-L1719`. The deadlock-prevention
   comment is the load-bearing piece of evidence.
2. `_EPLB` and `_EP` share the rank list but are distinct Python objects
   (and in production, distinct NCCL communicators).
3. Production EPLB additionally requires `supports_eplb` on the quant method
   and forbids round-robin placement (`layer.py:L548-L557, L168-L171`).

### Trap E: Aux loss is what makes MoE balanced in vLLM — WRONG

**Method**: `test_trap_E_no_loss_balance_in_eplb_module`,
`test_trap_E_no_optimizer_in_module`,
`test_trap_E_no_gradient_tracking_in_record_step`,
`test_trap_E_vllm_source_eplb_has_no_aux_loss_computation`.

Negative tests at three scopes:
1. **Our EPLB module**: AST-strip docstrings, then assert no `.backward(`,
   `torch.optim`, `Optimizer(`, `compute_aux_loss`, etc.
2. **No autograd tracking**: `record_step` detaches the load tensor; the
   loaded history has `requires_grad == False`.
3. **vLLM source verification**: grep `vllm/distributed/eplb/` for `.backward(`,
   `compute_aux_loss`, `compute_balance_loss`, `router_aux_loss_coef`,
   `aux_loss =`. ZERO hits across all 5 .py files in that directory.

EPLB is a runtime statistical rebalancer. Aux loss is a training mechanism
(Switch Transformer L_balance) — it shaped the trained model's expert
distribution, but vLLM is inference-only and never invokes it.

(Caveat carried in knowledge E10: vLLM's HF-config carriers like
`router_aux_loss_coef` ARE present as STORED attributes in some model
configs — but they are dead storage, never used in any forward or loss
computation. The directory `vllm/distributed/eplb/` is clean.)

### Trap F: FusedMoE.forward always calls dispatch then experts then combine — WRONG

**Method**: `test_use_all2all_kernels_requires_dp_and_ep` (test_ep_groups.py).

Real all-to-alls happen only when `dp_size > 1 AND use_ep` per
`config.py:L1019-L1020`. With `dp_size==1`, the fast-path
`MoEPrepareAndFinalizeNoDPEP` is selected — `dispatch` is identity, `combine`
is identity. The single-process simulation here makes this implicit (every
rank sees every token); the test pins the production gating condition.

### Trap G: Top-K then softmax = softmax then Top-K — WRONG (under no-renorm)

**Method**: `test_trap_G_softmax_topk_does_not_commute_with_topk_softmax`,
`test_section_31_renormalize_off_range_verbatim`.

Under `renormalize=False`, vLLM's order produces softmax-tail mass (sum < 1).
Topk-first-then-softmax always gives sum = 1. Demo §3.1 pins:
`renormalize=False → sum range [0.2730, 0.6171]  mean 0.3899` — these are
the softmax-tail probabilities and they're DIFFERENT numerical values from
the topk-first path.

(**Tester clarification — see knowledge E11**: under `renormalize=True`,
the two paths ARE algebraically equivalent. The non-commutativity is a
real effect, but its visibility depends on the renormalize flag. Writer
should frame Trap G with the no-renorm arm — that's the actual Mixtral
default at certain layers.)

## ep=1 vs ep=N forward equivalence — chain-break invariant

**Method**: `test_forward_invariance_ep1_vs_ep4`,
`test_forward_invariance_across_three_ep_sizes`,
`test_forward_invariance_grouped_path_ep1_vs_ep4`,
`test_smoke.py::test_ep1_eq_ep4_forward`.

Verified `ep ∈ {1, 2, 4, 8}` produce identical forward outputs (atol=1e-6) for
both Mixtral routing (`use_grouped_topk=False`) and DeepSeek grouped routing.
This is the chain-break invariant: EP is a partition of the expert sum, not
a different math.

## Coverage by module

1. **`routing.py`** (28 tests): fused_topk shapes/dtypes/scoring; renormalize
   on/off invariants; top-K picks K largest logits; pre-condition asserts;
   grouped_topk mask correctness; e_score_correction_bias unbiased weights;
   routed_scaling_factor; sum-of-top-2 group-score path; expert_load_counts
   identity Σ=M·K; Trap G renorm-vs-no-renorm divergence; §3.1 demo pin.

2. **`expert_map.py`** (24 tests): ep=1 short-circuit; rank validation;
   linear placement at E=8 P=4 (rank 0/1/3); int32 dtype; remainder
   block-start offsets; round-robin placement at E=8 P=2 + remainder;
   coverage = each expert owned exactly once; -1 sentinel; dense local
   indices; cross-rank uniqueness for both placements; per_rank_token_load.

3. **`ep_groups.py`** (27 tests): EP-vs-TP collapse rule under EP off/on;
   `flatten_tp_across_dp_and_pcp` formula; `use_all2all_kernels` gate;
   sequence-parallel flag; mesh init at world ∈ {4, 8}; world_size mismatch
   assert; dense-model returns None; uninitialized get raises; double-init
   asserts; **Trap D**: EP/EPLB distinct objects; reset_groups idempotency;
   `__repr__` content.

4. **`all2all_baseline.py`** (20 tests): all_gatherv concatenation;
   reduce_scatterv split sizes; AgRsAll2AllManager dispatch shape; combine
   round-trip; α-β formula at α-bound and β-bound regimes; AR/A2A ratio
   = 2; world_size==1 returns 0; §3.2 NVLink + IB pinned numerics.

5. **`fused_moe_block.py`** (30 tests): SiluAndMul split + math; gate weight
   replicated across EP; ep=1 vs ep=4 forward invariance; expert_load helper;
   forward shape preservation; expert FFN output is sum of contributions;
   memory_per_rank_MiB scaling; Mixtral path forward; DeepSeek grouped-topk
   forward; integration with routing module.

6. **`eplb.py`** (32 tests): initial physical_to_logical layout (logical
   then redundant round-robin); zero-redundant case; record_step interval
   gate; rearrange happens at step `interval`; hot experts rotate into
   redundant slots; imbalance_ratio at uniform vs skewed; per_rank load
   linear vs round_robin; **Trap E** (3 negatives + 1 vllm-source grep);
   §3.5 timeline pins (steps 0, 25, 50, 99); make_skewed_routing seed
   determinism.

7. **`mixtral_vs_deepseek.py`** (16 tests): MIXTRAL_8x7B config pinning
   (E=8 K=2, no shared); DEEPSEEK_V2_LITE config pinning (E=64 K=6 grouped,
   shared); build_block routes to right path; routing_fingerprint structure;
   §3.1 demo pin (Mixtral + DeepSeek).

8. **`test_integration.py`** (17 tests): §3.2 NVLink table verbatim; §3.3
   placement-skew table verbatim across (placement × ep_size); §3.4 EP×TP
   memory table verbatim; cross-cell mem inverse proportionality; §3.5 EPLB
   layout after 100 steps; ep ∈ {1, 2, 4, 8} forward invariance for both
   routing paths; cross-module composition (routing → expert_map → all2all);
   demo-output.txt presence; Trap roster meta-test.

9. **`test_smoke.py`** (9 tests): implementer's pre-handoff sanity (kept).

## Knowledge applied

- **E01-E10** (implementer-supplied) — every claim pinned by a test.
- **E11-E15** (tester-added, appended to
  `instances/vllm/knowledge/modules/expert-parallelism.md`) — Trap-G
  renorm-flag dependency; ep_size==1 None-branch; AgRs combine pre-sum
  contract; §3.5 seed=100+step pattern; Trap-D object-identity criterion.

## Wisdom applied

- **W02** (`wisdom/testing.md`, "don't pass for the wrong reason"): every
  demo number is asserted exactly (rank loads = `[5175, 980, 1017, 1020]`
  not "approximately balanced"; `physical_to_logical[-4:] = [5, 2, 0, 4]`
  exactly). The Trap-E AST-strip-docstrings refactor is the W02 pattern in
  action — initial test passed for the wrong reason (caught a docstring
  string), fix isolates the executable-code question.

## Linter status (informational)

Source-grounding linter on the implementation already passed pre-handoff
(implementer ran it; 66 # REFERENCE comments across 8 files, 10 source
files cited in impl-notes §1.1 — well above the v6 floor of 5).

## Framing tips for writer (5 surgical guidance items)

These are the test-derived narrative-shaping notes the writer should weave
into §9.x. Each comes from a place where a paraphrase risks subtly
misleading the reader.

### Tip 1 — Trap G is a REN-ORMALIZE-FLAG-dependent effect, not always-true

A casual phrasing like "softmax → top-K is not commutative with top-K → softmax"
is FALSE under `renormalize=True` (the two paths are algebraically equivalent;
both reduce to `softmax(g_i) / Σ_topk softmax(g_j)`). Under `renormalize=False`,
they DIVERGE — that's the actual evidence in Demo §3.1's
`[0.2730, 0.6171] mean 0.3899` block. Writer should:
1. State the order vLLM uses (softmax → topk → optional renormalize).
2. Show the no-renorm sums (0.27-0.62) as the visible non-commutativity.
3. Explicitly say "with renormalize=True, the two orderings become equivalent."
Otherwise readers will write a buggy alternative implementation thinking they
preserved the math. Knowledge E11 records this gotcha for future testers.

### Tip 2 — Lead Trap A with the placement-skew table, not the abstract claim

A common framing mistake is "EP scales memory, not throughput, because of
all-to-all overhead". That's true but ABSTRACT. Demo §3.3 has a concrete punch:
**linear placement at ep=8 produces max/mean=3.251** — rank 0 does 3.25× the
work of the mean rank. Throughput is gated by the slowest rank. Round-robin
brings it to 1.196 (a 2.7× improvement). Lead with the concrete `[5175, 980,
1017, 1020]` and `[3329, 1846, 483, ...]` numbers — readers respect concrete
imbalance more than the abstract statement.

### Tip 3 — Trap E needs the "router_aux_loss_coef config carrier" footnote

A naive negative-grep on vLLM source for `aux_loss` returns false positives:
`router_aux_loss_coef` IS stored as an attribute on `phimoe.py`,
`qwen3_5_moe.py`, etc. — these are HuggingFace config carriers, never used in
any forward/backward. The clean signal is a grep restricted to
`vllm/distributed/eplb/` (the EPLB module itself), which has zero hits. Writer
should frame Trap E as: "the trained model's `router_aux_loss_coef` is a
historical record of how it was *trained*, not a runtime mechanism. EPLB is
the only inference-time balancer." This avoids a confused reader finding the
config attribute and doubting the chapter.

### Tip 4 — The §3.5 EPLB timeline is the load-bearing narrative, not the
###          theory

The §3.5 trajectory (2.523 → 2.529 → 1.203 → 1.158 → 1.229 → 1.193) is the
single most pedagogically valuable artifact in the chapter. Show it as a
table early in §9.4 — readers SEE EPLB's effect (a 2.1× imbalance reduction
in one step). Don't lead with EplbState's class structure or the rebalance
algorithm — lead with the BEFORE/AFTER imbalance numbers, then explain WHY
(layout swap from linear to round_robin via the redundant-slot allocation
`[5, 2, 0, 4]`). The numerical contrast is more memorable than the policy.

### Tip 5 — _EP and _EPLB being distinct objects is the load-bearing fact for Trap D

Saying "EPLB has its own process group" reads as bureaucratic detail. Pin
WHY: `id(_EP) != id(_EPLB)` so an EPLB rebalance broadcast and a forward-pass
dispatch all-to-all run on DIFFERENT NCCL communicators. They CAN'T block on
each other's stream — that's the deadlock-prevention claim. Test
`test_trap_D_eplb_separate_group_prevents_aliasing` asserts the heap-object
inequality. The shared rank list is necessary but NOT sufficient — both
membership AND object identity matter. Use this construction in §9.5 when
introducing the EPLB group: "same ranks, different group object → different
NCCL handle → no deadlock."

## Backpressure gate

OPEN. Writer is clear to start.
