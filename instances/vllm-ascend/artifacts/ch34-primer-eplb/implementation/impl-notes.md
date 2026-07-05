# ch34 impl-notes — EPLB planning algorithm (primer reference implementation)

`kind: primer` — this is **not** a subtract-only companion. Per the implementer
contract's primer branch, `implementation/eplb.py` is a small, paper-faithful
reference port (NumPy, no torch, host-runnable), not a line-for-line subset of
`vllm_ascend/eplb/core/policy/policy_default_eplb.py`. Gate is
`lint_paper_grounding.py --expect-primer`, not `lint_fidelity.py`.

## What it is

The *global* load-balancing policy from DeepSeek-V3 (arXiv:2412.19437 §3.4) +
deepseek-ai/EPLB README ("Global Load Balancing"), as concretely landed in
vllm_ascend's default planner `DefaultEplb.rebalance_experts` (`policy_type=1`).
The hierarchical policy (group→node packing) from the same README appendix is
**not** implemented — the landing code doesn't implement it either (see chapter
narrative), so adding it here would invent a mechanism outside both sources.

## PAPER anchors (def-level, see also inline anchors on sub-steps)

| Function (reference impl) | PAPER anchor | Landing code anchor (canonical path) |
|---|---|---|
| `fold_physical_heat_to_experts` | §3.4 "statistics collected during the online deployment" | `vllm_ascend/eplb/core/policy/policy_default_eplb.py:L29-L41` (`add_redundant`) |
| `max_heat_per_layer` | §3.4 "ensure that each GPU processes approximately the same number of tokens" | `vllm_ascend/eplb/core/policy/policy_default_eplb.py:L242-L248` (`calculate_max_heat_per_layer`) |
| `count_redundant_slots` | §3.4 "one additional redundant expert" | `vllm_ascend/eplb/core/policy/policy_default_eplb.py:L238-L240` (`get_redundant_num`) |
| `replicate_hot_experts` | §3.4 redundant experts + README §Global Load Balancing | `vllm_ascend/eplb/core/policy/policy_default_eplb.py:L43-L57` (`original_compute_balanced_pack_redundancy` Step 1) |
| `pack_replicated_experts` | README §Global Load Balancing (pack to GPUs) | `vllm_ascend/eplb/core/policy/policy_default_eplb.py:L59-L121` (same fn, Step 2-5) |
| `balanced_pack_redundancy` | composition of the two above | `vllm_ascend/eplb/core/policy/policy_default_eplb.py:L43-L121` |
| `local_exchange` | §3.4 "rearrange ... without increasing the cross-node ... overhead" | `vllm_ascend/eplb/core/policy/policy_default_eplb.py:L250-L281` (`constraint_expert_local_exchange`) |
| `rebalance_experts` | §3.4 "adjusted periodically (e.g., every 10 minutes)" | `vllm_ascend/eplb/core/policy/policy_default_eplb.py:L283-L350` |
| `compute_imbalance` | §3.4 wall-clock-facing readout (par = max/mean) | `vllm_ascend/eplb/core/eplb_worker.py:L290-L309` (`EplbWorker._compute_imbalance`) |
| `PolicyFactory` dispatch (global policy = `policy_type=1`) | README §Global vs §Hierarchical Load Balancing | `vllm_ascend/eplb/core/policy/policy_factory.py:L12-L41`, base contract `vllm_ascend/eplb/core/policy/policy_abstract.py` |

## Faithfulness cross-check (not part of the lint gate, done for assurance)

Ran the *real* `vllm_ascend.eplb.core.policy.policy_default_eplb.DefaultEplb.rebalance_experts`
directly (it only imports `numpy`/`collections`/`typing`, no torch — importable
standalone by faking the parent package modules to dodge `vllm_ascend/__init__.py`'s
`import vllm_ascend.logger`) against the chapter's worked example
(`tests/test_eplb.py::test_worked_example_2rank_8expert_matches_hand_computation`'s
inputs). Result: identical `change=1`, identical per-card total weight (87.5 /
87.5), identical redundancy split (both hot experts 3 and 4 land one copy per
card). The only difference is which *equal-weight cold* experts land on which
card — a harmless tie-break artifact of argsort ordering among ties, not a
correctness divergence.

## Worked example baked into the tests (chapter's planned numeric walkthrough)

2 NPUs (ranks), 8 routed experts, 1 redundant slot per NPU (2 total) — but the
*naive* starting placement wastes both redundant copies on already-cold experts
(1 and 5) while co-locating the two genuinely hot experts (3: weight 60, 4:
weight 55) on the same card:

```
card0 = [3, 4, 1, 2, 1(dup)]   raw heat = 60+55+10+10+0 = 135
card1 = [0, 5, 6, 7, 5(dup)]   raw heat = 10+10+10+10+0 =  40
```

`par_before = 135 / 87.5 ≈ 1.543`. `rebalance_experts` redirects the two
redundant slots to experts 3 and 4 (diluting them to 30 and 27.5 respectively),
packs to an exact 87.5 / 87.5 split, and clears the 5% deadband
(87.5 < 0.95×135) so `change=1`. `par_after = 1.0` exactly.

## Tests

`tests/test_eplb.py` — unit tests per function (fold, max-heat, redundancy
count, replication recurrence `new_avg = old_avg*(k+1)/(k+2)`, LPT packing
invariants, local-exchange keep-in-place behavior, imbalance metric) plus the
end-to-end worked example above and a change-gate deadband case.

Run: `python3 -m pytest instances/vllm-ascend/artifacts/ch34-primer-eplb/tests/test_eplb.py -v`
(host-runnable, no NPU/CANN/vLLM needed — pure NumPy).
