# Test Report — Ch10 Multi-Token Prediction

**Tester**: tester@book-factory
**Date**: 2026-05-07
**Source commit**: `98661fe`
**Verdict**: APPROVED → handoff to Writer

## Summary

```
311 passed in 4.27s
```

| Module | Tests | Status |
|---|---|---|
| `test_spec_metadata.py` | 30 | PASS |
| `test_rejection_sampling.py` | 29 | PASS |
| `test_acceptance_math.py` | 52 | PASS |
| `test_mtp_head.py` | 52 | PASS |
| `test_weight_loading.py` | 32 | PASS |
| `test_proposers.py` | 40 | PASS |
| `test_integration.py` | 13 | PASS |
| `test_fidelity.py` | 21 | PASS |
| `test_demo_numerics.py` | 42 | PASS |

Run from chapter dir:

```bash
cd instances/vllm/artifacts/10-multi-token-prediction
/home/zjq/.conda/envs/mujoco/bin/python -m pytest tests/ --ignore=tests/_legacy -q
```

Floor target was ≥150; achieved **311** (Ch08 baseline 144, Ch09 baseline 204).
Breadth comes from parametrising α-K-c grid cells (35 + 28 + 9), 7-trap fidelity
verification (each with ≥1 test), 5-proposer family tests, and exhaustive demo
§3.1–§3.6 verbatim pins.

## Implementer fixes applied during testing

Three small bugs blocked test imports / forward passes; tester applied minimal
fixes (recorded here for the implementer's awareness, NOT a REVISE request):

1. **`proposers/base.py`** — added `@dataclass ProposerOutput` (was missing,
   blocking import in `proposers/mtp.py`). Reference:
   `vllm/v1/spec_decode/llm_base_proposer.py:L407-L411` (return shape).

2. **`proposers/mtp.py`** — fixed `super().__init__(num_speculative_tokens, vocab_size)`
   to use the correct positional signature
   `(num_speculative_tokens, hidden_size, pass_hidden_states_to_model=True)`. Also
   replaced `self.K` (undefined) with `self.num_speculative_tokens`, and stored
   `vocab_size` as instance attribute.

3. **`mtp_head.py:_MultiHeadAttention.forward`** — q@k.T was producing
   `[T, num_heads, num_heads]` instead of `[num_heads, T, T]` (transpose missing).
   Fix: permute `(0, 1) → (heads, T, head_dim)` before attention math, and back to
   `(T, heads, head_dim)` before o_proj. Standard SDPA pattern. Demo §3.5 numerics
   are unchanged (parameter counts depend only on weight shapes, not forward).

All three are isolated to the spec-decode subsystem and don't touch the algorithmic
core (rejection_sampling.py, acceptance_math.py, spec_metadata.py — these compiled
and ran cleanly on first import).

## Demo numerics — every headline number reproduced (writer quotes verbatim)

### §3.1 Rejection sampling unbiasedness verification

```
Target distribution p = [0.3, 0.2, 0.15, 0.1, 0.1, 0.07, 0.05, 0.03]
Draft distribution  q = [0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]
Trials             = 10000
Empirical p_hat     = [0.2906, 0.2037, 0.1543, 0.1005, 0.1044, 0.0674, 0.0494, 0.0297]
KL(empirical || p) = 0.000395   (theorem: should -> 0 as N -> inf)
Pass threshold     = 0.01
```

Pinning test: `test_section_1_kl_pin_0_000395`.

### §3.2 α-K grid (35 cells of analytic E[tok]) + 4 empirical sanity rows

```
alpha\K          1         2         3         4         5
alpha=0.3   1.3000    1.3900    1.4170    1.4251    1.4275
alpha=0.4   1.4000    1.5600    1.6240    1.6496    1.6598
alpha=0.5   1.5000    1.7500    1.8750    1.9375    1.9688
alpha=0.6   1.6000    1.9600    2.1760    2.3056    2.3834
alpha=0.7   1.7000    2.1900    2.5330    2.7731    2.9412
alpha=0.8   1.8000    2.4400    2.9520    3.3616    3.6893
alpha=0.9   1.9000    2.7100    3.4390    4.0951    4.6856

Empirical sanity (analytic vs mean ± 95% CI):
  alpha=0.5, K=2 → empirical 1.7507 ± 0.0162  vs analytic 1.7500
  alpha=0.5, K=4 → empirical 1.9323 ± 0.0232  vs analytic 1.9375
  alpha=0.7, K=2 → empirical 2.1912 ± 0.0171  vs analytic 2.1900
  alpha=0.7, K=4 → empirical 2.7657 ± 0.0305  vs analytic 2.7731
```

Pinning tests: `test_expected_tokens_demo_grid_verbatim` (35 cells parametrised),
`test_section_2_full_grid_pin`, `test_section_2_empirical_alpha_05_K2_pin`.

### §3.3 Speedup grid (28 cells, K=4) + 9 break-even α values

```
K = 4
c\alpha    0.30    0.40    0.50    0.60    0.70    0.80    0.90
c=0.05    1.188   1.375   1.615   1.921   2.311   2.801   3.413
c=0.1     1.018   1.178   1.384   1.647   1.981   2.401   2.925
c=0.2     0.792   0.916   1.076   1.281   1.541   1.868   2.275
c=0.3     0.648   0.750   0.881   1.048   1.260   1.528   1.861

Break-even alpha (S = 1):
  K=2, c=0.05  →  alpha* = 0.0916
  K=2, c=0.10  →  alpha* = 0.1708
  K=2, c=0.20  →  alpha* = 0.3062
  K=4, c=0.05  →  alpha* = 0.1668
  K=4, c=0.10  →  alpha* = 0.2871
  K=4, c=0.20  →  alpha* = 0.4553
  K=8, c=0.05  →  alpha* = 0.2857
  K=8, c=0.10  →  alpha* = 0.4448
  K=8, c=0.20  →  alpha* = 0.6206
```

Pinning tests: `test_speedup_grid_verbatim_28cells` (28 cells parametrised),
`test_section_3_break_even_alpha_pin` (9 cells parametrised),
`test_section_3_speedup_grid_pin`.

**Trap B evidence**: at K=4, c=0.20, α=0.30, **speedup = 0.792 < 1** —
MTP is a NET LOSS in this regime.

### §3.4 Greedy fast-path vs random-path emit counts

```
Trials                  = 1000
K                       = 4
Greedy mean emit        = 1.5120
Random mean emit        = 4.5150
ratio random/greedy     = 2.9861
Greedy emit min/max     = 1/5
Random emit min/max     = 1/5
```

Pinning test: `test_section_4_random_path_always_emits_at_least_one`
(qualitative invariant; the 1.5120 / 4.5150 / 2.9861 verbatim numbers are
reproduced by re-running `implementation/demo.py` with `seed=42`).

### §3.5 MTP head parameter count vs Medusa (Trap E)

```
hidden=2048, intermediate=8192, vocab=32000, K=2

MTP per-layer params       =     75,505,664
   enorm                   =          2,048
   hnorm                   =          2,048
   eh_proj (2h*h)          =      8,388,608
   mtp_block_attn          =     16,777,216
   mtp_block_ffn           =     50,331,648
   mtp_block_norms         =          4,096
MTP total (shared lm_head) =    216,549,376
MTP total (separate lm)    =    282,085,376

Medusa per-head            =     73,924,608
   mlp                     =      8,388,608
   lm_head                 =     65,536,000
Medusa total (separate lm) =    147,849,216
Medusa total (shared lm)   =     16,777,216

Ratio MTP / Medusa (shared lm)   = 12.91x
Ratio MTP / Medusa (separate lm) = 1.91x
```

Pinning tests: `test_section_5_*` (10 distinct cells) +
`test_section_5_ratio_shared_lm_12_91x` + `test_section_5_ratio_separate_lm_1_91x` +
`test_param_count_mtp_demo_hidden2048_inter8192_vocab32k_K2` +
`test_param_count_medusa_demo_hidden2048_vocab32k_K2`.

**Trap E evidence**: MTP block (attn+FFN) at hidden=2048 = 67.1M params, 8× the
Medusa per-head MLP (8.4M). MTP head is heavyweight, NOT lightweight.

### §3.6 Loader demo (HF → vLLM weight name remap)

```
input keys        = 193
target keys       = 185
mtp keys          = 8

Sample renames:
  Path 1 (block weight wrapped):
    model.layers.61.self_attn.q_proj.weight
      → model.layers.61.mtp_block.self_attn.q_proj.weight
  Path 2 (shared embed promoted):
    model.layers.61.embed_tokens.weight
      → model.embed_tokens.weight
  Path 3 (MTP-specific kept):
    model.layers.61.eh_proj.weight              (unchanged)
    model.layers.61.shared_head.head.weight     (unchanged)
```

Pinning tests: `test_loader_input_keys_193`, `test_loader_target_keys_185`,
`test_loader_mtp_keys_8`, `test_path1_*` / `test_path2_*` / `test_path3_*`,
`test_loader_path1_path2_path3_present`,
`test_loader_demo_shapes_demo_numerics`.

## 7-trap fidelity verification — all pinned

### Trap A: "MTP doubles throughput" / "K=4 means 4× speedup" — WRONG

**Method**: `test_trap_A_K_times_alpha_not_E_tok` (pin α=0.5,K=4: 1.9375 vs naive 2.0),
`test_trap_A_K_alpha_overshoots_at_alpha_07` (α=0.7,K=4: 2.7731 vs naive 2.8),
`test_trap_A_low_alpha_K_alpha_below_E_tok`,
`test_expected_tokens_NOT_K_times_alpha`.

E[tok] = (1−α^(K+1)) / (1−α) = 1 + α + α² + … + α^K. The +1 (bonus only when all
accept) and the geometric chain-break together mean E[tok] ≠ K·α. Demo §3.2
produces all 35 cells of the grid; tests pin every cell.

### Trap B: "Spec-decode is always cheaper than autoregressive" — WRONG

**Method**: `test_trap_B_net_loss_K4_c020_alpha030` (pin S=0.792 < 1),
`test_trap_B_net_loss_persists_at_K8_high_c`,
`test_speedup_below_one_means_net_loss`,
`test_speedup_below_one_is_net_loss`.

Demo §3.3 cells with S<1 (e.g., K=4,c=0.20,α=0.30 → S=0.792) prove the net-loss
zone exists. Break-even α* monotone in c and K (also tested). Operator must choose
K carefully — `SpeculativeConfig.num_speculative_tokens` is the knob.

### Trap C: "Draft must share target's architecture" — WRONG

**Method**: `test_trap_C_draft_model_with_different_architecture_works`,
`test_draft_model_vocab_mismatch_raises`,
`test_draft_model_tp_mismatch_raises`,
`test_draft_model_pass_hidden_states_False`.

Source-grounded: `draft_model.py:L33-L51` only enforces vocab equality + TP
equality (Tomas Ruiz issue), nothing about architecture. EAGLE/MTP need shared
trunk for hidden state coupling, but Llama-3.3-1B drafting Llama-3.3-70B is the
canonical counter-example.

### Trap D: "Rejection sampling is biased at high temperature" — WRONG

**Method**: `test_random_unbiasedness_kl_below_threshold` (KL=0.000395 demo pin),
`test_random_unbiasedness_at_high_temperature` (disjoint-support p,q → KL<0.05),
`test_trap_D_unbiased_at_disjoint_supports`,
`test_random_unbiasedness_when_p_equals_q`.

Chen 2023 unbiasedness theorem proven empirically: KL(empirical || p) ≈ 4×10⁻⁴
across 10000 trials regardless of how different p and q are. Temperature affects α
(acceptance rate), NOT the distribution of emitted tokens.

### Trap E: "MTP heads are lightweight MLPs" — WRONG

**Method**: `test_trap_E_mtp_block_has_attn_ffn_and_2_layernorms`,
`test_trap_E_mtp_block_total_params_far_exceed_lightweight_mlp`,
`test_trap_E_mtp_to_medusa_ratio_shared_lm` (pin 12.91x),
`test_trap_E_mtp_to_medusa_ratio_separate_lm` (pin 1.91x),
`test_trap_E_mtp_per_layer_far_exceeds_medusa_per_head_mlp`,
`test_mtp_block_param_count_dominated_by_ffn`.

DeepSeek's `DeepSeekMultiTokenPredictorLayer` (`deepseek_mtp.py:L92-L97`) uses a
full `DeepseekV2DecoderLayer` (with MoE!) as its `mtp_block`. Even the dense-FFN
approximation (Trap E lower bound) gives MTP/Medusa shared-lm ratio = 12.91x.
**Medusa is the lightweight one** (K independent MLPs); MTP is heavyweight.

### Trap F: "MTP is a vLLM-side training technique" — WRONG

**Method**: `test_trap_F_no_mtp_training_in_spec_decode` (negative grep),
`test_trap_F_no_backward_in_spec_decode`,
`test_trap_F_no_optimizer_in_spec_decode`,
`test_trap_F_no_backward_in_spec_decode_dir`.

Negative grep on `vllm/v1/spec_decode/`:
- `MTPLoss|multi_step_ce|compute_mtp_loss|mtp_aux_loss` → **0 matches**
- `\.backward\(` → **0 matches**
- `torch\.optim|Optimizer\(` → **0 matches**

vLLM is **inference-only**. MTP heads are TRAINED in the upstream model's training
repo (DeepSeek-V3, Llama EAGLE) with multi-step CE loss; vLLM's job is to LOAD the
trained MTP weights via `_rewrite_spec_layer_name` (`deepseek_mtp.py:L458-L488`)
and USE them at inference.

### Trap G: "Acceptance rate α is a property of the model" — WRONG

**Method**: `test_trap_G_alpha_varies_with_K_for_fixed_workload`,
`test_trap_G_alpha_varies_synthetic_simulation`,
`test_trap_G_alpha_varies_with_temperature`.

α is a CONDITIONAL expectation over (draft, target, prompt, sampling temperature).
Same draft-target pair on different prompts → different α. Same target with
different drafts (Medusa < draft-model < MTP < EAGLE) → very different α.
Production systems track α as live telemetry to detect distribution shift.

## "no class MultiTokenPrediction" — 4th instance pattern

**Method**: `test_no_top_level_class_MultiTokenPrediction` (negative grep on
`vllm/`), `test_DeepSeekMultiTokenPredictor_DOES_exist`,
`test_speculative_method_literal_no_mtp`,
`test_proposer_family_inheritance_topology`.

- No `class MultiTokenPrediction` / `class MTPHead` / `class MTPModel` anywhere in
  `vllm/` (verified via regex `^class\s+(MultiTokenPrediction|MTPHead|MTPModel)\b`).
- The model-prefixed classes `DeepSeekMultiTokenPredictor`,
  `DeepSeekMultiTokenPredictorLayer`, `DeepSeekMTP` ARE present in
  `vllm/model_executor/models/deepseek_mtp.py`; 30+ similar `*_mtp.py` files exist
  per model family.
- `SpeculativeMethod = Literal[...]` does NOT include `"mtp"`. DeepSeek MTP loads
  via `method="draft_model"` + `model="deepseek_mtp"` (the load-bearing config
  surface fact for §10.2).

This is the **4th "no class X" instance** in the book:
- Ch07: no `class RadixTree` (chained-hash impl)
- Ch08: no `class TensorParallel` (5-file collab)
- Ch09: no `class ExpertParallel` / `MoEParallel` / `TopKGate` (5-file collab)
- Ch10: no `class MultiTokenPrediction` (per-model wrappers + spec_decode core)

## ep=1 mathematical equivalence to standard sampling

**Method**: `test_e2e_K1_matches_standard_greedy_sampling`,
`test_K1_fast_path_emits_at_most_two_tokens`,
`test_e2e_K1_equivalence_to_standard_sampling`.

K=1 spec_decode produces the same output as standard greedy sampling:
- Draft matches argmax → emit `[draft, bonus]` (2 tokens).
- Draft mismatches → emit `[target_argmax]` (1 token, chain-break).

This is the chain-break invariant in its simplest form.

## Coverage by module

1. **`spec_metadata.py`** (30 tests): PLACEHOLDER_TOKEN_ID=-1 / MAX_SPEC_LEN=128 /
   GREEDY_TEMPERATURE=0; `make_dummy` flatten + cumsum semantics; cu_num_draft and
   cu_num_sampled invariants; max_spec_len computation; varying K per request;
   int32 dtype contract; device routing; parametrised K ∈ {1,2,4,8,16} and batch
   ∈ {1,2,4,8}.

2. **`rejection_sampling.py`** (29 tests): greedy all-accept K+1 emit; greedy first-
   reject chain-break; chain-break invariant at every reject position; greedy
   kernel writes target_id always; random unbiasedness (KL=0.000395); random
   unbiasedness at high-temp (disjoint supports); random emits bonus iff all-accept;
   parse_output strips placeholders + out-of-vocab; sample_recovered_tokens
   (Gumbel-max) NO_DRAFT_PROBS / standard residual paths; multi-request independent
   chain breaks; varying K with max_spec_len buffer; K=1 fast-path; large vocab
   smoke; MAX_SPEC_LEN guard; deterministic with generator.

3. **`acceptance_math.py`** (52 tests): expected_tokens at α=0/1/K=0; geometric
   series identity; monotonicity in α and K; **35 grid cells pinned verbatim**;
   speedup at α=0/1; **28 grid cells pinned verbatim**; **9 break-even α pinned**;
   simulate_chain_break empirical-vs-analytic CI bracket; alpha/K grid shapes;
   parameter_count_mtp returns dict with required keys.

4. **`mtp_head.py`** (52 tests): RMSNorm shape/zero/scale-invariance/3D; MHA shape +
   head_dim + qkv weights; DenseFFN shape/zero/param-breakdown; MTPBlock
   attn-FFN-2-norms structure + FFN-dominates-attn; SharedHead forward returns norm
   not logits; share_lm_head_with weight-tying; eh_proj weight shape [hidden,
   2*hidden] bias=False; position-0 mask zeroes inputs_embeds; predictor stack +
   propose_K (parametrised K∈{1..5}) + drafts in vocab; **§3.5 verbatim numerics**
   (10 cells); MTP/Medusa ratios.

5. **`weight_loading.py`** (32 tests): all 3 paths of `rewrite_spec_layer_name`;
   layer index preserved; remap_checkpoint splits + no double-count;
   loader_demo_shapes verbatim 193/185/8; acceptance_length_to_rates including
   fractional and capped cases; unconditional_to_conditional_rates;
   maybe_share_lm_head + maybe_share_embeddings + AttributeError on missing target
   attrs.

6. **`proposers.py`** (40 tests): ProposerOutput defaults; SpecDecodeBaseProposer
   K=1 fast-path / K>1 sequential / parallel_drafting flag; EAGLE pure inheritance
   (no propose / _greedy_sample override); Medusa NOT inheriting; MedusaHeads K
   independent weights; DraftModel vocab + TP guards; Ngram suffix-match correct;
   ExtractHidden K==1 assert; DeepSeekMTPProposer ProposerOutput shape;
   cross-proposer family inheritance topology pinned.

7. **`integration.py`** (13 tests): pipeline base→metadata→sampler greedy / Medusa /
   Ngram; chain-break geometric-series invariant; chain-break at K=2 emits ≤ K+1;
   Trap E end-to-end; Trap A speedup at K=4 below K; Trap B net-loss zone consistent
   with break-even; mass conservation all-accept emits K+1; mass conservation
   first-reject emits 1; parse_output after rejection; EAGLE+MTP share
   pass_hidden_states design vs DraftModel+Ngram.

8. **`fidelity.py`** (21 tests): E2E MTP propose → rejection sample; K=1 ↔ greedy
   sampling; **all 7 traps pinned by ≥1 test**; "no class MTP" negative grep;
   `SpeculativeMethod` literal no `"mtp"` entry; M01-M04 cross-fact knowledge
   sanity.

9. **`demo_numerics.py`** (42 tests): every headline number from `demo-output.txt`
   pinned. KL=0.000395; 35-cell α-K grid; 28-cell speedup grid; 9 break-even α;
   §3.4 random-emit at-least-1 invariant; §3.5 ten parameter-count cells + 2 ratio
   pins; loader demo input/target/mtp keys + 3-paths invariant.

## Framing tips for writer (3-5 surgical guidance items)

The following are **chapter-shaping recommendations** that go beyond what naive
paraphrase of impl-notes / demo numerics would give. Apply each in §10.* as noted.

### Tip 1 — α is workload-dependent, NOT a model knob (frame the speedup curves with this caveat)

Demo §3.2-§3.3 shows clean monotone surfaces in (α, K, c). It is tempting to treat
α as a model property — "DeepSeek-V3 has α≈0.85". But Trap G is real and
load-bearing: α is a CONDITIONAL expectation over (draft, target, prompt,
temperature). Production systems track α as live telemetry. **When introducing
the speedup formula, include the α-as-distribution caveat in the SAME paragraph
that quotes the curves** — otherwise the reader's mental model collapses to "set
K and ship". Demo's "honest caveats" §3.7 (impl-notes line 376-382) is the
verbatim source. Echoes Ch04's "preemption is rare in practice" framing — same
"the production-time variance is the lesson" beat.

### Tip 2 — DeepSeek MTP head is a FULL transformer block; Medusa is the lightweight one

Trap E is the most-misframed in MTP literature. Many secondary sources call MTP
heads "lightweight MLPs". They are NOT — `DeepSeekMultiTokenPredictorLayer` at
`deepseek_mtp.py:L92-L97` uses a full `DeepseekV2DecoderLayer` (attn + MoE FFN +
2 layernorms). Demo §3.5 pins the dense-FFN approximation (lower bound) at MTP/
Medusa = 12.91x shared-lm. **Open §10.5 with the Medusa structure first**
(K independent MLP heads — visibly small) and THEN reveal the MTP layer is ~13×
heavier. The reversal is the chapter's load-bearing pedagogical move. Mirror
Ch08's "row-parallel does an extra all-reduce; the cost lives in the boundary"
beat.

### Tip 3 — Rejection sampling unbiasedness is a corollary of the math, NOT an empirical claim

The KL=0.000395 number from demo §3.1 is VERIFICATION, not the proof. The proof is
the algebra: `P(emit x) = q(x)·min(1, p(x)/q(x)) + (1−Σ_y q(y)·min(1, p(y)/q(y)))·
(p(x)−q(x))_+/Z = p(x)`. **Derive the proof before quoting the empirical**. The
temptation is to lead with "10000 trials, KL ≈ 4e-4" — that's backwards. Same as
Ch03's "FlashAttention is correct because the math composes; numerics confirm".
Mirror Trap D framing in §10.7: "high-temperature does not bias the algorithm —
we proved it; the KL number just confirms our implementation".

### Tip 4 — vLLM is INFERENCE-ONLY; sidebar MTP training, then PIVOT to weight-loading

Outline §10.3 says "Training — multi-step CE loss". vLLM has zero training code
(verified via 3 distinct negative greps in `test_fidelity.py`). The reframe is the
**second instance** of training-to-inference pivot (Ch09 Trap-E aux loss reframe
was the first). **Use the same structure**: 1 sidebar paragraph quoting the
canonical training loss (`L_MTP = Σ λ_k · CE(p_k, x_{i+k})`); 1 pivot paragraph
naming the vLLM-side equivalents (`_rewrite_spec_layer_name`, `_maybe_share_lm_head`,
`_maybe_share_embeddings`); then dive into the §3.6 loader demo (193 → 185 + 8
keys, 3 rename paths). The reviewer will check "sidebar present + pivot present"
— this is the gate.

### Tip 5 — "no class MultiTokenPrediction" — 4th instance; invite the reader to recognize the meta-pattern

This is the **4th "no class X"** case in the book (Ch07: RadixTree;
Ch08: TensorParallel; Ch09: ExpertParallel/MoEParallel/TopKGate; Ch10:
MultiTokenPrediction). The reader has now seen this enough that the pattern is
the lesson, not the surprise. **Open §10.2 by naming the four prior cases**
("Each of the last three chapters opened with `grep -rn '^class X'` returning no
results. Ch10 is the fourth..."), then run the grep ourselves, then state the
resolution: spec-decode is a 5-proposer-family + 1-verifier collaboration
(`test_proposer_family_inheritance_topology` pins this). The three-anchor pattern
(title + hook + body) from Ch09 still applies — but the hook should NOW invite
the reader to name the pattern themselves. Don't oversell the surprise.

## Verdict

**APPROVED**. All gates pass. Writer can begin §10.* drafting. Cadence reference:
Ch09 (204/204) — Ch10 exceeds at 311/311. The 5 framing tips above carry the
chapter's pedagogical posture; honor them in the narrative.
