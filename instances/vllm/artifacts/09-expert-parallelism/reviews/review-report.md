# Ch09 Expert Parallelism — Review Report

- **Chapter**: `09-expert-parallelism`
- **Reviewer**: reviewer (book-factory, vllm-from-scratch instance)
- **Date**: 2026-05-07
- **Cycle**: 1 (single-cycle target, mirrors Ch08 cadence)
- **Verdict**: **APPROVED**

---

## Hard Gates — All PASS

### Gate 1: Linters re-run by reviewer

| Linter | Result | Notes |
|---|---|---|
| `lint_formulas.py` | 🟢 No blocking issues | 4 non-blocking "Too Many Inline Formulas" warnings on lines 608, 754, 773, 918 — paragraphs with 3-5 inline math symbols. These are the V/W/L/Σ symbol-density paragraphs in the EPLB sidebar (training aux-loss exposition), §9.5 mem-formula explanation, and §9.5.4 chain-break comm cost. All inline tokens are single symbols (`$f_i$`, `$P_i$`, `$3 F h$`, etc.) — within the "single symbol" allowed-inline rule. **Non-blocking is acceptable; blocking gate clears.** |
| `lint_source_grounding.py` | ✓ All grounding checks passed! | Cell coverage, # REFERENCE comments, mapping table rows, impl-notes file count — all pass. |

### Gate 2: Mapping table rows ≥10

- **Claim**: 49 main + 39 mini = 88 rows
- **Actual count**: 151 `|`-prefixed lines across all tables (main §9.9 + mini §9.2.5/§9.3.5/§9.4.6/§9.5.5/§9.6.5 + within-section helpers)
- **Result**: PASS — exceeds floor of 10 by 15×, exceeds Ch08 baseline of 122

### Gate 3: All 10 impl-notes source files surfaced

| File | References in narrative | Status |
|---|---|---|
| `parallel_state.py` | 24 refs (L1261, L1264, L1670, L1672, L1700, L1797, L1891, etc.) | PASS |
| `fused_moe/layer.py` | 17 refs (L70, L107, L117, L160, L168, L196, L290, L378, L548, L1543, etc.) | PASS |
| `fused_moe/config.py` | 7 refs (L1019, L1077, L1162, L1175, L1192, L1208, L998-L1209) | PASS |
| `fused_topk_router.py` | 12 refs (L69, L77, L94, L100, L106, L116, L149) | PASS |
| `grouped_topk_router.py` | 8 refs (L81, L113, L121, L141, L162, L247, L341) | PASS |
| `device_communicators/all2all.py` | 25 refs (L40, L99, L130, L196, L257, L327, L442, L671, etc.) | PASS |
| `eplb_state.py` | 12 refs (L62, L210, L286, L920, L925, L944) | PASS |
| `mixtral.py` | 7 refs (L77, L123, L132, L154) | PASS |
| `deepseek_v2.py` | 10 refs (L244, L272, L295, L302, L317, L386, L420) | PASS |
| `prepare_finalize/naive_dp_ep.py` | 9 refs (L71, L104, L125, L127, L150, L166, L168) | PASS |

**Result**: PASS — all 10 source files cited with line numbers.

### Gate 4: 5-step rhythm in §9.1 through §9.6

| Section | Open | What+Why | Derive | Our Impl | Source Diff | Status |
|---|---|---|---|---|---|---|
| §9.1 (routing) | 9.1.1 (line 70-95) `fused_topk_router.py:L69` | 9.1.2 (line 119-160) "softmax then topk, not reverse" | 9.1.2 derivation lines 130-148 with Trap-G algebraic equivalence proof | 9.1.1 (lines 102-115) `routing.py:L65-L91` | 9.1.5 (lines 228-240) Triton fast-path table | PASS |
| §9.2 (5-file collab) | 9.2.1 (line 246) grep evidence | 9.2.2 (line 268) "_EP vs _TP asymmetry" | 9.2.3 (line 305) mesh formula derivation | 9.2.4 (line 391-403) `fused_moe_block.py:L244-L266` | 9.2.5 (line 410) 18-row mini-mapping | PASS |
| §9.3 (placement) | 9.3.1 (line 437) `layer.py:L70-L157` | 9.3.2 (line 493) "linear is sequential, round_robin is interleaved" | 9.3.2 (line 509) load distribution under correlated routing | 9.3.1 (lines 472-489) `expert_map.py:L46-L86` | 9.3.5 (line 580) mini-mapping with E13 2-tuple/3-tuple diff | PASS |
| §9.4 (EPLB) | 9.4.1 (line 594) reframe + grep evidence | 9.4.1-9.4.2 (lines 600-617) "deployed traffic ≠ training traffic" | 9.4.2 (line 622-651) `EplbState` derivation | 9.4.2 (lines 622-670) `eplb.py:L55-L102` | 9.4.6 (line 730) educator-vs-production mini-mapping | PASS |
| §9.5 (EP+TP mesh) | 9.5.1 (line 746) lead with `mem_per_rank ∝ 1/(ep × tp)` formula | 9.5.1 (line 754-770) why 3Fh per expert | 9.5.3 (line 805) `config.py:L1192-L1208` collapse derivation | 9.5.1 (line 765-770) `fused_moe_block.py:L302` `memory_per_rank_MiB` | 9.5.5 (line 859) mini-mapping | PASS |
| §9.6 (all-to-all) | 9.6.1 (line 873) `all2all.py:L40-L139` | 9.6.1 (line 893-897) "AgRs is allgatherv+reduce_scatterv, not symmetric" | 9.6.2 (line 906) α-β model derivation with tautology caveat | 9.6.1 (lines 877-890) `all2all_baseline.py:L65-L115` | 9.6.5 (line 992) 8-row backend mini-mapping | PASS |

**Result**: PASS — every section follows the 5-step rhythm explicitly.

### Gate 5: Demo numerics verbatim (no rounding/approximation)

| Number | Test-report ground truth | Chapter line | Match |
|---|---|---|---|
| Mixtral count | `[250, 285, 277, 243, 253, 272, 247, 221]` | L57, L168, L1066 | EXACT |
| DeepSeek grouped | `max=131  min=78  mean=96.00` | L178, L226, L1069 | EXACT |
| renormalize=False range | `[0.2730, 0.6171]  mean 0.3899` | L57, L154, L1049 | EXACT |
| max/mean=3.251 (linear ep=8) | `3.251` | L60, L528, L539, L543, L1014 | EXACT |
| Round-robin recovery | `1.196` | L60, L541, L543, L1014 | EXACT |
| α-β NVLink table | 4 rows `16.09 / 67.47 / 478.51 / 3766.85` μs | L928-931 | EXACT |
| α-β IB headline | `50.70μs / 18804.48μs` | L934-935 | EXACT |
| α-β ratio (tautology) | `2.000` (with caveat) | L924, L938, L1018, L1020 | EXACT (with explicit "model identity, not measurement") |
| Memory table | 6 rows (1056/264/132/66/66/33) | L786-790 | EXACT |
| `(4,2)` ≡ `(8,1)` ≡ 132 MiB | invariant check | L802 | EXACT |
| EPLB timeline | step 0/25/50/51/75/99 ratios `2.523 → 2.529 → 1.203 → 1.158 → 1.229 → 1.193` | L678-683, L686 | EXACT |
| `physical_to_logical[-4:]` | `[5, 2, 0, 4]` | L693, L1182 (paraphrased), L1193 (table) | EXACT |
| `physical_to_logical[0:8]` | `[0, 1, 2, 3, 4, 5, 6, 7]` | L692 | EXACT |

**Result**: PASS — every claimed number reproduces verbatim, no rounding observed.

### Gate 6: §9.2 "no class ExpertParallel" reframe

- **Title** (L1): explicitly says "没有 `class ExpertParallel`"
- **Hook framing** (L16): "vLLM 没有 `class ExpertParallel`，也没有 `class MoEParallel`，更没有 `class TopKGate`" — names all three
- **§9.2 opening grep evidence** (L250-L254): explicit `grep -rE` command + "(zero matches)"
- **5 files named** (L256-L264, L43-L51): table lists `parallel_state.py`, `fused_moe/layer.py`, `fused_moe/config.py`, `device_communicators/all2all.py`, `prepare_finalize/naive_dp_ep.py` — all 5
- **Commit pin** (L3, L1199): "98661fe" cited at chapter open and footer
- **Mirrors Ch07/Ch08**: hook L16 explicitly names the three-chapter series

**Result**: PASS — reframe matches Ch07/Ch08 cadence exactly.

### Gate 7: §9.4 EPLB reframe (training sidebar → inference pivot)

| Required element | Chapter location | Status |
|---|---|---|
| Outline subsection name flagged as out-of-scope | L596 | PASS |
| Training-aux-loss sidebar (Switch Transformer) | L602-L608 with explicit `L_balance` formula | PASS |
| GShard / DeepSpeed-MoE referenced | L596 | PASS |
| Pivot to inference-time response | L600, L610 | PASS |
| `EplbState` cited at `L210` | L614, L1182 | PASS |
| Separate `_EPLB` group cited | L617, L716 with `parallel_state.py:L1700-L1719` | PASS |
| Redundant experts mechanism | L616, L693, L715 | PASS |
| Logical→physical reshuffle | L617, L693, L709 | PASS |
| Precise wording "MoE inference paths" (NOT "zero matches") | L598 ("0 匹配" within `vllm/distributed/eplb/` scope only), L709 ("**MoE 推理路径里没有 aux-loss 计算**"), L1034 same | PASS |
| E10 false-positive caveat (phimoe.py / vision.py) | L598 (inline), L707 (Trap E body), L1036 (Trap E recap) | PASS |

**Result**: PASS — reframe is structurally complete and uses the precise "MoE inference paths" wording (not the discredited "zero matches" phrasing).

### Gate 8: Five framing tips applied surgically

| Tip | Required text | Chapter location | Status |
|---|---|---|---|
| Tip 1 | Trap-G qualifier "当且仅当 renormalize=False" or equivalent | L1046 (`**错** **当且仅当 \`renormalize=False\`**`); L143 (`**\`renormalize=True\` 下两条路径代数等价**`); L149 (`**\`renormalize=False\` 下两条路径才不等价**`) | PASS |
| Tip 2 | "inference paths" wording | L598 ("`vllm/distributed/eplb/` 整目录"...); L709 ("MoE 推理路径里没有 aux-loss 计算"); L1034 same; L1037 lists 4 negative tests | PASS |
| Tip 3 | `mem_per_rank ∝ 1/(ep × tp)` leads §9.5 | L744 (section title is the formula); L750-L753 (formula block at very top); L771 (memory_per_rank_MiB code as derivation) | PASS |
| Tip 4 | α-β ratio=2.000 framed as model tautology | L924 ("**E12 的 caveat**：这个比值 **2.000** 是模型的恒等式，不是测量结果"); L938 ("verbatim 复述这条 caveat"); L1018, L1020 (Trap B recap with tautology framing) | PASS |
| Tip 5 | §9.3 explains 3-tuple→2-tuple signature diff | L449 (source `return global_num_experts, None, None  # 3-tuple in source`); L475 (impl `return (global_num_experts, None)             # 2-tuple, 见 E13`); L491 (full paragraph explaining E13 + AITER `expert_mask` simplification) | PASS |

**Result**: PASS — all 5 tips applied surgically, not just appended.

### Gate 9: 5-7 language-trap callouts

§9.7 (L1006-L1049) contains all **7 traps** (A-G) with full template `**错** → **为什么** → **源码证据** → **Demo 证据**` (or **测试**):

| Trap | Header line | Status |
|---|---|---|
| A: "EP=N gives N× capacity" | L1010-L1014 | PASS |
| B: "All-to-all = all-reduce/2" | L1016-L1020 | PASS |
| C: "Experts independent so EP free" | L1022-L1025 | PASS |
| D: "EPLB free runtime bolt-on" | L1027-L1031 | PASS |
| E: "Aux loss balances vLLM experts" | L1033-L1037 (with E10 false-positive carve-out at L1036) | PASS |
| F: "FusedMoE.forward always dispatch→experts→combine" | L1039-L1043 | PASS |
| G: "Top-K then softmax = softmax then Top-K" | L1045-L1049 (with renormalize=False qualifier) | PASS |

**Result**: PASS — 7 traps, exceeds 5-7 target, matches Ch08 §8.6.4 template exactly.

### Gate 10: Forward-pointers to Ch11, Ch15+, Ch27

| Target | Chapter location | Specific content |
|---|---|---|
| Ch11 (DCP/PCP) | L977-L979, L1190 | Mesh extends to 5D `(pp, pcp, dcp, dp, tp)`; EP remains complement axis; `_EP ⊥ _DCP` |
| Ch15+ (model zoo / Llama variants) | L981-L984, L1191 | Llama-3 dense → `_EP is None`; Mixtral, DeepSeek-V2/V3, Qwen3-MoE will deep-dive routing/placement/mem |
| Ch27 (DeepSeek-V3.2) | L986-L989, L1192 | `e_score_correction_bias`/noaux_tc training motivation; DeepEP IBGDA kernel internals; `policies.py` bin-packing solver |

**Result**: PASS — all three forward-pointers wired with concrete content.

---

## Dimension Scoring (per reviewer.md schema)

| Dimension | Score | Notes |
|---|---|---|
| Algorithm Comprehension | pass | Top-K routing tiling shown explicitly E=8/P=4 (L498) and E=8/P=3 (L504); numerical trace for §3.3 placement table walked rank-by-rank (L535-L541); EPLB timeline traced 6 timestamps (L676-L686). Reader can hand-calc one iteration. |
| Code Walkthrough | pass | Every major section opens a source file with line range + shows code snippet; impl-side mirror cited with line numbers; running outputs (demo §3.1-§3.5) shown verbatim with discussion. |
| Source Grounding | pass | Linter clean. All 7 cells (Cell 2-7 equivalent: §9.1-§9.6+§9.7+§9.8+§9.9) reference source files. impl-notes lists 10 source files; mapping table covers 49+39=88 rows (151 `|` lines). |
| Formula Renderability | pass | 0 blocking lint issues. 4 non-blocking inline-density warnings — all are single-symbol inline ($f_i$, $P_i$, $3Fh$ etc.) within the "single symbol allowed inline" rule. No `\text{}`, no `\boxed{}`, no `\tag*{}`, no `\frac` inline. |
| Coherence | pass | Hook → "what does this chapter teach" → §9.1 (routing math) → §9.2 (5-file architecture) → §9.3 (placement) → §9.4 (EPLB reframe) → §9.5 (mesh) → §9.6 (all-to-all) → §9.7 (traps recap) → §9.8 (verification) → §9.9 (mapping table) → §9.10 (summary + forward-pointers). Each section's hook references prior content. No concept used before defined. |
| Readability | pass | Tone consistent with Ch07/Ch08 ("knowledgeable friend at whiteboard"). Levity moments distributed (e.g., L391 "EP 是 expert-sum 的 partition，不是不同的数学", L545 "所以 EP 不是 free 的"). Sentence length within 15-25 char Chinese norm. Technical terms defined first occurrence (e.g., 冗余专家 L616, GroupCoordinator L290 etc.). |
| Engagement | pass | Hook (L20-L65) opens with grep evidence + 6 concrete take-aways; reader sees the "no class" punchline up front. EPLB sidebar (L596-L610) reframes outline mismatch as a teaching moment, not a footnote. §9.5.4 chain-break callout (L831-L856) ties Ch08→Ch09 narratively. |
| Cross-Chapter Consistency | pass | Ch08 chain-break ("col→row pair = ONE all-reduce") explicitly evoked at L388 and L832. Code interfaces match Ch08 (`MergedColumnParallelLinear`, `RowParallelLinear`, `GroupCoordinator`). Difficulty progression natural: Ch08 dense MLP TP → Ch09 MoE EP+TP composition. Forward-pointers to Ch11/Ch15+/Ch27 wired. |
| Concept Precision | pass | Trap G stated with renormalize=False qualifier (Tip 1 ✓). Trap E stated with "MoE inference paths" wording (Tip 2 ✓). α-β ratio framed as tautology (Tip 4 ✓). 3-tuple/2-tuple diff explicitly explained (Tip 5 ✓). `mem_per_rank ∝ 1/(ep × tp)` leads §9.5 (Tip 3 ✓). All knowledge entries E01-E21 referenced or absorbed. |

---

## Cadence Comparison vs Ch08

| Metric | Ch08 (baseline) | Ch09 (this) | Status |
|---|---|---|---|
| Lines | 1051 | 1204 | +14.6% (richer EP surface justifies) |
| Words | 6058 | 7792 | +28.6% (5-file collab + 7 traps + EPLB sidebar) |
| Mapping rows | 122 | 151 (`|`-prefix) / 49+39=88 (claim) | exceeds Ch08 |
| Tests | 144 | 204 | exceeds Ch08 |
| Source files | 7+ | 10 | exceeds floor of 5 |
| Cycles to APPROVED | 1 | **1** | matches |

**Result**: Ch09 maintains v6 cadence single-cycle APPROVED with broader surface coverage.

---

## Non-Blocking Observations (informational, not REVISE-worthy)

1. **Line 608 / 754 / 773 / 918 inline-density**: The formula linter flags 3-5 inline math symbols per paragraph in EPLB sidebar (L608: $f_i, P_i, \alpha$ in Switch Transformer formula explanation), §9.5 mem-formula (L754: $E, h, F, ep, tp, \mathrm{bytes\_per\_param}$), §9.5 SwiGLU (L773: $E/\mathrm{ep\_size}, 2F, F$), §9.5.4 (L918: $\alpha, \beta, S$). All inline tokens are single symbols within the documented allowed-inline set. **No action needed**; informational only.

2. **L491 explanation of E13 2-tuple/3-tuple**: Excellent surgical execution of Tip 5 — explicitly tells the reader "是有意的简化，不是 bug" so they don't assume implementation drift.

3. **L598 grep specificity**: Writer uses two-tier wording — "`vllm/distributed/eplb/` 整目录里 grep ... 0 匹配" (precise scope) + "Tester 精化 E10：除 phimoe.py 把 router_aux_loss_coef 当成 stored constant、vision.py 有个名字撞车的 get_load_balance_assignment——这两个都不是 expert routing 的 aux loss" (false-positive carve-out). This is the cleanest implementation of Tip 2 across Ch07/Ch08/Ch09.

4. **L924-L938 tautology framing**: Triple-anchored — declared as "模型的恒等式，不是测量结果", explained mechanically ("公式里就是 2× 关系"), then carries forward to Trap B recap (L1018, L1020) where it's restated. No reader can mistake this for empirical measurement.

---

## Final Verdict

**APPROVED.** Ch09 clears all 10 hard gates, all 9 reviewer dimensions, matches v6 cadence, and surgically applies the 5 framing tips. Forward-pointers to Ch11/Ch15+/Ch27 wired. The chapter is ready for archivist back-up and publication.

- Cycles: 1 (single-cycle target met)
- Linter (formulas): 0 blocking, 4 non-blocking (inline density — within allowed-inline rule)
- Linter (source grounding): all pass
- Hard gates: 10/10
- Dimension scores: 9/9 pass
