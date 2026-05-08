# Ch10 Multi-Token Prediction — Review Report

- **Chapter**: `10-multi-token-prediction`
- **Reviewer**: reviewer (book-factory, vllm-from-scratch instance)
- **Date**: 2026-05-07
- **Cycle**: 1 (single-cycle target, mirrors Ch08/Ch09 cadence)
- **Verdict**: **APPROVED**

---

## Hard Gates — All PASS

### Gate 1: Linters re-run by reviewer

| Linter | Result | Notes |
|---|---|---|
| `lint_formulas.py` | 🟢 No blocking issues | 11 non-blocking "Too Many Inline Formulas" warnings on lines 162, 235, 297, 570-573, 575, 1000, 1048-1059, 1118-1121, 1124-1127, 1130-1133, 1148-1151. Inspected each: every inline token is a single symbol (`$\alpha$`, `$K$`, `$E[\mathrm{tok}]$`, `$h_t$`, `$x_{t+1..t+k}$`, `$p_k$`, `$\lambda_k$`, `$L \in [1, K+1]$`, `$\lfloor L-1 \rfloor$`, `$S < 1$`, `$cK$`) or a simple expression — all within E24 single-symbol-density allowance. **Non-blocking is acceptable; blocking gate clears.** |
| `lint_source_grounding.py` | ✓ All grounding checks passed! | Cell coverage, `# REFERENCE:` comments, mapping table rows, impl-notes file count — all pass. |

### Gate 2: Mapping table rows ≥10

- **Claim**: 80 main + 51 mini-mapping = 131 rows (writer's stat)
- **Actual count**: 206 `|`-prefixed lines across all tables (main §10.9 + mini §10.2.5/§10.3.6/§10.4.6/§10.5.7/§10.6.4 + within-section helpers including production-config table at §10.6.1, proposer comparison §10.5.1, and 5-step structure rows)
- **Result**: PASS — exceeds floor of 10 by 20×, exceeds Ch09 baseline of 151

### Gate 3: All 11 impl-notes source files surfaced in narrative with `:Lxxx`

| File | References in narrative | Status |
|---|---|---|
| `rejection_sampler.py` | 61 occurrences (L30, L34, L37, L195, L246-L281, L392-L503, L425-L430, L450-L466, L491-L504, L659-L703, L708-L757, L723, L734, L743-L745, L751-L757, L760-L826, L797-L810, L811-L815, L853-L920, L877-L920, etc.) | PASS |
| `llm_base_proposer.py` | 23 occurrences (L60-L1820, L407-L412, L413-L656, L491-L494, L516-L654, L1402-L1469, L1471-L1539, L1522-L1538) | PASS |
| `eagle.py` | 8 occurrences (L1-L22, L10-L22, integral file) | PASS |
| `medusa.py` | 13 occurrences (L18-L78, L48-L49, L48-L55, L52-L53) | PASS |
| `draft_model.py` | 15 occurrences (L17-L88, L33-L34, L36-L51, L86-L88) | PASS |
| `ngram_proposer.py` | 8 occurrences (L12-L62, L12-L162, L131-L162, L198-L285) | PASS |
| `extract_hidden_states.py` | 8 occurrences (L26-L70, L26-L130, L29-L31, L72-L130) | PASS |
| `metadata.py` | 16 occurrences (L1-L66, L9-L24, L9-L66, L20, L26-L27, L29-L66) | PASS |
| `deepseek_mtp.py` | 34 occurrences (L43-L62, L63-L122, L92-L97, L99-L121, L107-L113, L124-L184, L160-L170, L172-L182, L186-L488, L271-L456, L458-L488, L464-L470, L480-L488) | PASS |
| `speculative.py` | 9 occurrences (L35-L67, L35-L70, L73-L210, L93, L93-L98, L213-L227) | PASS |
| `llama_eagle3.py` | 2 occurrences (L1-L425 in opening source list + body §10.2.2 referencing `Eagle3LlamaForCausalLM` and the `fc` projection) | PASS |

**Result**: PASS — all 11 source files cited with line numbers.

### Gate 4: 5-step rhythm in §10.1 through §10.6

| Section | Open | What+Why | Derive | Our Impl | Source Diff | Status |
|---|---|---|---|---|---|---|
| §10.1 (rejection math) | §10.1.1 L77-L113 opens `rejection_sampler.py:L392-L503` driver | §10.1.2 L129-L162 "K=4 ≠ 4×" intuition before math | §10.1.2-§10.1.4 derives geometric-series E[tok] + Chen 2023 5-line algebra (L138-L163, L249-L297) | §10.1.1 L115-L131 `rejection_sampling.py:L313-L318`; §10.1.2 L173-L184 `acceptance_math.py:L56-L74` | §10.1.5 L301-L312 6-row mini-mapping with `# vLLM 比我们多了什么` analysis | PASS |
| §10.2 (5 proposers) | §10.2.1 L320-L355 grep evidence + `SpeculativeMethod` literal | §10.2.2 L361-L467 "EAGLE is 22 lines because algorithm IS the base" | §10.2.3 L471-L490 base.propose K==1 fast-path vs K>1 sequential derivation | §10.2.2 L386-L394 `proposers/eagle.py:L20-L36`; L412-L421 `proposers/medusa.py:L80-L104` | §10.2.5 L515-L533 13-row mini-mapping | PASS |
| §10.3 (training→inference reframe) | §10.3.1 L539-L558 grep evidence + reframe declaration with M20 scope correction | §10.3.2 L560-L580 sidebar with literature L_MTP formula | §10.3.3 L583-L635 pivot to `_rewrite_spec_layer_name`; §10.3.4 L639-L688 `_maybe_share_lm_head`; §10.3.5 L692-L717 `SharedHead.forward` M09 correction | §10.3.3 L613 `weight_loading.py:L39-L103`; §10.3.5 L707-L715 `mtp_head.py:L162-L170` | §10.3.6 L719-L734 11-row mini-mapping | PASS |
| §10.4 (rejection kernels) | §10.4.1 L741-L756 driver branches | §10.4.2 L758-L799 "5 observations" of greedy kernel core; §10.4.3 L803-L851 random kernel + 5 observations | §10.4.4 L853-L885 (p-q)_+ Gumbel-max derivation | §10.4.2 L800 `rejection_sampling.py:L56-L131`; §10.4.4 L885 `L223-L279` | §10.4.6 L905-L919 11-row mini-mapping | PASS |
| §10.5 (proposer comparison) | §10.5.1 L926-L944 5-proposer (cost/α/coupling) table from impl-notes §1.5 | §10.5.2 L948-L979 EAGLE↔MTP shared-trunk parity with topology diff; §10.5.3 L981-L1000 Medusa parameter analysis | §10.5.3 L985-L1000 `parameter_count_medusa` closed-form; §10.5.4-§10.5.6 each proposer's hard guards | §10.5.2 L967-L977 `mtp_head.py:L240-L256` MTP layer forward; §10.5.3 L985-L996 closed-form formulas | §10.5.7 L1027-L1038 12-row mini-mapping | PASS |
| §10.6 (system + cross-chapter) | §10.6.1 L1046-L1062 production-config table with α-K-c real values | §10.6.1 L1061-L1063 framing tip 2 net-loss zone synthesis; §10.6.2 L1067-L1078 back/forward-pointers | §10.6.3 L1083-L1090 5 framing tips × 3-anchor verification table | §10.6 reuses Ch01/Ch08/Ch09 components; framing tips documented in test-report | §10.6.4 L1095-L1103 7-row cross-chapter mini-mapping | PASS |

**Result**: PASS — every section opens with source citation, derives theory, shows our impl, ends with mini-mapping. Ch08/Ch09 cadence preserved.

### Gate 5: Demo numerics verbatim (no rounding/approximation)

| Number | Test-report ground truth | Chapter line | Match |
|---|---|---|---|
| 35-cell α-K grid (rows for α=0.3..0.9, K=1..5) | exact 35 numbers | L153-L159 | EXACT |
| α=0.5, K=4 → 1.9375 | 1.9375 | L63, L155, L162, L167-L170, L1115 | EXACT |
| α=0.7, K=4 → 2.7731 | 2.7731 | L157, L162, L170, L1115 | EXACT |
| α=0.3, K=4 → 1.4251 | 1.4251 | L153, L162, L1115 | EXACT |
| Empirical sanity 4 rows | analytic & empirical pairs | L167-L170 | EXACT |
| 28-cell speedup grid (K=4) | exact 28 numbers | L203-L208 | EXACT |
| K=4, c=0.20, α=0.30 → S=0.792 | 0.792 | L64, L207, L211, L1059, L1118, L1330 | EXACT |
| 9 break-even α | 0.0916 / 0.1708 / 0.3062 / 0.1668 / 0.2871 / 0.4553 / 0.2857 / 0.4448 / 0.6206 | L222-L230, L1119 | EXACT |
| KL = 0.000395 | 0.000395 | L65, L294, L1133, L1168, L1307, L1322 | EXACT |
| §3.4 greedy 1.5120 / random 4.5150 / ratio 2.9861 | 1.5120 / 4.5150 / 2.9861 | L893-L898 | EXACT |
| §3.5 6-row breakdown (75,505,664; 216,549,376; 282,085,376; enorm 2,048; eh_proj 8,388,608; mtp_block_attn 16,777,216; mtp_block_ffn 50,331,648; mtp_block_norms 4,096; Medusa per-head 73,924,608; per-head MLP 8,388,608; per-head LM 65,536,000) | exact | L66, L668-L683 | EXACT |
| Ratio MTP/Medusa = 12.91× shared, 1.91× separate | 12.91 / 1.91 | L66, L682-L683, L686, L944, L981, L998, L1087, L1136, L1180, L1311, L1325 | EXACT |
| §3.6 loader 193 / 185 / 8 keys | 193 / 185 / 8 | L620-L633, L1182, L1324 | EXACT |
| 3-path renames | 3 verbatim path examples | L623-L633 | EXACT |
| K=4, c=0.10, α=0.30 → S = 1.018 (`MTP 配低 α 的 risk`) | 1.018 | L1058 | EXACT |
| K=4, c=0.30, α=0.30 → S=0.648 | 0.648 | L1118 | EXACT |

**Result**: PASS — every claimed number reproduces verbatim, no rounding observed. ≥85 verbatim numerics surfaced (exceeds the brief's ≥80 floor).

### Gate 6: §10.2 "no class MultiTokenPrediction" reframe — three-anchor + 4-instance lineage

- **Anchor 1 (Title)** — L1: "第10章：Multi-Token Prediction —— 没有 `class MultiTokenPrediction` 的 K 步并行解码"
- **Anchor 2 (Hook)** — L17: "第 7 章用'vLLM 没有 radix tree'开篇，第 8 章用'vLLM 没有 `class TensorParallel`'开篇，第 9 章用'vLLM 没有 `class ExpertParallel`'开篇——第 10 章是这条系列的 **第四件**". Names all four cases (Ch07 radix → Ch08 TP → Ch09 EP → Ch10 MTP).
- **Anchor 3 (§10.2 body)** — L320-L340: explicit `grep -rE "^class\s+(MultiTokenPrediction|MTPHead|MTPModel|TokenPredictor)\b"` command at commit `98661fe` returning "(zero matches)" + 30+ `*_mtp.py` file enumeration + `SpeculativeMethod` literal grep.
- **4-instance lineage explicit recap**: §10.10 L1323 also lists "Ch07 'no radix tree'、Ch08 'no class TensorParallel'、Ch09 'no class ExpertParallel' 之后的**第四件**'no class X'".

**Result**: PASS — three-anchor template fully satisfied; 4-instance series explicit at hook + recap.

### Gate 7: §10.3 training→inference reframe — sidebar + pivot, 2nd instance after Ch09 §9.4

- **Sidebar** (§10.3.2 L560-L580): Literature exposition of `L_MTP = Σ_{k=0..K-1} λ_k · CE(p_k, x_{t+1+k})` with DeepSeek-V3 / Switch Transformer / Better-MTP grounding, λ_k decay schedule (1.0, 0.5, 0.25, 0.125), teacher-forcing K-to-1 mapping. Closes with explicit "**这一段 100% 是文献简介，源码里没有 corresponding 代码**" (L579).
- **Pivot** (§10.3.3-§10.3.5 L583-L717): Three pivot anchors —
  1. `_rewrite_spec_layer_name` HF→vLLM three-path rewrite (L587-L613) with §3.6 loader demo verbatim 193 → (185, 8) at L620-L633.
  2. `_maybe_share_lm_head` 0.93 GB savings + DeepSeek-V3 vocab × hidden math (L639-L688), tying back to demo §3.5 12.91× ratio.
  3. `SharedHead.forward` M09 correction "只返回 `norm(hidden_states)`" (L692-L717).
- **2nd-instance acknowledgement**: L53 ("这是 Ch09 §9.4 之后的 **第二次** training-to-inference reframe，使用同一个 *sidebar + pivot* 模板") and L556 ("Ch09 §9.4 之后的**第二次**") — both explicit.
- **M20 scope correction surfaced**: L558 "grep 必须 scope 到 spec-decode subtree" with explicit false-positive list (`phimoe.py:router_aux_loss_coef`, `vision.py:get_load_balance_assignment`).
- **No "aux-loss/CE-loss in MoE inference paths" pattern**: spec-decode subtree variant — L552-L555 explicit `grep -rn '\.backward\(' instances/vllm/source/vllm/v1/spec_decode/` "(zero matches)" + L546 grep over `MTPLoss|multi_step_ce|compute_mtp_loss|mtp_aux_loss`.

**Result**: PASS — reframe matches Ch09 §9.4 sidebar+pivot template; both sidebar and pivot complete with M09 + M20 corrections wired in.

### Gate 8: 5 framing tips applied surgically (three-anchor verification each)

Per §10.6.3 L1083-L1090 (writer's own three-anchor table), and confirmed against the chapter body:

| Tip | Hook anchor | Body anchor | Recap anchor | Status |
|---|---|---|---|---|
| **Tip 1** "K=4 ≠ 4×; lead with formula" | Hook L57-L67 (导言): geometric formula + 35-cell number citations + "公式才是真的" | Body §10.1.2 L137-L184: derives chain-break geometry, declares L184 "数字只是公式的实例" | Recap §10.7 Trap A L1111-L1115 + §10.10 L1322 | PASS |
| **Tip 2** "Net-loss zone is THE operator risk" | Hook L64 + 9 break-even α list | Body §10.1.3 L211 "运维必须先测 α 再决定 K，不然 spec-decode 是搬起石头砸自己的脚——这是 framing tip 2 的'net-loss zone is THE headline operator risk'" + §10.6.1 L1059-L1062 production table 0.792 / 1.018 | Recap §10.7 Trap B L1117-L1121 with K=4 c=0.20 α=0.30 → S=0.792 explicit | PASS |
| **Tip 3** "DeepSeek MTP head ≈ 12.91× Medusa; Medusa is the foil" | Hook L66 introduces 12.91× / 1.91× ratios verbatim | Body §10.3.4 L663-L686 with full 6-row breakdown; §10.5.3 L981-L1000 reverses Medusa-first reveal | Recap §10.7 Trap E L1135-L1139 + §10.10 L1325 | PASS |
| **Tip 4** "Inference-only; sidebar→pivot mirroring Ch09 §9.4" | Hook L53 declares "Ch09 §9.4 之后的 **第二次** training-to-inference reframe，使用同一个 *sidebar + pivot* 模板" | Body §10.3.1 grep + §10.3.2 sidebar + §10.3.3-§10.3.5 three pivot anchors | Recap §10.7 Trap F L1141-L1145 + §10.10 L1324 | PASS |
| **Tip 5** "4th 'no class X'; three-anchor template" | Title L1 + Hook L17 with 4-instance enumeration | Body §10.2.1 grep + 5-proposer list | Recap §10.10 L1323 | PASS |

**Result**: PASS — all 5 framing tips applied at hook + body + recap (3 anchors each).

### Gate 9: 5-7 language traps (target 5-7; impl-notes lists 7 candidates A-G)

§10.7 L1109-L1151 contains all 7 traps (A through G):
- Trap A: "MTP 让吞吐翻倍 / K=4 意味着 4× 加速" — geometric chain-break analysis
- Trap B: "Speculative decoding 总是比纯自回归便宜" — net-loss zone with K=4 c=0.20 α=0.30 → S=0.792
- Trap C: "Draft 模型必须共享 target 架构才能高准确率" — DraftModel/Ngram counter-evidence
- Trap D: "Rejection sampling 在高温下有偏" — Chen 2023 unbiasedness for any p, q
- Trap E: "MTP 头是轻量 MLP" — `DeepseekV2DecoderLayer` full transformer block
- Trap F: "vLLM 训练 MTP" — grep 0 matches + M20 scope correction
- Trap G: "Acceptance rate 是模型属性" — α as conditional expectation (draft, target, prompt, temperature)

Each trap follows the *"声明 → 错 → 为什么 → 源码证据 → Demo/测试"* template (matches Ch09 §9.7 / Ch08 §8.6.4).

**Result**: PASS — exactly 7 traps, all from impl-notes A-G candidates, all 5 sub-bullets present per trap.

### Gate 10: Forward-pointers to Ch15+, Ch27, Ch28

| Forward-pointer | Locations | Status |
|---|---|---|
| **Ch15+** model zoo (Llama EAGLE3 / Qwen3 / Mistral / ERNIE per-model wrappers) | L1075 + L1101 + L1334 — three independent mentions; explicit list of `*_mtp.py` / `*_eagle3.py` files with fusion topology preview | PASS |
| **Ch27** DeepSeek-V3.2 deep-dive (MLA + 256-routed-expert MoE, ~120× Medusa, α=0.85 production traffic) | L1076 + L1102 + L1335 | PASS |
| **Ch28** DeepSeek-V4-Pro (`deepseek_v4_mtp.py` `hc_mult` carrier expansion 2-4×) | L1077 + L1103 + L1336 | PASS |

**Result**: PASS — all three forward-pointers cited with concrete content preview.

---

## Brief Corrections Verified

| Correction | Required statement | Where in chapter | Status |
|---|---|---|---|
| **M07** (`"mtp"` IS in `SpeculativeMethod`) | Use transitive containment `MTPModelTypes ⊂ EagleModelTypes ⊂ SpeculativeMethod`; should NOT say `"mtp"` is absent | L13 (header source list), L51 (§10.2 body explicit "**`MTPModelTypes` 通过 `MTPModelTypes ⊂ EagleModelTypes ⊂ SpeculativeMethod` 这条传递链合法存在**"), L357 (§10.2.1 explicit M07 correction call-out: "brief 里曾说 `'mtp'` 不在 `SpeculativeMethod` 里——验证后实际**是在的**") | PASS |
| **M09** (`SharedHead.forward` returns `norm(h)` only) | `forward` returns just `self.norm(hidden_states)`; `lm_head` invoked separately in `compute_logits` | L53 (导言: "`SharedHead.forward` 只返回 `norm(h)`（M09 校正）"), L692-L715 (§10.3.5 explicit M09 correction with code-side comparison) | PASS |

**Result**: Both brief corrections explicitly surfaced and applied in the narrative — not just absorbed silently.

---

## Counted statistics

| Metric | Ch08 | Ch09 | Ch10 (this) | Note |
|---|---|---|---|---|
| Lines | 933 | 1204 | 1345 | +12% over Ch09 |
| Words | ~6500 | 7792 | 8888 | +14% over Ch09 |
| Mapping rows | 122 | 151 | 206 | +37% over Ch09 |
| Demo verbatim numbers | ≥50 | ≥75 | ≥85 | exceeds brief floor of ≥80 |
| Source files surfaced | 8 | 10 | 11 | matches impl-notes |
| Language traps | 5 | 7 | 7 | matches Ch09 |
| Reframes | 1 (no class) | 2 (no class + training→inference) | 2 (no class 4th + training→inference 2nd) | matches Ch09 |
| Sections (10.1-10.10) | §8.1-§8.7 | §9.1-§9.10 | §10.1-§10.10 | full coverage |
| Forward-pointers | Ch09/Ch11 | Ch10/Ch11+ | Ch15+/Ch27/Ch28 | all explicit |

Pedagogical posture is consistent with Ch08/Ch09 cadence; chapter is denser as expected (rejection-sampling math + 5 proposers + 30+ model wrappers + reframes is more surface area than Ch09's 6 module families).

---

## Verdict

**APPROVED**. All 10 hard gates pass. Both brief corrections (M07, M09) explicitly applied. Both reframes (4th "no class X" + 2nd training→inference) follow the established three-anchor / sidebar+pivot templates. All 5 framing tips applied at three anchors each. Verbatim demo numerics reproduce exactly (≥85 numbers, exceeds ≥80 floor). 11 source files all surface. Single-cycle approval matches Ch09 cadence.

**Cycles**: 1 (single-cycle, mirrors Ch08/Ch09).

Writer-2 may proceed; Archivist may record delivery.
