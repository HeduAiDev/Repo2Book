# Ch10 Multi-Token Prediction v6 PUBLISHED — seventh v6 chapter, cadence holds at N=7; broadest source surface to date (11 vLLM modules + 5 proposer family classes); 4th "no class X" reframe graduates to chapter motif; 2nd training→inference reframe template stable

- **Type**: delivery
- **Chapter**: 10-multi-token-prediction
- **Date**: 2026-05-07
- **Timestamp**: 2026-05-07T08:00:00Z
- **Agents involved**: implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: v6, multi-token-prediction, no-class-reframe-4th, training-to-inference-reframe-2nd, rejection-sampling, chen-2023, geometric-chain-break, alpha-K-grid, deepseek-mtp, eagle, medusa, draft-model, ngram, framing-tips, language-trap

## What happened

Reviewer-2 APPROVED in 1 cycle (no REVISE iterations). All **10 hard gates pass**.
Both linters PASS at the BLOCKING bar: formula linter **0 blocking + 11
non-blocking inline-density warnings** (L162, L235, L297, L570-L573, L575,
L1000, L1048-L1059, L1118-L1121, L1124-L1127, L1130-L1133, L1148-L1151) —
every flagged line contains single-symbol inline tokens (`$\alpha$`, `$K$`,
`$E[\mathrm{tok}]$`, `$h_t$`, `$x_{t+1..t+k}$`, `$p_k$`, `$\lambda_k$`,
`$L \in [1, K+1]$`, `$\lfloor L-1 \rfloor$`, `$S < 1$`, `$cK$`) within the
documented "single-symbol allowed inline" rule (E24 from Ch09; M30 reviewer
calibration locks "warnings up to ~15 acceptable IF every flagged paragraph
passes the single-symbol-token check"). Higher than Ch09's 4 non-blocking but
strictly within the calibrated bar; source-grounding linter all PASS.

**311/311 tests pass at vLLM source commit `98661fe`**. Test count is **52% above
Ch09's 204** (Ch04: 48; Ch05: 74; Ch06: 97; Ch07: 83; Ch08: 144; Ch09: 204;
Ch10: **311**). Breadth comes from parametrising the α-K-c numerical grids:
**35 α-K cells × 28 speedup cells × 9 break-even α** + 5 propose_K cells +
4 K-uniform spec_metadata cells + 4 batch-size cells = ~85 parametrised cells
contributing ~30% of the test count (M25). Coverage by module: spec_metadata
30, rejection_sampling 29, acceptance_math 52, mtp_head 52, weight_loading 32,
proposers 40, integration 13, fidelity 21, demo_numerics 42 (test-report
header). 7-trap fidelity verification (each trap with ≥1 test), 5-proposer
family inheritance topology pin, exhaustive demo §3.1-§3.6 verbatim pins
(≥85 verbatim numerics surfaced — exceeds brief's ≥80 floor).

**Stats**: **1345 lines, 8888 words, 206 mapping rows** (80 main §10.9 + 51
mini across §10.2.5/§10.3.6/§10.4.6/§10.5.7/§10.6.4 + within-section helpers
including production-config §10.6.1, proposer comparison §10.5.1, and 5-step
structure rows — 206 total `|`-prefix lines counted by reviewer). Every
metric strictly exceeds all six prior v6 chapters — Ch04: 712/3064/13;
Ch05: 757/3849/21; Ch06: 655/3351/40; Ch07: 859/4440/72; Ch08: 1051/6058/122;
Ch09: 1204/7792/151; Ch10: **1345 / 8888 / 206**. Lines +12% over Ch09,
words +14% over Ch09, mapping rows +37% over Ch09. impl-notes "Source
Analysis" lists **11 distinct vLLM source files** (rejection_sampler.py,
metadata.py, llm_base_proposer.py, eagle.py, medusa.py, draft_model.py,
extract_hidden_states.py, ngram_proposer.py, deepseek_mtp.py, speculative.py,
llama_eagle3.py) — exceeds v6 floor of 5 (Ch04: 5, Ch05: 7, Ch06: 6, Ch07: 5,
Ch08: 8, Ch09: 10). **151 # REFERENCE: comments** across impl modules
(rejection_sampling.py 37, mtp_head.py 18, weight_loading.py 18, proposers
total 58 = base 14 + medusa 11 + extract_hidden 9 + draft_model 9 + ngram 7
+ eagle 4 + mtp 4, acceptance_math.py 10, spec_metadata.py 8, impl-notes.md
2) — count exceeds Ch09's 66 by 129%. Note: count inflation comes from the
proposer family's 7-file boilerplate (each proposer needs base-class +
override REFERENCEs); not a quality jump above Ch09's 66, just broader
source surface. Reviewer flagged all 11 source files with explicit `:Lxxx`
line refs in narrative — 61 occurrences for rejection_sampler.py (driver +
two kernels + recovered sampling + parse_output), 34 for deepseek_mtp.py
(SharedHead + MTP layer + predictor stack + load_weights + 3 path
rewrites), 23 for llm_base_proposer.py (init + propose + share-helpers),
16 for metadata.py, 15 for draft_model.py, 13 for medusa.py, 9 for
speculative.py, 8 each for eagle.py / extract_hidden_states.py /
ngram_proposer.py, 2 for llama_eagle3.py. Coverage is complete; nothing
referenced indirectly.

5-step rhythm verified §10.1-§10.6 by reviewer (gate 4): each section opens
with source location + `:Lxxx` → bridge → theory derive → our impl → source
diff/mapping. Two-tier mapping (K15 from Ch07) reused at higher density:
80-row master + 6 per-section mini-tables anchored to each section's
source walk — extends the Ch06/Ch07/Ch08/Ch09 pattern with the broadest
surface yet.

**§10.2 reframe applied** as designed in the implementer brief — outline
subsection "MTP头网络结构" reframed at chapter level as **"vLLM 没有
`class MultiTokenPrediction`，也没有 `class MTPHead` / `class MTPModel` —
30+ 个 `*_mtp.py` 模型族 wrapper + 5-proposer family + 1-verifier"**.
Established at THREE structural anchors:
1. Title L1: "第10章：Multi-Token Prediction —— 没有 `class MultiTokenPrediction` 的 K 步并行解码"
2. Hook L17: "第 7 章用'vLLM 没有 radix tree'开篇，第 8 章用'vLLM 没有
   `class TensorParallel`'开篇，第 9 章用'vLLM 没有 `class ExpertParallel`'
   开篇——第 10 章是这条系列的 **第四件**". Names all four cases (Ch07
   radix → Ch08 TP → Ch09 EP → Ch10 MTP).
3. §10.2 body L320-L340: explicit `grep -rE "^class\s+(MultiTokenPrediction|
   MTPHead|MTPModel|TokenPredictor)\b"` command at commit `98661fe` returning
   "(zero matches)" + 30+ `*_mtp.py` file enumeration + `SpeculativeMethod`
   literal grep showing `"mtp"` is NOT a literal (only enters via
   `MTPModelTypes ⊂ EagleModelTypes ⊂ SpeculativeMethod` transitive chain).

Plus 4-instance lineage explicit recap at §10.10 L1323 (lists "Ch07 'no
radix tree'、Ch08 'no class TensorParallel'、Ch09 'no class ExpertParallel'
之后的**第四件**'no class X'"). This is the **fourth instance of the
"no class X" framing pattern** — graduates from Ch07/Ch08/Ch09 series
convention to a **chapter motif** (recognized pattern explicitly carried
across chapters with grep evidence + 4-instance enumeration each time).

**§10.3 outline reframe applied** (gate 7) — the outline subsection
"Training——多步CE损失的加权策略" was identified as a training-only topic
that doesn't exist in vLLM's inference-only codebase. The chapter pivots
using the SAME *sidebar+pivot* template that Ch09 §9.4 introduced for
EPLB-vs-aux-loss:
- L53: hook acknowledges "Ch09 §9.4 之后的 **第二次** training-to-inference
  reframe，使用同一个 *sidebar + pivot* 模板"
- L556: §10.3.1 same reference repeated
- L552-L558: precise scope — `grep -rn '\.backward\(' instances/vllm/source/
  vllm/v1/spec_decode/` and `grep -rE 'MTPLoss|multi_step_ce|compute_mtp_loss|
  mtp_aux_loss' vllm/v1/spec_decode/` both "(zero matches)"
- L558: M20 scope correction — false-positive list (`phimoe.py:
  router_aux_loss_coef` HF-config carrier; `vision.py:get_load_balance_assignment`
  image-tile balancing — neither is MTP training)
- L560-L580: training-aux-loss sidebar (DeepSeek-V3 paper, Switch Transformer,
  Better-MTP) for grounding; closes with explicit "**这一段 100% 是文献简介,
  源码里没有 corresponding 代码**" (L579)
- L583-L717: three pivot anchors —
  1. `_rewrite_spec_layer_name` HF→vLLM three-path rewrite (L587-L613) with
     §3.6 loader demo verbatim 193 → (185, 8) at L620-L633
  2. `_maybe_share_lm_head` 0.93 GB savings + DeepSeek-V3 vocab × hidden
     math (L639-L688), tying back to demo §3.5 12.91× ratio
  3. `SharedHead.forward` M09 correction "只返回 `norm(hidden_states)`"
     (L692-L717) — IMPORTANT: brief said `forward` returns norm + lm_head;
     verified that the forward returns ONLY norm; lm_head is invoked
     separately in compute_logits

This is the **2nd instance** of the training-to-inference reframe
sub-pattern (Ch09 §9.4 EPLB → Ch10 §10.3 weight loading). The
*sidebar+pivot* template is now stable across two chapters per M28 —
M28 explicitly says "queue for wisdom promotion at instance #3" (i.e., the
template can be promoted to wisdom only after a third repo-instance
candidate; current count is intra-vllm-instance N=2 which does NOT meet
the strict 2+ INSTANCES bar).

**5 framing tips from tester applied surgically with three-anchor verification**
(gate 8 — hook + body + recap, per M29 reviewer rule):
1. **Tip 1: "K=4 ≠ 4×; lead with formula"** — Hook L57-L67 (geometric
   formula + 35-cell number citations + "公式才是真的"); Body §10.1.2
   L137-L184 (derives chain-break geometry, declares L184 "数字只是公式
   的实例"); Recap §10.7 Trap A L1111-L1115 + §10.10 L1322. Three
   anchors; reader can derive the speedup curve themselves rather than
   memorising 35 cells. Explicit M17 lesson — formula leads, numerics second.
2. **Tip 2: "Net-loss zone is THE operator risk"** — Hook L64 + 9
   break-even α list; Body §10.1.3 L211 "运维必须先测 α 再决定 K，
   不然 spec-decode 是搬起石头砸自己的脚——这是 framing tip 2 的
   'net-loss zone is THE headline operator risk'" + §10.6.1 L1059-L1062
   production table 0.792 / 1.018; Recap §10.7 Trap B L1117-L1121 with
   K=4 c=0.20 α=0.30 → S=0.792 explicit. M18 explicit.
3. **Tip 3: "DeepSeek MTP head ≈ 12.91× Medusa; Medusa is the foil"** —
   Hook L66 introduces 12.91× / 1.91× ratios verbatim; Body §10.3.4
   L663-L686 with full 6-row breakdown; §10.5.3 L981-L1000 reverses
   Medusa-first reveal; Recap §10.7 Trap E L1135-L1139 + §10.10 L1325.
4. **Tip 4: "Inference-only; sidebar→pivot mirroring Ch09 §9.4"** —
   Hook L53 declares "Ch09 §9.4 之后的 **第二次**"; Body §10.3.1 grep +
   §10.3.2 sidebar + §10.3.3-§10.3.5 three pivot anchors; Recap §10.7
   Trap F L1141-L1145 + §10.10 L1324. Same template reproduced.
5. **Tip 5: "4th 'no class X'; three-anchor template"** — Title L1 +
   Hook L17 with 4-instance enumeration; Body §10.2.1 grep + 5-proposer
   list; Recap §10.10 L1323. Three anchors; reader is invited to name
   the meta-pattern themselves.

**7 language-trap callouts** in §10.7 (gate 9) following Ch06/Ch07/Ch08/Ch09
lineage style "claim → 错 → 为什么 → 源码证据 → Demo/测试":
- Trap A (L1109-L1115): "MTP 让吞吐翻倍 / K=4 意味着 4× 加速" — geometric
  chain-break analysis with α=0.5 K=4 = 1.9375 verbatim
- Trap B (L1117-L1121): "Speculative decoding 总是比纯自回归便宜" —
  net-loss zone with K=4 c=0.20 α=0.30 → S=0.792
- Trap C (L1123-L1127): "Draft 模型必须共享 target 架构才能高准确率" —
  DraftModel/Ngram counter-evidence
- Trap D (L1129-L1133): "Rejection sampling 在高温下有偏" — Chen 2023
  unbiasedness for any p, q with KL=0.000395 verification
- Trap E (L1135-L1139): "MTP 头是轻量 MLP" — `DeepseekV2DecoderLayer`
  full transformer block (12.91× Medusa shared-lm)
- Trap F (L1141-L1145): "vLLM 训练 MTP" — grep 0 matches + M20 scope
  correction to spec-decode subtree
- Trap G (L1147-L1151): "Acceptance rate 是模型属性" — α as conditional
  expectation (draft, target, prompt, temperature)

7 traps **matches Ch09's** 7 (target floor was 5-7); each follows the same
5-substructure template per E19 from Ch09. Plus per-section cross-references
where each trap originates (Trap-A threaded through §10.1.2 derivation;
Trap-B through §10.1.3 break-even table; Trap-E through §10.5.3 12.91×
ratio reveal; Trap-F through §10.3.1 grep + §10.3.5 reframe).

**Forward-pointers** to Ch15+/Ch27/Ch28 wired (gate 10):
- Ch15+ (model zoo) at L1075, L1101, L1334: per-model wrappers — Llama
  EAGLE3 / Qwen3-MTP / Mistral / ERNIE; explicit list of `*_mtp.py` /
  `*_eagle3.py` files with fusion topology preview
- Ch27 (DeepSeek-V3.2) at L1076, L1102, L1335: deep-dive — MLA + 256-routed-
  expert MoE, ~120× Medusa, α=0.85 production traffic
- Ch28 (DeepSeek-V4-Pro) at L1077, L1103, L1336: `deepseek_v4_mtp.py`
  `hc_mult` carrier expansion 2-4× preview

**Brief corrections applied** (verified by reviewer in §"Brief Corrections
Verified"):
- **M07**: brief said `"mtp"` is NOT in `SpeculativeMethod`; verified at
  98661fe that it IS, transitively via `MTPModelTypes ⊂ EagleModelTypes ⊂
  SpeculativeMethod`. Surfaced at L13 (header source list), L51 (§10.2 body
  explicit "**`MTPModelTypes` 通过 `MTPModelTypes ⊂ EagleModelTypes ⊂
  SpeculativeMethod` 这条传递链合法存在**"), L357 (§10.2.1 explicit M07
  call-out: "brief 里曾说 `'mtp'` 不在 `SpeculativeMethod` 里——验证后
  实际**是在的**"). Important — brief was wrong on this point and chapter
  surfaces the correction explicitly rather than silently absorbing it.
- **M09**: `SharedHead.forward` returns `norm(h)` only; `lm_head` invoked
  separately in `compute_logits`. Surfaced at L53 (导言: "`SharedHead.forward`
  只返回 `norm(h)`（M09 校正）"), L692-L715 (§10.3.5 explicit M09 correction
  with code-side comparison).

**Implementation bug found+fixed during testing**: M16 — `_MultiHeadAttention.forward`
in `mtp_head.py` had a shape mismatch (q@k.T produced `[T, num_heads, num_heads]`
instead of `[num_heads, T, T]`; transpose missing). Tester applied minimal
fix: permute `(0, 1) → (heads, T, head_dim)` before attention math, and back
to `(T, heads, head_dim)` before o_proj. Standard SDPA pattern. Demo §3.5
numerics unchanged (parameter counts depend only on weight shapes, not
forward). Recorded as M22 with the reshape-then-permute lesson. Two other
small bugs: M23 — `proposers/base.py` was missing `@dataclass ProposerOutput`
(import-blocking); M24 — `acceptance_math.parameter_count_medusa` returned
flat dict missing `per_head_mlp` key (test contract divergence with
`mtp_head.parameter_count_medusa`).

**Knowledge appended**: M01-M15 (implementer-supplied, M07 with correction
note) + M16-M22 (tester-added: bug + framing tips + parametrisation lesson)
+ M23-M28 (writer-added: ProposerOutput, parameter_count divergence,
311-test parametrisation lesson, inline-density discipline, 5-step rhythm
fusion, reframe-template stability) + M29-M30 (reviewer-added: three-anchor
mechanical grep verification, inline-density warning calibration) =
**M01-M30 in `knowledge/modules/multi-token-prediction.md`**.

**Note**: multi-token-prediction.md now has **30 facts** (M01-M30) —
exceeds the 15-fact compaction trigger by 15. **Fifth chapter** to
demonstrate compact() brokenness in succession (Ch07: 17 → Ch08: 19 →
Ch09: 24 → Ch10: 30, plus pre-existing scheduler.md 12-after-2-compactions).
Rate of fact accumulation is accelerating: Ch10 added 30 in one shot vs
Ch09's 24, Ch08's 19, Ch07's 17. **P2-2 in system-improvements.md** —
implement working `learn.py compact` so chapter knowledge files don't grow
unbounded — is now blocking automated knowledge hygiene across **5
consecutive chapters**. Manual workaround remains in use.

**Minor non-blocking observations** (do NOT trigger REVISE; reviewer noted
for writer's future awareness):
1. L491 grep specificity in §10.2 — reviewer praised the M07 correction
   as "explicitly tells the reader 'brief was wrong, here's the verified
   transitive chain' so readers don't go memorise the wrong claim".
2. L558 M20 grep specificity — reviewer praised the spec-decode-subtree
   scoping ("vllm/v1/spec_decode/" not whole-vllm) as the cleanest "no
   training MTP loss" wording across Ch09/Ch10.
3. L692-L715 M09 — reviewer praised triple-anchoring (导言 + §10.3.5
   body + Trap F recap) as "no reader can mistake forward for the full
   logit projection".
4. 11 inline-density warnings — reviewer accepted under M30 calibration
   (warnings up to ~15 acceptable IF every flagged paragraph passes
   single-symbol token check).

Source pinned at vLLM commit `98661fe`. Snapshot location TBD (matching
prior pattern `trace/snapshots/{N}-{slug}/v6-2026-05-07/`).

## Why it matters

**SEVENTH chapter under v6 standards** — cadence holds at N=7. Critically,
Ch10 is the chapter where **two recognized chapter motifs graduate**: the
"no class X" reframe pattern reaches its 4th instance and is no longer a
nascent convention but an established motif; the training→inference
reframe sub-pattern reaches its 2nd instance and the *sidebar+pivot*
template stabilises. Three structural ways v6 is now strictly better at
N=7 than at N=6:

1. **Source surface scales without losing focus, AGAIN.** Ch08: 8 source
   files (Megatron TP); Ch09: 10 source files (EP routing + EP placement +
   EP groups + all-to-all backend + EPLB state + 2 reference model sites);
   Ch10: **11 source files** (rejection sampler kernel + spec-decode
   metadata contract + base proposer scaffolding + 5 proposer family
   subclasses + DeepSeek MTP model wrapper + speculative config +
   EAGLE3 reference). Two new code regions enter the book:
   `vllm/v1/sample/rejection_sampler.py` (the algorithmic core for ALL
   speculative decoding) and `vllm/v1/spec_decode/` (the entire spec-decode
   subtree). The 5-step rhythm and two-tier mapping survive the breadth —
   every section opens with a specific file:line, every file gets a mini-
   mapping table, the 80-row master indexes the chapter. Pattern from
   Ch08 (8 files) → Ch09 (10 files) → Ch10 (11 files) generalizes without
   dilution.

2. **Test breadth scales by parametrisation, ON THIRD AXIS.** 311 tests is
   a 52% increase over Ch09's 204. M25 explicitly captures the lever:
   parametrising over (α, K, c) numerical grids gives ~30% of test count
   for free — 35 α-K cells × 28 speedup cells × 9 break-even α = ~85
   parametrised cells. Ch08 used `tp_size ∈ {1,2,4,8}` (single axis);
   Ch09 used `ep_size × routing_path` (2 axes); Ch10 uses `α × K × c`
   (3 axes — the mathematical/operational parameter space rather than
   single config knob). Pattern: **whenever a chapter quotes a numerical
   grid, parametrise the cells** — near-free test breadth that surfaces
   drift. Sequence Ch08(144) → Ch09(204) → Ch10(311) is the cadence.

3. **Mapping density compounds through breadth.** 206 rows = 80 master +
   51 mini + ~75 helper. Two-tier pattern (K15) now scales to 11 source
   files; main density per source file ≈ 7 rows + per-section mini ≈ 9
   rows. Ch08 (122 rows / 8 files = 15.3 per file) → Ch09 (151 rows / 10
   files = 15.1 per file) → Ch10 (206 rows / 11 files = 18.7 per file).
   Per-file density INCREASING (not just total density). Pattern is
   **load-bearing for breadth and depth simultaneously**.

Beyond metrics, four patterns reproduce that lock in v6 robustness:

4. **The "no class X" framing pattern graduates to chapter motif (N=4).**
   Ch07 "no radix tree" → Ch08 "no class TensorParallel" → Ch09 "no class
   ExpertParallel" → Ch10 "no class MultiTokenPrediction". Three structural
   anchors (title + hook + §X.2 body) every time, plus 4-instance
   enumeration in hook starting from Ch09 (Ch09 listed 3 prior; Ch10 lists
   4 prior). **The pattern is now a recognized chapter motif** — readers
   can predict the framing on encountering an outline-vs-source-class-name
   mismatch in subsequent chapters. Ch11 (DCP/PCP) is the next likely
   candidate (DCP/PCP probably has no `class DecodeContextParallel` /
   `class PrefillContextParallel` either — verified in Ch11 brief). Would
   establish N=5. **STILL not eligible for wisdom promotion** per strict
   2+ INSTANCES rule — this is intra-vllm-instance N=4, not cross-instance.
   The motif is repo-specific patterning, not a universal pattern.

5. **§10.3 reproduces Ch09 §9.4's training→inference reframe template.**
   Sidebar+pivot template: (a) sidebar grounding from literature (1-2
   paragraphs of training-side math, properly cited), (b) explicit grep
   evidence that training code is absent from vLLM, (c) pivot to the 2-3
   inference-time helpers that ARE in vLLM and address the same concern,
   (d) demo numerics from those helpers. Ch09 §9.4 was EPLB-vs-aux-loss;
   Ch10 §10.3 is rewrite_spec_layer_name+share_lm_head-vs-multi-step-CE-loss.
   **2nd instance — template is stable**. M28 says queue for wisdom
   promotion at instance #3 (i.e., needs a 3rd cross-chapter or
   cross-instance recurrence). Future candidates: Ch16 quantisation
   (outline says QAT but vLLM only handles inference-time quant), Ch20
   distillation. **Reviewer expected sidebar + pivot per the same
   template** and verified both — the template is a reviewer-checkable
   pattern now, not a one-off Ch09 quirk.

6. **Tester framing-guidance loop reproduces (N=5).** Ch06 → Ch07 → Ch08 →
   Ch09 → Ch10. Each tester's surgical tips applied by writer and verified
   by reviewer with explicit three-anchor (hook + body + recap) check
   (E22/M29). Ch06: P05 sweep-pair language + K18 invariant-first; Ch07:
   no-asymptotic-faster + (N-1)×K-formula + chain-break-is-THE-invariant;
   Ch08: 1-AR-per-pair + α-bound-first + K17-OR-skip + bias-on-rank-0
   zero-weight + MergedColumn bug story; Ch09: Trap-G renorm-flag qualifier
   + MoE-inference-paths wording + mem-formula-leads + α-β-tautology +
   3-tuple-vs-2-tuple; Ch10: K=4≠4× formula-leads + net-loss zone-as-headline
   + MTP-head-12.91×-Medusa + sidebar→pivot template + 4th-no-class-X.
   **Five chapters in a row — testers consistently produce load-bearing
   narrative-shaping guidance, and writers/reviewers consistently apply
   it with three-anchor discipline.** Operationally reproducible enough
   that future testers can be expected to produce explicit framing tips
   of this caliber in their handoffs.

7. **APPROVED-at-cycle-1 discipline holds across all 7 v6 chapters.**
   Ch04, Ch05, Ch06, Ch07, Ch08, Ch09, Ch10 all 1-cycle. Pipeline cost is
   predictably low when writer pre-runs both linters AND reports outputs
   in handoff. **M30 reviewer wisdom calibrates the "0 blocking + every
   inline token single-symbol" bar even sharper**: warnings up to ~15
   acceptable IF every flagged paragraph passes the per-token single-symbol
   check. Ch10's 11 non-blocking warnings cleared the calibrated bar
   without REVISE. Sequence Ch08(0/0) → Ch09(0/4) → Ch10(0/11) shows the
   warning-tolerance band is a function of formula content density, not
   quality; M30 makes this explicit.

8. **Outline-as-topic-not-class-contract** (rule #6 from session pause) is
   now operationally proven on **FOUR chapters** (Ch07 radix-tree, Ch08
   TensorParallel-class, Ch09 ExpertParallel-class + ExpertLoadBalancingLoss-
   training, Ch10 MultiTokenPrediction-class + MTP-CE-loss-training).
   Outline JSON unchanged in all four cases; chapters honestly represent
   source reality. This protects outline JSON stability while letting
   chapters frame absent topics as "why isn't this here?".

9. **No new framework bugs surfaced this chapter; one fact-accumulation
   risk surfaced.** The known framework bugs (`scripts/learn.py
   _parse_module_file` returns [] → compact() non-functional; append-mode
   produces malformed double-prefix headings) remain open with manual
   workarounds. multi-token-prediction.md hitting 30 facts (15 over the
   compaction trigger) is now the **fifth chapter** to demonstrate compact()
   brokenness — escalating from "blocking automated knowledge hygiene
   across 4 chapters" (Ch09 delivery) to **5 consecutive chapters with
   ever-larger files** (Ch07: 17 → Ch08: 19 → Ch09: 24 → Ch10: 30).
   Rate of fact accumulation is accelerating. **P2-2 must be implemented
   before Ch12** as flagged in Ch09 — Ch11 will compound this further
   (DCP/PCP is a new module with D-prefix per brief plan).

cadence_holds_at_n7 unlocks the inference: v6 is robust enough to walk
**the broadest source surface yet** (11 vLLM modules including the entire
spec-decode subtree) in a single cycle. Pipeline can plan Ch11 (DCP/PCP)
→ Ch13 (prefix-cache-pooling) on this basis. The "no class X" pattern is
firmly established as a chapter motif, the §9.4/§10.3 training→inference
reframe template is stable enough to be a reviewer-checkable gate, and the
parametrise-numerical-grids test breadth lever is the third independent
axis after tp_size (Ch08) and ep_size (Ch09).

## What to remember

Reviewer-2 APPROVED in 1 cycle. Both linters PASS — formula 0 blocking +
11 non-blocking single-symbol-inline warnings (M30-acceptable);
source-grounding all PASS. **311/311 tests pass** at commit 98661fe (52%
test count increase over Ch09's 204). **1345 lines, 8888 words, 206
mapping rows** (80 master + 51 mini + 75 helper — two-tier per K15 with
broadest source surface yet). **11 vLLM source files cited**
(rejection_sampler, metadata, llm_base_proposer, eagle, medusa, draft_model,
extract_hidden_states, ngram_proposer, deepseek_mtp, speculative,
llama_eagle3) — 151 REFERENCE comments across 9 impl modules (count
inflated by proposer family boilerplate, not quality jump above Ch09's 66).
**§10.2 "no class MultiTokenPrediction" reframe** applied successfully —
4th in series after Ch07 (radix tree) + Ch08 (TensorParallel) + Ch09
(ExpertParallel) — graduates from intra-instance series convention to
recognized chapter motif. **§10.3 outline reframe** applied as 2nd
instance of training→inference *sidebar+pivot* template (Ch09 §9.4 was
1st); template now stable, M28 says queue for wisdom promotion at #3.
5 framing tips from tester applied with three-anchor verification (hook +
body + recap, per M29). 7 language-trap callouts in §10.7 (A-G) plus
per-section cross-refs — matches Ch09's 7. Forward-pointers to
Ch15+/Ch27/Ch28 wired. Brief corrections (M07: `"mtp"` IS in
`SpeculativeMethod` via transitive containment; M09: `SharedHead.forward`
returns norm only) explicitly surfaced — chapter does NOT silently absorb.
Implementation bug found+fixed during testing: M16 `_MultiHeadAttention.forward`
shape mismatch (now M22 with reshape-then-permute lesson). Knowledge M01-M15
(implementer) + M16-M22 (tester) + M23-M28 (writer) + M29-M30 (reviewer)
— **30 facts total in multi-token-prediction.md, 15 over compaction
trigger**; flag for P2-2 (5th compact() failure demo, now blocking 5
consecutive chapters; rate accelerating). **cadence_holds_at_n7** with
broadest source surface yet — v6 robust at N=7; not just holding but
actively scaling source breadth (11 files vs Ch09's 10) and test breadth
(311 vs 204) without dilution. Pipeline pattern of pre-run-linters-AND-
tester-tips-in-handoff confirmed across 5 consecutive chapters.
**Brief-on-approval discipline triggers Ch11 (DCP/PCP) brief immediately**
— likely 5th "no class X" candidate (verify DCP/PCP class absence in
brief), with sequence-parallel sharding math + Ring Attention as new
theory surface, and per outline 5 subsections covering the DCP/PCP
semantic distinction + CP+TP 3D parallel mesh.
