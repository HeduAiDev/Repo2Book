# Ch09 Expert Parallelism v6 PUBLISHED — sixth v6 chapter, cadence holds at N=6; broadest source surface yet (10 vLLM modules) and third "no class X" reframe in series

- **Type**: delivery
- **Chapter**: 09-expert-parallelism
- **Date**: 2026-05-07
- **Timestamp**: 2026-05-07T03:30:00Z
- **Agents involved**: implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: v6, expert-parallelism, no-class-reframe, eplb-pivot, alpha-beta-tautology, mesh-formula-lead, language-trap, two-tier-mapping, framing-tips

## What happened

Reviewer-2 APPROVED in 1 cycle (no REVISE iterations). All **10 hard gates pass**.
Both linters PASS at the BLOCKING bar: formula linter **0 blocking + 4 non-blocking
inline-density warnings** (L608, L754, L773, L918) — every flagged line contains
single-symbol inline tokens (`$f_i$`, `$P_i$`, `$3Fh$`, `$E$`, `$h$`, `$F$`, `$ep$`,
`$tp$`, `$\alpha$`, `$\beta$`, `$S$`) within the documented "single symbol allowed
inline" rule (writer wisdom rule). This is **slightly below Ch08's 0/0** but
strictly cleaner than Ch07's 5 non-blocking and well within v6 floor;
source-grounding linter all PASS.

**204/204 tests pass at vLLM source commit `98661fe`**. Test count is **42% above
Ch08's 144** (Ch04: 48; Ch05: 74; Ch06: 97; Ch07: 83; Ch08: 144; Ch09: **204**).
Breadth comes from parametrising `ep_size ∈ {1,2,4,8}` across routing/expert_map/
integration plus two routing-path families (Mixtral fused_topk + DeepSeek grouped_topk)
at every test level plus exhaustive demo-numerics pins across §3.1-§3.5. Demo
numerics reproduce verbatim in narrative — every ground-truth number from
`tests/test-report.md` greps clean: Mixtral count `[250, 285, 277, 243, 253, 272,
247, 221]`; DeepSeek `max=131 min=78 mean=96.00`; renormalize=False range
`[0.2730, 0.6171] mean 0.3899`; max/mean linear ep=8 = **3.251**; round-robin recovery
**1.196**; α-β NVLink table `16.09 / 67.47 / 478.51 / 3766.85 μs`; α-β IB headlines
`50.70μs / 18804.48μs`; α-β ratio = **2.000** (with explicit "model identity, not
measurement" tautology callout); memory table 6 rows
`1056/264/132/66/66/33`; `(4,2) ≡ (8,1) ≡ 132 MiB` invariant; EPLB timeline
`2.523 → 2.529 → 1.203 → 1.158 → 1.229 → 1.193`; `physical_to_logical[-4:] =
[5, 2, 0, 4]`.

**Stats**: **1204 lines, 7792 words, 151 mapping rows** (49 main §9.9 + 39 mini
across §9.2.5 / §9.3.5 / §9.4.6 / §9.5.5 / §9.6.5 + helper tables — 151 total
`|`-prefix lines counted by reviewer). Every metric strictly exceeds all five
prior v6 chapters — Ch04: 712/3064/13; Ch05: 757/3849/21; Ch06: 655/3351/40
(29 main + 11 mini); Ch07: 859/4440/72 (27 main + 45 mini); Ch08: 1051/6058/122;
Ch09: **1204 / 7792 / 151**. Lines +14.6%, words +28.6%, mapping rows +23.8% over
Ch08. impl-notes "Source Analysis" lists **10 distinct vLLM source files**
(parallel_state.py, all2all.py, fused_moe/layer.py, fused_moe/config.py,
router/fused_topk_router.py, router/grouped_topk_router.py,
prepare_finalize/naive_dp_ep.py, eplb/eplb_state.py, models/mixtral.py,
models/deepseek_v2.py) — exceeds v6 floor of 5 (Ch04: 5, Ch05: 7, Ch06: 6,
Ch07: 5, Ch08: 8). **66 # REFERENCE: comments** across 7 impl modules
(routing.py 14, fused_moe_block.py 15, ep_groups.py 15, expert_map.py 7,
all2all_baseline.py 6, eplb.py 5, mixtral_vs_deepseek.py 4) plus demo.py 0
and __init__.py 0. Reviewer flagged all 10 source files with explicit `:Lxxx`
line refs in narrative — 24 refs to parallel_state.py, 25 to all2all.py, 17 to
fused_moe/layer.py, 12 to fused_topk_router.py, 12 to eplb_state.py, etc.
Coverage is complete; nothing referenced indirectly.

5-step rhythm verified §9.1-§9.6 by reviewer (gate 4): each section opens with
source location + `:Lxxx` → bridge → theory derive → our impl → source diff/
mapping. Two-tier mapping (K15 from Ch07) reused at higher density: 49-row master
+ 5 per-section mini-tables anchored to each section's source walk — extends
the Ch06/Ch07/Ch08 pattern with the broadest surface yet.

**§9.2 reframe applied** as designed in the implementer brief — outline subsection
"TensorParallel/ExpertParallel 类详解" reframed at chapter level as **"vLLM 没有
`class ExpertParallel`，也没有 `class MoEParallel`，更没有 `class TopKGate` —
5 个文件协同实现 EP+TP composition"**. Established at THREE structural anchors:
1. Title L1: explicitly says "没有 `class ExpertParallel`"
2. Hook L16: names all three absent classes (ExpertParallel/MoEParallel/TopKGate)
3. §9.2 body L246-L264: grep evidence + `(zero matches)` + 5-file table
   (parallel_state.py / fused_moe/layer.py / fused_moe/config.py /
   device_communicators/all2all.py / prepare_finalize/naive_dp_ep.py)

Plus commit pin at L3 + L1199 ("98661fe") and explicit "mirrors Ch07/Ch08"
hook framing at L16. This is the **third instance of the "no class X" framing
pattern** (Ch07 "no radix tree" → Ch08 "no class TensorParallel" → Ch09 "no
class ExpertParallel") — graduates from Ch07/Ch08 series convention to a
**recognized chapter pattern** for outline-vs-source-mismatch chapters.

**§9.4 outline reframe applied** (gate 7) — the outline subsection "Expert Load
Balancing Loss 的梯度回传" was identified as a training-aux-loss topic that
doesn't exist in vLLM's inference-only codebase. The chapter pivots:
- L596: outline subsection name flagged as out-of-scope
- L598: precise scope wording — "`vllm/distributed/eplb/` 整目录里 grep 0 匹配"
  (NOT the discredited "zero matches" phrasing across all of vLLM)
- L598 + L707 + L1036: E10 false-positive carve-out (phimoe.py
  `router_aux_loss_coef` as stored config attribute; vision.py
  `get_load_balance_assignment` name collision — neither is expert routing aux
  loss)
- L602-L608: training-aux-loss sidebar (Switch Transformer L_balance, GShard,
  DeepSpeed-MoE) for grounding
- L600 + L610: pivot to inference-time response
- L614 + L1182: `EplbState` cited at `eplb_state.py:L210`
- L617 + L716: separate `_EPLB` group at `parallel_state.py:L1700-L1719`
- L616 + L693 + L715: redundant experts mechanism
- L617 + L693 + L709: logical→physical reshuffle
- L709: precise wording "**MoE 推理路径里没有 aux-loss 计算**"

This validates outline-as-topic-not-class-contract on a **third axis** —
not just "class doesn't exist" (Ch07/Ch08) but "training-time concept doesn't
exist in inference codebase". Outline JSON unchanged in all three cases.

**5 framing tips from tester applied surgically with three-anchor verification**
(gate 8 — hook + body + recap):
1. **Tip 1: Trap-G qualifier "当且仅当 renormalize=False"** — L143 (renormalize=True
   两条路径代数等价); L149 (renormalize=False 才不等价); L1046 (Trap-G recap with
   explicit qualifier). Three anchors; reader cannot write a buggy alternative
   implementation thinking they preserved the math under the wrong renorm flag.
2. **Tip 2: "MoE inference paths" wording, not "zero matches"** — L598 (`vllm/
   distributed/eplb/` 整目录 grep 0 匹配 + E10 carve-out); L709 (MoE 推理路径里没有
   aux-loss 计算); L1034 same; L1037 lists 4 negative tests. Cleanest implementation
   of Tip 2 across Ch07/Ch08/Ch09 (reviewer-noted).
3. **Tip 3: `mem_per_rank ∝ 1/(ep × tp)` LEADS §9.5** — L744 section title IS the
   formula; L750-L753 formula block at very top; L771 `memory_per_rank_MiB` code
   as derivation. Memory math leads abstract claim, not the other way around.
4. **Tip 4: α-β ratio = 2.000 framed as model tautology** — L924 ("E12 caveat:
   这个比值 2.000 是模型的恒等式，不是测量结果"); L938 (verbatim 复述 caveat);
   L1018 + L1020 (Trap-B recap with tautology framing). Triple-anchored — declared,
   explained mechanically, restated in trap recap. No reader can mistake α-β =
   2× as empirical measurement.
5. **Tip 5: 3-tuple → 2-tuple signature diff** — L449 (source returns
   `global_num_experts, None, None  # 3-tuple`); L475 (impl returns
   `(global_num_experts, None) # 2-tuple, 见 E13`); L491 (full paragraph
   explaining E13 + AITER `expert_mask` simplification). Reader sees the
   simplification is intentional, not implementation drift.

**7 language-trap callouts** in §9.7 (gate 9) following Ch06/Ch07/Ch08 lineage
style "claim → 错 → 为什么 → 源码证据 → Demo/测试":
- Trap A (L1010-L1014): "EP=N gives N× capacity" — wrong, max/mean=3.251
- Trap B (L1016-L1020): "All-to-all = all-reduce/2" — model tautology not empirics
- Trap C (L1022-L1025): "Experts independent so EP free" — shared experts +
  routing concentration
- Trap D (L1027-L1031): "EPLB free runtime bolt-on" — separate group + supports_eplb
  + round-robin ban
- Trap E (L1033-L1037): "Aux loss balances vLLM experts" — with E10 carve-out
- Trap F (L1039-L1043): "FusedMoE.forward always dispatch→experts→combine" —
  dp_size>1 AND use_ep gate
- Trap G (L1045-L1049): "Top-K then softmax = softmax then Top-K" — with
  renormalize=False qualifier

7 traps **exceeds** the v6 floor of ≥5; tester's brief had 7 candidates and
all 7 made it into recap. Plus per-section cross-references where each trap
originates (Trap-G threaded through L143/L149/§9.1.2 derivation; Trap-A
through §9.3.3 placement table; Trap-B through §9.6.2 α-β derivation; Trap-D
through §9.4.2 _EPLB group; Trap-E through L598/L707).

**Forward-pointers** to Ch11/Ch15+/Ch27 wired (gate 10):
- Ch11 (DCP/PCP) at L977-L979, L1190: mesh extends to 5D `(pp, pcp, dcp, dp, tp)`;
  `_EP ⊥ _DCP`
- Ch15+ (model zoo) at L981-L984, L1191: Llama-3 dense → `_EP is None`; Mixtral,
  DeepSeek-V2/V3, Qwen3-MoE deep dives
- Ch27 (DeepSeek-V3.2) at L986-L989, L1192: `e_score_correction_bias`/noaux_tc
  motivation; DeepEP IBGDA kernel internals; `policies.py` bin-packing solver

**Knowledge appended**: E01-E10 (implementer-supplied) + E11-E17 (tester-added,
including E10 refinement post-tester) + E18-E21 (writer-added) + E22-E24
(reviewer-added) = **E01-E24 in `knowledge/modules/expert-parallelism.md`**.
Note: E11-E13 appear duplicated in the file (implementer's E11-E13 plus tester's
E11-E15 with overlapping labels) — **flag for next compaction pass**. Newly added
this delivery:
- E18 (writer): inline `\frac` 不能塞复杂分式 — pedagogical formula rendering
- E19 (writer): trap recap "claim → 错 → 为什么 → 源码证据" 模板更结实
- E20 (writer): chapter-level reframe must be declared in opener hook, not buried in §X.4
- E21 (writer): single-process EP simulation must verbatim-quote honest-demo caveat
- E22 (reviewer): framing tip verification requires three-anchor check (hook + body + recap)
- E23 (reviewer): mapping row counts should be verified by `^|` line count, not writer claim
- E24 (reviewer): non-blocking inline-density warning acceptable IFF every inline token is single symbol

**Note**: expert-parallelism.md now has **24 facts** (E01-E24 with some duplication
overhead) — exceeds the 15-fact compaction trigger by 9. Combined with Ch08's
tensor-parallelism.md (19 facts), Ch07's prefix-cache.md (17 facts), and the
known `_parse_module_file` returns-[] bug in scripts/learn.py, the manual
workaround remains in use. **Fourth chapter** to demonstrate compact() brokenness
in succession (Ch07: 17 → Ch08: 19 → Ch09: 24, plus pre-existing scheduler.md
12-after-2-compactions). Escalation tracked as **P2-2 in
system-improvements.md** — implement working `learn.py compact` so chapter
knowledge files don't grow unbounded. Now blocking automated knowledge hygiene
across 4 chapters.

**Minor non-blocking observations** (do NOT trigger REVISE; reviewer noted for
writer's future awareness):
1. L491 explanation of E13 2-tuple/3-tuple — reviewer praised as "excellent
   surgical execution of Tip 5 — explicitly tells the reader '是有意的简化，不是
   bug' so they don't assume implementation drift".
2. L598 grep specificity — reviewer praised the two-tier wording (precise scope
   + false-positive carve-out) as the cleanest Tip 2 implementation across
   Ch07/Ch08/Ch09.
3. L924-L938 tautology framing — reviewer praised triple-anchoring as
   "no reader can mistake this for empirical measurement".
4. L608/L754/L773/L918 inline-density warnings (4 non-blocking) — reviewer
   accepted under E24 (single-symbol inline rule).

Source pinned at vLLM commit `98661fe`. Snapshot location TBD (matching prior
pattern `trace/snapshots/{N}-{slug}/v6-2026-05-07/`).

## Why it matters

**SIXTH chapter under v6 standards** — cadence holds at N=6. Critically, Ch09 is
the chapter where the **source surface broadens significantly** while quality
holds. Three structural ways v6 is now strictly better at N=6 than at N=5:

1. **Source surface scales without losing focus.** Ch08: 8 source files (Megatron
   TP); Ch09: **10 source files** (EP routing + EP placement + EP groups +
   all-to-all backend registry + EPLB state machine + 2 reference model sites).
   Two new code regions enter the book: `vllm/distributed/eplb/` (entire
   subdirectory not seen in any prior chapter) and `vllm/model_executor/layers/
   fused_moe/router/` (the routing kernel registry). The 5-step rhythm and
   two-tier mapping survive the breadth — every section opens with a specific
   file:line, every file gets a mini-mapping table, the 49-row master indexes
   the chapter. Pattern from Ch08 (8 files) generalizes to Ch09 (10 files)
   without dilution.

2. **Test breadth scales by parametrisation, not hand-writing.** 204 tests is
   a 42% increase over Ch08's 144 — but no test is busywork. Coverage by module
   (impl-notes §1.5): routing 28, expert_map 24, ep_groups 27, all2all_baseline
   20, fused_moe_block 30, eplb 32, mixtral_vs_deepseek 16, integration 18,
   smoke 9. Parametrising `ep_size ∈ {1,2,4,8}` across routing + expert_map +
   integration is the lever (same Ch08 trick on a new axis). Plus two routing-
   path families (Mixtral fused_topk + DeepSeek grouped_topk) at every test
   level — that's a 2× coverage multiplier without doubling test code.

3. **Mapping density compounds through the breadth.** 151 rows = 49 master + 39
   mini + ~63 helper. The two-tier pattern (K15 from Ch07) now scales to a
   chapter walking 10 source files. Each mini-table anchors to its section's
   source walk; master is a chapter-level index. Reader can find any cross-
   reference in O(1) per-section, then trace the section in detail. Ch08
   (122 rows / 8 files) → Ch09 (151 rows / 10 files): density per source file
   stays ~15 rows/file. Pattern is **load-bearing for breadth**, not just a
   nice-to-have.

Beyond metrics, four patterns reproduce that lock in v6 robustness:

4. **The "no class X" framing pattern reproduces (N=3).** Ch07 "no radix tree" →
   Ch08 "no class TensorParallel" → Ch09 "no class ExpertParallel/MoEParallel/
   TopKGate". Three structural anchors (title + hook + §X.2 body) every time.
   **Now graduates from "Ch07/Ch08 series convention" to a recognized chapter
   pattern** for any chapter where outline subsection names imply a class that
   doesn't exist in source. Ch10 (multi-token-prediction, "class
   MultiTokenPrediction"?) is the next likely candidate — would establish N=4.
   **STILL not eligible for wisdom promotion** per strict 2+ INSTANCES rule —
   this is intra-vllm-instance N=3, not cross-instance.

5. **§9.4 introduces a NEW reframe sub-pattern** — "training-time concept
   doesn't exist in inference codebase". Distinct from Ch07/Ch08 which were
   "specific class doesn't exist". §9.4 reframes "Expert Load Balancing Loss
   的梯度回传" (training-only mechanism) as "EPLB runtime statistical rebalance"
   (inference-time response). Same outline-as-topic-not-class-contract logic,
   different axis. Validates rule #6 on a third dimension. If Ch10/Ch11/Ch12
   surface similar training-vs-inference outline mismatches, this becomes a
   reframe sub-pattern of its own.

6. **Tester framing-guidance loop reproduces (N=4).** Ch06 → Ch07 → Ch08 → Ch09.
   Each tester's surgical tips applied by writer and verified by reviewer with
   explicit three-anchor (hook + body + recap) check (E22). Ch06: P05 sweep-pair
   language + K18 invariant-first; Ch07: no-asymptotic-faster + (N-1)×K-formula
   + chain-break-is-THE-invariant; Ch08: 1-AR-per-pair + α-bound-first +
   K17-OR-skip + bias-on-rank-0 zero-weight + MergedColumn bug story; Ch09:
   Trap-G renorm-flag qualifier + MoE-inference-paths wording + mem-formula-
   leads + α-β-tautology + 3-tuple-vs-2-tuple. Four chapters in a row —
   testers consistently produce load-bearing narrative-shaping guidance, and
   writers/reviewers consistently apply it with three-anchor discipline.
   **Operationally reproducible enough that future testers can be expected to
   produce explicit framing tips of this caliber in their handoffs.**

7. **APPROVED-at-cycle-1 discipline (K17/T19) holds across all 6 v6 chapters.**
   Ch04, Ch05, Ch06, Ch07, Ch08, Ch09 all 1-cycle. Pipeline cost is now
   predictably low when writer pre-runs both linters AND reports outputs in
   handoff. v6 Ch08 set the bar at "lint perfectly clean (0 blocking + 0
   non-blocking)"; Ch09 retreats slightly (4 non-blocking single-symbol-inline
   warnings) but still APPROVED — non-blocking-warning tolerance for
   single-symbol-inline is now codified in E24. The bar is **"0 blocking AND
   every inline token is single symbol"**, not "0 non-blocking warnings".
   This is a minor calibration of Ch08's "0/0" claim — Ch09 shows that 4
   single-symbol non-blocking warnings is acceptable.

8. **Outline-as-topic-not-class-contract** (rule #6 from session pause) is now
   operationally proven on **THREE chapters** (Ch07 radix-tree, Ch08
   TensorParallel-class, Ch09 ExpertParallel-class + Expert-Load-Balancing-Loss-
   training). Outline JSON unchanged in all three cases; chapters honestly
   represent source reality. This protects outline JSON stability while letting
   chapters frame absent topics as "why isn't this here?".

9. **No new framework bugs surfaced this chapter.** The known framework bugs
   (scripts/learn.py `_parse_module_file` returns [] → compact() non-functional;
   append-mode produces malformed double-prefix headings) remain open with
   manual workarounds. expert-parallelism.md hitting 24 facts (9 over the
   compaction trigger) is now the **fourth chapter** to demonstrate compact()
   brokenness — escalating from "annoying" to "blocking automated knowledge
   hygiene" with prefix-cache.md 17 + tensor-parallelism.md 19 + scheduler.md
   12-after-2-compactions + expert-parallelism.md 24. **P2-2 must be implemented
   before Ch12** or knowledge files become unmaintainable.

Cadence_holds_at_n6 unlocks the inference: v6 is robust enough to walk **the
broadest source surface yet** (10 vLLM modules including new EP territory) in
a single cycle. Pipeline can plan Ch10 (multi-token-prediction) → Ch13 (prefix-
cache-pooling) on this basis. The "no class X" pattern is now firmly
established and the §9.4 training-vs-inference reframe extends rule #6 to a
new dimension.

## What to remember

Reviewer-2 APPROVED in 1 cycle. Both linters PASS — formula 0 blocking + 4
non-blocking single-symbol-inline warnings (E24-acceptable); source-grounding
all PASS. **204/204 tests pass** at commit 98661fe (42% test count increase
over Ch08's 144). **1204 lines, 7792 words, 151 mapping rows** (49 master + 39
mini + 63 helper — two-tier per K15 with broadest source surface yet). **10
vLLM source files cited** (parallel_state, all2all, fused_moe/layer,
fused_moe/config, fused_topk_router, grouped_topk_router, naive_dp_ep,
eplb_state, mixtral, deepseek_v2) — 66 REFERENCE comments across 7 impl modules.
**§9.2 "no class ExpertParallel" reframe** applied successfully — third in series
after Ch07 (radix tree) + Ch08 (TensorParallel) — graduates from intra-instance
convention to recognized chapter pattern. **§9.4 outline reframe** applied with
NEW sub-pattern (training-time concept → inference-time response, distinct from
Ch07/Ch08 "specific class doesn't exist" axis). 5 framing tips from tester
applied with three-anchor verification (hook + body + recap, per E22). 7
language-trap callouts in §9.7 (A-G) plus per-section cross-refs — exceeds v6
floor of 5. Forward-pointers to Ch11/Ch15+/Ch27 wired. Knowledge E01-E10
(implementer) + E11-E17 (tester, E10 refined post-tester) + E18-E21 (writer) +
E22-E24 (reviewer) — **24 facts total in expert-parallelism.md, 9 over
compaction trigger**; flag for P2-2 system-improvements.md (fourth compact()
failure demo, now blocking). **Cadence_holds_at_n6** with broadest source
surface yet — v6 robust at N=6; not just holding but actively scaling source
breadth (10 files vs Ch08's 8) and test breadth (204 vs 144) without dilution.
Pipeline pattern of pre-run-linters-AND-tester-tips-in-handoff confirmed across
4 consecutive chapters. **Brief-on-approval discipline triggers Ch10
(multi-token-prediction) brief immediately** — likely fourth "no class X"
candidate (no `class MultiTokenPrediction` in vLLM).
