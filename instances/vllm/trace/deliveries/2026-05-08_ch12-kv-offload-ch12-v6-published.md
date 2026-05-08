# Ch12 KV Cache Offload v6 PUBLISHED — ninth v6 chapter, cadence holds at N=9; broadest source surface yet (22 vLLM modules); "no class X" series HONESTLY RETIRED at N=5 within vllm — Ch12 is NOT a 6th instance; new motif emerges: 4 TOPIC-level outline reframes (NVMe/LFU/attention-score/predictive absent at 98661fe); ARC-loses honest caveat verbatim across 8 anchors; two reviewers (reviewer-2 + reviewer-3) independently APPROVED — first cross-reviewer convergence in book

- **Type**: delivery
- **Chapter**: 12-kv-offload
- **Date**: 2026-05-08
- **Timestamp**: 2026-05-08T18:00:00Z
- **Agents involved**: implementer, tester, writer, reviewer-2, reviewer-3, archivist
- **User present**: False
- **Tags**: v6, kv-offload, hierarchical-storage, no-class-X-RETIRED-at-N5, topic-level-reframes, 4-topic-reframes, NVMe-2-tier-correction, LFU-LRU+ARC-correction, attention-score-block-hash-correction, predictive-REACTIVE-correction, ARC-loses-honest-caveat, megiddo-modha-2003, framing-tips, language-trap, two-reviewer-convergence, cross-reviewer-APPROVED

## What happened

**Reviewer-2 + reviewer-3 BOTH independently APPROVED Ch12 in 1 cycle (no
REVISE iterations). Both reports converged on the same verdict — first
cross-reviewer convergence on APPROVED in the book**. All **10 hard gates
pass**. Both linters PASS at the BLOCKING bar: formula linter **0 blocking
+ 2 non-blocking density warnings** (lines 158 and 205, both 3-inline-formulas
in one paragraph in alpha-beta derivation; defensible compactness because
each is a parameter-clarification line) — **major improvement vs Ch11's 15
non-blocking** (-86.7%, well within M30 calibrated band). Source-grounding
linter all PASS.

**314/314 tests pass at vLLM source commit `98661fe`**. Test count is
**33.8% below Ch11's 474** (Ch04: 48; Ch05: 74; Ch06: 97; Ch07: 83; Ch08:
144; Ch09: 204; Ch10: 311; Ch11: 474; Ch12: **314**). The drop reflects
focus shift: Ch11 had multi-fixture broad surface (DCP+PCP+5D mesh+LSE+
striped-layout); Ch12 is focused on 7 traps + ARC-honest + 18-connector
taxonomy + ≥26 verbatim numerics + alpha-beta + dry-run atomic. **314 still
exceeds the ≥250 v6 floor by 25.6%**, and 100% pass rate holds.

**Stats**: **1583 lines, 10178 words, 285 mapping rows** (142 main §12.9.1-9
+ 143 inline tables across §12.0-§12.7 → 285 grep-counted markdown table
rows total). Lines **exceed Ch11's 1394** (+13.5%); words **exceed Ch11's
8124** (+25.3%, well above the 5K v6 floor — prose density per topic is
intentionally higher because Ch12 walks 22 source files vs Ch11's 12);
mapping rows **exceed Ch11's 149 by 91%** (with the strict main-table count
of 142, still on par with Ch11). All three metrics strictly exceed v6 floors.
impl-notes "Source Analysis" §1.1 lists **22 distinct vLLM source files** —
**broadest source surface in the book to date**, exceeds Ch11's 12 by 10
files (+83%) and spans an entirely new territory: kv_offload base/factory/
reuse, cpu spec/manager, 3 cache policies (base/lru/arc), shared_offload_region,
gpu_worker, worker, 4 simple_kv_offload modules, and 7 connector v1 files
(base, factory, offloading_connector, lmcache_connector, multi_connector,
simple_cpu_offload_connector, kv_transfer_state). **81 # REFERENCE: comments**
across impl modules (cpu_gpu_worker.py 18, offload_manager.py 14,
offload_spec.py 13, policies.py 12, simple_offload_manager.py 6,
offloading_scheduler.py 5, connector_taxonomy.py 5, reuse_manager.py 3,
factory.py 1, __init__.py 1) — count is above Ch09's 66 baseline; lower
than Ch10's 151 (which was inflated by proposer-family boilerplate); above
Ch11's 78 baseline by 3.8%. Coverage is complete; nothing referenced
indirectly.

5-step rhythm verified §12.1-§12.6 by reviewer (gate 4): each section opens
with source files + `:Lxxx` (§12.1.1 grep + per-tier table; §12.2.1 ls
policies; §12.3.1 grep predict; §12.4.1 copy_backend.py:L43-44; §12.5.1
base.py:L170-660; §12.6.1 demo §5 + L170-195) → bridge → theory derive
(alpha-beta, ARC adapt eqs, prefix scan semantics, 1.3-1.5x cost model,
headroom formula) → our impl with file:line → closes with "与源码的差距"
subsection.

**§12.0 explicit "NOT a 6th 'no class X' instance" framing** —
established at FOUR structural anchors:

1. §12.0 quote-block (L28) explicitly enumerates the 5 prior chapters'
   "class 缺位" motif (Ch7 radix tree / Ch8 TensorParallel / Ch9
   ExpertParallel / Ch10 MultiTokenPrediction / Ch11 RingAttention/
   ContextParallel) and states **"N=5 的「class 缺位」母题在第 12 章不延续"**.
2. §12.0 honestly enumerates the classes that DO exist in Ch12:
   `OffloadingManager`, `CPUOffloadingManager`, `OffloadingSpec`,
   `OffloadingHandler`, `SingleDirectionOffloadingHandler`,
   `CpuGpuOffloadingHandlers`, `OffloadingWorker`, `OffloadingConnector`,
   `KVConnectorBase_V1`, `SupportsHMA`, `MultiConnector` + 18 connectors.
3. §12.0 body recap at L99 ("前 5 章的母题在 N=5 已经停了").
4. §12.10 framing-tip recap at L1084: cross-anchor that the chapter is
   honestly NOT a 6th instance.

This is **the honest retirement of the "no class X" series at N=5 within
vllm** — series stays at N=5; Ch12 is NOT artificially forced to be a 6th
instance because the chapter genuinely has many concrete classes. **CRITICAL
CLARIFICATION**: the wisdom-promotion gate of "2+ INSTANCES" remains
unmet. The retirement at N=5 within vllm is documented in
`knowledge/modules/kv-offload.md` O01 as a chapter-internal motif. Future
instances (other repo books) will determine whether this pattern is
universally promotable.

**4 TOPIC-level outline reframes applied** (gate 7 — distinct from prior
chapters' "no class X" reframes):

1. **§12.1 NVMe SSD third tier → HBM↔CPU pinned 2-tier** (L116-276):
   - Outline says "GPU HBM→CPU DRAM→NVMe SSD的访问延迟阶梯" — implies 3-tier.
   - Source reality: vLLM at 98661fe is **2-tier (HBM ↔ CPU pinned)**. Recursive
     grep of `vllm/v1/kv_offload/` for `nvme | ssd | disk | fs_offload` returns
     ZERO matches.
   - Treatment: Demo 1 includes NVMe Gen5 (14 GB/s, 1198.37 µs per 16 MB block)
     for academic context; chapter pivots to 2-tier reality with grep evidence.
   - Anchors: L122-134 (grep), L188-195 (per-tier table), §12.1.4-§12.1.5
     derivation.

2. **§12.2 LFU eviction → LRU+ARC** (L280-507):
   - Outline says "LRU/LFU/attention-score-based选择策略".
   - Source reality: `vllm/v1/kv_offload/cpu/policies/` contains
     `base.py + lru.py + arc.py`. **NO `lfu.py`**. ARC IS the production
     sophisticated alternative, NOT LFU. ARC was published by Megiddo & Modha
     (IBM Almaden, FAST 2003).
   - Treatment: Demo 2 walks LRU + ARC honestly with the **ARC-loses
     phase_shift caveat** (LRU=2.60% / ARC=14.15% — ARC LOSES by 5×).
   - Anchors: L287 (ls evidence), L344-348 (LRU touch reverse-iteration),
     L389-422 (ARC dry-run code listing), L425-427 (atomicity rationale).

3. **§12.2 attention-score-based → block-hash semantics** (sidebar L304-307):
   - Outline says "...attention-score-based选择策略".
   - Source reality: vLLM uses **block-hash semantics throughout**. There is
     NO token-level attention-statistic policy. H2O / HeavyHitter / StreamingLLM
     are research papers (NeurIPS 2023, etc.), NOT in vLLM at 98661fe.
   - Treatment: Sidebar in §12.2.1 callout: research has explored
     attention-score eviction; vLLM ships block-hash. Forward-pointer to Ch28.

4. **§12.3 predictive ML prefetch → REACTIVE block-hash prefix lookup**
   (L511-689):
   - Outline says "Prefetch——预测哪些KV block会用到，提前搬回GPU" — implies ML predictor.
   - Source reality: `OffloadingConnectorScheduler.get_num_new_matched_tokens`
     is a **REACTIVE block-hash prefix lookup**. Linear scan over
     `block_hashes` calling `manager.lookup`. NO Markov chain, NO ML predictor.
   - Treatment: §12.3 corrects "predictive prefetch" to "reactive cache
     lookup". The async-deferral path (`return None`) IS pipelining — backends
     warm cache lines while scheduler retries — but it's NOT prediction.
   - Anchors: L513-526 (grep predict/markov returns 0), L538-550
     (`_maximal_prefix_lookup` ref), L562-572 (prefix-scan derivation).

This is a **distinct motif from prior chapters' "no class X" pattern**:
- Ch07-Ch11: each chapter had ONE specific class absent at outline level.
- Ch12: FOUR specific TOPIC-level concepts absent (NVMe, LFU, attention-score,
  predictive). Outline lists CONCEPTS as bullets, source ships DIFFERENT
  CONCEPTS. The reframe is "outline concept absent, source has different
  concept" — chapter pivots topic-by-topic.

**ARC-loses honest caveat preserved verbatim across 8 anchors** (gate 5 +
new "honesty discipline" rule emerges):
1. L33 — chapter learning outcomes flagging ARC-loses
2. L105 — §12.0 hook teaser
3. L449 — §12.2.5 title "ARC 不严格优于 LRU"
4. L456 — §12.2.5 phase_shift demo body (LRU=2.60% / ARC=14.15%)
5. L490 — Megiddo-Modha 2003 Table 4 reference (partial-shift wins)
6. L1073 — §12.6 Invariant #7 "ARC 适配代价 ≠ ARC 总优"
7. L1224 — §12.7.3 Trap B nuance ("ARC pays cost of T2/B2 for partial-shift wins")
8. L1546-L1571 — §12.10 chapter summary (final framing)

This is **the strongest demonstration to date of "honest counter-intuitive
caveat preservation"** — earlier chapters had isolated honest-callouts
(Ch06 §6.5.4 1-Pareto-point-artifact, Ch10 K17 OR-skip), but Ch12 anchors
the same caveat across 8 places in the chapter, refusing the tempting
narrative ("ARC > LRU always") and citing the 2003 academic paper for the
nuanced reality. Reviewer-3 explicitly praised this as "the cleanest
honest-caveat surfacing in the book to date".

**5 framing tips from tester applied with three-anchor verification** (gate 8 —
D28 rule):
1. **Tip 1 NOT-6th-no-class-X**: Hook §12.0 quote (L28) + body L37+L99
   outline-corrects-itself + recap L1084. Three anchors PASS.
2. **Tip 2 ARC-NOT-strictly-better**: Hook §12.2.5 title (L441) + body
   §12.2.5 phase_shift data (L443-462) + Megiddo-Modha 2003 Table 4
   reference (L462) + chapter summary L1571 (cross-anchor) + L1073
   invariant #7 (cross-anchor). Three+ anchors PASS.
3. **Tip 3 Two-streams-not-2x**: Hook §12.4 title (L692) + body §12.4.2
   1.3-1.5× implementation (L712-720) + recap L1086 framing-tip table.
   Three anchors PASS.
4. **Tip 4 REACTIVE-not-PREDICTIVE**: Hook §12.3.1 grep (L513-526) + body
   §12.3.3 prefix-scan derivation (L562-572) + `_maximal_prefix_lookup`
   ref (L538-550) + recap §12.7.6 Trap E (L1166-1176) + chapter summary
   L1573. Five anchors PASS.
5. **Tip 5 connectors-NOT-interchangeable**: Hook §12.5.1 (L830) + body
   §12.5.4 18-row taxonomy (L865-899) + recap §12.7.5 Trap D (L1154-1164).
   Three anchors PASS.

All 5 tips have ≥3 anchors. D28 rule re-validated at N=9.

**7 language-trap callouts** in §12.7 (gate 9) following the
Ch06-Ch11 lineage style "claim → 错 → 为什么 → 源码证据 → demo/test ref → nuance":
- Trap A: offload ≠ swap (§12.7.2 L1108) — offload preserves logical liveness
- Trap B: LFU/attn-score absent (§12.7.3 L1123) — only LRU + ARC at 98661fe
- Trap C: PCIe-bound not free latency (§12.7.4 L1142) — alpha-beta beta-bound
- Trap D: connectors not interchangeable (§12.7.5 L1154) — 18 distinct protocols
- Trap E: reactive not predictive (§12.7.6 L1166) — block-hash lookup, no ML
- Trap F: pin not free (§12.7.7 L1178) — pinned DRAM cannot page out
- Trap G: v0≠v1 (§12.7.8 L1190) — KVConnectorBase_V1 strictly richer

7 traps **matches Ch09/Ch10/Ch11's** 7. Each follows the same 6-substructure
template per E19/M29/D28.

**Forward-pointers** to Ch13/Ch22/Ch23/Ch24/Ch27/Ch28 wired (gate 10):
- Ch13 (prefix-cache-pooling): L1045, L1057, L1534, L1577 — pool size composes
  with offload tier sizes; chapter ending §12.10 closes on Ch13 forward-pointer
- Ch22 (PD architecture): L1058, L1535 — KV-transfer protocols re-appear at
  PD-disagg boundary
- Ch23 (PD prefix-cache): L1059, L1536 — multi-tier prefix-cache lookup
- Ch24 (layerwise-connectors): L276, L1060, L1537 — per-layer KV streaming
- Ch27/Ch28 (DeepSeek): L1061, L1538 — production offload uses LMCache /
  Mooncake / NixlConnector

**Knowledge appended**: O01-O28 in `knowledge/modules/kv-offload.md`:
- O01-O15: implementer-supplied (NOT-6th-no-class-X + 4-TOPIC reframes +
  alpha-beta cost model + LRU+ARC distinction + ref_cnt=-1 sentinel +
  proactive eviction + connector taxonomy + pinned-memory + lazy registry)
- O16-O22: tester-added (factory-registration full-path quirk + lazy-load
  timing + ARC dry-run atomicity + LRU touch reverse-iteration semantics +
  reuse-manager LRU eviction quirk + complete_store success/failure asymmetry)
- O23-O25: writer-added (ARC-loses 8-anchor preservation discipline + 4-TOPIC
  reframe template stability + chapter-honest-retirement-of-no-class-X)
- O26-O28: reviewer-added (cross-reviewer convergence as new gate-passable
  signal + 8-anchor caveat-preservation as new "honesty discipline" rule +
  density-warning M30 calibration validates at 2 non-blocking)

At **28 facts**, kv-offload.md exceeds the 15-fact compaction trigger by 13
facts. **Seventh chapter** to demonstrate compact() opportunity in succession
(Ch07: 17 → Ch08: 19 → Ch09: 24 → Ch10: 30 → Ch11: 29 → Ch12: 28). **P1-1
fix is operational** (since Ch11) — recommend `python3 scripts/learn.py
compact kv-offload` to LLM-summarise oldest 5 facts (O01-O05) and bring
file under threshold.

**Two-reviewer convergence — first in book**. The dispatch ran reviewer-2
+ reviewer-3 INDEPENDENTLY (no information-share between them); both arrived
at APPROVED with the same hard-gate evidence and the same minor non-blocking
observations (50.92 ms wall-time variance, L158/L205 density warnings).
This is the **first cross-reviewer convergence pattern** in the book. It
suggests the v6 standards are now codified enough that two independent
reviewers reach the same verdict — a structural quality signal that the
review gate is reproducible across reviewer instances. Document in
`kv-offload.md` O26 as a candidate cross-chapter pattern; if Ch13 reproduces
this, it becomes a Ch13+ baseline expectation.

**Minor non-blocking observations** (do NOT trigger REVISE; both reviewers
noted for writer's future awareness):
1. **Demo 5 wall-time variance** (L978, L1241): chapter cites `50.92 ms`
   verbatim; sample re-runs produce 50.83-51.01 ms. Body softens with
   "≈ 50 ms" at L981. Test-report uses "~50 ms" loose. The other 25 demo
   numerics are deterministic. Acceptable as the single timing-dependent
   measurement.
2. **Formula-density warnings on L158 + L205**: Both within alpha-beta
   derivation. Parameter clarification ($\alpha$, $\beta$, $1/\beta$ in one
   line; $\alpha=\beta\cdot B_{be}$ + concrete numeric + KiB conversion in
   one line) is defensible compactness. M30 calibration at ≤15 holds; Ch12's
   2 is dramatically under — this is **the cleanest formula-density profile
   since Ch08's 0** (Ch08:0, Ch09:4, Ch10:11, Ch11:15, Ch12:**2**).
3. **Framing-tip recap mapping** (L1084-1088): self-mapping in the
   framing-tips table is loose, but real anchors are present (chapter summary
   L1571 for Tip 2; §12.0 enumeration + L99 for Tip 1). Recommend tightening
   in future cadence; not blocking.

Source pinned at vLLM commit `98661fe`. Snapshot location TBD (matching
prior pattern `trace/snapshots/{N}-{slug}/v6-2026-05-08/`).

## Why it matters

**NINTH chapter under v6 standards** — cadence holds at N=9. Ch12 establishes
**the broadest source surface in the book to date** (22 vLLM modules — 83%
above Ch11's 12) and **the honest retirement of the "no class X" series at
N=5 within vllm**. This is structurally significant: Ch12 chose NOT to force
a 6th instance because the chapter genuinely has many concrete classes; the
honesty here protects the framework from artificial pattern-extension. Three
structural ways v6 is now strictly better at N=9 than at N=8:

1. **Honest-caveat preservation reproduces and scales.** Ch11 had isolated
   honest-callouts; Ch12 anchors the ARC-loses caveat across 8 places in
   the chapter. The pattern: **counter-intuitive findings get preserved
   verbatim across hook + body (with academic citation) + invariant +
   trap-callout + chapter summary**. Document in kv-offload.md O23 as a
   candidate cross-chapter pattern. **8-anchor preservation is a strict
   baseline for any future chapter where the demo numerics contradict
   conventional wisdom**.

2. **Topic-level reframe motif emerges as distinct from "no class X" motif.**
   Ch07-Ch11 each had ONE class-level absence (radix tree / TensorParallel /
   ExpertParallel / MultiTokenPrediction / RingAttention). Ch12 has FOUR
   topic-level absences (NVMe / LFU / attention-score / predictive). The
   reframe template stays the same (outline says X, source ships Y, chapter
   pivots with grep evidence) but the unit changes from "class" to "concept
   bundle". This **expands the reframe vocabulary** — Ch13+ chapters can
   honestly cite either pattern. Document in kv-offload.md O24.

3. **Cross-reviewer convergence on APPROVED.** First time in the book that
   two independent reviewers reach the same APPROVED verdict on the same
   chapter. This is a **structural quality signal** that the v6 hard gates
   are now codified enough to be reproducible across reviewer instances.
   Document in kv-offload.md O26 as a candidate baseline for Ch13+. If Ch13
   reproduces this, it elevates from "interesting observation" to "expected
   structural property of the v6 review gate".

Beyond metrics, four patterns reproduce that lock in v6 robustness:

4. **Source surface continues to scale without losing focus.** Ch08 (8
   files) → Ch09 (10 files) → Ch10 (11 files) → Ch11 (12 files) → Ch12
   (**22 files**). The Ch12 surface spans entirely new territory:
   `vllm/v1/kv_offload/` (10 files) + `vllm/v1/simple_kv_offload/` (4 files)
   + 7 connector v1 files + `kv_transfer_state`. The 5-step rhythm and
   two-tier mapping survive the breadth — every section opens with a
   specific file:line, every file gets a mini-mapping table, the 142-row
   master + 143-row inline tables combine to **285 mapping rows** (91%
   above Ch11's 149).

5. **Outline-as-topic-not-class-contract pattern proves at concept-bundle
   scale.** Ch07-Ch11 reframed at class-level. Ch12 reframes at concept-level
   (4 absent topics). Outline JSON unchanged in all six cases (N=5 class-level
   + N=1 topic-level so far); chapters honestly represent source reality.
   Pattern is now stable for ≥6 instances within vllm.

6. **Tester framing-guidance loop reproduces (N=7).** Ch06 → Ch07 → Ch08
   → Ch09 → Ch10 → Ch11 → Ch12. Each tester's surgical tips applied by
   writer and verified by reviewer with explicit three-anchor (hook + body
   + recap) check, formalised as the **D28 rule** in dcp-pcp.md. Seven
   chapters in a row — testers consistently produce load-bearing
   narrative-shaping guidance.

7. **APPROVED-at-cycle-1 discipline holds across all 9 v6 chapters.**
   Ch04 through Ch12 all 1-cycle. Pipeline cost is predictably low.
   Cross-reviewer convergence at Ch12 means the cycle-1 pattern now has
   a structural quality signal beyond just "reviewer-3 reaches APPROVED".

8. **Knowledge file growth pattern stabilises around 28-30 facts at chapter
   close.** Ch10: 30, Ch11: 29, Ch12: 28. Compact threshold at 15 means
   roughly half the file gets compacted at the close-out — P1-1 fix is
   now demonstrating its value across 3 chapters in a row.

9. **Lint formula-density profile improves dramatically.** Ch11's 15
   non-blocking sat at the edge of the M30 calibrated band (≤15). Ch12's
   2 demonstrates that the warning is content-driven (not a v6 standards
   regression); the M30 calibration band of ≤15 holds with margin.

cadence_holds_at_n9 unlocks the inference: v6 is robust enough to walk
**the broadest source surface in the book to date** (22 vLLM modules
across 4 distinct territories — kv_offload core, simple_kv_offload reference,
connector taxonomy, v1 base ABCs) in a single cycle WITH cross-reviewer
convergence WITH 4-TOPIC-level outline reframes WITH 8-anchor honest-caveat
preservation. Pipeline can plan Ch13 (prefix-cache-pooling) on this basis.
The "no class X" pattern is **honestly retired at N=5 within vllm**; future
chapters use either class-level OR topic-level reframe, whichever the source
warrants. The 4-TOPIC reframe pattern at Ch12 may apply at Ch13 if the
prefix-cache-pooling outline diverges from source similarly.

## What to remember

Reviewer-2 + reviewer-3 BOTH independently APPROVED Ch12 in 1 cycle —
**first cross-reviewer convergence on APPROVED in the book**. Both linters
PASS — formula 0 blocking + 2 non-blocking density warnings (lines 158, 205,
both alpha-beta-derivation parameter-clarification compactness; **86.7%
fewer non-blocking than Ch11**); source-grounding all PASS. **314/314 tests
pass** at commit 98661fe (above ≥250 floor by 25.6%). **1583 lines, 10178
words, 285 mapping rows** (142 main + 143 inline → broader surface
representation). **22 vLLM source files cited** — broadest in book; 81 #
REFERENCE comments. **§12.0 explicit "NOT a 6th 'no class X' instance"
framing** — series **honestly retired at N=5 within vllm**; the 5 prior
chapters' motif enumerated at L28; classes that DO exist enumerated at L37;
recap at L99 + L1084. **4 TOPIC-level outline reframes applied** (NVMe→2-tier,
LFU→LRU+ARC, attention-score→block-hash, predictive→REACTIVE) — distinct
motif from "no class X" series; outline JSON unchanged; chapter pivots
topic-by-topic with grep evidence. **ARC-loses honest caveat preserved
verbatim across 8 anchors** (LRU 2.60% vs ARC 14.15% on phase_shift) with
Megiddo-Modha 2003 Table 4 reference — strongest honest-caveat preservation
discipline in book. 5 framing tips with three-anchor verification (D28 rule).
7 language-trap callouts (A-G) matching Ch09/Ch10/Ch11. Forward-pointers
to Ch13/Ch22/Ch23/Ch24/Ch27/Ch28 wired. Knowledge O01-O28 in kv-offload.md
(28 facts; recommend `python3 scripts/learn.py compact kv-offload`).
**cadence_holds_at_n9** with broadest source surface yet — v6 robust at
N=9; not just holding but actively scaling source breadth (22 files vs
Ch11's 12, +83%) without dilution. **NEW MOTIFS EMERGING**: (a) 4-TOPIC
reframe pattern (distinct from "no class X"); (b) 8-anchor honest-caveat
preservation (strongest yet); (c) cross-reviewer convergence on APPROVED
(first in book; structural quality signal). **Brief-on-approval discipline
triggers Ch13 (prefix-cache-pooling) brief immediately** — Ch13 is the
**LAST `needs_rewrite` chapter**; after Ch13 ships, all 10 v5→v6 rewrites
are done.
