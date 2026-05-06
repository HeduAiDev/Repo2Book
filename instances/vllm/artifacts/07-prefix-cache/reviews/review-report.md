# Ch07 Prefix-Cache — Review Report

**Reviewer**: reviewer@book-factory
**Date**: 2026-05-06
**Source commit**: `98661fe`
**File reviewed**: `instances/vllm/artifacts/07-prefix-cache/narrative/chapter.md`
**Verdict**: **APPROVED**
**Cycles**: 1

---

## Stats verified

| Claim | Verified |
|---|---|
| 859 lines | 859 (`wc -l`) |
| ~4440 words | 4440 (`wc -w`) |
| 72 mapping rows | 72 lines starting with `\| ` (`grep -cE '^\\| '`) |
| Two-tier pattern (main §7.6.3 + mini at §7.1.5, §7.4.5) | confirmed L150-157, L535-541, L731-759 |
| 4 language traps | confirmed L24, L132/L471, L663/L778, L780 + recap §7.6.4 |
| Demo numerics verbatim | confirmed (see Gate 5 below) |

## Linters (re-run by reviewer)

```
$ python3 scripts/lint_formulas.py instances/vllm/artifacts/07-prefix-cache/narrative/chapter.md
🟢 No blocking issues
❌ Too Many Inline Formulas (5) — non-blocking density warnings:
  L32, L86, L88, L628, L841

$ python3 scripts/lint_source_grounding.py instances/vllm/artifacts/07-prefix-cache/
✓ All grounding checks passed!
```

Both pass the BLOCKING bar. The 5 inline-density warnings are non-blocking (K14-aligned: writer applied K14 mitigation but a few proof paragraphs still pile up symbols; tolerable for math-heavy proofs).

---

## Hard gate walk

### Gate 1 — Linters (BLOCKING) — **PASS**
Re-run results above: 0 blocking formula issues, 0 source-grounding failures.

### Gate 2 — Mapping table ≥10 rows — **PASS**
72 total table rows. Two-tier confirmed:
- §7.1.5 mini-table (block_hash) — 6 rows at L150-157
- §7.4.5 mini-table (cache_blocks lifecycle) — 6 rows at L536-541
- §7.6.3 main mapping — 27 rows at L731-759
- §7.3.3 4-variant table (find_longest_cache_hit) — 4 rows at L344-349
- §7.5.4 sweep verification table — 3 rows at L657-659
- demo §1 hit-rate table at L476-481
- §7.4.3 three-state lifecycle at L492-496

K15 two-tier pattern correctly applied; baseline Ch06 had 40 with two-tier.

### Gate 3 — impl-notes source files ≥5 appear in narrative — **PASS**
impl-notes "Source Analysis" lists 5 files. Each appears in narrative with `:Lxxx`:
- `kv_cache_utils.py` — L4, L50, L53, L97, L116, L138, L154, L737-754, L774
- `block_pool.py` — L5, L16, L167, L170, L218, L233, L236, L275, L278, L490, L508, L511, L538-541, L695, L774, L740-744, L750-752
- `single_type_kv_cache_manager.py` — L6, L300, L303, L342, L346-349, L358, L446, L449, L537, L541, L748-749, L753, L776, L780
- `kv_cache_manager.py` — L7, L549, L555, L558, L574, L586, L608, L674, L755-757, L778
- `request.py` — L155, L738 (via `Request.update_block_hashes`)

All 5 files cited multiple times with line numbers.

### Gate 4 — 5-step rhythm in §7.1-§7.5 — **PASS**

Each major section opens with source location → bridge → theory → impl → diff:

- **§7.1**: 7.1.1 source open `kv_cache_utils.py:L539-L566` → 7.1.2 derive chain monotonicity lemma + proof → 7.1.3 our `block_hash.py` impl → diff (`sha256(repr())` vs `sha256_cbor`) → 7.1.5 mini mapping.
- **§7.2**: 7.2.1 source open `block_pool.py:L34-L127` → 7.2.2 radix-tree foil derivation → 7.2.3 our `prefix_cache_index.py` 3 branches → 7.2.4 K02 NOTE + invariant I1 derivation → 7.2.6 multi-group impl + K07 derivation.
- **§7.3**: 7.3.1 source open `single_type_kv_cache_manager.py:L446-L494` → 7.3.2 why-not-binary-search derivation → 7.3.3 4-variant map → 7.3.4 our impl + diff → 7.3.5 demo trace + 7.3.6 microbench analysis.
- **§7.4**: 7.4.1 source open `single_type_kv_cache_manager.py:L277-L301` → 7.4.2 partial-block math derivation (9.3% etc.) → 7.4.3 K06 three-state lifecycle table → 7.4.4 K05 lazy eviction reasoning → 7.4.5 mini mapping.
- **§7.5**: 7.5.1 source open `kv_cache_manager.py:L183-L223 + L225-L416` → 7.5.2 K04 `-1` derivation → 7.5.3 our 6-step `prefix_aware_allocate` → 7.5.4 (N-1)×K formula derivation FIRST then 78% plug-in → 7.5.5 trap-3 derivation → 7.5.6 diff table.

No section is missing any step.

### Gate 5 — Demo numerics verbatim — **PASS**

All ground-truth numbers from `tests/test-report.md` appear unchanged:

| Number | Locations |
|---|---|
| 9.3% | L37 (in objectives), L477, L484, L486, L827 |
| 49.5% | L479, L486, L827 |
| 88.2% | L481, L486, L827 |
| 373,348/s | L420, L828 |
| 80,291/s | L418, L828 |
| 77,232/s | L419, L828 |
| 4.65× | L38, L412 (header), L420, L429, L431, L436, L828 |
| 4.65× framed as Python overhead, NOT asymptotic | L412 header "4.65× 不是渐近差异"; L425-L436 explicit derivation; L431 "剩下的 4.65 倍来自 **Python 解释器开销**"; cross-checked against K12. |
| naive=2000 | L646, L830 |
| prefix-aware=432 | L647, L830 |
| saved=1568 | L633, L636, L648, L659, L830, L845 |
| ratio=0.784 (78%) | L639, L648, L830 |
| 1568=49×32 | L633 (block formula), L659 (sweep table) |

### Gate 6 — (N-1)×K formula leads §7.5 — **PASS**

§7.5.4 (L610-L660) opens with prose intro at L613 ("先讲公式，再讲数字"), then derives at L616-L628:
- Naive: `N × (K + U)`
- Prefix-aware: `(K + U) + (N-1) × U`
- Saved (block math): `(N-1) × K`
- saving_ratio formula

Demo plug-in (50/32/8) only happens at L630 AFTER the general formula. K13 anti-pattern ("先公式，后数字") explicitly stated at L651.

### Gate 7 — §7.2 radix-tree reframe applied — **PASS**

§7.2.1 (L163-L181):
- L165: "**vLLM v1 没有 radix tree**"
- L167-L177: cites `block_pool.py:L34-L127` and shows it's a `dict`
- §7.2.2 (L183-L211): `radix_tree.py` framed as "教学对照" foil that quantifies what vLLM avoided

Hook §7.0 (L24) also opens with the reframe: "整个 vLLM v1 源码树里搜不到一个 `class.*Radix|Trie|PrefixTree`（commit `98661fe` 实测 0 个匹配）"

### Gate 8 — Chain-break = THE invariant — **PASS**

- Hook L39: "看懂'链断裂'现象"
- §7.3.5 demo trace L383-L408 walks chain-break step-by-step
- L410: "**链断裂不是一个 bug，它就是 prefix cache 的核心不变量** I3"
- §7.6.1 invariant I3 framed as "chained hash 让 match length 单调"
- Closing summary L843-L847 frames I3 alongside I1, I2 as the load-bearing trio

### Gate 9 — ≥1 language-trap callout — **PASS** (4 traps, exceeds 1)

- Trap 1 (no radix tree): hook L24, §7.2.1, recap §7.6.4 L774
- Trap 2 (partial block): §7.1.3 L132, §7.4.2 L471, recap §7.6.4 L776
- Trap 3 (cache hit ≠ free): §7.5.5 L663-L676, recap §7.6.4 L778
- Trap 4 (exhaustive search): §7.3.2 L334-L338, recap §7.6.4 L780
- All 4 explicitly enumerated in §7.6.4 dedicated section "4 个语言陷阱回顾"

### Gate 10 — Forward-pointers to Ch13 and Ch23 — **PASS**

- Ch13 forward-pointers: L651 (cross-rank pooling re-uses (N-1)×K), L676 (decode block shortage), L711 (I4 cross-rank invariant), L722, L725 (dedicated paragraph), L767 (real benchmarks), L855 (final outro)
- Ch23 forward-pointers: L723, L727 (dedicated paragraph), L855
- Bonus: Ch12 (multi-group) at L146, L347, L377, L685, L721, L763; Ch15/Ch20 (variants) at L348-L349, L765
- Back-pointers Ch03/Ch04/Ch05/Ch06 also wired (L551, L719-L720, L843)

---

## Cross-chapter consistency (Dimension 5)

- Ch03/Ch05/Ch06 references match prior chapters' framing
- KVCacheBlock three-state lifecycle (K06) consistent with Ch05's BlockPool
- Chain-hash invariant builds on but doesn't contradict Ch04's continuous-batching narrative
- Cadence (5-step rhythm, mini-table at section end + main at chapter end) matches Ch06

## Algorithmic comprehension (Dimension 0)

- Tiling visualization: §7.3.5 demo trace (steps 1-5) and §7.4.3 three-state table both serve as concrete visualizations
- Numerical trace: §7.3.5 walks A/B/C/D allocations with concrete block_size=4 and explicit hit counts
- Math proof: §7.1.2 chain monotonicity lemma has explicit base case + inductive step + biconditional proof
- Quantification: hit-rate sweep, microbench (4.65×), savings formula, all concrete

## Code walkthrough (Dimension -1) — PASS

Every section shows our impl side-by-side with source. `# REFERENCE:` annotations consistent. impl-notes's 27-row mapping spine reflected in narrative.

## Source grounding (Dimension 1) — PASS

Linter clean. Every Cell has 1+ source ref. Mapping table 72 rows (≥5 floor exceeded by 14×).

## Readability + engagement — PASS

- Sentence length within target (Chinese)
- Hook (`这章要讲什么？` L14) sets up the radix-tree mystery effectively
- Levity moments: "tree 都没有，何来 tree 操作？" (L32), "这一行是本章最重要的 7 个字符" (L381)
- Narrator voice consistent

## Concept precision — PASS

Technical terms canonical. Simplifications explicitly marked (single group → Ch12-13; KVCacheBlock object → int block_id with Ch05 boundary called out).

---

## Minor (non-blocking) observations

These do NOT block APPROVED but the writer may want to know for future chapters:

1. **L725 backtick/dollar typo**: `` `(N-1) \cdot K$ 公式 `` — opens with backtick code-span but closes with `$`. Renders as raw text. Cosmetic, not lint-blocking.
2. **§7.6.4 trap headers in English**: "**陷阱 1：vLLM 用 radix tree。**" mixes Chinese trap label with English trap content (e.g., "Cache hit on partial block."). Internally consistent with the impl-notes language traps section (which is also bilingual), but readers may notice the language switch.
3. **5 inline-density warnings**: L32, L86, L88, L628, L841 each have 3-4 inline `$...$` in one paragraph. K14 says these are non-blocking but flag readability risk. Writer applied the mitigation in proof paragraphs (rendering "第 0 块 hash" as plain text); these 5 spots could benefit from one more pass but not required.

None of the above is a Hard Gate failure.

---

## Verdict

**APPROVED** — all 10 hard gates PASS, all auto-REJECT triggers clean.
Cycles: 1 (no REVISE iterations needed).

Chapter is publication-ready. Backpressure gate OPEN for archivist (Task #20).
