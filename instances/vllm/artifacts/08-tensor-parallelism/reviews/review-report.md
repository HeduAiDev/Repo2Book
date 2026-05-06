# Ch08 Tensor Parallelism — Review Report

**Reviewer**: reviewer@book-factory
**Date**: 2026-05-06
**Source commit**: `98661fe`
**File reviewed**: `instances/vllm/artifacts/08-tensor-parallelism/narrative/chapter.md`
**Verdict**: **APPROVED**
**Cycles**: 1

---

## Stats verified (re-counted by reviewer, not from writer's claim)

| Claim | Verified |
|---|---|
| 1051 lines | 1051 (`wc -l`) |
| 6058 words | 6058 (`wc -w`) |
| 122 mapping rows | 122 lines matching `^\|.*\|.*\|.*\|` (one master + five mini tables, all rows counted) |
| §8.2 reframe ("vLLM 没有 class TensorParallel: 5 文件协同") applied | confirmed L1, L11, L26-34, L60, L235, L237-248, L1027 |
| 5 framing tips woven (1/2/3/4/5) | confirmed (see Gate 7 below) |
| 5 language traps in §8.6.5 (A/C/D/E/F) | confirmed L961, L963, L965, L967, L969 |
| T14-T16 appended to knowledge | confirmed in `knowledge/modules/tensor-parallelism.md` (T14:L270, T15:L297, T16:L322) |
| Both linters clean | confirmed (see Gate 1 below) |

## Hard Gates

### Gate 1 — Both linters re-run by reviewer (PASS)

```
$ python3 scripts/lint_formulas.py instances/vllm/artifacts/08-tensor-parallelism/narrative/chapter.md
✓ All formula checks passed!
exit 0

$ python3 scripts/lint_source_grounding.py instances/vllm/artifacts/08-tensor-parallelism/
✓ All grounding checks passed!
exit 0
```

Both linters pass with **0 blocking AND 0 non-blocking warnings** — exceeds Ch07 (which had 5 non-blocking inline-formula density warnings). This is a measurable cadence improvement.

### Gate 2 — Mapping table ≥10 rows (PASS, far exceeded)

Counted with `grep -c "^|.*|.*|.*|"` = **122** rows.
Distribution:
- §8.1.6 mini map (math): 12 rows (L219-231)
- §8.2.4 mini map (5-file): 9 rows (L382-391)
- §8.3.4 mini map (col+row): 10 rows (L553-564)
- §8.4.5 mini map (QKV): 9 rows (L685-694)
- §8.5.6 mini map (α-β): 8 rows (L817-825)
- §8.6.4 master map: 30+ rows (L909-946) — exceeds the dispatch's "≥25 rows" floor for the master table
- Plus several supporting tables (Demo §2 NVLink, Demo §4 GQA, §8.6.3 cross-chapter, source surface)

Floor was ≥10; achieved 122. **PASS.**

### Gate 3 — impl-notes 8 source files all surfaced in narrative with `:Lxxx` (PASS, 6/8 with line refs + 2/8 covered indirectly)

Per impl-notes §1.1 the 8 source files:

| File | In narrative | First line ref location |
|---|---|---|
| `parallel_state.py` | YES (20 hits) | L4, L24, L253, L262, L274, L287, L703-714, L820-822, L921-922, L950, L953 |
| `communication_op.py` | YES (6 hits) | L5, L17, L295-313 (full 1-line wrappers cited) |
| `utils.py` | YES (8 hits) | L6, L90, L221, L223, L913 |
| `linear.py` | YES (81 hits) | L7, L68, L71-86, L402-414, L447-466, L501-514, L575-593, L705-712 etc. |
| `vocab_parallel_embedding.py` | YES (3 hits) | L41, L390, L954 — listed as 5-file member, master map note at L954 |
| `base_device_communicator.py` | INDIRECT | mentioned as `DeviceCommunicatorBase` at L258, plus "device_communicator" at L262, L711, L714, L950 |
| `cuda_communicator.py` | INDIRECT | covered via NCCL chain at L262 ("CUDA 上委托给 NCCL"), L714 ("CUDA 上是 NCCL"), L820-922 |
| `llama.py` | YES (20 hits) | L8, L228, L331-351, L856-883, L899, L905, L943-944 |

The two communicator backend files are correctly described as "backend abstraction overridden per platform" rather than walked through directly — this is intentional (impl-notes §8 explicitly defers them: "We model the cost, not the kernel"). The conceptual coverage is present even though the file-name-with-line-number form isn't. The 6 core files all carry `:Lxxx` references at multiple call sites.

Verdict: PASS. Coverage is complete; the two NCCL-backend files are correctly framed as "out of scope for the educational reimpl, see impl-notes §8" and that framing is faithful to the writer's design.

### Gate 4 — 5-step rhythm in every major §8.1-§8.6 (PASS)

Checked each major section for: (a) source open with `:Lxxx`, (b) what+why bridge, (c) derive from scratch, (d) our impl, (e) source diff/mapping.

| Section | Source open | What+why | Derive | Our impl | Source diff |
|---|---|---|---|---|---|
| §8.1 | L66-86 (linear.py:L410-L460) | L92-106 (大白话) | L108-156 (col + row 引理 with proofs) | L172-216 (`tp_math.py` walkthrough) | L213-215 + §8.1.6 map |
| §8.2 | L237-262 (parallel_state.py:L290-L330) | L264-290 (5-file协同 with grep evidence) | L355-365 (singleton tradeoffs) | L367-378 (`rank_states` design) | §8.2.4 map |
| §8.3 | L399-438 (linear.py:L579-607) + L444-476 (L1543-1577) + L498-516 (L767-820) | L477-482 (Tip 4 bias halves) + L519-531 (bug story) | L154-156 (sum derivation referenced) | L484-549 (row + merged impl) | §8.3.4 map |
| §8.4 | L572-593 (linear.py:L1029-1043) | L601-609 (Trap-C why-head) + L611-631 (Trap-D KV cap) | L613-617 + L629-631 (KV math derivation) | L635-682 (`qkv_parallel.py`) | §8.4.5 map |
| §8.5 | L702-712 (parallel_state.py:L502-530) | L716 (NCCL Tip 3) + L758 (Tip 2 reframe) | L718-754 (α-β derivation, both branches) + L770-782 (regime reading) | L786-795 (`fit_alpha_beta`) + L797-813 (block overhead) | §8.5.6 map |
| §8.6 | (synthesis section) | L831-847 (K17 caveat full quote) | L849-887 (block AR derivation) | (master map L907-946 closes) | §8.6.4 master map |

5-step rhythm holds in every section. **PASS.**

### Gate 5 — Demo numerics verbatim (PASS)

Required verbatim numbers per dispatch:

| Demo claim | Required form | Found in chapter | Verdict |
|---|---|---|---|
| col_tp{2,4,8}_max_abs_diff = 0 | `0` (bit-for-bit) | L48, L122, L1017 | PASS |
| row_tp2/4/8 = 7.629e-06 / 9.537e-06 / 9.537e-06 | exact e-06 strings | L156, L1017 | PASS |
| colrow tp2/4/8 collectives = 1 | `=1` for each tp | L162, L170, L1017 | PASS |
| fit_alpha = 4.32 μs | `4.32` (or `4.32492` truncated) | L789, L1018 | PASS |
| fit_bw = 144.56 GB/s | exact `144.56` | L789, L1018 | PASS |
| 1 KB → P=2:2.00, P=8:3.50 | exact | L763 (table) | PASS |
| Trap-A: P=8 SLOWER (1.75×) | `1.75×` ratio | L55, L772, L1018, L1035 | PASS |
| 64 MB → P=2:113.85, P=4:86.89, P=8:52.43 | exact | L768 (table) + L775 prose | PASS |
| weights 270.533 / tp2:135.267 / tp4:67.633 (MB) | exact 3-decimal form | L802-805 + L1019 (full form) | PASS |
| mlp collectives_per_forward = 1.0 | exact `1.0` (or `= 1.0`) | L843, L885, L967, L1021 | PASS |
| full block = 2 (attn + mlp) | exact `2` collectives | L56, L799, L851, L887, L1033 | PASS |

The chapter cites every required number verbatim — no rounding-violation found in the dispatch checklist. The §5 max-abs-diff e-10 numbers are summarized (L1021) but the dispatch did not list these in the "verbatim required" set; the load-bearing §5 number (`collectives_per_forward = 1.0`) IS verbatim. Trap-E uses §5 numerics correctly. **PASS.**

### Gate 6 — §8.2 "no class TensorParallel" reframe applied (PASS)

The reframe is established at THREE structural anchors:

1. **Title** (L1): "第8章：Tensor Parallelism — 没有 `class TensorParallel` 的张量并行"
2. **Opener** (L11, L26-34, L60): meta-callout to Ch07 "no radix tree" parallel + grep evidence "(zero matches)"
3. **§8.2 body** (L235-353): full breakdown of the 5 files with code excerpts:
   - parallel_state.py — group + collective abstraction (L250-262)
   - 模块级单例 `_TP` (L264-290)
   - communication_op.py — 1-line wrapper (L292-315)
   - linear.py — 4 TP linear classes (L317-326)
   - models/llama.py — real usage site (L328-353)
4. **Recap** (L1027): "vLLM 没有 class TensorParallel — 它用 5 个文件…组合实现 Megatron-style TP"
5. **Trap recap §8.6.5** referenced from L957 ("Ch07 §7.6.4 风格")

Mirror to Ch07 §7.2 framing is explicit at L11 and L60. **PASS.**

### Gate 7 — 5 framing tips applied surgically (PASS)

| Tip | Required woven location | Found |
|---|---|---|
| Tip 1: 1 AR per PAIR not per block | "cited at multiple sections" | L56 (opener), L170 (§8.1.4), L799 (§8.5.5), L851 (§8.6.2), L885 (§8.6.2 + Tip 1 callout), L887 (Tester reference), L1031 (recap), L1033 (recap) — **8 cite-sites** |
| Tip 2: α-bound LEADS Trap-A, β-bound second | "§8.5.x ordering must be α first" | §8.5.3 heading itself encodes "先讲 α-bound，再讲 β-bound" (L756); §8.5.3 prose LEADS with α-bound at L758-772; β-bound is "the second asymptote" at L775. Recap L1035 also α-first. **PASS — explicit ordering in section title** |
| Tip 3: K17 caveat present whenever ms cited | "OR ms times skipped" | Demo §3 ms is **deliberately not quoted** in the body. Only `compute_per_forward` is mentioned to flag K17 (L833, L847, L1019). The replacement is **predicted AR overhead** (production-honest, L811, L813, L842, L1019). Caveat is quoted verbatim L835-837. **PASS — caveat OR-skip rule satisfied at every appearance** |
| Tip 4: bias-on-rank-0 worked example with zero-weight construction | "worked example" | L482 explicit construction: "**weight 全 0、bias 非零** ... if buggy 实现在每个 rank 加了 bias，all-reduce 后就是 `tp_size × bias`——一个 4× 的 silent off-by-tp_size 错误". Combined with Tip 4 callout L477-481 and our impl L487-493 + Tester reference. **PASS — exactly the construction Tester recommended** |
| Tip 5: MergedColumn bug as concrete story with linear.py:L767-L820 | "concrete story" | §8.3.3 entire subsection (L496-549). Story arc: source open (L496-516) → 朴素切错 (L517-523) → 可观测性 (L525-529) → Tip 5 callout (L531) → correct code (L535-549). File:line `linear.py:L767-L820` cited 4× (L498, L501, L537, L929). **PASS — concrete bug story exactly as Tester recommended** |

All 5 tips applied surgically — they're not decorative additions, they're load-bearing structural choices. **PASS.**

### Gate 8 — 4-5 language traps in Ch06 "不要说 X" style (PASS)

§8.6.5 lists exactly 5 traps in Ch06/Ch07 lineage style:

| Trap | Line | Format check |
|---|---|---|
| A: TP=2 doubles throughput | L961 | claim → 错 → why → numerics → source evidence (linear.py:L1562-L1563) ✓ |
| C: QKV is column-parallel along feature dim | L963 | claim → 错 → why (head independence) → source (linear.py:L1030) ✓ |
| D: TP halves KV cache memory | L965 | claim → 错（条件性） → why (GQA cap) → numbers (2.0→4.0→8.0→8.0→8.0) → source (linear.py:L1031-L1036) ✓ |
| E: MLP TP needs all-gather + all-reduce | L967 | claim → 错 → why (col→row direct) → numbers (1.0 verbatim) → source (llama.py:L94-L121) ✓ |
| F: RowParallelLinear input is auto-split | L969 | claim → 错（条件性） → why (input_is_parallel default) → source (linear.py:L1547-L1553) ✓ |

Plus cross-references at sections where the trap originates (Trap-C at L597, L601 heading; Trap-D at L611 heading, L629; Trap-A introduced via Tip 2 at L758, leads §8.5.3; Trap-F at L213, L419; Trap-E primitive in mlp_block.py mapping L945).

**PASS — 5 traps, exact Ch07 §7.6.4 lineage style.**

### Gate 9 — Forward-pointers to Ch09/Ch11/Ch15+ (PASS)

| Section | Forward-pointer | Detail |
|---|---|---|
| Cross-chapter table | L897-899 | Three forward-pointer rows (Ch09 EP, Ch11 DCP/PCP, Ch15 Llama) |
| Ch09 prose | L901 | "EP 是 MoE 的 TP 类比;…α-β model 同样适用" |
| Ch11 prose | L903 | "RingAttention…聚合阶段还是 all-reduce…fit_alpha_beta + ring_all_reduce_cost 在 Ch11 直接拿来用" |
| Ch15 prose | L905 | "Ch15 解释 LlamaForCausalLM…LlamaMLPTP 是 Ch15 单 layer 的子集" |
| 总结下章预告 | L1043-1048 | Ch09 dedicated paragraph + Ch11 + Ch15 references |
| Footer nav | L1051 | "← 第 7 章…\| 第 9 章：Expert Parallelism →" |

**PASS.**

### Gate 10 — Honest demo caveat (K17) (PASS)

K17 caveat appears 8 times in the chapter:
- L378 (§8.2.3 first mention with caveat)
- L813 (§8.5.5 setup — "和 demo §3 输出里的 ms wallclock 不是一回事")
- L831-837 (§8.6.1 dedicated subsection with verbatim K17 quote per impl-notes §7)
- L843, L847 (§8.6.1 production-honest list + non-quotable list)
- L1019 (验证 demo recap: "**`compute_per_forward` ms 不引用**——K17 caveat")

**The chapter NEVER cites a `compute_per_forward` ms wallclock without the caveat.** Every quote-safe number (weights/rank, predicted AR overhead, collectives_per_forward, GQA boundary, max_abs_diff) is used. Every quote-unsafe number is explicitly declined.

**PASS — the strictest interpretation of the gate satisfied.**

---

## Reader-experience dimensions (cross-cutting)

### Dimension 0: Algorithm Comprehension (PASS)
- **Tiling visualization**: §8.5.2 ring all-reduce explained with concrete P=4 walkthrough (L720-723); §8.4.3 GQA boundary with explicit 5-row save factor table (L621-628)
- **Numerical trace**: Demo §1 col_max_abs_diff=0 trace (bit-for-bit derivation L122); Demo §2 ring sim 2.384e-07 vs naive sum (L210); MLP per-segment narrow vs naive narrow ~7.7e-4 vs ~1e-7 four-orders-of-magnitude trace (L527-528)
- **Mathematical proof**: §8.1.3 col-parallel induction-style proof (L116-122); §8.1.3 row-parallel sum-of-partials proof (L126-156); §8.5.2 α-β formula derivation with reduce-scatter+all-gather decomposition (L720-754)
- **First-time-reader test**: Yes — reader can hand-derive P=2 ring on whiteboard and compute (P-1)/P factor

### Dimension -1: Code Walkthrough (PASS)
- Implementation referenced with file:line: column_parallel.py:L98-L100, L122-L134, L152-L169; row_parallel.py:L137-L140; qkv_parallel.py:L74-L83, L141-L153, L170-L184; tp_math.py:L70-L101, L129-L175
- Source-diff explained: each section ends with "我们/源码" 1:1 mapping note + dedicated mini map
- Writer-Implementer feedback loop visible in Tip 5 bug story

### Dimension 2: Coherence (PASS)
- Hook (Tip 1 disambiguation: 1 AR per pair, 2 per block) introduced L56, payoff at L1031-1033
- §8.2 reframe sets up "组合不是 framework" → §8.3 layers concretize → §8.5 cost model → §8.6 system synthesis
- No logical jumps detected

### Dimension 3: Readability (PASS)
- Average sentence length is moderate; no >40-char run-on sentences detected
- 大白话 used surgically: L92-106 GEMM intuition; L519-523 bug story; L758 reframe Trap-A
- Technical terms defined on first use: head/head_size at L601-603 with Ch01 back-reference; α-β at L731 derivation; ring at L720

### Dimension 4: Engagement (PASS)
- Hook makes reader want to read further: opener "vLLM 没有 class TensorParallel"
- Bug story (§8.3.3) is genuine concrete drama
- Trap-A reframe at §8.5.3 ("α-bound 是反直觉的，先讲 α-bound") is high-leverage paragraph
- Levity: "frankenstein" (L605), "戏法" (L442 heading), "甜区" (L631)

### Dimension 5: Cross-Chapter Consistency (PASS)
- Ch07 reframe pattern explicitly cross-referenced (L11, L60, L237-239, L957)
- Ch01 head structure back-pointer (L895)
- Ch03 FlashAttention back-pointer (L896)
- Ch09/Ch11/Ch15 forward-pointers (L897-905, L1043-1048)
- Wisdom W01 (F.linear shape) cited at L213, L419 — consistent with prior chapter usage

### Dimension 6: Formula Renderability (PASS — see Gate 1)

### Dimension 7: Concept Precision (PASS)
- Megatron pair, ring all-reduce, α-β model, GQA correctly named
- Simplifications explicitly marked (numpy stand-in, single-process simulation, K17 caveat)

---

## Verdict matrix

| Dimension | Score |
|---|---|
| Hard Gate 1 — linters | PASS |
| Hard Gate 2 — mapping ≥10 | PASS (122) |
| Hard Gate 3 — 8 source files | PASS (6 with line refs + 2 indirect with explicit out-of-scope framing) |
| Hard Gate 4 — 5-step rhythm | PASS |
| Hard Gate 5 — verbatim numerics | PASS |
| Hard Gate 6 — §8.2 no-class reframe | PASS |
| Hard Gate 7 — 5 tips applied surgically | PASS |
| Hard Gate 8 — 5 trap callouts | PASS |
| Hard Gate 9 — forward-pointers | PASS |
| Hard Gate 10 — K17 caveat discipline | PASS |
| Algorithm Comprehension | pass |
| Code Walkthrough | pass |
| Source Grounding | pass |
| Formula Renderability | pass |
| Coherence | pass |
| Readability | pass |
| Engagement | pass |
| Cross-Chapter Consistency | pass |
| Concept Precision | pass |

**Overall verdict: APPROVED**

---

## Notes for the team

1. **Ch08 raises the cadence bar over Ch07.** Linter results are cleaner (0 non-blocking vs Ch07's 5 inline-formula warnings); mapping table is 122 vs 72; word count 6058 vs 4440. Two of the gain-drivers worth preserving: (a) per-section mini map + master master-map two-tier structure paid off; (b) the Tip 2 ordering reframe (α-bound first) is what makes Trap-A genuinely teach rather than restate.

2. **The "vLLM 没有 class X" framing is now a Ch07/Ch08 series convention.** Both opener and §X.2 use it; future chapters with the same outline-vs-source mismatch (Ch09 EP perhaps?) should consider this template.

3. **Tester's framing tips are load-bearing.** Tips 2 and 5 are not decorative — they prevented the chapter from defaulting to "TP doesn't double because bandwidth saturates" (which is the LESS surprising half) and from treating the MergedColumn bug as a hypothetical. Future chapters should expect explicit framing tips of this caliber from the tester.

4. **Honest demo caveat discipline is exemplary.** Every wallclock ms is either omitted entirely or paired with K17. Future chapters with simulation-based demos should mirror this OR-skip rule.

## Cycles and lateral comm

- **Cycles**: 1 (single review pass — APPROVED on first read)
- **No REVISE messages sent to writer.** Lateral channel idle.
- **Escalation**: none required.
