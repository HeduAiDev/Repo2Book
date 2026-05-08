# Ch11 DCP/PCP — Review Report

- **Chapter**: `11-dcp-pcp`
- **Reviewer**: reviewer (book-factory, vllm-from-scratch instance)
- **Date**: 2026-05-08
- **Cycle**: 1 (single-cycle target, mirrors Ch08/Ch09/Ch10 cadence)
- **Verdict**: **APPROVED**

---

## Hard Gates — All PASS

### Gate 1: Linters re-run by reviewer

| Linter | Result | Notes |
|---|---|---|
| `lint_formulas.py` | 🟢 No blocking issues | 15 non-blocking warnings (4 "Too Many Inline Formulas" at L126/L554/L739/L911-919, 11 "Complex Inline Formulas"). Inspected: each warning is a derivation step where the inline expression is part of a *narrative chain* (e.g. L102 spec parameters list, L412 LSE substitution, L571 α/β H100 reference numbers, L1316 mapping-table cell). Promoting all of these to block formulas would fragment the prose. Per Ch10 precedent, non-blocking is acceptable when each inline serves narrative cohesion. **Blocking gate clears.** |
| `lint_source_grounding.py` | ✓ All grounding checks passed! | Cell coverage, `# REFERENCE:` comments, mapping table rows, impl-notes file count — all pass. |

### Gate 2: Mapping table rows ≥10

**Floor**: ≥10. **Achieved**: §11.9 main table = **81 rows** (Ch10 was 206; Ch11 writer claimed 149 in dispatch — actual count of `^|` lines after `## 11.9` and before `### 11.9.1` is 81 main rows, plus 16 rows in `### 11.9.1` cross-chapter forward + `### 11.9.2` back tables, plus per-section mini-tables in §11.1.6 / §11.2.5 / §11.3.5 / §11.4.7 / §11.5.7 / §11.6.8 / §11.7.1 / §11.8 yielding total 149+ table rows across the chapter). **Floor far exceeded.**

### Gate 3: 12 source files surfaced with `:Lxxx`

| File | Anchored in chapter | First citation | Pass |
|---|---|---|---|
| `parallel_state.py` | 28 mentions, anchors L1234-L1290, L1497-L1782, L1569-L1575, L1594-L1633, L1741-L1782, L1791-L1797, L1847-L1854 | L4 (front-matter) | ✓ |
| `dcp_alltoall.py` | 41 mentions, anchors L1-L20, L39-L103, L66-L70, L72-L78, L81-L84, L89-L91, L93-L94, L96-L101, L106-L130, L134-L196, L197-L319, L320-L450, L431-L436, L448 | L5 (front-matter) | ✓ |
| `v1/attention/backend.py` | 20 mentions, anchors L685-L757, L700, L703, L705-L706, L731-L757, L751-L752, L754-L756 | L6 (front-matter) | ✓ |
| `v1/attention/backends/utils.py` | 10 mentions, anchors L820-L857, L844-L849, L851-L855 | L7 (front-matter) | ✓ |
| `v1/attention/backends/mla/flashattn_mla.py` | 10 mentions, anchors L125, L175, L196-L250, L353-L355 | L9 (front-matter) | ✓ |
| `v1/kv_cache_interface.py` | 13 mentions, anchors L150-L195, L185-L195, L195-L205, L196-L204 | L10 (front-matter) | ✓ |
| `v1/executor/multiproc_executor.py` | 12 mentions, anchors L116-L121, L985-L1004, L985-L1001 | L11 (front-matter) | ✓ |
| `config/parallel.py` | 12 mentions, anchors L115, L310-L313, L315-L321, L322-L328, L330-L342, L469-L478, L474-L478, L480-L483 | L12 (front-matter) | ✓ |
| `flashinfer.py` | 4 mentions, anchor L213 (`class BatchDCPPrefillWrapper` — the reframe anchor) | L8, L227 | ✓ |
| `mla/rocm_aiter_mla.py` | 1 mention at L1339 (mapping table) | L1339 | ✓ |
| `moe_runner.py` | 2 mentions at L248 (file roster) and L1173 (Ch09 cross-chapter) | L248 | ✓ |
| `flash_attn.py` | 1 mention at L1337 (mapping table — "FlashAttnBackend") | L1337 | ✓ |

**12/12 files surfaced**. Note: `flashmla.py` (impl-notes optional 13th file) is not surfaced, which matches the impl-notes positioning of it as an additional MLA backend variant — not required by the dispatch list. **Gate clears.**

### Gate 4: 5-step rhythm in §11.1-§11.6

Each major section opens with a source file at a specific line, bridges via "what+why", derives the principle, presents our impl, then a "差距" diff table.

| Section | Source open | What+why | Derive | Our impl | Diff table |
|---|---|---|---|---|---|
| §11.1 | §11.1.1 `kv_cache_interface.py:L195-L205` | §11.1.2 | §11.1.3 | §11.1.4 + §11.1.5 demo | §11.1.6 |
| §11.2 | §11.2.1 grep evidence + `flashinfer.py:L213` | §11.2.2 | §11.2.3 (toy: 3 transports) | §11.2.4 (3 blocks) | §11.2.5 |
| §11.3 | §11.3.1 `dcp_alltoall.py:L39-L103` | (implicit; LSE algebra) | §11.3.2 | §11.3.3 + §11.3.4 demo | §11.3.5 |
| §11.4 | §11.4.1 `parallel.py:L322-L328` (DCPCommBackend) | §11.4.2 | §11.4.3 (α-β) | §11.4.6 + demo | §11.4.7 |
| §11.5 | §11.5.1 `parallel.py:L330-L342` + `utils.py:L820-L857` | §11.5.2 | §11.5.3 (base+remainder+clip) | §11.5.4 + §11.5.5 demo | §11.5.7 |
| §11.6 | §11.6.1 `parallel_state.py:L1569-L1575` + `multiproc_executor.py:L116-L121` | §11.6.2 | §11.6.3 (reshape order) | §11.6.4 + §11.6.5 demo | §11.6.8 |

**6/6 sections**. **Gate clears.**

### Gate 5: Demo numerics verbatim

Every headline number from `test-report.md` reproduced verbatim:

| Number | Required | Found in chapter | Lines |
|---|---|---|---|
| HBM 40.0 GB → 2.5 GB at (dcp=4, pcp=4) | NOT 33.5 GB | Verbatim at L41, L63, L188-L195, L1384 | ✓ |
| 8-cell HBM table (40.0/20.0/20.0/10.0/10.0/10.0/5.0/2.5) | All 8 cells | Table at L188-L195 verbatim | ✓ |
| `42,949,672,960` (naive bytes) | Verbatim | L185 | ✓ |
| LSE max 2.106473 | Verbatim | L473, L475 | ✓ |
| Weights 0.209762/0.133496/0.405190/0.251552 | All 4 | L477-L480 | ✓ |
| max abs error 3.33e-16 | Verbatim | L64, L481, L487, L1209, L1386 | ✓ |
| associativity error 2.22e-16 | Verbatim | L482, L487 | ✓ |
| AG+RS vs A2A: 33% NCCL ops reduction | "33%" | L65, L514, L594, L656, L1388 | ✓ |
| 2.87× / 5.44× / 9.85× at dcp ∈ {2,4,8} | All 3 | L65, L590-L592, L594, L652-L654, L1118, L1388 | ✓ |
| Striped vs contiguous: 13.44× / 1.55× / 1.24× | All 3 | L66, L725, L735, L788, L801-L803, L811, L1165, L1219, L1390 | ✓ |
| 5D mesh world=16 | Demo §5 numerics | L989, L991, L1031, L1222 | ✓ |
| (tp=8, dcp=2, pcp=4) → world_size=32 | NOT 64 | L67, L1062-L1068, L1392 (and Trap D summary at L1099) | ✓ |
| `8,192` (per_rank_len at (4,4)) | Verbatim | L195 | ✓ |
| Payloads 67,108,864 / 34,078,720 / 17,039,360 / 8,519,680 | All 4 | L590-L592, L652-L654 | ✓ |
| 1036.6/360.8/190.4/105.2 µs | All 4 | L590-L592, L652-L654 | ✓ |

**`33.5 GB` does NOT appear anywhere** (grep confirmed clean). The HBM correction landed surgically. **Gate clears.**

### Gate 6: §11.2 "no class RingAttention" reframe (5th in series) — three-anchor

| Anchor | Required | Verified |
|---|---|---|
| 1 — chapter title | Names the technique by vLLM-side feature name | L1: "DCP/PCP — 没有 `class RingAttention` 的上下文并行" — names DCP/PCP, names the absence | ✓ |
| 2 — opening hook | 5-instance enumeration (Ch07→Ch08→Ch09→Ch10→Ch11) + `_DCP`/`_PCP` GroupCoordinator + module-level pattern | L15: "第 7 章用'vLLM 没有 radix tree'开篇，第 8 章 ... `class TensorParallel` ... 第 9 章 ... `class ExpertParallel` ... 第 10 章 ... `class MultiTokenPrediction`——第 11 章是这条系列的 **第五件**：**vLLM 没有 `class RingAttention` / `class StripedAttention` / `class ContextParallel` / `class DecodeContextParallel` / `class PrefillContextParallel`**" — five-instance lineage explicit | ✓ |
| 3 — §11.2 body | Full grep evidence at 98661fe + 1-DCP-prefixed-class enumeration + Liu et al. 2023 comparison | L218-L232 (grep blocks), L227 (only `flashinfer.py:213:class BatchDCPPrefillWrapper`), L232 ("outline 描述的是技术名词，源码实现的是另一种结构"), L236-L262 (Liu 2023 vs NCCL collectives toy derivation) | ✓ |

**Mirrors Ch07/Ch08/Ch09/Ch10 lineage.** **Gate clears.**

### Gate 7: §11.4 AG+RS-vs-A2A reframe (surgical correction of outline §11.3)

| Required | Verified |
|---|---|
| Cite `DCPCommBackend = Literal["ag_rs", "a2a"]` at `parallel.py:L322-L328` | L12 (front-matter), L233 (impl-notes echo at §2 reframe), L507-L515 (chapter §11.4.1 quotes the Literal verbatim) | ✓ |
| Algebra-level equivalence (D18) framing | L162-L165, L424-L425, L487-L489, L630-L640 (`simulate_a2a_combine`/`simulate_ag_rs_combine` both call `lse_weighted_combine`), L1154 (test_section_2_a2a_equals_ag_rs_combine), L1386 | ✓ |
| Transport-pattern difference framing | L518-L520 ("差别在 NCCL 调用次数和 buffer packing"), L526-L539 (AG+RS vs A2A op-by-op walk), L550 ("op 数从 3 减到 2 是数学的，payload 缩小是工程的"), L606 (production decision rule) | ✓ |

**Surgical correction landed.** Outline JSON unchanged per consensus rule; chapter calls out the rename explicitly at L518. **Gate clears.**

### Gate 8: §11.6 5D mesh reframe (surgical correction of outline §11.5 "3D 并行")

| Required | Verified |
|---|---|
| Walk reshape order at `parallel_state.py:L1569-L1582` (writer cited L1569-L1575) | L4 (front-matter at L1569-L1575), L860-L867 (chapter §11.6.1 quotes the 5D reshape verbatim), L901-L919 (§11.6.3 derives every group's reshape/transpose pattern with line refs) | ✓ |
| world_size = `tp × pp × pcp × dp` (DCP excluded) | L867-L879 (chapter §11.6.1 quotes `multiproc_executor.py:L116-L121` with ✗ no dcp), L981 (`world_topology.py::MeshConfig.world_size` excludes dcp), L1065-L1068 ("不是 8×2×4 = 64. dcp 不进乘积"), L1392 | ✓ |
| DCP folded inside TP rationale | L897 (§11.6.2: "DCP 为什么折叠在 TP 里"... "decode 时每 rank 已经在 TP-group 里复制了 Q ... 直接在 TP-group 内部分 DCP 子组、共用 TP 的 NVLink 带宽，是工程上最划算的选择"), L920-L924 (DCP不需要 transpose), L1070-L1072 (production walkthrough) | ✓ |

**Reframe landed with full source-line walk.** **Gate clears.**

### Gate 9: 5 framing tips applied surgically (three-anchor each)

| Tip | Hook | Body | Recap |
|---|---|---|---|
| **Tip 1** (40.0 GB → 2.5 GB absolute, not 16× ratio) | L41 ("40 GB ... 砍 16 倍 ... 降到 2.5 GB"), L63 (learning bullets) | L185 (`Naive total KV bytes ... = 40.0 GB`), L188-L195 (full 8-cell table with absolute GB, not just ratios), L198 (takeaway: "**16× 减少，跨过 H100 的 80 GB 红线**") | L1384 (summary point 1: "40.0 GB 砍到 2.5 GB（16×），跨过 H100 80 GB 红线") | ✓ |
| **Tip 2** ("same algebra, different transport"; both backends delegate to `lse_weighted_combine`) | L65 ("33% 的 NCCL op 削减是数学的、payload 缩小是工程的，二者叠加"), L424 ("这条式子和**通信方式无关**") | L626-L640 (both `simulate_a2a_combine` and `simulate_ag_rs_combine` call `lse_weighted_combine`), L487-L489 (max abs error = ε), L518-L520, L611-L617 (the L448 `dist.all_to_all_single` quote) | L1386 (summary point 2: "三种 transport 走同一条数学路径") | ✓ |
| **Tip 3** (1.24× near-balanced, NOT perfectly balanced) | L66 ("1.24× 不是 1.0×... rank 7 还是比 rank 0 多 24% 的 work") | L735 ("1.24× **不是 1.0×** ... **绝对完美的 1.0× 不存在**"), L809 ("demo 文本里有 'perfectly balanced'，但**章节文字必须改成 'near-balanced'**") | L1390 (summary point 4: "near-balanced，不是 perfectly balanced") | ✓ |
| **Tip 4** (total_cp_rank PCP-major formula with worked examples) | L1043-L1044 quotes `backend.py:L751-L752` verbatim | L1046-L1054 ("multiplier 是 `dcp_world_size`，**不是** `total_cp_world_size`"), L1052 (worked: pcp=1, dcp=0, dcp_world=2 → 2; pcp=1, dcp=1, dcp_world=2 → 3; pcp=0, dcp=1 → 1), L1056-L1058 (off-by-one consequence) | L1392 (summary point 5: "PCP-major，multiplier 是 dcp_world") | ✓ |
| **Tip 5** ((tp=8, dcp=2, pcp=4) → world=32, NOT 64) | L67 (learning bullets: "world_size 是 32，**不是** 64") | L1062-L1076 (full §11.6.7 walkthrough: tp=8 → 4 TP groups of 8 each → 4 DCP sub-groups of 2 inside each TP), L1099 (Trap D table cell) | L1392 (summary point 5: "world_size=32 (不是 64)") | ✓ |

**5/5 tips applied with three-anchor verification.** **Gate clears.**

### Gate 10: 5-7 language-trap callouts in §11.7

| Trap | Title | Where called out | Anchor in §11.7 |
|---|---|---|---|
| A | "DCP doubles decode throughput at dcp_size=2" | §11.7.2 (L1104-L1114) + summary table L1096 | ✓ |
| B | "PCP halves prefill latency at pcp_size=2" | §11.7.3 (L1116-L1124) + summary table L1097 | ✓ |
| C | "CP is just SP renamed" | §11.7.4 (L1126-L1128) + summary table L1098 | ✓ |
| D | "DCP must equal PCP" | §11.7.5 (L1130-L1137) + summary table L1099 | ✓ |
| E | "CP equals TP for the attention layer" | §11.7.6 (L1139-L1146) + summary table L1100 | ✓ |
| F | "Ring is the canonical implementation in vLLM" | §11.7.7 (L1148-L1154) + summary table L1101 | ✓ |
| G | "Striped is renamed Ring" | §11.7.8 (L1156-L1165) + summary table L1102 | ✓ |

**7 traps (matches Ch10 cadence).** Each has the §11.7.1 quick-reference table cell + a dedicated §11.7.{N} subsection with hard evidence. **Gate clears.**

### Gate 11: Forward-pointers to Ch12 (KV offload), Ch15+, Ch22, Ch25, Ch27

| Target | Inline pointers | §11.7.9 cross-chapter wrap | §11.9.1 forward table |
|---|---|---|---|
| Ch12 (KV offload) | L212 ("Ch12（KV offload）会被 SSD 容量再乘一倍"), L1394 ("下一章 Ch12 接着讲 KV cache offload") | L1175 | L1358 | ✓ |
| Ch15+ (model zoo) | L354, L1086 | L1176 | L1359 | ✓ |
| Ch18 (Triton attention) | L673 | L1177 | L1360 | ✓ |
| Ch22 (PD architecture) | L212, L1086 | L1178 | L1361 | ✓ |
| Ch25 (PD ratio) | L212 | L1179 | L1362 | ✓ |
| Ch27 (DeepSeek-V3.2) | L1086 | L1180 | L1363 | ✓ |

**6/6 forward chapters wired (5 required + Ch18 bonus).** **Gate clears.**

---

## Summary Statistics

| Metric | Ch11 (this) | Ch10 (prior) | Floor |
|---|---|---|---|
| Lines | 1394 | 1345 | — |
| Words | ~8124 (writer-claimed) | 8888 | — |
| Mapping rows (main) | 81 | 206 | ≥10 |
| Mapping rows (incl. mini-tables) | 149 | — | — |
| Source files surfaced | 12 | 11 | ≥5 |
| Tests passed | 402 | 311 | ≥150 |
| Lint formulas | 🟢 0 blocking, 15 non-blocking | 🟢 0 blocking | 0 blocking |
| Lint source-grounding | ✓ all pass | ✓ all pass | ✓ |
| Reframes | 3 (§11.2 / §11.4 / §11.6) | — | — |
| Framing tips applied | 5/5 | 3/3 | — |
| Language traps | 7 | 7 | ≥5 |
| Forward-pointer chapters | 6 | — | ≥5 |

---

## What's strong

1. **The 5-instance "no class X" lineage promotion is delivered cleanly.** L15 names the four prior cases and the new fifth one in a single sentence — exactly the three-anchor template Ch08/Ch09/Ch10 used. Reader sees the motif as a confirmed pattern, not a coincidence.
2. **The HBM correction (40.0 GB → 2.5 GB, NOT 33.5) is verbatim everywhere.** Grep for "33.5" returns zero hits. The 16× ratio framing leads with the absolute numbers (Tip 1) and only references the ratio as a derived quantity.
3. **The "same algebra, different transport" framing is structurally enforced** at the *code* level — `simulate_a2a_combine` and `simulate_ag_rs_combine` literally call the same `lse_weighted_combine` (L630-L640), and the test `test_section_2_a2a_equals_ag_rs_combine` is cited as bit-identical proof.
4. **The 5D mesh reframe carries the reshape order through every group construction** — L901-L919 walks TP / PP / PCP / DP / DCP and shows exactly which transpose-or-not produces which group. The "DCP doesn't add hardware" narrative (L1070-L1076) is the cleanest possible illustration of Trap D.
5. **`total_cp_rank` PCP-major off-by-one (Tip 4)** is broken out as its own subsection (§11.6.6) with worked examples — the kind of subtle invariant that silently corrupts production if missed.
6. **Source-grounding is rich**: anchors include not just file:line but per-method and per-conditional ranges (e.g. `L66-L70` for the NaN/+inf path, `L72-L78` for lse_max stability). Every claim is back-pointed.

## Non-blocking observations (no action required)

- 15 non-blocking lint warnings on inline-formula density. Each occurrence is a derivation step in a narrative paragraph (e.g. L412 LSE substitution, L739 base+remainder+clip formula, L911-919 5D mesh group construction). Promoting them all to block formulas would fragment the prose. Per Ch10 precedent these are acceptable.
- `flashmla.py` (the FlashMLA backend) is listed in impl-notes §1.1 but is not surfaced in the chapter. The dispatch list does not require it (it's marked optional via "plus any others impl-notes lists"). Three other MLA backends (`flashattn_mla`, `rocm_aiter_mla`) and the FA backend are cited; this is sufficient for the lesson about per-backend `__new__` integration breadth.
- `flashinfer.py` is cited 4× but only at `:L213` (the `BatchDCPPrefillWrapper` reframe anchor). This is intentional — the only DCP-prefixed class IS the §11.2 anchor — additional flashinfer wiring would dilute the no-class-X story.

---

## Verdict

**APPROVED**. All 11 hard gates pass. Cycles: 1.

Handoff:
- Archivist: record delivery for Ch11, update `state.json`, write Ch12 brief.
- Team-lead: Part 3 chapter 1/8 complete (Ch11 of 28); next dispatch Ch12 (KV offload) — implementer brief to follow archivist handoff.
