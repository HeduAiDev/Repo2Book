# Prefix Cache Knowledge

---

## K01–K05: [COMPACTED 2026-05-07] Prefix-cache foundation — hash-not-radix, append-only, chained-hash, max_hit=N-1, lazy eviction

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Compacted by**: book-editor (manual; `learn.py compact` non-functional — `_parse_module_file` returns `[]`)
**Archive**: `knowledge/archive/prefix-cache-20260507-k01-k05.json` (full original text, 5 facts)
**Access count (combined)**: 0 at compaction time, but ALL FIVE are heavily externally cited in artifacts/07-prefix-cache/ (narrative, tests, reviews, impl-notes — see archive `external_citations` for per-fact citation lists).
**Citation anchors preserved**: The five `### K01:` / `### K02:` / `### K03:` / `### K04:` / `### K05:` headings below ALL survive heading-level grep. Test files referencing these IDs in docstrings remain stable.

### K01: Hash table, not radix tree — flat dict
`vllm/v1/core/block_pool.py:L34-L127`. The prefix cache is `cached_block_hash_to_block: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]`. Per-block O(1) lookup; the chained-hash invariant (K03) gives the "tree property" (miss at K → miss at K+1) without the tree. Demo §2 shows hash table is ~4.6× faster than a hand-rolled RadixTree on realistic prompt lengths. Don't say "vLLM uses a radix tree" — common misconception (likely from SGLang's RadixAttention literature). The 4.6× gap is Python interpreter overhead (K12), not asymptotics.

### K02: Append-only block_id assignment — no deduplication
`block_pool.py:L48-L52` NOTE: *"we don't check if there is already an identical block in the cache. This is because we want to make sure the allocated block IDs won't change so that block tables are append-only."* Why: PagedAttention reads block_table mid-decode; dedup re-pointing a request's block_id would silently corrupt GPU output. Cost: `BlockHashToBlockMap` has a dict-of-blocks fallback branch (`block_pool.py:L83-L86`) for the rare collision where two physical blocks share one hash key.

### K03: Chained hash invariant — H_k depends on H_(k-1) transitively
`kv_cache_utils.py:L539-L566` (`hash_block_tokens`). Each block hash includes the parent block's hash in its payload. Hence: if H_k matches across two requests, `tokens[0..(k+1)*B]` must be byte-identical. This lets `find_longest_cache_hit` (`single_type_kv_cache_manager.py:L473-L483`) `break` on the first miss — comment there: *"if a block hash is not in cached_block_hash_to_id, the following block hashes are not computed yet for sure"*. Tests must NOT assume any non-leading block can hit independently. K11 chain-break test pins this load-bearingly.

### K04: max_cache_hit_length = num_tokens - 1 to preserve logits
`kv_cache_manager.py:L208` magic line `max_cache_hit_length = request.num_tokens - 1`. Even if an entire prompt hits the cache, vLLM forces ONE token of recompute to produce logits for that position. Without this, a 100%-cache-hit request has no logits to sample → RuntimeError downstream. Small but load-bearing. Comment notes block-size alignment also forces recompute of an entire trailing block — *"removing this limitation could slightly improve performance in the future"*.

### K05: Lazy eviction inside get_new_blocks, not a separate sweep
`block_pool.py:L322-L352` (`get_new_blocks`). Eviction inline when popping LRU head — `_maybe_evict_cached_block(block)` on each popped block (L341, L347). No background thread. Why: (a) no GIL contention; (b) eviction work scales with allocation work, not cache size; (c) evict-on-pop guarantees the cache index never points to a popped block. A "cron-style sweep" would have to re-acquire the alloc lock — vLLM's design avoids that entire class of bugs.

---

## K06: Three-state block lifecycle: cached-in-use / cached-idle / uncached-idle

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: implementer
**TTL**: 2026-06-05
**Access count**: 0
**Tags**: prefix-cache, lifecycle, data-structure

block_pool.py:L391-L422 (touch + free_blocks). A cached block is in one of three states: (1) ref_cnt>0, hash set: in use; NOT in free queue, IS in cache index. (2) ref_cnt=0, hash set: idle but cached; IS in free queue, IS in cache. (3) ref_cnt=0, hash None: idle and uncached; IS in free queue, NOT in cache. State 2→1 happens via `touch()` which calls `free_block_queue.remove(block)` — that's the O(1) middle-removal payoff of the doubly-linked list (kv_cache_utils.py:L162-L183). Without the doubly-linked list, the prefix-cache fast path would be O(n).

---

## K07: Multi-group all-must-hit semantic for hybrid attention models

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: implementer
**TTL**: 2026-06-05
**Access count**: 0
**Tags**: prefix-cache, multi-group, hybrid-attention

block_pool.py:L184-L209 (get_cached_block) takes `kv_cache_group_ids: list[int]` and returns None if ANY group misses. This matters for hybrid models with full-attention + sliding-window layers in the same model — they have separate cache groups but a request needs cached blocks for ALL groups to actually skip recomputation. The for-loop at L196-L208 does the per-group lookup with early return. Single-group models (the common case) collapse to one iteration.

---

## K08: BlockHashWithGroupId packs hash + 4-byte big-endian group_id

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: implementer
**TTL**: 2026-06-05
**Access count**: 0
**Tags**: prefix-cache, encoding, group-id

kv_cache_utils.py:L53-L62 (make_block_hash_with_group_id). Same content under different KV cache groups gets distinct cache entries. Implementation: `BlockHashWithGroupId(block_hash + group_id.to_bytes(4, 'big', signed=False))`. Endianness matters — must match between insert and lookup. Helpers `get_block_hash` (L65-L67) and `get_group_id` (L70-L72) unpack via `[:-4]` and `int.from_bytes(key[-4:], 'big')`.

---

## K09: extra_keys composition: lora + mm + cache_salt + prompt_embeds

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: implementer
**TTL**: 2026-06-05
**Access count**: 0
**Tags**: prefix-cache, extra-keys, multimodal, lora

kv_cache_utils.py:L501-L536 (generate_block_hash_extra_keys). Order is `lora_extra_keys + mm_extra_keys + cache_salt_keys + prompt_embeds_keys`. cache_salt is included ONLY for block 0 (L522-L524): `[request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else []`. mm_extra_keys are tuples `(mm_hash, offset_within_block)` — distinct hashes when the same MM item appears at different positions. need_extra_keys (L373-L390) gates whether the composer is invoked at all.

---

## K10: Cached blocks survive free_request — eviction is decoupled

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: implementer
**TTL**: 2026-06-05
**Access count**: 0
**Tags**: prefix-cache, lifecycle, system-prompt

single_type_kv_cache_manager.py:L303-L318 (free) + block_pool.py:L408-L422 (free_blocks). When a request finishes, its blocks ref_cnt-- and (if 0) re-enter the free queue — but they STAY in `cached_block_hash_to_block`. A future request that prefix-cache-hits the same hash chain will `touch()` them back to in-use without recomputing. This is what makes the system-prompt-reuse pattern work: requests come and go but the system prompt's cached blocks stay alive across the request boundary. Eviction only happens later when the block reaches the LRU head AND is popped by `get_new_blocks` (lazy eviction).

---

## K11: Chain-break test — evict-block-0 must produce hit=0, not hit=N-1

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: tester (Ch07 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: testing, chain-break, invariant

The chain-break property (K03's natural consequence) is the prefix-cache fundamental: evict block i → ALL of {i, i+1, ..., end} unreachable via `find_longest_cache_hit`, even if blocks i+1..end remain in the index. The naive test temptation is to assert "after eviction, hit count drops by 1" — that would still pass if `find_longest_cache_hit` silently switched to exhaustive search and skipped the evicted block.

**Correct test pattern** (used in `test_chain_break_after_evicting_first_block`):

```python
# Cache 2 blocks for request A, then evict block 0.
mgr.evict_block(blocks[0], chain[0])
# A NEW lookup of the SAME chain must return EMPTY.
hits = mgr.find_longest_cache_hit(chain, max_length=8)
assert hits == []   # NOT [block_id_1] — chain breaks at first miss
```

The complementary "evict middle, head still hits" test (`test_evicting_middle_block_keeps_prefix_hit`) confirms the asymmetry. Both tests together pin the scan-and-stop semantic load-bearingly.

For testers on Ch08+/distributed prefix cache: the chain-break property must survive cross-rank coordination. Replicate this test at the distributed boundary.

---

## K12: Hash-vs-radix speedup is driven by Python interpreter overhead, not asymptotics

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: tester (Ch07 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: testing, micro-bench, performance, language-trap

The 4.6x speedup of hash table over Trie/RadixTree (demo §2) is NOT primarily an O(1) vs O(L) story. Both Trie and RadixTree are O(L) per token in asymptotic analysis, and the chained hash table is also O(L/B) per query (one dict lookup per chained hash). The asymptotic gap is just 1/B (block size), typically 16x — not 4.6x.

The actual gap comes from **Python interpreter overhead per node walk**: Trie/RadixTree traverse Python `dict[int, Node]` per token — each step costs a Python attribute access, a dict lookup, and a function call frame. The hash table does ONE Python `dict.get()` per chained hash — that's a single C-level call with `__hash__` precomputed. For B=16, the chained hash does ~16x fewer Python-level operations than per-token Trie walking.

For testers: when CI-stable speedup tests are needed, assert `hash_time < tree_time` (not a fixed ratio). The 4.6x demo number is reproducible at large N (1000 prefixes × 5000 iterations), but smaller-N tests show variance from 2x to 6x. Used in `test_hash_table_faster_than_trie` and `test_hash_table_faster_than_radix_tree` — both use the strict `<` assertion, no ratio.

For writers: do NOT say "hash table is asymptotically faster than radix tree". Both are O(L) per query. The right framing is "hash table avoids Python interpreter overhead for tree traversal" — implementation cost vs algorithmic cost.

---

## K13: KV savings formula — (num_reqs - 1) * sys_prompt_blocks scales linearly

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: tester (Ch07 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: testing, kv-savings, sweep-design

The demo §4 78% claim isn't magic — it's `(N-1)/N * sys_blocks / (sys_blocks + user_blocks)`. For N=50, sys=32, user=8: `49/50 * 32/40 = 0.98 * 0.80 = 0.784`. Verified at three sweep points in `test_savings_scale_with_shared_prefix`: (10, 4)→36 saved; (20, 8)→152 saved; (50, 32)→1568 saved. All exact.

For writers framing the headline number: "78% KV savings" is workload-specific. The portable claim is "savings = (N-1) × sys_blocks", which lets the reader plug their own N and sys_prompt size. Don't lead with 78% — lead with the formula, then plug demo numbers.

For testers: pin both the exact arithmetic for the demo workload AND the formula generalization across at least 3 sweep points. A test that only pins 1568 would still pass if the underlying algorithm got a worse-than-linear scaling bug.

---

## K14: lint_formulas "too many inline" is NON-BLOCKING but flags real density

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: writer (Ch07 v6 narrative)
**TTL**: permanent
**Access count**: 0
**Tags**: writing, lint, formula-style

`scripts/lint_formulas.py` flags 4 issue classes as BLOCKING (`\text{}`, `\boxed{}`, `\tag*{}`, `\frac` in inline `$...$`, `$$` on content line) and 2 NON-BLOCKING warnings: "too many inline formulas in one paragraph" (>2) and "complex inline formula >40 chars". The non-blocking warnings still indicate readability risk — proof paragraphs that pile up `$H_k$, $H_{k-1}$, $\mathrm{tokens}$` symbols 7+ in one bullet read like APL. Mitigation that worked on Ch07 §7.1.2: keep ≤2 inline formulas per bullet, and **render mid-proof variable references as plain text** ("第 0 块 hash" instead of `$H_0$`) when the symbol isn't doing math work. Block formulas (`$$ ... $$`) for the headline equation, prose for the chain of consequences.

For writers on math-heavy chapters: do a dedicated lint pass after first draft and refactor any paragraph the linter flags. Even non-blocking warnings are signal.

---

## K15: Two-tier mapping table layout — main §X.6 + per-section mini-tables

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: writer (Ch07 v6 narrative, lifted from Ch06)
**TTL**: permanent
**Access count**: 0
**Tags**: writing, source-mapping, structure

When a chapter's source surface is broad (Ch07 cites 5 source files, 60+ `# REFERENCE:` comments), one giant mapping table at chapter end becomes opaque. Pattern that worked: **main 27-row mapping at §7.6.3 + mini-tables (5-6 rows each) at §7.1.5 (block_hash) and §7.4.5 (cache_blocks lifecycle)**. Total Ch07 markdown rows ~72 (incl. demo numerics tables, 4-variant table). Reader can read mini-tables in section context, then consult §7.6 main as the index.

Density floor: chapters with ≥4 source files should adopt two-tier; chapters with 1-2 source files stay with one main table (Ch04 13-row, Ch05 21-row both single-table; Ch06 introduced two-tier at 40 rows). Ch07 maintains the v6 pattern with 27 main + 12 mini = 39 row two-tier structure (counting only "our code | source | change" rows, excluding heading rows).

For reviewers: prefer asking "is the mapping discoverable from the section the reader's currently in?" over "is the mapping table long enough?". A 50-row table at chapter end nobody reads is worse than a 5-row mini-table at §X.3 anchored to that section's source walk.

---

## K16: Hard-gate review checklist for prefix-cache style chapters

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: reviewer (Ch07 v6 review pass, 2026-05-06)
**TTL**: permanent
**Access count**: 0
**Tags**: reviewing, hard-gates, checklist, language-trap

When reviewing a chapter where the source structure is a common misconception target (Ch07: "vLLM uses radix tree" — false; future chapters likely have similar traps for "vLLM uses XYZ standard pattern" — verify on the actual source), the hard-gate checklist must include:

1. **Linters re-run by reviewer, never trusted from writer claim.** Run `lint_formulas.py` and `lint_source_grounding.py` yourself. Writer can be wrong about what they ran — verify.
2. **Demo numerics verbatim grep.** Ground-truth numbers in `tests/test-report.md` must appear in narrative without rounding (9.3% not 9%; 4.65× not "~5×"). Grep each one.
3. **Formula leads numerics.** General formula must appear BEFORE the workload-specific value (e.g., (N-1)×K block-math derived BEFORE plugging in 78%). K13 anti-pattern.
4. **Language traps explicitly enumerated.** A misconception-heavy chapter should have a dedicated trap recap section with each trap framed as "claim → 错 → why → source-evidence". Ch07 §7.6.4 is the template.
5. **Source-files ≥ N test.** impl-notes "Source Analysis" lists files; verify each appears with `:Lxxx` reference in narrative (≥1 occurrence per file).
6. **Forward-pointer test.** Specific Ch numbers (Ch12 / Ch13 / Ch23 etc) must be wired so cross-chapter references compile across the book.
7. **5-step rhythm spot-check.** For each major section §X.1-§X.5, verify all 5 of: source-open / what+why / theory / our-impl / source-diff. Missing any = REVISE.

For Ch08+ reviewer: when the writer claims "I ran the linter and it passed", run it yourself — they may have run it on a stale draft or missed a section.

For Ch07 specifically (verdict APPROVED at 1 cycle): non-blocking issues that did NOT trigger REVISE: 5 inline-density warnings (K14 mitigation partial); one cosmetic backtick/dollar typo at L725; trap recap headers mix Chinese label + English content. None of these block publication when all 10 hard gates pass.

---

## K17: APPROVED at cycle 1 is achievable when writer pre-runs both linters

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: reviewer (Ch07 v6 review pass, 2026-05-06)
**TTL**: permanent
**Access count**: 0
**Tags**: reviewing, cycle-count, writer-discipline

Ch07 went APPROVED on first review cycle. Distinguishing factor vs Ch04 (which needed REVISE rounds): writer pre-ran lint_formulas.py + lint_source_grounding.py before delivery and reported both results in the handoff. Reviewer re-ran both as gate 1 and got the same outputs (0 blocking, 5 non-blocking density warnings; all source grounding pass). When writer's claimed stats match reviewer's re-run within a few warnings, the rest of the review collapses to walking 9 conceptual gates against a clean baseline.

For book-editor / pipeline design: the standard handoff template should include "writer claims" with linter outputs verbatim. Reviewer's first action is `diff` between claimed and re-run. Match → fast review. Mismatch → REVISE pre-emptively.

For writer: invest the 30 seconds to run both linters and paste outputs into your delivery message. It's strictly cheaper than a REVISE round.

---

## K18: Outline subsection name = topic question, not class contract — verify source surface BEFORE implementer dispatch

**Module**: prefix-cache
**Chapter**: 07-prefix-cache
**Discovered by**: archivist (Ch07 v6 delivery, 2026-05-06)
**TTL**: permanent
**Access count**: 0
**Tags**: archivist, outline, dispatch-protocol, source-verification

Ch07's outline subsection #2 read "Radix Tree 数据结构详解 — 从 Trie 到压缩 Radix Tree" — but vLLM v1 has zero radix-tree classes (verified `grep -rE "class.*Radix|Trie|PrefixTree"` at commit 98661fe returns 0 matches). The outline JSON was NOT updated mid-run (per consensus protocol — would need 2nd repo / 2nd reviewer to vote, and structural change rules apply). Instead the chapter reframed §7.2 at chapter level as "为什么没用 radix tree:链式 hash 替代方案的设计权衡" — and APPROVED on first cycle.

For archivist before any implementer dispatch on a future chapter:

1. **Source-verify the outline subsection names.** For each subsection that names a structure ("X 数据结构详解", "Y 实现", "Z 类的设计"), grep for matching `class X|class .*Y|class .*Z` in source at the pinned commit. Zero matches = topic-not-class-contract case.
2. **Capture the verification in the brief.** Brief §2 must explicitly enumerate "outline says X exists; source says X does not exist; reframe at chapter level as Y". This protects the implementer from chasing absent classes.
3. **Record to state.json `outline_notes`.** Even when the outline JSON itself is unchanged, the chapter-level reframe decision is a trace-worthy event. Format: `{id, date, decided_by, chapter, outline_subsection, decision, trigger, reasoning, operational_status}`.
4. **The reframe is NOT a chapter-quality compromise** — it's a faithful reading of source. Chapters that honestly say "the textbook structure isn't here, here's what vLLM uses instead" tend to be the best chapters (Ch07 had 4 language-trap callouts and APPROVED at cycle 1).

Operational rule (carried forward to Ch08): **verify source surface BEFORE implementer dispatch**. Specifically, for Ch08 Tensor Parallelism, the outline implies a `class TensorParallel` may exist — `grep` confirmed it does NOT. TP is implemented as `ColumnParallelLinear` + `RowParallelLinear` + `parallel_state.py` collectives. Brief §2 makes this explicit so implementer doesn't search for a single TP class.

For book-editor / future pipeline: this rule is operationally proven within instance N=2 (Ch07 + Ch08 brief). Still NOT eligible for wisdom promotion per strict 2+ repos rule (this is intra-instance reproducibility). Wait until 2nd repo instance to promote.

---
