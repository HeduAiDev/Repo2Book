# Rehydration Brief — Ch07 Prefix Cache (Implementer)

- **Chapter**: `07-prefix-cache` (slug-aligned ID; legacy ID was `08-prefix-cache`)
- **Title**: Prefix Cache 与 APCAwareAllocation
- **Outline level**: core (Part 2)
- **Status**: `needs_rewrite` — full v6-grade rewrite per Ch04/Ch05/Ch06 baseline (cadence holds at N=3)
- **Dependencies (per outline)**: `03-flashattention-pagedattention` (PagedAttention block tables — required); plus interaction with `05-memory-management` (KVCacheManager / BlockPool)
- **Dependents downstream**: `13-prefix-cache-pooling` (cross-request sharing + pooling), `23-pd-prefix-cache` (PD-architecture variant)
- **Source pin**: vLLM commit `98661fe` at `instances/vllm/source/`
- **Brief generated**: 2026-05-06 by archivist
- **Recipient**: book-editor-2 / team-lead → forward to implementer-2 with full context

---

## 1. Project state context

- Part 1 complete (Ch01-Ch04 published). Part 2 4/9 complete after Ch07 (Ch05 → Ch06 → Ch07 in v6 sequence).
- v6 cadence holds at **N=3** — Ch04, Ch05, Ch06 all published_v6 with strictly increasing mapping density (13 → 21 → 40 crossrefs). Ch07 should match the floor without surprises.
- **Floor thresholds** (v6, confirmed at N=3): ≥5 source files in impl-notes, ≥60 `# REFERENCE:` comments, ≥10-row mapping (preferably ≥20-30), demo numerics verbatim in narrative, 5-step rhythm per major section, both linters green, 100% test pass.
- **Patterns from Ch06 to reach for** (recorded in `state.json:v6_compliance.patterns_promoted_to_baseline`): two_tier_mapping (main + per-section mini-tables when surface is broad), tester_framing_guidance (testers shape narrative through Knowledge facts), language_trap_callouts (explicit "don't say X"), honest_demo_caveats (flag what's model artifact vs vLLM truth).

## 2. CRITICAL: outline subsection #2 is misleading — there is NO radix tree in vLLM v1

**Read this section before opening any source file.**

The book-outline.json subsection 2 reads "Radix Tree 数据结构详解 — 从 Trie 到压缩 Radix Tree". **This is misleading.** vLLM v1's prefix cache is NOT a radix tree. It's a flat hash-to-block map keyed on chained block hashes:

- `block_pool.py:L34 BlockHashToBlockMap` — flat `dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]`.
- `kv_cache_utils.py:L539 hash_block_tokens()` — primitive that chains hashes: `H_k = hash((H_{k-1}, tokens[k*B : (k+1)*B], extra_keys))`. The "tree" structure is implicit through hash chains, not an explicit radix tree class.
- The whole vLLM source tree contains ZERO `class.*Radix` / `class.*Trie` / `class.*PrefixTree` (verified at commit 98661fe).

**Source of confusion**: SGLang and the RadixAttention paper popularized "radix-tree prefix cache" vocabulary. vLLM uses a different design — chained hash + flat hash table + linear scan from the head. The legacy chapter (`narrative/_legacy/chapter.md:L40-L80`) gets this exactly right with its §7.1 title literally **"链式 Hash：没有 Radix Tree 的前缀匹配"** and a clean explanation of why hash chains can replace a radix tree (next bullet).

**For the writer**: lead with this language-trap, in the spirit of Ch06's §6.3.2 "不要说 recompute 更快" callout. Suggested framing: **"vLLM v1 没有 radix tree——它用链式 hash + 平面 hash 表替代。"** Explain the math: if two requests share the first `k*B` tokens, their `H_k` values must be identical (by the hash chain's collision-resistance). A single O(1) hash lookup answers "is block k cached?", and you scan forward from the head until the first miss (because the chain guarantees all subsequent blocks are also misses). This is asymptotically equivalent to a radix-tree match and *simpler to implement*.

**For the implementer**: do NOT implement a radix tree. Implement the same data structure vLLM uses: a flat `dict[BlockHash, KVCacheBlock]` plus the `parent_hash` chain. If you find yourself writing TrieNode or RadixNode classes — STOP. The chapter has gone wrong.

The outline subsection #2 should be reread as "Block hash chain — the radix-tree alternative", and the writer should structure it that way regardless of what the outline literal says. Flag with team-lead if they push back, but the source is unambiguous: 0 radix trees at commit 98661fe.

## 3. Chapter scope (what Ch07 actually covers — and what it does NOT)

**Core question**: when 100 concurrent users hit a chatbot with the same 2000-token system prompt, how does vLLM avoid recomputing the K, V values 100 times?

**The spine of the chapter** (legacy intro nails it):
1. `kv_cache_utils.py:L539 hash_block_tokens()` — the chained-hash primitive. **THE math foundation**: `H_k` depends on parent hash, so collision-resistance guarantees no false-positive prefix matches.
2. `block_pool.py:L34 BlockHashToBlockMap` — the flat hash-to-block lookup table. Note the explicit no-deduplication design (NOTE #1 in the source: "we don't check if there is already an identical block in the cache" — block IDs must be append-only for block-table stability). This is an interview-level engineering rationale.
3. `single_type_kv_cache_manager.py:L338+L448+L513+L650+L810` — five `find_longest_cache_hit()` implementations, one per cache type (full attention, sliding window, hybrid, etc.). **Pick the simplest one** (`FullAttentionManager` at L338) and walk through it. Mention the others exist as variants.
4. `kv_cache_manager.py:L515 cache_blocks()` — the insert path: iterate through fresh blocks, hash them, register in `BlockHashToBlockMap`.
5. `kv_cache_manager.py:L?? eviction interaction` — when a cached block gets evicted, what happens to subsequent cached blocks in the chain? (The chain breaks — subsequent blocks become unreachable via prefix lookup, but their KV is still in memory until they're individually evicted. This is the W10 "ref_cnt = -1 means not ready" wisdom in action.)
6. **APCAware allocation** — the per-request flow: when a new request arrives, `find_longest_cache_hit` returns the longest matching prefix; the matched blocks get their `ref_cnt` bumped (touch); only the SUFFIX needs new allocation. This is the §7.4 / §7.5 territory.

**OUT of scope** (do NOT re-cover; reference back to prior chapters):
- PagedAttention block-table mechanics → Ch03 already covered. Reference, don't re-walk.
- KV cache memory budget / num_blocks derivation → Ch05 §5.3 (the 35148 derivation). Reference, don't re-walk.
- BlockPool internals (free queue, watermark, get_new_blocks) → Ch05 §5.4-5.5. Reference, don't re-walk.
- Cross-request sharing and pool dynamics under load → Ch13 (prefix-cache-pooling). Forward-pointer.
- PD-architecture variant of prefix cache → Ch23. Forward-pointer.

If the implementer is re-deriving block-table mechanics or BlockPool free-queue semantics — STOP. Those belong to prior chapters; Ch07 is the prefix-cache LAYER on top.

## 4. Source modules to reference (commit 98661fe verified)

| File | Lines | What |
|---|---|---|
| `instances/vllm/source/vllm/v1/core/kv_cache_utils.py` | L539+ | `hash_block_tokens()` — chained-hash primitive (THE math foundation) |
| `instances/vllm/source/vllm/v1/core/kv_cache_utils.py` | L2056 | `BlockHashListWithBlockSize` — request-side block-hash list |
| `instances/vllm/source/vllm/v1/core/block_pool.py` | L34-L100 | `BlockHashToBlockMap` — flat hash → block lookup; explicit no-dedup design (NOTE #1) |
| `instances/vllm/source/vllm/v1/core/block_pool.py` | L184+ | `BlockPool.get_cached_block()` — the lookup entry point |
| `instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py` | L338+ | `FullAttentionManager.find_longest_cache_hit()` — start here, simplest variant |
| `instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py` | L277, L1052, L1083 | `cache_blocks()` variants (Full, sliding, hybrid) |
| `instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py` | L448, L513, L650, L810, L1094 | other `find_longest_cache_hit` variants — mention exist, don't deep-dive |
| `instances/vllm/source/vllm/v1/core/kv_cache_manager.py` | L515 | top-level `cache_blocks()` (the API surface) |
| `instances/vllm/source/vllm/v1/core/kv_cache_coordinator.py` | L198-L487 | coordinator-level dispatch — multiple `find_longest_cache_hit` shims |

The implementer MUST verify these line numbers against `git rev-parse HEAD` in `instances/vllm/source/` matching `98661fe` before writing references.

## 5. Knowledge module recommendation

**Create `knowledge/modules/prefix-cache.md`** — Ch07 deserves its own module:
- Distinct from `kv-cache.md` (block structures, free queue) and `scheduler.md` (scheduling logic).
- Forward-shared with Ch13 (prefix-cache-pooling) and Ch23 (PD prefix-cache).
- Use **P-prefix IDs** (P01, P02, ...) like `preemption.md` does — DO NOT use K-prefix (K is for `scheduler.md`).
- **WARNING**: per `trace/cross-chapter/learn-py-append-id-bug.md`, `learn.py` append-mode is buggy. It will likely produce malformed `## K01: P01: ...` headings even on a fresh file. Verify after running learn.py and **immediately fix any `## [KP]\d+: [KP]\d+:` double-prefix headings to plain `## P0X: ...`** (this is a standing rule per team-lead 2026-05-06).
- Update `knowledge/INDEX.md` with the new module row when created.

## 6. Strengths to preserve from Ch04+Ch05+Ch06 (the v6 cadence)

These are confirmed precedents at N=3 — replicate them in Ch07:

1. **Demo with reproducible numerics**. Concrete proposal: a system-prompt sharing benchmark — N requests with `system_prompt_len = 2000`, distinct user suffixes of length 100. Measure: (a) hit-rate as N grows, (b) cache lookup latency vs N, (c) total KV bytes saved vs naive. Numbers must appear verbatim in narrative.
2. **5-step rhythm per major section** (Source Trail → Bridge → Theory Deep Dive → Implementation → Source Diff). Suggested 6 sections from outline reframed: §7.1 hash chain math (replaces "Radix Tree" subsection), §7.2 BlockHashToBlockMap structure + no-dedup rationale, §7.3 find_longest_cache_hit walk, §7.4 cache_blocks insert path, §7.5 APCAware allocation flow, §7.6 PagedAttention coordination + cross-request ref_cnt sharing.
3. **Source mapping table density** (Ch04: 13, Ch05: 21, Ch06: 40). Ch07 should aim for ≥20. Use Ch06's two-tier pattern if scope allows: main §7.6 mapping + mini-tables in §7.1 (hash chain → kv_cache_utils.py:L539+) and §7.3 (find_longest variants table).
4. **`# REFERENCE:` saturated** (Ch04: 65, Ch05: 61, Ch06: 60). Target ≥60.
5. **Engineering-rationale call-outs** (interview-level). Ch07 has rich material:
   - Why hash chain replaces radix tree (asymptotically equivalent, simpler) — §7.1 punch line
   - Why no deduplication of identical block contents (block IDs must be append-only for block-table stability) — `block_pool.py:L34` NOTE #1
   - Why iterate from the head until first miss (chain collision-resistance guarantees subsequent miss) — §7.3 walkthrough
   - Why `extra_keys` in hash (LoRA, multimodal — different requests with same tokens but different LoRA must NOT collide) — `hash_block_tokens` signature
   - Why the BlockHashToBlockMap can return EITHER a single block OR a dict of {block_id: block} (the union-type GC-cost optimization) — `block_pool.py:L34-L60`
6. **Language-trap callouts** (Ch06 had 4 explicit ones). Ch07 candidates:
   - "vLLM 没有 radix tree" — primary trap, lead §7.1 with this
   - "Block hash 不去重相同内容" — counterintuitive design choice
   - "Cache hit 不等于免费" — touch() bumps ref_cnt + reorders LRU; not zero-cost
   - "find_longest_cache_hit 遇到第一个 miss 就停" — readers may expect a tree-walk; explain why head-first scan is sufficient
7. **Honest demo caveats** (Ch06 §6.5.4 precedent): if the demo benchmark assumes ideal hit conditions (all requests share the same prefix), say so. Real production hit-rate distributions are bimodal/multimodal; the demo number is the optimistic envelope.
8. **Cross-chapter coherence**: front-matter back-pointer to Ch03 (PagedAttention block tables) + Ch05 (KVCacheManager); forward-pointer to Ch13 (cross-request pooling) + Ch23 (PD variant).
9. **Scope discipline** (the Ch06 payoff). §7.0 / §7.6 should reference Ch03/Ch05 by section number, not re-walk.

## 7. Implementation skeleton hints

The implementer's `implementation/` for Ch07 likely needs:
- `simple_block_hash.py` — minimal `hash_block_tokens()` matching source 1:1, plus `BlockHash` newtype.
- `simple_block_hash_map.py` — flat `dict[BlockHash, SimpleKVCacheBlock]` with `get_one_block`, `add_block`, `pop_block` mirroring `BlockHashToBlockMap`. SHOULD NOT do dedup (preserve the source's design choice).
- `simple_apc_manager.py` — minimal manager with `find_longest_cache_hit`, `cache_blocks`, `touch`, `evict`. Imports a thin SimpleKVCacheManager stub from Ch05's impl OR re-stubs the BlockPool interface (the latter is cleaner; don't entangle with Ch05 impl).
- `demo.py` — system-prompt-sharing benchmark with reproducible numerics.
- `impl-notes.md` — list ≥5 source files (kv_cache_utils.py, block_pool.py, single_type_kv_cache_manager.py, kv_cache_manager.py, kv_cache_coordinator.py).

The legacy `implementation/_legacy/prefix_cache.py` is reference-only (~300 lines from a glance). Inspect for structure but rewrite from current source.

The legacy `narrative/_legacy/chapter.md` (235 lines) is well-pitched — particularly §7.1 "链式 Hash：没有 Radix Tree 的前缀匹配". Re-read it for tone and structural guidance; the writer can lift that framing wholesale (with v6-grade source grounding upgrades).

## 8. Process / handoff requirements

Per `trace/cross-chapter/handoff-protocol-2026-05-06.md` (candidate, operationally enforced):
1. Implementer MUST `SendMessage` tester explicitly when impl is ready.
2. Tester MUST `SendMessage` writer with test-pass evidence.
3. Writer MUST `SendMessage` reviewer when narrative is ready.
4. Reviewer MUST `SendMessage` archivist with verdict + status-JSON path.
5. Each stage's `TaskUpdate` → completed BEFORE the SendMessage.
6. **NEW for Ch07** (handoff already proven N=3 within instance, still candidate per CLAUDE.md strict-2-repos): if you (implementer) finish but tester doesn't ack within ~10min, ping team-lead.

Also: **post-AFTER WORK extract**, verify `prefix-cache.md` for any `## [KP]\d+: [KP]\d+:` double-prefix headings and clean immediately. Standing rule per team-lead 2026-05-06.

## 9. Knowledge / wisdom queries before starting

```bash
# Knowledge — Ch07 builds on kv-cache + scheduler modules
cat knowledge/modules/kv-cache.md       # block structures, free queue, ref_cnt invariants
cat knowledge/modules/scheduler.md      # K10 (allocate_slots takes Request) is relevant
cat knowledge/modules/preemption.md     # preemption-vs-cache interaction (when preempt frees cached blocks)
# create knowledge/modules/prefix-cache.md (this chapter populates it for the first time)

# Wisdom for implementer (priority: debugging > architecture > testing > writing)
cat wisdom/debugging.md                 # any hash-collision / cache-staleness gotchas
cat wisdom/architecture.md              # W04 backpressure, W08 lateral, W12 pipeline
cat wisdom/testing.md                   # W02 preemption-test sizing (relevant for §7.5 eviction interaction tests), W10 ref_cnt=-1 invariant
cat instances/vllm/trace/cross-chapter/handoff-protocol-2026-05-06.md
cat instances/vllm/trace/cross-chapter/learn-py-append-id-bug.md
```

## 10. Open questions for book-editor / team-lead

1. **Outline subsection #2 wording**: should I (archivist) update `book/book-outline.json` to reframe "Radix Tree 数据结构详解" as "块哈希链 — radix-tree 的替代设计"? Or leave the outline as-is and let the writer reframe in the chapter? **Recommendation**: reframe in the chapter only. Don't touch the outline without explicit team-lead authorization (outline changes are framework-level decisions).
2. **Demo workload realism**: the system-prompt-sharing benchmark gives optimistic hit-rates. Should the demo include a "realistic distribution" comparison? **Recommendation**: stick with the optimistic envelope and add an honest §7.5.4 caveat (Ch06 §6.5.4 precedent). Don't bloat the demo.
3. **Scope of `find_longest_cache_hit` variants**: 5 implementations exist. Should §7.3 walk all 5? **Recommendation**: walk only `FullAttentionManager` at L338. Mention the other 4 variants in a single table with their use-cases (sliding-window, hybrid, etc.) but don't deep-dive — saves ~100 lines. Forward-pointer to a future "attention-variant chapter" if one ever exists.
4. **Topology**: default `linear` should suffice — Ch07 is structurally similar to Ch06 (single layer, well-bounded source surface). Reconsider `pair` only if implementer hits unknowns in `BlockHashWithGroupId` or the GroupId concept (which is multimodal/LoRA-related and might pull in Ch08+ scope).

## 11. Deliverables checklist for implementer-2

- [ ] `implementation/simple_block_hash.py` with ≥10 `# REFERENCE:` comments
- [ ] `implementation/simple_block_hash_map.py` with ≥10 `# REFERENCE:` comments
- [ ] `implementation/simple_apc_manager.py` with ≥20 `# REFERENCE:` comments (the meatiest module)
- [ ] `implementation/demo.py` with reproducible system-prompt-sharing benchmark
- [ ] `implementation/impl-notes.md` listing ≥5 source files
- [ ] `knowledge/modules/prefix-cache.md` populated with ≥3 facts; verify NO `## [KP]\d+: [KP]\d+:` headings
- [ ] `knowledge/INDEX.md` updated with the new module row
- [ ] On completion: `TaskUpdate #16 → completed` AND **explicit `SendMessage` to tester-2** with: (a) impl paths, (b) test-pass status, (c) "vLLM 没有 radix tree" framing reminder for the tester to verify

## 12. Tasks expected (book-editor / team-lead should create on kickoff)

Following the Ch04/Ch05/Ch06 pattern:
- #16 Ch07 implementer: rewrite prefix-cache implementation
- #17 Ch07 tester: validate prefix-cache implementation
- #18 Ch07 writer: produce narrative chapter.md (v6 standard)
- #19 Ch07 reviewer: gate APPROVE/REVISE on chapter.md
- #20 Ch07 archivist: record delivery and update state

---

**This brief should be forwarded by book-editor-2 / team-lead to implementer-2 with the full body. Reviewer-2 may benefit from skimming §2 (the radix-tree language-trap) and §6 (strengths-to-preserve catalog).**
