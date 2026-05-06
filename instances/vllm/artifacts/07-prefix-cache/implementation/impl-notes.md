# Implementation Notes — Ch07 Automatic Prefix Caching

Strategy A1 rewrite. vLLM source pinned at `98661fe`. **Ch07 is the prefix-
cache layer ON TOP of Ch05's BlockPool**, not a re-derivation of allocation.

## Source Analysis (HARD GATE)

### 1. What vLLM files implement this feature?

| Role | Path (verified at 98661fe) | Lines actually read |
|------|----------------------------|---------------------|
| Hash primitives | `instances/vllm/source/vllm/v1/core/kv_cache_utils.py` | L40-L78 (BlockHash, group-id pack/unpack), L83-L110 (NONE_HASH init), L373-L536 (extra_keys composition), L539-L566 (hash_block_tokens), L635-L686 (get_request_block_hasher) |
| Block pool + cache index | `instances/vllm/source/vllm/v1/core/block_pool.py` | L34-L127 (BlockHashToBlockMap), L130-L182 (BlockPool init), L184-L209 (get_cached_block), L211-L320 (cache_full_blocks), L322-L352 (get_new_blocks + lazy eviction), L354-L389 (_maybe_evict_cached_block), L391-L422 (touch + free_blocks), L424-L441 (evict_blocks), L478-L497 (get_num_free_blocks, get_usage) |
| Per-spec manager | `instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py` | L30-L82 (SingleTypeKVCacheManager init), L277-L318 (cache_blocks, free), L336-L383 (find_longest_cache_hit ABC), L446-L494 (FullAttentionManager.find_longest_cache_hit) |
| Top-level KVCacheManager | `instances/vllm/source/vllm/v1/core/kv_cache_manager.py` | L21-L103 (KVCacheBlocks dataclass), L106-L161 (init), L183-L223 (get_computed_blocks), L225-L416 (allocate_slots — the prefix-cache integration entry) |
| Free queue | `instances/vllm/source/vllm/v1/core/kv_cache_utils.py` | L162-L370 (FreeKVCacheBlockQueue — for the touch O(1)-removal cost discussion) |

### 2. Key classes and responsibilities

| Class | Owns | Delegates to |
|-------|------|--------------|
| `BlockHash` (NewType bytes) | the bytes of one chained block hash | — |
| `BlockHashWithGroupId` (NewType bytes) | block hash + 4-byte group id | — |
| `chain_block_hashes` | a list of chained hashes for a token sequence | hash_block_tokens |
| `BlockHashToBlockMap` | dict[hash → block_id \| dict[id, id]] | — |
| `get_cached_block` | per-group cache lookup with all-groups-must-hit semantic | BlockHashToBlockMap |
| `Trie` | naive baseline (one node per token) | — |
| `RadixTree` | path-compressed trie (edges hold token sequences) | — |
| `PrefixCacheManager` | match (find_longest_cache_hit) + insert (cache_blocks) + evict | BlockHashToBlockMap |
| `prefix_aware_allocate` | match → touch → fresh-allocate → cache_blocks | PrefixCacheManager |
| `AllocateResult` (KVCacheBlocks equivalent) | block_ids, num_cache_hit_blocks, num_cache_hit_tokens | — |
| `verify_invariants` | assert I1/I2/I3 hold on a manager | — |

### 3. Data flow (one allocate cycle)

```
User submits request with token_ids
  ↓
chain_block_hashes(tokens, B, extra_keys)        ← Ch07.block_hash
  produces: [H_0, H_1, H_2, ...] = chained hashes per FULL block
  ↓
KVCacheManager.get_computed_blocks(req)           ← Ch07.paged_integration
  for h in chain[:max_cache_hit_length // B]:
    block = cache.get_one_block(pack(h, group_id))
    if block is None: break
    cached.append(block)
  return KVCacheBlocks(cached), num_tokens_hit
  ↓
Scheduler decides whether num_new_tokens fit budget
  ↓
KVCacheManager.allocate_slots(req, num_new_tokens, ...)
  ├─ allocate_new_computed_blocks()              ← appends cached blocks first
  ├─ allocate_new_blocks()                       ← block_pool.get_new_blocks(N)
  │    ├─ free_block_queue.popleft_n(N)
  │    └─ for each: _maybe_evict_cached_block()  ← lazy eviction here
  └─ cache_blocks(req, num_tokens_to_cache)      ← block_pool.cache_full_blocks
       └─ for each new full block:
            cached_block_hash_to_block.insert(packed_hash, block)
  ↓
Future request with same prefix:
  get_computed_blocks → cache hit → block_pool.touch() bumps ref_cnt
  → free_block_queue.remove(block) (O(1))        ← the doubly-linked list payoff
```

### 4. Design decisions and WHY

**D1. Hash table, not radix tree.** vLLM's prefix cache is a flat
`dict[BlockHashWithGroupId, KVCacheBlock | dict]` (`block_pool.py:L34-L127`).
It is NOT a radix tree, even though radix-tree mental models are common in
prefix-cache literature. Why:
- O(1) per-block lookup vs O(L/B) tree traversal. The microbench in §2 shows
  the hash table is ~4.6× faster on realistic prompt lengths.
- Cache locality. A dict bucket fits a cache line; a tree node walks pointers.
- Concurrent insertion is much simpler. A dict's invariants are atomic;
  a tree's restructuring is not.
- The chained hash already gives the "tree property" (a miss at depth K
  guarantees no hit at depth > K) without paying for an actual tree.

When a real radix tree DOES win:
- You need to ENUMERATE prefixes (admin tooling, eviction heuristics).
- You need RANGE queries.
- Memory is so tight that path compression's savings dominate.

vLLM is in none of those regimes, so the hash table dominates. The
`radix_tree.py` module is pedagogical — it lets the reader see WHY vLLM made
the choice rather than taking it on faith.

**D2. Append-only block_id assignment** (block_pool.py:L48-L52 NOTE).
"NOTE: We currently don't de-duplicate the blocks in the cache, meaning that
if a block becomes full and is cached, we don't check if there is already an
identical block in the cache. This is because we want to make sure the
allocated block IDs won't change so that block tables are append-only."

The block table is the input to PagedAttention's GPU kernel. If a request's
block_table held block_id=N and prefix-cache deduplication later "merged"
that with a different physical block, the GPU would silently read the
wrong page mid-decode → corrupted output. vLLM avoids this by NOT
deduplicating at insert time. The cost is a `dict[int, KVCacheBlock]`
inner branch in BlockHashToBlockMap to handle the rare collision case.

**D3. Chained hash → scan-and-stop match** (kv_cache_utils.py:L539-L566 +
single_type_kv_cache_manager.py:L473-L483). The hash for block K depends on
the hash for block K-1 transitively, which is why
`find_longest_cache_hit` can `break` on the first miss
(`single_type_kv_cache_manager.py:L482-L483` — "if a block hash is not in
cached_block_hash_to_id, the following block hashes are not computed yet
for sure"). This is the I3 invariant in `paged_integration.verify_invariants`.

**D4. Lazy eviction** (block_pool.py:L322-L352 inside `get_new_blocks`).
Eviction is NOT a separate cron-style sweep. It happens when
`get_new_blocks` pops the LRU head off the free queue and discovers the
block is cached — `_maybe_evict_cached_block` is called inline. Why lazy:
- No background thread / GIL contention.
- Eviction work scales with allocation work, not cache size.
- Evict-on-pop guarantees the cache index never points to a popped block.

**D5. `max_cache_hit_length = num_tokens - 1`**
(kv_cache_manager.py:L208). Even if an entire prompt hits the cache, vLLM
forces ONE token of recompute to obtain logits for that position. Without
this, a 100%-cache-hit request would have no logits to sample from →
RuntimeError downstream. The `-1` is small but load-bearing.

**D6. ref_cnt + free_block_queue + cache index = three-state lifecycle**
(block_pool.py:L391-L422). A cached block is in one of three states:
- ref_cnt > 0 (in use by ≥1 running request) — NOT in free queue, IS in cache.
- ref_cnt = 0, hash set (idle but cached) — IS in free queue, IS in cache.
- ref_cnt = 0, hash None (idle and uncached) — IS in free queue, NOT in cache.
The transitions are: alloc → 1st state; finish/free → 2nd state; LRU pop +
evict → 3rd state. `touch` moves 2nd → 1st (and removes from queue with
the doubly-linked list's O(1) middle-removal — Ch05 K05).

### 5. Complexity preserved (NOT simplified away)

- Chained hashing with `(parent_hash, tuple(token_ids), extra_keys)` payload.
- BlockHashWithGroupId packing (4-byte big-endian appendix).
- BlockHashToBlockMap's no-deduplication invariant.
- `get_cached_block` returning `None` if any group misses (multi-group
  semantics for hybrid attention models).
- `find_longest_cache_hit` scan-and-stop on first miss.
- Lazy eviction inside `get_new_blocks`.
- ref_cnt + free queue + cache index three-state lifecycle.
- `max_cache_hit_length = num_tokens - 1` to preserve logits.
- prefix-aware allocation order: cached blocks first, fresh after.

### What we deliberately simplified (each annotated)

- `extra_keys` is just a tuple — vLLM composes it from mm_features /
  lora / cache_salt / prompt_embeds (kv_cache_utils.py:L501-L536).
- `BlockHashToBlockMap` uses `int` for block_ids; vLLM uses `KVCacheBlock`
  metadata objects.
- Multi-group `find_longest_cache_hit` collapses to single group_id —
  Ch12-13 will reintroduce hybrid attention.
- `KVCacheManager` is collapsed into `PrefixCacheManager` — single-spec only,
  no coordinator / multi-group routing.
- `cache_full_blocks`'s KV-event emission for distributed prefix cache logging
  (block_pool.py:L276-L320) is omitted.
- `MambaManager.cache_blocks` (single_type_kv_cache_manager.py:L1052) and the
  cross-attention/sliding-window variants are deferred.
- `verify_invariants` returns a flat dict of bools; production diagnostics
  would be richer.

## 1:1 Source Mapping

| Our code | vLLM source | What we changed | Why |
|----------|-------------|-----------------|-----|
| `BlockHash`, `BlockHashWithGroupId` | `kv_cache_utils.py:L40-L45` | identical NewType wrappers | typing |
| `make_block_hash_with_group_id` | `kv_cache_utils.py:L53-L62` | identical (4-byte big-endian) | — |
| `get_block_hash`, `get_group_id` | `kv_cache_utils.py:L65-L72` | identical | — |
| `init_none_hash` | `kv_cache_utils.py:L83-L110` | dropped CBOR warning | — |
| `hash_block_tokens` | `kv_cache_utils.py:L539-L566` | sha256(repr()) instead of sha256_cbor | dependency-free |
| `chain_block_hashes` | `request.py:Request.update_block_hashes` | extracted as pure function | testability |
| `BlockHashToBlockMap.get_one_block` | `block_pool.py:L62-L73` | identical | — |
| `BlockHashToBlockMap.insert` | `block_pool.py:L75-L91` | identical (3 branches: empty/int/dict) | — |
| `BlockHashToBlockMap.pop` | `block_pool.py:L93-L121` | identical | — |
| `get_cached_block` | `block_pool.py:L184-L209` | identical (multi-group all-must-hit) | — |
| `Trie` | NOT in vLLM | new (pedagogical baseline) | radix-vs-hash benchmark |
| `RadixTree` | NOT in vLLM | new (pedagogical comparator) | shows what hash table replaces |
| `PrefixCacheManager.find_longest_cache_hit` | `single_type_kv_cache_manager.py:L446-L494` | single-group only | Ch12-13 has multi-group |
| `PrefixCacheManager.cache_blocks` | `single_type_kv_cache_manager.py:L277-L301` | identical idempotent guard | — |
| `PrefixCacheManager.evict_block` | `block_pool.py:L354-L389` | identical | — |
| `PrefixCacheManager.evict_blocks` | `block_pool.py:L424-L441` | identical | — |
| `PrefixCacheManager.touch` | `block_pool.py:L391-L406` | dropped metrics_collector | optional |
| `PrefixCacheManager.free_request` | `single_type_kv_cache_manager.py:L303-L318` | identical reverse-order free | — |
| `prefix_aware_allocate` | composition: `kv_cache_manager.py:L183-L223` + `L225-L416` | full match→touch→fresh→cache pipeline | — |
| `AllocateResult` | `kv_cache_manager.py:L21-L103` (KVCacheBlocks) | trimmed to 3 fields | — |
| `get_computed_blocks` | `kv_cache_manager.py:L183-L223` | identical (max_hit = N-1) | — |
| `verify_invariants` | composition (no single source) | new | I1/I2/I3 diagnostic |

## Files in this directory

- `__init__.py` — module map.
- `block_hash.py` — BlockHash, NONE_HASH, hash_block_tokens, chain_block_hashes.
- `prefix_cache_index.py` — BlockHashToBlockMap (dict + collision dict).
- `radix_tree.py` — Trie + RadixTree (pedagogical, NOT in vLLM).
- `prefix_cache_manager.py` — match / cache_blocks / evict + prefix_aware_allocate.
- `paged_integration.py` — get_computed_blocks, allocate_with_prefix_cache, verify_invariants.
- `demo.py` — runnable trace of all five outline sections.
- `impl-notes.md` — this file.
- `_legacy/` — old v5 attempt (read-only reference).

## Running the demo

```bash
python3 -m instances.vllm.artifacts.07-prefix-cache.implementation.demo
```

Expected highlights:
- §1 hit-rate sweep: 0.10 prefix → 9.3% hit; 0.90 prefix → 88.2% hit.
  (rate ~= shared_frac, modulo the partial-block remainder.)
- §2 microbench: hash table ~4.6× faster than RadixTree, ~4.6× faster than Trie.
- §3 match/insert/evict trace: A admits, B reuses 2 blocks of A's prefix,
  evict A's block 0 → D's chain breaks at the FIRST block (chain monotonicity).
- §4 prefix-aware vs naive: 78% KV-block savings (32-block sys_prompt × 50 reqs).
- §5 invariants: I1, I2, I3 all PASS.

## Language Traps (writer/reviewer must read)

These four phrasings are easy to write and almost-but-not-quite right. Each
trap is a direct invitation to either a misinformed reader or a wrong fix
later in the codebase.

**Trap 1 — "vLLM 使用 radix tree."** vLLM v1 does NOT have a radix tree.
The implementation is a flat `dict[BlockHashWithGroupId, KVCacheBlock]` at
`block_pool.py:L34-L127`. The "tree" structure is implicit in the
`parent_hash` chain (`kv_cache_utils.py:L539-L566`), not a tree class.
Lookup is O(1) per block; the linear scan over the chain stops at the first
miss. SGLang's "RadixAttention" literature uses a real radix tree — vLLM
does not. The chain hash + dict gives the same correctness guarantees with
simpler code, lock-free reads, and no path-compression maintenance.
The pedagogical `radix_tree.py` in this chapter exists ONLY to quantify
the trade-off (hash table ~4.6× faster on the §2 microbench).

**Trap 2 — "Cache hit on partial block."** Prefix caching only operates on
FULL blocks. `cache_blocks` (`single_type_kv_cache_manager.py:L286-L290`)
bails out when `num_cached_blocks >= num_full_blocks`, where
`num_full_blocks = num_tokens // block_size` (integer division). A
trailing partial block is NEVER cached, EVEN IF the same first 17 tokens
appear in many requests with `block_size=16`. This is why hit rate sweeps
in §1 sit slightly below the shared-prefix fraction (e.g. 0.10 fraction
→ 9.3% hit rate, not 10%): the partial-block remainder steals from the cache.

**Trap 3 — "Cache hit makes the request free."** A cache hit only saves
PREFILL compute on the matching prefix. The request still allocates output
blocks for decode tokens it produces (`kv_cache_manager.py:L380-L391`,
`allocate_new_blocks` runs after `allocate_new_computed_blocks`). The
saving is in TFLOPs (skipped attention forward) and KV bandwidth (no
re-write of identical KV), not in block-pool capacity for the decode
phase. A 100%-cache-hit request still costs `ceil(max_output_tokens / B)`
blocks for its output; if the pool is near-full, prefix cache does NOT
prevent admission stalls there.

**Trap 4 — "Need exhaustive search to find longest cached prefix."** No.
The chained-hash invariant (`kv_cache_utils.py:L539-L566`) guarantees that
if `H_k` is uncached, then `H_{k+1}` is uncached too — because `H_{k+1}`
depends on `H_k` transitively. So `find_longest_cache_hit`
(`single_type_kv_cache_manager.py:L473-L483`) does a HEAD-FIRST SCAN with
break-on-first-miss. Binary search, exhaustive comparison, or any other
"find longest match" pattern would be wrong (in addition to being slower):
the chain hash is what makes the simple scan correct.

## Cross-chapter dependency

- **Inherits from Ch03**: PagedAttention block_table semantics, the
  fixed-size page abstraction.
- **Inherits from Ch05**: BlockPool, KVCacheBlock, FreeKVCacheBlockQueue,
  ref_cnt accounting, lazy eviction inside `get_new_blocks`.
- **Provides to Ch08-Ch11**: cache-aware admission for distributed inference;
  the chain invariant lets data-parallel ranks agree on prefix matches with
  no cross-rank coordination.

## Notes for the next implementer / writer

- vLLM does NOT use a radix tree, despite literature on "RadixAttention" /
  SGLang. If a reviewer or writer says "vLLM uses a radix tree" — show them
  `block_pool.py:L34-L127` (it's a dict). Be charitable: the chained-hash
  invariant gives the "tree property" without the tree.
- The `dict[int, int]` collision branch in BlockHashToBlockMap is rarely
  exercised in tests because hash collisions on sha256 are basically
  impossible. It exists for the (provable-to-the-reader) edge case where
  block_pool inserts under a key that already exists. Ch07 demo's §3 uses
  block_size=4 and small token vocab where collisions are still
  astronomically unlikely; if you write a test that exercises the dict
  branch, force-collide by inserting the same hash twice with different
  block_ids manually.
- The `extra_keys=()` default in `chain_block_hashes` is fine for
  text-only/non-LoRA workloads. Multimodal Ch needs to thread mm_features
  through; LoRA Ch needs lora_name. See `kv_cache_utils.py:L501-L536` for
  the composition rule.
- `verify_invariants` returns optimistic results — I1 only checks ref_cnt
  consistency, I3 checks structural assumptions. A real audit needs to
  replay the request log against the manager state. For Ch07's purposes,
  the demo's PASS/PASS/PASS shows the manager doesn't violate invariants
  on the workloads we exercise.
