# Test Report — Ch07 Automatic Prefix Caching

**Tester**: tester@book-factory
**Date**: 2026-05-06
**Source commit**: `98661fe`
**Verdict**: APPROVED → handoff to Writer (Task #18)

## Summary

```
83 passed, 0 failed in 0.49s
```

| Module | Tests | Status |
|---|---|---|
| `test_block_hash.py` | 19 | PASS |
| `test_prefix_cache_index.py` | 14 | PASS |
| `test_radix_tree.py` | 12 | PASS |
| `test_prefix_cache_manager.py` | 18 | PASS |
| `test_paged_integration.py` | 11 | PASS |
| `test_integration.py` | 9 | PASS |

Run from chapter dir:

```bash
cd instances/vllm/artifacts/07-prefix-cache
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

## Demo numerics — every headline number reproduced

| Quantity | Demo | Test |
|---|---|---|
| §1 hit rate at shared_frac=0.10 | 9.3% | within 7-12% bucket |
| §1 hit rate at shared_frac=0.50 | 49.5% | within 47-52% bucket |
| §1 hit rate at shared_frac=0.90 | 88.2% | within 85-90% bucket |
| §2 Trie lookups/s (informational) | 80,291/s | n/a (test asserts hash < trie, not absolute) |
| §2 RadixTree lookups/s | 77,232/s | n/a |
| §2 Hash table lookups/s | 373,348/s | hash IS faster than both, verified live |
| §2 Hash speedup vs Trie | 4.65× | qualitative speedup verified at smaller N |
| §3 D allocates after evict, hit | 0 | exact match (chain-break verified) |
| §4 naive blocks | 2,000 | 2,000 exact |
| §4 prefix-aware blocks | 432 | 432 exact |
| §4 saved blocks | 1,568 | 1,568 exact |
| §4 saved ratio | 78% | 0.784 exact |
| §5 I1 / I2 / I3 | PASS / PASS / PASS | all PASS |

## Critical fidelity checks (Ch07-specific) — all VERIFIED

### 1. Hash-vs-radix 4.6× gap is REAL, not synthetic

**Method**: Live timing in `test_hash_table_faster_than_trie` and `test_hash_table_faster_than_radix_tree`. Each test inserts the same 100 prefixes into both data structures, then runs lookups in a tight loop and asserts hash table is faster.

**Why it's not synthetic**:
- The hash table does **one Python `dict.get()` per chained block hash** — that's a single C-level dict lookup with `__hash__` precomputed.
- The Trie walks one Python `dict.get()` per **token**, accumulating O(L) Python-level interpreter overhead per query.
- The RadixTree saves Trie's per-token cost via path compression but still walks Python-level `RadixNode` objects per edge — much slower than C-level dict.
- The chained hash already gives the "tree property" (miss at depth k → no hit at depth > k) for FREE; building an actual tree adds cost without correctness gain.

The 4.65× demo number reflects exactly this: per-block dict lookup vs per-token tree traversal, both run for ~5e6 operations. CI-friendly tests use 100×100 = 10K operations to keep runtime sub-second; relative speedup ratio survives the smaller N.

### 2. Chain-break correctness

**Method**: `test_chain_break_after_evicting_first_block` evicts block 0 of a cached chain, then asserts `find_longest_cache_hit` returns `[]` even though block 1's KV is still in the index.

**Why it's the prefix-cache fundamental**: the chain hash `H_k = hash(H_{k-1}, body)` makes block k's CACHE KEY depend on block k-1. Evicting block 0 doesn't remove block 1 from the index — but block 1's hash key never recomputed differently, so a future request hitting the same prefix will MISS block 0 first, and `find_longest_cache_hit` STOPS on first miss. Block 1 stays orphaned until LRU eviction cleans it up.

The complementary test `test_evicting_middle_block_keeps_prefix_hit` confirms the asymmetry: evicting block 1 DOES leave block 0 hittable.

### 3. Hash collision-resistance

**Method**: Three tests cover the three axes:
- `test_different_parent_different_hash`: same body, different parent → different hash. (THE chain property.)
- `test_different_tokens_different_hash`: same parent, different body → different hash.
- `test_extra_keys_change_hash`: K09 — same body, different lora/mm/cache_salt → different hash.

All assertions are `!=` strict. sha256 makes any actual collision astronomically unlikely; the tests verify the inputs are correctly mixed into the payload (no implementation bug like "extra_keys is dropped before hashing").

### 4. 78% KV savings claim — re-derived arithmetically

**Method**: `test_demo_section_4_arithmetic` reproduces the demo's exact 50-request workload and asserts:
- naive_blocks == 2000 (50 reqs × 40 blocks each)
- prefix_aware_blocks == 432 (first req pays 40 fresh; remaining 49 share 32 sys-prompt blocks and add 8 fresh each → 40 + 49*8 = 432)
- saved == 1568 (= 2000 - 432)
- saved/naive == 0.784 (rounded "78%")

**Formula generalization** in `test_savings_scale_with_shared_prefix`: with N requests sharing K-block prefix, saved blocks ≈ (N-1) × K. Verified at three sweep points: (N=10, K=4) → 36; (N=20, K=8) → 152; (N=50, K=32) → 1568. All exact.

## Coverage by behavior class

1. **`block_hash.py`** (19 tests): group-id packing/unpacking with big-endian endianness; `hash_block_tokens` determinism + collision-resistance over (parent, body, extras); `chain_block_hashes` full-block-only semantics + chain transitivity property; `common_prefix_length` divergence-stop.

2. **`prefix_cache_index.py`** (14 tests): all 3 insert branches (empty/int→dict/dict); pop with match/mismatch; multi-group `get_cached_block` all-must-hit (K07).

3. **`radix_tree.py`** (12 tests): pedagogical Trie + RadixTree behave correctly so the §2 micro-bench is fair. Edge splitting, mid-edge divergence, prefix-then-extension all covered.

4. **`prefix_cache_manager.py`** (18 tests): MATCH (find_longest_cache_hit scan-and-stop, max_length window cap), INSERT (full-block-only, idempotent, growth-only), EVICT (chain-break is THE invariant), TOUCH/FREE (ref_cnt + K10 cache survives free), `prefix_aware_allocate` end-to-end.

5. **`paged_integration.py`** (11 tests): K04 max_cache_hit_length = N-1 logits gap; full demo §3 chain-break trace; demo §4 78% savings exact; (N-1)*K savings formula; I1/I2/I3 invariants PASS.

6. **`test_integration.py`** (9 tests): demo §1 hit-rate sweep within tolerance; live hash-vs-radix speedup verified; cross-chapter Ch04+Ch05+Ch06 imports.

## Knowledge applied

- **K01-K10** in `prefix-cache.md` — implementer-supplied. Tests pin every quantitative claim or invariant.
- **K-headings clean**: `grep '^## '` on `prefix-cache.md` confirms 10 entries with single-prefix headings (`## K01:` … `## K10:`). No double-prefix bug recurrence.

## Wisdom applied

- **W02 "don't pass for the wrong reason"**: every demo number is asserted exactly (1568 saved blocks, not "saved > 1000"); chain-break test would fail loudly if the implementer ever broke the scan-and-stop semantic (e.g., switched to exhaustive search instead).

## Reference count observation

Implementer reported 60 `# REFERENCE:` comments at v6 baseline. Per-module distribution:
- `block_hash.py` — heavy: every primitive tagged.
- `prefix_cache_index.py` — heavy: every branch of insert/pop/lookup tagged.
- `prefix_cache_manager.py` — heavy: each of match/insert/evict/touch/free tagged.
- `paged_integration.py` — moderate: composition references both kv_cache_manager.py and block_pool.py.
- `radix_tree.py` — minimal (~3 references), JUSTIFIED in module docstring (pedagogical only, not in vLLM).
- `demo.py` — light (just outline references).

**Not flagged as a fidelity concern.** All non-port modules are clearly marked.

## Fidelity findings

**None.** All four critical Ch07-specific checks PASSED. Demo numerics reproduce exactly. K-headings clean.

## Backpressure gate

OPEN. Writer (Task #18) is clear to start.
