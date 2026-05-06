# Knowledge Index — vLLM Repository

Repo-specific facts organized by module. These have TTL and decay. Query before working on a chapter.

## Module Map

| Module | Chapters | Key Files |
|--------|----------|-----------|
| [scheduler](modules/scheduler.md) | 04, 06, 07 | scheduler.py:L352-L945 |
| [kv-cache](modules/kv-cache.md) | 02, 12, 13 | kv_cache_manager.py, block_pool.py |
| [attention](modules/attention.md) | 01, 03 | flash_attn.py, triton_decode_attention.py |
| [tensor-parallelism](modules/tensor-parallelism.md) | 08, 09, 11, 15+ | linear.py, parallel_state.py, communication_op.py |
| [prefix-cache](modules/prefix-cache.md) | 07, 13, 23 | block_pool.py:L34-L127, kv_cache_utils.py, single_type_kv_cache_manager.py |
| [preemption](modules/preemption.md) | 06 | scheduler.py preemption path |
| [memory](modules/memory.md) | 05 | vllm.utils format_gib + memory profiling |
| [prefill-decode](modules/prefill-decode.md) | 04-cp, 22-25 | scheduler.py (chunked prefill) |

## Query Protocol

Before starting work on chapter `{id}`:
1. Check this INDEX for the relevant module
2. Read the module knowledge file
3. Filter entries by your role (implementer/tester/writer/reviewer)
4. Apply relevant facts to your work

## Anti-Bloat Rules

- Max 15 facts per module file
- When exceeded: oldest 5 facts are LLM-compacted into a single summary fact
- Facts unused for 30 days → archived to `knowledge/archive/{module}-{date}.md`
- Access count tracked per fact; top 5 are pinned (never compacted)
