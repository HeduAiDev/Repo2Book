"""GPU memory management — faithful reimplementation of vLLM v1.

Module layout mirrors `vllm/v1/` and `vllm/utils/`:

    mem_snapshot.py    <-> vllm/utils/mem_utils.py
    kv_cache_spec.py   <-> vllm/v1/kv_cache_interface.py
    kv_cache_block.py  <-> vllm/v1/core/kv_cache_utils.py
    block_pool.py      <-> vllm/v1/core/block_pool.py
    memory_layout.py   <-> vllm/v1/worker/gpu_worker.py (determine_available_memory)
    recompute.py       (pedagogical: recompute-vs-swap trade-off; v1 chose recompute)
    demo.py            (end-to-end runnable trace)
"""
