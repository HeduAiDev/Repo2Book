"""Ch07 — Automatic Prefix Caching (APC).

Modules mirror vLLM's prefix-cache surface:

    block_hash.py            <-> vllm/v1/core/kv_cache_utils.py (L40-L78, L539-L566)
    prefix_cache_index.py    <-> vllm/v1/core/block_pool.py (L34-L127, L184-L209)
    radix_tree.py            (pedagogical: trie + path-compressed radix tree;
                              vLLM does NOT ship one — see module docstring)
    prefix_cache_manager.py  <-> vllm/v1/core/single_type_kv_cache_manager.py
                                 (L277-L301, L338-L383, L446-L494) +
                                 vllm/v1/core/kv_cache_manager.py (L183-L223)
    paged_integration.py     <-> vllm/v1/core/kv_cache_manager.py (L225-L416)
    demo.py                  (hit-rate sweep, radix-vs-hash microbench,
                              prefix-aware reuse savings)
"""
