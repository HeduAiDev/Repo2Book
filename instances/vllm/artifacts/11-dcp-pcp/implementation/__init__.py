# Chapter 11: DCP/PCP — Decode/Prefill Context Parallelism
# Pedagogical re-implementation of vLLM Context Parallelism at commit 98661fe.
#
# Modules:
#   parallel_state_dcp_pcp  — _DCP/_PCP singleton mirrors + initialize_model_parallel
#   world_topology          — 5D device mesh (external_dp x dp x pp x pcp x tp)
#   attention_backend       — AttentionImplBase __new__ discovery, total_cp = pcp x dcp
#   lse_combine             — LSE-weighted combine math (FlashAttention online softmax across ranks)
#   dcp_alltoall            — A2A communication backend (Triton kernel mirror in pure Python)
#   seq_sharding            — get_dcp_local_seq_lens with cp_kv_cache_interleave_size
#   kv_cache_per_rank       — max_memory_usage_bytes formula
#   dcp_vs_pcp_demo         — separable-axes facts (D-prefix knowledge entries D01, D06, D07)
#   demo                    — 5+ demos producing verbatim numerics for the writer
