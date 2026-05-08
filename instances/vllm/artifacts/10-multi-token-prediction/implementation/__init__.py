"""Ch10 Multi-Token Prediction (MTP) — pedagogical reimplementation.

Mirrors vLLM @ commit 98661fe:
  - vllm/v1/sample/rejection_sampler.py            -> rejection_sampling.py
  - vllm/v1/spec_decode/metadata.py                -> spec_metadata.py
  - vllm/v1/spec_decode/llm_base_proposer.py       -> proposers/base.py
  - vllm/v1/spec_decode/eagle.py                   -> proposers/eagle.py
  - vllm/v1/spec_decode/medusa.py                  -> proposers/medusa.py
  - vllm/v1/spec_decode/ngram_proposer.py          -> proposers/ngram.py
  - vllm/v1/spec_decode/extract_hidden_states.py   -> proposers/extract_hidden.py
  - vllm/v1/spec_decode/draft_model.py             -> proposers/draft_model.py
  - vllm/model_executor/models/deepseek_mtp.py     -> mtp_head.py + weight_loading.py
  - vllm/config/speculative.py                     -> (referenced via SpeculativeMethod literals)

The implementation is plain PyTorch / NumPy. Triton kernels in the source are
mirrored as Python loops with the same algorithmic semantics; performance is
~100x worse but the chain-break invariant, recovered-token math, and bonus-token
handling are identical.
"""
