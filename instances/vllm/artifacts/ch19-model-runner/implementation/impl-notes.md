# ch19 reduced companion — implementation notes

Subtract-only reduced version of the GPU model runner's two-phase per-step loop
(`execute_model()` → `sample_tokens()`), the persistent-batch write-back
(`_bookkeeping_sync`, f13), and CUDA-graph dispatch
(`CudagraphDispatcher.dispatch`). Pin `f3fef123`.

The persistent-batch infrastructure (`gpu_input_batch.py`, `block_table.py`,
`logits_processor_state.py`, `_support.py`) carries over verbatim from the ch18
reduced companion — ch19 reuses it so the chapter is self-contained. Only
`gpu_model_runner.py` (the two-phase orchestration) and `cudagraph_dispatcher.py`
are new for ch19.

The model and sampler are *not* implemented here — `GPUModelRunner` takes them as
constructor arguments and only orchestrates calls into them (the real ones belong
to the model-loading / sampling chapters). The tests inject tiny deterministic
stand-ins so the two-phase + write-back flow can be observed on host CPU.

## Source Map (reduced ↔ real vLLM ↔ change ↔ why)

| reduced symbol | vllm source | change | reason |
|---|---|---|---|
| `ExecuteModelState` | `vllm/v1/worker/gpu_model_runner.py:L378` | verbatim 10-field NamedTuple (field type annotations loosened to `object` for the spec/connector payloads we don't import) | the physical two-phase bridge; `must_keep` |
| `GPUModelRunner.execute_model` | `gpu_model_runner.py:L3825` | kept entry assertion + preprocess + dispatch + forward + cache-state + `return None`; subtracted MoE capture, ngram copy, KV/EC connector, mamba, ubatch, PP, pooling, cascade, `_build_attention_metadata`/`_preprocess` | phase-1 entry; `must_keep`. Subtractions are the approved optional deployment branches |
| `GPUModelRunner.sample_tokens` | `gpu_model_runner.py:L4178` | kept unpack-state → `_sample` → `_update_states_after_model_execute` → `_bookkeeping_sync` → assemble `ModelRunnerOutput`; subtracted grammar bitmask, draft proposal, PP broadcast, KV finalize, eplb, async wrapping | phase-2 entry; `must_keep` |
| `GPUModelRunner.execute_model_state` | `gpu_model_runner.py` (instance field) | single slot, `None` ↔ `ExecuteModelState`, paired with the entry assertion | `must_keep` |
| `GPUModelRunner._bookkeeping_sync` | `gpu_model_runner.py:L3397` | kept the write-back loop (`token_ids_cpu[slot,start:end]=sampled_ids`, `num_tokens_no_spec[slot]=end`, `output_token_ids.extend`); subtracted NaN scan, async-scheduling GPU caching / `-1` placeholder, spec-decode rejection parse, prompt logprobs, `is_token_ids` (already subtracted in ch18 InputBatch) | f13 write-back side; `must_keep` |
| `GPUModelRunner._prepare_inputs` | `gpu_model_runner.py:L1787` | ch18-reduced; `index_select` from `token_ids_cpu` via `token_indices = positions + slot*M`; slot_mapping kernel guarded to CUDA | f13 read-back side; `must_keep` |
| `GPUModelRunner._determine_batch_execution_and_padding` | `gpu_model_runner.py:L3591` | kept `_is_uniform_decode` + `dispatcher.dispatch`; subtracted LoRA count, SP/DP padding, ubatch coordination, enc-dec FULL-disable | `must_keep`; calls the dispatcher |
| `GPUModelRunner._model_forward` | `gpu_model_runner.py:L3538` | verbatim `self.model(...)` | `must_keep`; the inspectable forward |
| `GPUModelRunner._sample` | `gpu_model_runner.py:L3367` | kept plain-sampler branch; subtracted async output-token update + rejection-sampler spec path | `must_keep` |
| `GPUModelRunner._update_states_after_model_execute` | `gpu_model_runner.py:L1421` | kept call site; body returns immediately (hybrid/spec only) | `must_keep`; shows flow position |
| `set_forward_context` | `vllm/forward_context.py` | reduced context manager publishing `cudagraph_runtime_mode` + `batch_descriptor` + `slot_mapping`; subtracted DP/ubatch/attention plumbing | `must_keep`; the graph-replay publication point |
| `CudagraphDispatcher` / `dispatch` | `vllm/v1/cudagraph_dispatcher.py:L15` / `L234` | kept FULL→PIECEWISE→NONE selection + `max_size`/uninitialized cutoff; subtracted LoRA normalization, separate-routine composite modes | `must_keep` |
| `CUDAGraphMode` | `vllm/config/compilation.py:L53` | three scalar runtime modes only (NONE/PIECEWISE/FULL); subtracted composite tuple modes | `must_keep` |
| `compute_slot_mapping` / `commit_block_table` | `vllm/v1/worker/block_table.py` | carried from ch18 (Triton kernel, CUDA-only) | `must_keep`; slot_mapping source |
| `add_request` / `InputBatch` / `CachedRequestState` / `token_ids_cpu` / `num_tokens_no_spec` / `num_computed_tokens_cpu` / `output_token_ids` / `req_output_token_ids` | `vllm/v1/worker/gpu_input_batch.py` | carried from ch18 verbatim | `must_keep`; persistent-batch truth + f13 aliasing |

## Validation

Verdict: deleting every `# SUBTRACTED:` branch from the real vLLM
`execute_model` / `sample_tokens` / `_bookkeeping_sync` / dispatcher leaves a body
≈ this reduced companion. Host tests in `../tests/test_two_phase_and_writeback.py`
(15 passing) assert: the two-phase contract (return None / entry assertion / slot
reset), f13 write-back (token lands at the slot row, counter advances,
`req_output_token_ids[slot] is output_token_ids`), the f13 read-back closure
(written token re-enters as next step's input), and dispatch FULL/PIECEWISE/NONE.
