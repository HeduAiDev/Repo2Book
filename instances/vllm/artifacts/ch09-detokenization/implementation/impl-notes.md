# ch09 implementation notes — incremental detokenization & stop strings

Subtract-only companion. Every module mirrors a real vLLM file at pin `f3fef123`,
keeping names / structure / control flow; only `subtraction_plan.delete` items are
removed (each marked `# SUBTRACTED:`). The Fast path imports the real
`tokenizers` (>=0.22.0) / `transformers` libraries unconditionally — exactly as
real vLLM does — so its tests run inside the vLLM container. The slow-path
algorithm and the pure update / stop / holdback / min_tokens logic run on host.

## Files

| companion | real vLLM source | notes |
|-----------|------------------|-------|
| `detokenizer.py` | `vllm/v1/engine/detokenizer.py` | full 3-way hierarchy + factory + check_stop_strings |
| `detokenizer_utils.py` | `vllm/tokenizers/detokenizer_utils.py` | slow-path double-window + UTF-8 boundary |
| `output_processor.py` | `vllm/v1/engine/output_processor.py` (`process_outputs`) | only the detokenize→stop_string→reqs_to_abort slice |
| `_types.py` | external dep surface | EngineCoreRequest / SamplingParams / EngineCoreOutput / FinishReason / TokenizerLike / length_from_prompt_token_ids_or_embeds — narrowed to fields the detokenizer reads |

## 1:1 Source Map (companion ↔ vllm/...:Lxxx ↔ change ↔ reason)

| companion symbol | vllm source | change | reason |
|------------------|-------------|--------|--------|
| `IncrementalDetokenizer` + `from_new_request` | `detokenizer.py:L30,L48` | verbatim | empty shell + 3-way factory dispatch |
| `USE_FAST_DETOKENIZER` / `INVALID_PREFIX_ERR_MSG` | `detokenizer.py:L24,L27` | verbatim | version gate + error-string discriminator |
| `BaseIncrementalDetokenizer.__init__` | `detokenizer.py:L69` | verbatim | stop / min_tokens / `stop_buffer_length = max(len)-1` holdback |
| `BaseIncrementalDetokenizer.update` | `detokenizer.py:L95` | verbatim | skip stop token, per-token decode, min_tokens guard, check_stop_strings truncate |
| `get_next_output_text` | `detokenizer.py:L148` | verbatim | delta/cumulative + tail holdback release on finish |
| `FastIncrementalDetokenizer.__init__` | `detokenizer.py:L168` | SUBTRACTED spaces-between-special-tokens precompute (L185-L205) | optional suppression branch off by default (delete #2) |
| `FastIncrementalDetokenizer.decode_next` | `detokenizer.py:L207` | SUBTRACTED special-token space-suppression branch (L210-L216) | delete #2; degrades to `return token or ""` |
| `FastIncrementalDetokenizer._protected_step` | `detokenizer.py:L220` | verbatim | UTF-8 invalid-prefix → rebuild DecodeStream; swallow OverflowError/TypeError |
| `SlowIncrementalDetokenizer.*` | `detokenizer.py:L245-L301` | verbatim | prompt-primed offsets; `output_token_ids`/`num_output_tokens` drop prompt |
| `check_stop_strings` | `detokenizer.py:L304` | verbatim | windowed find + include/exclude truncation offset |
| `detokenize_incrementally` | `detokenizer_utils.py:L110` | SUBTRACTED `else` added-encoders branch (L180-L192) | delete #3; fast/no-added-vocab path behaviour-equivalent |
| `convert_prompt_ids_to_tokens` + `INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET` | `detokenizer_utils.py:L54,L59` | verbatim | initial prefix/read offsets |
| (removed) `_convert_tokens_to_string_with_added_encoders` | `detokenizer_utils.py:L14-L51` | SUBTRACTED | delete #3 |
| (removed) `convert_ids_list_to_tokens` | `detokenizer_utils.py:L83-L104` | SUBTRACTED | delete #3, unused in this call chain |
| `OutputProcessor.process_outputs` | `output_processor.py:L572` | SUBTRACTED stats/logprobs/pooling/streaming-input/RequestOutput/tracing/parallel-sampling | delete #1; keep detokenize + stop_string + reqs_to_abort |

## Verification

- Host: `python3 -m pytest instances/vllm/artifacts/ch09-detokenization/tests/`
  → 25 passed, 1 skipped (Fast path is container-only).
- Container (Fast path, real tokenizers/transformers):
  `scripts/vllm_docker.sh -m pytest /work/instances/vllm/artifacts/ch09-detokenization/tests/test_fast_detokenizer.py -v`
  (not runnable in the authoring environment — docker unavailable here).
