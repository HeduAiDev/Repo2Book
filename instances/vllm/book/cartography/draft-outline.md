# 草案大纲 (DRAFT) — vLLM v1 源码解读

> 由架构测绘 synthesis 生成，待人工评审。7 Parts / 28 章，按依赖拓扑排序。


## Part I — The Big Picture: One Request, End to End

- **[ch01] What vLLM v1 Is and How to Read This Book** _(intro)_  
  子系统: `config-and-wiring` · 依赖: —  
  vLLM v1 mental model, why v1 differs from v0 (async-decoupled, persistent batch, torch.compile), the offline LLM vs OpenAI server surfaces, the subtract-only companion convention, and how source-reading chapters are structured (Source Trail + Theory).
- **[ch02] The Lifecycle of a Request (Bird's-Eye Trace)** _(intro)_  
  子系统: `entrypoints` · 依赖: ch01  
  Walk the full spine at low resolution: generate()/HTTP -> render -> InputProcessor -> EngineCore (schedule+execute+sample+update) -> OutputProcessor -> streamed tokens. Establishes the map every later chapter zooms into. Uses async_llm.py:524, core.py:402, output_processor.py:572 as anchor points.
- **[ch03] From EngineArgs to VllmConfig: Assembling the Stack** _(core)_  
  子系统: `config-and-wiring` · 依赖: ch02  
  EngineArgs.create_engine_config() building ModelConfig/CacheConfig/ParallelConfig/SchedulerConfig/CompilationConfig into VllmConfig; optimization levels O0-O3; factory selection of Executor, Scheduler, and EngineCoreClient driven by feature flags; compute_hash().

## Part II — The Async 3-Stage Decoupling (Flagship)

- **[ch04] AsyncLLM: The 3-Stage Decoupled Facade** _(core)_  
  子系统: `async-engine` · 依赖: ch03  
  The crown-jewel architecture: in-process input preprocessing, out-of-process EngineCore, in-process background output handler. generate() async generator, add_request() fan to OutputProcessor + EngineCore, lazy output_handler task, chunked processing with asyncio.sleep(0). async_llm.py:280-707.
- **[ch05] Stage 1 — Input Processing: Prompt to EngineCoreRequest** _(core)_  
  子系统: `input-processor` · 依赖: ch04  
  InputProcessor.process_inputs(): tokenization via Renderer, param validation, request-id assignment, multimodal MultiModalFeatureSpec extraction/sorting, and EngineCoreRequest construction. input_processor.py:234-377.
- **[ch06] Parallel Sampling Fan-Out (n>1)** _(advanced)_  
  子系统: `input-processor` · 依赖: ch05  
  ParentRequest spawning n independent child EngineCoreRequests with unique ids and deterministic seed progression; why independent scheduling beats batched n; aggregation hooks set up for the output stage. parallel_sampling.py, async_llm.py:376-412.
- **[ch07] The IPC Boundary: ZMQ, msgpack, and Zero-Copy Tensors** _(advanced)_  
  子系统: `engine-core` · 依赖: ch04, ch05  
  EngineCoreClient hierarchy (Inproc/Sync/AsyncMP), ZMQ DEALER sockets, EngineCoreRequestType byte-tagged protocol, msgpack encode/decode, EngineCoreProc input/output socket threads, and TensorIpcSender/Receiver drain-and-buffer for multimodal tensors. core_client.py, core.py:1320-1531, tensor_ipc.py.
- **[ch08] Stage 3 — Output Processing: Tokens to RequestOutput** _(core)_  
  子系统: `output-processor` · 依赖: ch07, ch06  
  The single-loop OutputProcessor.process_outputs(), per-request RequestState, RequestOutputCollector queues with DELTA merging, stream_interval, finish handling, and parent aggregation for n>1. output_processor.py:572-687, 45-107.
- **[ch09] Incremental Detokenization and Stop Strings** _(advanced)_  
  子系统: `output-processor` · 依赖: ch08  
  IncrementalDetokenizer hierarchy (Fast tokenizers.DecodeStream vs Slow Python), stop-string buffer holdback, min_tokens guard, UTF-8 boundary recovery. detokenizer.py:30-340.
- **[ch10] Logprobs Assembly with Byte-Fallback Correction** _(advanced)_  
  子系统: `output-processor` · 依赖: ch09  
  Sample and prompt logprobs from EngineCore tensors, context-aware UTF-8 multi-byte reconstruction, cumulative logprob tracking, flat vs nested formats. logprobs.py:30-353.

## Part III — Inside EngineCore: The Busy Loop

- **[ch11] EngineCore and the Busy Loop** _(core)_  
  子系统: `engine-core` · 依赖: ch07  
  EngineCore.step() orchestration (schedule -> execute_model non_block -> grammar bitmask -> sample_tokens -> update_from_output), run_busy_loop, lifecycle (pause/resume/sleep), and where the batch queue plugs in. core.py:91-432, 1164-1205.
- **[ch12] Pipeline Parallelism in the Loop: the Batch Queue** _(advanced)_  
  子系统: `engine-core` · 依赖: ch11  
  step_with_batch_queue() buffering SchedulerOutput+future pairs to keep PP stages busy, deferred sampling for structured output, async_scheduling flag. core.py:443-559.

## Part IV — Scheduling and KV Cache

- **[ch13] Continuous Batching with Token Budgets** _(core)_  
  子系统: `scheduler` · 依赖: ch11  
  Token-centric scheduling (no prefill/decode phases), RUNNING-first, max_num_scheduled_tokens budget, SchedulerOutput (NewRequestData vs CachedRequestData persistent-batch protocol), FCFS vs Priority queues. scheduler.py:352-525, output.py.
- **[ch14] Preemption, Waiting Queues, and Update-From-Output** _(advanced)_  
  子系统: `scheduler` · 依赖: ch13  
  Preemption loop on allocation failure, dual waiting/skipped queues for head-of-line avoidance, update_from_output token append + stop detection + request lifecycle/cleanup. scheduler.py:466-846, 1290-1551.
- **[ch15] Paged KV Cache: Block Pool and Prefix Caching** _(core)_  
  子系统: `kv-cache` · 依赖: ch13  
  BlockPool, KVCacheBlock, FreeKVCacheBlockQueue doubly-linked LRU, block hashing with extra keys (MM/LoRA/salt), BlockHashToBlockMap, reference counting touch/free. block_pool.py, kv_cache_utils.py.
- **[ch16] Allocation and Multi-Attention Coordination** _(advanced)_  
  子系统: `kv-cache` · 依赖: ch15  
  KVCacheManager three-stage allocate_slots (free skipped / prefix hits + external / new + lookahead), KVCacheCoordinator (Unitary vs Hybrid), fixed-point cache-hit search for hybrid full+SWA+MLA models. kv_cache_manager.py:225-416, kv_cache_coordinator.py:487-591.

## Part V — Execution: Workers, Runner, Kernels

- **[ch17] Executors and Worker Lifecycle** _(core)_  
  子系统: `worker-and-executor` · 依赖: ch11  
  Executor.get_class() factory (uni/mp/ray), collective_rpc, MultiprocExecutor shared-memory MessageQueues + FutureWrapper + worker monitoring, WorkerWrapperBase lazy init, GPU memory profiling before KV allocation. abstract.py, multiproc_executor.py, gpu_worker.py.
- **[ch18] The Persistent Batch and _prepare_inputs** _(core)_  
  子系统: `model-runner` · 依赖: ch17, ch14, ch16  
  InputBatch persisting across iterations (slot reuse via batch_update_builder), _update_states reconciling SchedulerOutput, _prepare_inputs (token index_select, positions, block table commit, Triton slot_mapping), CachedRequestState. gpu_model_runner.py:1065-1936, gpu_input_batch.py, block_table.py.
- **[ch19] Decoupled Forward and Sampling: ExecuteModelState** _(advanced)_  
  子系统: `model-runner` · 依赖: ch18  
  Two-phase execute_model() returning None + caching ExecuteModelState, sample_tokens() unpacking it, _bookkeeping_sync writing tokens back, CUDA graph dispatch. gpu_model_runner.py:378-391, 3825-4225.
- **[ch20] Distributed Parallelism: Groups and Collectives** _(advanced)_  
  子系统: `distributed-parallelism` · 依赖: ch17  
  GroupCoordinator (TP/PP/DP/EP), CPU(gloo)+device(NCCL) groups, all_reduce/all_gather/reduce_scatter, custom-op registration for torch.compile, P2P for PP. parallel_state.py:290-809, communication_op.py.
- **[ch21] Async Communication and Data Parallelism** _(advanced)_  
  子系统: `async-engine` · 依赖: ch20, ch12  
  AsyncIntermediateTensors lazy PP sync (isend/irecv), DPCoordinator wave synchronization with all-reduce consensus, DP load balancing (DPLBAsyncMPClient). gpu_worker.py:74-861, coordinator.py, core.py:1622-1863.

## Part VI — Models, Attention, Sampling

- **[ch22] Model Definitions and Weight Loading** _(core)_  
  子系统: `model-definitions` · 依赖: ch20  
  Llama as case study: v1 (vllm_config, prefix) signature, ColumnParallel/QKVParallel linear with TP, get_rope() factory, model loader pipeline (initialize_model -> load_weights -> process_weights_after_loading), AutoWeightsLoader, stacked QKV/gate-up mapping. llama.py, model_loader/, linear.py.
- **[ch23] Attention Backends and Metadata** _(advanced)_  
  子系统: `attention` · 依赖: ch22, ch16, ch18  
  AttentionBackend abstraction + registry + selector, CommonAttentionMetadata -> backend-specific metadata (FlashAttention/FlashInfer/Triton), KV cache shape/stride (NHD vs HND), AOT scheduler for FA3, cascade and DCP. backend.py, selector.py, backends/flash_attn.py.
- **[ch24] The Sampling Pipeline** _(core)_  
  子系统: `sampling` · 依赖: ch19  
  Sampler 9-step pipeline (raw logprobs, penalties, bad words, logits processors argmax-invariant vs not, temperature, top-k/top-p), multi-backend TopKTopPSampler dispatch, persistent batch BatchUpdate for logits processors. sampler.py, ops/topk_topp_sampler.py, logits_processor/.
- **[ch25] Speculative Decoding: Proposers and Rejection Sampling** _(advanced)_  
  子系统: `spec-decode` · 依赖: ch24, ch14, ch23  
  Proposers (ngram CPU/GPU, EAGLE, DFlash), SpecDecodeMetadata index indirection, rejection sampling (probabilistic vs synthetic) against target logits, fused Triton slot-mapping kernels with rejection masking. llm_base_proposer.py, rejection_sampler.py, spec_decode/metadata.py, utils.py.

## Part VII — The Serving Surface

- **[ch26] The Offline LLM API** _(core)_  
  子系统: `entrypoints` · 依赖: ch08, ch11  
  LLM class: generate/chat/embed/encode, EngineArgs -> LLMEngine over EngineCore, request batching and rendering, the sync LLMEngine.step() path vs async. llm.py, v1/engine/llm_engine.py.
- **[ch27] The OpenAI-Compatible Server** _(core)_  
  子系统: `entrypoints` · 依赖: ch26, ch04  
  FastAPI lifespan + build_async_engine_client, OpenAIServing base + chat/completion handlers, Renderer, tool/reasoning parsers, SSE streaming vs JSON, launcher (uvicorn, watchdog, SSL). api_server.py, openai/*/serving.py, launcher.py.
- **[ch28] Advanced Engine Operation: Elastic Scaling and Multi-Turn** _(advanced)_  
  子系统: `engine-core` · 依赖: ch27, ch21, ch14  
  Elastic EP scale-up/scale-down state machine, Responses API multi-turn/tool context, KV connector (disaggregated prefill) integration touchpoints in the scheduler. core.py:1865-1977, core_client.py:1542-1695, openai/responses/serving.py, scheduler.py:621-805.