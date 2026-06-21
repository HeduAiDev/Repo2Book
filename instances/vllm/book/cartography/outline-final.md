# 最终大纲 — vLLM v1 源码解读 (8 Parts / 33 章)

> 合并两轮架构测绘；按依赖拓扑排序；async 三段式解耦为旗舰 Part II。


## Part I — The Big Picture: One Request, End to End

- **[ch01] What vLLM v1 Is and How to Read This Book** _(intro)_ · 依赖: —  
  vLLM v1 mental model, why v1 differs from v0 (async-decoupled, persistent batch, torch.compile), the offline LLM vs OpenAI server surfaces, the subtract-only companion convention, and how source-reading chapters are structured (Source Trail + Theory).
- **[ch02] The Lifecycle of a Request (Bird's-Eye Trace)** _(intro)_ · 依赖: ch01  
  Walk the full spine at low resolution: generate()/HTTP -> render -> InputProcessor -> EngineCore (schedule+execute+sample+update) -> OutputProcessor -> streamed tokens. Establishes the map every later chapter zooms into. Uses async_llm.py:524, core.py:402, output_processor.py:572 as anchor points.
- **[ch03] From EngineArgs to VllmConfig: Assembling the Stack** _(core)_ · 依赖: ch02  
  EngineArgs.create_engine_config() building ModelConfig/CacheConfig/ParallelConfig/SchedulerConfig/CompilationConfig into VllmConfig; optimization levels O0-O3; factory selection of Executor, Scheduler, and EngineCoreClient driven by feature flags; compute_hash().

## Part II — The Async 3-Stage Decoupling (Flagship)

- **[ch04] AsyncLLM: The 3-Stage Decoupled Facade** _(core)_ · 依赖: ch03  
  The crown-jewel architecture: in-process input preprocessing, out-of-process EngineCore, in-process background output handler. generate() async generator, add_request() fan to OutputProcessor + EngineCore, lazy output_handler task, chunked processing with asyncio.sleep(0). async_llm.py:280-707.
- **[ch05] Stage 1 — Input Processing: Prompt to EngineCoreRequest** _(core)_ · 依赖: ch04  
  InputProcessor.process_inputs(): tokenization via Renderer, param validation, request-id assignment, multimodal MultiModalFeatureSpec extraction/sorting, and EngineCoreRequest construction. input_processor.py:234-377.
- **[ch06] Parallel Sampling Fan-Out (n>1)** _(advanced)_ · 依赖: ch05  
  ParentRequest spawning n independent child EngineCoreRequests with unique ids and deterministic seed progression; why independent scheduling beats batched n; aggregation hooks set up for the output stage. parallel_sampling.py, async_llm.py:376-412.
- **[ch07] The IPC Boundary: ZMQ, msgpack, and Zero-Copy Tensors** _(advanced)_ · 依赖: ch04, ch05  
  EngineCoreClient hierarchy (Inproc/Sync/AsyncMP), ZMQ DEALER sockets, EngineCoreRequestType byte-tagged protocol, msgpack encode/decode, EngineCoreProc input/output socket threads, and TensorIpcSender/Receiver drain-and-buffer for multimodal tensors. core_client.py, core.py:1320-1531, tensor_ipc.py.
- **[ch08] Stage 3 — Output Processing: Tokens to RequestOutput** _(core)_ · 依赖: ch07, ch06  
  The single-loop OutputProcessor.process_outputs(), per-request RequestState, RequestOutputCollector queues with DELTA merging, stream_interval, finish handling, and parent aggregation for n>1. output_processor.py:572-687, 45-107.
- **[ch09] Incremental Detokenization and Stop Strings** _(advanced)_ · 依赖: ch08  
  IncrementalDetokenizer hierarchy (Fast tokenizers.DecodeStream vs Slow Python), stop-string buffer holdback, min_tokens guard, UTF-8 boundary recovery. detokenizer.py:30-340.
- **[ch10] Logprobs Assembly with Byte-Fallback Correction** _(advanced)_ · 依赖: ch09  
  Sample and prompt logprobs from EngineCore tensors, context-aware UTF-8 multi-byte reconstruction, cumulative logprob tracking, flat vs nested formats. logprobs.py:30-353.

## Part III — Inside EngineCore: The Busy Loop

- **[ch11] EngineCore and the Busy Loop** _(core)_ · 依赖: ch07  
  EngineCore.step() orchestration (schedule -> execute_model non_block -> grammar bitmask -> sample_tokens -> update_from_output), run_busy_loop, lifecycle (pause/resume/sleep), and where the batch queue plugs in. core.py:91-432, 1164-1205.
- **[ch12] Pipeline Parallelism in the Loop: the Batch Queue** _(advanced)_ · 依赖: ch11  
  step_with_batch_queue() buffering SchedulerOutput+future pairs to keep PP stages busy, deferred sampling for structured output, async_scheduling flag. core.py:443-559.

## Part IV — Scheduling and KV Cache

- **[ch13] Continuous Batching with Token Budgets** _(core)_ · 依赖: ch11  
  Token-centric scheduling (no prefill/decode phases), RUNNING-first, max_num_scheduled_tokens budget, SchedulerOutput (NewRequestData vs CachedRequestData persistent-batch protocol), FCFS vs Priority queues. scheduler.py:352-525, output.py.
- **[ch14] Preemption, Waiting Queues, and Update-From-Output** _(advanced)_ · 依赖: ch13  
  Preemption loop on allocation failure, dual waiting/skipped queues for head-of-line avoidance, update_from_output token append + stop detection + request lifecycle/cleanup. scheduler.py:466-846, 1290-1551.
- **[ch15] Paged KV Cache: Block Pool and Prefix Caching** _(core)_ · 依赖: ch13  
  BlockPool, KVCacheBlock, FreeKVCacheBlockQueue doubly-linked LRU, block hashing with extra keys (MM/LoRA/salt), BlockHashToBlockMap, reference counting touch/free. block_pool.py, kv_cache_utils.py.
- **[ch16] Allocation and Multi-Attention Coordination** _(advanced)_ · 依赖: ch15  
  KVCacheManager three-stage allocate_slots (free skipped / prefix hits + external / new + lookahead), KVCacheCoordinator (Unitary vs Hybrid), fixed-point cache-hit search for hybrid full+SWA+MLA models. kv_cache_manager.py:225-416, kv_cache_coordinator.py:487-591.

## Part V — Execution: Workers, Runner, Kernels

- **[ch17] Executors and Worker Lifecycle** _(core)_ · 依赖: ch11  
  Executor.get_class() factory (uni/mp/ray), collective_rpc, MultiprocExecutor shared-memory MessageQueues + FutureWrapper + worker monitoring, WorkerWrapperBase lazy init, GPU memory profiling before KV allocation. abstract.py, multiproc_executor.py, gpu_worker.py.
- **[ch18] The Persistent Batch and _prepare_inputs** _(core)_ · 依赖: ch17, ch14, ch16  
  InputBatch persisting across iterations (slot reuse via batch_update_builder), _update_states reconciling SchedulerOutput, _prepare_inputs (token index_select, positions, block table commit, Triton slot_mapping), CachedRequestState. gpu_model_runner.py:1065-1936, gpu_input_batch.py, block_table.py.
- **[ch19] Decoupled Forward and Sampling: ExecuteModelState** _(advanced)_ · 依赖: ch18  
  Two-phase execute_model() returning None + caching ExecuteModelState, sample_tokens() unpacking it, _bookkeeping_sync writing tokens back, CUDA graph dispatch. gpu_model_runner.py:378-391, 3825-4225.
- **[ch20] Distributed Parallelism: Groups and Collectives** _(advanced)_ · 依赖: ch17  
  GroupCoordinator (TP/PP/DP/EP), CPU(gloo)+device(NCCL) groups, all_reduce/all_gather/reduce_scatter, custom-op registration for torch.compile, P2P for PP. parallel_state.py:290-809, communication_op.py.
- **[ch21] Async Communication and Data Parallelism** _(advanced)_ · 依赖: ch20, ch12  
  AsyncIntermediateTensors lazy PP sync (isend/irecv), DPCoordinator wave synchronization with all-reduce consensus, DP load balancing (DPLBAsyncMPClient). gpu_worker.py:74-861, coordinator.py, core.py:1622-1863.

## Part VI — 模型、算子、注意力与采样 (Models, Kernels, Attention, Sampling)

- **[ch22] Model Definitions and Weight Loading: The Llama Baseline** _(core)_ · 依赖: ch20  
  Establish the canonical vLLM v1 model contract using Llama as the simplest reference. The (vllm_config, prefix) constructor convention (llama.py:256, 354, 518); the four-level module nesting LlamaForCausalLM -> LlamaModel -> LlamaDecoderLayer -> {LlamaAttention(qkv_proj), LlamaMLP(gate_up/down)} (llama.py:81,124,253,350,501); ColumnParallel/RowParallel/QKVParallel linear with TP sharding and get_rope() factory; the forward signature (input_ids, positions, intermediate_tensors, inputs_embeds) and what each argument carries from the model runner; the weight-loading pipeline initialize_model -> load_weights -> process_weights_after_loading, AutoWeightsLoader, and stacked_params_mapping that fuses checkpoint q/k/v into qkv_proj and gate+up into gate_up_proj (llama.py:436-471). This is the foundation every later model chapter (custom ops, DeepSeek-V4) builds on. Contrast hook: explicitly note what Llama LACKS (no compression, no experts, no latent bottleneck, no hybrid-compute residual) so the DeepSeek-V4 chapter can introduce each addition as a delta.
- **[ch23] Custom Operators and torch.compile: Plugging Kernels into the Graph** _(advanced)_ · 依赖: ch22  
  How a model layer becomes a fused, graph-captured kernel. Two-tier dispatch: the CustomOp base class selecting forward_cuda vs forward_native once at __init__ via dispatch_forward() and current_platform (custom_op.py:103-208), the op_registry enable/disable via compilation_config.custom_ops with +name/-name and the enabled()/default_on() check (custom_op.py:272-305), and PluggableLayer for out-of-tree replacement (custom_op.py:32). Binding C++/CUDA: vllm/_custom_ops.py Python wrappers calling torch.ops._C.* (paged_attention, rms_norm), torch.library.register_fake() shape stubs for compile-time shape inference (_custom_ops.py:83-104), and platform.import_kernels() loading vllm._C/_moe_C. Compilation: the @support_torch_compile decorator (decorators.py:118-252) shown decorating the real LlamaModel (llama.py:340), dynamic_arg_dims inference and mark_dynamic/mark_unbacked (decorators.py:414-501), TorchCompileWithNoGuardsWrapper guard dropping (wrapper.py:147), and piecewise CUDA-graph capture/replay via PiecewiseBackend shape-range dispatch + CUDAGraphWrapper batch_descriptor keying (piecewise_backend.py:86-150, cuda_graph.py:145-200). This chapter is the prerequisite vocabulary (torch.ops.vllm.* registration, custom-op dispatch, compile boundaries) that the DeepSeek-V4 chapter relies on for mhc_pre/post, mega-MoE, and MLA custom ops.
- **[ch24] Attention Backends and Metadata** _(advanced)_ · 依赖: ch22, ch23, ch16, ch18  
  AttentionBackend abstraction + registry + selector, CommonAttentionMetadata -> backend-specific metadata (FlashAttention/FlashInfer/Triton), KV cache shape/stride (NHD vs HND), AOT scheduler for FA3, cascade and DCP. Adds an MLA-backend on-ramp: the MLAAttention base class (mla_attention.py:318) and its compute-friendly forward_mha (prefill, mla_attention.py:2251) vs data-movement forward_mqa (decode, mla_attention.py:2324) split, the kv_b_proj W_UK/W_UV transforms and nope/rope query split, MLA backend selection (TRITON_MLA/FLASHMLA_SPARSE/FLASHINFER_MLA_SPARSE, mla_attention.py:374), MLAAttentionSpec KV-cache allocation, and the unified_mla_kv_cache_update / unified_mla_attention_with_output custom ops. This makes MLA an attention-backend concept here so the DeepSeek-V4 chapter can focus on model assembly (the latent projections) rather than re-deriving the attention kernel. backend.py, selector.py, backends/flash_attn.py, mla_attention.py.
- **[ch25] Reading a Full Model: DeepSeek-V4 (MLA + MoE + MTP)** _(high)_ · 依赖: ch22, ch23, ch24  
  The capstone model-reading chapter: assemble DeepSeek-V4 end to end as a stack of deltas over the Llama baseline (ch22). (1) Entry + skeleton: DeepseekV4ForCausalLM -> DeepseekV4Model -> DeepseekV4DecoderLayer (deepseek_v4.py:1507,1220,1089), the forward loop with hc_mult expansion (deepseek_v4.py:1315) and final_norm. (2) MLA model side: DeepseekV4Attention (deepseek_v4.py:921) wrapping DeepseekV4MultiHeadLatentAttentionWrapper (deepseek_v4_attention.py:106) — fused_wqa_wkv low-rank projection (q_lora_rank/kv_lora_rank), q/kv RMSNorm, wq_b latent->full-q expansion, wo_a/wo_b output projection with inverse RoPE + FP8 einsum, and multi-stream gemm parallelism (deepseek_v4_attention.py:337); the actual attention kernel is deferred to ch24. (3) Dual-backend MoE: DeepseekV4MoE (deepseek_v4.py:707) selecting DeepseekV4MegaMoEExperts (deep_gemm, deepseek_v4.py:392) via use_mega_moe vs the tensor-parallel FusedMoE path, gate router topk + shared_experts aggregation (deepseek_v4.py:854-919), expert sharding via FusedMoE expert_map (fused_moe/layer.py:71-158). (4) Hybrid-Computation residual: hc_pre/hc_post via torch.ops.vllm.mhc_pre and the hc_head final mixing (deepseek_v4.py:1166-1193,1274-1329) with learnable hc_{attn,ffn}_{fn,base,scale}. (5) MTP draft interface: the pre-hc_head _mtp_hidden_buffer (deepseek_v4.py:1297-1327) and DeepSeekV4MTP (deepseek_v4_mtp.py:244) consumed by speculative decoding (ch28). (6) Weight loading deltas: stacked expert mapping, expert-dtype resolution (DeepseekV4FP8Config MXFP4 vs FP8, deepseek_v4.py:121), float8_e8m0fnu -> uint8 raw-bit view, and lazy finalize_weights for deep_gemm scale layout. Every subsystem (custom ops, compile boundary, attention backend, MoE) is cross-referenced to its dedicated chapter.
- **[ch26] From Model Code to Architecture Diagram: A Worked Example** _(intermediate)_ · 依赖: ch25  
  A methodology chapter that turns the DeepSeek-V4 source (ch25) into a precise architecture diagram, teaching a repeatable code->diagram procedure the reader can apply to any model. Procedure: (1) extract the module tree by reading __init__ (which nn.Module owns which children) — DeepseekV4ForCausalLM -> DeepseekV4Model -> DeepseekV4DecoderLayer[] -> {DeepseekV4Attention(MLA wrapper), DeepseekV4MoE(gate + experts + shared_experts)}; (2) extract the data-flow edges by reading forward() in tensor order (embed -> hc expand -> per-layer attn/MoE residual with hc_pre/post -> hc_head -> final_norm -> lm_head), annotating tensor shapes [B,H]/[B,hc_mult,H] and the prefill/decode branch; (3) mark subsystem boundaries (which boxes are custom ops vs torch.compile regions vs collective ops) so the diagram doubles as a subsystem map; (4) render with the svg-diagram skill (Python script -> xmllint validate -> ImageMagick PNG), using nested containers for the module tree and directed arrows for the forward data flow, exactly the dense-many-element case the skill exists for. Deliverable: a layered SVG architecture diagram of DeepSeek-V4 plus the narration of how each box/edge was read out of the source. Doubles as the template for diagrams in earlier chapters.
- **[ch27] The Sampling Pipeline** _(core)_ · 依赖: ch19  
  Sampler 9-step pipeline (raw logprobs, penalties, bad words, logits processors argmax-invariant vs not, temperature, top-k/top-p), multi-backend TopKTopPSampler dispatch, persistent batch BatchUpdate for logits processors. sampler.py, ops/topk_topp_sampler.py, logits_processor/.
- **[ch28] Speculative Decoding: Proposers and Rejection Sampling** _(advanced)_ · 依赖: ch27, ch25, ch14, ch24  
  Proposers (ngram CPU/GPU, EAGLE, DFlash, and the DeepSeek-V4 MTP draft from ch25 via DeepSeekV4MTP), SpecDecodeMetadata index indirection, rejection sampling (probabilistic vs synthetic) against target logits, fused Triton slot-mapping kernels with rejection masking. llm_base_proposer.py, rejection_sampler.py, spec_decode/metadata.py, deepseek_v4_mtp.py, utils.py.

## Part VII — Prefill/Decode 分离 (Disaggregation)

- **[ch29] PD Disaggregation I — The KV Connector Contract and Scheduler Integration** _(advanced)_ · 依赖: ch14, ch16, ch20  
  Why prefill and decode get split across engines, and the abstraction that makes it work. The KVConnectorBase_V1 role-split contract (base.py:170): KVConnectorRole.SCHEDULER (metadata planning + state) vs WORKER (load/save execution), built as two separate instances to enforce separation (factory.py:69-82). Scheduler-side lifecycle inside scheduler.py: get_num_new_matched_tokens for remote-prefix matching (scheduler.py:620-647), update_state_after_alloc after block reservation, the WAITING_FOR_REMOTE_KVS state machine and skipped-waiting queue (scheduler.py:770-805), build_connector_meta at step-end (scheduler.py:947-950), async-free decision via request_finished, and the load-error recovery path get_block_ids_with_load_errors -> _update_requests_with_invalid_blocks resetting num_computed_tokens to the longest valid prefix. The factory registry, lazy-load-by-name, signature evolution, and HMA/SupportsHMA gating (factory.py:27-143, base.py:84-113). Establishes the metadata-driven, plan-then-execute design that Chapter ch30 turns into actual cross-process transfers.
- **[ch30] PD Disaggregation II — Worker Execution and Pluggable Backends (P2P/NIXL/Offloading)** _(high)_ · 依赖: ch29, ch19, ch24  
  The worker half of the contract and the concrete transports. Worker-side lifecycle: ActiveKVConnector.pre_forward binding metadata + start_load_kv, per-layer save_kv_layer during the model forward, post_forward wait_for_save + get_finished (kv_connector.py:62-102), wrapped by KVConnectorModelRunnerMixin's _get_kv_connector_output context manager and the cross-layer-blocks / uniform-KV-cache optimization (kv_connector_model_runner_mixin.py:84-190). Producer/consumer role split via is_kv_producer (prefill sends, decode receives). Three backends compared: P2pNcclConnector direct NCCL send/recv with extract/inject per layer (p2p_nccl_connector.py:74-312); OffloadingConnector facade delegating to scheduler/worker with OffloadingManager, HMA, and async transfer-job tracking (offloading_connector.py:46-192); NixlConnector handshake + side-channel metadata negotiation and async P2P (nixl/connector.py). The end-to-end data-flow walk: scheduler plan -> metadata delivery -> worker load -> forward+save -> get_finished -> scheduler completion, including the layer-by-layer pipelining interface (start_load_kv bulk async, wait_for_layer_load, save_kv_layer, wait_for_save). Closes with how this integrates with the model runner forward (ch19) and where it surfaces in engine operation.

## Part VIII — 服务接口 (The Serving Surface)

- **[ch31] The Offline LLM API** _(core)_ · 依赖: ch08, ch11  
  LLM class: generate/chat/embed/encode, EngineArgs -> LLMEngine over EngineCore, request batching and rendering, the sync LLMEngine.step() path vs async. llm.py, v1/engine/llm_engine.py.
- **[ch32] The OpenAI-Compatible Server** _(core)_ · 依赖: ch31, ch04  
  FastAPI lifespan + build_async_engine_client, OpenAIServing base + chat/completion handlers, Renderer, tool/reasoning parsers, SSE streaming vs JSON, launcher (uvicorn, watchdog, SSL). api_server.py, openai/*/serving.py, launcher.py.
- **[ch33] Advanced Engine Operation: Elastic Scaling and Multi-Turn** _(advanced)_ · 依赖: ch32, ch21, ch14  
  Elastic EP scale-up/scale-down state machine, Responses API multi-turn/tool context (PD/KV-connector material moved to Part VII).