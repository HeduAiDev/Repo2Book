# ch03 Implementation Notes — subtract-only companion

Source pin: `f3fef123`. Companion: `implementation/config_wiring.py`. Tests:
`tests/test_config_wiring.py` (35 passing, pure host unit tests, no `import vllm`).

## What this companion is
A faithful, *subtract-only* subset of the real vLLM config-assembly + wiring path:
flat `EngineArgs` → `create_engine_config()` → structured `VllmConfig` →
`VllmConfig.__post_init__` cross-config derivation → the three factories
(`Executor.get_class` / `SchedulerConfig.get_scheduler_cls` /
`EngineCoreClient.make_client`) → `EngineCore` convergence. Same names, same
structure, same control flow as vLLM. Every deletion is an approved
`subtraction_plan.delete` item, marked `# SUBTRACTED:`. Every `def`/`class`
carries a `# SOURCE: vllm/...` ref.

Verification: `lint_fidelity` passes with 0 BLOCKING; deleting every
`# SUBTRACTED:` branch from real vLLM should reproduce (approximately) this file.

## Host-runnability seams (faithful, not invented)
- Sub-Config dataclasses (`ModelConfig`/`CacheConfig`/`ParallelConfig`/
  `SchedulerConfig`/`CompilationConfig`/`KernelConfig`/`DeviceConfig`) are reduced
  to the *fields this chapter's control flow reads*. Every kept field maps 1:1 to
  a real vLLM attribute of the same name; unread fields are SUBTRACTED, never
  renamed/invented.
- `current_platform` is a tiny injectable `Platform` stub (the same
  `current_platform.is_cuda()/device_count()/...` seam vLLM goes through) so the
  backend-derivation and compilation branches run on a CPU host. Not a toy model.
- Fusion predicates keep their *function-valued preset* shape (the dossier elides
  the bodies, not the mechanism); bodies reduced to platform-independent defaults.
- `__version__`, `safe_hash`, `os.environ`, `resolve_obj_by_qualname` replaced by
  small private helpers (`_hash10`/`_safe_hash`/`_env_get`/`_resolve_obj_by_qualname`)
  so `compute_hash` etc. run without the vllm package; algorithm is identical.

## 1:1 Source Map (companion ↔ real vLLM ↔ change ↔ why)

| Companion symbol | Real vLLM source | Change | Why faithful |
|---|---|---|---|
| `EngineArgs` (+ field defaults from sub-Configs) | `vllm/engine/arg_utils.py:L403-L462` | kept representative fields; SUBTRACTED ~hundreds of MM/LoRA/spec/KV/quant fields | single-source-of-truth default pattern preserved; deleted fields off this chapter's single-DP path |
| `EngineArgs.__post_init__` | `vllm/engine/arg_utils.py:L690-L720` | kept compilation_config dict→obj promotion; SUBTRACTED other promotions + plugin load | one representative promotion shows the mechanism (dossier delete) |
| `EngineArgs.create_model_config` | `vllm/engine/arg_utils.py` create_model_config | SUBTRACTED HF download/parse, dtype/runner inference | heaviest IO step; flat fields packed directly, derived flags keep defaults so downstream decisions stay observable |
| `EngineArgs.create_engine_config` | `vllm/engine/arg_utils.py:L1622-L2177` | kept Cache/Parallel/Scheduler/Compilation repacking + VllmConfig build; SUBTRACTED Ray/DP-LB/TurboQuant/other sub-config repacking | the first-level mapping; deleted items are dossier `delete` (Ray/DP/quant edge) |
| `VllmConfig` (+fields) | `vllm/config/vllm.py:L274-L366` | kept model/cache/parallel/scheduler/device/kernel/compilation + optimization_level/performance_mode; SUBTRACTED load/offload/attention/mamba/... | aggregate-config role intact; deleted fields not read by this chapter |
| `VllmConfig.compute_hash` | `vllm/config/vllm.py:L367-L473` | kept version + model/cache/parallel/scheduler/compilation/kernel factor appends; SUBTRACTED the rest | identical collect→hash→`[:10]` algorithm; deleted appends are same-shape (dossier delete) |
| `VllmConfig._set_config_default` | `vllm/config/vllm.py:L637-L650` | verbatim | only-fill-None + callable-eval rule is load-bearing for opt-level priority |
| `VllmConfig._apply_optimization_level_defaults` | `vllm/config/vllm.py:L652-L679` | verbatim (incl. nested `apply_recursive`) | recursive only-None apply is the opt-level mechanism |
| `VllmConfig.__post_init__` | `vllm/config/vllm.py:L721-L981` | kept async_scheduling tri-state, enforce_eager/env override, mode default, opt-level apply; SUBTRACTED deep_gemm/Turing/blocked-weights/NCCL-DP/cascade/mamba/spec-detail branches | the three kept decisions are the chapter's spine; deleted are hardware/feature edges (dossier delete) keeping original control flow |
| `OptimizationLevel`, `OPTIMIZATION_LEVEL_0{0,1,2,3}`, `OPTIMIZATION_LEVEL_TO_CONFIG` | `vllm/config/vllm.py:L68-L80, L184-L270` | verbatim presets; predicate bodies reduced | preset-dict + lookup-table is the literal mechanism; O3 is O2 (real) |
| `ParallelConfig.__post_init__` | `vllm/config/parallel.py:L829-L874` | kept None→mp/uni derivation + world-size>GPU raise; SUBTRACTED Ray/TPU/DP-backend branches | single-node mp/uni story exact; Ray is dossier delete |
| `Executor.get_class` | `vllm/v1/executor/abstract.py:L47-L92` | kept type/mp/uni/external_launcher + error branches; SUBTRACTED Ray V2/V1 + qualname dynamic resolve | factory lookup pattern intact; Ray/dynamic are dossier delete |
| `Executor.supports_async_scheduling` (+subclasses) | `vllm/v1/executor/abstract.py` + executor modules | kept per-class boolean | feeds the async_scheduling decision (real return values) |
| `SchedulerConfig.get_scheduler_cls` | `vllm/config/scheduler.py:L168-L188` | kept async→Async/else Scheduler + custom-class; SUBTRACTED warning log | factory #2 logic verbatim |
| `EngineCoreClient.make_client` / `make_async_mp_client` | `vllm/v1/engine/core_client.py:L80-L130` | verbatim 2-D + DP-LB selection (marker-stub clients) | factory #3 logic intact; client bodies are runtime IPC (out of scope) |
| `EngineCore.__init__` | `vllm/v1/engine/core.py:L116-L153` | kept executor instantiation + get_scheduler_cls selection; SUBTRACTED KV/profiling/PP/GC init | convergence point of the three factories; chapter stops at scheduler selection |
| `LLMEngine.from_engine_args` / `__init__` | `vllm/v1/engine/llm_engine.py:L151-L177` | kept create_engine_config + get_class + make_client(asyncio_mode=False); SUBTRACTED env override + tokenizer/output plumbing | entry-facade data-flow start; asyncio_mode=False is the load-bearing fact (AsyncLLM is ch04) |

## Foreshadowing seams (kept observable for later chapters)
- `async_scheduling` tri-state + `AsyncScheduler` selection → ch06.
- `make_async_mp_client` / `AsyncMPClient` → ch04.
- `CacheConfig.enable_prefix_caching` / `block_size` → ch05.
- `compute_hash` as torch.compile cache key → compilation chapter.
