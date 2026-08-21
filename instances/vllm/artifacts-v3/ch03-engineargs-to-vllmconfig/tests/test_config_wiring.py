"""TDD tests for the v3 ch03 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

These assert the *observable vLLM behavior* this chapter teaches:
- flat EngineArgs defaults borrowed from sub-Config class attributes (single
  source of truth),
- get_batch_defaults: (device memory x device name x usage context) defaults,
- ParallelConfig.__post_init__ backend derivation (uni/mp),
- factory #1 Executor.get_class + the supports_async_scheduling reverse query,
- the async_scheduling tri-state decision (True hard-fails / None auto-resolves
  to True on v0.27.1 defaults),
- O0-O3 presets applied via recursive only-fill-None (user > env > preset),
- compute_hash scope (TP yes / max_num_batched_tokens yes / max_num_seqs no /
  executor backend no),
- factory #3 EngineCoreClient.make_client 2D table,
- factory #2 SchedulerConfig.get_scheduler_cls,
- end-to-end inproc assembly: LLM -> EngineArgs -> VllmConfig -> factories ->
  EngineCore.

Pure unit tests (no `import vllm`, no torch/CUDA): they run on the CPU host
via `python3 -m pytest tests/`. Every assertion is grounded in the v0.27.1
source (line refs in comments point into the real tree).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the companion module directly by path (no package install needed).
_IMPL = Path(__file__).resolve().parent.parent / "implementation" / "config_wiring.py"
_spec = importlib.util.spec_from_file_location("config_wiring", _IMPL)
cw = importlib.util.module_from_spec(_spec)
sys.modules["config_wiring"] = cw
_spec.loader.exec_module(cw)

GiB = 1024**3
H100 = dict(device_type="cuda", _is_cuda=True, _device_count=1,
            _total_memory=80 * GiB, _device_name="NVIDIA H100 80GB HBM3")
A100_80G = dict(device_type="cuda", _is_cuda=True, _device_count=1,
                _total_memory=80 * GiB, _device_name="NVIDIA A100-SXM4-80GB")
SMALL_GPU = dict(device_type="cuda", _is_cuda=True, _device_count=1,
                 _total_memory=24 * GiB, _device_name="NVIDIA RTX 4090")


@pytest.fixture(autouse=True)
def _single_h100_platform():
    """Default each test to a single-H100 host so defaults are deterministic."""
    saved = cw.current_platform
    saved_mp = cw.envs.VLLM_ENABLE_V1_MULTIPROCESSING
    cw.current_platform = cw.Platform(**H100)
    cw.envs.VLLM_ENABLE_V1_MULTIPROCESSING = True
    yield
    cw.current_platform = saved
    cw.envs.VLLM_ENABLE_V1_MULTIPROCESSING = saved_mp


def make_cfg(**kwargs) -> "cw.VllmConfig":
    """create_engine_config on the LLM_CLASS usage context (chapters'主线)."""
    kwargs.setdefault("model", "test-model")
    kwargs.setdefault("max_model_len", 2048)
    return cw.EngineArgs(**kwargs).create_engine_config(cw.UsageContext.LLM_CLASS)


# ---------------------------------------------------------------------------
# Station 1-2: flat EngineArgs borrows sub-Config defaults (single source of
# truth) — vllm/engine/arg_utils.py:L421-L463
# ---------------------------------------------------------------------------

def test_engine_args_defaults_borrow_subconfig_attrs():
    assert cw.EngineArgs.model == cw.ModelConfig.model == "Qwen/Qwen3-0.6B"
    assert cw.EngineArgs.kv_cache_dtype == cw.CacheConfig.cache_dtype == "auto"
    # v0.27.1 default is 0.92 (vllm/config/cache.py:L68).
    assert cw.EngineArgs.gpu_memory_utilization == cw.CacheConfig.gpu_memory_utilization == 0.92
    assert cw.EngineArgs.enable_prefix_caching is None   # L518: None sentinel
    assert cw.EngineArgs.block_size is None              # L517
    assert cw.EngineArgs.max_num_batched_tokens is None  # L533
    assert cw.EngineArgs.max_num_seqs is None            # L536
    assert cw.EngineArgs.enable_chunked_prefill is None  # L619
    assert cw.EngineArgs.async_scheduling is None        # L727
    assert cw.EngineArgs.optimization_level == cw.OptimizationLevel.O2
    assert cw.EngineArgs.performance_mode == "balanced"


def test_post_init_upgrades_dict_subconfigs():
    # EngineArgs(compilation_config={...}) without manual construction (L755+).
    args = cw.EngineArgs(compilation_config={"use_inductor_graph_partition": True})
    assert isinstance(args.compilation_config, cw.CompilationConfig)
    assert args.compilation_config.use_inductor_graph_partition is True
    args = cw.EngineArgs(kernel_config={"enable_flashinfer_autotune": False})
    assert isinstance(args.kernel_config, cw.KernelConfig)


def test_post_init_fault_tolerance_auto_enable():
    args = cw.EngineArgs(fault_tolerance_config={})
    assert args.enable_fault_tolerance is True  # L773-L782 warning + auto-enable


# ---------------------------------------------------------------------------
# Station 5: batch defaults are a function of (memory, device, usage context)
# — vllm/engine/arg_utils.py:L2515-L2596
# ---------------------------------------------------------------------------

def test_get_batch_defaults_h100_vs_a100_vs_small():
    ws = 1
    toks, seqs = cw.EngineArgs.get_batch_defaults(ws)
    assert toks[cw.UsageContext.LLM_CLASS] == 16384
    assert toks[cw.UsageContext.OPENAI_API_SERVER] == 8192
    assert seqs[cw.UsageContext.LLM_CLASS] == 1024

    cw.current_platform = cw.Platform(**A100_80G)
    toks, seqs = cw.EngineArgs.get_batch_defaults(ws)
    # A100 80GiB still takes the *smaller* branch: #17885 regression guard.
    assert toks[cw.UsageContext.LLM_CLASS] == 8192
    assert toks[cw.UsageContext.OPENAI_API_SERVER] == 2048
    assert seqs[cw.UsageContext.LLM_CLASS] == 256

    cw.current_platform = cw.Platform(**SMALL_GPU)
    toks, seqs = cw.EngineArgs.get_batch_defaults(ws)
    assert toks[cw.UsageContext.LLM_CLASS] == 8192
    assert seqs[cw.UsageContext.LLM_CLASS] == 256


def test_get_batch_defaults_probe_failure_falls_back():
    cw.current_platform = cw.Platform(device_type="cuda", _is_cuda=True,
                                      _device_count=0, _total_memory=None,
                                      _device_name=None)
    toks, _ = cw.EngineArgs.get_batch_defaults(1)
    assert toks[cw.UsageContext.LLM_CLASS] == 8192  # non-H100 defaults


def test_batch_defaults_end_to_end_usage_context_and_throughput():
    # LLM_CLASS on H100 -> 16384 tokens / 1024 seqs.
    cfg = make_cfg()
    assert cfg.scheduler_config.max_num_batched_tokens == 16384
    assert cfg.scheduler_config.max_num_seqs == 1024
    # throughput mode doubles *only* fields the user left unset (L2745-L2749).
    cfg = make_cfg(performance_mode="throughput")
    assert cfg.scheduler_config.max_num_batched_tokens == 16384 * 2
    assert cfg.scheduler_config.max_num_seqs == 1024 * 2
    cfg = make_cfg(performance_mode="throughput", max_num_batched_tokens=512)
    assert cfg.scheduler_config.max_num_batched_tokens == 512  # user wins
    # seqs doubled to 2048, then floored to the token budget (L2793-L2795).
    assert cfg.scheduler_config.max_num_seqs == 512


def test_batch_defaults_non_chunked_raises_to_max_model_len():
    # L2755-L2760: no chunked prefill -> budget at least max_model_len.
    cfg = make_cfg(enable_chunked_prefill=False, max_model_len=32768)
    assert cfg.scheduler_config.max_num_batched_tokens == 32768
    # ... and capped at max_num_seqs * max_model_len (L2782-L2785): here the
    # cap (131072) is above the budget, so the budget itself stands.
    cfg = make_cfg(enable_chunked_prefill=False, max_model_len=65536,
                   max_num_seqs=2)
    assert cfg.scheduler_config.max_num_batched_tokens == 65536


# ---------------------------------------------------------------------------
# Station 7 / mechanism ch03-backend-derivation — vllm/config/parallel.py:
# L911-L956
# ---------------------------------------------------------------------------

def test_parallel_backend_derivation_uni_and_mp():
    p = cw.ParallelConfig(tensor_parallel_size=1)
    assert p.world_size == 1
    assert p.distributed_executor_backend == "uni"  # L955-L956

    cw.current_platform = cw.Platform(**{**H100, "_device_count": 2})
    p = cw.ParallelConfig(tensor_parallel_size=2)
    assert p.world_size == 2
    assert p.distributed_executor_backend == "mp"  # L917 default under CUDA

    # Not enough GPUs on the node -> hard error (L923-L934).
    cw.current_platform = cw.Platform(**H100)  # 1 device
    with pytest.raises(ValueError, match="larger than the number of"):
        cw.ParallelConfig(tensor_parallel_size=2)


# ---------------------------------------------------------------------------
# Factory #1: Executor.get_class — vllm/v1/executor/abstract.py:L47-L92
# ---------------------------------------------------------------------------

def _vcfg(backend):
    return cw.VllmConfig(parallel_config=cw.ParallelConfig(
        distributed_executor_backend=backend))


def test_executor_factory_lookup_table():
    assert cw.Executor.get_class(_vcfg("uni")) is cw.UniProcExecutor
    assert cw.Executor.get_class(_vcfg("mp")) is cw.MultiprocExecutor
    assert cw.Executor.get_class(_vcfg("ray")) is cw.RayDistributedExecutor
    assert cw.Executor.get_class(
        _vcfg("external_launcher")) is cw.ExecutorWithExternalLauncher

    class MyExecutor(cw.Executor):
        pass

    assert cw.Executor.get_class(_vcfg(MyExecutor)) is MyExecutor
    with pytest.raises(TypeError, match="must be a subclass of Executor"):
        cw.Executor.get_class(_vcfg(int))
    with pytest.raises(ValueError, match="Unknown distributed executor"):
        cw.Executor.get_class(_vcfg("bogus"))


def test_supports_async_scheduling_reverse_query():
    # Base class default False (abstract.py:L364-L368); uni/mp override True.
    assert cw.Executor.supports_async_scheduling() is False
    assert cw.UniProcExecutor.supports_async_scheduling() is True
    assert cw.MultiprocExecutor.supports_async_scheduling() is True
    # external_launcher subclasses UniProcExecutor and inherits True.


# ---------------------------------------------------------------------------
# Station 10-11: VllmConfig.__post_init__ cross-checks + async tri-state
# — vllm/config/vllm.py:L972-L1143
# ---------------------------------------------------------------------------

def test_model_parallel_head_divisibility_crosscheck():
    cw.current_platform = cw.Platform(**{**H100, "_device_count": 8})
    mc = cw.ModelConfig(model="m", max_model_len=2048,
                        model_arch_config=cw.ModelArchConfig(
                            total_num_attention_heads=16))
    good = cw.ParallelConfig(tensor_parallel_size=2)
    mc.verify_with_parallel_config(good)  # 16 % 2 == 0 -> ok
    bad = cw.ParallelConfig(tensor_parallel_size=3)
    with pytest.raises(ValueError, match="divisible by tensor parallel size"):
        mc.verify_with_parallel_config(bad)


def test_async_scheduling_none_defaults_true_on_v0271():
    # The v0.27.1 default heartbeat: None + generate model + uni executor ->
    # True (vllm.py:L1142-L1143).
    cfg = make_cfg()
    assert cfg.scheduler_config.async_scheduling is True
    assert cfg.parallel_config.disable_nccl_for_dp_synchronization is True


def test_async_scheduling_pooling_disables():
    cfg = make_cfg(runner="pooling")
    assert cfg.scheduler_config.async_scheduling is False


def test_async_scheduling_explicit_true_hard_fails_on_executor():
    class NoAsyncExecutor(cw.Executor):
        pass  # inherits supports_async_scheduling() -> False

    with pytest.raises(ValueError, match="does not support async scheduling"):
        make_cfg(async_scheduling=True,
                 distributed_executor_backend=NoAsyncExecutor)


def test_async_scheduling_explicit_true_hard_fails_on_spec_method():
    with pytest.raises(ValueError, match="EAGLE/MTP/Draft Model/NGram GPU"):
        make_cfg(async_scheduling=True,
                 speculative_config={"method": "medusa"})


def test_async_scheduling_explicit_true_allows_eagle():
    cfg = make_cfg(async_scheduling=True,
                   speculative_config={"method": "eagle"})
    assert cfg.scheduler_config.async_scheduling is True


def test_async_scheduling_explicit_false_skips_checks():
    cfg = make_cfg(async_scheduling=False, runner="pooling")
    assert cfg.scheduler_config.async_scheduling is False
    assert cfg.parallel_config.disable_nccl_for_dp_synchronization is False


# ---------------------------------------------------------------------------
# Station 12 + mechanism ch03-optimization-levels — vllm/config/vllm.py:
# L104-L327, L811-L853, L1193-L1300
# ---------------------------------------------------------------------------

def test_o2_default_lands_compile_and_full_and_piecewise():
    cfg = make_cfg()
    cc = cfg.compilation_config
    assert cc.mode == cw.CompilationMode.VLLM_COMPILE
    assert cc.cudagraph_mode == cw.CUDAGraphMode.FULL_AND_PIECEWISE
    assert cc.pass_config.fuse_norm_quant is False  # predicate resolved
    assert cfg.kernel_config.enable_flashinfer_autotune is True
    # inductor + compiling -> custom_ops default "none" (L1285-L1292).
    assert "none" in cc.custom_ops


def test_o0_is_strictly_eager():
    cfg = make_cfg(optimization_level=cw.OptimizationLevel.O0)
    cc = cfg.compilation_config
    assert cc.mode == cw.CompilationMode.NONE
    assert cc.cudagraph_mode == cw.CUDAGraphMode.NONE
    assert cc.pass_config.fuse_norm_quant is False
    assert cfg.kernel_config.enable_flashinfer_autotune is False
    assert "all" in cc.custom_ops  # mode NONE -> "all" (L1285-L1292)


def test_o1_is_piecewise():
    cfg = make_cfg(optimization_level=cw.OptimizationLevel.O1)
    assert cfg.compilation_config.cudagraph_mode == cw.CUDAGraphMode.PIECEWISE


def test_user_explicit_beats_preset():
    cfg = make_cfg(compilation_config={"cudagraph_mode": cw.CUDAGraphMode.PIECEWISE})
    assert cfg.compilation_config.cudagraph_mode == cw.CUDAGraphMode.PIECEWISE


def test_enforce_eager_overrides_everything():
    cfg = make_cfg(enforce_eager=True)
    cc = cfg.compilation_config
    assert cc.mode == cw.CompilationMode.NONE
    assert cc.cudagraph_mode == cw.CUDAGraphMode.NONE
    assert cc.max_cudagraph_capture_size == 0      # L1424-L1430
    assert cc.cudagraph_capture_sizes == []
    assert cc.cudagraph_num_of_warmups is None     # warmups only set otherwise


def test_torch_compile_disable_env_beats_preset(monkeypatch):
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    cfg = make_cfg()
    cc = cfg.compilation_config
    assert cc.mode == cw.CompilationMode.NONE       # L1201-L1206
    assert cc.cudagraph_mode == cw.CUDAGraphMode.NONE  # guard L1310-L1321


# ---------------------------------------------------------------------------
# Station 17 + mechanism ch03-compute-hash — vllm/config/vllm.py:L431-L537,
# vllm/config/scheduler.py:L193-L219, vllm/config/parallel.py:L774-L830
# ---------------------------------------------------------------------------

def test_compute_hash_is_ten_hex_chars_and_deterministic():
    h1, h2 = make_cfg().compute_hash(), make_cfg().compute_hash()
    assert h1 == h2
    assert len(h1) == 10
    int(h1, 16)  # hex


def test_compute_hash_scope_what_matters():
    base = make_cfg(max_num_batched_tokens=2048, max_num_seqs=128)
    h0 = base.compute_hash()
    # max_num_seqs: NOT a graph factor (scheduler.py hash only collects the
    # token budget).
    assert make_cfg(max_num_batched_tokens=2048, max_num_seqs=256).compute_hash() == h0
    # max_num_batched_tokens: IS a graph factor (LoRA buffers + index width).
    assert make_cfg(max_num_batched_tokens=4096, max_num_seqs=128).compute_hash() != h0
    # Executor backend: NOT a graph factor (parallel.py ignored_factors).
    other = make_cfg(max_num_batched_tokens=2048, max_num_seqs=128)
    other.parallel_config.distributed_executor_backend = "mp"
    assert other.compute_hash() == h0
    # TP: IS a graph factor (collectives enter the graph).
    cw.current_platform = cw.Platform(**{**H100, "_device_count": 2})
    assert make_cfg(max_num_batched_tokens=2048, max_num_seqs=128,
                    tensor_parallel_size=2).compute_hash() != h0


def test_scheduler_compute_hash_only_collects_token_budget():
    s1 = cw.SchedulerConfig(max_model_len=2048, is_encoder_decoder=False,
                            max_num_batched_tokens=2048, max_num_seqs=128)
    s2 = cw.SchedulerConfig(max_model_len=8192, is_encoder_decoder=False,
                            max_num_batched_tokens=2048, max_num_seqs=999)
    assert s1.compute_hash() == s2.compute_hash()
    s3 = cw.SchedulerConfig(max_model_len=2048, is_encoder_decoder=False,
                            max_num_batched_tokens=4096, max_num_seqs=128)
    assert s3.compute_hash() != s1.compute_hash()


# ---------------------------------------------------------------------------
# Factory #3 + #2 — vllm/v1/engine/core_client.py:L89-L139,
# vllm/config/scheduler.py:L170-L190
# ---------------------------------------------------------------------------

def test_make_client_two_by_two_table():
    cfg = make_cfg()
    ex = cw.UniProcExecutor
    assert isinstance(
        cw.EngineCoreClient.make_client(False, False, cfg, ex, True),
        cw.InprocClient)
    assert isinstance(
        cw.EngineCoreClient.make_client(True, False, cfg, ex, True),
        cw.SyncMPClient)
    assert isinstance(
        cw.EngineCoreClient.make_client(True, True, cfg, ex, True),
        cw.AsyncMPClient)  # DP=1 -> plain AsyncMPClient
    with pytest.raises(NotImplementedError):
        cw.EngineCoreClient.make_client(False, True, cfg, ex, True)


def test_get_scheduler_cls_factory():
    kw = dict(max_model_len=2048, is_encoder_decoder=False)
    s = cw.SchedulerConfig(async_scheduling=True, **kw)
    assert s.get_scheduler_cls() is cw.AsyncScheduler
    s = cw.SchedulerConfig(async_scheduling=False, **kw)
    assert s.get_scheduler_cls() is cw.Scheduler
    # Custom class passes through with a warning (scheduler.py:L189-L191).
    class MySched:
        pass
    s = cw.SchedulerConfig(scheduler_cls=MySched, **kw)
    assert s.get_scheduler_cls() is MySched


def test_scheduler_verify_max_model_len_guards():
    kw = dict(max_model_len=2048, is_encoder_decoder=False)
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        cw.SchedulerConfig(max_num_batched_tokens=1024,
                           enable_chunked_prefill=False, **kw)
    with pytest.raises(ValueError, match="greater than or equal to max_num_seqs"):
        cw.SchedulerConfig(max_num_batched_tokens=8, max_num_seqs=16, **kw)


# ---------------------------------------------------------------------------
# End-to-end: the minimal assembly line runs inproc on host.
# LLM -> EngineArgs -> create_engine_config -> VllmConfig -> factories ->
# EngineCore (vllm/entrypoints/llm.py:L295-L341, vllm/v1/engine/core.py)
# ---------------------------------------------------------------------------

def test_llm_end_to_end_inproc_assembly():
    cw.envs.VLLM_ENABLE_V1_MULTIPROCESSING = False
    llm = cw.LLM(model="test-model", max_model_len=2048, enforce_eager=True)
    assert isinstance(llm.llm_engine, cw.LLMEngine)
    vc = llm.llm_engine.vllm_config
    assert vc.model_config.model == "test-model"
    assert vc.scheduler_config.max_num_batched_tokens == 16384  # H100/LLM_CLASS
    # factory #3 picked the inproc escape hatch.
    client = llm.llm_engine.engine_core
    assert isinstance(client, cw.InprocClient)
    core = client.engine_core
    assert isinstance(core, cw.EngineCore)
    # factory #1: uni world -> UniProcExecutor, instantiated *here* (L132).
    assert isinstance(core.model_executor, cw.UniProcExecutor)
    # factory #2: async_scheduling None resolved True -> AsyncScheduler (L160).
    assert isinstance(core.scheduler, cw.AsyncScheduler)
    assert isinstance(core.structured_output_manager, cw.StructuredOutputManager)
    # graph fingerprint available for the compile cache key.
    assert len(vc.compute_hash()) == 10


def test_llm_engine_multiprocessing_default_spawns_mp_client():
    # envs.VLLM_ENABLE_V1_MULTIPROCESSING defaults True (envs.py:L149):
    # even offline LLM talks to a separate EngineCore process (WC2).
    args = cw.EngineArgs(model="test-model", max_model_len=2048,
                         enforce_eager=True)
    engine = cw.LLMEngine.from_engine_args(
        args, usage_context=cw.UsageContext.LLM_CLASS)
    assert isinstance(engine.engine_core, cw.SyncMPClient)


def test_async_engine_args_same_assembly_line():
    aea = cw.AsyncEngineArgs(model="test-model", max_model_len=2048,
                             enforce_eager=True, enable_log_requests=True)
    assert isinstance(aea, cw.EngineArgs)
    cfg = aea.create_engine_config(cw.UsageContext.OPENAI_API_SERVER)
    # H100 + API server face is more conservative: 8192 tokens.
    assert cfg.scheduler_config.max_num_batched_tokens == 8192
    aengine = cw.AsyncLLM.from_engine_args(aea)
    assert isinstance(aengine, cw.AsyncLLM)
    # Async face goes through make_async_mp_client -> AsyncMPClient.
    assert isinstance(aengine.engine_core, cw.AsyncMPClient)
