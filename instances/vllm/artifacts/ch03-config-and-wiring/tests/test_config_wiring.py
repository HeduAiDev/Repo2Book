"""TDD tests for the ch03 subtract-only companion.

These assert the *observable vLLM behavior* this chapter teaches — the two-level
mapping (flat EngineArgs -> structured VllmConfig -> implementation class), the
O0-O3 optimization-level application, the async_scheduling tri-state decision,
the three factories, and compute_hash. Pure unit tests (no `import vllm`), so
they run on a CPU host: `python3 -m pytest`.

Behaviors are grounded in real vLLM source at pin f3fef123 (line refs in each
test name / comment point to the original).
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Load the companion module directly by path (no package install needed).
_IMPL = Path(__file__).resolve().parent.parent / "implementation" / "config_wiring.py"
_spec = importlib.util.spec_from_file_location("config_wiring", _IMPL)
cw = importlib.util.module_from_spec(_spec)
sys.modules["config_wiring"] = cw
_spec.loader.exec_module(cw)


@pytest.fixture(autouse=True)
def _single_gpu_platform():
    """Default each test to a single-CUDA-GPU host so backend derivation is
    deterministic. Mirrors current_platform behavior on a 1-GPU node."""
    saved = cw.current_platform
    cw.current_platform = cw.Platform(device_type="cuda", _is_cuda=True,
                                      _device_count=1)
    yield
    cw.current_platform = saved


# --------------------------------------------------------------------------
# First-level mapping: flat EngineArgs -> structured VllmConfig
# vllm/engine/arg_utils.py:create_engine_config
# --------------------------------------------------------------------------

def test_create_engine_config_packs_flat_into_structured():
    args = cw.EngineArgs(model="my-model", max_num_seqs=42,
                         tensor_parallel_size=1)
    cfg = args.create_engine_config()
    assert isinstance(cfg, cw.VllmConfig)
    # Flat fields landed in the right sub-configs.
    assert cfg.model_config.model == "my-model"
    assert cfg.scheduler_config.max_num_seqs == 42
    assert cfg.parallel_config.tensor_parallel_size == 1
    assert isinstance(cfg.cache_config, cw.CacheConfig)


def test_engine_args_default_is_single_source_of_truth():
    # EngineArgs defaults equal the sub-Config defaults (vllm L406-L440).
    assert cw.EngineArgs.model == cw.ModelConfig.model
    assert cw.EngineArgs.kv_cache_dtype == cw.CacheConfig.cache_dtype
    assert cw.EngineArgs.distributed_executor_backend == \
        cw.ParallelConfig.distributed_executor_backend


def test_engine_args_post_init_promotes_dict_compilation_config():
    # EngineArgs(compilation_config={...}) -> CompilationConfig (vllm L690-L696).
    args = cw.EngineArgs(compilation_config={"backend": "eager"})
    assert isinstance(args.compilation_config, cw.CompilationConfig)
    assert args.compilation_config.backend == "eager"


# --------------------------------------------------------------------------
# distributed_executor_backend default derivation
# vllm/config/parallel.py:L829-L874
# --------------------------------------------------------------------------

def test_backend_defaults_to_uni_for_single_gpu():
    pc = cw.ParallelConfig(tensor_parallel_size=1)
    pc.__post_init__()
    assert pc.distributed_executor_backend == "uni"


def test_backend_defaults_to_mp_for_multi_gpu():
    cw.current_platform = cw.Platform(_is_cuda=True, _device_count=2)
    pc = cw.ParallelConfig(tensor_parallel_size=2)
    pc.__post_init__()
    assert pc.distributed_executor_backend == "mp"


def test_backend_world_size_exceeds_gpus_raises():
    cw.current_platform = cw.Platform(_is_cuda=True, _device_count=1)
    # ParallelConfig.__post_init__ runs at construction (dataclass), so the
    # ValueError surfaces during construction.
    with pytest.raises(ValueError, match="larger than the number"):
        cw.ParallelConfig(tensor_parallel_size=4)


def test_explicit_backend_is_not_overridden():
    pc = cw.ParallelConfig(tensor_parallel_size=4, distributed_executor_backend="mp")
    pc.__post_init__()
    assert pc.distributed_executor_backend == "mp"


# --------------------------------------------------------------------------
# Factory #1: Executor.get_class — vllm/v1/executor/abstract.py:L47-L92
# --------------------------------------------------------------------------

@pytest.mark.parametrize("backend,expected", [
    ("uni", cw.UniProcExecutor),
    ("mp", cw.MultiprocExecutor),
    ("external_launcher", cw.ExecutorWithExternalLauncher),
])
def test_executor_get_class_selects_by_backend(backend, expected):
    cfg = cw.VllmConfig.__new__(cw.VllmConfig)
    cfg.parallel_config = cw.ParallelConfig(distributed_executor_backend=backend)
    assert cw.Executor.get_class(cfg) is expected


def test_executor_get_class_accepts_executor_subclass():
    class MyExec(cw.Executor):
        pass
    cfg = cw.VllmConfig.__new__(cw.VllmConfig)
    cfg.parallel_config = cw.ParallelConfig(distributed_executor_backend=MyExec)
    assert cw.Executor.get_class(cfg) is MyExec


def test_executor_get_class_unknown_backend_raises():
    cfg = cw.VllmConfig.__new__(cw.VllmConfig)
    cfg.parallel_config = cw.ParallelConfig(distributed_executor_backend=12345)
    with pytest.raises(ValueError, match="Unknown distributed executor backend"):
        cw.Executor.get_class(cfg)


# --------------------------------------------------------------------------
# Factory #2: SchedulerConfig.get_scheduler_cls — vllm/config/scheduler.py:L168-L188
# --------------------------------------------------------------------------

def test_scheduler_factory_async_true_returns_async_scheduler():
    sc = cw.SchedulerConfig(async_scheduling=True)
    assert sc.get_scheduler_cls() is cw.AsyncScheduler


def test_scheduler_factory_async_false_returns_scheduler():
    sc = cw.SchedulerConfig(async_scheduling=False)
    assert sc.get_scheduler_cls() is cw.Scheduler


def test_scheduler_factory_custom_class_passthrough():
    class MyScheduler:
        pass
    sc = cw.SchedulerConfig(scheduler_cls=MyScheduler)
    assert sc.get_scheduler_cls() is MyScheduler


# --------------------------------------------------------------------------
# Factory #3: EngineCoreClient.make_client — vllm/v1/engine/core_client.py:L80-L130
# --------------------------------------------------------------------------

def _vllm_config(dp_size=1, external_lb=False):
    cfg = cw.VllmConfig.__new__(cw.VllmConfig)
    cfg.parallel_config = cw.ParallelConfig(
        data_parallel_size=dp_size, data_parallel_external_lb=external_lb)
    return cfg


def test_make_client_inproc_when_not_multiprocess():
    c = cw.EngineCoreClient.make_client(False, False, _vllm_config(),
                                        cw.UniProcExecutor, log_stats=False)
    assert isinstance(c, cw.InprocClient)
    assert not isinstance(c, cw.SyncMPClient)


def test_make_client_sync_mp():
    c = cw.EngineCoreClient.make_client(True, False, _vllm_config(),
                                        cw.MultiprocExecutor, log_stats=False)
    assert isinstance(c, cw.SyncMPClient)


def test_make_client_async_requires_multiprocessing():
    with pytest.raises(NotImplementedError):
        cw.EngineCoreClient.make_client(False, True, _vllm_config(),
                                        cw.MultiprocExecutor, log_stats=False)


def test_make_client_async_mp_single_dp():
    c = cw.EngineCoreClient.make_client(True, True, _vllm_config(dp_size=1),
                                        cw.MultiprocExecutor, log_stats=False)
    assert type(c) is cw.AsyncMPClient


def test_make_async_mp_client_dp_internal_vs_external_lb():
    internal = cw.EngineCoreClient.make_async_mp_client(
        _vllm_config(dp_size=2, external_lb=False), cw.MultiprocExecutor, False)
    external = cw.EngineCoreClient.make_async_mp_client(
        _vllm_config(dp_size=2, external_lb=True), cw.MultiprocExecutor, False)
    assert type(internal) is cw.DPLBAsyncMPClient
    assert type(external) is cw.DPAsyncMPClient


# --------------------------------------------------------------------------
# async_scheduling tri-state decision — vllm/config/vllm.py:L777-L852
# --------------------------------------------------------------------------

def test_async_scheduling_auto_enabled_by_default():
    # None -> auto -> enabled when executor supports it (uni does).
    cfg = cw.EngineArgs(tensor_parallel_size=1).create_engine_config()
    cfg.__post_init__()
    assert cfg.scheduler_config.async_scheduling is True


def test_async_scheduling_disabled_for_pooling_model():
    # Build with a pooling model so the single (auto) __post_init__ sees it.
    cfg = cw.VllmConfig(
        model_config=cw.ModelConfig(runner_type="pooling"),
        parallel_config=cw.ParallelConfig(tensor_parallel_size=1),
        scheduler_config=cw.SchedulerConfig(async_scheduling=None),
    )
    assert cfg.scheduler_config.async_scheduling is False


def test_external_launcher_supports_async_scheduling():
    # ExecutorWithExternalLauncher subclasses UniProcExecutor and does NOT
    # override supports_async_scheduling, so it inherits True (real vLLM
    # uniproc_executor.py:L144 + L139-L141). External launcher therefore does
    # NOT reject explicit async scheduling.
    assert cw.ExecutorWithExternalLauncher.supports_async_scheduling() is True
    cfg = cw.EngineArgs(tensor_parallel_size=1,
                        async_scheduling=True).create_engine_config()
    cfg.parallel_config.distributed_executor_backend = "external_launcher"
    cfg.__post_init__()  # must NOT raise
    assert cfg.scheduler_config.async_scheduling is True


def test_async_scheduling_explicit_true_raises_if_executor_unsupported():
    # An executor whose supports_async_scheduling() returns False (the base
    # Executor default, real vLLM abstract.py:L367-L372) makes explicit
    # async_scheduling=True a hard error.
    class NoAsyncExecutor(cw.Executor):
        pass  # inherits base supports_async_scheduling() -> False

    assert NoAsyncExecutor.supports_async_scheduling() is False
    cfg = cw.EngineArgs(tensor_parallel_size=1,
                        async_scheduling=True).create_engine_config()
    # distributed_executor_backend may be a custom Executor subclass (real vLLM
    # abstract.py:L53-L57); get_class returns it as-is.
    cfg.parallel_config.distributed_executor_backend = NoAsyncExecutor
    with pytest.raises(ValueError, match="does not support async scheduling"):
        cfg.__post_init__()


def test_async_scheduling_explicit_false_stays_false():
    cfg = cw.EngineArgs(tensor_parallel_size=1,
                        async_scheduling=False).create_engine_config()
    cfg.__post_init__()
    assert cfg.scheduler_config.async_scheduling is False


# --------------------------------------------------------------------------
# O0-O3 optimization-level application — vllm/config/vllm.py:L184-L270, L652-L976
# --------------------------------------------------------------------------

def test_o0_disables_compilation_and_cudagraph():
    cfg = cw.EngineArgs(optimization_level=cw.OptimizationLevel.O0,
                        tensor_parallel_size=1).create_engine_config()
    cfg.__post_init__()
    assert cfg.compilation_config.mode == cw.CompilationMode.NONE
    assert cfg.compilation_config.cudagraph_mode == cw.CUDAGraphMode.NONE
    assert cfg.kernel_config.enable_flashinfer_autotune is False


def test_o2_default_full_and_piecewise_cudagraph():
    cfg = cw.EngineArgs(optimization_level=cw.OptimizationLevel.O2,
                        tensor_parallel_size=1).create_engine_config()
    cfg.__post_init__()
    assert cfg.compilation_config.mode == cw.CompilationMode.VLLM_COMPILE
    assert cfg.compilation_config.cudagraph_mode == \
        cw.CUDAGraphMode.FULL_AND_PIECEWISE
    assert cfg.kernel_config.enable_flashinfer_autotune is True


def test_o3_equals_o2():
    assert cw.OPTIMIZATION_LEVEL_03 is cw.OPTIMIZATION_LEVEL_02


def test_enforce_eager_overrides_optimization_level():
    # Priority: user enforce_eager > optimization-level preset (vllm L904-L910).
    cfg = cw.EngineArgs(optimization_level=cw.OptimizationLevel.O2,
                        enforce_eager=True,
                        tensor_parallel_size=1).create_engine_config()
    cfg.__post_init__()
    assert cfg.compilation_config.mode == cw.CompilationMode.NONE
    assert cfg.compilation_config.cudagraph_mode == cw.CUDAGraphMode.NONE


def test_torch_compile_disable_env_forces_none(monkeypatch):
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    cfg = cw.EngineArgs(optimization_level=cw.OptimizationLevel.O2,
                        tensor_parallel_size=1).create_engine_config()
    cfg.__post_init__()
    assert cfg.compilation_config.mode == cw.CompilationMode.NONE


def test_user_explicit_cudagraph_not_overridden_by_opt_level():
    # _set_config_default only fills None fields (vllm L645, L652-L679).
    args = cw.EngineArgs(optimization_level=cw.OptimizationLevel.O2,
                         tensor_parallel_size=1,
                         compilation_config={"cudagraph_mode": cw.CUDAGraphMode.PIECEWISE})
    cfg = args.create_engine_config()
    cfg.__post_init__()
    assert cfg.compilation_config.cudagraph_mode == cw.CUDAGraphMode.PIECEWISE


# --------------------------------------------------------------------------
# compute_hash — vllm/config/vllm.py:L367-L473
# --------------------------------------------------------------------------

def test_compute_hash_is_10_chars_and_deterministic():
    cfg = cw.EngineArgs(tensor_parallel_size=1).create_engine_config()
    cfg.__post_init__()
    h1 = cfg.compute_hash()
    h2 = cfg.compute_hash()
    assert len(h1) == 10
    assert h1 == h2


def test_compute_hash_changes_with_graph_affecting_config():
    a = cw.EngineArgs(max_model_len=4096, tensor_parallel_size=1).create_engine_config()
    b = cw.EngineArgs(max_model_len=8192, tensor_parallel_size=1).create_engine_config()
    a.__post_init__(); b.__post_init__()
    assert a.compute_hash() != b.compute_hash()


# --------------------------------------------------------------------------
# End-to-end entry: LLMEngine.from_engine_args — vllm/v1/engine/llm_engine.py:L151-L177
# --------------------------------------------------------------------------

def test_from_engine_args_builds_engine_with_inproc_client():
    args = cw.EngineArgs(model="m", tensor_parallel_size=1)
    engine = cw.LLMEngine.from_engine_args(args, enable_multiprocessing=False)
    assert isinstance(engine, cw.LLMEngine)
    # LLMEngine always uses asyncio_mode=False -> Inproc (non-mp) client.
    assert isinstance(engine.engine_core, cw.InprocClient)


def test_engine_core_selects_scheduler_via_factory():
    cfg = cw.EngineArgs(tensor_parallel_size=1).create_engine_config()
    cfg.__post_init__()
    executor_class = cw.Executor.get_class(cfg)
    core = cw.EngineCore(cfg, executor_class, log_stats=False)
    # async_scheduling auto-enabled -> AsyncScheduler chosen.
    assert core.scheduler is cw.AsyncScheduler


def test_executor_class_chosen_not_instantiated_at_from_engine_args():
    # from_engine_args returns a *class* for the executor (vllm L163).
    args = cw.EngineArgs(tensor_parallel_size=1)
    cfg = args.create_engine_config()
    cfg.__post_init__()
    executor_class = cw.Executor.get_class(cfg)
    assert isinstance(executor_class, type)
