# TDD tests for the v3 ch17 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).
#
# These assert the *observable pinned vLLM behavior* this chapter teaches — the
# three execution-arm layers (Executor / Worker / ModelRunner). Three layers:
#
# * Unit / contract, in-process: Executor.get_class factory dispatch (m2),
#   UniProcExecutor direct-call control plane + run_method three-branch
#   dispatch (m3), WorkerWrapperBase lazy init + __getattr__ (m5), the
#   platform axis worker_cls='auto' (m6), Worker.init_device ordering +
#   unsupported-device raise (m7), the CuMem pool tags of the three memory
#   anchors (m8), compile_or_warm_up_model startup orchestration (m9),
#   FutureWrapper FIFO pairing (m12), the two-phase runner state machine +
#   grammar-bitmask application point (m15), AsyncIntermediateTensors lazy
#   comm wait (m16), gpu_sync_check gate (m20 declaration faces).
# * Real-multiprocessing end-to-end: MultiprocExecutor spawns real WorkerProc
#   children through make_worker_process/worker_main/READY handshake (m4),
#   collective_rpc broadcast + single-point output_rank harvest (m11/m14),
#   worker_busy_loop dispatch + callable pickling (m13), the FAILURE/timeout
#   RPC error path and the process-death sentinel path (m17), shutdown /
#   death-pipe / weakref finalizer (m18), async output copy thread (m20).
# * Worker doubles live in tests/_worker_double.py and are resolved through
#   the real qualname mechanism (worker_cls string -> resolve_obj_by_qualname).
#
# Run:  cd instances/vllm/artifacts-v3/ch17-executor-worker-model-runner
#       python -m pytest tests/ -q
#
# Host: Windows (mp spawn), real torch / pyzmq / cloudpickle; no vllm package.

from __future__ import annotations

import gc
import importlib
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pytest
import torch

_TESTS_DIR = Path(__file__).resolve().parent
_CHAPTER_DIR = _TESTS_DIR.parent
# chapter dir for `import implementation`, tests dir for `import _worker_double`
sys.path.insert(0, str(_CHAPTER_DIR))
sys.path.insert(0, str(_TESTS_DIR))

implementation = importlib.import_module("implementation")
_host_seams = implementation._host_seams
_worker_double = importlib.import_module("_worker_double")

from implementation.executor.abstract import Executor  # noqa: E402
from implementation.executor.multiproc_executor import (  # noqa: E402
    FutureWrapper,
    MultiprocExecutor,
    WorkerProc,
)
from implementation.executor.uniproc_executor import (  # noqa: E402
    AsyncOutputFuture,
    UniProcExecutor,
)
from implementation.platforms.cuda import CudaPlatformBase  # noqa: E402
from implementation.serial_utils import run_method  # noqa: E402
from implementation.utils.import_utils import resolve_obj_by_qualname  # noqa: E402
from implementation.worker.gpu_model_runner import (  # noqa: E402
    ExecuteModelState,
    GPUModelRunner,
)
from implementation.worker.gpu_worker import (  # noqa: E402
    AsyncIntermediateTensors,
    Worker,
    init_worker_distributed_environment,
)
from implementation.worker.worker_base import (  # noqa: E402
    WorkerBase,
    WorkerWrapperBase,
)
from implementation.utils import gpu_sync_debug  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mk_config(**over):
    """Build the seam VllmConfig (the real assembly line is ch03's product)."""
    parallel = dict(
        world_size=2,
        local_world_size=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        nnodes_within_dp=1,
        node_rank_within_dp=0,
        node_rank=0,
        master_addr="127.0.0.1",
        distributed_executor_backend="mp",
        worker_cls="auto",
        worker_extension_cls=None,
        data_parallel_rank_local=None,
        data_parallel_index=0,
        data_parallel_backend="mp",
        assigned_physical_gpu_ids=None,
        enable_expert_parallel=False,
        distributed_timeout_seconds=None,
        disable_custom_all_reduce=True,
        enable_dbo=False,
    )
    parallel.update(over.pop("parallel_config", {}) or {})
    cfg = _host_seams.VllmConfig(
        parallel_config=_host_seams.ParallelConfig(**parallel),
        scheduler_config=_host_seams.SchedulerConfig(
            async_scheduling=over.pop("async_scheduling", False)
        ),
        device_config=_host_seams.DeviceConfig(
            device_type=over.pop("device_type", "cuda")
        ),
        use_v2_model_runner=over.pop("use_v2_model_runner", False),
    )
    assert not over, f"unknown test config overrides: {over}"
    return cfg


def mk_scheduler_output(total=4, req_ids=("r0", "r1")):
    return _host_seams.SchedulerOutput(
        num_scheduled_tokens={rid: total // len(req_ids) for rid in req_ids},
        total_num_scheduled_tokens=total,
    )


def spawn_mp_executor(worker_cls="_worker_double.ProbeWorker", world_size=2, **cfg_over):
    """Bring up a REAL MultiprocExecutor over spawned WorkerProc children."""
    cfg = mk_config(
        parallel_config={
            "world_size": world_size,
            "local_world_size": world_size,
            "tensor_parallel_size": world_size,
        },
        **cfg_over,
    )
    cfg.parallel_config.worker_cls = worker_cls
    return MultiprocExecutor(cfg)


def wait_until(pred, timeout=15.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# m2 — Executor.get_class factory dispatch
# ---------------------------------------------------------------------------


class TestGetClass:
    def test_mp_and_uni_dispatch(self):
        cfg = mk_config(parallel_config={"distributed_executor_backend": "mp"})
        assert Executor.get_class(cfg) is MultiprocExecutor
        cfg = mk_config(parallel_config={"distributed_executor_backend": "uni"})
        assert Executor.get_class(cfg) is UniProcExecutor

    def test_type_instance_subclass_returned(self):
        cfg = mk_config(
            parallel_config={"distributed_executor_backend": UniProcExecutor}
        )
        assert Executor.get_class(cfg) is UniProcExecutor

    def test_type_instance_non_executor_rejected(self):
        cfg = mk_config(parallel_config={"distributed_executor_backend": int})
        with pytest.raises(TypeError, match="must be a subclass"):
            Executor.get_class(cfg)

    def test_unknown_value_rejected(self):
        cfg = mk_config(parallel_config={"distributed_executor_backend": 3})
        with pytest.raises(ValueError, match="Unknown distributed executor backend"):
            Executor.get_class(cfg)

    def test_custom_qualname_resolved(self):
        # The canonical vllm qualnames resolve to this companion through the
        # host-seam alias table (real vllm package absent on this host).
        cfg = mk_config(
            parallel_config={
                "distributed_executor_backend": "vllm.v1.executor.uniproc_executor.UniProcExecutor"
            }
        )
        assert Executor.get_class(cfg) is UniProcExecutor

    def test_custom_qualname_must_be_executor_subclass(self):
        cfg = mk_config(
            parallel_config={
                "distributed_executor_backend": "vllm.v1.worker.worker_base.WorkerWrapperBase"
            }
        )
        with pytest.raises(TypeError, match="must be a subclass"):
            Executor.get_class(cfg)

    def test_ray_and_external_launcher_subtracted(self):
        # SUBTRACTED consequence (dossier delete #1): the ray branch is gone,
        # so "ray" falls through to the qualname branch and fails there — the
        # documented post-subtraction behavior of the companion.
        cfg = mk_config(parallel_config={"distributed_executor_backend": "ray"})
        with pytest.raises(ValueError):
            Executor.get_class(cfg)
        src = "\n".join(
            line
            for line in Path(
                implementation.executor.abstract.__file__
            ).read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "RayDistributedExecutor" not in src
        assert "ExecutorWithExternalLauncher" not in src


# ---------------------------------------------------------------------------
# m10 — Executor as thin collective_rpc wrappers (abstract.py)
# ---------------------------------------------------------------------------


class _ConcreteExecutor(Executor):
    """Bare concrete stub so the thin-wrapper methods can be spied (the real
    class is an ABC)."""

    def _init_executor(self) -> None:
        pass

    def collective_rpc(self, *a, **k):  # overridden per-test via instance attr
        raise NotImplementedError

    def check_health(self) -> None:
        return


class TestThinControlPlane:
    def _spy_executor(self):
        ex = _ConcreteExecutor.__new__(_ConcreteExecutor)  # skip __init__
        calls = []

        def fake_rpc(method, timeout=None, args=(), kwargs=None, non_block=False):
            calls.append((method, args, kwargs, non_block))
            return ["ok"]

        ex.collective_rpc = fake_rpc
        return ex, calls

    def test_execute_model_is_thin_wrapper(self):
        ex, calls = self._spy_executor()
        so = mk_scheduler_output()
        out = ex.execute_model(so, non_block=True)
        assert out == "ok"
        assert calls == [("execute_model", (so,), None, True)]

    def test_sample_tokens_is_thin_wrapper(self):
        ex, calls = self._spy_executor()
        out = ex.sample_tokens(None, non_block=False)
        assert out == "ok"
        assert calls[0][0] == "sample_tokens"

    def test_initialize_from_config_and_memory_go_through_rpc(self):
        ex, calls = self._spy_executor()
        ex.initialize_from_config([_host_seams.KVCacheConfig(num_blocks=8)])
        assert calls[0][:2] == ("initialize_from_config", ([_host_seams.KVCacheConfig(num_blocks=8)],))
        ex.determine_available_memory()
        assert calls[1][0] == "determine_available_memory"
        ex.get_kv_cache_specs()
        assert calls[2][0] == "get_kv_cache_spec"

    def test_control_plane_docstring_verbatim(self):
        # The control/data-plane separation contract must survive verbatim.
        # (The docstring lives on the first @overload — runtime attribute is
        # the abstract tail — so assert against the module source.)
        src = Path(implementation.executor.abstract.__file__).read_text(
            encoding="utf-8"
        )
        assert (
            "It is recommended to use this API to only pass control messages,\n"
            "            and set up data-plane communication to pass data." in src
        )

    def test_compile_or_warm_up_times_max_propagation(self):
        # abstract.py:L127-L137: with TP>1 the compilation happens in worker
        # processes; the main process takes the max across workers.
        from implementation.worker.worker_base import CompilationTimes

        ex = _ConcreteExecutor.__new__(_ConcreteExecutor)
        ex.vllm_config = mk_config()
        ex.collective_rpc = lambda *a, **k: [
            CompilationTimes(language_model=1.0, encoder=0.25),
            CompilationTimes(language_model=3.0, encoder=0.5),
        ]
        ex.compile_or_warm_up_model()
        assert ex.vllm_config.compilation_config.compilation_time == 3.0
        assert ex.vllm_config.compilation_config.encoder_compilation_time == 0.5

    def test_base_supports_async_scheduling_false(self):
        assert Executor.supports_async_scheduling() is False


# ---------------------------------------------------------------------------
# m3 — UniProcExecutor: direct call + run_method three branches
# ---------------------------------------------------------------------------


class _InlineWorker:
    def __init__(self):
        self.calls = []

    def echo(self, *args, **kwargs):
        self.calls.append(("echo", args, kwargs))
        return {"echo": True}

    def execute_model(self, scheduler_output):
        self.calls.append(("execute_model", scheduler_output))
        return {"executed": scheduler_output.total_num_scheduled_tokens}


class TestUniProc:
    def _uni(self, worker):
        ex = UniProcExecutor.__new__(UniProcExecutor)
        ex.driver_worker = worker
        return ex

    def test_blocking_rpc_returns_list(self):
        w = _InlineWorker()
        out = self._uni(w).collective_rpc("echo", args=(1,), kwargs={"k": 2})
        assert out == [{"echo": True}]
        assert w.calls == [("echo", (1,), {"k": 2})]

    def test_blocking_single_value(self):
        w = _InlineWorker()
        out = self._uni(w).collective_rpc("echo", single_value=True)
        assert out == {"echo": True}

    def test_non_block_completes_future(self):
        w = _InlineWorker()
        fut = self._uni(w).collective_rpc("echo", non_block=True)
        assert fut.done() and fut.result() == [{"echo": True}]

    def test_non_block_exception_future(self):
        ex = self._uni(_InlineWorker())
        fut = ex.collective_rpc("no_such_method", non_block=True)
        assert fut.done()
        with pytest.raises(NotImplementedError, match="no_such_method"):
            fut.result()

    def test_non_block_async_output_wrapped(self):
        async_out = _worker_double.ScriptedAsyncOutput({"payload": 7}, ready_after=0.1)
        w = _InlineWorker()
        w.echo = lambda *a, **k: async_out  # type: ignore[method-assign]
        fut = self._uni(w).collective_rpc("echo", non_block=True, single_value=True)
        assert isinstance(fut, AsyncOutputFuture)
        assert not fut.done()  # result() is what waits the (D2H) event
        assert async_out.get_output_calls == 0
        assert fut.result() == {"payload": 7}
        assert async_out.get_output_calls == 1
        # second result() is instant, does not call get_output again
        assert fut.result() == {"payload": 7}
        assert async_out.get_output_calls == 1

    def test_blocking_rpc_unwraps_async_output(self):
        async_out = _worker_double.ScriptedAsyncOutput({"payload": 9})
        w = _InlineWorker()
        w.echo = lambda *a, **k: async_out  # type: ignore[method-assign]
        out = self._uni(w).collective_rpc("echo")
        assert out == [{"payload": 9}]
        assert async_out.get_output_calls == 1

    def test_async_output_future_timeout_not_implemented(self):
        async_out = _worker_double.ScriptedAsyncOutput(None)
        fut = AsyncOutputFuture(async_out, single_value=False)
        with pytest.raises(RuntimeError, match="timeout not implemented"):
            fut.result(timeout=1)

    def test_execute_model_inlines_early_failure(self):
        # uniproc_executor.py:L117-L121: non_blocking mode surfaces exceptions
        # as early as possible — execute_model calls output.result() in-line.
        class _FailingWorker(_InlineWorker):
            def execute_model(self, scheduler_output):
                raise RuntimeError("forward blew up")

        ex = self._uni(_FailingWorker())
        with pytest.raises(RuntimeError, match="forward blew up"):
            ex.execute_model(mk_scheduler_output(), non_block=True)

    def test_uni_supports_async_scheduling_true(self):
        assert UniProcExecutor.supports_async_scheduling() is True


class TestRunMethod:
    def test_str_branch(self):
        w = _InlineWorker()
        assert run_method(w, "echo", (1,), {}) == {"echo": True}

    def test_str_missing_raises_notimplemented(self):
        with pytest.raises(NotImplementedError, match="is not implemented"):
            run_method(_InlineWorker(), "missing", (), {})

    def test_bytes_branch_cloudpickle(self):
        import cloudpickle

        w = _InlineWorker()
        pickled = cloudpickle.dumps(lambda obj, v: ("called-with", type(obj).__name__, v))
        # bytes -> cloudpickle.loads -> partial(fn, obj): the worker object is
        # passed as the first argument
        assert run_method(w, pickled, (5,), {}) == ("called-with", "_InlineWorker", 5)

    def test_callable_branch(self):
        w = _InlineWorker()
        out = run_method(w, lambda obj: obj, (), {})
        assert out is w


# ---------------------------------------------------------------------------
# m5 — WorkerWrapperBase lazy initialization
# ---------------------------------------------------------------------------


class TestWorkerWrapper:
    def test_construction_records_ranks_only(self):
        wrap = WorkerWrapperBase(rpc_rank=1, global_rank=3)
        assert wrap.rpc_rank == 1 and wrap.global_rank == 3
        # worker is NOT constructed yet — attribute *access* is undefined
        # before init_worker (the __getattr__ passthrough recurses on any
        # non-instance attribute: documented cost of the magic, verified
        # separately below). The instance dict is the honest probe.
        assert "worker" not in wrap.__dict__

    def test_pre_init_attribute_access_recurses(self):
        # worker_base.py:L333-L334 的既定行为：init_worker 前的属性访问是
        # 未定义行为——__getattr__ 透传到不存在的 self.worker 上会递归。
        wrap = WorkerWrapperBase(rpc_rank=0)
        with pytest.raises(RecursionError):
            wrap.some_attribute

    def test_init_worker_resolves_qualname(self):
        wrap = WorkerWrapperBase(rpc_rank=0)
        cfg = mk_config(parallel_config={"world_size": 1})
        cfg.parallel_config.worker_cls = "_worker_double.ProbeWorker"
        kwargs = dict(
            vllm_config=cfg,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
        )
        wrap.init_worker([kwargs])
        assert type(wrap.worker).__name__ == "ProbeWorker"
        assert wrap.worker.rank == 0

    def test_canonical_vllm_qualname_resolves_to_companion_worker(self):
        wrap = WorkerWrapperBase(rpc_rank=0)
        cfg = mk_config(parallel_config={"world_size": 1})
        cfg.parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
        wrap.init_worker(
            [
                dict(
                    vllm_config=cfg,
                    local_rank=0,
                    rank=0,
                    distributed_init_method="tcp://127.0.0.1:1",
                    is_driver_worker=True,
                )
            ]
        )
        assert type(wrap.worker) is Worker

    def test_worker_cls_must_be_string(self):
        wrap = WorkerWrapperBase(rpc_rank=0)
        cfg = mk_config(parallel_config={"world_size": 1})
        cfg.parallel_config.worker_cls = Worker  # a class, not a qualname
        with pytest.raises(ValueError, match="no longer supported"):
            wrap.init_worker(
                [
                    dict(
                        vllm_config=cfg,
                        local_rank=0,
                        rank=0,
                        distributed_init_method="tcp://127.0.0.1:1",
                        is_driver_worker=True,
                    )
                ]
            )

    def test_getattr_passthrough_after_init(self):
        wrap = WorkerWrapperBase(rpc_rank=0)
        cfg = mk_config(parallel_config={"world_size": 1})
        cfg.parallel_config.worker_cls = "_worker_double.ProbeWorker"
        wrap.init_worker(
            [
                dict(
                    vllm_config=cfg,
                    local_rank=0,
                    rank=0,
                    distributed_init_method="tcp://127.0.0.1:1",
                    is_driver_worker=True,
                )
            ]
        )
        # collective_rpc's `getattr(self.worker, method)` lands here.
        assert wrap.rank == 0

    def test_update_environment_variables(self):
        from implementation.utils.system_utils import update_environment_variables

        old = dict(os.environ)
        try:
            update_environment_variables({"CH17_TEST_ENV": "42"})
            assert os.environ["CH17_TEST_ENV"] == "42"
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_wrapper_execute_model_applies_mm_cache_then_delegates(self):
        wrap = WorkerWrapperBase(rpc_rank=0)
        cfg = mk_config(parallel_config={"world_size": 1})
        cfg.parallel_config.worker_cls = "_worker_double.ProbeWorker"
        wrap.init_worker(
            [
                dict(
                    vllm_config=cfg,
                    local_rank=0,
                    rank=0,
                    distributed_init_method="tcp://127.0.0.1:1",
                    is_driver_worker=True,
                )
            ]
        )
        so = mk_scheduler_output()
        out = wrap.execute_model(so)
        assert out["scheduled"] == 4
        # mm_receiver_cache is None in the companion (build branch
        # subtracted), so _apply_mm_cache is a no-op — new reqs survive.
        assert so.scheduled_new_reqs == []


# ---------------------------------------------------------------------------
# m6 — platform axis
# ---------------------------------------------------------------------------


class TestPlatformAxis:
    def test_auto_resolves_to_canonical_gpu_worker_qualname(self):
        # platforms/cuda.py:L307-L313, verbatim mapping.
        cfg = mk_config(parallel_config={"world_size": 1})
        CudaPlatformBase.check_and_update_config(cfg)
        assert cfg.parallel_config.worker_cls == "vllm.v1.worker.gpu_worker.Worker"

    def test_resolve_obj_by_qualname_roundtrip(self):
        cls = resolve_obj_by_qualname("vllm.v1.worker.gpu_worker.Worker")
        assert cls is Worker
        fn = resolve_obj_by_qualname("vllm.v1.serial_utils.run_method")
        assert fn is run_method


# ---------------------------------------------------------------------------
# m7 — Worker.init_device
# ---------------------------------------------------------------------------


class TestInitDevice:
    def _worker(self, **cfg_over):
        cfg = mk_config(**cfg_over)
        w = Worker(
            vllm_config=cfg,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
        )
        return w

    def test_unsupported_device_type_raises(self):
        w = self._worker(device_type="tpu")
        with pytest.raises(RuntimeError, match="Unsupported device type: tpu"):
            w.init_device()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs a CUDA context on host"
    )
    def test_distributed_init_precedes_memory_snapshot(self, monkeypatch):
        w = self._worker()
        gpu_worker_mod = implementation.worker.gpu_worker
        order = []
        monkeypatch.setattr(
            gpu_worker_mod, "init_worker_distributed_environment",
            lambda *a, **k: order.append("dist_init"), raising=True,
        )
        monkeypatch.setattr(
            gpu_worker_mod, "MemorySnapshot",
            lambda device=None: order.append("snapshot") or _host_seams.MemorySnapshot(),
            raising=True,
        )
        monkeypatch.setattr(
            gpu_worker_mod, "request_memory",
            lambda snap, cache_cfg: order.append("request_memory") or 0,
            raising=True,
        )
        w.init_device()
        assert order[0] == "dist_init"
        assert order.index("snapshot") > order.index("dist_init")
        # the model runner is the V1 canon when the V2 flag is off
        from implementation.worker.gpu_model_runner import GPUModelRunner as V1

        assert type(w.model_runner) is V1
        assert w.device is not None

    def test_dp_local_rank_correction(self, monkeypatch):
        monkeypatch.setattr(
            torch.accelerator, "device_count", lambda: 4
        )
        # the host really owns 1 GPU: cuda:2 selection is recorded, not applied
        monkeypatch.setattr(
            torch.accelerator, "set_device_index", lambda device: None
        )
        w = self._worker(
            parallel_config={
                "world_size": 2,
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "data_parallel_rank_local": 1,
            }
        )
        gpu_worker_mod = implementation.worker.gpu_worker
        seen = {}
        monkeypatch.setattr(
            gpu_worker_mod,
            "init_worker_distributed_environment",
            lambda cfg, rank, method, local_rank, backend: seen.update(
                local_rank=local_rank
            ),
            raising=True,
        )
        monkeypatch.setattr(gpu_worker_mod, "MemorySnapshot",
                            lambda device=None: _host_seams.MemorySnapshot(),
                            raising=True)
        monkeypatch.setattr(gpu_worker_mod, "request_memory",
                            lambda s, c: 0, raising=True)
        w.init_device()
        # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK = 1 * 2 + 0
        assert w.local_rank == 2
        assert seen["local_rank"] == 2


# ---------------------------------------------------------------------------
# m8 — three memory anchors + CuMem pool tags
# ---------------------------------------------------------------------------


class _RecordingRunner:
    """Spy runner: records calls; stands where ch18/ch19's runner lives."""

    def __init__(self):
        self.calls = []

    def load_model(self, *, load_dummy_weights=False):
        self.calls.append("runner.load_model")

    def initialize_kv_cache(self, cfg):
        self.calls.append("runner.initialize_kv_cache")

    def get_model(self):
        return torch.nn.Linear(2, 2)

    def shutdown(self):
        pass


class TestMemoryAnchors:
    def _worker(self, monkeypatch, cumem: bool = True):
        # The real conditional ladder (gpu_worker.py:L256-L267): the CuMem pool
        # only engages on a cuda-like platform with enable_cumem_allocator on
        # (the default CUDA deployment takes the nullcontext fast path).
        monkeypatch.setattr(_host_seams.current_platform, "is_cuda_alike", lambda: True)
        monkeypatch.setattr(_host_seams.current_platform, "is_xpu", lambda: False)
        monkeypatch.setattr(_host_seams.current_platform, "is_cpu", lambda: False)
        cfg = mk_config()
        cfg.model_config.enable_cumem_allocator = cumem
        w = Worker(
            vllm_config=cfg,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
        )
        w.model_runner = _RecordingRunner()
        _host_seams.CUMEM_SEAM.log.clear()
        return w

    def test_load_model_tag_weights(self, monkeypatch):
        w = self._worker(monkeypatch)
        w.load_model()
        tags = [e["tag"] for e in _host_seams.CUMEM_SEAM.log]
        assert tags[-1] == "weights"
        assert w.model_runner.calls == ["runner.load_model"]

    def test_default_cuda_config_takes_nullcontext_fast_path(self, monkeypatch):
        # gpu_worker.py:L257-L261: cuda-like + cumem DISABLED (the default)
        # short-circuits to nullcontext — no pool tag recorded.
        w = self._worker(monkeypatch, cumem=False)
        w.load_model()
        assert _host_seams.CUMEM_SEAM.log == []
        assert w.model_runner.calls == ["runner.load_model"]

    def test_initialize_from_config_tag_kv_cache(self, monkeypatch):
        w = self._worker(monkeypatch)
        w.initialize_from_config(_host_seams.KVCacheConfig(num_blocks=16))
        tags = [e["tag"] for e in _host_seams.CUMEM_SEAM.log]
        assert tags[-1] == "kv_cache"
        assert w.cache_config.num_gpu_blocks == 16
        assert w.model_runner.calls == ["runner.initialize_kv_cache"]

    def test_determine_available_memory_is_anchor_skeleton(self, monkeypatch):
        # The ledger itself is ch14's companion; here the anchor must exist
        # and answer through the control plane (see the mp e2e test).
        w = self._worker(monkeypatch)
        assert callable(w.determine_available_memory)
        assert w.determine_available_memory() == 0  # HOST SEAM ledger -> ch14

    def test_worker_base_contract_docstrings(self):
        # The two-phase contract text must survive verbatim (worker_base.py:L145-L149).
        assert (
            "If this method returns None, sample_tokens should be called "
            "immediately after" in WorkerBase.execute_model.__doc__
        )
        assert "may be changed in future" in WorkerBase.execute_model.__doc__
        assert (
            "immediately after execute_model iff it returned None"
            in WorkerBase.sample_tokens.__doc__
        )


# ---------------------------------------------------------------------------
# m9 — compile_or_warm_up_model startup orchestration
# ---------------------------------------------------------------------------


class TestWarmupOrchestration:
    def test_orchestration_order(self, monkeypatch):
        w = Worker(
            vllm_config=mk_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
        )
        runner = _RecordingRunner()
        runner.calls = []
        captured = {}

        def dummy_run(num_tokens, **kw):
            captured.setdefault("warmup_sizes", []).append(num_tokens)
            return (torch.zeros(1), torch.zeros(1))

        runner._dummy_run = dummy_run
        runner.maybe_remove_all_loras = lambda cfg: runner.calls.append(
            "maybe_remove_all_loras"
        )
        runner.capture_model = lambda: runner.calls.append("capture_model") or 0
        runner._dummy_sampler_run = lambda hidden_states=None: runner.calls.append(
            "_dummy_sampler_run"
        )
        runner.is_pooling_model = False
        runner.lora_config = None
        w.model_runner = runner
        gpu_worker_mod = implementation.worker.gpu_worker
        monkeypatch.setattr(
            gpu_worker_mod, "kernel_warmup",
            lambda worker: runner.calls.append("kernel_warmup"), raising=True,
        )
        monkeypatch.setattr(
            gpu_worker_mod, "freeze_gc_heap",
            lambda: runner.calls.append("freeze_gc_heap"), raising=True,
        )
        monkeypatch.setattr(
            _host_seams, "activate_jit_monitor",
            lambda **kw: runner.calls.append("jit_monitor"), raising=True,
        )
        monkeypatch.setattr(
            gpu_worker_mod, "enable_gpu_sync_check",
            lambda: runner.calls.append("enable_gpu_sync_check"), raising=True,
        )
        w.vllm_config.compilation_config.compile_sizes = [4, 8, 2]
        w.vllm_config.compilation_config.cudagraph_capture_sizes = [4]
        w.vllm_config.compilation_config.cudagraph_mode = (
            _host_seams.CUDAGraphMode.PIECEWISE
        )
        w.vllm_config.compilation_config.mode = (
            _host_seams.CompilationMode.VLLM_COMPILE
        )

        times = w.compile_or_warm_up_model()
        # warmup runs big-to-small, sizes in the capture list skipped (4), then
        # the sampler warmup adds max_num_reqs = min(128, 2048) = 128
        assert captured["warmup_sizes"][:2] == [8, 2]
        assert captured["warmup_sizes"][-1] == 128
        # the fixed startup choreography order (m9)
        assert runner.calls.index("maybe_remove_all_loras") < runner.calls.index(
            "kernel_warmup"
        )
        assert runner.calls.index("kernel_warmup") < runner.calls.index(
            "capture_model"
        )
        assert runner.calls.index("capture_model") < runner.calls.index(
            "_dummy_sampler_run"
        )
        assert runner.calls[-4:] == [
            "_dummy_sampler_run",
            "jit_monitor",
            "freeze_gc_heap",
            "enable_gpu_sync_check",
        ]
        assert isinstance(times, implementation.worker.worker_base.CompilationTimes)

    def test_gpu_sync_check_gate_off_during_setup(self, monkeypatch):
        # gpu_sync_debug.py: the gate starts OFF; enable flips it only when
        # VLLM_GPU_SYNC_CHECK is set — startup syncs pass through.
        monkeypatch.setattr(_host_seams.envs, "VLLM_GPU_SYNC_CHECK", "error")
        try:
            assert gpu_sync_debug._sync_check_enabled is False
            gpu_sync_debug.enable_gpu_sync_check()
            assert gpu_sync_debug._sync_check_enabled is True

            @gpu_sync_debug.with_gpu_sync_check
            def f():
                return 11

            assert f() == 11  # non-CUDA host: real no-op branch (L164-L165)
        finally:
            gpu_sync_debug._sync_check_enabled = False

    def test_unconfigured_sync_check_leaves_gate_off(self):
        gpu_sync_debug.enable_gpu_sync_check()  # envs seam: VLLM_GPU_SYNC_CHECK=None
        assert gpu_sync_debug._sync_check_enabled is False


# ---------------------------------------------------------------------------
# m15 — GPUModelRunner two-phase state machine
# ---------------------------------------------------------------------------


class TestTwoPhaseRunner:
    def _runner(self):
        r = GPUModelRunner.__new__(GPUModelRunner)  # ch18's __init__ subtracted
        r.execute_model_state = None  # vllm/v1/worker/gpu_model_runner.py:L942
        r.kv_connector_output = None
        r.input_batch = _SimpleBatch(["r0", "r1"])
        r.use_async_scheduling = False
        return r

    def test_execute_twice_raises_state_error(self):
        r = self._runner()
        assert r.execute_model(mk_scheduler_output()) is None
        with pytest.raises(RuntimeError) as ei:
            r.execute_model(mk_scheduler_output())
        assert "State error: sample_tokens() must be called " in str(ei.value)
        assert "after execute_model() returns None." in str(ei.value)

    def test_execute_packs_state_and_returns_none(self):
        r = self._runner()
        so = mk_scheduler_output()
        assert r.execute_model(so) is None
        st = r.execute_model_state
        assert isinstance(st, ExecuteModelState)
        assert st.scheduler_output is so
        # the ten-field ephemeral protocol (L437-L451)
        assert len(ExecuteModelState._fields) == 10

    def test_sample_unpacks_clears_and_calls_bitmask_then_sample(self):
        r = self._runner()
        so = mk_scheduler_output(total=2, req_ids=("r0",))
        r.execute_model(so)
        seq = []

        class _SpySamplerOutput:
            sampled_token_ids = [[7]]

        r.__dict__["_sample"] = lambda logits, spec: seq.append("_sample") or _SpySamplerOutput()
        r.__dict__["_update_states_after_model_execute"] = (
            lambda sampled, sched: seq.append("_update_states")
        )
        # real grammar application on real torch logits (structured_output/utils.py)
        import numpy as np

        logits = torch.tensor([[0.1, 5.0, 0.2, 0.3]])  # argmax = token 1
        r.execute_model_state = r.execute_model_state._replace(logits=logits)
        grammar = _host_seams.GrammarOutput(
            structured_output_request_ids=["r0"],
            grammar_bitmask=np.array(
                [[0b10101]], dtype=np.int32
            ),  # forbid token 1 (bit 1 = 0)
        )
        r.sample_tokens(grammar)
        assert r.execute_model_state is None  # unpack-then-clear
        assert seq == ["_sample", "_update_states"]
        # bitmask applied in-place: token 1 -> -inf, argmax flips
        assert logits[0, 1].item() == float("-inf")
        assert int(logits.argmax()) != 1

    def test_sample_without_state_is_pass_through_branch(self):
        r = self._runner()
        out = r.sample_tokens(None)
        # the early-exit branch (kv-conn pass-through shape, ch34/ch16 domain)
        assert isinstance(out, _host_seams.ModelRunnerOutput)

    def test_worker_layer_two_phase_delegation(self):
        w = Worker(
            vllm_config=mk_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
        )
        seq = []

        class _Runner:
            def execute_model(self, so, intermediate_tensors=None):
                seq.append("runner.execute_model")
                return None

            def sample_tokens(self, grammar_output):
                seq.append("runner.sample_tokens")
                return "MRO"

        w.model_runner = _Runner()
        assert w.execute_model(mk_scheduler_output()) is None
        assert w.sample_tokens(None) == "MRO"
        assert seq == ["runner.execute_model", "runner.sample_tokens"]

    def test_worker_execute_model_pp1_fast_path(self):
        # TP=1/PP=1: no irecv (first rank), runner returns ModelRunnerOutput
        # -> early return; None (two-phase) -> returns None
        w = Worker(
            vllm_config=mk_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:1",
            is_driver_worker=True,
        )
        w.model_runner = _RecordingRunner()
        mro = _host_seams.ModelRunnerOutput(req_ids=["r0"])
        w.model_runner.execute_model = lambda so, it=None: mro
        assert w.execute_model(mk_scheduler_output()) is mro
        # zero-token scheduler output: forward_pass False -> still delegates
        w.model_runner.execute_model = lambda so, it=None: None
        assert w.execute_model(mk_scheduler_output(total=0)) is None


class _SimpleBatch:
    def __init__(self, req_ids):
        self.req_ids = list(req_ids)


# ---------------------------------------------------------------------------
# m16 — AsyncIntermediateTensors lazy comm sync
# ---------------------------------------------------------------------------


class TestAsyncIntermediateTensors:
    def test_lazy_wait_on_tensors_access(self):
        waited = []

        class _H:
            def wait(self):
                waited.append("wait")

        class _Post:
            def __call__(self):
                waited.append("post")

        t = AsyncIntermediateTensors(
            {"hidden": torch.zeros(2)}, comm_handles=[_H()], comm_postprocess=[_Post()]
        )
        assert waited == []  # construction does not wait
        assert waited == [] or True
        tensors = t.tensors  # first access triggers the lazy sync
        assert "hidden" in tensors
        assert waited == ["wait", "post"]
        t.tensors  # second access: no re-wait
        assert waited == ["wait", "post"]

    def test_wait_for_comm_idempotent(self):
        t = AsyncIntermediateTensors({"hidden": torch.zeros(1)})
        t.wait_for_comm()
        t.wait_for_comm()


# ---------------------------------------------------------------------------
# m12 — FutureWrapper FIFO pairing
# ---------------------------------------------------------------------------


class TestFutureWrapper:
    def test_result_drains_futures_ahead_of_self(self):
        fq: deque = deque()
        got = []

        def resp_for(tag):
            def get():
                got.append(tag)
                return f"resp-{tag}"

            return get

        f1 = FutureWrapper(fq, get_response=resp_for(1))
        f2 = FutureWrapper(fq, get_response=resp_for(2))
        f3 = FutureWrapper(fq, get_response=resp_for(3))
        # appendleft+pop => FIFO: f1 oldest
        assert f3.result() == "resp-3"
        # collecting f3 drained f1 and f2 first — both are done now
        assert f1.done() and f1.result() == "resp-1"
        assert f2.done() and f2.result() == "resp-2"
        assert got == [1, 2, 3]

    def test_exception_forwarded(self):
        fq: deque = deque()

        def boom():
            raise RuntimeError("mq says no")

        f = FutureWrapper(fq, get_response=boom)
        with pytest.raises(RuntimeError, match="mq says no"):
            f.result()

    def test_timeout_not_implemented(self):
        fq: deque = deque()
        f = FutureWrapper(fq, get_response=lambda: None)
        with pytest.raises(RuntimeError, match="timeout not implemented"):
            f.result(timeout=1)


# ---------------------------------------------------------------------------
# worker_main / wait_for_ready / enqueue_output unit faces
# ---------------------------------------------------------------------------


class TestWorkerProcUnits:
    def test_response_status_members(self):
        assert {m.name for m in WorkerProc.ResponseStatus} == {"SUCCESS", "FAILURE"}

    def test_signal_handler_raises_systemexit_once(self):
        ev = threading.Event()

        # reconstruct worker_main's closure shape (L830-L837)
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested.is_set():
                shutdown_requested.set()
                raise SystemExit()

        shutdown_requested = ev
        with pytest.raises(SystemExit):
            signal_handler(15, None)
        assert ev.is_set()
        signal_handler(15, None)  # second time: silent, no raise

    def test_wait_for_ready_rejects_failure_status(self):
        parent, child = mp.Pipe()
        import multiprocessing.connection as mpc

        def feed():
            child.send({"status": "FAILED", "handle": None, "peer": []})

        threading.Thread(target=feed, daemon=True).start()

        class _Proc:
            pass

        class _Handle:
            def __init__(self):
                self.proc = _Proc()
                self.rank = 0
                self.ready_pipe = parent

        with pytest.raises(
            Exception, match="WorkerProc initialization failed"
        ):
            WorkerProc.wait_for_ready([_Handle()])

    def test_wait_for_ready_eof_raises(self):
        parent, child = mp.Pipe()
        child.close()  # EOF

        class _Handle:
            def __init__(self):
                self.proc = object()
                self.rank = 0
                self.ready_pipe = parent

        with pytest.raises(
            Exception, match="WorkerProc initialization failed"
        ):
            WorkerProc.wait_for_ready([_Handle()])

    def test_enqueue_output_success_and_failure(self):
        wp = WorkerProc.__new__(WorkerProc)
        sent = []
        wp.worker_response_mq = _FakeMQ(sent)
        wp.enqueue_output({"ok": 1})
        assert sent[-1] == (WorkerProc.ResponseStatus.SUCCESS, {"ok": 1})
        wp.enqueue_output(RuntimeError("bad"))
        status, payload = sent[-1]
        assert status == WorkerProc.ResponseStatus.FAILURE
        assert "bad" in payload

    def test_enqueue_output_unwraps_async_output(self):
        wp = WorkerProc.__new__(WorkerProc)
        sent = []
        wp.worker_response_mq = _FakeMQ(sent)
        wp.enqueue_output(
            _worker_double.ScriptedAsyncOutput({"unwrapped": True})
        )
        assert sent[-1] == (WorkerProc.ResponseStatus.SUCCESS, {"unwrapped": True})

    def test_handle_output_sync_direct_vs_async_queue(self):
        wp = WorkerProc.__new__(WorkerProc)
        sent = []
        wp.worker_response_mq = _FakeMQ(sent)
        wp.use_async_scheduling = False
        wp.handle_output("x")
        assert sent == [(WorkerProc.ResponseStatus.SUCCESS, "x")]
        wp.use_async_scheduling = True
        wp.async_output_queue = queue_mod.Queue()
        wp.handle_output("y")
        assert len(sent) == 1  # parked on the async queue instead
        assert wp.async_output_queue.get_nowait() == "y"

    def test_failure_note_attached(self):
        # worker_busy_loop's except arm: add_note with the child traceback.
        wp = WorkerProc.__new__(WorkerProc)
        wp.rpc_broadcast_mq = None
        wp.worker = _worker_double.ProbeWorker(
            mk_config(parallel_config={"world_size": 1}), 0, 0, "tcp://x"
        )
        wp.rank = 0
        wp.use_async_scheduling = False
        sent = []
        wp.worker_response_mq = _FakeMQ(sent)
        # one iteration of the loop: dequeue one RPC then break via fake MQ
        wp.rpc_broadcast_mq = _OneShotMQ(("boom", (), {}, None))
        try:
            wp.worker_busy_loop()
        except StopIteration:
            pass
        status, payload = sent[-1]
        assert status == WorkerProc.ResponseStatus.FAILURE
        assert "boom in worker" in payload

    def test_busy_loop_output_rank_filter(self):
        # non-output ranks must NOT write a response
        wp = WorkerProc.__new__(WorkerProc)
        wp.rank = 1
        wp.use_async_scheduling = False
        sent = []
        wp.worker_response_mq = _FakeMQ(sent)
        wp.worker = _worker_double.ProbeWorker(
            mk_config(parallel_config={"world_size": 2}), 1, 1, "tcp://x"
        )
        wp.rpc_broadcast_mq = _OneShotMQ(("echo", (1,), {}, 0))  # only rank 0 replies
        try:
            wp.worker_busy_loop()
        except StopIteration:
            pass
        assert sent == []


class _FakeMQ:
    def __init__(self, sink):
        self.sink = sink

    def enqueue(self, item):
        self.sink.append(item)


class _OneShotMQ:
    """Broadcast-MQ stand-in: returns one message, then raises StopIteration
    (test transport out of the infinite busy loop)."""

    def __init__(self, msg):
        self.msg = msg
        self.served = False

    def dequeue(self, indefinite=False, timeout=None):
        if self.served:
            raise StopIteration
        self.served = True
        return self.msg


# ---------------------------------------------------------------------------
# m17 engine-side death wiring (replica of core.py:L1029-L1031)
# ---------------------------------------------------------------------------


class TestFailureCallbackWiring:
    def test_executor_failed_sentinel_replica(self):
        # verbatim construction from vllm/v1/engine/core.py:L1029-L1031:
        #   executor_fail_callback = lambda: self.input_queue.put_nowait(
        #       (EngineCoreRequestType.EXECUTOR_FAILED, b""))
        EXECUTOR_FAILED = "EXECUTOR_FAILED"
        input_queue = queue_mod.Queue()

        class _Self:
            pass

        s = _Self()
        s.input_queue = input_queue
        executor_fail_callback = lambda: s.input_queue.put_nowait(  # noqa: E731
            (EXECUTOR_FAILED, b"")
        )
        ex = _ConcreteExecutor.__new__(_ConcreteExecutor)
        ex.register_failure_callback(executor_fail_callback)
        # base-class registration is a no-op (mp overrides it) — use the mp face
        mp_ex = MultiprocExecutor.__new__(MultiprocExecutor)
        mp_ex.is_failed = False
        mp_ex.register_failure_callback(executor_fail_callback)
        mp_ex.failure_callback()
        assert input_queue.get_nowait() == (EXECUTOR_FAILED, b"")


# ===========================================================================
# Real-multiprocessing end-to-end (spawn on win32)
# ===========================================================================


@pytest.fixture(scope="module")
def mp_executor():
    ex = spawn_mp_executor()
    yield ex
    ex.shutdown()


@pytest.mark.e2e
class TestMultiprocE2E:
    # ---- m4 bring-up -----------------------------------------------------
    def test_bringup_shape(self, mp_executor):
        assert len(mp_executor.workers) == 2
        assert mp_executor.rpc_broadcast_mq is not None
        assert len(mp_executor.response_mqs) == 2
        assert isinstance(mp_executor.futures_queue, deque)
        assert mp_executor.is_failed is False
        assert all(h.proc.is_alive() for h in mp_executor.workers)

    def test_ready_handshake_loaded_workers(self, mp_executor):
        ids = mp_executor.collective_rpc("get_identity")
        assert {i["rank"] for i in ids} == {0, 1}
        assert all(i["loaded"] for i in ids)  # load_model ran before READY
        assert len({i["pid"] for i in ids}) == 2  # separate processes

    # ---- m11 broadcast + callable ---------------------------------------
    def test_broadcast_reaches_all_workers(self, mp_executor):
        out = mp_executor.collective_rpc("echo", args=(42,), kwargs={"k": "v"})
        assert sorted(o["args"][0] for o in out) == [42, 42]
        assert {o["rank"] for o in out} == {0, 1}

    def test_callable_method_cloudpickled(self, mp_executor):
        out = mp_executor.collective_rpc(
            lambda worker, salt: (worker.rank, salt, os.getpid()), args=("s",)
        )
        assert sorted(o[0] for o in out) == [0, 1]
        assert all(o[1] == "s" for o in out)

    def test_rpc_failure_raises_with_worker_text(self):
        # A FAILURE poisons the response MQs (the real engine shuts down after
        # one) — dedicated single-worker executor for the error-path tests.
        ex = spawn_mp_executor(world_size=1)
        try:
            with pytest.raises(RuntimeError) as ei:
                ex.collective_rpc("boom")
            assert "boom in worker" in str(ei.value)
            assert "Worker failed with error" in str(ei.value)
        finally:
            ex.shutdown()

    def test_rpc_timeout(self):
        ex = spawn_mp_executor(world_size=1)
        try:
            with pytest.raises(TimeoutError, match="RPC call to slowpoke timed out"):
                ex.collective_rpc("slowpoke", timeout=1.0)
        finally:
            ex.shutdown()

    # ---- m14 output_rank single harvest ----------------------------------
    def test_output_rank_formula(self):
        ex = Executor.__new__(MultiprocExecutor)
        ex.parallel_config = mk_config(
            parallel_config={
                "world_size": 32,
                "tensor_parallel_size": 8,
                "prefill_context_parallel_size": 1,
            }
        ).parallel_config
        ex.world_size = 32
        assert ex._get_output_rank() == 24  # TP=8, PP=4 -> last PP stage TP-0

    def test_execute_model_single_result_from_output_rank(self, mp_executor):
        so = mk_scheduler_output(total=6)
        out = mp_executor.execute_model(so)
        assert isinstance(out, dict) and not isinstance(out, list)
        assert out["scheduled"] == 6

    def test_two_phase_pair_over_control_plane(self):
        ex = spawn_mp_executor(worker_cls="_worker_double.TwoPhaseProbeWorker")
        try:
            so = mk_scheduler_output(total=3)
            assert ex.execute_model(so) is None  # beat ② returns None
            out = ex.sample_tokens(None)  # beat ④
            assert out["scheduled"] == 3
            assert out["grammar"] is False
        finally:
            ex.shutdown()

    # ---- m17 process death path ------------------------------------------
    def test_worker_death_fires_failure_callback(self):
        ex = spawn_mp_executor()
        try:
            fired = []
            ex.register_failure_callback(lambda: fired.append("EXECUTOR_FAILED"))
            victim = ex.workers[0].proc
            victim.kill()
            victim.join(5)
            assert wait_until(lambda: bool(fired), timeout=15)
            assert ex.is_failed is True
            # after the monitor's shutdown() the broadcast MQ is None, so the
            # follower-node assert can trip before the is_failed check — both
            # are post-mortem refusals of the control plane
            with pytest.raises((RuntimeError, AssertionError)):
                ex.collective_rpc("echo")
        finally:
            ex.shutdown()

    # ---- m18 shutdown / death pipe / weakref ------------------------------
    def test_shutdown_terminates_workers_gracefully(self):
        ex = spawn_mp_executor()
        procs = [h.proc for h in ex.workers]
        ex.shutdown()
        assert wait_until(lambda: all(not p.is_alive() for p in procs))
        # MQ shut down with the executor
        assert ex.rpc_broadcast_mq is None
        assert ex.response_mqs == []

    def test_weakref_finalizer_registered_and_fires(self):
        # multiproc_executor.py:L113: weakref.finalize(self, self.shutdown) is
        # the leak guard. NOTE the bound-method pattern (real vLLM's own) keeps
        # a strong ref through the finalizer registry, so the finalizer is the
        # at-exit / explicit-call guard, not a gc hook — invoke it directly.
        ex = spawn_mp_executor(world_size=1)
        procs = [h.proc for h in ex.workers]
        assert ex._finalizer.alive is True
        ex._finalizer()  # fires shutdown() exactly as interpreter exit would
        assert wait_until(lambda: all(not p.is_alive() for p in procs))
        assert ex.rpc_broadcast_mq is None
        ex.shutdown()  # idempotent (shutting_down guard)

    # ---- m20 async output copy thread -------------------------------------
    def test_async_scheduling_output_copy_thread(self):
        ex = spawn_mp_executor(
            worker_cls="_worker_double.AsyncProbeWorker", async_scheduling=True
        )
        try:
            assert MultiprocExecutor.supports_async_scheduling() is True
            fut = ex.execute_model(mk_scheduler_output(), non_block=True)
            out = fut.result()
            # the async wrapper was resolved inside the child (enqueue_output)
            assert out["async"] is True
        finally:
            ex.shutdown()


import weakref as weakref_mod  # noqa: E402  (used above)


# ---------------------------------------------------------------------------
# lifecycle over the real control plane (m8 e2e)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestLifecycleOverControlPlane:
    def test_initialize_and_memory_anchors_over_rpc(self):
        ex = spawn_mp_executor()
        try:
            mem = ex.determine_available_memory()
            assert mem == [12345, 12345]
            cfg = _host_seams.KVCacheConfig(num_blocks=32)
            ex.initialize_from_config([cfg, cfg])
            status = ex.collective_rpc("get_identity")
            # worker state driven purely through the real RPC surface
            assert all(s["loaded"] for s in status)
        finally:
            ex.shutdown()

    def test_compile_times_max_over_workers(self):
        ex = spawn_mp_executor()
        try:
            ex.compile_or_warm_up_model()
            # ranks 0,1 report language_model 1.0 / 2.0 -> max propagated
            assert ex.vllm_config.compilation_config.compilation_time == 2.0
        finally:
            ex.shutdown()

    def test_check_health_roundtrip(self):
        ex = spawn_mp_executor()
        try:
            ex.check_health()  # collective_rpc("check_health", timeout=10)
        finally:
            ex.shutdown()
