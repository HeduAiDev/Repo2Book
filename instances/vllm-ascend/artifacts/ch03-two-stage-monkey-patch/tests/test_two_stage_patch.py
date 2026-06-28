"""TDD tests for ch03 — two-stage monkey-patch.

These tests exercise the *observable behavior* of the real vllm-ascend patch
code (faithfully subtracted into ../implementation): the single entry point's
two-way dispatch, the idempotent global-patch guard, and the 5 rebinding
techniques. Ascend NPU/CANN code cannot run on host, but the *rebinding logic
itself is pure Python* — so we stub the vLLM/vllm_ascend target namespaces in
sys.modules, import the (subtracted) patch modules, and assert the rebind took
effect exactly as in the real repo.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class ModRegistry:
    """Register fake dotted modules in sys.modules, with auto-cleanup.

    Never clobbers an already-present real module (e.g. torch); only creates
    the missing links and records what it added so the fixture can pop them.
    """

    def __init__(self):
        self.added = []

    def module(self, dotted):
        parts = dotted.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[: i + 1])
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m
                self.added.append(name)
                if i > 0:
                    setattr(sys.modules[".".join(parts[:i])], parts[i], m)
        return sys.modules[dotted]

    def cleanup(self):
        for n in reversed(self.added):
            sys.modules.pop(n, None)


@pytest.fixture
def reg():
    r = ModRegistry()
    yield r
    r.cleanup()


_counter = [0]


def load_fresh(filename):
    """Import implementation/<filename> as a brand-new module (re-executed)."""
    _counter[0] += 1
    name = f"_impl_{_counter[0]}_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# single entry point: adapt_patch two-way dispatch
# --------------------------------------------------------------------------- #
def test_adapt_patch_dispatches_to_platform_or_worker(reg):
    triggers = load_fresh("triggers.py")

    recorded = []
    patch_pkg = reg.module("vllm_ascend.patch")

    def _getattr(item):  # PEP 562: records which subpackage adapt_patch imports
        if item.startswith("__"):  # import machinery probes __path__/__spec__/...
            raise AttributeError(item)
        recorded.append(item)
        return types.ModuleType("vllm_ascend.patch." + item)

    patch_pkg.__getattr__ = _getattr

    triggers.adapt_patch(is_global_patch=True)
    assert recorded == ["platform"], "is_global_patch=True must import the platform stage"

    triggers.adapt_patch()  # default False
    assert recorded == ["platform", "worker"], "default must import the worker stage"


# --------------------------------------------------------------------------- #
# platform stage trigger ②: idempotent process-wide guard (ch02 f1 recovery)
# --------------------------------------------------------------------------- #
def test_ensure_global_patch_is_idempotent(reg):
    triggers = load_fresh("triggers.py")

    utils = reg.module("vllm_ascend.utils")
    calls = []

    def adapt_patch(is_global_patch=False):
        calls.append(is_global_patch)

    utils.adapt_patch = adapt_patch

    assert triggers._GLOBAL_PATCH_APPLIED is False
    triggers._ensure_global_patch()
    triggers._ensure_global_patch()
    triggers._ensure_global_patch()

    assert calls == [True], "guard must apply the platform stage exactly once"
    assert triggers._GLOBAL_PATCH_APPLIED is True


def test_register_connector_triggers_global_patch(reg):
    triggers = load_fresh("triggers.py")
    utils = reg.module("vllm_ascend.utils")
    calls = []
    utils.adapt_patch = lambda is_global_patch=False: calls.append(is_global_patch)

    kv = reg.module("vllm_ascend.distributed.kv_transfer")
    kv.register_connector = lambda: None

    triggers.register_connector()
    assert calls == [True], "register_connector must ensure the platform stage first"


# --------------------------------------------------------------------------- #
# technique ③: method replacement (patch_scheduler)
# --------------------------------------------------------------------------- #
def test_scheduler_method_replacement(reg):
    sched_mod = reg.module("vllm.v1.core.sched.scheduler")

    class Scheduler:
        def _mamba_block_aligned_split(self, *a, **k):
            return "ORIGINAL"

    sched_mod.Scheduler = Scheduler
    req_mod = reg.module("vllm.v1.request")
    req_mod.Request = type("Request", (), {})

    mod = load_fresh("patch_scheduler.py")

    # exactly the module-level function is now bound on the target class
    assert Scheduler._mamba_block_aligned_split is mod._mamba_block_aligned_split
    # no subclass was created — it is the same class object, only one method swapped
    assert sched_mod.Scheduler is Scheduler


# --------------------------------------------------------------------------- #
# technique ②: factory / registry replacement (patch_mamba_manager)
# --------------------------------------------------------------------------- #
def test_mamba_manager_factory_table_replacement(reg):
    m = reg.module("vllm.v1.core.single_type_kv_cache_manager")

    class MambaManager:
        def __init__(self, kv_cache_spec, block_pool, **kwargs):
            self.enable_caching = False

    class MambaSpec:
        pass

    m.BlockPool = type("BlockPool", (), {})
    m.MambaManager = MambaManager
    m.MambaSpec = MambaSpec
    m.spec_manager_map = {MambaSpec: MambaManager}

    mod = load_fresh("patch_mamba_manager.py")

    # both the class name AND the factory dispatch table must point to the subclass
    assert m.MambaManager is mod.AscendMambaManager
    assert m.spec_manager_map[MambaSpec] is mod.AscendMambaManager
    assert issubclass(mod.AscendMambaManager, MambaManager)


# --------------------------------------------------------------------------- #
# technique ①: whole-class replacement (patch_multiproc_executor)
# --------------------------------------------------------------------------- #
def test_multiproc_executor_whole_class_replacement(reg):
    m = reg.module("vllm.v1.executor.multiproc_executor")

    class MultiprocExecutor:
        pass

    m.MultiprocExecutor = MultiprocExecutor
    m.WorkerProc = type("WorkerProc", (), {"worker_main": staticmethod(lambda: None)})

    mod = load_fresh("patch_multiproc_executor.py")

    assert m.MultiprocExecutor is mod.AscendMultiprocExecutor
    assert issubclass(mod.AscendMultiprocExecutor, MultiprocExecutor)


# --------------------------------------------------------------------------- #
# techniques ④ + ⑤: library-function wrapper + from-import cache trap
# --------------------------------------------------------------------------- #
def test_distributed_wrapper_and_cache_trap(reg):
    utils = reg.module("vllm_ascend.utils")

    class AscendDeviceType:
        _310P = "310P"
        OTHER = "other"

    utils.AscendDeviceType = AscendDeviceType
    # not 310P → module-level guard must NOT auto-run communication_adaptation_310p
    utils.get_ascend_device_type = lambda: AscendDeviceType.OTHER

    dist = torch.distributed
    c10d = torch.distributed.distributed_c10d
    saved = (dist.broadcast, dist.all_reduce, c10d.broadcast, c10d.all_reduce)
    try:
        mod = load_fresh("patch_distributed_platform.py")
        # guard did not fire → originals untouched after import
        assert dist.broadcast is saved[0]

        # install sentinels so the wrapper captures *them* as the fallback fn
        dist.broadcast = lambda tensor, src=0, group=None, async_op=False: "TOP_FALLBACK"
        c10d.broadcast = lambda tensor, src=0, group=None, async_op=False: "C10D_FALLBACK"

        mod.communication_adaptation_310p()

        # technique ⑤: BOTH the top-level name and the distributed_c10d alias rebound
        assert dist.broadcast(torch.zeros(2)) == "TOP_FALLBACK"
        assert c10d.broadcast(torch.zeros(2)) == "C10D_FALLBACK"
        # technique ④: the new callables are the nested `broadcast310p` wrapper
        assert dist.broadcast.__name__ == "broadcast310p"
        assert c10d.broadcast.__name__ == "broadcast310p"
    finally:
        dist.broadcast, dist.all_reduce, c10d.broadcast, c10d.all_reduce = saved


# --------------------------------------------------------------------------- #
# technique ④ minimal: give a library module a missing function (patch_triton)
# --------------------------------------------------------------------------- #
def test_triton_next_power_of_2_rebind(reg):
    conv = reg.module("vllm.model_executor.layers.mamba.ops.causal_conv1d")
    triton_utils = reg.module("vllm.triton_utils")
    triton_mod = types.ModuleType("triton")  # the library module we patch onto
    triton_utils.HAS_TRITON = True
    triton_utils.triton = triton_mod

    math_utils = reg.module("vllm.utils.math_utils")
    sentinel = lambda x: x  # noqa: E731
    math_utils.next_power_of_2 = sentinel

    ascend_conv = reg.module("vllm_ascend.ops.triton.mamba.causal_conv1d")
    ascend_conv.causal_conv1d_fn = lambda *a, **k: None
    ascend_conv.causal_conv1d_update_npu = lambda *a, **k: None

    assert not hasattr(triton_mod, "next_power_of_2")
    load_fresh("patch_triton.py")
    # the function vLLM bundles is now attached to the triton module
    assert triton_mod.next_power_of_2 is sentinel
    # and the causal_conv1d ops were swapped on the library module too
    assert conv.causal_conv1d_update is ascend_conv.causal_conv1d_update_npu


# --------------------------------------------------------------------------- #
# comprehensive sample (worker patch_distributed): ① + ④ + ⑤ together
# --------------------------------------------------------------------------- #
def test_worker_distributed_comprehensive(reg):
    vllm = reg.module("vllm")
    dist = reg.module("vllm.distributed")
    ps = reg.module("vllm.distributed.parallel_state")

    class GroupCoordinator:
        pass  # base intentionally has no all_to_all

    def _orig_destroy():
        return "destroyed"

    ps.GroupCoordinator = GroupCoordinator
    ps._get_unique_name = lambda name: name
    ps._register_group = lambda self: None
    ps.destroy_distributed_environment = _orig_destroy
    dist.destroy_distributed_environment = _orig_destroy

    hccl = reg.module("vllm_ascend.patch.worker._hccl_pg_registry")
    hccl.HcclPgRegistry = type("HcclPgRegistry", (), {"clear": lambda self: None})

    mod = load_fresh("patch_distributed_worker.py")

    # technique ①: whole-class replacement
    assert ps.GroupCoordinator is mod.GroupCoordinatorPatch
    assert issubclass(mod.GroupCoordinatorPatch, GroupCoordinator)
    # motivation for the subclass: it adds all_to_all the base never had
    assert hasattr(mod.GroupCoordinatorPatch, "all_to_all")
    assert not hasattr(GroupCoordinator, "all_to_all")

    # technique ⑤: same wrapped fn bound on BOTH the top-level + re-exported alias
    assert ps.destroy_distributed_environment is dist.destroy_distributed_environment
    wrapped = ps.destroy_distributed_environment
    assert wrapped is not _orig_destroy
    # technique ④: idempotency marker prevents double-wrapping
    assert wrapped._hccl_registry_clearing_wrapped is True
    assert mod._wrap_destroy_distributed_environment(wrapped) is wrapped
    # the wrapper still delegates to the original
    assert wrapped() == "destroyed"
