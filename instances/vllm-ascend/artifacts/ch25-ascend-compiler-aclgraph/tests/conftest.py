"""Host scaffold for ch25 runnable tests (NOT part of the subtract-only source).

The implementation files are faithful subtract-only copies of vllm_ascend/compilation/*.py,
vllm_ascend/platform.py and vllm_ascend/ops/__init__.py. The real ones import torch_npu, vllm,
npugraph_ex/torchair — none available on a plain host. So BEFORE importing the impl modules we
inject lightweight stubs into sys.modules and synthesize a `torch.npu` namespace (host CUDA torch
has none). The stubs only stand in for *boundary* objects (NPUGraph, forward context, vllm config
enums); the control flow under test (compile() 二分 dispatch / ACLGraphWrapper capture-replay-分桶 /
207008 兜底 / pass manager 串 pass / dummy fusion op 锚点) is the real code, exercised for real.
"""
import enum
import importlib.util
import sys
import types
from pathlib import Path

import torch

IMPL = Path(__file__).resolve().parent.parent / "implementation"


def _mod(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# ---------------------------------------------------------------- torch.npu stub
class FakeNPUGraph:
    """Stands in for torch.npu.NPUGraph; records replay() invocations."""

    def __init__(self):
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1


class _FakeGraphCtx:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False  # do not swallow exceptions -> 207008 propagates to except


class _FakeStream:
    def __init__(self):
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1


_FAKE_STREAM = _FakeStream()

if not hasattr(torch, "npu"):
    torch.npu = types.SimpleNamespace()
torch.npu.NPUGraph = FakeNPUGraph
torch.npu.graph = lambda *a, **k: _FakeGraphCtx()
torch.npu.current_stream = lambda: _FAKE_STREAM
torch.npu.empty_cache = lambda: None
torch.npu.set_compile_mode = lambda **k: None

_tnpu = _mod("torch_npu")
_tnpu.__version__ = "stub"
_tnpu.npu = torch.npu


# ---------------------------------------------------------------- vllm.* stubs
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class VllmConfig:  # only used as a type hint target
    pass


class Range:
    pass


class CompilerInterface:
    def __init__(self, *a, **k):
        pass


class CUDAGraphOptions:
    def __init__(self, debug_log_enable=False, gc_disable=False, weak_ref_output=False):
        self.debug_log_enable = debug_log_enable
        self.gc_disable = gc_disable
        self.weak_ref_output = weak_ref_output


class _Logger:
    def info(self, *a, **k):
        pass

    debug = info_once = warning = error = info


class _Counter:
    num_cudagraph_captured = 0


class _Offloader:
    def sync_prev_onload(self):
        pass

    def join_after_forward(self):
        pass


# vllm package tree
_mod("vllm")
_envs = _mod("vllm.envs")
_envs.VLLM_LOGGING_LEVEL = "INFO"

_vlogger = _mod("vllm.logger")
_vlogger.logger = _Logger()

_vconfig = _mod("vllm.config")
_vconfig.VllmConfig = VllmConfig
_vconfig.CUDAGraphMode = CUDAGraphMode
_vcfg_utils = _mod("vllm.config.utils")
_vcfg_utils.Range = Range
_vcfg_compilation = _mod("vllm.config.compilation")
_vcfg_compilation.Range = Range

_mod("vllm.compilation")
_vci = _mod("vllm.compilation.compiler_interface")
_vci.CompilerInterface = CompilerInterface
_vcounter = _mod("vllm.compilation.counter")
_vcounter.compilation_counter = _Counter()
_vcg = _mod("vllm.compilation.cuda_graph")
_vcg.CUDAGraphOptions = CUDAGraphOptions
_vmon = _mod("vllm.compilation.monitor")
_vmon.validate_cudagraph_capturing_enabled = lambda: None

_vfc = _mod("vllm.forward_context")


class BatchDescriptor:
    pass


_vfc.BatchDescriptor = BatchDescriptor
# forward context is settable per-test through this holder
_vfc._CTX = types.SimpleNamespace(batch_descriptor=None, cudagraph_runtime_mode=CUDAGraphMode.NONE, capturing=False)
_vfc.get_forward_context = lambda: _vfc._CTX

_vplatforms = _mod("vllm.platforms")
_vplatforms.current_platform = types.SimpleNamespace(get_global_graph_pool=lambda: None)

_mod("vllm.model_executor")
_mod("vllm.model_executor.offloader")
_voff = _mod("vllm.model_executor.offloader.base")
_voff.get_offloader = lambda: _Offloader()

# vllm.compilation.passes.* (for graph_fusion_pass_manager)
_mod("vllm.compilation.passes")
_vip = _mod("vllm.compilation.passes.inductor_pass")
_vip.get_pass_context = lambda: types.SimpleNamespace(compile_range=Range())
_vvip = _mod("vllm.compilation.passes.vllm_inductor_pass")


class VllmInductorPass:
    def __init__(self, *a, **k):
        pass

    def is_applicable_for_range(self, compile_range):
        return True


_vvip.VllmInductorPass = VllmInductorPass


# ---------------------------------------------------------------- vllm_ascend.* stubs
def _weak_ref_tensors(x):
    return x  # on host, identity stands in for the weak-ref view


_va = _mod("vllm_ascend")
_va_utils = _mod("vllm_ascend.utils")
_va_utils.COMPILATION_PASS_KEY = "graph_fusion_manager"
_va_utils.weak_ref_tensors = _weak_ref_tensors
_va_utils.is_310p = lambda: False
_va_utils.enable_custom_op = lambda: True

_va_cfg = _mod("vllm_ascend.ascend_config")


class AscendCompilationConfig:
    pass


_va_cfg.AscendCompilationConfig = AscendCompilationConfig
# settable global config (tests flip enable_npugraph_ex)
_va_cfg._CFG = types.SimpleNamespace(
    ascend_compilation_config=types.SimpleNamespace(enable_npugraph_ex=False, enable_static_kernel=False)
)
_va_cfg.get_ascend_config = lambda: _va_cfg._CFG

_va_afc = _mod("vllm_ascend.ascend_forward_context")
_va_afc._EXTRA_CTX = types.SimpleNamespace(is_draft_model=False)

# package skeleton so relative imports inside the impl resolve
_va_comp = _mod("vllm_ascend.compilation")


# ---------------------------------------------------------------- load impl as canonical modules
def _load(dotted: str, filename: str, package: str | None = None):
    spec = importlib.util.spec_from_file_location(dotted, IMPL / filename)
    m = importlib.util.module_from_spec(spec)
    if package:
        m.__package__ = package
    sys.modules[dotted] = m
    spec.loader.exec_module(m)
    return m


acl_graph = _load("vllm_ascend.compilation.acl_graph", "acl_graph.py", package="vllm_ascend.compilation")
compiler_interface = _load(
    "vllm_ascend.compilation.compiler_interface", "compiler_interface.py", package="vllm_ascend.compilation"
)
graph_fusion_pass_manager = _load(
    "vllm_ascend.compilation.graph_fusion_pass_manager",
    "graph_fusion_pass_manager.py",
    package="vllm_ascend.compilation",
)
ops_dummy_fusion = _load("ops_dummy_fusion", "ops_dummy_fusion.py")
platform_hooks = _load("platform_hooks", "platform_hooks.py")

# expose to tests
sys.modules["_ch25_acl_graph"] = acl_graph
sys.modules["_ch25_compiler_interface"] = compiler_interface
sys.modules["_ch25_pass_manager"] = graph_fusion_pass_manager
sys.modules["_ch25_ops"] = ops_dummy_fusion
sys.modules["_ch25_platform_hooks"] = platform_hooks
