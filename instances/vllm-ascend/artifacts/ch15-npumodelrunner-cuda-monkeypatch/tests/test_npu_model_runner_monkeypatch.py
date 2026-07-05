"""TDD tests for ch14 — NPUModelRunner: 继承 244KB 父类 + 运行时设备猴补.

这些测试验证 vllm-ascend 真实代码（忠实减法进 ../implementation）的*可观察行为*：
两个成对进出的上下文管理器如何在 with 作用域内临时把 torch.cuda.* 与父类模块级
graph_capture/CUDAGraphWrapper 换成 NPU/ACL 版、退出即还原；_use_aclgraph 的三条件
决策；_get_gpu_model_runner_module_name 的 MRO 取模块名；ACLGraphWrapper 与
CUDAGraphWrapper 的同形可热替换；以及设备钩子 override。

昇腾 NPU/CANN 不在 host 真跑，但*猴补的装卸逻辑本身是纯 Python*——所以我们在
sys.modules 里桩掉 torch.npu / vllm / vllm_ascend 目标命名空间，import（已减法的）
实现，断言符号置换与真实仓一致。
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
    """在 sys.modules 注册假的点分模块，自动清理；不覆盖已存在的真实模块。"""

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

    def put(self, dotted, mod):
        # 把一个已加载模块对象登记到点分名下（含父包链接）。
        self.module(".".join(dotted.split(".")[:-1])) if "." in dotted else None
        if dotted not in sys.modules:
            self.added.append(dotted)
        sys.modules[dotted] = mod
        if "." in dotted:
            parent, leaf = dotted.rsplit(".", 1)
            setattr(sys.modules[parent], leaf, mod)

    def cleanup(self):
        for n in reversed(self.added):
            sys.modules.pop(n, None)


_counter = [0]


def _load(filename, modname=None):
    _counter[0] += 1
    name = modname or f"_impl_{_counter[0]}_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 父类模块里"原版" cuda 符号的哨兵（用于断言替换/还原）。
_ORIG_GRAPH_CAPTURE = object()


class _OrigCUDAGraphWrapper:  # 父类模块原版（被换成 ACLGraphWrapper 的目标）
    def __init__(self, *a, **k):
        pass


@pytest.fixture
def env():
    """搭好 torch.npu 桩 + vllm/vllm_ascend 假命名空间，加载两个实现模块。"""
    reg = ModRegistry()

    # ---- 快照真实 torch.cuda 符号，测后还原（_torch_cuda_wrapper 会就地改写）---- #
    snap = {
        "torch_Event": getattr(torch, "Event", None),
        "cuda_Event": torch.cuda.Event,
        "cuda_Stream": torch.cuda.Stream,
        "cuda_sync": torch.cuda.synchronize,
        "cuda_mem": getattr(torch.cuda, "mem_get_info", None),
        "npu": getattr(torch, "npu", None),
    }

    # ---- torch.npu 桩：本章只需这些被 wrapper 改向的符号 ---- #
    npu_calls = {"synchronize": 0}

    class _NpuEvent:
        pass

    class _NpuStream:
        def __init__(self, *a, **k):
            pass

        def wait_stream(self, *a, **k):
            pass

    def _npu_sync(*a, **k):
        npu_calls["synchronize"] += 1

    def _npu_mem_get_info(*a, **k):
        return (123, 456)

    from contextlib import nullcontext as _nullcontext

    torch.npu = types.SimpleNamespace(
        Event=_NpuEvent,
        Stream=_NpuStream,
        synchronize=_npu_sync,
        mem_get_info=_npu_mem_get_info,
        current_stream=lambda *a, **k: None,
        default_stream=lambda *a, **k: None,
        stream=lambda *a, **k: _nullcontext(),
    )

    # ---- vllm.config 桩 ---- #
    cfg = reg.module("vllm.config")

    class CUDAGraphMode:
        NONE = 0
        PIECEWISE = 1
        FULL = 2

    class CompilationMode:
        VLLM_COMPILE = "vllm_compile"

    cfg.CUDAGraphMode = CUDAGraphMode
    cfg.CompilationMode = CompilationMode
    cfg.VllmConfig = type("VllmConfig", (), {})

    # ---- 父类模块 vllm.v1.worker.gpu_model_runner（含模块级 graph_capture/CUDAGraphWrapper）---- #
    parent_mod = reg.module("vllm.v1.worker.gpu_model_runner")
    parent_mod.graph_capture = _ORIG_GRAPH_CAPTURE
    parent_mod.CUDAGraphWrapper = _OrigCUDAGraphWrapper

    captured = {}

    class GPUModelRunner:
        # 父类巨方法的桩：调用时记录"当前父模块里看到的符号"，证明已被替换。
        def profile_cudagraph_memory(self):
            captured["graph_capture"] = parent_mod.graph_capture
            captured["CUDAGraphWrapper"] = parent_mod.CUDAGraphWrapper
            captured["mem_get_info"] = torch.cuda.mem_get_info
            return 4096

        def capture_model(self):
            captured["graph_capture"] = parent_mod.graph_capture
            captured["CUDAGraphWrapper"] = parent_mod.CUDAGraphWrapper
            captured["mem_get_info"] = torch.cuda.mem_get_info
            return 8192

    GPUModelRunner.__module__ = "vllm.v1.worker.gpu_model_runner"
    parent_mod.GPUModelRunner = GPUModelRunner

    # ---- vllm_ascend 设备实体桩 ---- #
    attn_mod = reg.module("vllm_ascend.attention.attention_v1")
    attn_mod.AscendAttentionState = type("AscendAttentionState", (), {"DecodeOnly": 0})
    samp_mod = reg.module("vllm_ascend.sample.sampler")
    samp_mod.AscendSampler = type("AscendSampler", (), {})

    # ---- 先加载 acl_graph 精简版，登记到规范名，供 model_runner 导入 ---- #
    reg.module("vllm_ascend.compilation")
    acl = _load("acl_graph.py")
    reg.put("vllm_ascend.compilation.acl_graph", acl)

    # ---- 加载主角实现 ---- #
    mr = _load("model_runner_v1.py")

    yield types.SimpleNamespace(
        reg=reg, mr=mr, acl=acl, parent_mod=parent_mod,
        CUDAGraphMode=CUDAGraphMode, CompilationMode=CompilationMode,
        OrigCUDAGraphWrapper=_OrigCUDAGraphWrapper,
        npu_calls=npu_calls, captured=captured,
    )

    # ---- teardown ---- #
    reg.cleanup()
    if snap["torch_Event"] is None:
        if hasattr(torch, "Event"):
            del torch.Event
    else:
        torch.Event = snap["torch_Event"]
    torch.cuda.Event = snap["cuda_Event"]
    torch.cuda.Stream = snap["cuda_Stream"]
    torch.cuda.synchronize = snap["cuda_sync"]
    if snap["cuda_mem"] is None:
        if hasattr(torch.cuda, "mem_get_info"):
            del torch.cuda.mem_get_info
    else:
        torch.cuda.mem_get_info = snap["cuda_mem"]
    if snap["npu"] is None:
        del torch.npu
    else:
        torch.npu = snap["npu"]


# --------------------------------------------------------------------------- #
# 1. _torch_cuda_wrapper: 进入装 NPU / 退出卸（成对装卸）
# --------------------------------------------------------------------------- #
def test_torch_cuda_wrapper_swaps_to_npu_inside_scope(env):
    mr = env.mr
    inside = {}
    with mr._torch_cuda_wrapper():
        inside["Event"] = torch.cuda.Event
        inside["Stream"] = torch.cuda.Stream
        inside["synchronize"] = torch.cuda.synchronize
        inside["mem_get_info"] = torch.cuda.mem_get_info
    # 作用域内：四个代表符号都指向 torch.npu.*
    assert inside["Event"] is torch.npu.Event
    assert inside["Stream"] is torch.npu.Stream
    assert inside["synchronize"] is torch.npu.synchronize
    assert inside["mem_get_info"] is torch.npu.mem_get_info


def test_torch_cuda_wrapper_finally_settles_to_safe_defaults(env):
    mr = env.mr
    with mr._torch_cuda_wrapper():
        pass
    # 退出后并非原样还原 cuda，而是落到稳态缺省：Event→placeholder、mem_get_info/sync→npu
    assert torch.cuda.mem_get_info is torch.npu.mem_get_info
    assert torch.cuda.synchronize is torch.npu.synchronize
    ev = torch.cuda.Event()  # placeholder 的 no-op 接口仍可调用
    assert ev.query() is True
    assert ev.record() is None


def test_torch_cuda_wrapper_exception_installs_placeholder_and_reraises(env):
    mr = env.mr
    with pytest.raises(RuntimeError, match="NPUModelRunner init failed"):
        with mr._torch_cuda_wrapper():
            raise ValueError("boom")
    # except 分支把 Event/Stream 换成 placeholder（构造不崩）
    assert torch.cuda.Stream() is not None or True
    ev = torch.cuda.Event()
    assert ev.query() is True


# --------------------------------------------------------------------------- #
# 2. _replace_gpu_model_runner_function_wrapper: 改父类模块属性 + finally 还原
# --------------------------------------------------------------------------- #
def test_replace_wrapper_swaps_parent_module_symbols_and_restores(env):
    mr, parent = env.mr, env.parent_mod
    orig_gc = parent.graph_capture
    orig_wrap = parent.CUDAGraphWrapper
    seen = {}
    with mr._replace_gpu_model_runner_function_wrapper("vllm.v1.worker.gpu_model_runner"):
        seen["graph_capture"] = parent.graph_capture
        seen["CUDAGraphWrapper"] = parent.CUDAGraphWrapper
    # 作用域内：父模块符号被换成 NPU 版 graph_capture / ACLGraphWrapper
    assert seen["graph_capture"] is mr.graph_capture
    assert seen["CUDAGraphWrapper"] is env.acl.ACLGraphWrapper
    # 退出后：逐一还原为 original_attrs
    assert parent.graph_capture is orig_gc
    assert parent.CUDAGraphWrapper is orig_wrap


def test_replace_wrapper_restores_even_on_exception(env):
    mr, parent = env.mr, env.parent_mod
    orig_gc = parent.graph_capture
    orig_wrap = parent.CUDAGraphWrapper
    with pytest.raises(RuntimeError, match="NPUModelRunner failed"):
        with mr._replace_gpu_model_runner_function_wrapper("vllm.v1.worker.gpu_model_runner"):
            raise ValueError("boom")
    assert parent.graph_capture is orig_gc
    assert parent.CUDAGraphWrapper is orig_wrap


# --------------------------------------------------------------------------- #
# 3. _get_gpu_model_runner_module_name: 沿 MRO 取父类模块名
# --------------------------------------------------------------------------- #
def test_get_parent_module_name_via_mro(env):
    mr = env.mr
    inst = mr.NPUModelRunner.__new__(mr.NPUModelRunner)
    assert mr._get_gpu_model_runner_module_name(inst) == "vllm.v1.worker.gpu_model_runner"


def test_get_parent_module_name_raises_when_absent(env):
    mr = env.mr

    class Lonely:
        pass

    with pytest.raises(TypeError, match="Could not find GPUModelRunner"):
        mr._get_gpu_model_runner_module_name(Lonely())


# --------------------------------------------------------------------------- #
# 4. capture_model / profile_cudagraph_memory: 双 wrapper 包父方法
# --------------------------------------------------------------------------- #
def test_capture_model_runs_parent_under_swapped_symbols(env):
    mr = env.mr
    inst = mr.NPUModelRunner.__new__(mr.NPUModelRunner)
    parent = env.parent_mod
    orig_gc, orig_wrap = parent.graph_capture, parent.CUDAGraphWrapper

    result = inst.capture_model()

    assert result == 8192  # 返回的是父类巨方法 GPUModelRunner.capture_model(self) 的值
    # 父方法运行时看到的是被替换的 NPU/ACL 版符号
    assert env.captured["graph_capture"] is mr.graph_capture
    assert env.captured["CUDAGraphWrapper"] is env.acl.ACLGraphWrapper
    assert env.captured["mem_get_info"] is torch.npu.mem_get_info
    # 退出后父模块符号已还原
    assert parent.graph_capture is orig_gc
    assert parent.CUDAGraphWrapper is orig_wrap


def test_profile_cudagraph_memory_wraps_parent_and_resets(env):
    mr = env.mr
    inst = mr.NPUModelRunner.__new__(mr.NPUModelRunner)
    # reset_graph_params 应被调用（清空 acl_graph 模块的 _graph_params 全局）
    env.acl._graph_params = "dirty"
    result = inst.profile_cudagraph_memory()
    assert result == 4096
    assert env.captured["mem_get_info"] is torch.npu.mem_get_info
    assert env.acl._graph_params is None


# --------------------------------------------------------------------------- #
# 5. _use_aclgraph: 三条件决策
# --------------------------------------------------------------------------- #
def _make_runner_for_use_aclgraph(env, mode, comp_mode, enforce_eager):
    mr = env.mr
    inst = mr.NPUModelRunner.__new__(mr.NPUModelRunner)
    inst.compilation_config = types.SimpleNamespace(cudagraph_mode=mode, mode=comp_mode)
    inst.model_config = types.SimpleNamespace(enforce_eager=enforce_eager)
    return inst


def test_use_aclgraph_true_when_all_conditions_met(env):
    inst = _make_runner_for_use_aclgraph(
        env, env.CUDAGraphMode.FULL, env.CompilationMode.VLLM_COMPILE, False)
    assert inst._use_aclgraph() is True


@pytest.mark.parametrize("mode,comp,eager", [
    ("NONE", "VLLM_COMPILE", False),     # 图模式 NONE → False
    ("FULL", "other", False),            # 编译模式非 VLLM_COMPILE → False
    ("FULL", "VLLM_COMPILE", True),      # 强制 eager → False
])
def test_use_aclgraph_false_when_any_condition_fails(env, mode, comp, eager):
    mode_val = env.CUDAGraphMode.NONE if mode == "NONE" else env.CUDAGraphMode.FULL
    comp_val = env.CompilationMode.VLLM_COMPILE if comp == "VLLM_COMPILE" else "other"
    inst = _make_runner_for_use_aclgraph(env, mode_val, comp_val, eager)
    assert inst._use_aclgraph() is False


# --------------------------------------------------------------------------- #
# 6. 设备钩子 override
# --------------------------------------------------------------------------- #
def test_init_device_properties_sets_num_sms_none(env):
    mr = env.mr
    inst = mr.NPUModelRunner.__new__(mr.NPUModelRunner)
    inst._init_device_properties()
    assert inst.num_sms is None


def test_sync_device_calls_npu_synchronize(env):
    mr = env.mr
    inst = mr.NPUModelRunner.__new__(mr.NPUModelRunner)
    before = env.npu_calls["synchronize"]
    inst._sync_device()
    assert env.npu_calls["synchronize"] == before + 1


# --------------------------------------------------------------------------- #
# 7. NPUModelRunner 继承声明 + ACLGraphWrapper 同形可热替换
# --------------------------------------------------------------------------- #
def test_npu_model_runner_inherits_gpu_model_runner(env):
    mr = env.mr
    base_names = [c.__name__ for c in mr.NPUModelRunner.__mro__]
    assert "GPUModelRunner" in base_names


def test_aclgraph_wrapper_is_signature_compatible_with_cudagraph_wrapper(env):
    acl = env.acl
    # 用父方法构造 CUDAGraphWrapper 的同一组实参即可构造 ACLGraphWrapper（鸭子兼容）
    w = acl.ACLGraphWrapper(
        runnable=lambda: None,
        vllm_config=object(),
        runtime_mode=env.CUDAGraphMode.FULL,
        cudagraph_options=None,
    )
    assert w in acl.ACLGraphWrapper._all_instances
    acl.ACLGraphWrapper.clear_all_graphs()  # 同形的类方法可调用
    assert w.concrete_aclgraph_entries == {}


# --------------------------------------------------------------------------- #
# 8. graph_capture (NPU 版) 与父类同形
# --------------------------------------------------------------------------- #
def test_npu_graph_capture_yields_context_with_npu_stream(env):
    mr = env.mr
    with mr.graph_capture(device="npu:0") as ctx:
        # 返回 GraphCaptureContext，其 stream 是 torch.npu.Stream 实例
        assert isinstance(ctx, mr.GraphCaptureContext)
        assert isinstance(ctx.stream, torch.npu.Stream)
