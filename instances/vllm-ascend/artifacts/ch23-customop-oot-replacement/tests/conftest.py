"""ch23 测试脚手架：host 无 NPU/CANN，在 sys.modules 桩掉 torch_npu / _C_ascend / vllm 重运行时依赖，
再把（已减法的）implementation/ 模块按**规范模块名**注册进去，让它们彼此 import 解析到精简版。

可在 host 验证、与真仓一致的纯 Python 控制流：
  (1) register_ascend_customop —— 建 REGISTERED_ASCEND_OPS 表 + 遍历 register_oot 写入 op_registry_oot；幂等闸只生效一次；
  (2) CustomOp.__new__ —— 按 op_registry_oot 把 RMSNorm()/SiluAndMul() 换身成 Ascend 子类；
  (3) dispatch_forward —— enabled 且 is_out_of_tree()→forward_oot；未 enabled→forward_native；
  (4) AscendRMSNorm.forward_oot —— enable_custom_op() 真二分：融合算子 vs 原子算子回退；
  (5) register_oot 把 op.name 覆盖成类名键（'RMSNorm' 而非 'rms_norm'）。
真实 torch_npu / _C_ascend 算子由「记录调用」替身承接——只验入参/分流，不真算（昇腾才有内核）。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _load(filename, modname):
    spec = importlib.util.spec_from_file_location(modname, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    if "." in modname:
        parent = modname.rsplit(".", 1)[0]
        if parent in sys.modules:
            setattr(sys.modules[parent], modname.rsplit(".", 1)[1], mod)
    spec.loader.exec_module(mod)
    return mod


class _Stubs:
    def __init__(self):
        self.added = []

    def mod(self, dotted):
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


class _Recorder:
    """记录每个算子调用 (name, args)。算子不真算，返回形状自洽的占位张量。"""

    def __init__(self):
        self.calls = []

    def names(self):
        return [c[0] for c in self.calls]

    def _rec(self, name, args):
        self.calls.append((name, args))


class _Namespace:
    """torch.ops.<ns> 的替身：任意属性都返回一个记录型可调用。"""

    def __init__(self, ns, rec, returns):
        self._ns = ns
        self._rec = rec
        self._returns = returns

    def __getattr__(self, op):
        def _call(*args, **kwargs):
            self._rec._rec(f"{self._ns}.{op}", args)
            fn = self._returns.get(op)
            return fn(*args, **kwargs) if fn else None
        return _call


class _Ops:
    """torch.ops 的替身（_C / _C_ascend / vllm 命名空间）。"""

    def __init__(self, rec):
        self._rec = rec
        self._returns = {
            # 融合分支目标算子：返回 (x, None, residual)
            "npu_add_rms_norm_bias": lambda x, residual, w, b, eps: (x, None, residual),
            # residual 切块：透传
            "maybe_chunk_residual": lambda x, residual: residual,
        }

    def __getattr__(self, ns):
        return _Namespace(ns, self._rec, self._returns)


class _TorchNpu:
    """torch_npu 替身：记录调用，返回形状自洽占位。"""

    def __init__(self, rec):
        self._rec = rec

    def npu_add_rms_norm(self, x, residual, weight, eps):
        self._rec._rec("torch_npu.npu_add_rms_norm", (x, residual, weight, eps))
        return x, None, residual

    def npu_rms_norm(self, x, weight, eps):
        self._rec._rec("torch_npu.npu_rms_norm", (x, weight, eps))
        return x, x

    def npu_swiglu(self, x):
        self._rec._rec("torch_npu.npu_swiglu", (x,))
        return x

    def npu_fast_gelu(self, x):
        self._rec._rec("torch_npu.npu_fast_gelu", (x,))
        return x


def _make_platform(knobs):
    p = types.SimpleNamespace()
    p.is_out_of_tree = lambda: knobs.is_out_of_tree
    p.is_rocm = lambda: False
    p.is_cpu = lambda: False
    p.is_tpu = lambda: False
    p.is_xpu = lambda: False
    p.is_cuda_alike = lambda: False
    return p


def _make_compilation_config(knobs):
    return types.SimpleNamespace(
        enabled_custom_ops=set(),
        disabled_custom_ops=set(),
        custom_ops=knobs.custom_ops,
    )


def _make_vllm_config():
    priority = types.SimpleNamespace(rms_norm=["cuda"], fused_add_rms_norm=["cuda"])
    return types.SimpleNamespace(
        kernel_config=types.SimpleNamespace(ir_op_priority=priority),
        quant_config=None,
    )


@pytest.fixture
def env():
    stubs = _Stubs()
    rec = _Recorder()

    knobs = types.SimpleNamespace(
        is_out_of_tree=True,
        custom_ops=["all"],  # default_on → True（算子默认启用）
    )

    # ---- torch.ops / torch_npu 替身 ---- #
    orig_ops = torch.ops
    torch.ops = _Ops(rec)
    sys.modules["torch_npu"] = _TorchNpu(rec)
    stubs.added.append("torch_npu")

    # ---- vllm 顶层 + 子模块 ---- #
    stubs.mod("vllm")

    # vllm.logger
    logger_mod = stubs.mod("vllm.logger")
    def _init_logger(_name):
        lg = types.SimpleNamespace()
        lg.debug = lambda *a, **k: None
        lg.warning = lambda *a, **k: None
        lg.warning_once = lambda *a, **k: None
        return lg
    logger_mod.init_logger = _init_logger

    # vllm.platforms.current_platform
    plat = stubs.mod("vllm.platforms")
    plat.current_platform = _make_platform(knobs)

    # vllm.config
    cfg = stubs.mod("vllm.config")
    cfg.VllmConfig = type("VllmConfig", (), {})
    _comp = _make_compilation_config(knobs)
    cfg.get_cached_compilation_config = lambda: _comp
    cfg.get_current_vllm_config = lambda: _make_vllm_config()

    # vllm.envs / vllm.ir / batch_invariant（base_layernorm 用，本章不走其热路径）
    envs = stubs.mod("vllm.envs")
    envs.VLLM_BATCH_INVARIANT = False
    ir = stubs.mod("vllm.ir")
    ir.ops = types.SimpleNamespace(
        rms_norm=lambda *a: a[0],
        fused_add_rms_norm=types.SimpleNamespace(maybe_inplace=lambda *a: (a[0], a[1])),
    )
    bi = stubs.mod("vllm.model_executor.layers.batch_invariant")
    bi.rms_norm_batch_invariant = lambda x, w, eps: x

    # base FusedMoE（ch26 专题，本章作注册表第四代表项；out-of-scope 用桩基类）
    fm = stubs.mod("vllm.model_executor.layers.fused_moe.layer")

    # ---- vllm_ascend 包占位 ---- #
    stubs.mod("vllm_ascend")
    stubs.mod("vllm_ascend.ops")
    stubs.mod("vllm_ascend.ops.fused_moe")

    # 注：host 无编译产物，`import vllm_ascend.vllm_ascend_C` 天然抛 ImportError → enable_custom_op() 回退。
    # 要验融合分支时，测试直接置 utils._CUSTOM_OP_ENABLED=True 模拟「库在」。

    # ---- 按依赖顺序加载精简版 ---- #
    custom_op = _load("custom_op.py", "vllm.model_executor.custom_op")
    fm.FusedMoE = type("FusedMoE", (custom_op.CustomOp,), {})
    _load("base_activation.py", "vllm.model_executor.layers.activation")
    _load("base_layernorm.py", "vllm.model_executor.layers.layernorm")
    utils = _load("utils.py", "vllm_ascend.utils")
    _load("ascend_activation.py", "vllm_ascend.ops.activation")
    _load("ascend_layernorm.py", "vllm_ascend.ops.layernorm")
    _load("ascend_fused_moe.py", "vllm_ascend.ops.fused_moe.fused_moe")

    mods = types.SimpleNamespace(
        custom_op=custom_op,
        utils=utils,
        rec=rec,
        knobs=knobs,
        comp=_comp,
        activation=sys.modules["vllm.model_executor.layers.activation"],
        layernorm=sys.modules["vllm.model_executor.layers.layernorm"],
        ascend_activation=sys.modules["vllm_ascend.ops.activation"],
        ascend_layernorm=sys.modules["vllm_ascend.ops.layernorm"],
    )
    try:
        yield mods, knobs
    finally:
        torch.ops = orig_ops
        stubs.cleanup()
        for n in (
            "vllm.logger",
            "vllm.platforms",
            "vllm.config",
            "vllm.envs",
            "vllm.ir",
            "vllm.model_executor.layers.batch_invariant",
            "vllm.model_executor.layers.fused_moe.layer",
            "vllm.model_executor.custom_op",
            "vllm.model_executor.layers.activation",
            "vllm.model_executor.layers.layernorm",
            "vllm_ascend.utils",
            "vllm_ascend.ops.activation",
            "vllm_ascend.ops.layernorm",
            "vllm_ascend.ops.fused_moe.fused_moe",
        ):
            sys.modules.pop(n, None)
