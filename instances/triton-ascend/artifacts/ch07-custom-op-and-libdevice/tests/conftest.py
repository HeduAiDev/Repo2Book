"""ch07 测试脚手架。

本章的机制(dossier.mechanisms m1-m8)落在 third_party/ascend/language/cann/ 的几个
纯 Python 文件(custom_op.py/builtin_custom_ops.py/libdevice.py/math_ops.py/两个
__init__.py)里，但它们真实调用的 `ir.builder`(create_custom_op/
create_extern_elementwise/create_fadd/...)是编译期生成的 C++ 绑定——本仓构建产物只
在有昇腾 NPU/CANN 工具链的机器上存在(见 INSTANCE.md「运行验证需昇腾 NPU/CANN
工具链，宿主无此环境」)，host 既没有真绑定，也没有等价的可信 Python 替代(pip 装的
官方 triton 是不同版本/不同 fork，用它对照会静默引入版本漂移)。

所以这里在 sys.modules 里搭桩，做法与 ch04/ch05 conftest.py 一致：
  - 真正是 C++ 绑定、host 无法拥有的名字(triton._C.libtriton)——换成本文件的
    FakeBuilder 测试替身。标量算术类方法(create_fadd/create_fsub/create_fmul/
    create_fdiv/create_fcmpOLT/create_select/create_fabs/create_sqrt/get_fp32/
    get_int1/get_int32)做成"真的算浮点数"(handle 直接就是 Python float/bool)——
    这是本章唯一"数值可验证"的部分(acos 的纯 IR 多项式逼近本身是可在 CPU 上复现的
    数学，不依赖昇腾硬件)，让测试能验证精简版复现的 acos 与 math.acos 数值相符，
    而不只是"调用顺序对不对"。custom_op/属性类方法(create_custom_op/
    create_extern_elementwise/get_core_type_attr/get_pipe_attr/get_vf_mode_attr/
    get_str_attr/get_int_attr/...)做成"记录调用 + 返回可预测的哨兵值"——这些是真正
    只有编译期 C++/MLIR 才能回答"IR 属性长什么样"的地方，测试要看的是"调用被路由到
    哪个符号/建了哪个属性"，不模拟 MLIR 语义。
  - 真正是本仓 Python 源码的名字(triton.language.{core,semantic,math}、
    triton.runtime.jit、third_party/.../extension/*、libdevice.py、两个
    __init__.py)——按**规范模块名**把 implementation/ 下(已减法)的同名文件加载进
    sys.modules，不是另造桩。
  - 真正是"与本章机制无关的其它后端/工具模块"(triton.backends.ascend.utils.
    triton_enable_libdevice_simt、triton.tools.get_ascend_devices.
    is_compile_on_910_95)——按最小桩提供，可在测试里按需 monkeypatch 切换分支。

加载顺序：extension/math_ops.py 用 `from ..libdevice import atan, isnan, isinf`
(两个点，越过 extension 包直接找 cann 包下的 libdevice)，所以 libdevice.py 必须先于
extension 下任何子模块加载并登记进 sys.modules['triton.language.extra.cann.libdevice']。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _load(rel_path, modname):
    """按规范模块名加载精简版文件并登记进 sys.modules(含父包属性挂接)。"""
    path = IMPL_DIR / rel_path
    is_pkg = path.name == "__init__.py"
    spec = importlib.util.spec_from_file_location(
        modname, path, submodule_search_locations=[str(path.parent)] if is_pkg else None)
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


# ---------------------------------------------------------------------------
# FakeBuilder —— 站在真实 ir.builder 位置上的测试替身。
# ---------------------------------------------------------------------------
class ExternCall:
    """create_extern_elementwise 的返回哨兵：记录被选中的 __hmf_ 符号名与入参 handle。"""

    def __init__(self, symbol, handles):
        self.symbol = symbol
        self.handles = tuple(handles)

    def __repr__(self):
        return f"ExternCall({self.symbol!r}, {self.handles!r})"


class FakeBuilder:
    def __init__(self):
        self.calls = []

    # ---- 标量装箱：直接用 Python 值本身当 handle ----
    def get_int1(self, v):
        return bool(v)

    def get_int32(self, v):
        return int(v)

    def get_fp32(self, v):
        return float(v)

    # ---- 真·浮点算术(handle 即 Python float/bool)——供 semantic.py/math.py 的
    # 精简版调用，让 acos 的多项式逼近可在 CPU 上端到端算出真实数值 ----
    def create_fadd(self, a, b):
        return a + b

    def create_fsub(self, a, b):
        return a - b

    def create_fmul(self, a, b):
        return a * b

    def create_fdiv(self, a, b):
        return a / b

    def create_fcmpOLT(self, a, b):
        return a < b

    def create_select(self, cond, a, b):
        return a if cond else b

    def create_fabs(self, a):
        return abs(a)

    def create_sqrt(self, a):
        import math as _m
        return _m.sqrt(a)

    # ---- extern 调度：不模拟 __hmf_ 符号的真实计算(需真机)，只记录"选中了哪个符号" ----
    def create_extern_elementwise(self, symbol, handles):
        call = ExternCall(symbol, handles)
        self.calls.append(("create_extern_elementwise", call))
        return call

    # ---- 自定义算子 IR 属性：只记录调用，返回可预测的哨兵值 ----
    def get_core_type_attr(self, v):
        return ("core_type_attr", v)

    def get_pipe_attr(self, v):
        return ("pipe_attr", v)

    def get_vf_mode_attr(self, v):
        return ("vf_mode_attr", v)

    def get_str_attr(self, v):
        return ("str_attr", v)

    def get_int_attr(self, v):
        return ("int_attr", v)

    def get_affine_map_array_attr(self, v):
        return ("affine_map_array_attr", tuple(v))

    def get_iterator_types_attr(self, v):
        return ("iterator_types_attr", tuple(v))

    def get_type_array_attr(self, v):
        return ("type_array_attr", tuple(v))

    def get_i64_array_attr(self, v):
        return ("i64_array_attr", tuple(v))

    def create_custom_op(self, name, attrs, inputs, outputs, arg_attrs):
        self.calls.append(("create_custom_op", name, dict(attrs), list(inputs), list(outputs), list(arg_attrs)))
        return [f"{name}-result#{i}" for i in range(len(outputs))]


# ---------------------------------------------------------------------------
# 环境装配：按依赖顺序把桩与精简版登记进 sys.modules。
#
# `simt_enabled` 控制 cann/__init__.py 里 `if not triton_enable_libdevice_simt():
# libdevice.atan2 = extension.math_ops.atan2` 这条 m8 分支——这个标志在 import 期
# 就被读取一次并决定是否做覆盖，之后不可变，故要测两条分支需要两次完整的环境装配
# (两个 fixture 共用这份 builder 逻辑)，而不是装好一次后 monkeypatch。
# ---------------------------------------------------------------------------
def _build_env(simt_enabled: bool):
    stubs = _Stubs()

    # ---- triton._C.libtriton：C++ 绑定的桩，acos 用它做 `_builder: ir.builder`
    # 类型注解(本文件顶部无 `from __future__ import annotations`，注解会在 def 时
    # 求值，故 ir.builder 必须是个存在的名字) ----
    stubs.mod("triton")
    stubs.mod("triton._C")
    libtriton = stubs.mod("triton._C.libtriton")
    libtriton.ir = types.SimpleNamespace(builder=FakeBuilder)

    # ---- triton.language.{core,semantic,math}：基座 triton 值系统 + 标量算子的
    # 精简真实子集(non-ascend，未被 fork 改动，见各文件头注释) ----
    stubs.mod("triton.language")
    tl_core = _load("python/triton/language/core.py", "triton.language.core")
    tl_semantic = _load("python/triton/language/semantic.py", "triton.language.semantic")
    tl_math = _load("python/triton/language/math.py", "triton.language.math")

    # ---- triton.runtime.jit：JITFunction 只保留 `.fn` 这一件事(见文件头注释) ----
    stubs.mod("triton.runtime")
    tl_jit = _load("python/triton/runtime/jit.py", "triton.runtime.jit")
    sys.modules["triton.runtime"].JITFunction = tl_jit.JITFunction

    # ---- 与本章机制无关的后端/工具模块：最小桩，可在单测里 monkeypatch 切换分支 ----
    stubs.mod("triton.backends")
    stubs.mod("triton.backends.ascend")
    ascend_utils = stubs.mod("triton.backends.ascend.utils")
    ascend_utils.get_ascend_arch_from_env = lambda: "Ascend910B"
    ascend_utils.triton_enable_libdevice_simt = lambda: simt_enabled

    stubs.mod("triton.tools")
    ascend_devices = stubs.mod("triton.tools.get_ascend_devices")
    ascend_devices.is_compile_on_910_95 = False

    # ---- triton.language.extra.cann：先注册父包占位(math_ops.py 的两点相对导入
    # `from ..libdevice import ...` 需要它)，libdevice.py 必须先于 extension/* 加载 ----
    stubs.mod("triton.language.extra")
    cann_pkg = stubs.mod("triton.language.extra.cann")

    libdevice = _load("third_party/ascend/language/cann/libdevice.py",
                       "triton.language.extra.cann.libdevice")

    stubs.mod("triton.language.extra.cann.extension")
    ext_core = _load("third_party/ascend/language/cann/extension/core.py",
                      "triton.language.extra.cann.extension.core")
    ext_utils = _load("third_party/ascend/language/cann/extension/_utils.py",
                       "triton.language.extra.cann.extension._utils")
    ext_custom_op = _load("third_party/ascend/language/cann/extension/custom_op.py",
                           "triton.language.extra.cann.extension.custom_op")
    ext_builtin_custom_ops = _load(
        "third_party/ascend/language/cann/extension/builtin_custom_ops.py",
        "triton.language.extra.cann.extension.builtin_custom_ops")
    ext_math_ops = _load("third_party/ascend/language/cann/extension/math_ops.py",
                          "triton.language.extra.cann.extension.math_ops")

    extension = _load("third_party/ascend/language/cann/extension/__init__.py",
                       "triton.language.extra.cann.extension")

    cann = _load("third_party/ascend/language/cann/__init__.py",
                  "triton.language.extra.cann")

    mods = types.SimpleNamespace(
        tl_core=tl_core,
        tl_semantic=tl_semantic,
        tl_math=tl_math,
        tl_jit=tl_jit,
        ascend_utils=ascend_utils,
        ascend_devices=ascend_devices,
        libdevice=libdevice,
        ext_core=ext_core,
        ext_utils=ext_utils,
        custom_op=ext_custom_op,
        builtin_custom_ops=ext_builtin_custom_ops,
        math_ops=ext_math_ops,
        extension=extension,
        cann=cann,
        FakeBuilder=FakeBuilder,
        ExternCall=ExternCall,
    )

    def _cleanup():
        stubs.cleanup()
        for n in (
            "triton.language.core",
            "triton.language.semantic",
            "triton.language.math",
            "triton.runtime.jit",
            "triton.backends.ascend.utils",
            "triton.tools.get_ascend_devices",
            "triton.language.extra.cann.libdevice",
            "triton.language.extra.cann.extension.core",
            "triton.language.extra.cann.extension._utils",
            "triton.language.extra.cann.extension.custom_op",
            "triton.language.extra.cann.extension.builtin_custom_ops",
            "triton.language.extra.cann.extension.math_ops",
            "triton.language.extra.cann.extension",
            "triton.language.extra.cann",
        ):
            sys.modules.pop(n, None)

    return mods, _cleanup


@pytest.fixture
def env():
    """默认环境：triton_enable_libdevice_simt() 恒 False(与真实默认配置一致)。"""
    mods, cleanup = _build_env(simt_enabled=False)
    try:
        yield mods
    finally:
        cleanup()


@pytest.fixture
def env_simt_enabled():
    """m8 的另一条分支：triton_enable_libdevice_simt() 恒 True——cann/__init__.py
    不会把 libdevice.atan2 覆盖成 extension.math_ops.atan2(见 test_cann_package_overrides.py)。"""
    mods, cleanup = _build_env(simt_enabled=True)
    try:
        yield mods
    finally:
        cleanup()
