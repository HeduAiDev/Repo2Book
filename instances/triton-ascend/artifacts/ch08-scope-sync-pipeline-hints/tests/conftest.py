"""ch08 测试脚手架。

本章的机制(dossier.mechanisms m1-m18)落在 third_party/ascend/language/cann/
extension/{scope,dispatch,code_generator,core,semantic,aux_ops,_utils}.py 与
python/triton/compiler/code_generator.py 几个纯 Python 文件里，但它们真实调用的
`ir.builder`/`ascendnpu_ir_builder`(create_scope_op/scope_return/sync_block_set/
create_custom_op_for_inter_core_sync/create_annotation_mark/get_t_core_type_*_attr/...)
是编译期生成的 C++ 绑定——本仓构建产物只在有昇腾 NPU/CANN 工具链的机器上存在(见
INSTANCE.md「运行验证需昇腾 NPU/CANN 工具链，宿主无此环境」)，host 既没有真绑定，也没
有等价的可信 Python 替代。

所以这里在 sys.modules 里搭桩，做法与 ch04/ch05/ch07 conftest.py 一致：
  - 真正是 C++ 绑定、host 无法拥有的名字(triton._C.libtriton[.ascend])——换成本文件的
    FakeBuilder 测试替身，只提供本章代码路径 duck-type 用到的方法，全部"记录调用 + 返回
    可预测的哨兵值"，不模拟任何 MLIR/硬件语义——本章要看的是『调用被路由到哪个符号/建了
    哪个属性』，不是『IR dump 长什么样』。ascend_ir.CoreType/PIPE/MODE 三个 pybind 枚举
    也在这里搭成等价的 Python 占位对象(名字/档数与 ascend_ir.cc 的 py::enum_ 导出一致：
    CoreType 4 档、PIPE 8 档、MODE 3 档)。
  - 真正是本仓 Python 源码的名字(triton.language.core、triton.compiler.code_generator、
    third_party/.../extension/*)——按**规范模块名**把 implementation/ 下(已减法)的同名
    文件加载进 sys.modules，不是另造桩。
  - `triton.language.semantic.to_tensor`(base 基座、未被 fork 改动的真实函数)——它在真实
    仓库里还要经过完整 dtype 系统与 `full()`张量构造，这些与本章 dossier(scope 的编译器
    特判/sync_block 两代协议/PIPE 口径收窄/compile_hint)完全正交、无出处支持在本章重现，
    故按"外部依赖"处理，用一个只保留『value -> 带 .handle 的容器』这一可观察行为的测试
    替身代替，不在 implementation/ 下重造 dtype/full()。
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
# ascend_ir 占位枚举 —— 站在 triton._C.libtriton.ascend.ir 的 CoreType/PIPE/MODE
# pybind 导出位置上。名字与档数据实(ascend_ir.cc:L420-436)：CoreType 4 档、PIPE 8 档
# (真实 .td 定义 15 档，pybind 只导出这 8 档——本章 M12 口径收窄机制)、MODE 3 档。
# ---------------------------------------------------------------------------
class _EnumVal:
    def __init__(self, group, name):
        self.group = group
        self.name = name

    def __repr__(self):
        return f"{self.group}.{self.name}"


def _make_enum_group(group_name, names):
    ns = types.SimpleNamespace()
    for n in names:
        setattr(ns, n, _EnumVal(group_name, n))
    return ns


CoreType = _make_enum_group("CoreType", ["VECTOR", "CUBE", "CUBE_OR_VECTOR", "CUBE_AND_VECTOR"])
PIPE_ENUM = _make_enum_group(
    "PIPE", ["PIPE_S", "PIPE_V", "PIPE_M", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_ALL", "PIPE_FIX"])
MODE_ENUM = _make_enum_group("MODE", ["SIMD", "SIMT", "MIX"])


# ---------------------------------------------------------------------------
# FakeBuilder —— 站在真实 ir.builder / ascendnpu_ir_builder 位置上的测试替身。
# ---------------------------------------------------------------------------
class FakeBlock:
    def __init__(self, region=None):
        self.region = region
        self.erased = False

    def erase(self):
        self.erased = True


class FakeRegion:
    pass


class FakeOp:
    """create_scope_op 的返回值替身：记录属性/结果类型，get_region/get_result 可用。"""

    def __init__(self, attrs, result_types):
        self.attrs = attrs
        self.result_types = result_types
        self.region = FakeRegion()

    def get_region(self, i):
        assert i == 0
        return self.region

    def get_result(self, i):
        return f"scope-result-{i}"


class FakeBuilder:
    def __init__(self):
        self.calls = []
        self._loc = None
        self._ip = "ip-0"
        self._ip_seq = 0
        self._simt_mode = False

    # ---- 插入点/位置：enter_sub_region 与 handle_scope_with 的两趟 visit 都要用 ----
    def get_loc(self):
        return self._loc

    def set_loc(self, *args):
        self._loc = args

    def get_insertion_point(self):
        return self._ip

    def restore_insertion_point(self, ip):
        self._ip = ip

    def get_insertion_block(self):
        return "main-block"

    def set_insertion_point_to_start(self, block):
        self._ip_seq += 1
        self._ip = f"ip-{self._ip_seq}"

    def set_insertion_point_to_end(self, block):
        self._ip_seq += 1
        self._ip = f"ip-{self._ip_seq}"

    def create_block(self):
        self.calls.append(("create_block",))
        return FakeBlock()

    def create_block_with_parent(self, region, args):
        self.calls.append(("create_block_with_parent", region, args))
        return FakeBlock(region)

    # ---- scope.scope 的建造/终结 ----
    def create_scope_op(self, attrs, result_types):
        self.calls.append(("create_scope_op", dict(attrs), list(result_types)))
        return FakeOp(attrs, result_types)

    def scope_return(self, values):
        self.calls.append(("scope_return", list(values)))

    def get_t_core_type_attr_name(self):
        return "t_core_type"

    def get_t_core_type_cube_attr(self):
        return "cube-attr"

    def get_t_core_type_vector_attr(self):
        return "vector-attr"

    # ---- Python 值 -> MLIR 属性(_py_value_to_mlir_attr / compile_hint_impl 共用) ----
    def get_unit_attr(self):
        return "unit-attr"

    def get_bool_attr(self, v):
        return ("bool-attr", v)

    def get_str_attr(self, v):
        return ("str-attr", v)

    def get_int32_attr(self, v):
        return ("int32-attr", v)

    def get_i64_array_attr(self, v):
        return ("i64-array-attr", tuple(v))

    # ---- 旧代核间同步：落到通用 CustomOp ----
    def create_custom_op_for_inter_core_sync(self, op_name, mode_or_sender, event_id):
        self.calls.append(("create_custom_op_for_inter_core_sync", op_name, mode_or_sender, event_id))
        return f"customop-{op_name}"

    # ---- 新代核间同步：落到 hivm.sync_block_set/wait/all(记录调用，不模拟 hivm 语义) ----
    def sync_block_set(self, sender, receiver, event_id_handle, sender_pipe_value, receiver_pipe_value):
        self.calls.append(("sync_block_set", sender, receiver, event_id_handle, sender_pipe_value, receiver_pipe_value))

    def sync_block_wait(self, sender, receiver, event_id_handle, sender_pipe_value, receiver_pipe_value):
        self.calls.append(("sync_block_wait", sender, receiver, event_id_handle, sender_pipe_value, receiver_pipe_value))

    def sync_block_all(self, mode, event_id):
        self.calls.append(("sync_block_all", mode, event_id))

    # ---- compile_hint 落地 ----
    def is_simt_mode(self):
        return self._simt_mode

    def create_annotation_mark(self, ptr_handle, hint_name, hint_val):
        self.calls.append(("create_annotation_mark", ptr_handle, hint_name, hint_val))


# ---------------------------------------------------------------------------
# 环境装配
# ---------------------------------------------------------------------------
def _build_env():
    stubs = _Stubs()

    # ---- triton._C.libtriton[.ascend]：C++ 绑定的桩 ----
    stubs.mod("triton")
    stubs.mod("triton._C")
    libtriton = stubs.mod("triton._C.libtriton")
    libtriton.ir = types.SimpleNamespace(builder=FakeBuilder)
    ascend_pkg = stubs.mod("triton._C.libtriton.ascend")
    ascend_pkg.ir = types.SimpleNamespace(
        ascendnpu_ir_builder=FakeBuilder,
        CoreType=CoreType,
        PIPE=PIPE_ENUM,
        MODE=MODE_ENUM,
    )

    # ---- triton.language：基座值系统(精简版) ----
    tl_pkg = stubs.mod("triton.language")
    core = _load("python/triton/language/core.py", "triton.language.core")
    tl_pkg.core = core

    # ---- triton.language.semantic(base，未被 fork 改动)：只桩 to_tensor，理由见文件头 ----
    real_semantic = stubs.mod("triton.language.semantic")

    def _to_tensor(x, builder, check_type=True):
        from triton.language.core import constexpr, tensor
        if isinstance(x, constexpr):
            x = x.value
        if isinstance(x, tensor):
            return x
        return tensor(x, "scalar-ty")

    real_semantic.to_tensor = _to_tensor

    # ---- triton.language.extra.cann.extension：按依赖顺序加载本章精简版 ----
    stubs.mod("triton.language.extra")
    stubs.mod("triton.language.extra.cann")
    stubs.mod("triton.language.extra.cann.extension")

    ext_semantic = _load("third_party/ascend/language/cann/extension/semantic.py",
                          "triton.language.extra.cann.extension.semantic")
    ext_core = _load("third_party/ascend/language/cann/extension/core.py",
                      "triton.language.extra.cann.extension.core")
    ext_scope = _load("third_party/ascend/language/cann/extension/scope.py",
                       "triton.language.extra.cann.extension.scope")
    ext_codegen = _load("third_party/ascend/language/cann/extension/code_generator.py",
                         "triton.language.extra.cann.extension.code_generator")
    ext_dispatch = _load("third_party/ascend/language/cann/extension/dispatch.py",
                          "triton.language.extra.cann.extension.dispatch")
    ext_utils = _load("third_party/ascend/language/cann/extension/_utils.py",
                       "triton.language.extra.cann.extension._utils")
    ext_aux_ops = _load("third_party/ascend/language/cann/extension/aux_ops.py",
                         "triton.language.extra.cann.extension.aux_ops")

    # ---- triton.compiler.code_generator：基座 with-分派缝(本章精简版) ----
    stubs.mod("triton.compiler")
    code_generator = _load("python/triton/compiler/code_generator.py",
                            "triton.compiler.code_generator")

    mods = types.SimpleNamespace(
        core=core,
        real_semantic=real_semantic,
        ext_semantic=ext_semantic,
        ext_core=ext_core,
        ext_scope=ext_scope,
        ext_codegen=ext_codegen,
        ext_dispatch=ext_dispatch,
        ext_utils=ext_utils,
        ext_aux_ops=ext_aux_ops,
        code_generator=code_generator,
        FakeBuilder=FakeBuilder,
    )

    def _cleanup():
        stubs.cleanup()
        for n in (
            "triton.language.core",
            "triton.language.semantic",
            "triton.language.extra.cann.extension.semantic",
            "triton.language.extra.cann.extension.core",
            "triton.language.extra.cann.extension.scope",
            "triton.language.extra.cann.extension.code_generator",
            "triton.language.extra.cann.extension.dispatch",
            "triton.language.extra.cann.extension._utils",
            "triton.language.extra.cann.extension.aux_ops",
            "triton.compiler.code_generator",
        ):
            sys.modules.pop(n, None)

    return mods, _cleanup


@pytest.fixture
def env():
    mods, cleanup = _build_env()
    try:
        yield mods
    finally:
        cleanup()


def make_generator(mods):
    """按精简版 CodeGenerator.__init__(builder) 构造一个实例。"""
    CodeGenerator = mods.code_generator.CodeGenerator
    return CodeGenerator(builder=mods.FakeBuilder())
