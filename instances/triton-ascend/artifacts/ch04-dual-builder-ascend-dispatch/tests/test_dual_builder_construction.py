"""m1 —— 双 builder 构造:CodeGenerator.__init__ 在同一个实例上并挂
self.builder（标准 Triton IR）与 self.ascend_builder（ascendnpu_ir_builder），
并把 setup_unified_builder(self.builder, self.ascend_builder) 接上。

对照真实源码 python/triton/compiler/code_generator.py:L215-231：
    self.builder = ir.builder(context, compile_mode="simd" 或 "simt")
    self.ascend_builder = ascend_ir.ascendnpu_ir_builder(context, arch)
    setup_unified_builder(self.builder, self.ascend_builder)
"""
import types

from conftest import make_generator, FakeAscendBuilder, FakeBuilder


def test_constructs_both_builders_on_same_context(env):
    gen = make_generator(env)

    assert isinstance(gen.builder, FakeBuilder)
    assert isinstance(gen.ascend_builder, FakeAscendBuilder)
    # 两个 builder 共享同一 MLIR context（dossier design_decisions 第 3 条）。
    assert gen.builder.context == gen.ascend_builder.context == "ctx"
    # 两者不是同一个对象——的确是「第二个」builder，不是复用同一个。
    assert gen.builder is not gen.ascend_builder


def test_compile_mode_follows_force_simt_only_option(env):
    # 基座 NPU 后端特有的分岔：options.force_simt_only 决定主 builder 的 compile_mode。
    gen_simd = make_generator(env, options=types.SimpleNamespace(force_simt_only=False))
    assert gen_simd.builder.compile_mode == "simd"

    gen_simt = make_generator(env, options=types.SimpleNamespace(force_simt_only=True))
    assert gen_simt.builder.compile_mode == "simt"

    # options 没有 force_simt_only 属性时（基座默认，非 NPU 后端）走 simd 分支。
    gen_default = make_generator(env, options=types.SimpleNamespace())
    assert gen_default.builder.compile_mode == "simd"


def test_ascend_builder_gets_arch_from_options(env):
    gen = make_generator(env, options=types.SimpleNamespace(arch="ascend910b"))
    assert gen.ascend_builder.arch == "ascend910b"

    # options 没有 arch 属性时用空字符串兜底（getattr(options, "arch", "")）。
    gen_default = make_generator(env, options=types.SimpleNamespace())
    assert gen_default.ascend_builder.arch == ""


def test_setup_unified_builder_is_invoked_during_init(env):
    gen = make_generator(env)
    # setup_unified_builder(main, ascend) 把 ascend_builder 反挂为 main._ascend_builder。
    assert gen.builder._ascend_builder is gen.ascend_builder
    # 且把 create_scope_op 等方法作为 wrapper 挂到了主 builder 上（m4，见另一测试文件
    # 里更细的验证）——这里只确认「__init__ 确实触发了这一步」。
    assert hasattr(gen.builder, "create_scope_op")
    assert hasattr(gen.builder, "scope_return")
