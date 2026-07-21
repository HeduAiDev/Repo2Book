"""m8 —— cann/__init__.py 在 import 期动态覆盖挂载 libdevice 命名空间。

对照真实源码 third_party/ascend/language/cann/__init__.py:L27-52：libdevice 的一部分
符号直接复用基座 triton math(umulhi/exp/log/.../sqrt/abs)，另一部分被昇腾专属实现
覆盖(isfinited/finitef/恒定覆盖；atan2 只在 `not triton_enable_libdevice_simt()`
为真时才覆盖)。用 import 期属性赋值而非重复定义，既复用基座又插入昇腾差异。
"""


def test_isfinited_and_finitef_are_always_overridden_from_math_ops(env):
    assert env.cann.libdevice.isfinited is env.cann.extension.math_ops.isfinited
    assert env.cann.libdevice.finitef is env.cann.extension.math_ops.finitef


def test_base_math_functions_are_reused_verbatim(env):
    """libdevice.sqrt/abs 直接就是基座 triton.language.math 的同一个对象——
    不是重新定义，是"引用同一个函数"。"""
    assert env.cann.libdevice.sqrt is env.tl_math.sqrt
    assert env.cann.libdevice.abs is env.tl_math.abs


def test_atan2_is_overridden_when_simt_disabled(env):
    """env fixture 默认 triton_enable_libdevice_simt() 恒 False -> `not ...()` 为真
    -> 覆盖分支执行。"""
    assert env.cann.libdevice.atan2 is env.cann.extension.math_ops.atan2


def test_atan2_override_is_skipped_when_simt_enabled(env_simt_enabled):
    """triton_enable_libdevice_simt() 恒 True -> `not ...()` 为假 -> 覆盖分支被跳过，
    libdevice.atan2 保持 import 前的原样(本章精简版里，libdevice.py 本身没有保留
    atan2 这个 extern 实现——它在 subtraction_plan 里与其余 ~30 个数学函数一起被删，
    真实全量仓库里这里会是 libdevice 自己的 extern 版 atan2，而不是 math_ops 版；
    本测试只验证"覆盖分支被跳过"这件事本身，不断言跳过后剩下的是谁)。"""
    env = env_simt_enabled
    assert not hasattr(env.cann.libdevice, "atan2")
