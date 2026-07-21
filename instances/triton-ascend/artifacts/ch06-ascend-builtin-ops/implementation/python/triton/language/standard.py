# 基座（非 ascend）standard 层的最小子集——flip 的 SIMT 回退分支用到的
# _is_power_of_two/xor_sum。本文件属于基座 Triton、未被昇腾 fork 改动。
#
# SOURCE: python/triton/language/standard.py（节选）

from . import core


# SOURCE: python/triton/language/standard.py:L19-21（全量保留，很小）
def _is_power_of_two(i: "core.constexpr"):
    n = i.value
    return core.constexpr((n & (n - 1)) == 0 and n != 0)


# SOURCE: python/triton/language/standard.py:L296-302
# SUBTRACTED: 真实 xor_sum 委托 core.reduce(input, axis, _xor_combine, ...)，其内部
# 经 semantic.reduction 建 MLIR combine region、由 _generator.call_JitFunction 把
# `_xor_combine`（一个 @triton.jit 组合子）内联进区域——这一整套是上游 Triton 通用
# 规约机制的编译期展开细节，既不在本章 code_spine 内，也不在 dossier must_keep 里
# （must_keep 只列到 flip/flip_impl/static_range——调用 xor_sum 这件事本身才是本章
# 要讲的，规约算子内部如何建 region 不是）。这里只保留 xor_sum 的公开签名与
# "只接受整型" 的类型校验，规约结果用形状正确的占位 tensor 表达：
# flip_impl 的 SIMT 分支（must_keep，逐字保留）能借此完整跑到底，验证的是"走了几步
# xor-swap、每步前后 reshape/bitcast 是否配对"这条控制流，而不是规约算子的数值语义
# （这本就需要真实 MLIR 才谈得上"数值"，interpreter 也不例外——ascend_interpreter.py
# 里同样没有单独还原 xor_sum 的 numpy 语义）。
def xor_sum(input, axis=None, keep_dims=False, _builder=None, _generator=None):  # SOURCE: python/triton/language/standard.py:L296-302(节选，见文件顶部注释)
    scalar_ty = input.type.scalar
    if not scalar_ty.is_int():
        raise ValueError("xor_sum only supported for integers")
    ax = core._constexpr_to_value(axis)
    shape = [core._constexpr_to_value(s) for s in input.shape]
    if keep_dims:
        shape[ax] = 1
    else:
        shape.pop(ax)
    handle = _builder.create_reduce(input.handle, ax)
    if shape:
        ret_ty = core.block_type(input.type.scalar, shape)
    else:
        ret_ty = input.type.scalar
    return core.tensor(handle, ret_ty)
