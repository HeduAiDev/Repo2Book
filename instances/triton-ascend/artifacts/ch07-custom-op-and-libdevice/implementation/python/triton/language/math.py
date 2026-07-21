# 支撑层——triton.language.math 的通用数学算子与"数据类型门禁"装饰器，属基座
# Triton、未被昇腾 fork 改动。libdevice.acos 的多项式分支调 math.abs；acos/isfinited/
# atan2/finitef 的函数签名上都挂着 math._check_dtype/_add_math_*arg_docstr——这两个
# 装饰器不是本章要讲的机制，但 _check_dtype 是真实会在调用期执行的 dtype 校验(不是
# 纯文档)，予以保留以复现"传错 dtype 会在 Python 层直接报错"这一真实可观察行为。
#
# SOURCE: python/triton/language/math.py(节选，见每个符号上的行号)
# SUBTRACTED: 真实文件还有 fdiv/div_rn/erf/floor/ceil/sin/cos/exp/log/... 约 40 个
# 数学自由函数，以及 _add_math_3arg_docstr。本章 libdevice 样例只用到 abs(acos 的
# |x| 分支)与 sqrt(acos mid 分支的 sqrt((1-|x|)/(1+|x|)))，故只留这两个。

from functools import wraps

import triton.language.core as tl
from . import semantic


def _check_dtype(dtypes):  # SOURCE: python/triton/language/math.py:L10-33
    """We're following libdevice's convention to check accepted data types for math functions."""

    def wrapper(fn):  # SOURCE: python/triton/language/math.py:L19-33(节选)

        @wraps(fn)
        def check(*args, **kwargs):  # SOURCE: python/triton/language/math.py:L22-31(节选)
            all_args = list(args) + list(kwargs.values())
            for arg in [a for a in all_args if isinstance(a, tl.tensor)]:
                arg_type = arg.type.scalar.name
                if arg_type not in dtypes:
                    raise ValueError(f"Expected dtype {dtypes} but got {arg_type}")
            return fn(*args, **kwargs)

        return check

    return wrapper


def _add_math_1arg_docstr(name):  # SOURCE: python/triton/language/math.py:L39-50
    def _decorator(func):
        func.__doc__ = f"Computes the element-wise {name} of :code:`x`."
        return func

    return _decorator


def _add_math_2arg_docstr(name):  # SOURCE: python/triton/language/math.py:L54-67
    def _decorator(func):
        func.__doc__ = f"Computes the element-wise {name} of :code:`x` and :code:`y`."
        return func

    return _decorator


# SUBTRACTED: fp8e4b15 掩码分支与整数(有符号/无符号)分支(python/triton/language/
# math.py:L188-195)——本章 acos 只在 fp16/fp32/bf16 上调用 abs，恒进浮点分支。
def abs(x, _builder=None):  # SOURCE: python/triton/language/math.py:L184-196(节选)
    x = semantic.to_tensor(x, _builder)
    return tl.tensor(_builder.create_fabs(x.handle), x.type)


def sqrt(x, _builder=None):  # SOURCE: python/triton/language/math.py:L158-160
    x = semantic.to_tensor(x, _builder)
    return tl.tensor(_builder.create_sqrt(x.handle), x.type)
