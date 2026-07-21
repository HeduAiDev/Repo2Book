# 支撑层——本章机制(register_custom_op/libdevice 三分野)的间接依赖，非本章主角。
#
# SOURCE: python/triton/runtime/jit.py:L654-659(节选 JITFunction.__init__)、L822-844(节选 jit())
# SUBTRACTED: 真实 JITFunction 是 @triton.jit 装饰器返回的完整可调用包装类——签名特化
# 缓存(cache_key)、递归 AST 解析(parse)、noinline/repr、完整调用协议(__call__ 触发
# 编译)等，属"JITFunction 组合子递归 codegen"(dossier 理论所称另一岔)，与本章
# custom_op/libdevice 的语言层扩展主线无关。math_ops.py 的 isfinited/atan2/finitef
# 只用 @jit 把函数包成"看起来像内建"的对象，本章测试直接经 `.fn` 取原始 Python 函数
# 验证真实控制流(不触发编译)，故这里只保留 JITFunction.__init__ 真正做的那一件事：
# `self.fn = fn`。
class JITFunction:
    def __init__(self, fn):  # SOURCE: python/triton/runtime/jit.py:L659
        self.fn = fn


def jit(fn):  # SOURCE: python/triton/runtime/jit.py:L822(节选，去掉可选参数重载)
    return JITFunction(fn)
