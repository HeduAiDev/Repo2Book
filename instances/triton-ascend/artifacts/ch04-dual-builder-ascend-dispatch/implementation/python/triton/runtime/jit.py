# SOURCE: python/triton/runtime/jit.py:L445 class JITFunction(KernelInterface[T])
# SUBTRACTED: @triton.jit 装饰器返回的完整可调用包装类——签名特化缓存(cache_key)、
# 递归 AST 解析(parse)、noinline/repr、调用协议等一整套（dossier 理论小节称其为
# visit_Call 的『第②岔』：JITFunction 组合子递归 codegen，与本章『第④岔——路由到
# ascend_builder』正交，属另一批章节的讲解范围）。这里只保留类型标识本身，供
# CodeGenerator.visit_Call 的 `isinstance(fn, JITFunction)` 分支判定使用；本章的
# 测试与叙事都只走 builtin 分支（第③/④岔），从不构造真正的 JITFunction 实例。
class JITFunction:
    # SOURCE: python/triton/runtime/jit.py:L445
    pass
