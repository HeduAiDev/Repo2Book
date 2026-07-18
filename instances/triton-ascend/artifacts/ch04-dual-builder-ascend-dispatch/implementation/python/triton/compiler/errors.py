class CompilationError(Exception):
    # SOURCE: python/triton/compiler/errors.py
    # SUBTRACTED: 真实实现还有 _format_message()（把 AST 节点位置格式化成
    # "at line:col: <source excerpt>" 的错误提示）与 __reduce__（pickle 支持）；本章
    # visit_Call 只在 except 分支 `raise CompilationError(...) from e` 里把它当异常类型
    # 用，不依赖其消息格式化细节，故这里只保留能被实例化/抛出/捕获的最小异常语义。
    def __init__(self, src, node, error_message=None):
        # SOURCE: python/triton/compiler/errors.py
        self.src = src
        self.node = node
        self.error_message = error_message
        super().__init__(error_message or "")


# SOURCE: python/triton/compiler/errors.py（CompileTimeAssertionFailure(CompilationError)）
class CompileTimeAssertionFailure(CompilationError):
    pass


# SOURCE: python/triton/compiler/errors.py（UnsupportedLanguageConstruct(CompilationError)）
class UnsupportedLanguageConstruct(CompilationError):
    pass
