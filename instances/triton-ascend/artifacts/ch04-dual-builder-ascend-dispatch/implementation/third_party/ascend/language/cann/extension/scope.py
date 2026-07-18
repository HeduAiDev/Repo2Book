# SOURCE: third_party/ascend/language/cann/extension/scope.py（全文件，未删减——本身很小，
# __enter__/__exit__ 只做 kernel 内使用校验，作为 ASCEND_WITH_DISPATCH 的键、被
# visit_With 查表命中，本章按原样保留）。
__all__ = ["scope"]

from triton.language.core import _constexpr_to_value


class scope:
    """
    Context manager for entering and exiting a scope, where operations within a scope
    shares some common characteristics.

    Example:
    ```python
        import triton.language.extra.cann.extension as extension

        @triton.jit
        def kernel(x_ptr, y_ptr, N):
            with extension.scope(feature_a=True):
                a = tl.load(x_ptr)
                b = tl.load(y_ptr)
                result = tl.dot(a, b)
    ```

    Reserved keywords:
        - `core_mode`: Allows explicitly specify which core type should be used for
          operations within a code block.
    """
    # SOURCE: third_party/ascend/language/cann/extension/scope.py:L28-45

    def __init__(self, core_mode: str, _builder=None, _semantic=None, **kwargs):
        # SOURCE: third_party/ascend/language/cann/extension/scope.py:L49-63
        self.core_mode = _constexpr_to_value(core_mode) if _builder is None else core_mode
        self._builder = _builder
        self._semantic = _semantic

        if self.core_mode not in ("cube", "vector"):
            raise ValueError(f'core_mode must be "cube" or "vector", got {self.core_mode}')

    def __enter__(self):
        # SOURCE: third_party/ascend/language/cann/extension/scope.py:L65-68
        if self._builder is None:
            raise RuntimeError("scope can only be used inside a Triton kernel")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # SOURCE: third_party/ascend/language/cann/extension/scope.py:L70-71
        return False
