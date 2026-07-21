# SOURCE: third_party/ascend/language/cann/extension/scope.py
# SUBTRACTED: 文件头 Huawei/OpenAI license 块(原 L1-L21)——纯法务声明，不参与任何控制流。
__all__ = ["scope"]

from triton.language.core import _constexpr_to_value


class scope:  # SOURCE: third_party/ascend/language/cann/extension/scope.py:L28-71
    """
    Context manager for entering and exiting a scope, where operations within a scope shares some common characteristics.

    Example:
    ```python
        import triton.language.extra.cann.extension as extension

        @triton.jit
        def kernel(x_ptr, y_ptr, N):
            # specify annotation
            with extension.scope(feature_a=True):
                a = tl.load(x_ptr)
                b = tl.load(y_ptr)
                result = tl.dot(a, b)
    ```

    Reserved keywords:
        - `core_mode`: Allows explicitly specify which core type should be used for operations within a code block, helping the compiler generate appropriate code for cube or vector cores.
    """

    def __init__(self, core_mode: str, _builder=None, _semantic=None, **kwargs):  # SOURCE: third_party/ascend/language/cann/extension/scope.py:L49-63
        """
        :param core_mode: Either "cube" or "vector" to specify the core type
        :param _builder: Internal builder object (set by code_generator)
        :param _semantic: Internal semantic object (set by code_generator)
        :param kwargs: Additional internal parameters
        """
        # Convert constexpr to value if not being called from code generator
        self.core_mode = _constexpr_to_value(core_mode) if _builder is None else core_mode
        self._builder = _builder
        self._semantic = _semantic

        # Validate core_mode
        if self.core_mode not in ("cube", "vector"):
            raise ValueError(f'core_mode must be "cube" or "vector", got {self.core_mode}')

    def __enter__(self):  # SOURCE: third_party/ascend/language/cann/extension/scope.py:L65-68
        if self._builder is None:
            raise RuntimeError("scope can only be used inside a Triton kernel")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # SOURCE: third_party/ascend/language/cann/extension/scope.py:L70-71
        return False
