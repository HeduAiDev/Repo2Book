# SOURCE: vllm/compilation/decorators.py
# ch23 切面（m11 挂钩位之一）：@support_torch_compile 装饰器——模型类声明
# 「我支持 piecewise 编译」的标记（llama.py:L334-L343 消费面）。
# SUBTRACTED：装饰器的编译包装本体（_support_torch_compile 的 forward 重写/
#   dynamic_arg_dims 标记/计数器登记，decorators.py:L150 起）——ch19 编译域；
#   精简版保签名与标记语义（原类直返）。
from __future__ import annotations

from typing import Callable, Optional, TypeVar, Union

_T = TypeVar("_T")


# SOURCE: vllm/compilation/decorators.py:L118-L126 support_torch_compile
#   减法子集（签名逐字；包装本体 → ch19）
def support_torch_compile(
    cls: Optional[type[_T]] = None,
    *,
    dynamic_arg_dims: Optional[dict[str, Union[int, list[int], dict[int, str]]]] = None,
    mark_unbacked_dims: Optional[dict[str, Union[int, list[int]]]] = None,
    enable_if: Optional[Callable] = None,
    is_encoder: bool = False,
) -> Union[Callable[[type[_T]], type[_T]], type[_T]]:
    """
    A decorator to add support for compiling the forward method of a class.

    Usage 1: use directly as a decorator without arguments:

    ```python
    @support_torch_compile
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]): ...
    ```

    Usage 2: use as a decorator with arguments:

    ```python
    @support_torch_compile(dynamic_arg_dims={"x": 0, "y": 0})
    class MyModel(nn.Module):
        def forward(self, x: torch.Tensor, y: Optional[torch.Tensor]): ...
    ```

    `dynamic_arg_dims` is a dictionary that maps argument names to the dynamic
    dimensions of the argument.
    """
    # SUBTRACTED: _support_torch_compile 包装本体（forward 重写与编译计数）
    #   ——ch19 编译域；精简版原类直返（标记语义保留：装饰即声明可编译）

    # SOURCE: vllm/compilation/decorators.py:L118-L126 support_torch_compile
    def decorator(_cls: type[_T]) -> type[_T]:
        # SOURCE: vllm/compilation/decorators.py:L118-L126 support_torch_compile
        return _cls

    if cls is not None:
        return decorator(cls)
    return decorator
