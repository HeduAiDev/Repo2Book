# SOURCE: vllm/v1/utils.py
# 只做减法的忠实精简版：本章消费面只有 ConstantList（Request 的只读视图
# output_token_ids/all_token_ids——append_output_token_ids 的双列表同步语义
# 依赖它防直改）。类本体逐字。
# SUBTRACTED: SPDX 版权头；文件其余（CpuGpuBuffer/install_frontend_uses 等）。
from typing import Generic, Sequence, TypeVar, overload

T = TypeVar("T")


# SOURCE: vllm/v1/utils.py:L48-L121 ConstantList —— 逐字
class ConstantList(Generic[T], Sequence):
    # SOURCE: vllm/v1/utils.py:L49（ConstantList.__init__）
    def __init__(self, x: list[T]) -> None:
        self._x = x

    # SOURCE: vllm/v1/utils.py:L52
    def append(self, item):
        raise TypeError("Cannot append to a constant list")

    # SOURCE: vllm/v1/utils.py:L55
    def extend(self, item):
        raise TypeError("Cannot extend a constant list")

    # SOURCE: vllm/v1/utils.py:L58
    def insert(self, item):
        raise TypeError("Cannot insert into a constant list")

    # SOURCE: vllm/v1/utils.py:L61
    def pop(self, item):
        raise TypeError("Cannot pop from a constant list")

    # SOURCE: vllm/v1/utils.py:L64
    def remove(self, item):
        raise TypeError("Cannot remove from a constant list")

    # SOURCE: vllm/v1/utils.py:L67
    def clear(self):
        raise TypeError("Cannot clear a constant list")

    # SOURCE: vllm/v1/utils.py:L70
    def index(self, item: T, start: int = 0, stop: int | None = None) -> int:
        return self._x.index(item, start, stop if stop is not None else len(self._x))

    @overload
    # SOURCE: vllm/v1/utils.py:L74（overload 桩）
    def __getitem__(self, item: int) -> T: ...

    @overload
    # SOURCE: vllm/v1/utils.py:L77（overload 桩）
    def __getitem__(self, s: slice, /) -> list[T]: ...

    # SOURCE: vllm/v1/utils.py:L79
    def __getitem__(self, item: int | slice) -> T | list[T]:
        return self._x[item]

    @overload
    # SOURCE: vllm/v1/utils.py:L83（overload 桩）
    def __setitem__(self, item: int, value: T): ...

    @overload
    # SOURCE: vllm/v1/utils.py:L86（overload 桩）
    def __setitem__(self, s: slice, value: T, /): ...

    # SOURCE: vllm/v1/utils.py:L88
    def __setitem__(self, item: int | slice, value: T | list[T]):
        raise TypeError("Cannot set item in a constant list")

    # SOURCE: vllm/v1/utils.py:L91
    def __delitem__(self, item):
        raise TypeError("Cannot delete item from a constant list")

    # SOURCE: vllm/v1/utils.py:L94
    def __iter__(self):
        return iter(self._x)

    # SOURCE: vllm/v1/utils.py:L97
    def __contains__(self, item):
        return item in self._x

    # SOURCE: vllm/v1/utils.py:L100
    def __len__(self):
        return len(self._x)

    # SOURCE: vllm/v1/utils.py:L103
    def __repr__(self) -> str:
        return f"ConstantList({self._x})"

    # SOURCE: vllm/v1/utils.py:L106
    def copy(self) -> list[T]:
        return self._x.copy()
