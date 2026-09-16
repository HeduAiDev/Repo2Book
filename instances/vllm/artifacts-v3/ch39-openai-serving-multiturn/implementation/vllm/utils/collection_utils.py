# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/utils/collection_utils.py —— HOST SEAM（最小承载）：本章
# 消费面只有 as_list（流式生成器 token_ids 直通）与 is_list_of（两个
# parser 注册表的注册名校验）；FactoryDict/common_prefix 等归各自域。
from collections.abc import Iterable
from typing import Literal, TypeVar, assert_never

# SOURCE: vllm/utils/collection_utils.py:L6 —— TypeIs 位（HOST SEAM：真实
# 自 typing 导入（Python 3.13+）；本 host 为 3.11，从 typing_extensions 取
# 同一对象——语义一致）
from typing_extensions import TypeIs

T = TypeVar("T")


# SOURCE: vllm/utils/collection_utils.py:L49-L51 —— as_list 逐字
def as_list(maybe_list: Iterable[T]) -> list[T]:
    """Convert iterable to list, unless it's already a list."""
    return maybe_list if isinstance(maybe_list, list) else list(maybe_list)


# SOURCE: vllm/utils/collection_utils.py:L54-L66 —— is_list_of 逐字
def is_list_of(
    value: object,
    typ: type[T] | tuple[type[T], ...],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)


# SUBTRACTED: vllm/utils/collection_utils.py 其余（FactoryDict/common_prefix/
# merge_dicts…）——本章消费面不触及。
