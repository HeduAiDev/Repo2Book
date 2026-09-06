# SOURCE: vllm/v1/sample/logits_processor/state.py
# ch29 m2 结构基础：LogitsProcessors 容器——构造期按 is_argmax_invariant()
# 把处理器分进两列（非不变列 step5 greedy 前 / 不变列 step7c 温度后随机路径）。
# SUBTRACTED：delete[5] BatchUpdateBuilder 全类（L18-L145）——持久批状态机
#   的变更聚合器（removed/added/moved 三事件），调用方 InputBatch 不在
#   本章精简范围（归 ch18）；连带修剪 import 的四个类型名（L7-L12：
#   AddedRequest/BatchUpdate/MovedRequest/RemovedRequest——均只被已删
#   符号消费）。
from collections.abc import Iterable, Iterator
from itertools import chain
from typing import TYPE_CHECKING

# SUBTRACTED: vllm/v1/sample/logits_processor/state.py:L7-L12
#   `from vllm.v1.sample.logits_processor.interface import (AddedRequest,
#   BatchUpdate, MovedRequest, RemovedRequest)` —— delete[5] 计划明列的
#   连带 import 修剪（四类型只被已删的 BatchUpdateBuilder 消费）。

if TYPE_CHECKING:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor


# SUBTRACTED: vllm/v1/sample/logits_processor/state.py:L18-L145
#   BatchUpdateBuilder 全类 —— delete[5]（持久批状态机维护面归 ch18：
#   removed 降序聚合/added/moved 三事件账本与 get_and_reset 收口）。
#   精简版直接以构造好的 LogitsProcessors 容器为输入（测试自建 state），
#   apply()/is_argmax_invariant() 路径不变。


# SOURCE: vllm/v1/sample/logits_processor/state.py:L148-L160 LogitsProcessors
#   —— 逐字（两列分类）
class LogitsProcessors:
    """Encapsulates initialized logitsproc objects."""

    def __init__(self, logitsprocs: Iterable["LogitsProcessor"] | None = None) -> None:
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L151-L160 LogitsProcessors.__init__ —— 逐字（构造期两列分类）
        self.argmax_invariant: list[LogitsProcessor] = []
        self.non_argmax_invariant: list[LogitsProcessor] = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L162-L165 LogitsProcessors.all —— 逐字
        """Iterator over all logits processors."""
        return chain(self.argmax_invariant, self.non_argmax_invariant)
