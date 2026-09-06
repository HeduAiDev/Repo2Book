# SOURCE: vllm/v1/sample/logits_processor/interface.py
# ch29 m2 契约位：LogitsProcessor 抽象基类——apply 与 is_argmax_invariant
# 是「决定处理器进哪条流水线」的语义承诺（分类错=静默破坏 greedy 语义）。
# SUBTRACTED：delete[5] BatchUpdate/MoveDirectionality/AddedRequest 等类型
#   （L17-L57）与 update_state 抽象方法（L96-L108）——持久批增量维护面
#   （BatchUpdate(removed/added/moved) 三事件 + 共享 output_tok_ids 引用
#   不变量）归 ch18；精简版以构造好的处理器为输入，接口面只剩
#   构造/apply/is_argmax_invariant/validate_params。
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# SUBTRACTED: vllm/v1/sample/logits_processor/interface.py:L17-L21
#   MoveDirectionality —— delete[5]（moved 事件的 UNIDIRECTIONAL/SWAP
#   两义枚举，只被已删的 BatchUpdate 家族消费）。
# SUBTRACTED: vllm/v1/sample/logits_processor/interface.py:L24-L33
#   RemovedRequest/AddedRequest/MovedRequest 三类型别名 —— delete[5]（同上）。
# SUBTRACTED: vllm/v1/sample/logits_processor/interface.py:L36-L57
#   BatchUpdate frozen dataclass —— delete[5]（持久批状态变更描述符；
#   「added 里的 output_tok_ids 是共享 list 引用」的不变量注释在
#   正文引真源码讲，归 ch18）。


# SOURCE: vllm/v1/sample/logits_processor/interface.py:L60-L109 LogitsProcessor
#   —— 逐字（update_state 抽象方法已按 delete[5] 减去）
class LogitsProcessor(ABC):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L61-L69 LogitsProcessor.validate_params —— 逐字
        """Validate sampling params for this logits processor.

        Raise ``VLLMValidationError`` (preferred) / ``ValueError`` (backward compatible)
        for invalid params. Bare ``ValueError`` is converted to ``VLLMValidationError``
        at the engine boundary so online serving returns HTTP 400.
        """
        return None

    @abstractmethod
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L71-L75 LogitsProcessor.__init__ —— 逐字
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L77-L84 LogitsProcessor.apply —— 逐字
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be
        modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L86-L94 LogitsProcessor.is_argmax_invariant —— 逐字（章节核心概念的可检测锚点）
        """True if logits processor has no impact on the
        argmax computation in greedy sampling.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        """
        raise NotImplementedError

    # SUBTRACTED: vllm/v1/sample/logits_processor/interface.py:L96-L108
    #   LogitsProcessor.update_state 抽象方法 —— delete[5]（随批增量维护面
    #   归 ch18；本精简版不实现该面）。
