# 只做减法的忠实精简版 —— 镜像两处真实源码（pin f3fef123）：
#   - vllm/v1/sample/logits_processor/state.py    （LogitsProcessors 分类容器）
#   - vllm/v1/sample/logits_processor/builtin.py  （MinP / LogitBias / MinTokens）
#   - vllm/v1/sample/logits_processor/interface.py（LogitsProcessor 抽象基类）
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# SUBTRACTED: 各文件 SPDX 版权头与无关 import。
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from itertools import chain

import torch


# ===== vllm/v1/sample/logits_processor/interface.py: 抽象基类 =====
class LogitsProcessor(ABC):
    # SOURCE: vllm/v1/sample/logits_processor/interface.py:L60-92
    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L75-82
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        # SOURCE: vllm/v1/sample/logits_processor/interface.py:L84-92
        """True if logits processor has no impact on the
        argmax computation in greedy sampling."""
        raise NotImplementedError

    # SUBTRACTED: validate_params / __init__(vllm_config,...) / update_state 抽象方法
    # （interface.py:L61-73, L94-106）—— 持久批（persistent batch）的状态机契约，
    # 属另一子系统；本章只关心 apply() 时 state 已是张量切片这一结果。


# ===== vllm/v1/sample/logits_processor/state.py: 分类容器 =====
class LogitsProcessors:
    # SOURCE: vllm/v1/sample/logits_processor/state.py:L148-165
    """Encapsulates initialized logitsproc objects."""

    def __init__(self, logitsprocs: Iterable["LogitsProcessor"] | None = None) -> None:
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L151-160
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
        # SOURCE: vllm/v1/sample/logits_processor/state.py:L162-165
        """Iterator over all logits processors."""
        return chain(self.argmax_invariant, self.non_argmax_invariant)


# SUBTRACTED: BatchUpdateBuilder / BatchUpdate / RemovedRequest 等（state.py:L18-145,
# interface.py:L17-57）—— persistent batch 的 add/remove/move 状态机，属另一子系统。


# ===== vllm/v1/sample/logits_processor/builtin.py: 三个内建 processor =====
class MinPLogitsProcessor(LogitsProcessor):
    # SUBTRACTED: 原 __init__(vllm_config, device, is_pin_memory)（builtin.py:L23-44）
    # 预分配 cpu/device 双张量并切片维护 self.min_p；本精简版直接接收已就绪的
    # min_p 切片张量（apply 时它就是 [batch,1] 张量），与 update_state 后的状态等价。
    def __init__(self, min_p: torch.Tensor, min_p_count: int):
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L23-44（构造被精简为接收就绪状态）
        self.min_p = min_p
        self.min_p_count = min_p_count

    def is_argmax_invariant(self) -> bool:
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L46-48
        """Min-p never impacts greedy sampling"""
        return True

    # SUBTRACTED: get_min_p_by_index / update_state（builtin.py:L50-99）—— persistent batch 维护。

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L101-115
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):
    # SUBTRACTED: 原 __init__ + update_state + _device_tensor（builtin.py:L119-159）
    # 从逐请求 logit_bias 字典构建 (req_idx, tok_id) 索引与 bias 张量；本精简版直接接收
    # 已就绪的 logits_slice 与 bias_tensor（apply 所需的最终状态）。
    def __init__(self, biases, logits_slice, bias_tensor):
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L119-128（构造被精简为接收就绪状态）
        self.biases = biases
        self.logits_slice = logits_slice
        self.bias_tensor = bias_tensor

    def is_argmax_invariant(self) -> bool:
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L130-133
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L161-164
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):
    # SUBTRACTED: 原 __init__ + add_request + update_state + _device_tensor +
    # apply_with_spec_decode（builtin.py:L168-291）—— persistent batch 维护与投机解码变体。
    # 本精简版直接接收 min_toks 标志、就绪的 logits_slice 索引与 neg_inf_tensor。
    def __init__(self, min_toks, logits_slice, neg_inf_tensor):
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L168-184（构造被精简为接收就绪状态）
        self.min_toks = min_toks
        self.logits_slice = logits_slice
        self.neg_inf_tensor = neg_inf_tensor

    def is_argmax_invariant(self) -> bool:
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L186-189
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L234-238
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits


# SUBTRACTED: process_dict_updates（builtin.py:L294-332）—— persistent batch 稀疏状态字典维护。
