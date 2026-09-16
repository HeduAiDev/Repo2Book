# SOURCE: vllm/v1/sample/logits_processor/builtin.py
# ch29 三件套：MinP（argmax 不变列唯一内置代表）+ MinTokens/LogitBias
# （非不变列）——构造器、is_argmax_invariant 声明、apply 算式全保留。
# SUBTRACTED：delete[5] 持久批 update_state 侧——MinP.update_state
#   （L54-L100）、LogitBias.update_state（L135-L154）、MinTokens.update_state
#   （L197-L224）、MinTokens.add_request（L188-L195）、process_dict_updates
#   （L289-L327）；delete[2] MinTokens.apply_with_spec_decode（L235-L286，
#   spec decode 归 ch32/33）；连带 import 修剪（L11-L16 的
#   BatchUpdate/MoveDirectionality——计划明列；以及只被已删符号消费的
#   numpy/TypeVar/Callable/SamplingParams 名）。构造器与 _device_tensor、
#   is_argmax_invariant、apply 按「禁删」清单全保留（build_logitsprocs
#   三件套构造的必经链）。测试自建 state（直接写处理器内部张量）。
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import LogitsProcessor

# SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L6 `import numpy as np`
#   —— 只被已删的 apply_with_spec_decode 消费（delete[2] 连带）。
# SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L9
#   `from vllm import SamplingParams` 与 L11-L16 interface import 里的
#   BatchUpdate/MoveDirectionality —— 只被已删的 update_state 家族消费
#   （delete[5] 计划明列的连带 import 修剪）。
# SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L4
#   `from collections.abc import Callable` 与 L20 TypeVar —— 只被已删的
#   process_dict_updates 消费（delete[5] 连带）。

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# SOURCE: vllm/v1/sample/logits_processor/builtin.py:L23 MinPLogitsProcessor
#   —— 类位（构造器 L24-L46 逐字：max_num_seqs 槽位预分配 + CPU numpy
#   视图 + 设备双缓冲）
class MinPLogitsProcessor(LogitsProcessor):
    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L24-L46 MinP 构造器 —— 逐字
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=is_pin_memory
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L47-L49 MinP.is_argmax_invariant —— 逐字（阈值 min_p×max_prob、最高位恒留）
    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling"""
        return True

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L51-L52 MinP.get_min_p_by_index —— 逐字
    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    # SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L54-L100
    #   MinP.update_state —— delete[5]（持久批三事件增量维护，归 ch18；
    #   精简版测试自建 state：min_p_cpu 落值 + min_p_count 计数 + min_p
    #   设备切片刷新，即 update_state 的等价终态）。

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L102-L116 MinP.apply —— 逐字（min_p×max_prob 阈值砍尾）
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
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


# SOURCE: vllm/v1/sample/logits_processor/builtin.py:L119-L162 LogitBiasLogitsProcessor
#   —— 逐字（update_state 已按 delete[5] 减去）
class LogitBiasLogitsProcessor(LogitsProcessor):
    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L120-L128 LogitBias 构造器 —— 逐字
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L130-L133 LogitBias.is_argmax_invariant —— 逐字
    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    # SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L135-L154
    #   LogitBias.update_state —— delete[5]（biases dict 的批事件维护 +
    #   稀疏坐标张量重建，归 ch18；测试自建 state）。

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L156-L157 LogitBias._device_tensor —— 逐字
    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L159-L162 LogitBias.apply —— 逐字（稀疏坐标 +=）
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


# SOURCE: vllm/v1/sample/logits_processor/builtin.py:L165-L233 MinTokensLogitsProcessor
#   —— 逐字（update_state/add_request/apply_with_spec_decode 已减去）
class MinTokensLogitsProcessor(LogitsProcessor):
    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L166-L182 MinTokens 构造器 —— 逐字
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=self.device
        )

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L183-L186 MinTokens.is_argmax_invariant —— 逐字
    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    # SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L188-L195
    #   MinTokens.add_request —— delete[5]（update_state 的 new_state 回调，
    #   随维护面删除；测试自建 min_toks dict）。

    # SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L197-L224
    #   MinTokens.update_state —— delete[5]（达标请求摘除 + 张量重建，归 ch18）。

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L226-L227 MinTokens._device_tensor —— 逐字
    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L229-L233 MinTokens.apply —— 逐字（未达标请求封 stop/EOS）
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits

    # SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L235-L286
    #   MinTokens.apply_with_spec_decode —— delete[2]（spec decode 的多 draft
    #   行掩码版（num_draft_tokens cumsum 行位图），归 ch32/33；非 spec
    #   路径无消费者）。


# SUBTRACTED: vllm/v1/sample/logits_processor/builtin.py:L289-L327
#   process_dict_updates —— delete[5]（稀疏 dict 状态的批事件更新工具，
#   只被已删的三个 update_state 与 AdapterLogitsProcessor 消费；归 ch18）。
