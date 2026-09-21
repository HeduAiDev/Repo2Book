# SOURCE: vllm/v1/sample/logits_processor/builtin.py
# ch34 HOST SEAM（沿 ch30 镜像 + 本章消费面回补）：三件套 MinP/MinTokens/
# LogitBias 的构造器、is_argmax_invariant、apply 全保留；本章消费面回补
# MinTokens 的 add_request/update_state/apply_with_spec_decode 与
# process_dict_updates（RejectionSampler.apply_logits_processors 经
# logitsprocs.non_argmax_invariant 消费 MinTokens.apply_with_spec_decode——
# 互斥清单里唯一的幸存处理器；min_toks 的填充走真实 update_state 面）。
# SUBTRACTED（沿 ch30 镜像）：MinP.update_state（L54-L100）、LogitBias.
#   update_state（L135-L154）——持久批三事件增量维护面归 ch18，本章无消费位
#   （测试自建 state）。
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import torch

from vllm import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")


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
    #   MinP.update_state —— 持久批三事件增量维护（归 ch18；本章无消费位，
    #   测试自建 state：min_p_cpu 落值 + min_p_count 计数 + min_p 设备切片刷新）。

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
#   —— 逐字（update_state 已减去）
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
    #   LogitBias.update_state —— biases dict 的批事件维护 + 稀疏坐标张量重建
    #   （归 ch18；测试自建 state）。

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L156-L157 LogitBias._device_tensor —— 逐字
    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L159-L162 LogitBias.apply —— 逐字（稀疏坐标 +=）
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


# SOURCE: vllm/v1/sample/logits_processor/builtin.py:L165-L286 MinTokensLogitsProcessor
#   —— 本章消费面全保留（含 spec 特化版）
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

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L188-L195 MinTokens.add_request —— 逐字
    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int]] | None:
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L188-L195 MinTokens.add_request
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        return min_tokens, output_tok_ids, params.all_stop_token_ids

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L197-L224 MinTokens.update_state —— 逐字
    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L226-L227 MinTokens._device_tensor —— 逐字
    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(data, device=self.device, dtype=dtype)

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L229-L233 MinTokens.apply —— 逐字（未达标请求封 stop/EOS）
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits

    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L235-L286 MinTokens.apply_with_spec_decode
    #   —— 逐字（spec 特化版：num_draft_tokens cumsum 行位图 + 逐请求
    #   『还要封几位』算术；docstring 自带 worked example）
    def apply_with_spec_decode(
        self,
        logits: torch.Tensor,
        num_draft_tokens: list[int],
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L235-L286 MinTokens.apply_with_spec_decode（spec 特化版）
        """Spec-decode version of apply().
        Priority: ``min_tokens`` > ``stop_token_ids`` / EOS.
        Example: ``num_draft_tokens = [2, 3, 1]``
          → ``logits`` shape ``[6, V]``, ``cumsum = [0, 2, 5, 6]``
          → request 0 owns rows 0‑1, request 1 rows 2‑4, request 2 row 5.
        """
        if not self.min_toks:
            return logits

        num_draft_arr = np.array(num_draft_tokens, dtype=np.int64)
        cumsum = np.concatenate([[0], np.cumsum(num_draft_arr)])

        entries = [
            (req_idx, min_tok, len(out_tok_ids), list(stop_tok_ids))
            for req_idx, (min_tok, out_tok_ids, stop_tok_ids) in self.min_toks.items()
            if stop_tok_ids
        ]

        if not entries:
            return logits

        all_rows: list[np.ndarray] = []  # row indices to mask
        all_toks: list[np.ndarray] = []  # stop-token ids at those rows

        for req_idx, min_tok, current_len, stop_toks in entries:
            remaining = min_tok - current_len
            # How many leading draft positions still need stop-token masking.
            n_mask = int(min(max(remaining, 0), num_draft_arr[req_idx]))

            if n_mask > 0:
                offset = cumsum[req_idx]
                row_indices = np.arange(offset, offset + n_mask, dtype=np.int64)
                n_stop = len(stop_toks)
                all_rows.append(np.repeat(row_indices, n_stop))
                all_toks.append(np.tile(stop_toks, n_mask))

        if all_rows:
            rows_arr = np.concatenate(all_rows)
            toks_arr = np.concatenate(all_toks)
            # (row_indices, token_indices) for index_put_ to set -inf.
            logits_slice = (
                async_tensor_h2d(rows_arr, device=self.device),
                async_tensor_h2d(toks_arr, device=self.device),
            )
            logits.index_put_(logits_slice, self.neg_inf_tensor)

        return logits


# SOURCE: vllm/v1/sample/logits_processor/builtin.py:L289-L327 process_dict_updates
#   —— 逐字（稀疏 dict 状态的批事件更新工具：added/removed/moved 三事件，
#   MinTokens.update_state 的消费位）
def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    # SOURCE: vllm/v1/sample/logits_processor/builtin.py:L289-L327 process_dict_updates（批事件三段更新）
    """Utility function to update dict state for sparse LogitsProcessors."""

    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Process removed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Process moved requests, unidirectional (a->b) and
        # swapped (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
