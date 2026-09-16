# SOURCE: vllm/v1/outputs.py
# HOST SEAM：本章消费面三件——LogprobsTensors（sampler 出件的载体，
# gather_logprobs/num_logprobs==-1 两处构造）、LogprobsLists（tolists 的
# 返回类型）、SamplerOutput（step9 出件）。逐字；文件其余类（RoutedExperts/
# KVConnector/ModelRunnerOutput 等）不在本章消费面、不载入。
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


# SOURCE: vllm/v1/outputs.py:L28-L52 LogprobsLists —— 逐字
class LogprobsLists(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: np.ndarray
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: np.ndarray
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: np.ndarray
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    cu_num_generated_tokens: list[int] | None = None

    # SOURCE: vllm/v1/outputs.py:L45-L52 LogprobsLists.slice_request —— 逐字
    def slice_request(self, req_idx: int, num_positions: int):
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )


# SOURCE: vllm/v1/outputs.py:L53-L80 LogprobsTensors —— 逐字
class LogprobsTensors(NamedTuple):
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: torch.Tensor
    # [num_reqs]
    cu_num_generated_tokens: list[int] | None = None

    # SOURCE: vllm/v1/outputs.py:L59-L68 LogprobsTensors.tolists —— 逐字
    def tolists(self, cu_num_generated_tokens: list[int] | None = None):
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens
            if cu_num_generated_tokens is not None
            else self.cu_num_generated_tokens,
        )

    # SOURCE: vllm/v1/outputs.py:L70-L79 LogprobsTensors.to_cpu_nonblocking —— 逐字
    def to_cpu_nonblocking(self) -> "LogprobsTensors":
        if self.logprob_token_ids.device.type == "cpu":
            return self
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
            self.cu_num_generated_tokens,
        )

    # SUBTRACTED: vllm/v1/outputs.py:L81-L138 LogprobsTensors.filter/cat/
    #   empty_cpu——D2H 之后的切行/装配/拼接面（logprobs 支路全景归 ch8，
    #   本章只出 GPU 张量到 SamplerOutput）。


@dataclass
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    # SOURCE: vllm/v1/outputs.py:L212-L220 SamplerOutput —— 逐字
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
