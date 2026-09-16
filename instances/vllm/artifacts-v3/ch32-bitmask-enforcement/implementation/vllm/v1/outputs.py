# SOURCE: vllm/v1/outputs.py
# HOST SEAM：本章消费面三件——ModelRunnerOutput（update_from_output 的入参：
# req_ids/req_id_to_index/sampled_token_ids）、SamplerOutput（_sample 的出参
# 载体）、DraftTokenIds（草稿回传通道的载荷，spec_decode/utils.py 与
# update_draft_token_ids_in_output 消费）。字段按消费切片承载，
# logprobs/pooler/connector/routed_experts 等旁路面删（归 ch8/ch16）。
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    pass


# SOURCE: vllm/v1/outputs.py:L213-L218 SamplerOutput —— 字段面承载
@dataclass
# SOURCE: vllm/v1/outputs.py:L213-L218 SamplerOutput —— 字段面承载
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: Any | None = None  # 真实类型 LogprobsTensors | None（ch8）


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
# SOURCE: vllm/v1/outputs.py:L261-L322 ModelRunnerOutput —— 消费切片承载
@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    # SUBTRACTED: logprobs/prompt_logprobs/pooler/kv_connector/ec_connector/
    #   num_nans/cudagraph_stats/routed_experts 字段（L267-L322——ch8/ch16/ch19
    #   的消费面）与 with_kv_conn_output_only（KV-only 拍早退分支归 ch16）。

    @staticmethod
    def with_kv_conn_output_only(kv_connector_output) -> "ModelRunnerOutput":
        # SOURCE: vllm/v1/outputs.py:L305-L322 with_kv_conn_output_only ——
        #   HOST SEAM 最小承载（本章 KV-only 拍不展开）
        return ModelRunnerOutput(req_ids=[], req_id_to_index={})


# ModelRunnerOutput wrapper for async scheduling.
# SOURCE: vllm/v1/outputs.py:L325-L336 AsyncModelRunnerOutput —— 逐字
class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    # SOURCE: vllm/v1/outputs.py:L325-L336 AsyncModelRunnerOutput.get_output —— 逐字
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        pass


# SOURCE: vllm/v1/outputs.py:L338-L343 DraftTokenIds —— 逐字
@dataclass
# SOURCE: vllm/v1/outputs.py:L338-L343 DraftTokenIds —— 逐字
class DraftTokenIds:
    # [num_reqs]
    req_ids: list[str]
    # num_reqs x num_draft_tokens
    draft_token_ids: list[list[int]]
