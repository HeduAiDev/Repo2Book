# SOURCE: vllm/v1/outputs.py
# 本章消费面：SamplerOutput（_sample 的返回型）/ ModelRunnerOutput（sample_
# tokens 的组装与 EMPTY 单例）。SUBTRACTED: AsyncModelRunnerOutput 及
# AsyncGPUModelRunnerOutput 的包裹协议（ch12 已全文立——逐字在 ch12 精简版）、
# KVConnectorOutput/ECConnectorOutput/DraftTokenIds/PoolerOutput 族（ch16/ch33/
# ch29 域，本章只留注解占位）。
from __future__ import annotations

from dataclasses import dataclass, field

import torch

# 注解占位（真实 import 自各域文件；字段均默认 None/不构造，行为无涉）
# SOURCE: vllm/v1/outputs.py:L13-L14 CUDAGraphStat 等类型注解面
CUDAGraphStat = object  # ch19 域（真实 vllm/compilation/cuda_graph.py）
LogprobsLists = object  # ch08 域（真实 vllm/v1/sample/logprobs_processor.py）
LogprobsTensors = object  # ch08 域（真实 vllm/v1/outputs.py 内定义）
KVConnectorOutput = object  # ch16 域（真实 vllm/v1/outputs.py 内定义）
ECConnectorOutput = object  # mm 域（真实 vllm/v1/outputs.py 内定义）
RoutedExpertsLists = object  # MoE 域（真实 vllm/v1/outputs.py 内定义）


# SOURCE: vllm/v1/outputs.py:L213 SamplerOutput
@dataclass
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    # SOURCE: vllm/v1/outputs.py:L218-L219
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
# SOURCE: vllm/v1/outputs.py:L261 ModelRunnerOutput
@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    # SOURCE: vllm/v1/outputs.py:L263-L265
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None = None

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )

    # [num_reqs, hidden_size]
    pooler_output: list[torch.Tensor | None] | None = None

    kv_connector_output: KVConnectorOutput | None = None

    ec_connector_output: ECConnectorOutput | None = None

    # req_id -> num_nans_in_logits
    num_nans_in_logits: dict[str, int] | None = None

    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None

    # Per-step routed experts data captured by the worker.
    # ``routing_data`` shape: (num_scheduled_tokens, num_layers,
    #                         num_experts_per_tok); expert IDs as uint8/uint16.
    # ``slot_mapping`` shape: (num_scheduled_tokens,); physical KV-cache
    # slot for each row of routing_data.
    # ``num_scheduled_tokens`` is step-level (total across all requests
    # in this step), not per-request.
    # ``None`` when ``enable_return_routed_experts`` is off.
    routed_experts: RoutedExpertsLists | None = None

    # SUBTRACTED: with_kv_conn_output_only（L311-L323——KV connector 传递面，
    #   消费点在已删除的 sample_tokens 空槽早退支，ch16 域）。


# SOURCE: vllm/v1/outputs.py:L375 EMPTY_MODEL_RUNNER_OUTPUT 单例
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})
