# SOURCE: vllm/v1/outputs.py
# ModelRunnerOutput / AsyncModelRunnerOutput —— worker→engine 的输出契约面。
# 本章保留：req_ids/req_id_to_index/sampledtoken_ids（update_from_output 热循环
# 的定位与采样行）+ get_output 抽象契约（AsyncGPUModelRunnerOutput 实现在
# gpu_model_runner.py）。完整字段版（logprobs/pooler/routed/cudagraph）归
# ch8/ch9/ch17。
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/v1/outputs.py:L289 ModelRunnerOutput（v0.27.1 字段群精简到本章
# 消费面）
@dataclass
class ModelRunnerOutput:
    # SUBTRACTED: logprobs/prompt_logprobs_dict/pooler_output/kv_connector_
    #   output/num_nans_in_logits/cudagraph_stats/routed_experts——观测与
    #   connector 面（ch8/ch9/ch17 全文已立）。
    # SOURCE: vllm/v1/outputs.py:L293-L296 [num_reqs]
    req_ids: list[str] = field(default_factory=list)
    # SOURCE: vllm/v1/outputs.py:L298-L300 req_id → batch row
    req_id_to_index: dict[str, int] = field(default_factory=dict)
    # SOURCE: vllm/v1/outputs.py:L302-L305 [num_reqs][num_schedule_tokens]
    sampled_token_ids: list[list[int]] | None = None

    # SOURCE: vllm/v1/outputs.py with_kv_conn_output_only（PP 特例）——第 5 条
    #   删除批准（PP 面），不保留。


# SOURCE: vllm/v1/outputs.py:L341-L374 EMPTY_MODEL_RUNNER_OUTPUT（逐字语义）
# SOURCE: vllm/v1/outputs.py:L375 EMPTY_MODEL_RUNNER_OUTPUT
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput(req_ids=[], req_id_to_index={})


# SOURCE: vllm/v1/outputs.py AsyncModelRunnerOutput — ABC (get_output 契约)
class AsyncModelRunnerOutput(ABC):
    # SOURCE: vllm/v1/outputs.py AsyncModelRunnerOutput.get_output (契约逐字)
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        # SOURCE: vllm/v1/outputs.py AsyncModelRunnerOutput.get_output (契约逐字)
        raise NotImplementedError


# SUBTRACTED: IntermediateTensors / DraftTokenIds / LogprobsTensors 等伴生类型
#   （PP/spec/观测面——第 5/6 条删除批准与邻章）。
