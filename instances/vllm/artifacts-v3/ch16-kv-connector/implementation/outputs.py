# SOURCE: vllm/v1/outputs.py
# KVConnectorOutput——worker→scheduler 的回传包（L223-L248）：finished_
# sending/finished_recving（异步收/发完成——m9/m11 的输入）、invalid_block_
# ids（失败块——m10 的输入）。ModelRunnerOutput 的 connector 面账位与
# with_kv_conn_output_only（no_forward 步的回传形态）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 3 条观测面：kv_connector_stats/kv_cache_events 字段与 expected_
#     finished_count（KVOutputAggregator 聚合——握手计数归 ch36）；
#   第 1 条 ECConnectorOutput 及 logprobs/pooler/routed_experts 等
#     worker 输出面（ch07/08 各章）；
#   AsyncModelRunnerOutput/DraftTokenIds（ch12/ch33）。
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import KVConnectorWorkerMetadata

# SOURCE: vllm/v1/outputs.py:L~230 ModelRunnerOutput 的空实例常量（with_
#   kv_conn_output_only 的底——本章空步回传用）


# SOURCE: vllm/v1/outputs.py:L223 KVConnectorOutput
@dataclass
class KVConnectorOutput:
    # [req_ids]
    # SOURCE: vllm/v1/outputs.py:L225-L226
    finished_sending: set[str] | None = None
    finished_recving: set[str] | None = None
    # SUBTRACTED: kv_connector_stats/kv_cache_events（L227-L228——第 3 条
    #   观测面）。
    kv_connector_worker_meta: "KVConnectorWorkerMetadata | None" = None
    # IDs of externally computed KV blocks that failed to load.
    # Requests referencing these blocks should be rescheduled to recompute them
    # SOURCE: vllm/v1/outputs.py:L231-L233 invalid_block_ids
    invalid_block_ids: set[int] = field(default_factory=set)
    # SUBTRACTED: expected_finished_count（L234-L239——握手计数/KVOutput
    #   Aggregator → ch36）。

    # SOURCE: vllm/v1/outputs.py:L241 is_empty
    def is_empty(self):
        # SOURCE: vllm/v1/outputs.py:L242-L249（stats/events 字段随第 3 条删）
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.invalid_block_ids
            and not self.kv_connector_worker_meta
        )


# SUBTRACTED: ECConnectorOutput（L252 起——第 1 条 ECConnector 全删）。


# SOURCE: vllm/v1/outputs.py:L~180 ModelRunnerOutput（connector 面切面）
@dataclass
class ModelRunnerOutput:
    # [num_reqs]
    # SOURCE: vllm/v1/outputs.py:L182-L184
    req_ids: list[str] = field(default_factory=list)
    # req_id -> index
    req_id_to_index: dict[str, int] = field(default_factory=dict)

    # num_reqs x num_generated_tokens
    # SOURCE: vllm/v1/outputs.py:L186-L189
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # SUBTRACTED: logprobs/prompt_logprobs/pooler/cudagraph/routed_experts/
    #   ec_connector_output/num_nans（各邻章输出面）。

    # SOURCE: vllm/v1/outputs.py:L212 kv_connector_output 账位
    kv_connector_output: KVConnectorOutput | None = None

    # SOURCE: vllm/v1/outputs.py:L311 with_kv_conn_output_only
    @staticmethod
    def with_kv_conn_output_only(
        kv_connector_output: KVConnectorOutput | None,
    ) -> "ModelRunnerOutput":
        """Return ModelRunnerOutput containing the provided KVConnectorOutput,
        otherwise empty. Returns None if kv_connector_output is passed as None.
        """
        # SOURCE: vllm/v1/outputs.py:L317-L323
        if kv_connector_output is None or kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output


# SOURCE: vllm/v1/outputs.py EMPTY_MODEL_RUNNER_OUTPUT 模块级常量
EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput()
