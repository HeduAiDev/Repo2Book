# SOURCE: vllm/v1/core/sched/output.py
# 只做减法的忠实精简版：本章消费面 = SchedulerOutput（schedule() 出件 /
# update_from_output 入件——num_scheduled_tokens 迭代序即结构化请求行序的
# 契约载体 + has_structured_output_requests 旗标）。
# SUBTRACTED: SPDX 版权头；NewRequestData/CachedRequestData 的 mm/PP 字段面
#   （worker 增量下发域，ch10/ch18）与 spec/encoder 统计字段；GrammarOutput
#   （L287-L291——本章交棒点 get_grammar_bitmask 不进精简版（delete[2]），
#   GrammarOutput 是 ch31 主角的出件载体）。
from dataclasses import dataclass, field


# SOURCE: vllm/v1/core/sched/output.py SchedulerOutput —— 字段面精简
@dataclass
class SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py:L213-L214（scheduled_new_reqs 形位
    #   ——HOST SEAM 以空 list 默认承载）
    scheduled_new_reqs: list = field(default_factory=list)
    # SOURCE: vllm/v1/core/sched/output.py:L217 scheduled_cached_reqs（同上形位
    #   ——HOST SEAM：None 占位，真实是 CachedRequestData）
    scheduled_cached_reqs: object = None

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    # SOURCE: vllm/v1/core/sched/output.py:L220-L223
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    # SOURCE: vllm/v1/core/sched/output.py:L224-L225
    total_num_scheduled_tokens: int = 0
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    # SOURCE: vllm/v1/core/sched/output.py:L226-L229
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    # SUBTRACTED: scheduled_encoder_inputs/num_common_prefix_blocks/
    #   finished_req_ids/free_encoder_mm_hashes/scheduled_encoder_input_stats/
    #   preempted_req_ids/num_invalid_spec_tokens（encoder/KV/spec 统计面）。

    # Whether any of the scheduled requests use structured output.
    # Set only in async scheduling case.
    # SOURCE: vllm/v1/core/sched/output.py:L236-L238 —— 逐字
    has_structured_output_requests: bool = False

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    # SOURCE: vllm/v1/core/sched/output.py:L240-L241 —— 逐字（消费面归 ch31）
    pending_structured_output_tokens: bool = False
