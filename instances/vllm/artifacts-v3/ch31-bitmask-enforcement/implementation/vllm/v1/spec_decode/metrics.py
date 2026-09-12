# SOURCE: vllm/v1/spec_decode/metrics.py
# HOST SEAM：本章消费面一件——SpecDecodingStats（update_from_output 的
# make_spec_decoding_stats 调用位：num_invalid_spec_tokens 进接受率统计的
# 落点，scheduler.py:L1785-L1790）。new/observe_draft 逐字；
# dataclass 字段面承载。
from __future__ import annotations

from dataclasses import dataclass, field


# SOURCE: vllm/v1/spec_decode/metrics.py:L17-L30 SpecDecodingStats —— 逐字
@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in EngineCoreOutputs->SchedulerStats.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)

    # SOURCE: vllm/v1/spec_decode/metrics.py:L33-L39 new —— 逐字
    @classmethod
    # SOURCE: vllm/v1/spec_decode/metrics.py:L33-L39 new —— 逐字
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_draft_tokens_per_pos=[0] * num_spec_tokens,
        )

    # SOURCE: vllm/v1/spec_decode/metrics.py:L41-L50 observe_draft —— 逐字
    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        for i in range(num_draft_tokens):
            self.num_draft_tokens_per_pos[i] += 1
