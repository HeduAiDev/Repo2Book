# SOURCE: vllm/v1/core/sched/scheduler.py
# 只做减法的忠实精简版（pin ad7125a4 / v0.21.0）。真实 Scheduler 是几千行的调度器
# 核心（KV cache 分配/抢占/cascade attention/多种 connector），本章只抽取与
# 结构化输出掩码装配、投机耦合直接相关的四个切面：
#   1. _update_after_schedule 里置位 has_structured_output_requests（m01 门控）
#   2. get_grammar_bitmask（上下篇交棒点，本章入口）
#   3. update_draft_token_ids_in_output（投机耦合·装配前：-1 补齐 + 掩码行数对齐）
#   4. make_spec_decoding_stats（num_invalid_spec_tokens 的消费点：接受率统计
#      要扣掉被语法过滤掉的草稿数）
# 与 code_spine/must_keep 覆盖范围之外的调度逻辑（抢占、KV block 分配、cascade
# attention 等，ch13 范围）一并省略——不是需要单独批准的删除项，而是未被选入
# 本章切面。
#
# SUBTRACTED: SPDX 版权头。
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from output import GrammarOutput, SchedulerOutput
    from outputs import DraftTokenIds
    from request import Request
    from structured_output_manager import StructuredOutputManager


@dataclasses.dataclass
class SpecDecodingStats:
    # SOURCE: vllm/v1/spec_decode/metrics.py（精简为本章唯一用到的两个字段与方法）
    #
    # SUBTRACTED: 真实 SpecDecodingStats 另有 num_drafts/num_emitted_tokens 等
    # 累计字段与按草稿位置分桶的接受计数（ch34 范围）——本章只关心
    # observe_draft 被调用时 num_draft_tokens 是否已被 num_invalid_spec_tokens
    # 扣减过。
    num_spec_tokens: int
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        # SOURCE: vllm/v1/spec_decode/metrics.py:L32-36
        # SUBTRACTED: `num_accepted_tokens_per_pos` per-position accept counter
        # (metrics.py:L29-30) -- per-position acceptance breakdown is a display
        # concern for the stats logger, not the num_invalid_spec_tokens
        # consumption this chapter is tracing.
        return cls(num_spec_tokens=num_spec_tokens)

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int) -> None:
        # SOURCE: vllm/v1/spec_decode/metrics.py:L39-45
        # SUBTRACTED: `num_drafts` counter increment and the
        # `num_accepted_tokens_per_pos` per-position loop -- same reason as
        # above.
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens


class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py class Scheduler（精简到本章四个切面）
    def __init__(
        self,
        structured_output_manager: "StructuredOutputManager",
        num_spec_tokens: int = 0,
        log_stats: bool = True,
    ) -> None:
        self.requests: "dict[str, Request]" = {}
        self.structured_output_manager = structured_output_manager
        self.num_spec_tokens = num_spec_tokens
        self.log_stats = log_stats

    def _update_after_schedule(self, scheduler_output: "SchedulerOutput") -> None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L932-951（精简到结构化输出相关的
        # 两行；num_computed_tokens/is_prefill_chunk 的完整推进逻辑属 ch13）
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )

    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1224-1246
        from output import GrammarOutput

        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_draft_token_ids_in_output(
        self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
    ) -> None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1623-1657
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: "SpecDecodingStats | None",
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: "dict[str, int] | None",
        request_id: str,
    ) -> "SpecDecodingStats | None":
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1901-1917
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats
