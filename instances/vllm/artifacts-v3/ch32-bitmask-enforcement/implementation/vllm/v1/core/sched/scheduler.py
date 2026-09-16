# SOURCE: vllm/v1/core/sched/scheduler.py
# v3 ch31 脊柱④：Scheduler 的掩码账本切面——_update_after_schedule 的
# has_structured_output_requests 置位（L1317-L1343，m05）、get_grammar_bitmask
# （L1646-L1668，m03 行序账本）、update_from_output 的 spec 接受率统计 +
# 语法真推进段（L1746-L1843 切面：should_advance → trim_reasoning_for_advance
# → accept_tokens，拒绝即 FINISHED_ERROR）、update_draft_token_ids（L2147-L2166）
# 与 update_draft_token_ids_in_output（L2168-L2203，m17 草稿语法过滤+(-1) 补齐
# +num_invalid 记账）。
# SUBTRACTED：调度循环本体 schedule()（L439-L1315，ch10/ch11/ch12 全文已立）、
# 真实 __init__ 的几百行装配（L70-L360）、KV/connector/encoder/抢占/统计尾巴。
from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar

logger = init_logger(__name__)
# SUBTRACTED: from vllm.v1.request import Request（L39——Request 本体归 ch02，
#   本章 requests 字典的值以测试替身承载；注解面以字符串引用）。


# SOURCE: vllm/v1/core/sched/scheduler.py:L69 Scheduler —— 本章切面
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py:L70-L72 __init__ 签名 —— 保留
    #   （真实 (vllm_config, kv_cache_config, structured_output_manager,
    #   include_finished_set, log_stats, block_size, hash_block_size)——
    #   本章消费面裁剪）
    def __init__(
        self,
        vllm_config,
        structured_output_manager=None,
        log_stats: bool = False,
    ) -> None:
        # SUBTRACTED: 真实 __init__ 装配 L70-L360（策略/预算/KV 管理/connector/
        #   encoder/抢占簿记/观测——ch10/ch11/ch13/ch16 各自的域）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L87-L88
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        # SOURCE: vllm/v1/core/sched/scheduler.py:L95
        self.structured_output_manager = structured_output_manager
        # SOURCE: vllm/v1/core/sched/scheduler.py:L120-L123
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = (
            1 if not vllm_config.model_config.is_diffusion else 0
        )
        # SOURCE: vllm/v1/core/sched/scheduler.py:L177-L178
        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # SOURCE: vllm/v1/core/sched/scheduler.py:L246
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        # SOURCE: vllm/v1/core/sched/scheduler.py:L360
        self._inflight_prefills: set[Request] = set()  # noqa: F821（Request 注解面，ch02）

    # SUBTRACTED: vllm/v1/core/sched/scheduler.py:L439-L1315 schedule() ——
    #   调度循环本体（预算分配/抢占/前缀缓存/continue-generation——
    #   ch10/ch11/ch12/ch15 全文已立）。本章以构造好的 SchedulerOutput 为起点。

    #   —— 逐字（仅删 defer_block_free 栅栏记账与 routed_experts 快照两段）
    # SOURCE: vllm/v1/core/sched/scheduler.py:L1317-L1343 _update_after_schedule
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            # SUBTRACTED: L1332-L1334 defer_block_free 的 in-flight 栅栏记账
            #   （ch11 内存域的延迟释放栅栏，dossier 摘录 elide 注明归 ch11）。
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # SUBTRACTED: L1345-L1365 routed_experts 块号快照（enable_return_
        #   routed_experts 的消费面——ch27 量化/路由域）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1646-L1668 get_grammar_bitmask —— 逐字
    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
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

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1670-L2145 update_from_output ——
    #   掩码账本切面（每请求循环里的 spec 接受率统计 L1769-L1791 + 语法真推进
    #   L1797-L1843 逐字；扣在途/stale/KV 失败/编码器/停止判据/输出聚合等删）
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, "EngineCoreOutputs"]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        # SUBTRACTED: logprobs/pooler_outputs/num_nans/kv_connector_output/
        #   cudagraph_stats 取值与 defer_block_free 释放栅栏、perf_stats、
        #   failed_kv_load、routed_experts 落槽（L1672-L1690——ch8/ch11/ch16）。

        spec_decoding_stats: SpecDecodingStats | None = None
        # SUBTRACTED: outputs 聚合字典与 stopped 集合（L1695-L1697——输出聚合
        #   归 ch9/ch12 切面；本章返回空 dict 占位同一签名）。

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best to
        # avoid expensive operations inside the loop.
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            # SUBTRACTED: 扣在途 num_in_flight_tokens / stale 锁步 drain /
            #   KV 失败重排（L1728-L1767——ch11/ch12/ch16）。
            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and (
                generated_token_ids or self.num_sampled_tokens_per_step == 0
            ):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                # Rejections roll back num_computed_tokens (and, under async
                # scheduling, num_output_placeholders, which covers the spec
                # tokens). A stale rejection count predates the preemption
                # rollback and must not apply.
                # SUBTRACTED: L1780-L1784 拒绝对 computed/placeholders 的回扣
                #   （ch12 m15 全文已立）。
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # SUBTRACTED: encoder 输入释放与 stop 判据
            #   _update_request_with_output（L1793-L1816——ch11）。
            new_token_ids = generated_token_ids

            # SOURCE: vllm/v1/core/sched/scheduler.py:L1817-L1843 语法真推进 —— 逐字
            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert isinstance(grammar, StructuredOutputGrammar)
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                advance_token_ids = (
                    self.structured_output_manager.trim_reasoning_for_advance(
                        request, new_token_ids
                    )
                )
                if advance_token_ids and not grammar.accept_tokens(
                    req_id, advance_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False

            # SUBTRACTED: routed_experts 读槽/输出聚合/EngineCoreOutput 组装/
            #   抢占处理/清理（L1845-L2145——ch7/ch9/ch11/ch27）。

        # SUBTRACTED: 输出聚合 L2140-L2145——返回空 dict 承载同一签名
        #   （聚合细节归 ch9/ch12 切面）。
        return {}

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2147-L2166 update_draft_token_ids —— 逐字
    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2168-L2203 update_draft_token_ids_in_output —— 逐字
    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
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
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2533-L2550 make_spec_decoding_stats —— 逐字
    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
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
