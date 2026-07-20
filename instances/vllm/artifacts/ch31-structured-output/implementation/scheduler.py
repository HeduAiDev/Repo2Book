# SOURCE: vllm/v1/core/sched/scheduler.py
# 只做减法的忠实精简版——真实 Scheduler 是几千行的调度核心（schedule/
# update_from_output 主循环已在 ch13 详细讲过）。本章只抽取与"语法编译异步门 +
# 谁在推进语法状态机"直接相关的方法/片段：阻塞态判定、阻塞态晋级（结构化输出分支）、
# accept_tokens/validate_tokens 两个调用点、bitmask 交棒点的 id 过滤。
# KV cache 分配、抢占、优先级队列、P/D、投机解码主流程等一律不在本章范围。
from request import RequestStatus


class Scheduler:
    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1515-1521
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _try_promote_blocked_waiting_request(self, request) -> bool:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1998-2023
        #
        # SUBTRACTED: `WAITING_FOR_REMOTE_KVS`（P/D，读 finished_recving_kv_req_ids /
        # num_preemptions）与 `WAITING_FOR_STREAMING_REQ` 两个分支——不在本章
        # must_keep 范围，且依赖的状态本就不是这个精简版 Scheduler 持有的字段。
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True
        return False

    @staticmethod
    def _advance_grammar_on_sampled_tokens(request, new_token_ids: list[int]) -> None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1360-1369（update_from_output 内的
        # 内联片段，抽成独立方法便于单测）——accept_tokens 真正的调用点：用这一步
        # 真实采样出的 token 推进语法状态机，而不是候选/草稿 token。
        #
        # SUBTRACTED: `self.structured_output_manager.should_advance(request)` 门控
        # ——真实版还要过 reasoning_ended 判断（推理模型"思考"阶段跳过语法约束，
        # 思考结束才开始推进），reasoning 相关代码整体不在本章范围（批准项3）。
        # 精简版直接以 `request.use_structured_output` 为准，等价于非推理模型场景。
        if not (new_token_ids and request.use_structured_output):
            return
        struct_output_request = request.structured_output_request
        assert struct_output_request.grammar is not None
        if not struct_output_request.grammar.accept_tokens(
            request.request_id, new_token_ids
        ):
            # SUBTRACTED: logger.error(...) + 后续终止请求逻辑（scheduler.py:
            # L1372-1379，批准项7可观测性 + 错误处理旁支，非本章讲解焦点）。
            raise RuntimeError(
                f"grammar rejected tokens {new_token_ids} for request "
                f"{request.request_id}"
            )

    @staticmethod
    def _validate_spec_tokens_against_grammar(
        request, spec_token_ids: list[int]
    ) -> list[int]:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1617-1621（validate_tokens 调用点
        # 之一，抽成独立方法便于单测）——投机解码草稿 token 的"不推进的试走"，
        # 与上面 accept_tokens 的"真推进"是同一契约里两个不同语义的方法，二者的
        # 等价性只在投机 token 数以内成立（见 backend_xgrammar.py 的
        # max_rollback_tokens）。
        #
        # SUBTRACTED: 同上，should_advance 门控替换为 use_structured_output。
        if not request.use_structured_output:
            return spec_token_ids
        metadata = request.structured_output_request
        return metadata.grammar.validate_tokens(spec_token_ids)

    @staticmethod
    def _collect_structured_output_request_ids(scheduled_request_ids, requests_by_id):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1224-1246（get_grammar_bitmask 的
        # 请求 id 过滤逻辑）——本章到这里为止："语法对象造好、能 fill_bitmask"。
        #
        # SUBTRACTED: 之后调用 `structured_output_manager.grammar_bitmask(...)` 做
        # 批量装配（批准项4，归下一章）——精简版只演示交棒点在哪，不实现装配本身。
        return [
            req_id
            for req_id in scheduled_request_ids
            if (req := requests_by_id.get(req_id))
            and (
                req.use_structured_output
                and not getattr(req, "is_prefill_chunk", False)
            )
        ]
