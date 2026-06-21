# 只做减法的忠实精简版 —— 镜像 vllm/v1/engine/output_processor.py 的 logprobs 取用切片
# （pin f3fef123）。本章只保留 logprobs 装配链路的下游接口示意。
#
# SUBTRACTED: RequestOutputCollector 队列合并、abort_requests、do_tracing/_update_stats_*
#             统计、StreamingUpdate、parallel sampling 合并（output_processor.py 大部）——
#             属其它章节的编排逻辑，删去不影响 logprobs 正确性。
# SUBTRACTED: RequestState 其余字段/方法、detokenizer 增量去 token、_new_request_output 的
#             prompt 取用细节 —— 保留 _new_completion_output 中 sample logprobs/cumulative
#             进 CompletionOutput 的 DELTA 切尾这一处作为下游接口示意。
from dataclasses import dataclass
from enum import Enum

from logprobs import SampleLogprobs
from logprobs_processor import LogprobsProcessor


class RequestOutputKind(Enum):
    # SOURCE: vllm/sampling_params.py(RequestOutputKind) —— 仅保留本章用到的 DELTA/FINAL。
    # SUBTRACTED: CUMULATIVE 等其余取值与序列化字段，本章只区分 DELTA / 非 DELTA。
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


@dataclass
class CompletionOutput:
    # SOURCE: vllm/outputs.py(CompletionOutput) —— 仅保留本章解读涉及的 logprobs 相关字段。
    # SUBTRACTED: index/finish_reason/stop_reason/routed_experts 等非 logprobs 字段的完整
    #             语义（属输出汇编章节），此处留作占位以示装配落点。
    text: str
    token_ids: list[int]
    logprobs: SampleLogprobs | None
    cumulative_logprob: float | None


class RequestState:
    # SOURCE: vllm/v1/engine/output_processor.py(RequestState) —— 只保留 logprobs 取用所需字段。
    def __init__(
        self,
        logprobs_processor: LogprobsProcessor,
        output_kind: RequestOutputKind,
        detokenizer,
        request_index: int = 0,
    ) -> None:
        self.logprobs_processor = logprobs_processor
        self.output_kind = output_kind
        self.detokenizer = detokenizer
        self.request_index = request_index

    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason=None,
        stop_reason=None,
        routed_experts=None,
    ) -> CompletionOutput:
        # SOURCE: vllm/v1/engine/output_processor.py:L376-L407
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            logprobs = logprobs[-len(token_ids):]

        return CompletionOutput(
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
        )
