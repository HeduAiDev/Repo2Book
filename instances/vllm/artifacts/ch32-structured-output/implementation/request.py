# SOURCE: vllm/v1/request.py
# 只做减法的忠实精简版（pin ad7125a4 / v0.21.0）。真实 Request 类另有 KV cache block /
# 多模态 / LoRA / 优先级等几十个字段（ch05/ch13 范围），与本章「掩码怎么装配」
# 的控制流正交，一并省略（不在 code_spine/must_keep 覆盖范围，不是需要单独批准的
# 删除项——ch31 impl-notes 同一惯例）。本章相对 ch31 的 request.py 额外保留
# num_computed_tokens / num_tokens / num_output_placeholders / spec_token_ids /
# is_prefill_chunk / all_token_ids / prompt_token_ids——它们是 m01 门控判据与
# m05/m12 投机耦合的直接输入。
#
# SUBTRACTED: SPDX 版权头、pooling_params/lora_request/mm_features/cache_salt/
# trace_headers/block_hasher 等与本章无关的构造参数与字段（vllm/v1/request.py:
# L60-121 里除下列保留项外的其余部分）、ConstantList 只读视图包装（本章直接用
# 普通 list，读写语义不变，只是失去"防止直接 append"的防御性包装）。
import enum
import time
from typing import Any

from so_request import StructuredOutputRequest


class RequestStatus(enum.IntEnum):
    # SOURCE: vllm/v1/request.py:L316-321（其余成员——PREEMPTED/FINISHED_* 一族——
    # 属调度章 ch13，非本章 must_keep，省略）
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()


class Request:
    # SOURCE: vllm/v1/request.py:L60-170（__init__，精简到本章相关字段）
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: "list[int] | None" = None,
        sampling_params: Any | None = None,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # SOURCE: vllm/v1/request.py:L87-89
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = time.time()

        # SOURCE: vllm/v1/request.py:L97
        self.status = RequestStatus.WAITING

        if sampling_params is not None:
            # SOURCE: vllm/v1/request.py:L107-112 —— Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

        # SOURCE: vllm/v1/request.py:L121-145
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(prompt_token_ids) if prompt_token_ids else 0
        self._all_token_ids: list[int] = (
            list(prompt_token_ids) if prompt_token_ids is not None else []
        )
        self.all_token_ids = self._all_token_ids

        # SOURCE: vllm/v1/request.py:L140-145 —— Used in async scheduling.
        self.num_output_placeholders = 0
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0

        # SOURCE: vllm/v1/request.py:L161 —— True if this request is scheduled
        # as a non-final prefill chunk.
        self.is_prefill_chunk = False

    @property
    def use_structured_output(self) -> bool:
        # SOURCE: vllm/v1/request.py:L236-238
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/request.py:L240-241
        return len(self._all_token_ids)

    def is_finished(self) -> bool:
        # SOURCE: vllm/v1/request.py:L272-273 —— 真实实现是
        # RequestStatus.is_finished(self.status)，依赖 PREEMPTED/FINISHED_* 等
        # 本章未保留的状态成员（调度章 ch13 范围，已在本章顶部说明省略理由）。
        #
        # SUBTRACTED: 精简版的请求在本章测试里总是活跃请求，恒返回 False——不影响
        # update_draft_token_ids_in_output 的控制流路径（该守卫只在请求已结束时
        # 提前 continue，本章测试场景不触发它）。
        return False
