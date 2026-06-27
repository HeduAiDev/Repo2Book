"""本进程侧的归并与登记：每个 child 一份独立 RequestState；

parent_requests / external_req_ids 两张表使输出归并与级联 abort 成为可能。
只保留 parent_req.get_outputs 归并 + external_req_id 改写两条本章关键线。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .parallel_sampling import ParentRequest
from .types import (
    CompletionOutput,
    EngineCoreRequest,
    RequestOutput,
    RequestOutputCollector,
)


# SOURCE: vllm/v1/engine/output_processor.py:L131 class RequestState（本章只保留归并相关）
class RequestState:
    # SOURCE: vllm/v1/engine/output_processor.py:L133 def __init__
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
    ) -> None:
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.queue = queue
        # SUBTRACTED: detokenizer/logprobs_processor/stats/lora_name/prompt 等真实字段
        #            （vllm/v1/engine/output_processor.py:L133-L163）——属 ch08 输出处理，
        #            本章只需 parent_req/request_index/external_req_id/queue。

    @classmethod
    def from_new_request(
        cls,
        request: EngineCoreRequest,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
    ) -> "RequestState":
        # SOURCE: vllm/v1/engine/output_processor.py:L246 @classmethod def from_new_request
        assert request.external_req_id is not None
        # SUBTRACTED: tokenizer/detokenizer/logprobs/stats 构造（vllm/v1/engine/output_processor.py:L246-L293）
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
        )

    # SOURCE: vllm/v1/engine/output_processor.py — make_request_output（含 L340-L354 归并）
    def make_request_output(
        self,
        output: CompletionOutput,
    ) -> RequestOutput | None:
        # SUBTRACTED: 上文 detokenize / DELTA token 偏移 / pooling_output 分支
        #            （vllm/v1/engine/output_processor.py:L303-L336）——属 ch08；这里直接拿到
        #            本 child 的 CompletionOutput，只演示归并与 external_req_id 改写。
        external_req_id = self.external_req_id
        finished = output.finished()

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        # SUBTRACTED: _new_request_output 的 prompt/kv_transfer_params 组装
        #            （vllm/v1/engine/output_processor.py:L355-L389）——只保留 request_id 改回
        #            external_req_id 这一关键不变量。
        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            outputs=outputs,
            finished=finished,
        )


# SOURCE: vllm/v1/engine/output_processor.py:L440 class OutputProcessor
class OutputProcessor:
    # SOURCE: vllm/v1/engine/output_processor.py:L441 def __init__
    def __init__(self) -> None:
        # SUBTRACTED: log_stats/tokenizer/stream_interval/lora_states/tracing_enabled
        #            （vllm/v1/engine/output_processor.py:L449-L456）——非本章主线。
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)

    # SOURCE: vllm/v1/engine/output_processor.py:L491 def abort_requests
    def abort_requests(
        self, request_ids: Iterable[str], internal: bool
    ) -> list[str]:
        """Abort a list of requests.

        In the case of parallel sampling, a request ID may identify a parent
        request, in which case the associated child requests are aborted also.
        """
        internal_req_ids: list[str] = []
        for request_id in request_ids:
            if internal:
                # Internal ID - this may be a parent request
                internal_req_ids.append(request_id)
                # Remove internal ID from the external->internal mapping
                if req_state := self.request_states.get(request_id):
                    external_req_id = req_state.external_req_id
                    internal_ids = self.external_req_ids[external_req_id]
                    internal_ids.remove(request_id)
                    if not internal_ids:
                        del self.external_req_ids[external_req_id]
            elif internal_ids := self.external_req_ids.pop(request_id, []):
                # External ID - abort all requests in the external->internal mapping
                internal_req_ids.extend(internal_ids)

        request_ids_to_abort: list[str] = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                request_ids_to_abort.append(request_id)
                # SUBTRACTED: lora_states.request_finished + 产出 abort 终态 output
                #            （vllm/v1/engine/output_processor.py:L509-L523）——本章只演示
                #            「父→级联未完成 child」的 id 归集。
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                if parent.child_requests:
                    child_reqs = list(parent.child_requests)
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    request_ids_to_abort.extend(child_reqs)
                self.parent_requests.pop(request_id, None)
        return request_ids_to_abort

    # SOURCE: vllm/v1/engine/output_processor.py:L533 def add_request
    def add_request(
        self,
        request: EngineCoreRequest,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        if req_state is not None:
            # SUBTRACTED: _update_streaming_request_state（流式输入复用，
            #            vllm/v1/engine/output_processor.py:L543-L545）——与本章无关。
            return

        req_state = RequestState.from_new_request(
            request=request,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
        )
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)
