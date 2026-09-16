# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/v1/engine/output_processor.py —— 忠实承载（本章窗口）：
# RequestOutputCollector 逐字（单槽邮箱，generate() 消费侧的对面——ch7
# 站 12 已立）、OutputProcessor.abort_requests 逐字（must_keep：ABORT
# 终态解阻塞，F5 回收的 output_processor 半跳）。RequestState/
# process_outputs/output_handler 全链（ch7 域）删；RequestState 按
# abort 路径消费面做 HOST SEAM 退化承载。
import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch

from vllm.outputs import (
    CompletionOutput,
    PoolingRequestOutput,
    RequestOutput,
)
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason
from vllm.v1.metrics.stats import LoRARequestStates

if TYPE_CHECKING:
    from vllm.v1.engine.parallel_sampling import ParentRequest

# SOURCE: vllm/v1/engine/output_processor.py:L41-L42
# shared empty CPU tensor used as a placeholder pooling output
EMPTY_CPU_TENSOR = torch.empty(0, device="cpu")


# SOURCE: vllm/v1/engine/output_processor.py:L45-L106 —— RequestOutputCollector 逐字
class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    # SOURCE: vllm/v1/engine/output_processor.py:L54-L60
    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self.output: RequestOutput | PoolingRequestOutput | Exception | None = None
        self.ready = asyncio.Event()

        self._input_stream_task: asyncio.Task | None = None

    # SOURCE: vllm/v1/engine/output_processor.py:L62-L76
    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            self.output.add(output, aggregate=self.aggregate)
        elif isinstance(self.output, PoolingRequestOutput) and isinstance(
            output, PoolingRequestOutput
        ):
            self.output = output

    # SOURCE: vllm/v1/engine/output_processor.py:L78-L86
    async def get(self) -> RequestOutput | PoolingRequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SOURCE: vllm/v1/engine/output_processor.py:L88-L96
    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SOURCE: vllm/v1/engine/output_processor.py:L98-L100
    def close(self):
        if self._input_stream_task is not None:
            self._input_stream_task.cancel()
        self._input_stream_task = None

    # SOURCE: vllm/v1/engine/output_processor.py:L103-L106
    def __del__(self):
        if (task := self._input_stream_task) is not None:
            task.get_loop().call_soon_threadsafe(task.cancel)
            self._input_stream_task = None


# SUBTRACTED: vllm/v1/engine/output_processor.py:L109-L127 OutputProcessorOutput
# 与 StreamingUpdate——output_handler 处理链载体（ch7 域，本章后台循环删）。


# SOURCE: vllm/v1/engine/output_processor.py:L129-L427 —— HOST SEAM：
# RequestState 按 abort 路径消费面退化承载（真实 300 行持有
# detokenizer/logprobs_processor/stream_interval 游标并实现
# make_request_output 全链增量输出——ch7 站 5-13 已立；本章只需 abort
# 终态投递契约：new_token_ids=[] + finish_reason=ABORT → 一条 finished
# RequestOutput 进 queue）
class RequestState:
    # SOURCE: vllm/v1/engine/output_processor.py:L130 —— 构造位（消费面字段）
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: "ParentRequest | None",
        request_index: int,
        lora_request,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        queue,
        **kwargs,
    ):
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.lora_request = lora_request
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.queue = queue
        self.detokenizer = None
        # SUBTRACTED: vllm/v1/engine/output_processor.py:L163-L190 其余
        # 状态位（logprobs_processor/routed_experts_chunks/stream_interval
        # 游标/streaming_input 队列）——ch7 域。

    # SOURCE: vllm/v1/engine/output_processor.py:L276-L340 —— HOST SEAM：
    # make_request_output 退化（abort 终态契约；真实的 FINAL_ONLY 中间零
    # 构造 / stream_interval 节流 / DELTA-CUMULATIVE 分叉归 ch7）
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None:
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        external_req_id = self.external_req_id

        if pooling_output is not None:
            from vllm.outputs import PoolingOutput

            return PoolingRequestOutput(
                request_id=external_req_id,
                outputs=PoolingOutput(data=pooling_output),
                prompt_token_ids=self.prompt_token_ids or [],
                num_cached_tokens=0,
                finished=finished,
            )

        # SUBTRACTED: vllm/v1/engine/output_processor.py:L292-L313
        # stream_interval 节流与 DELTA 偏移重放——ch7 域。
        output = CompletionOutput(
            index=self.request_index,
            text="",
            token_ids=new_token_ids,
            cumulative_logprob=None,
            logprobs=None,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )

        if self.parent_req is None:
            outputs = [output]
        else:
            # SUBTRACTED: vllm/v1/engine/output_processor.py:L326-L332
            # parent_req.get_outputs 的 n>1 扇出收集——ch7 m15 域。
            outputs = [output]

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=self.prompt_token_ids,
            prompt_logprobs=None,
            outputs=outputs,
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
        )


# SOURCE: vllm/v1/engine/output_processor.py:L429 —— OutputProcessor 类位
# （HOST SEAM：只承载 abort_requests 消费面；真实的 process_outputs/
# add_request/update_scheduler_stats/propagate_error 全链归 ch7）
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    # SOURCE: vllm/v1/engine/output_processor.py:L432 —— 构造位（消费面
    # 状态字典）
    def __init__(
        self,
        tokenizer=None,
        vllm_config=None,
        log_stats: bool = False,
        stream_interval: int = 1,
    ):
        self.request_states: dict[str, RequestState] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        self.parent_requests: dict[str, Any] = {}
        self.lora_states = LoRARequestStates(log_stats)
        # SUBTRACTED: vllm/v1/engine/output_processor.py:L432-L470 其余
        # 构造位（scheduler stats/enum 输出枚举/abort 队列）——ch7 域。

    # SOURCE: vllm/v1/engine/output_processor.py:L460-L476 —— add_request 位
    # （HOST SEAM 退化：真实走 RequestState.from_new_request 全装配并处理
    # streaming input 更新；本章按消费面直装配）
    def add_request(
        self,
        request,
        prompt: str | None,
        parent_req: "ParentRequest | None" = None,
        request_index: int = 0,
        queue=None,
    ) -> None:
        request_id = request.request_id
        self.request_states[request_id] = RequestState(
            request_id=request_id,
            external_req_id=request.external_req_id or request_id,
            parent_req=parent_req,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=request.params.output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            queue=queue,
        )
        self.external_req_ids[request.external_req_id or request_id].append(request_id)

    # SOURCE: vllm/v1/engine/output_processor.py:L476-L523 —— abort_requests 逐字
    # （must_keep：外部/内部 id 展开 → 移 RequestState → 投 finish_reason=
    # ABORT 终态解阻塞在等消费者 → 返回待跨进程停算的 id 列表）
    def abort_requests(
        self, request_ids: list[str], internal: bool = False
    ) -> list[str]:
        """Aborts the requests in request_ids.

        Returns the aborted request IDs.

        If a request ID is an external ID (i.e. the one supplied by the
        client), then it may correspond to more than one request, e.g. when
        using parallel sampling with n > 1. In this case, all requests
        associated with that external request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        internal_req_ids = []
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

        request_ids_to_abort = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                self.lora_states.request_finished(request_id, req_state.lora_name)
                request_ids_to_abort.append(request_id)
                # Produce final abort output.
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],
                        # Set pooling_output is not None to
                        # correctly enter the abort pooling branch
                        pooling_output=EMPTY_CPU_TENSOR
                        if req_state.detokenizer is None
                        else None,
                        finish_reason=FinishReason.ABORT,
                        stop_reason=None,
                        kv_transfer_params=None,
                        ec_transfer_params=None,
                    )
                ):
                    req_state.queue.put(request_output)
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                if parent.child_requests:
                    child_reqs = list(parent.child_requests)
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    request_ids_to_abort.extend(child_reqs)
                self.parent_requests.pop(request_id, None)
        return request_ids_to_abort

    # SUBTRACTED: vllm/v1/engine/output_processor.py:L525-L836 其余全链
    # （process_outputs 增量输出/stop-string abort/parent 扇出/
    # propagate_error）——ch7 域。
