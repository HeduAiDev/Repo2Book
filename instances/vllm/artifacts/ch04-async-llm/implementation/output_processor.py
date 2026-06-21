"""Stage3 后处理与 per-request 队列登记/分发（精简版）。

本章只用 OutputProcessor 的「登记 queue」(add_request) 与「按 req_id 解多路复用分发」
(process_outputs) 两个骨架方法；去 tokenize/logprobs/统计等内部细节留 ch08。

RequestOutputCollector 是异步多路复用的关键（伏笔 f1，ch08 回收）：单槽 + asyncio.Event
的轻量收集器，连接背景 output_handler（生产者）与各 per-request generate() 协程（消费者）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from messages import EngineCoreOutput, EngineCoreRequest


@dataclass
class RequestOutput:
    """generate() yield 给调用方的对象（精简版）。

    真实 vLLM 的 RequestOutput 含 prompt/outputs(CompletionOutput 列表)/logprobs/metrics 等；
    本章只需 request_id + token_ids + finished，以演示「生产者-消费者 + merge」与「finished 判停」。
    """

    # SOURCE: vllm/outputs.py:L85 (class RequestOutput)
    # SUBTRACTED: prompt/prompt_token_ids/encoder_prompt/prompt_logprobs/metrics/lora_request/
    #   num_cached_tokens/CompletionOutput 结构等字段 —— Stage3 输出装配细节留 ch08。
    request_id: str
    token_ids: list[int] = field(default_factory=list)
    finished: bool = False

    # 保留 merge 语义：DELTA(aggregate=True) 累加增量 token；FINAL(aggregate=False) 以最新覆盖。
    # 这是 RequestOutputCollector.put 在「生产者超前」时调用的合帧逻辑（背压的替代）。
    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:  # SOURCE: vllm/outputs.py:L145 (RequestOutput.add)
        """Merge subsequent RequestOutput into this one"""
        self.finished |= next_output.finished
        if aggregate:
            # SUBTRACTED: 真实版按 CompletionOutput.index 配对 merge（n>1 不互相覆盖）+
            #   text/logprobs/cumulative_logprob/finish_reason 累加。原 vllm/outputs.py:L151-L173。
            #   精简版收敛到 n==1，直接累加 token_ids 即可体现 DELTA 聚合。
            self.token_ids.extend(next_output.token_ids)
        else:
            # Replace the output with the new one
            self.token_ids = next_output.token_ids


class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    # SOURCE: vllm/v1/engine/output_processor.py:L45 (class RequestOutputCollector)
    def __init__(self, output_kind: str, request_id: str):
        # SUBTRACTED: 真实版用 RequestOutputKind.DELTA 枚举；精简版用字符串 "DELTA"
        #   表达同一判断。原 vllm/v1/engine/output_processor.py:L55
        self.aggregate = output_kind == "DELTA"
        self.request_id = request_id
        self.output: RequestOutput | Exception | None = None
        self.ready = asyncio.Event()
        # SUBTRACTED: self._input_stream_task —— 流式输入清理用，本章删流式输入。
        #   原 vllm/v1/engine/output_processor.py:L60

    # SOURCE: vllm/v1/engine/output_processor.py:L62 (put)
    def put(self, output: RequestOutput | Exception) -> None:
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
        # SUBTRACTED: PoolingRequestOutput 分支（embedding/pooling 任务）。
        #   原 vllm/v1/engine/output_processor.py:L73-L76

    # SOURCE: vllm/v1/engine/output_processor.py:L78 (get)
    async def get(self) -> RequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SOURCE: vllm/v1/engine/output_processor.py:L88 (get_nowait)
    def get_nowait(self) -> RequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SOURCE: vllm/v1/engine/output_processor.py:L98 (close)
    def close(self):
        # SUBTRACTED: 取消 _input_stream_task（流式输入清理）；本章无流式输入，保留空骨架。
        #   原 vllm/v1/engine/output_processor.py:L99-L101
        pass


@dataclass
class _RequestState:
    """per-request 状态（精简版）。

    真实 vLLM 的 RequestState 还含 detokenizer/logprobs_processor/parent_req/stats 等（ch08）；
    本章只需 request_id 与本请求专属的 queue，以演示「req_id -> queue」的查找表与分发。
    """

    # SOURCE: vllm/v1/engine/output_processor.py (RequestState, 见 from_new_request L522)
    # SUBTRACTED: tokenizer/detokenizer/logprobs_processor/parent_req/request_index/log_stats/
    #   stream_interval/is_prefilling 等 Stage3 细节字段 —— 留 ch08。
    request_id: str
    queue: RequestOutputCollector | None


@dataclass
class OutputProcessorOutput:
    # SOURCE: vllm/v1/engine/output_processor.py (OutputProcessorOutput, 见 process_outputs 返回 L684)
    # SUBTRACTED: 其余统计字段；本章只需 request_outputs（异步路径恒为空）与 reqs_to_abort。
    request_outputs: list[RequestOutput]
    reqs_to_abort: list[str]


class OutputProcessor:
    # SOURCE: vllm/v1/engine/output_processor.py:L413 (class OutputProcessor)
    def __init__(self, tokenizer=None, log_stats: bool = False, stream_interval: int = 1):
        # SUBTRACTED: tokenizer/log_stats/stream_interval/tracing_enabled 的实际使用 +
        #   lora_states/parent_requests/external_req_ids 等映射 —— Stage3/n>1 细节留 ch08。
        #   保留 request_states：req_id -> _RequestState 的查找表，是解多路复用的前提。
        self.request_states: dict[str, _RequestState] = {}

    # SOURCE: vllm/v1/engine/output_processor.py:L508 (add_request)
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req=None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        req_state = self.request_states.get(request_id)
        # SUBTRACTED: 已存在 req_state 的流式更新分支 _update_streaming_request_state。
        #   原 vllm/v1/engine/output_processor.py:L517-L520（流式输入，本章删）

        # SUBTRACTED: RequestState.from_new_request(tokenizer=..., detokenizer 等)；精简版只存
        #   request_id + queue（Stage3 状态机留 ch08）。原 vllm/v1/engine/output_processor.py:L522-L531
        req_state = _RequestState(request_id=request_id, queue=queue)
        # 把本请求 queue 记入查找表 —— 后续 process_outputs 按 req_id 解多路复用的前提。
        self.request_states[request_id] = req_state
        # SUBTRACTED: parent_requests / external_req_ids 映射（n>1 与外部 id 追踪）。
        #   原 vllm/v1/engine/output_processor.py:L533-L537

    # SOURCE: vllm/v1/engine/output_processor.py:L572 (process_outputs)
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats=None,
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        ...
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.
            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.
        """
        request_outputs: list[RequestOutput] = []
        reqs_to_abort: list[str] = []
        # 多路复用解扇出：一个 EngineCore 批 -> 按 req_id 分发回 N 个 per-request 队列。
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # SUBTRACTED: _update_stats_from_output 统计、is_prefilling/prefill_stats 处理、
            #   detokenizer.update + stop-string 检测 + logprobs_processor —— Stage3 去 tokenize
            #   细节留 ch08。原 vllm/v1/engine/output_processor.py:L609-L641
            new_token_ids = engine_core_output.new_token_ids
            finish_reason = engine_core_output.finish_reason

            # 4) Create and handle RequestOutput objects.
            # SUBTRACTED: req_state.make_request_output(...) 的字段装配（detokenized text/logprobs
            #   等）留 ch08；精简版直接构造一个带 finished 标志的 RequestOutput。
            #   原 vllm/v1/engine/output_processor.py:L644-L651
            request_output = RequestOutput(
                request_id=req_id,
                token_ids=list(new_token_ids),
                finished=finish_reason is not None,
            )
            if req_state.queue is not None:
                # AsyncLLM: put into queue for handling by generate().
                req_state.queue.put(request_output)
            else:
                # LLMEngine: return list of RequestOutputs.
                request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                # SUBTRACTED: _finish_request 的统计/parent_req/tracing 收尾留 ch08；
                #   精简版只从查找表移除 req_state。原 vllm/v1/engine/output_processor.py:L662-L682
                self.request_states.pop(req_id, None)

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L446 (abort_requests)
    def abort_requests(self, request_ids, internal: bool = False) -> list[str]:
        # SUBTRACTED: internal/external id 解析与统计；精简版只从查找表移除并回传 id 列表，
        #   供 AsyncLLM.abort 转投 EngineCore。原 vllm/v1/engine/output_processor.py:L446-L506
        aborted: list[str] = []
        for req_id in request_ids:
            if self.request_states.pop(req_id, None) is not None:
                aborted.append(req_id)
        return aborted

    # SOURCE: vllm/v1/engine/output_processor.py (propagate_error)
    def propagate_error(self, e: Exception) -> None:
        # 背景任务故障要传播给所有正等待的 generate()：把异常 put 进每个请求队列，
        # 各 generate 的 get()/get_nowait() 会 raise。
        for req_state in self.request_states.values():
            if req_state.queue is not None:
                req_state.queue.put(e)
