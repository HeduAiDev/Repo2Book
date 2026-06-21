"""AsyncLLM —— 三段式异步解耦 facade（精简版，只做减法）。

三段：
  Stage1  InputProcessor      —— 本进程，prompt -> EngineCoreRequest（细节留 ch05）
  Stage2  EngineCore          —— 真实是独立进程经 IPC；本章用 in-process stub 替代（IPC 留 ch07）
  Stage3  OutputProcessor     —— 本进程背景 asyncio 任务，EngineCoreOutput -> RequestOutput（细节留 ch08）

与真实 vllm/v1/engine/async_llm.py 同名、同结构、同控制流：__init__ 构造三段并急切/懒启
output_handler；generate() 是异步生成器（消费者）；_add_request 是三段解耦的扇出点；
_run_output_handler 起背景生产者任务。把所有 # SUBTRACTED 分支删回去 ≈ 真实 AsyncLLM。
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator

from engine_core_stub import InProcessEngineCore
from input_processor import InputProcessor
from messages import EngineCoreRequest, SamplingParams
from output_processor import OutputProcessor, RequestOutput, RequestOutputCollector

# SOURCE: vllm/v1/engine/async_llm.py:L654 (envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE)
# 分块常量：output_handler 把一批输出按它切块，块间 await asyncio.sleep(0) 让出事件循环，
# 避免长循环饿死其它协程（尤其 add_request 接客）。真实版从 envs 读环境变量；精简版同样支持
# 环境变量覆盖，默认与 vLLM 一致（128）。
VLLM_V1_OUTPUT_PROC_CHUNK_SIZE = int(os.environ.get("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128"))


class AsyncLLM:
    # SOURCE: vllm/v1/engine/async_llm.py:L70 (class AsyncLLM)
    # SUBTRACTED: 继承 EngineClient 基类（仅抽象接口约定，不影响三段式控制流）。
    #   原 vllm/v1/engine/async_llm.py:L70

    # SOURCE: vllm/v1/engine/async_llm.py:L82 (AsyncLLM.__init__)
    def __init__(
        self,
        vllm_config=None,
        log_requests: bool = False,
        log_stats: bool = False,
    ):
        # SUBTRACTED: maybe_register_config_serialize_by_value()/tracing/custom_stat_loggers/
        #   StatLoggerManager/torch profiler 配置（__init__ L107-L131, L155-L166, L178-L200）——
        #   与三段式编排正交的观测/运维分支。原 vllm/v1/engine/async_llm.py:L107-L200
        self.log_requests = log_requests
        self.log_stats = log_stats
        self.errored = False

        # SUBTRACTED: self.renderer = renderer_from_config(...)（tokenizer 装配，留 ch05）。
        #   原 vllm/v1/engine/async_llm.py:L132

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(vllm_config, renderer=None)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            tokenizer=None,
            log_stats=self.log_stats,
            stream_interval=1,
        )

        # EngineCore (starts the engine in background process).
        # SUBTRACTED: EngineCoreClient.make_async_mp_client(...) 启动独立进程 + DP 扩展参数
        #   (client_addresses/client_count/client_index)。本章用 in-process stub 顶替 Stage2，
        #   接口同名（add_request_async/get_output_async/abort_requests_async），IPC 留 ch07。
        #   原 vllm/v1/engine/async_llm.py:L146-L153
        self.engine_core = InProcessEngineCore()

        # SUBTRACTED: self.logger_manager / self._client_count（观测旁路）。
        #   原 vllm/v1/engine/async_llm.py:L155-L168

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

    # SOURCE: vllm/v1/engine/async_llm.py:L280 (add_request)
    async def add_request(
        self,
        request_id: str,
        prompt,
        params: SamplingParams,
        arrival_time: float | None = None,
        prompt_text: str | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        if self.errored:
            # SUBTRACTED: raise EngineDeadError()；精简版用通用 RuntimeError 表达同一语义。
            #   原 vllm/v1/engine/async_llm.py:L300-L301
            raise RuntimeError("engine dead")

        # SUBTRACTED: is_pooling 判断 + kv_sharing_fast_prefill 的 prompt_logprobs 校验
        #   (L303-L314)：错误/特性分支。
        # SUBTRACTED: isinstance(prompt, AsyncGenerator) 流式输入分支 (L316-L331) 与
        #   isinstance(prompt, EngineCoreRequest) 的 deprecated 直传分支 (L334-L347)：本章
        #   只讲常规一次性 prompt。原 vllm/v1/engine/async_llm.py:L303-L347

        # Convert Input --> Request.
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            supported_tasks=None,
            arrival_time=arrival_time or 0.0,
        )
        # SUBTRACTED: prompt_text = extract_prompt_components(...) 的真实抽取（留 ch05）+
        #   reasoning_ended/reasoning_parser_kwargs 透传 (L361-L366)。
        #   原 vllm/v1/engine/async_llm.py:L361-L366

        self.input_processor.assign_request_id(request)

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        self._run_output_handler()

        # Create a new output collector for the request.
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        params = request.sampling_params

        if params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        # SUBTRACTED: n>1 的 ParentRequest 扇出子请求 (L385-L398)：并行采样是同机制重复，
        #   精简版收敛到 n==1 主路径。原 vllm/v1/engine/async_llm.py:L385-L398
        raise NotImplementedError("n>1 (parallel sampling) 留作并行采样章节")

    # SOURCE: vllm/v1/engine/async_llm.py:L400 (_add_request)
    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req,
        index: int,
        queue: RequestOutputCollector,
    ):
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        # SUBTRACTED: log_requests 日志 (L414-L415)。

    # SOURCE: vllm/v1/engine/async_llm.py:L524 (generate)
    async def generate(
        self,
        prompt,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_text=prompt_text,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                assert isinstance(out, RequestOutput)
                finished = out.finished
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            raise

        # SUBTRACTED: EngineDeadError/ValueError/InputStreamError/通用 Exception 的多个 except
        #   分支 (L598-L632)：错误分类与日志。保留 CancelledError->abort 一条已足够演示
        #   「客户端断开则中止」的生命周期。原 vllm/v1/engine/async_llm.py:L598-L632
        finally:
            if q is not None:
                q.close()

    # SOURCE: vllm/v1/engine/async_llm.py:L637 (_run_output_handler)
    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        # SUBTRACTED: log_stats / self._logger_ref(弹性 EP 扩缩用的 mutable list) / renderer
        #   等统计与日志捕获 (L647-L653)。原 vllm/v1/engine/async_llm.py:L647-L653
        chunk_size = VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        # SOURCE: vllm/v1/engine/async_llm.py:L656 (output_handler 内层协程)
        async def output_handler():
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    # SUBTRACTED: iteration_stats = IterationStats() if ... (L663-L665)：统计旁路。

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, None
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    # SUBTRACTED: update_scheduler_stats + logger_ref[0].record(...) 日志
                    #   (L691-L702)：观测旁路。原 vllm/v1/engine/async_llm.py:L691-L702
            except Exception as e:
                # 背景任务故障要传播给所有正等待的 generate()。
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    # SOURCE: vllm/v1/engine/async_llm.py:L709 (abort)
    async def abort(self, request_id, internal: bool = False) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""
        request_ids = (request_id,) if isinstance(request_id, str) else list(request_id)
        # 双向清理：先在本进程 OutputProcessor 移除，再投递 abort 到 EngineCore。
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)
        # SUBTRACTED: log_requests 日志 (L720-L721)。
