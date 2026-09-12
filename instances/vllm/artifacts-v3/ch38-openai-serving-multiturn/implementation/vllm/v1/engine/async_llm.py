# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/v1/engine/async_llm.py —— 忠实承载（本章窗口）：dossier
# scope_note 边界——「AsyncLLM 内部（add_request/双登记/output_handler/
# collector/detokenizer）ch4/ch6/ch7 已讲透，本章自 generate() 的
# AsyncGenerator 消费侧接手」。故本文件保留：generate() 消费循环
# （L596-L655，含 CancelledError/GeneratorExit → abort 的 F5 引擎侧半跳）、
# abort() 两跳（L729-L741）、add_request 的 reasoning_ended 过线盖章
# （L383-L386）。引擎启动/执行器/output_handler 后台循环（ch4/ch5/ch7
# 域）删，构造与 from_vllm_config 按消费面退化承载。
import asyncio
from collections.abc import Iterable, Mapping
from typing import Any

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient, StreamingInput
from vllm.exceptions import VLLMClientError
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.outputs import STREAM_FINISHED, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils.collection_utils import as_list
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/async_llm.py:L60-L69 —— InputStreamError 逐字
class InputStreamError(Exception):
    """Wrapper for errors from the input stream generator.

    This is used to propagate errors from the user's input generator
    without wrapping them in EngineGenerateError.
    """

    def __init__(self, cause: Exception):
        self.cause = cause
        super().__init__(str(cause))


# SOURCE: vllm/v1/engine/async_llm.py:L72 —— AsyncLLM 类位
class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""

    # SOURCE: vllm/v1/engine/async_llm.py:L75-L262 —— HOST SEAM：构造位
    # 退化（真实 190 行：executor 解析、output_handler 启动、多客户端
    # 注册、stat logger 装配——ch4/ch5 域；本章按消费面直挂三个组件位）
    def __init__(
        self,
        vllm_config: VllmConfig,
        log_stats: bool = True,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        *,
        engine_core: EngineCoreClient | None = None,
        input_processor: InputProcessor | None = None,
        output_processor: OutputProcessor | None = None,
        renderer=None,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.log_requests = log_requests
        self.log_stats = log_stats
        self.engine_core: EngineCoreClient = (
            engine_core if engine_core is not None else EngineCoreClient()
        )
        self.input_processor: InputProcessor = (
            input_processor
            if input_processor is not None
            else InputProcessor(
                vllm_config=vllm_config,
                model_config=vllm_config.model_config,
                renderer=renderer,
            )
        )
        self.output_processor: OutputProcessor = (
            output_processor if output_processor is not None else OutputProcessor()
        )
        self.renderer = renderer
        # We start the output_handler on the first call to add_request()
        # (real semantics preserved via the None sentinel used by is_running).
        self.output_handler: asyncio.Task | None = None
        # SUBTRACTED: vllm/v1/engine/async_llm.py:L109-L204 executor 解析 /
        # fault-tolerance / 客户端注册 / reset_mm_cache 预热——ch4 域。

    # SOURCE: vllm/v1/engine/async_llm.py:L205-L231 —— from_vllm_config
    # （HOST SEAM：Executor.get_class 装配链退化为直接构造——ch4 在线面
    # 入口，本章黑盒回指；签名保留）
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context=None,
        stat_loggers=None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            start_engine_loop=start_engine_loop,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
        )

    # SOURCE: vllm/v1/engine/async_llm.py:L234-L260 —— from_engine_args 位
    # （HOST SEAM：create_engine_config 校验链归 ch3 域）
    @classmethod
    def from_engine_args(cls, engine_args, start_engine_loop: bool = True,
                         usage_context=None, stat_loggers=None) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""
        # SUBTRACTED: vllm/v1/engine/async_llm.py:L249-L260
        # engine_args.create_engine_config（ch3 EngineArgs→VllmConfig 全链）。
        raise NotImplementedError(
            "from_engine_args full chain lives in ch3 territory; use "
            "from_vllm_config or direct construction in the reduced build"
        )

    # SOURCE: vllm/v1/engine/async_llm.py:L1086-L1101 —— is_running/is_stopped/
    # errored/dead_error 逐字（errored 是 WC3 渲染前预检与 watchdog 的判定源）
    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    # SOURCE: vllm/v1/engine/async_llm.py:L1091-L1093
    @property
    def is_stopped(self) -> bool:
        return self.errored

    # SOURCE: vllm/v1/engine/async_llm.py:L1095-L1097
    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    # SOURCE: vllm/v1/engine/async_llm.py:L1099-L1101
    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()

    # SOURCE: vllm/v1/engine/async_llm.py:L262-L293 —— HOST SEAM：shutdown
    # 退化（prometheus/renderer 关停与 handler 取消按消费面承载）
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown, cleaning up the background proc and IPC."""
        if renderer := getattr(self, "renderer", None):
            getattr(renderer, "shutdown", lambda: None)()

        if engine_core := getattr(self, "engine_core", None):
            getattr(engine_core, "shutdown", lambda **kw: None)(timeout=timeout)

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            handler.cancel()

    # SOURCE: vllm/v1/engine/async_llm.py:L276-L281 —— get_supported_tasks 位
    # （HOST SEAM：真实向 engine_core 查询并缓存）
    async def get_supported_tasks(self) -> tuple:
        if not hasattr(self, "_supported_tasks"):
            self._supported_tasks = ("generate",)
        return self._supported_tasks

    # SOURCE: vllm/v1/engine/async_llm.py:L283-L418 —— add_request（本章窗口
    # = L336-L403 的已渲染 EngineInput 主路 + L383-L386 reasoning_ended
    # 盖章；EngineCoreRequest 直传支路、流式输入支路、n>1 扇出（ch7 m15）
    # 与 pooling 校验删）
    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request=None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        if self.errored:
            raise EngineDeadError()

        # SUBTRACTED: vllm/v1/engine/async_llm.py:L303-L334 —— pooling
        # kv_sharing 校验与 AsyncGenerator 流式输入支路（流式输入面）。

        # Convert Input --> Request.
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            if isinstance(prompt, dict) and "type" in prompt:
                # Rendered EngineInput; no blocking preprocessing needed.
                request = self.input_processor.process_inputs(
                    request_id,
                    prompt,
                    params,
                    supported_tasks=await self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                )
            else:
                # SUBTRACTED: vllm/v1/engine/async_llm.py:L366-L380 原始
                # prompt 的 process_inputs_async 预处理（ch6 渲染边界）。
                request = self.input_processor.process_inputs(
                    request_id,
                    prompt,
                    params,
                    supported_tasks=await self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                )

        # SOURCE: vllm/v1/engine/async_llm.py:L383-L386 —— 站 6 盖章：
        # serving 层预判的 reasoning_ended 随请求过线（引擎侧据此延迟启用
        # 结构化输出窗口）
        if reasoning_ended is not None:
            request.reasoning_ended = reasoning_ended
        if reasoning_parser_kwargs is not None:
            request.reasoning_parser_kwargs = reasoning_parser_kwargs

        self.input_processor.assign_request_id(request)

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        # SUBTRACTED: vllm/v1/engine/async_llm.py:L393 self._run_output_handler()
        # ——output_handler 后台循环（ch7 域；精简环境由测试注入产出）。

        # Create a new output collector for the request.
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        # SUBTRACTED: vllm/v1/engine/async_llm.py:L401-L418 pooling 分支与
        # n>1 子请求扇出（ParentRequest，ch7 m15 已立）。
        await self._add_request(request, prompt_text, None, 0, queue)
        return queue

    # SOURCE: vllm/v1/engine/async_llm.py:L420-L435 —— _add_request 逐字
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

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    # SUBTRACTED: vllm/v1/engine/async_llm.py:L437-L537 _add_streaming_input_
    # request 与 _validate_streaming_input_sampling_params——流式输入面。

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    # SOURCE: vllm/v1/engine/async_llm.py:L544-L655 —— generate 逐字
    # （站 7/10：消费循环 q.get_nowait() or await q.get() + 断连/取消 →
    # abort 两跳的引擎侧入口；流式输入 prompt 类型注解保留）
    async def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | EngineInput
        | "asyncio.AsyncGenerator[StreamingInput, None]",
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request=None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> "asyncio.AsyncGenerator[RequestOutput, None]":
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
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
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
                if out is not STREAM_FINISHED:
                    yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except VLLMClientError as e:
            if self.log_requests:
                logger.info("Request %s failed (bad request): %s.", request_id, e)
            raise

        # Error from input stream generator - propagate directly.
        except InputStreamError as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed (input error): %s.", request_id, e)
            raise e.cause from e

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                try:
                    s = f"{e.__class__.__name__}: {e}"
                except Exception as e2:
                    s = (
                        f"{e.__class__.__name__}: "
                        "error during printing an exception of class"
                        + e2.__class__.__name__
                    )
                logger.info("Request %s failed due to %s.", request_id, s)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    # SUBTRACTED: vllm/v1/engine/async_llm.py:L657-L727 _run_output_handler
    # 与 output_handler 后台循环——ch7 站 1-4 域（get_output_async →
    # process_outputs 分块 → 单槽邮箱投递）。

    # SOURCE: vllm/v1/engine/async_llm.py:L729-L741 —— abort 两跳逐字
    # （must_keep：①本进程 OutputProcessor.abort_requests 移状态并投
    # ABORT 终态；②engine_core.abort_requests_async 跨进程停算释放 KV）
    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    # SUBTRACTED: vllm/v1/engine/async_llm.py:L743-L1085 其余（notify_kv_
    # transfer_request_rejected / pause_generation / resume_generation /
    # scale_elastic_ep / collective_rpc / check_health / reset_mm_cache…）
    # ——P/D 复用（ch36）、弹性 EP、容错与观测面，各归其章。
