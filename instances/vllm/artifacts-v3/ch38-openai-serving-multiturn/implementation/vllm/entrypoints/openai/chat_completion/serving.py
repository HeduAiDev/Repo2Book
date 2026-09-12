# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/openai/chat_completion/serving.py —— 忠实承载（本章 m4/m5/m6/m7/m14 主角）：
# OpenAIServingChat 编排主体、chat_completion_stream_generator（SSE 主路径）
# 与 chat_completion_full_generator（非流式平行路）逐字；减法项见各
# SUBTRACTED 注记（delete[0]-[11] 对应 dossier subtraction_plan）。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from http import HTTPStatus
from typing import Any, Final

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    make_tool_call_id,
)
from vllm.entrypoints.generate.base.serving import (
    GenerateBaseServing,
    GenerationError,
    clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens, should_include_usage
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import as_list

# SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L10/L13/L15/L17-L18
# io/numpy/pybase64/cast 导入——delete[7]（routed_experts 编码与多模态采集）。
# SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L30/L32
# build_per_request_timing_metrics / format_token_id_placeholder 导入
# ——delete[3] per_request metrics、delete[1] logprobs 组装。
# SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L35-L37/L50
# ChatCompletionLogProb 族与 PerRequestTimingMetrics 导入——delete[1]/[3]。
# SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L62/L64
# MultiModalPlaceholders / Logprob 导入——delete[7]/[1]。
# SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L69
# BeamSearchParams 导入——delete[0] beam 全家。
# SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L71-L85
# _get_mm_token_counts 本体——delete[7]（纯文本请求 mm_placeholders 恒空；
# 删采集点后 mm_token_counts 保持 None 流入两生成器）。

logger = init_logger(__name__)


# SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L88 (_make_prompt_tokens_details)
def _make_prompt_tokens_details(
    enable_prompt_tokens_details: bool,
    num_cached_tokens: int | None,
    num_cache_creation_tokens: int | None,
    mm_token_counts: dict[str, int] | None,
) -> PromptTokenUsageInfo | None:
    """Build ``prompt_tokens_details`` from cached + multimodal token counts."""
    if not enable_prompt_tokens_details:
        return None
    if (
        num_cached_tokens is None
        and num_cache_creation_tokens is None
        and not mm_token_counts
    ):
        return None
    return PromptTokenUsageInfo(
        cached_tokens=num_cached_tokens,
        created_cache_tokens=num_cache_creation_tokens,
        multimodal_tokens=mm_token_counts or None,
    )


# SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L110 (OpenAIServingChat)
class OpenAIServingChat(GenerateBaseServing):
    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L111 (__init__)
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        online_renderer: "OnlineRenderer",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        enable_per_request_metrics: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.online_renderer = online_renderer
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        self.enable_log_deltas = enable_log_deltas

        self.enable_auto_tools: bool = enable_auto_tools
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
            is_harmony=self.model_config.hf_config.model_type == "gpt_oss",
        )
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.enable_per_request_metrics = enable_per_request_metrics
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L180 (_effective_chat_template_kwargs)
    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L192 (render_chat_request)
    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OnlineRenderer, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
            or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        return await self.online_renderer.render_chat(request)

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L219 (create_chat_completion)
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L231-L233
        # _with_kv_transfer_rejection_cleanup 包装——delete[11]（P/D 分离的
        # KV-transfer 拒绝清理；无 kv_transfer_params 的普通请求直接透传）。
        return await self._create_chat_completion(request, raw_request)

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L235 (_create_chat_completion)
    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        parser: Parser | None = None
        if self.parser_cls is not None:
            parser = self.parser_cls(
                tokenizer,
                request.tools,
                chat_template_kwargs=chat_template_kwargs,
                model_config=self.model_config,
            )
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        mm_token_counts: dict[str, int] | None = None
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
            # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L279
            # mm_token_counts = _get_mm_token_counts(engine_input)——
            # delete[7]（删采集点后 mm_token_counts 保持 None 流入两生成器，
            # _make_prompt_tokens_details 四参签名原样保留）。

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L298-L302
            # use_beam_search 的 to_beam_search_params 分支——delete[0]
            # （beam_search 全家；普通采样请求控制流不变，保留下方唯一赋值）。
            sampling_params: SamplingParams
            sampling_params = request.to_sampling_params(
                max_tokens,
                self.default_sampling_params,
            )

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L322-L329
            # isinstance(sampling_params, BeamSearchParams) 的 self.beam_search
            # 发车分支——delete[0]（保留下方采样发车主路）。
            if not request.include_reasoning:
                reasoning_ended = True
            # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L333-L337
            # request._grammar_from_parser 的 Mistral grammar 分支——
            # delete[5]（Mistral 特例：grammar 自带 think? 规则）。
            elif parser is not None and parser.reasoning_parser is not None:
                reasoning_ended = parser.is_reasoning_end(prompt_token_ids or [])
            else:
                reasoning_ended = None

            generator = self.engine_client.generate(
                engine_input,
                sampling_params,
                sub_request_id,
                lora_request=lora_request,
                trace_headers=trace_headers,
                priority=request.priority,
                data_parallel_rank=data_parallel_rank,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs={
                    "chat_template_kwargs": chat_template_kwargs,
                }
                if parser is not None and parser.reasoning_parser is not None
                else None,
            )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                chat_template_kwargs=chat_template_kwargs,
                mm_token_counts=mm_token_counts,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
        )

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L389 (get_chat_request_role)
    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L394 (_create_chat_message)
    def _create_chat_message(self, *args: Any, **kwargs: Any) -> ChatMessage:
        """Construct the response :class:`ChatMessage` for the non-streaming path.

        The full-generator calls this at every construction site so
        subclasses can swap in a specialized :class:`ChatMessage`
        subclass (e.g. :class:`CohereServingChatV2` returning
        :class:`CohereChatMessage`) without duplicating the branchy
        tool-choice / auto-tools logic that decides which fields are
        populated. The default returns a plain :class:`ChatMessage`.
        """
        return ChatMessage(*args, **kwargs)

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L406 (_finalize_response_message)
    def _finalize_response_message(
        self,
        message: ChatMessage,
        *,
        parser: Parser | None,
    ) -> ChatMessage:
        """Subclass hook to enrich a fully-constructed :class:`ChatMessage`.

        Default is a no-op. Subclasses that need to surface parser-side
        extras (e.g. :class:`CohereServingChatV2` reading grounding
        citations off the reasoning parser and populating
        :class:`CohereChatMessage.citations`) override this to inspect
        ``parser`` and mutate/replace ``message``.
        """
        return message

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L422 (chat_completion_stream_generator)
    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        chat_template_kwargs: dict[str, Any] | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        num_cache_creation_tokens = None
        tools_streamed = [False] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        previous_texts = [""] * num_choices

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                parsers: list[Parser | None] = [
                    self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                        model_config=self.model_config,
                    )
                    for _ in range(num_choices)
                ]
            else:
                parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        last_res: RequestOutput | None = None
        try:
            async for res in result_generator:
                last_res = res
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    num_cache_creation_tokens = res.num_cache_creation_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # ``res.prompt`` is the rendered chat-templated prompt
                    prompt_text = res.prompt if request.return_prompt_text else None

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                            prompt_text=prompt_text,
                        )

                        # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L533-L539
                        # include_continuous_usage 的首块 usage——delete[3]
                        # （continuous usage 每块增强）。

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L544-L579
                    # request.echo 的尾消息回显块（含其 continuous usage 位
                    # L570-L575）——delete[2] echo / delete[3] continuous usage。
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    parser = parsers[i]
                    if finish_reason_sent[i]:
                        continue

                    # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L587-L600
                    # request.logprobs 的 _create_chat_logprobs 组装分支——
                    # delete[1]（ch8 域；保留 logprobs=None 占位，字段面不变）。
                    logprobs = None

                    delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    if parser is not None:
                        delta_message = parser.parse_delta(
                            delta_text=delta_text,
                            delta_token_ids=as_list(output.token_ids),
                            request=request,
                            prompt_token_ids=res.prompt_token_ids,
                            finished=output.finish_reason is not None,
                        )
                        if delta_message is not None and delta_message.tool_calls:
                            tools_streamed[i] = True

                    # handle streaming just a content delta (no parsers)
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    # When reasoning is hidden, suppress per-token
                    # metadata (logprobs, token_ids) on every chunk to
                    # prevent leaking reasoning tokens through decoded
                    # token text in logprob entries or raw token IDs.
                    hide_stream_metadata = (
                        not request.include_reasoning and parser is not None
                    )
                    if hide_stream_metadata:
                        logprobs = None

                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if output.finish_reason is None and (
                            not request.return_token_ids or hide_stream_metadata
                        ):
                            continue
                        delta_message = DeltaMessage()

                    # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L659-L684
                    # enable_log_outputs/enable_log_deltas 的流式 delta 日志块
                    # ——delete[4]（观测侧路，不改变请求语义）。

                    include_token_ids = (
                        request.return_token_ids and not hide_stream_metadata
                    )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids) if include_token_ids else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check for error finish reason and abort streaming
                        # finish_reason='error' indicates a retryable error
                        self._raise_if_error(output.finish_reason, request_id)

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids) if include_token_ids else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    # Stamp the fingerprint on terminal chunks only (those with
                    # finish_reason set). When ``include_usage`` is on, the
                    # trailing usage chunk below overrides this as the true
                    # final message.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint

                    # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L751-L758
                    # include_continuous_usage 的每块 usage——delete[3]。

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                final_usage.prompt_tokens_details = _make_prompt_tokens_details(
                    self.enable_prompt_tokens_details,
                    num_cached_tokens,
                    num_cache_creation_tokens,
                    mm_token_counts,
                )

                # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L779-L792
                # enable_per_request_metrics 的流式计时指标——delete[3]。

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L817-L833
            # enable_log_outputs 的流式收尾日志块——delete[4]。

        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L844 (chat_completion_full_generator)
    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        parser: Parser | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        tool_parser_cls = (
            self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
        )
        for output in final_res.outputs:
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            self._raise_if_error(output.finish_reason, request_id)
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L885-L898
            # request.logprobs 的 _create_chat_logprobs 组装分支——delete[1]
            # （ch8 域；保留 logprobs=None 占位，hide_stream_metadata 对
            # logprobs 的置 None 独立逻辑保留）。
            logprobs = None

            if parser is not None:
                reasoning, content, tool_calls = parser.parse(
                    output.text,
                    request,
                    enable_auto_tools=self.enable_auto_tools,
                    model_output_token_ids=token_ids,
                )
                suppress_metadata = not request.include_reasoning and parser is not None
                if not request.include_reasoning:
                    reasoning = None
                if suppress_metadata:
                    logprobs = None
            else:
                reasoning = None
                content = output.text
                tool_calls = []
                suppress_metadata = False

            auto_tools_called = False
            is_named_tool_choice = (
                request.tool_choice is not None
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            )
            is_required_tool_choice = request.tool_choice == "required"

            # All six construction sites route through ``self._create_chat_message``
            # so subclasses can swap in a specialized :class:`ChatMessage`
            # (e.g. the Cohere v2 handler's ``CohereChatMessage``) without
            # having to duplicate this branch logic.
            if (not self.enable_auto_tools or not tool_parser_cls) and (
                not is_named_tool_choice and not is_required_tool_choice
            ):
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            elif is_named_tool_choice or is_required_tool_choice:
                message = self._create_chat_message(
                    role=role,
                    reasoning=reasoning,
                    content=content or "",
                    tool_calls=[
                        ToolCall(id=tc.id or make_tool_call_id(), function=tc)
                        for tc in (tool_calls or [])
                    ],
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and tool_parser_cls
            ):
                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    message = self._create_chat_message(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=[
                            ToolCall(id=tc.id or make_tool_call_id(), function=tc)
                            for tc in tool_calls
                        ],
                    )

                else:
                    message = self._create_chat_message(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            # Subclass hook: enrich the constructed message with any
            # parser-side extras that don't fit through the plain
            # ``(reasoning, content, tool_calls)`` tuple. Base is a no-op;
            # citation-aware handlers use this to surface grounding
            # metadata cached on the reasoning parser.
            message = self._finalize_response_message(message, parser=parser)

            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )

            # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L1007-L1015
            # routed_experts 的 .npy+base64 编码——delete[7]（MoE 路由观测
            # 字段，删后响应语义不变；协议字段面按计划保留）。
            routed_experts_b64 = None

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids)
                    if request.return_token_ids and not suppress_metadata
                    else None
                ),
                routed_experts=routed_experts_b64,
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L1038-L1051
        # request.echo 的尾消息内容前置块——delete[2] echo。

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        usage.prompt_tokens_details = _make_prompt_tokens_details(
            self.enable_prompt_tokens_details,
            final_res.num_cached_tokens,
            final_res.num_cache_creation_tokens,
            mm_token_counts,
        )

        request_metadata.final_usage_info = usage

        # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L1074-L1084
        # enable_per_request_metrics 的非流式计时指标——delete[3]。

        # ``final_res.prompt`` is the rendered chat-templated prompt text
        prompt_text = final_res.prompt if request.return_prompt_text else None

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            prompt_text=prompt_text,
            kv_transfer_params=final_res.kv_transfer_params,
            ec_transfer_params=final_res.ec_transfer_params,
        )

        # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L1106-L1136
        # enable_log_outputs 的非流式收尾日志块——delete[4]。

        return response

    # SUBTRACTED: vllm/entrypoints/openai/chat_completion/serving.py:L1140-L1231
    # _get_top_logprobs / _create_chat_logprobs——delete[1]（logprobs 组装
    # 全家归 ch8；两生成器的调用分支已删，logprobs 字段保留 None 占位）。
