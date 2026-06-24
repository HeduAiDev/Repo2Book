"""OpenAIServingChat —— chat/completions handler（精简版，只做减法）。

与真实 vllm/entrypoints/openai/chat_completion/serving.py 同名、同结构、同控制流。本章主线保留：
  * render_chat_request：_check_model + engine.errored 检查 → 委托 OpenAIServingRender.render_chat。
  * create_chat_completion：render → request_id('chatcmpl-...') → SamplingParams → engine_client.generate
                            → 按 request.stream 分流 stream_generator / full_generator。
  * chat_completion_stream_generator：SSE DELTA —— 首块发 role 空 delta，逐 res 逐 output 推 delta_text，
                                      末块 finish_reason，可选 usage，最后 'data: [DONE]\\n\\n'。
  * chat_completion_full_generator：FINAL_ONLY —— async for 聚合到 final_res，组装 ChatCompletionResponse。

把所有 # SUBTRACTED 删回去 ≈ 真实 chat handler 在普通采样（n==1、无 harmony/mistral/beam）下的主干。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from http import HTTPStatus

from messages import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    GenerationError,
    RequestOutput,
    SamplingParams,
    RequestOutputKind,
    UsageInfo,
)
from engine_serving import OpenAIServing


def get_max_tokens(max_model_len, requested, prompt_len, default_params, override):
    # SOURCE: vllm/entrypoints/openai/engine/utils.py:get_max_tokens
    # SUBTRACTED: override / default_sampling_params 的 min 取值细节；主线：请求值或按上下文余量。
    if requested is not None:
        return requested
    return max(max_model_len - prompt_len, 0)


class OpenAIServingChat(OpenAIServing):
    """chat/completions handler 主体。"""

    # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:OpenAIServingChat
    def __init__(self, engine_client, models, openai_serving_render, **kwargs):
        # SOURCE: OpenAIServingChat.__init__
        super().__init__(engine_client, models, **kwargs)
        self.openai_serving_render = openai_serving_render
        self.default_sampling_params: dict = {}
        self.override_max_tokens = None
        self.reasoning_parser_cls = None  # SUBTRACTED: reasoning parser 装配（点到即止）

    def _effective_chat_template_kwargs(self, request) -> dict:
        # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:_effective_chat_template_kwargs
        return request.chat_template_kwargs or {}

    async def render_chat_request(self, request):
        # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L202 render_chat_request
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        return await self.openai_serving_render.render_chat(request)

    async def create_chat_completion(
        self,
        request,
        raw_request=None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L229 create_chat_completion
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        # SUBTRACTED: reasoning_parser 实例化（dossier delete 批准 logprobs/parser 细节）。

        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)
        model_name = self.models.model_name(lora_request)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_input in enumerate(engine_inputs):
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
                len(engine_input.token_ids),
                self.default_sampling_params,
                self.override_max_tokens,
            )

            # SUBTRACTED: use_beam_search 分支（dossier delete 批准）；只保留普通采样。
            sampling_params = request.to_sampling_params(
                max_tokens, self.default_sampling_params,
            )

            self._log_inputs(
                sub_request_id, engine_input,
                params=sampling_params, lora_request=lora_request,
            )
            trace_headers = (
                None if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            # SUBTRACTED: reasoning_ended 预判分支（依赖 reasoning_parser，已删）→ None。
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
                reasoning_parser_kwargs=None,
            )
            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, model_name,
                conversation, tokenizer, request_metadata,
            )

        return await self.chat_completion_full_generator(
            request, result_generator, request_id, model_name,
            conversation, tokenizer, request_metadata,
        )

    async def chat_completion_stream_generator(
        self,
        request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation,
        tokenizer,
        request_metadata,
    ) -> AsyncGenerator[str, None]:
        # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L408 chat_completion_stream_generator
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        num_choices = 1  # SUBTRACTED: n>1 多 choice 扇出（主线 assert len(generators)==1）
        first_iteration = True
        num_prompt_tokens = 0
        previous_num_tokens = [0] * num_choices
        include_usage = bool(
            request.stream_options and request.stream_options.get("include_usage")
        )

        try:
            async for res in result_generator:
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)

                # First chunk: declare the assistant role with an empty delta.
                # Must be the FIRST response (sent inside try so generator
                # exceptions surface as the first chunk).
                if first_iteration:
                    role = self.get_chat_request_role(request)
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role=role, content=""),
                            logprobs=None,
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=request_id, object=chunk_object_type,
                            created=created_time, choices=[choice_data],
                            model=model_name,
                        )
                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"
                    # SUBTRACTED: echo 首块回显分支（非主线）。
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    # SUBTRACTED: tool/reasoning parser、logprobs、harmony 增量构建分支
                    #   （dossier delete 批准）；主线 delta_text = output.text。
                    delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message = DeltaMessage(content=delta_text)
                    previous_num_tokens[i] += len(output.token_ids)

                    choice_data = ChatCompletionResponseStreamChoice(
                        index=i,
                        delta=delta_message,
                        logprobs=None,
                        finish_reason=output.finish_reason,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=request_id, object=chunk_object_type,
                        created=created_time, choices=[choice_data],
                        model=model_name,
                    )
                    # Stamp the fingerprint on terminal chunks only.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # Once the final token is handled, if stream_options.include_usage
            # is sent, send the usage.
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id, object=chunk_object_type, created=created_time,
                    choices=[], model=model_name, usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=sum(previous_num_tokens),
                total_tokens=num_prompt_tokens + sum(previous_num_tokens),
            )

        except GenerationError as e:
            # 200 已发出无法改状态码 → 把错误作为下一个 SSE data 帧推给客户端。
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        except Exception as e:
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all responses are finished.
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation,
        tokenizer,
        request_metadata,
    ) -> ErrorResponse | ChatCompletionResponse:
        # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L1148 chat_completion_full_generator
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
        for output in final_res.outputs:
            self._raise_if_error(output.finish_reason, request_id)
            # SUBTRACTED: logprobs + reasoning/tool parsing 的长 if/elif 链（dossier delete
            #   批准 logprobs/tool/harmony/mistral）；主线把 output.text 直接作为 message.content。
            message = ChatMessage(role=role, content=output.text)
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=None,
                finish_reason=output.finish_reason if output.finish_reason else "stop",
                stop_reason=output.stop_reason,
            )
            choices.append(choice_data)

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(len(o.token_ids) for o in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        request_metadata.final_usage_info = usage

        response = ChatCompletionResponse(
            id=request_id, created=created_time, model=model_name,
            choices=choices, usage=usage,
            system_fingerprint=self.system_fingerprint,
        )
        return response


class RequestResponseMetadata:
    # SOURCE: vllm/entrypoints/openai/protocol.py:RequestResponseMetadata
    def __init__(self, request_id: str):
        # SOURCE: RequestResponseMetadata.__init__
        self.request_id = request_id
        self.final_usage_info: UsageInfo | None = None
