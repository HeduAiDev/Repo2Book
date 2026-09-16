# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/renderers/online_renderer.py —— 忠实承载（站 3/4，m12）：
# OnlineRenderer 构造（use_harmony 判定 + ParserManager 装配）/ warmup /
# render_chat（tool_choice 门控 + 非和谐/和谐二分）/ _make_request_with_
# harmony（harmony 渲染主入口）/ validate_chat_template / preprocess_chat
# 逐字。删：Mistral 序列化三行与 _grammar 判定（delete[5]）、
# _reused_prompt_token_ids decode 复用（delete[11]）、completion 面三方法
# （render_completion/preprocess_completion/preprocess_cmpl——m18 平行面）。
from http import HTTPStatus
from typing import Any

from openai_harmony import Message as OpenAIMessage

from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.parser.harmony_utils import (
    build_harmony_preamble,
    extract_instructions_from_messages,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
)
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import (
    EngineInput,
    tokens_input,
)
from vllm.logger import init_logger
from vllm.parser import Parser, ParserManager
from vllm.renderers import BaseRenderer, ChatParams, merge_kwargs
from vllm.utils.mistral import is_mistral_tokenizer

# SUBTRACTED: vllm/renderers/online_renderer.py:L18-L20 CompletionRequest 导入
# ——completion 面（m18 同构省略）。
# SUBTRACTED: vllm/renderers/online_renderer.py:L28 ResponsesRequest 导入
# ——delete[13]。
# SUBTRACTED: vllm/renderers/online_renderer.py:L40-L43 parse_model_prompt/
# prompt_to_seq 导入——completion 面与 ch6 渲染内部。
# SUBTRACTED: vllm/renderers/online_renderer.py:L44-L45 mt 导入——delete[5]
# Mistral 特例。

logger = init_logger(__name__)


# SUBTRACTED: vllm/renderers/online_renderer.py:L50-L60
# _reused_prompt_token_ids——delete[11]（P/D 分离 decode 侧 token 复用，
# 归 ch36；其两个消费点 _make_request_with_harmony:L226-L231 与
# preprocess_chat:L415-L424 随删）。


# SOURCE: vllm/renderers/online_renderer.py:L63-L106 —— OnlineRenderer 构造逐字
# （use_harmony = model_type=='gpt_oss'；parser 装配）
class OnlineRenderer:
    # SOURCE: vllm/renderers/online_renderer.py:L64-L79
    def __init__(
        self,
        model_config: ModelConfig,
        renderer: BaseRenderer,
        *,
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        log_error_stack: bool = False,
    ) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.request_logger = request_logger

        self.enable_auto_tools = enable_auto_tools
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        self.parser: type[Parser] | None = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=model_config.model,
            is_harmony=self.use_harmony,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: ChatTemplateContentFormatOption = (
            chat_template_content_format
        )
        self.default_chat_template_kwargs: dict[str, Any] = (
            default_chat_template_kwargs or {}
        )
        self.trust_request_chat_template = trust_request_chat_template

        self.log_error_stack = log_error_stack
        self.supports_browsing = False
        self.supports_code_interpreter = False

    # SOURCE: vllm/renderers/online_renderer.py:L108-L115 —— warmup 逐字
    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    # SOURCE: vllm/renderers/online_renderer.py:L117-L218 —— render_chat 逐字
    # （must_keep：站 3 二分分支；Mistral 重序列化三行删——delete[5]）
    async def render_chat(
        self,
        request: ChatCompletionRequest,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """Core preprocessing logic for chat requests (no model/engine check).

        Called directly by render_chat_request and delegated to by
        OpenAIServingChat.render_chat_request after its engine-aware checks.

        Decode-side token reuse (ids forwarded in ``kv_transfer_params``) is
        handled deeper, in ``preprocess_chat`` / ``_make_request_with_harmony``,
        so it skips only templating and tokenization while tool-choice
        validation and ``adjust_request`` still run and the output is
        detokenized (text-out).
        """
        tokenizer = self.renderer.tokenizer

        tool_parser = self.parser.tool_parser_cls if self.parser is not None else None

        # SUBTRACTED: vllm/renderers/online_renderer.py:L138-L143 Mistral
        # tokenizer 的 maybe_serialize_tool_calls/truncate_tool_call_ids/
        # validate_request_params 三行——delete[5]。

        # Check if tool parsing is unavailable (common condition)
        tool_parsing_unavailable = (
            tool_parser is None
            and not is_mistral_tokenizer(tokenizer)
            and not self.use_harmony
        )

        # Validate tool_choice when tool parsing is required but unavailable
        if tool_parsing_unavailable and request.tool_choice not in (
            None,
            "none",
        ):
            if request.tool_choice == "auto" and not self.enable_auto_tools:
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )
            elif request.tool_choice != "auto":
                # "required" or named tool requires tool parser
                if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
                    tool_choice_desc = f'function "{request.tool_choice.function.name}"'
                else:
                    tool_choice_desc = f'"{request.tool_choice}"'
                return self.create_error_response(
                    f"tool_choice={tool_choice_desc} requires "
                    "--tool-call-parser to be set"
                )

        if request.tools is None or (
            request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none
        ):
            tool_dicts = None
        else:
            tool_dicts = [tool.model_dump() for tool in request.tools]

        if not self.use_harmony:
            # Common case.
            error_check_ret = self.validate_chat_template(
                request_chat_template=request.chat_template,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=self.trust_request_chat_template,
            )
            if error_check_ret is not None:
                return error_check_ret

            conversation, engine_inputs = await self.preprocess_chat(
                request,
                request.messages,
                default_template=self.chat_template,
                default_template_content_format=self.chat_template_content_format,
                default_template_kwargs=self.default_chat_template_kwargs,
                tool_dicts=tool_dicts,
                parser=self.parser,
                skip_mm_cache=skip_mm_cache,
            )
        else:
            # For GPT-OSS.
            if self.parser is not None:
                # HarmonyParser doesn't need chat_template_kwargs
                # TODO: Unify adjust_request() call with non-harmony branch
                self.parser(
                    self.renderer.get_tokenizer(),
                    request.tools,
                    model_config=self.model_config,
                ).adjust_request(request=request)

            should_include_tools = tool_dicts is not None
            conversation, engine_inputs = self._make_request_with_harmony(
                request, should_include_tools
            )

        return conversation, engine_inputs

    # SOURCE: vllm/renderers/online_renderer.py:L220-L267 —— _make_request_with_harmony 逐字
    # （must_keep：站 4 前半；_reused_prompt_token_ids 复用分支删——delete[11]；
    # maybe_serialize_tool_calls 删——delete[5]）
    def _make_request_with_harmony(
        self,
        request: ChatCompletionRequest,
        should_include_tools: bool = True,
    ):
        """Build Harmony (GPT-OSS) messages and engine prompt from a chat request."""
        # SUBTRACTED: vllm/renderers/online_renderer.py:L226-L231
        # reuse_ids = _reused_prompt_token_ids(request) 的 decode 侧 token
        # 复用分支——delete[11]（P/D 分离，归 ch36）。

        messages: list[OpenAIMessage] = []

        # SUBTRACTED: vllm/renderers/online_renderer.py:L238
        # _mt.maybe_serialize_tool_calls(request)——delete[5] Mistral 兼容。

        chat_messages = list(request.messages)
        instructions, chat_messages = extract_instructions_from_messages(chat_messages)

        # Add system message.
        # NOTE: In Chat Completion API, browsing is enabled by default
        # if the model supports it. TODO: Support browsing.
        assert not self.supports_browsing
        assert not self.supports_code_interpreter
        if (reasoning_effort := request.reasoning_effort) == "none":
            raise ValueError(f"Harmony does not support {reasoning_effort=}")
        tools = request.tools if should_include_tools else None
        messages.extend(
            build_harmony_preamble(
                instructions=instructions,
                tools=tools,  # type: ignore[arg-type]
                reasoning_effort=reasoning_effort,
                with_custom_tools=should_include_tools,
            )
        )

        # Add remaining conversation messages.
        messages.extend(parse_chat_inputs_to_harmony_messages(chat_messages))

        # Render prompt token ids.
        prompt_token_ids = render_for_completion(messages)
        engine_input = tokens_input(prompt_token_ids, cache_salt=request.cache_salt)

        return messages, [engine_input]

    # SUBTRACTED: vllm/renderers/online_renderer.py:L269-L299 render_completion
    # 与 L331-L377 preprocess_completion/preprocess_cmpl——completion 面三方法
    # （m18：/v1/completions 是无 messages/无 parser 的最简对照面，同构点到
    # 为止）。

    # SOURCE: vllm/renderers/online_renderer.py:L301-L308 —— create_error_response 逐字
    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    # SOURCE: vllm/renderers/online_renderer.py:L310-L329 —— validate_chat_template 逐字
    def validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> ErrorResponse | None:
        """Copied from GenerateBaseServing._validate_chat_template."""
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    # SOURCE: vllm/renderers/online_renderer.py:L379-L477 —— preprocess_chat 逐字
    # （ch6 黑盒回指的交棒点：renderer.render_chat_async 是 ch6 四步流水边界；
    # Mistral tokenize 判定与 grammar-eligible 判定删——delete[5]；
    # _reused_prompt_token_ids 复用分支删——delete[11]；ResponsesRequest
    # isinstance 判定删——delete[13]）
    async def preprocess_chat(
        self,
        request: Any,
        messages: list[Any],
        default_template: str | None,
        default_template_content_format: ChatTemplateContentFormatOption,
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: list[dict[str, Any]] | None = None,
        parser: type[Parser] | None = None,
        *,
        skip_mm_cache: bool = False,
    ) -> tuple[list[ConversationMessage], list[EngineInput]]:
        """Copied from GenerateBaseServing._preprocess_chat."""
        renderer = self.renderer
        mm_config = self.model_config.multimodal_config

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                # SUBTRACTED: vllm/renderers/online_renderer.py:L398-L403
                # tokenize=(is_mistral_tokenizer(...) or ...) 的 Mistral 判定
                # ——delete[5]（删 mistral 项后仅剩 enable_prompt_embeds 支路，
                # 普通请求语义不变）。
                tokenize=self.model_config.enable_prompt_embeds,
            ),
        )

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        # SUBTRACTED: vllm/renderers/online_renderer.py:L415-L424
        # reuse_ids = _reused_prompt_token_ids(request) 的 decode 侧复用分支
        # ——delete[11]（归 ch36）。

        (conversation,), (engine_input,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
            skip_mm_cache=skip_mm_cache,
        )

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        #
        # Exception: Mistral grammar-capable tokenizers always call
        # adjust_request — even for tool_choice="none" — so that the grammar
        # factory can prevent special-token leakage.
        if parser is not None:
            tokenizer = renderer.get_tokenizer()
            tool_parser = parser.tool_parser_cls
            tool_choice = getattr(request, "tool_choice", "none")
            # SUBTRACTED: vllm/renderers/online_renderer.py:L449-L454
            # is_mistral_grammar_eligible 判定——delete[5]。
            should_adjust_request = (
                parser.reasoning_parser_cls is not None
                or tool_choice != "none"
            )
            if should_adjust_request:
                # SUBTRACTED: vllm/renderers/online_renderer.py:L461-L467
                # isinstance(request, ChatCompletionRequest | ResponsesRequest)
                # 的 ResponsesRequest 判定——delete[13]。
                request = parser(
                    tokenizer,
                    request.tools,
                    model_config=self.model_config,
                    chat_template_kwargs=chat_params.chat_template_kwargs,
                ).adjust_request(
                    request=request,
                )

        return conversation, [engine_input]
