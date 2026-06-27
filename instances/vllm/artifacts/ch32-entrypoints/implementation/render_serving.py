"""OpenAIServingRender —— Renderer 适配层（精简版，只做减法）。

与真实 vllm/entrypoints/serve/render/serving.py 同名、同结构、同控制流。本章主线保留：
  * render_chat：tool_choice 合法性校验 → validate_chat_template → preprocess_chat → (conversation, engine_inputs)。
  * preprocess_chat：build_chat_params / build_tok_params → renderer.render_chat_async（chat template→prompt token_ids）。

render 与 engine 解耦：OpenAIServingRender 可无引擎独立运行（GPU-less render server），
OpenAIServingChat 仅在其上加引擎相关校验（LoRA / 健康）。
"""

from __future__ import annotations

from typing import Any

from messages import ErrorResponse
from engine_serving import create_error_response


class EngineInput:
    """渲染产物：一个 prompt 的 token_ids（vLLM 中是 EngineInput dataclass）。

    # SUBTRACTED: 真实 EngineInput 还携带 multi-modal data / cache_salt 等，本章主线只用 token_ids。
    #   原 vllm/entrypoints/serve/render/_engine_input。
    """

    # SOURCE: vllm/entrypoints/serve/render —— EngineInput（减法版）
    def __init__(self, token_ids: list[int]):
        # SOURCE: EngineInput.__init__ —— 减法版
        self.token_ids = token_ids


class _StubRenderer:
    """renderer 的极小替身：把 messages 渲染成确定的 token_ids。

    # SUBTRACTED: 真实 renderer.render_chat_async 走 HF/Mistral chat template + tokenizer +
    #   多模态处理（与本章『请求穿过 vLLM 到引擎』主线正交，自成体系）。这里给一个确定映射，
    #   仅用于跑通控制流。原 vllm/renderers/ 的 render_chat_async。
    """

    # SOURCE: vllm/renderers/...:BaseRenderer —— stub
    tokenizer = object()

    async def render_chat_async(self, messages_list, chat_params, tok_params,
                                prompt_extras=None, skip_mm_cache=False):
        # SOURCE: BaseRenderer.render_chat_async —— stub：chat messages → (conversation, engine_input)
        (messages,) = messages_list
        # 渲染：把每条消息内容拼成 token_ids（长度=字符数），模拟 chat template→prompt。
        text = " ".join(m.get("content", "") for m in messages)
        token_ids = list(range(len(text)))
        conversation = list(messages)
        return ([conversation], [EngineInput(token_ids)])


class OpenAIServingRender:
    """Renderer 适配层。"""

    # SOURCE: vllm/entrypoints/serve/render/serving.py:OpenAIServingRender
    def __init__(
        self,
        model_config=None,
        renderer=None,
        *,
        chat_template: str | None = None,
        chat_template_content_format: str = "auto",
        trust_request_chat_template: bool = False,
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        default_chat_template_kwargs: dict | None = None,
    ):
        # SOURCE: OpenAIServingRender.__init__
        self.model_config = model_config
        self.renderer = renderer if renderer is not None else _StubRenderer()
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.enable_auto_tools = enable_auto_tools
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.use_harmony = False  # SUBTRACTED: gpt_oss harmony 路径恒关（dossier delete 批准）

    def create_error_response(self, message, **kwargs) -> ErrorResponse:
        # SOURCE: vllm/entrypoints/serve/render/serving.py —— 继承自基类工厂
        return create_error_response(message, **kwargs)

    def validate_chat_template(self, *, request_chat_template, chat_template_kwargs,
                               trust_request_chat_template) -> ErrorResponse | None:
        # SOURCE: vllm/entrypoints/serve/render/serving.py:validate_chat_template
        # SUBTRACTED: 信任校验细节（请求自带模板需 --trust-request-chat-template）。
        #   主线：默认无请求级模板 → 放行。
        if request_chat_template and not trust_request_chat_template:
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set."
            )
        return None

    async def render_chat(
        self,
        request,
        *,
        skip_mm_cache: bool = False,
    ):
        # SOURCE: vllm/entrypoints/serve/render/serving.py:L185 render_chat
        """Core preprocessing logic for chat requests (no model/engine check)."""
        tool_parser = self.tool_parser

        # SUBTRACTED: is_mistral_tokenizer 重序列化分支（Mistral 特例，dossier delete 批准）。
        #   原 serve/render/serving.py:L199-L204。

        # Check if tool parsing is unavailable (common condition)
        tool_parsing_unavailable = tool_parser is None and not self.use_harmony

        # Validate tool_choice when tool parsing is required but unavailable
        if tool_parsing_unavailable and request.tool_choice not in (None, "none"):
            if request.tool_choice == "auto" and not self.enable_auto_tools:
                return self.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )
            elif request.tool_choice != "auto":
                return self.create_error_response(
                    f'tool_choice="{request.tool_choice}" requires '
                    "--tool-call-parser to be set"
                )

        if request.tools is None or (
            request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none
        ):
            tool_dicts = None
        else:
            tool_dicts = [t for t in request.tools]

        # SUBTRACTED: use_harmony (gpt_oss) 分支（dossier delete 批准）；只保留普通路径。
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
            tool_parser=tool_parser,
            skip_mm_cache=skip_mm_cache,
            reasoning_parser=self.reasoning_parser,
        )
        return conversation, engine_inputs

    async def preprocess_chat(
        self,
        request,
        messages,
        *,
        default_template=None,
        default_template_content_format="auto",
        default_template_kwargs=None,
        tool_dicts=None,
        tool_parser=None,
        skip_mm_cache=False,
        reasoning_parser=None,
    ):
        # SOURCE: vllm/entrypoints/serve/render/serving.py:L525 preprocess_chat
        renderer = self.renderer

        # SUBTRACTED: build_tok_params / build_chat_params.with_defaults 的参数合并细节
        #   （media_io_kwargs / mm_processor_kwargs 等多模态项）。主线：构造两组 params 后
        #   交给 renderer.render_chat_async。原 serve/render/serving.py:L551-L563。
        tok_params = {"model_config": self.model_config}
        chat_params = {
            "default_template": default_template,
            "content_format": default_template_content_format,
            "tool_dicts": tool_dicts,
        }

        (conversation,), (engine_input,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={},
            skip_mm_cache=skip_mm_cache,
        )

        # SUBTRACTED: reasoning/tool parser 的 adjust_request 后处理（按 parser 微调请求），
        #   dossier 标注点到即止。原 serve/render/serving.py:L571-L606。
        return conversation, [engine_input]
