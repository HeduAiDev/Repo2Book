# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/generate/api_router.py —— 忠实承载（m18，站 0）：
# register_generate_api_routers 的家族装配面与 init_generate_state 的
# OpenAIServingChat 实例化段逐字。删：responses/anthropic/cohere 三面注册
# 与实例化（delete[13]）、completion/generative_scoring 同构面（m18 一句
# 枚举）、batch（略）、tool_server 与 cohere_format CLI 折叠段。
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.serve.utils.request_logger import RequestLogger
else:
    RequestLogger = object


# SOURCE: vllm/entrypoints/generate/api_router.py:L21-L54 —— register_generate_
# api_routers（must_keep：API 面家族装配处；四面注册删——delete[13] 与 m18
# 同构省略，正文按 m18 一句枚举六面共享 engine_client/OnlineRenderer）
def register_generate_api_routers(app: FastAPI):
    from vllm.entrypoints.openai.chat_completion.api_router import (
        attach_router as register_chat_api_router,
    )

    register_chat_api_router(app)

    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L28-L54 其余五面
    # 注册：responses/anthropic/cohere——delete[13]（Responses API 全目录与
    # anthropic/cohere 面）；completion/generative_scoring——m18 同构面
    # （正文一句枚举，精简树不携带其 serving 文件）。六面共享同一
    # engine_client 与 OnlineRenderer 的事实不受影响。


# SOURCE: vllm/entrypoints/generate/api_router.py:L57-L243 —— init_generate_state
# （消费面：OpenAIServingChat 实例化段逐字；其余平行 serving 对象的实例化
# 段删——delete[9]/[13]/batch 略）
async def init_generate_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple,
):
    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L64-L81
    # AnthropicServingMessages/CohereServingChatV2 的导入与 env 开关段
    # ——delete[13]。
    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L83-L111 tool_server
    # 装配段——MCP 工具服务器域。
    from vllm.entrypoints.chat_utils import load_chat_template

    # SOURCE: vllm/entrypoints/generate/api_router.py:L91 —— OpenAIServingChat
    # 导入位（真实文件在函数体内延迟导入）
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    resolved_chat_template = load_chat_template(args.chat_template)

    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L114-L126
    # --cohere-format CLI 旗标折叠进 default_chat_template_kwargs 的段
    # ——delete[13] cohere 面。
    default_chat_template_kwargs = dict(args.default_chat_template_kwargs or {})

    # Render endpoints are always backed by OnlineRenderer so that
    # /v1/chat/completions/render and /v1/completions/render work on both
    # generate-mode and render-only servers. Created in init_app_state.

    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L131-L151
    # OpenAIServingResponses 实例化——delete[13]。

    # SOURCE: vllm/entrypoints/generate/api_router.py:L152-L172 —— _chat_kwargs
    # 装配段逐字
    _chat_kwargs = dict(
        engine_client=engine_client,
        models=state.openai_serving_models,
        response_role=args.response_role,
        online_renderer=state.online_renderer,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        enable_log_deltas=args.enable_log_deltas,
        enable_per_request_metrics=args.enable_per_request_metrics,
    )
    # SOURCE: vllm/entrypoints/generate/api_router.py:L173-L175
    state.openai_serving_chat = (
        OpenAIServingChat(**_chat_kwargs) if "generate" in supported_tasks else None
    )
    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L176-L180
    # OpenAIServingChatBatch 实例化——batch 面（略）。
    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L181-L194
    # OpenAIServingCompletion 实例化——m18 同构面。
    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L195-L235
    # AnthropicServingMessages / CohereServingChatV2 实例化——delete[13]。
    # SUBTRACTED: vllm/entrypoints/generate/api_router.py:L237-L243
    # ServingGenerativeScoring 实例化——m18 同构面。
