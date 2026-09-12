# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/renderers/base.py —— HOST SEAM（最小承载）：BaseRenderer 是
# 本章的 ch6 边界对象——OnlineRenderer 构造期注入（engine_client.renderer），
# render_chat 交棒后非和谐路径的 preprocess_chat 调 renderer.render_chat_async
# （chat 模板+tokenize 四步流水，ch6 已立，本章黑盒回指）。此处只承载
# 消费的方法面；真实 1109 行含模板解析/tokenize/多模态全链。
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from vllm.renderers.params import ChatParams, TokenizeParams

_T = TypeVar("_T")


# SOURCE: vllm/renderers/base.py:L72 —— BaseRenderer 类位（HOST SEAM：
# 方法面按消费承载）
class BaseRenderer(ABC, Generic[_T]):
    # SUBTRACTED: vllm/renderers/base.py 其余 ~1000 行（模板解析、
    # tokenize、多模态渲染、render/render_cmpl 同步族）——ch6 渲染域。

    # SOURCE: vllm/renderers/base.py:L155 —— get_tokenizer
    def get_tokenizer(self) -> _T: ...

    # SOURCE: vllm/renderers/base.py:L178 —— stat_mm_cache
    def stat_mm_cache(self) -> Any | None:
        return None

    # SOURCE: vllm/renderers/base.py:L237 —— warmup（chat 模板预热）
    def warmup(self, chat_params: ChatParams) -> None: ...

    # SOURCE: vllm/renderers/base.py:L1071 —— render_chat_async（ch6 四步
    # 流水异步入口；preprocess_chat 的交棒点）
    @abstractmethod
    async def render_chat_async(
        self,
        messages_list: list[Any],
        chat_params: ChatParams,
        tok_params: TokenizeParams,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        """Render chat messages to engine inputs (ch6 boundary)."""
        ...
