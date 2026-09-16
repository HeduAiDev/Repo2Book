# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/renderers/__init__.py —— 承载 re-export 面（RendererRegistry/
# renderer_from_config 删——引擎侧渲染器装配归 ch4/ch6，本章经
# engine_client.renderer 注入现成实例）。
from .base import BaseRenderer
from .params import ChatParams, TokenizeParams, merge_kwargs

__all__ = [
    "BaseRenderer",
    "ChatParams",
    "TokenizeParams",
    "merge_kwargs",
]
