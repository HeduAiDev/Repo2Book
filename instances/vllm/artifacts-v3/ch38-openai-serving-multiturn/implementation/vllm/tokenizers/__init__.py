# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/tokenizers/__init__.py —— HOST SEAM（最小承载）：真实包经
# hf/registry 提供完整 tokenizer 栈；本章服务面只把 TokenizerLike 当不透明
# 协议（get_vocab/decode/convert_ids_to_tokens…）传给 parser 构造器。
from typing import Any, Protocol, runtime_checkable


# SOURCE: vllm/tokenizers/protocol.py —— HOST SEAM：TokenizerLike 协议面
# （真实协议在 vllm/tokenizers/protocol.py，此处按消费面最小重述）
@runtime_checkable
class TokenizerLike(Protocol):
    # SOURCE: vllm/tokenizers/protocol.py —— HOST SEAM 协议位
    def get_vocab(self) -> dict[str, int]: ...

    # SOURCE: vllm/tokenizers/protocol.py —— HOST SEAM 协议位
    def decode(self, token_ids: list[int]) -> str: ...


# SUBTRACTED: vllm/tokenizers/__init__.py 的 TokenizerRegistry /
# cached_get_tokenizer / get_tokenizer / maybe_make_thread_pool——tokenizer
# 装配与线程池归 ch6 渲染域；本章经 renderer/engine_client 注入拿现成实例。
