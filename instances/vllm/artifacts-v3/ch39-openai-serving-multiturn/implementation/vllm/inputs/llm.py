# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/inputs/llm.py —— HOST SEAM（最小承载）：真实 700+ 行定义全套
# prompt schema（含 multimodal UUID/嵌入提示词族）；本章消费面只有
# TokensPrompt/SingletonPrompt/PromptType 类型别名面（EngineClient.generate
# 签名与 preprocess 侧引用）。TokensPrompt/_PromptOptions 逐字，多模态
# 专用 NotRequired 键族按消费面保留。
from typing import Any, TypeAlias, Union

from typing_extensions import NotRequired, TypedDict

from vllm.multimodal.inputs import MultiModalPlaceholders


# SOURCE: vllm/inputs/llm.py:L64 —— _PromptOptions 逐字（多模态键族保留）
class _PromptOptions(TypedDict):
    """
    Additional options available to all
    [`SingletonPrompt`][vllm.inputs.llm.SingletonPrompt] types.
    """

    mm_processor_kwargs: NotRequired[dict[str, Any] | None]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


# SOURCE: vllm/inputs/llm.py:L98 —— TokensPrompt 逐字
class TokensPrompt(_PromptOptions):
    """Schema for a tokenized prompt."""

    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""


# SUBTRACTED: vllm/inputs/llm.py 的 TextPrompt/EmbedsPrompt/
# ExplicitEncoderDecoderPrompt/DataPrompt 与 multimodal UUID 键族——
# 本章消费面只有 TokensPrompt 一族（渲染产物 tokens_input）。

# SOURCE: vllm/inputs/llm.py:L221 —— PromptType 别名（HOST SEAM：精简版
# 退化为 TokensPrompt 一支）
PromptType: TypeAlias = Union[TokensPrompt, str]

# SOURCE: vllm/inputs/llm.py:L213 —— SingletonPrompt 别名（HOST SEAM 同上）
SingletonPrompt: TypeAlias = Union[TokensPrompt, str]

# SOURCE: vllm/inputs/llm.py:L125 —— MultiModalPlaceholders 再导出位
# （HOST SEAM：经 vllm/multimodal/inputs 承载）
_ = MultiModalPlaceholders
