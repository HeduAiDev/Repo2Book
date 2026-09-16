# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/renderers/params.py —— 忠实承载（消费面）：merge_kwargs /
# ChatParams / TokenizeParams 是 render_chat 交棒（站 3）与 preprocess_chat
# 默认参数合并的参数载体，逐字；RenderParams 等渲染器配置族删（ch6 域）。
from dataclasses import dataclass, field
from typing import Any, Literal

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

logger = init_logger(__name__)

# SUBTRACTED: vllm/renderers/params.py:L7-L11 EmbedsPrompt/TextPrompt/
# TokensPrompt、merge_media_io_kwargs 与 LazyLoader 导入面——with_defaults 的
# media_io 合并按 HOST SEAM 退化（见 _merge_media_io_kwargs_seam 位）；
# torch 仅 TYPE_CHECKING。


# SOURCE: vllm/renderers/params.py:L28-L40 —— merge_kwargs 逐字
def merge_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, Any]:
    if defaults is None:
        defaults = {}
    if overrides is None:
        overrides = {}

    return defaults | {k: v for k, v in overrides.items() if v not in unset_values}


# SOURCE: vllm/renderers/params.py:L43-L68 —— recursively_merge_kwargs 逐字
def recursively_merge_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, Any]:
    if defaults is None:
        defaults = {}
    if overrides is None:
        overrides = {}

    merged = dict(defaults)

    for k, v in overrides.items():
        if v in unset_values:
            continue

        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = recursively_merge_kwargs(
                merged[k], v, unset_values=unset_values
            )
        else:
            merged[k] = v

    return merged


# SOURCE: vllm/multimodal/media/connector.py —— HOST SEAM：
# merge_media_io_kwargs 退化位（真实按模态分键深合并；本章纯文本主线
# media_io 恒 None，空字典合并语义不变）
def merge_media_io_kwargs(defaults, overrides):
    return {**(defaults or {}), **(overrides or {})}


# SOURCE: vllm/renderers/params.py:L71-L137 —— ChatParams 逐字
# （with_defaults 的 merge_media_io_kwargs 调用改指本文件 HOST SEAM 位）
@dataclass(frozen=True)
class ChatParams:
    """Configuration to control how to parse chat messages."""

    chat_template: str | None = None
    """The chat template to apply."""

    chat_template_content_format: object = "auto"
    """The format of the chat template."""

    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """The kwargs to pass to the chat template."""

    media_io_kwargs: dict[str, dict[str, Any]] | None = None
    """Per-modality kwargs for media I/O (loading/decoding images, videos, etc.)."""

    mm_processor_kwargs: dict[str, Any] | None = None
    """The kwargs to pass to the multi-modal processor."""

    return_assistant_tokens_mask: bool = False
    """Request a per-token assistant mask from apply_chat_template."""

    tool_choice: Any | None = None
    """Request-level tool choice for renderers that need API metadata."""

    response_format: Any | None = None
    """Request-level response format for renderers that need API metadata."""

    # SOURCE: vllm/renderers/params.py:L99-L130
    def with_defaults(
        self,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        default_media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        default_mm_processor_kwargs: dict[str, Any] | None = None,
    ):
        if (
            not default_chat_template_kwargs
            and not default_media_io_kwargs
            and not default_mm_processor_kwargs
        ):
            return self

        return ChatParams(
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                default_chat_template_kwargs,
                self.chat_template_kwargs,
            ),
            media_io_kwargs=merge_media_io_kwargs(
                default_media_io_kwargs,
                self.media_io_kwargs,
            ),
            mm_processor_kwargs=recursively_merge_kwargs(
                default_mm_processor_kwargs,
                self.mm_processor_kwargs,
            ),
            return_assistant_tokens_mask=self.return_assistant_tokens_mask,
            tool_choice=self.tool_choice,
            response_format=self.response_format,
        )

    # SOURCE: vllm/renderers/params.py:L132-L137
    def get_apply_chat_template_kwargs(self) -> dict[str, Any]:
        """The arguments to pass to `tokenizer.apply_chat_template`."""
        return merge_kwargs(
            self.chat_template_kwargs,
            dict(chat_template=self.chat_template, return_dict=False),
        )


# SOURCE: vllm/renderers/params.py:L140-L251 —— TokenizeParams 逐字
@dataclass(frozen=True)
class TokenizeParams:
    """Configuration to control how prompts are tokenized."""

    max_total_tokens: int | None
    """
    Maximum allowed number of input + output tokens.

    Usually, this refers to the model's context length.
    """

    max_output_tokens: int = 0
    """Maximum requested number of output tokens."""

    pad_prompt_tokens: int | None = None
    """
    Number of tokens to pad to:
    - `None` means no padding.
    - `-1` maps to `max_input_tokens`.
    """

    truncate_prompt_tokens: int | None = None
    """
    Number of tokens to keep:
    - `None` means no truncation.
    - `-1` maps to `max_input_tokens`.
    """

    truncation_side: Literal["left", "right"] | None = None
    """
    Which side to truncate from when ``truncate_prompt_tokens`` is active:
    - ``"right"`` keeps the first N tokens (truncate from the end).
    - ``"left"``  keeps the last  N tokens (truncate from the start).
    - ``None``    falls back to the tokenizer default.
    """

    do_lower_case: bool = False
    """Whether to normalize text to lower case before tokenization."""

    add_special_tokens: bool = True
    """Whether to add special tokens."""

    return_token_offsets: bool = False
    """If true, request char-level (start, end) offsets per token. Honored
    only for Fast (Rust-backed) tokenizers with text input and no multimodal
    data; otherwise silently ignored."""

    needs_detokenization: bool = False
    """
    Whether the tokenized prompt needs to contain the original text.

    Not to be confused with `SamplingParams.detokenize` which deals
    with the output generated by the model.
    """

    max_total_tokens_param: str = "max_total_tokens"
    """Override this to edit the message for validation errors."""

    max_output_tokens_param: str = "max_output_tokens"
    """Override this to edit the message for validation errors."""

    truncate_prompt_tokens_param: str = "truncate_prompt_tokens"
    """Override this to edit the message for validation errors."""

    # SOURCE: vllm/renderers/params.py:L204-L210
    @property
    def max_input_tokens(self) -> int | None:
        """Maximum allowed number of input tokens."""
        if self.max_total_tokens is None:
            return None

        return self.max_total_tokens - self.max_output_tokens

    # SOURCE: vllm/renderers/params.py:L212-L251
    def __post_init__(self) -> None:
        max_total_tokens = self.max_total_tokens
        max_output_tokens = self.max_output_tokens
        max_input_tokens = self.max_input_tokens
        truncate_prompt_tokens = self.truncate_prompt_tokens

        if self.truncation_side not in (None, "left", "right"):
            raise VLLMValidationError(
                "`truncation_side` must be either 'left' or 'right'.",
                parameter="truncation_side",
                value=self.truncation_side,
            )

        if (
            max_output_tokens is not None
            and max_total_tokens is not None
            and max_output_tokens > max_total_tokens
        ):
            raise VLLMValidationError(
                f"{self.max_output_tokens_param}={max_output_tokens} "
                f"cannot be greater than "
                f"{self.max_total_tokens_param}={max_total_tokens=}. "
                f"Please request fewer output tokens.",
                parameter=self.max_output_tokens_param,
                value=max_output_tokens,
            )

        if (
            max_input_tokens is not None
            and truncate_prompt_tokens is not None
            and truncate_prompt_tokens > max_input_tokens
        ):
            raise VLLMValidationError(
                f"{self.truncate_prompt_tokens_param}={truncate_prompt_tokens} "
                f"cannot be greater than {self.max_total_tokens_param} - "
                f"{self.max_output_tokens_param} = {max_input_tokens}. "
                f"Please request a smaller truncation size.",
                parameter=self.truncate_prompt_tokens_param,
                value=truncate_prompt_tokens,
            )

    # SUBTRACTED: vllm/renderers/params.py:L253-L310 TokenizeParams.
    # with_kwargs——tokenizer 侧 HF 参数归一（ch6 渲染四步流水内部消费），
    # 本章 preprocess_chat 黑盒回指不触达。

# SUBTRACTED: vllm/renderers/params.py:L311-L496 RenderParams/
# RendererSpecificParams 族——渲染器配置归 ch6 域。
