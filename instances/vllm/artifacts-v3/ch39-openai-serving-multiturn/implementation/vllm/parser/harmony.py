# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/parser/harmony.py —— 忠实承载（本章 introduces 主角，m11）：
# _SegmentType 三通道映射表 / Segment / ChunkResult / HarmonyParser 全方法
# 逐字；xgrammar StructuralTag 组装段删（delete[12]：_END_TAG.._ANY_CONTENT
# 常量、_assemble_tag、get_harmony_structural_tag、_params_to_final_content
# ——xgrammar 语法树构造归 ch30 结构化输出域），保留 _adjust_output_format
# 的折叠骨架；ResponsesRequest 注解面删（delete[13]）。
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

from openai_harmony import HarmonyError, Message, Role

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
)
from vllm.entrypoints.openai.parser.harmony_utils import (
    extract_function_from_recipient,
    get_streamable_parser_for_assistant,
    is_function_recipient,
)
from vllm.logger import init_logger
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai_harmony import Message, StreamableParser

# SUBTRACTED: vllm/parser/harmony.py:L13-L27 xgrammar 导入面（StructuralTag/
# Format 族）与 L50-L54 structural_tag_registry 导入面——delete[12]。
# SUBTRACTED: vllm/parser/harmony.py:L44 ResponsesRequest 导入——delete[13]。


logger = init_logger(__name__)


# SOURCE: vllm/parser/harmony.py:L63-L79 —— _SegmentType 三通道映射表逐字
# （must_keep：本章 introduces 的核心符号——analysis→REASONING、final 或无
# recipient 的 commentary→CONTENT、functions.*→TOOL、其余→IGNORE）
class _SegmentType(Enum):
    TOOL = auto()
    REASONING = auto()
    CONTENT = auto()
    IGNORE = auto()

    # SOURCE: vllm/parser/harmony.py:L69-L79 —— from_channel_and_recipient 逐字
    @staticmethod
    def from_channel_and_recipient(
        channel: str | None, recipient: str | None
    ) -> _SegmentType:
        if recipient and is_function_recipient(recipient):
            return _SegmentType.TOOL
        if channel == "analysis":
            return _SegmentType.REASONING
        if channel == "final" or (channel == "commentary" and recipient is None):
            return _SegmentType.CONTENT
        return _SegmentType.IGNORE


# SOURCE: vllm/parser/harmony.py:L82-L86 —— Segment 逐字（must_keep）
class Segment(NamedTuple):
    channel: str | None
    recipient: str | None
    delta: str
    completed_message: Message | None = None


# SOURCE: vllm/parser/harmony.py:L89-L92 —— ChunkResult 逐字（must_keep：
# segments + reasoning_token_count 思考预算计数）
@dataclass
class ChunkResult:
    segments: list[Segment]
    reasoning_token_count: int


# SOURCE: vllm/parser/harmony.py:L95-L387 —— HarmonyParser 逐字（must_keep）
class HarmonyParser(DelegatingParser):
    # SOURCE: vllm/parser/harmony.py:L96-L118
    def __init__(self, tokenizer, tools=None, *args, **kwargs):
        super().__init__(tokenizer, tools, *args, **kwargs)

        if self.reasoning_parser and not isinstance(
            self.reasoning_parser, GptOssReasoningParser
        ):
            raise ValueError(
                "Harmony requires GptOssReasoningParser, "
                f"got {self.reasoning_parser.__class__.__name__}."
            )

        if self.tool_parser and not isinstance(self.tool_parser, GptOssToolParser):
            raise ValueError(
                "Harmony requires GptOssToolParser, "
                f"got {self.tool_parser.__class__.__name__}."
            )

        self._parser: StreamableParser | None = None
        self._next_tool_call_index = 0
        self._num_processed_messages = 0

        # For error recovery
        self._current_message_tokens: list[int] = []

    # SOURCE: vllm/parser/harmony.py:L120-L125 —— _harmony_parser 惰性初始化
    @property
    def _harmony_parser(self) -> StreamableParser:
        """Lazily initializes the Harmony parser."""
        if self._parser is None:
            self._parser = get_streamable_parser_for_assistant()
        return self._parser

    # SOURCE: vllm/parser/harmony.py:L127-L134 —— _poll_completed_message 逐字
    def _poll_completed_message(self) -> Message | None:
        messages = self._harmony_parser.messages
        if len(messages) <= self._num_processed_messages:
            return None
        msg = messages[self._num_processed_messages]
        msg.recipient = self._normalize_recipient(msg.recipient)
        self._num_processed_messages += 1
        return msg

    # SOURCE: vllm/parser/harmony.py:L136-L177 —— flush 逐字（must_keep：
    # 流收口与 HarmonyError 错误恢复——final 通道兜底重解码）
    def flush(self) -> list[Segment]:
        segments: list[Segment] = []
        try:
            self._harmony_parser.process_eos()
            msg = self._poll_completed_message()
        except HarmonyError:
            logger.warning(
                "Harmony parser ended in a non-terminal state; returning the "
                "recovered raw output."
            )

            final_channel = "final"
            text = self.model_tokenizer.decode(self._current_message_tokens)
            segments.append(
                Segment(
                    channel=final_channel,
                    recipient=None,
                    delta=text,
                    completed_message=None,
                )
            )
            msg = Message.from_role_and_content(Role.ASSISTANT, text).with_channel(
                final_channel
            )

        # Reset to the initial assistant-parser state for the next turn.
        self._parser = None
        self._num_processed_messages = 0
        self._current_message_tokens.clear()

        if msg is None:
            return segments

        segments.append(
            Segment(
                channel=msg.channel,
                recipient=msg.recipient,
                delta="",
                completed_message=msg,
            )
        )
        return segments

    # SOURCE: vllm/parser/harmony.py:L179-L235 —— parse 逐字（非流式一次成型）
    def parse(
        self,
        model_output: str,
        request: ChatCompletionRequest,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        """Parse Harmony output from token IDs.

        Tool calls are always extracted regardless of ``enable_auto_tools``.
        Callers must decide whether to surface them.
        """
        result = self.process_chunk(model_output_token_ids)
        flushed_segments = self.flush()
        if flushed_segments:
            result.segments.extend(flushed_segments)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: list[FunctionCall] = []

        for segment in result.segments:
            msg = segment.completed_message
            if msg is None:
                continue
            if msg.author.role != "assistant" or not msg.content:
                continue
            text = msg.content[0].text
            segment_type = _SegmentType.from_channel_and_recipient(
                msg.channel, msg.recipient
            )
            match segment_type:
                case _SegmentType.REASONING if self.reasoning_parser and text:
                    reasoning_parts.append(text)
                case _SegmentType.CONTENT if text:
                    content_parts.append(text)
                case _SegmentType.TOOL if self.tool_parser:
                    recipient = msg.recipient
                    content_type = msg.content_type
                    assert recipient is not None
                    if content_type is not None and "json" not in content_type:
                        arguments = text
                    else:
                        try:
                            arguments = json.dumps(json.loads(text))
                        except json.JSONDecodeError:
                            arguments = text
                    tool_calls.append(
                        FunctionCall(
                            name=extract_function_from_recipient(recipient),
                            arguments=arguments,
                        )
                    )

        reasoning = "\n".join(reasoning_parts) or None
        content = "\n".join(content_parts) or None
        return reasoning, content, tool_calls or None

    # SOURCE: vllm/parser/harmony.py:L237-L329 —— parse_delta 逐字（must_keep：
    # 站 8 harmony 路——按 Segment 分拣，recipient 变化开新 DeltaToolCall）
    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        prev_recipient = self._normalize_recipient(
            self._harmony_parser.current_recipient
        )
        result = self.process_chunk(delta_token_ids)
        if finished:
            flushed_segments = self.flush()
            if flushed_segments:
                result.segments.extend(flushed_segments)
        combined_content = ""
        combined_reasoning = ""
        tool_messages: list[DeltaToolCall] = []

        for segment in result.segments:
            if segment.completed_message is not None:
                prev_recipient = None
                continue

            segment_type = _SegmentType.from_channel_and_recipient(
                segment.channel, segment.recipient
            )
            match segment_type:
                case _SegmentType.REASONING if self.reasoning_parser:
                    combined_reasoning += segment.delta
                case _SegmentType.CONTENT:
                    combined_content += segment.delta
                case _SegmentType.TOOL if self.tool_parser:
                    assert segment.recipient is not None
                    if prev_recipient != segment.recipient:
                        tool_name = extract_function_from_recipient(segment.recipient)
                        tool_messages.append(
                            DeltaToolCall(
                                # HarmonyParser does not use _stream_state;
                                # "random" tool_call_id_type is always used
                                id=make_tool_call_id(),
                                type="function",
                                function=DeltaFunctionCall(
                                    name=tool_name,
                                    arguments=segment.delta,
                                ),
                                index=self._next_tool_call_index,
                            )
                        )
                        self._next_tool_call_index += 1
                        prev_recipient = segment.recipient
                    elif segment.delta:
                        idx = self._next_tool_call_index - 1
                        if tool_messages:
                            tool_msg = tool_messages[-1]
                            assert tool_msg.index == idx
                            fn = tool_msg.function
                            assert fn is not None and fn.arguments is not None
                            fn.arguments += segment.delta
                        else:
                            tool_messages.append(
                                DeltaToolCall(
                                    index=idx,
                                    function=DeltaFunctionCall(arguments=segment.delta),
                                )
                            )

        if finished:
            self._next_tool_call_index = 0

        if not combined_content and not combined_reasoning and not tool_messages:
            return None

        delta_message = DeltaMessage()
        if combined_content:
            delta_message.content = combined_content
        if combined_reasoning:
            delta_message.reasoning = combined_reasoning
        if tool_messages:
            delta_message.tool_calls = tool_messages

        # Suppress reasoning deltas if not requested
        if delta_message and not request.include_reasoning:
            delta_message.reasoning = None

            # If only reasoning was in the message (no content, no tool_calls)
            # skip emitting entirely
            if not delta_message.content and not delta_message.tool_calls:
                return None

        return delta_message

    # SOURCE: vllm/parser/harmony.py:L331-L370 —— process_chunk 逐字（must_keep：
    # 逐 token 喂 StreamableParser 出 Segment）
    def process_chunk(self, token_ids: Sequence[int]) -> ChunkResult:
        if not token_ids:
            return ChunkResult(segments=[], reasoning_token_count=0)

        segments: list[Segment] = []
        reasoning_token_count = 0
        for token_id in token_ids:
            self._harmony_parser.process(token_id)
            channel = self._harmony_parser.current_channel
            recipient = self._normalize_recipient(
                self._harmony_parser.current_recipient
            )
            delta = self._harmony_parser.last_content_delta or ""
            completed_message = self._poll_completed_message()

            if completed_message is not None:
                self._current_message_tokens.clear()
            else:
                self._current_message_tokens.append(token_id)

            if channel == "analysis" or (
                channel == "commentary" and recipient is not None
            ):
                reasoning_token_count += 1

            segments.append(
                Segment(
                    channel=channel,
                    recipient=recipient,
                    delta=delta,
                    completed_message=completed_message,
                )
            )

            # TODO: Optionally merge and suppress empty Segments

        return ChunkResult(
            segments=segments,
            reasoning_token_count=reasoning_token_count,
        )

    # SOURCE: vllm/parser/harmony.py:L372-L376 —— adjust_request 逐字
    # （must_keep：和谐分支折 StructuralTag 的入口）
    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        request = _adjust_output_format(request)
        return super().adjust_request(request)

    # SOURCE: vllm/parser/harmony.py:L378-L387 —— _normalize_recipient 逐字
    @staticmethod
    def _normalize_recipient(recipient: str | None) -> str | None:
        """Remove constrained formats misparsed into recipients by older Harmony."""
        if recipient is None:
            return None

        constrain_index = recipient.find("<|constrain|>")
        if constrain_index == -1:
            return recipient
        return recipient[:constrain_index].rstrip() or None


# SUBTRACTED: vllm/parser/harmony.py:L390-L547 —— harmony StructuralTag 组装
# 段（delete[12]）：_END_TAG/_FINAL_BEGIN/_TOOL_CALL_CHANNELS/
# _FUNCTION_CALL_BEGINS/_JSON_CONTENT/_ANY_CONTENT 常量、_assemble_tag、
# get_harmony_structural_tag（@register_vllm_structural_tag("harmony")）、
# _params_to_final_content——xgrammar StructuralTag 语法树的逐 Tag 构造归
# ch30 结构化输出域；本章只需「约束折进 final 通道」这一层折叠语义（骨架
# 见下方 _adjust_output_format）。


# SOURCE: vllm/parser/harmony.py:L550-L586 —— _adjust_output_format 折叠骨架
# （m15：约束折进 final 通道 StructuralTag、reasoning-aware、清空原
# response_format；折叠构造段删——delete[12]，见函数体 SUBTRACTED 注记）
def _adjust_output_format(
    request: ChatCompletionRequest,
) -> ChatCompletionRequest:
    """Canonicalize request constraints into a reasoning-aware StructuralTag."""
    params = request.extract_structured_outputs()
    if params is None:
        return request

    # SUBTRACTED: vllm/parser/harmony.py:L558-L581 —— 折叠构造段：真实链路为
    # final_content = _params_to_final_content(params)（json_object/json/
    # regex/choice/grammar/structural_tag → xgrammar Format）→ 按
    # isinstance(final_content, JSONSchemaFormat) 选 <|constrain|>json 前缀 →
    # _assemble_tag(allow_analysis=True, allow_commentary=False, content=
    # TagFormat(begin=_FINAL_BEGIN…, content=final_content, end=_END_TAG)) 折出
    # StructuralTag → request.structured_outputs = replace(params, json=None,
    # regex=None, choice=None, grammar=None, json_object=None, structural_tag=
    # json.dumps(structural_tag.model_dump())) 并清空 request.response_format
    # （约束只作用于最终答案通道，analysis 思考不受限）。Format 树构造归
    # ch30（delete[12]）；骨架保留「无约束请求原样返回」的早退语义——带约束
    # 请求在精简版退化为未经折叠的原始 response_format 直通（下游按通用
    # 结构化输出路径约束，仅少 harmony final 通道折叠）。
    return request
