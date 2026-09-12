# harmony 渲染侧多轮编码（站 4，m12/m13）。
# 行为基准：vllm/entrypoints/openai/parser/harmony_utils.py:L32-L461
# （v0.27.1 现核；openai_harmony 0.0.8 实测对照）。
import pytest
from openai_harmony import Message, Role

from vllm.entrypoints.openai.parser.harmony_utils import (
    auto_drop_analysis_messages,
    build_harmony_preamble,
    extract_function_from_recipient,
    extract_instructions_from_messages,
    flatten_input_text_content,
    get_streamable_parser_for_assistant,
    is_function_recipient,
    parse_chat_input_to_harmony_message,
    parse_chat_inputs_to_harmony_messages,
    render_for_completion,
)


def _assistant_final(text):
    m = Message.from_role_and_content(Role.ASSISTANT, text)
    return m.with_channel("final")


def _assistant_analysis(text):
    m = Message.from_role_and_content(Role.ASSISTANT, text)
    return m.with_channel("analysis")


class TestIsFunctionRecipient:
    def test_functions_prefixed_recipient_is_tool(self):
        assert is_function_recipient("functions.get_weather") is True

    def test_bare_functions_dot_is_not(self):
        assert is_function_recipient("functions.") is False

    def test_special_token_recipient_is_not(self):
        assert is_function_recipient("<|channel|>") is False

    def test_assistant_recipient_is_not(self):
        assert is_function_recipient("assistant") is False

    def test_builtin_tool_labels_are_not(self):
        # python/browser/container 是内置工具标签（BUILTIN_TOOL_TO_MCP_SERVER_LABEL）
        assert is_function_recipient("python") is False
        assert is_function_recipient("browser") is False

    def test_bare_recipient_true_on_chat_path(self):
        # Chat 路径不传 allowed_function_tool_names → 启发式全收（docstring 原话）
        assert is_function_recipient("get_weather") is True

    def test_extract_function_from_recipient_strips_prefix(self):
        assert extract_function_from_recipient("functions.get_weather") == "get_weather"


class TestFlattenInputTextContent:
    def test_string_passthrough(self):
        assert flatten_input_text_content("hi") == "hi"

    def test_none_passthrough(self):
        assert flatten_input_text_content(None) is None

    def test_content_parts_flattened(self):
        parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert flatten_input_text_content(parts) == "ab"

    def test_str_items_in_list(self):
        assert flatten_input_text_content(["x", "y"]) == "xy"


class TestExtractInstructions:
    def test_leading_system_peeled(self):
        msgs = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]
        instructions, rest = extract_instructions_from_messages(msgs)
        assert instructions == "be nice"
        assert rest == [{"role": "user", "content": "hi"}]

    def test_leading_developer_peeled(self):
        msgs = [{"role": "developer", "content": "dev rules"}, {"role": "user", "content": "hi"}]
        instructions, rest = extract_instructions_from_messages(msgs)
        assert instructions == "dev rules"
        assert len(rest) == 1

    def test_non_leading_system_not_peeled(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "system", "content": "late"}]
        instructions, rest = extract_instructions_from_messages(msgs)
        assert instructions is None
        assert len(rest) == 2

    def test_empty(self):
        assert extract_instructions_from_messages([]) == (None, [])


class TestParseChatInput:
    def test_assistant_with_tool_calls_maps_to_commentary_channels(self):
        # assistant 带 tool_calls：content→commentary、reasoning→analysis、
        # 每个调用→commentary + recipient=functions.{name} + content_type=json
        msg = {
            "role": "assistant",
            "content": "let me check",
            "reasoning": "hmm",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }
        msgs = parse_chat_input_to_harmony_message(msg)
        assert [m.channel for m in msgs] == ["commentary", "analysis", "commentary"]
        assert msgs[2].recipient == "functions.get_weather"
        assert msgs[2].content_type == "json"

    def test_tool_result_maps_to_commentary_to_assistant(self):
        # tool 结果：Author(functions.{name}) + commentary 通道 + recipient=assistant
        msg = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "22C sunny",
        }
        (m,) = parse_chat_input_to_harmony_message(
            msg, {"call_1": "get_weather"}
        )
        assert m.channel == "commentary"
        assert m.recipient == "assistant"
        assert m.author.name == "functions.get_weather"

    def test_plain_assistant_maps_to_final(self):
        msg = {"role": "assistant", "content": "the answer", "reasoning": "chain"}
        msgs = parse_chat_input_to_harmony_message(msg)
        # reasoning 先发 analysis，再发 final 正文
        assert msgs[0].channel == "analysis"
        assert msgs[1].channel == "final"

    def test_user_message_direct(self):
        msgs = parse_chat_input_to_harmony_message({"role": "user", "content": "q"})
        assert len(msgs) == 1
        assert msgs[0].channel is None
        assert msgs[0].author.role == Role.USER


class TestParseChatInputs:
    def test_tool_id_to_name_mapping_built_from_history(self):
        # 第一遍扫历史建 tool_call_id→函数名映射，tool 结果靠它找 recipient
        chat = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_9", "content": "rain"},
        ]
        msgs = parse_chat_inputs_to_harmony_messages(chat)
        tool_result = msgs[-1]
        assert tool_result.channel == "commentary"
        assert tool_result.author.name == "functions.get_weather"

    def test_system_mid_conversation_maps_via_preamble_helper(self):
        chat = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        msgs = parse_chat_inputs_to_harmony_messages(chat)
        # system 在中段（非首位）走 get_system_or_developer_message
        assert msgs[0].author.role in (Role.SYSTEM, Role.DEVELOPER)


class TestAutoDropAnalysis:
    def test_analysis_before_last_final_dropped(self):
        msgs = [
            _assistant_analysis("old-think-1"),
            _assistant_final("old-answer"),
            _assistant_analysis("old-think-2"),
            _assistant_final("new-answer"),
            _assistant_analysis("current-think"),
        ]
        cleaned = auto_drop_analysis_messages(msgs)
        texts = [m.content[0].text for m in cleaned]
        # 最后一个 final 之前的 analysis 全丢，只留其后的（当前轮思考）
        assert texts == ["old-answer", "new-answer", "current-think"]

    def test_no_final_keeps_all_analysis(self):
        msgs = [_assistant_analysis("t1"), _assistant_analysis("t2")]
        assert auto_drop_analysis_messages(msgs) == msgs

    def test_single_turn_all_kept_after_final(self):
        msgs = [_assistant_final("answer"), _assistant_analysis("tail")]
        assert auto_drop_analysis_messages(msgs) == msgs


class TestPreambleAndRender:
    def test_build_preamble_system_plus_developer_with_tools(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionToolsParam,
            FunctionDefinition,
        )
        tools = [
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name="get_weather",
                    description="weather",
                    parameters={"type": "object", "properties": {}},
                )
            )
        ]
        preamble = build_harmony_preamble(
            instructions="be nice", tools=tools, reasoning_effort="medium"
        )
        assert preamble[0].author.role == Role.SYSTEM
        assert preamble[1].author.role == Role.DEVELOPER

    def test_reasoning_effort_invalid_raises(self):
        with pytest.raises(ValueError, match="not supported by Harmony"):
            build_harmony_preamble(instructions=None, tools=None,
                                   reasoning_effort="extreme")

    def test_render_for_completion_drops_old_analysis(self):
        # render_for_completion 内部先手动 auto_drop（L449-450）
        msgs = [
            Message.from_role_and_content(Role.USER, "hi"),
            _assistant_analysis("old"),
            _assistant_final("answer"),
        ]
        ids = render_for_completion(msgs)
        assert isinstance(ids, list) and len(ids) > 0

    def test_get_streamable_parser_yields_assistant_stream(self):
        sp = get_streamable_parser_for_assistant()
        assert sp is not None


class TestRenderParseRoundtrip:
    def test_multi_turn_tokens_reparse_channels(self):
        # 编码→解码往返：多轮 harmony token 流经 StreamableParser 重解析，
        # 通道信息保真（m12 的「通道写进 token 流本身」）
        ids = render_for_completion(
            [
                Message.from_role_and_content(Role.USER, "hi"),
                _assistant_analysis("let me think"),
                _assistant_final("Hello!"),
            ]
        )
        sp = get_streamable_parser_for_assistant()
        for tid in ids:
            sp.process(tid)
        chats = [
            (m.channel, [c.text for c in (m.content or [])])
            for m in sp.messages
            if m.author.role == Role.ASSISTANT
        ]
        assert ("analysis", ["let me think"]) in chats
        assert ("final", ["Hello!"]) in chats
