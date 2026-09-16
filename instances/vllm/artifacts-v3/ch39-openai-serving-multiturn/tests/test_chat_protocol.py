# 协议定型（站 5，m3）：to_sampling_params 装配 + 默认值合并 + 校验器。
# 行为基准：vllm/entrypoints/openai/chat_completion/protocol.py:L212-L1030
# （v0.27.1 现核）。
import pytest

from conftest import make_request

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams


class TestToSamplingParams:
    def test_defaults_passthrough(self):
        p = make_request(temperature=0.5, top_p=0.9, max_tokens=32).to_sampling_params(
            64, {}
        )
        assert p.temperature == 0.5
        assert p.top_p == 0.9
        assert p.max_tokens == 64  # get_max_tokens 结果作为入参
        assert p.n == 1

    def test_server_defaults_fill_unset(self):
        # 请求未给 temperature → server 默认（generation_config 差量）填上
        p = make_request().to_sampling_params(64, {"temperature": 0.7})
        assert p.temperature == 0.7

    def test_all_five_default_params_merged(self):
        # repetition_penalty/temperature/top_p/top_k/min_p 五连合并（L651-672）
        defaults = {
            "repetition_penalty": 1.1,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 5,
            "min_p": 0.05,
        }
        p = make_request().to_sampling_params(64, defaults)
        assert p.repetition_penalty == 1.1
        assert p.temperature == 0.7
        assert p.top_p == 0.9
        assert p.top_k == 5
        assert p.min_p == 0.05

    def test_request_overrides_server_defaults(self):
        p = make_request(temperature=0.2).to_sampling_params(64, {"temperature": 0.7})
        assert p.temperature == 0.2

    def test_stop_token_ids_from_request(self):
        p = make_request(stop_token_ids=[7, 9]).to_sampling_params(64, {})
        assert p.stop_token_ids == [7, 9]

    def test_stop_token_ids_merge_dedup_preserving_order(self):
        # server 默认与请求值合并去重（dict.fromkeys 保序，L674-684）
        p = make_request(stop_token_ids=[7, 9]).to_sampling_params(
            64, {"stop_token_ids": [9, 11]}
        )
        assert p.stop_token_ids == [7, 9, 11]

    def test_skip_clone_true(self):
        # 每请求新建，安全免克隆（L732）
        p = make_request().to_sampling_params(64, {})
        assert p.skip_clone is True

    def test_structured_outputs_from_response_format_json_schema(self):
        req = make_request(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "out",
                    "schema": {"type": "object", "properties": {"a": {}}},
                },
            }
        )
        p = req.to_sampling_params(64, {})
        assert p.structured_outputs is not None
        assert p.structured_outputs.json == {"type": "object", "properties": {"a": {}}}

    def test_structured_outputs_json_object(self):
        req = make_request(response_format={"type": "json_object"})
        assert req.extract_structured_outputs() == StructuredOutputsParams(
            json_object=True
        )

    def test_extract_structured_outputs_none(self):
        assert make_request().extract_structured_outputs() is None


class TestRequestValidators:
    def test_tools_default_tool_choice_auto(self):
        # 有 tools 未给 tool_choice → 默认 auto（check_tool_usage，L893-894）
        req = make_request(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
        assert req.tool_choice == "auto"

    def test_empty_tools_rejected(self):
        with pytest.raises(VLLMValidationError):
            make_request(tools=[])

    def test_tool_choice_without_tools_rejected(self):
        with pytest.raises(VLLMValidationError):
            make_request(tool_choice="auto")

    def test_stream_options_requires_stream(self):
        with pytest.raises(VLLMValidationError):
            make_request(
                stream=False, stream_options={"include_usage": True}
            )

    def test_stream_options_with_stream_ok(self):
        req = make_request(stream=True, stream_options={"include_usage": True})
        assert req.stream_options.include_usage is True

    def test_continue_final_message_conflicts_add_generation_prompt(self):
        with pytest.raises(VLLMValidationError):
            make_request(
                continue_final_message=True, add_generation_prompt=True
            )

    def test_reasoning_content_renamed_to_reasoning(self):
        # _normalize_messages_before：deprecated reasoning_content → reasoning
        req = ChatCompletionRequest(
            model="test-model",
            messages=[
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "thinking",
                }
            ],
        )
        assert req.messages[0]["reasoning"] == "thinking"
        assert "reasoning_content" not in req.messages[0]


class TestStreamResponseSerialization:
    def test_exclude_unset_drops_unset_fields(self):
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionResponseStreamChoice,
        )
        from vllm.entrypoints.openai.engine.protocol import DeltaMessage

        chunk = ChatCompletionStreamResponse(
            id="chatcmpl-x",
            created=123,
            model="test-model",
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        data = chunk.model_dump_json(exclude_unset=True)
        # 未显式设的 system_fingerprint/usage/prompt_token_ids 不出现
        assert "system_fingerprint" not in data
        assert "usage" not in data
        assert '"role":"assistant"' in data
