"""测试 Responses API 多轮有状态会话的真实 vLLM 可观察行为。

验证：construct_input_messages 非 harmony 拼接（滤上轮 system、上轮 output
转 assistant、append 本轮 input）；msg_store 的 list 与 HarmonyContext._messages
共享同一对象 → append_output 后 msg_store 自动含本轮 output；create_responses
据 previous_response_id 取回 prev_response（缺失 404）；两轮闭环。
"""
import pytest

from responses_multiturn import (
    HarmonyContext,
    OpenAIServingResponses,
    ResponseOutputMessage,
    _Content,
    _NotFoundError,
    construct_input_messages,
)


# ---------------------------------------------------------------------------
# construct_input_messages: 非 harmony 拼接
# ---------------------------------------------------------------------------
def test_construct_messages_basic_str_input():
    msgs = construct_input_messages(
        request_instructions="be nice",
        request_input="hello",
    )
    assert msgs == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ]


def test_construct_messages_filters_prev_system_messages():
    prev = [
        {"role": "system", "content": "OLD INSTRUCTIONS"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    msgs = construct_input_messages(
        request_instructions="new instructions",
        request_input="q2",
        prev_msg=prev,
    )
    # 上轮 system 被滤掉，instructions 不跨轮携带
    assert {"role": "system", "content": "OLD INSTRUCTIONS"} not in msgs
    assert msgs[0] == {"role": "system", "content": "new instructions"}
    assert {"role": "user", "content": "q1"} in msgs
    assert {"role": "assistant", "content": "a1"} in msgs
    assert msgs[-1] == {"role": "user", "content": "q2"}


def test_construct_messages_prev_output_becomes_assistant():
    prev_output = [ResponseOutputMessage(content=[_Content("prior answer")])]
    msgs = construct_input_messages(
        request_input="next",
        prev_response_output=prev_output,
    )
    assert {"role": "assistant", "content": "prior answer"} in msgs
    assert msgs[-1] == {"role": "user", "content": "next"}


def test_construct_messages_skips_non_output_message_items():
    # 非 ResponseOutputMessage（如 reasoning）应被跳过
    class Reasoning:
        pass

    msgs = construct_input_messages(
        request_input="x",
        prev_response_output=[Reasoning()],
    )
    assert msgs == [{"role": "user", "content": "x"}]


# ---------------------------------------------------------------------------
# HarmonyContext: _messages 与传入 list 共享同一对象 + append_output extend
# ---------------------------------------------------------------------------
class FakeParser:
    """模拟 harmony streamable parser：process 累积，messages 暴露解出的消息。"""

    def __init__(self, out_messages):
        self._out = out_messages
        self.processed = []

    def process(self, token_id):
        self.processed.append(token_id)

    @property
    def messages(self):
        return self._out


class FakeOutputItem:
    def __init__(self, token_ids, finish_reason="stop"):
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class FakeRequestOutput:
    def __init__(self, token_ids, out_msgs, finish_reason="stop"):
        self.outputs = [FakeOutputItem(token_ids, finish_reason)]
        self.kv_transfer_params = None
        self._out_msgs = out_msgs


def test_harmony_context_shares_message_list():
    shared = [{"role": "user", "content": "q"}]
    ctx = HarmonyContext(shared, available_tools=[])
    assert ctx._messages is shared
    assert ctx.messages is shared


def test_append_output_extends_shared_list():
    shared = [{"role": "user", "content": "q"}]
    ctx = HarmonyContext(shared, available_tools=[])
    out_msgs = [{"role": "assistant", "content": "answer"}]
    ctx.parser = FakeParser(out_msgs)
    output = FakeRequestOutput(token_ids=[1, 2, 3], out_msgs=out_msgs)
    ctx.append_output(output)
    # 本轮 output 已 extend 进共享 list；finish_reason 落地
    assert {"role": "assistant", "content": "answer"} in shared
    assert ctx.finish_reason == "stop"
    assert ctx.parser.processed == [1, 2, 3]


# ---------------------------------------------------------------------------
# OpenAIServingResponses: create_responses 多轮主线
# ---------------------------------------------------------------------------
class Req:
    _counter = 0

    def __init__(
        self,
        input="hi",
        instructions=None,
        previous_response_id=None,
        store=True,
    ):
        Req._counter += 1
        self.request_id = f"req-{Req._counter}"
        self.input = input
        self.instructions = instructions
        self.previous_response_id = previous_response_id
        self.previous_input_messages = None
        self.store = store
        self.tools = []
        self.tool_choice = None


class Resp:
    def __init__(self, rid, output, status="completed"):
        self.id = rid
        self.output = output
        self.status = status


def make_server(use_harmony=False):
    server = OpenAIServingResponses(use_harmony=use_harmony, enable_store=True)

    # 注入生成器/响应构造替身（对应真实 _generate_with_builtin_tools / from_request）
    def make_generator(request, messages, available_tools):
        ctx = HarmonyContext(messages, available_tools)
        out_msgs = [{"role": "assistant", "content": f"answer-to:{request.input}"}]
        ctx.parser = FakeParser(out_msgs)
        gen = [FakeRequestOutput(token_ids=[9], out_msgs=out_msgs)]
        return gen, ctx

    def make_response(request, context):
        # response.id 用 request_id 关联，便于多轮
        return Resp(request.request_id, output=list(context.messages))

    server._make_generator = make_generator
    server._make_response_from_request = make_response
    server._preprocess_chat = lambda m: ("inputs", m)
    return server


def test_create_responses_unknown_prev_id_returns_not_found():
    server = make_server()
    req = Req(previous_response_id="does-not-exist")
    result = server.create_responses(req)
    assert isinstance(result, _NotFoundError)
    assert result.response_id == "does-not-exist"


def test_create_responses_stores_msg_and_response():
    server = make_server()
    req = Req(input="round1")
    resp = server.create_responses(req)
    # store 后 msg_store / response_store 都落库
    assert req.request_id in server.msg_store
    assert resp.id in server.response_store
    # 本轮 output 已通过共享 list 自动留存于 msg_store
    stored_msgs = server.msg_store[req.request_id]
    assert any(
        isinstance(m, dict) and m.get("role") == "assistant" for m in stored_msgs
    )


def test_two_turn_conversation_threads_history():
    server = make_server()
    # 轮1
    req1 = Req(input="my name is Bob")
    resp1 = server.create_responses(req1)
    # 轮2 携 previous_response_id
    req2 = Req(input="what is my name?", previous_response_id=resp1.id)
    resp2 = server.create_responses(req2)
    # 轮2 拼接的 messages（= msg_store[req2]）应含轮1的 user+assistant 历史
    msgs2 = server.msg_store[req2.request_id]
    contents = [m.get("content") for m in msgs2 if isinstance(m, dict)]
    assert "my name is Bob" in contents
    assert "answer-to:my name is Bob" in contents
    assert "what is my name?" in contents


def test_store_disabled_when_enable_store_false():
    server = OpenAIServingResponses(use_harmony=False, enable_store=False)
    server._make_generator = make_server()._make_generator
    server._make_response_from_request = make_server()._make_response_from_request
    server._preprocess_chat = lambda m: ("inputs", m)
    req = Req(input="x", store=True)
    server.create_responses(req)
    # enable_store=False -> store 被隐式关闭，msg_store/response_store 不落库
    assert req.request_id not in server.msg_store
    assert req.store is False


# ---------------------------------------------------------------------------
# harmony 续轮: _construct_input_messages_with_harmony 取 msg_store 拼接
# ---------------------------------------------------------------------------
class HarmonyMsg:
    def __init__(self, channel, role="assistant", text=""):
        self.channel = channel
        self.role = role
        self.text = text


def test_harmony_continue_extends_from_msg_store():
    server = OpenAIServingResponses(use_harmony=True, enable_store=True)
    prev_id = "prev-1"
    prev_msgs = [
        HarmonyMsg("analysis", text="thinking"),
        HarmonyMsg("final", text="prev answer"),
    ]
    server.msg_store[prev_id] = prev_msgs
    prev_response = Resp(prev_id, output=[])

    req = Req(input="follow up", previous_response_id=prev_id)
    server._get_user_message = lambda t: HarmonyMsg("user_input", role="user", text=t)

    messages = server._construct_input_messages_with_harmony(req, prev_response)
    # 续轮历史来自 msg_store[prev_id]，再 append 本轮 user input
    assert any(getattr(m, "text", "") == "prev answer" for m in messages)
    assert messages[-1].role == "user"
    assert messages[-1].text == "follow up"


def test_harmony_slice_reappend_is_noop():
    """源码 FIXME 直言 slice-delete-reappend 是 no-op：消息集合不变。"""
    server = OpenAIServingResponses(use_harmony=True, enable_store=True)
    prev_id = "p"
    prev_msgs = [
        HarmonyMsg("final", text="a"),
        HarmonyMsg("analysis", text="b"),
        HarmonyMsg("final", text="c"),
    ]
    server.msg_store[prev_id] = prev_msgs
    prev_response = Resp(prev_id, output=[])
    req = Req(input="", previous_response_id=prev_id)
    req.previous_input_messages = [object()]  # 让空 input 被跳过
    before = [(m.channel, m.text) for m in prev_msgs]
    messages = server._construct_input_messages_with_harmony(req, prev_response)
    after = [(m.channel, m.text) for m in messages]
    # no-op：所有原消息原样保留（顺序/内容不丢）
    assert set(before) <= set(after)
