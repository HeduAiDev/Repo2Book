"""ch32 精简版测试 —— 复现真实 vLLM entrypoints 主线的【可观察行为】。

钉住 dossier 记录的真实 vLLM 行为（非测精简版自洽）：
  1. _base_request_id：X-Request-Id 头优先，否则 random_uuid；chat 前缀 'chatcmpl-'。
  2. _check_model：未知模型 → 404 NotFoundError ErrorResponse。
  3. 渲染前 engine.errored → 抛 dead_error（流式 200 前暴露引擎已死）。
  4. 流式 SSE：首块只含 role 空 delta；逐 output 推 delta_text；末块带 finish_reason；
     收尾 'data: [DONE]\\n\\n'；可选 include_usage 末块带 usage。
  5. 非流式 FINAL_ONLY：async for 取末个 RequestOutput 聚合成 ChatCompletionResponse + UsageInfo。
  6. stream generator 内异常 → 写成 SSE error data 帧而非抛出（200 已发）。
  7. router 分流：ErrorResponse/ChatCompletionResponse → JSONResponse；其余 → StreamingResponse(text/event-stream)。
  8. AuthenticationMiddleware：Bearer token sha256 比对，/v1 强制、非 /v1 与 OPTIONS 跳过。
  9. terminate_if_errored：engine.errored and not is_running → server.should_exit（除非 KEEP_ALIVE）。
 10. build_async_engine_client：退出上下文必 shutdown 引擎（finally）。
 11. with_cancellation：handler 先完成则返回其结果。
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL) not in sys.path:
    sys.path.insert(0, str(IMPL))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 _fakes 可 import

from _fakes import FakeEngine, FakeRequest  # noqa: E402


def _make_chat(engine=None):
    from engine_serving import OpenAIServingModels
    from render_serving import OpenAIServingRender
    from chat_serving import OpenAIServingChat
    engine = engine or FakeEngine()
    models = OpenAIServingModels(base_model_names=["companion-model"])
    render = OpenAIServingRender(model_config=engine.model_config,
                                 renderer=engine.renderer)
    return OpenAIServingChat(engine, models, render), engine


# --- 1: request_id ---

def test_base_request_id_prefers_header():
    from engine_serving import OpenAIServing
    from _framework import Request
    req = Request(headers={"X-Request-Id": "abc-123"})
    assert OpenAIServing._base_request_id(req) == "abc-123"


def test_base_request_id_falls_back_to_uuid():
    from engine_serving import OpenAIServing
    from _framework import Request
    rid = OpenAIServing._base_request_id(Request(headers={}))
    assert rid and rid != "abc-123"


def test_chat_request_id_has_chatcmpl_prefix():
    chat, _ = _make_chat()
    resp = asyncio.run(chat.create_chat_completion(FakeRequest(stream=False)))
    assert resp.id.startswith("chatcmpl-")


# --- 2: _check_model 404 ---

def test_check_model_unknown_returns_404():
    chat, _ = _make_chat()
    resp = asyncio.run(chat.create_chat_completion(FakeRequest(model="nope")))
    from messages import ErrorResponse
    assert isinstance(resp, ErrorResponse)
    assert resp.error.code == 404
    assert resp.error.type == "NotFoundError"


# --- 3: render-time engine.errored raises dead_error ---

def test_errored_engine_raises_before_generation():
    dead = ValueError("DEAD")
    chat, _ = _make_chat(FakeEngine(errored=True, dead=dead))
    with pytest.raises(ValueError, match="DEAD"):
        asyncio.run(chat.create_chat_completion(FakeRequest(stream=True)))


# --- 4: streaming SSE shape ---

def _collect(agen):
    async def run():
        return [c async for c in agen]
    return asyncio.run(run())


def test_stream_first_chunk_is_role_only_delta():
    chat, _ = _make_chat()
    gen = asyncio.run(chat.create_chat_completion(FakeRequest(stream=True)))
    chunks = _collect(gen)
    assert chunks[0].startswith("data: ")
    first = json.loads(chunks[0][len("data: "):])
    delta = first["choices"][0]["delta"]
    assert delta["role"] == "assistant"
    assert delta.get("content", "") == ""
    assert first["choices"][0]["finish_reason"] is None


def test_stream_ends_with_done_sentinel():
    chat, _ = _make_chat()
    chunks = _collect(asyncio.run(chat.create_chat_completion(FakeRequest(stream=True))))
    assert chunks[-1] == "data: [DONE]\n\n"


def test_stream_delta_content_and_finish_reason():
    chat, _ = _make_chat(FakeEngine(deltas=("Hel", "lo")))
    chunks = _collect(asyncio.run(chat.create_chat_completion(FakeRequest(stream=True))))
    # 去掉首块 role 与末尾 [DONE]，中间是逐 token delta。
    payloads = [json.loads(c[len("data: "):]) for c in chunks if c != "data: [DONE]\n\n"]
    contents = [p["choices"][0]["delta"].get("content", "") for p in payloads[1:]]
    assert "".join(contents) == "Hello"
    # 末个 delta 块带 finish_reason='stop'
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_stream_include_usage_appends_usage_chunk():
    chat, _ = _make_chat()
    chunks = _collect(asyncio.run(
        chat.create_chat_completion(FakeRequest(stream=True, include_usage=True))))
    # [DONE] 前一块应携带 usage（choices 为空）。
    usage_chunk = json.loads(chunks[-2][len("data: "):])
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens"] >= 1


# --- 5: non-streaming FINAL_ONLY aggregation ---

def test_full_generator_aggregates_final_output():
    chat, _ = _make_chat(FakeEngine(deltas=("Hel", "lo")))
    resp = asyncio.run(chat.create_chat_completion(FakeRequest(stream=False)))
    from messages import ChatCompletionResponse
    assert isinstance(resp, ChatCompletionResponse)
    assert resp.choices[0].message.content == "Hello"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.total_tokens == (
        resp.usage.prompt_tokens + resp.usage.completion_tokens
    )


# --- 6: stream exception becomes error data frame (not raised) ---

def test_stream_generation_error_becomes_data_frame():
    chat, _ = _make_chat()

    async def boom():
        if False:
            yield  # pragma: no cover
        raise RuntimeError("kaboom")

    req = FakeRequest(stream=True)
    gen = chat.chat_completion_stream_generator(
        req, boom(), "chatcmpl-x", "m", [], object(),
        type("M", (), {"final_usage_info": None})(),
    )
    chunks = _collect(gen)
    # 不抛出；倒数第二帧是 error data，末帧是 [DONE]。
    assert chunks[-1] == "data: [DONE]\n\n"
    err = json.loads(chunks[-2][len("data: "):])
    assert "error" in err


# --- 7: router dispatch ---

def test_router_streaming_returns_streaming_response():
    import api_router
    from _framework import Request, FastAPI, StreamingResponse
    chat, _ = _make_chat()
    app = FastAPI()
    app.state.openai_serving_chat = chat
    raw = Request(app=app, headers={})
    handler = api_router.router.routes["/v1/chat/completions"]
    resp = asyncio.run(handler(FakeRequest(stream=True), raw_request=raw))
    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"


def test_router_nonstreaming_returns_json_response():
    import api_router
    from _framework import Request, FastAPI, JSONResponse
    chat, _ = _make_chat()
    app = FastAPI()
    app.state.openai_serving_chat = chat
    raw = Request(app=app, headers={})
    handler = api_router.router.routes["/v1/chat/completions"]
    resp = asyncio.run(handler(FakeRequest(stream=False), raw_request=raw))
    assert isinstance(resp, JSONResponse)
    assert resp.content["object"] == "chat.completion"


def test_router_error_returns_json_with_status():
    import api_router
    from _framework import Request, FastAPI, JSONResponse
    chat, _ = _make_chat()
    app = FastAPI()
    app.state.openai_serving_chat = chat
    raw = Request(app=app, headers={})
    handler = api_router.router.routes["/v1/chat/completions"]
    resp = asyncio.run(handler(FakeRequest(model="nope"), raw_request=raw))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 404


# --- 8: AuthenticationMiddleware ---

def _auth_mw(token="secret"):
    from server_utils import AuthenticationMiddleware
    sentinel = {}

    def downstream(scope, receive, send):
        sentinel["passed"] = True
        return "DOWNSTREAM"

    return AuthenticationMiddleware(downstream, tokens=[token]), sentinel


def test_auth_rejects_missing_bearer_on_v1():
    mw, sentinel = _auth_mw()
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers_obj": {}}
    out = mw(scope, None, None)
    from _framework import JSONResponse
    assert isinstance(out, JSONResponse)
    assert out.status_code == 401
    assert "passed" not in sentinel


def test_auth_accepts_valid_bearer():
    mw, sentinel = _auth_mw("secret")
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers_obj": {"Authorization": "Bearer secret"}}
    assert mw(scope, None, None) == "DOWNSTREAM"
    assert sentinel["passed"]


def test_auth_skips_non_v1_and_options():
    mw, sentinel = _auth_mw()
    scope = {"type": "http", "method": "GET", "path": "/health", "headers_obj": {}}
    assert mw(scope, None, None) == "DOWNSTREAM"
    scope2 = {"type": "http", "method": "OPTIONS", "path": "/v1/chat/completions",
              "headers_obj": {}}
    assert mw(scope2, None, None) == "DOWNSTREAM"


# --- 9: watchdog terminate_if_errored ---

def test_terminate_if_errored_sets_should_exit(monkeypatch):
    import envs
    import launcher
    monkeypatch.setattr(envs, "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", False)
    server = type("S", (), {"should_exit": False})()
    engine = FakeEngine(errored=True)
    # is_running == not errored == False → engine_errored True
    launcher.terminate_if_errored(server, engine)
    assert server.should_exit is True


def test_terminate_no_op_when_keep_alive(monkeypatch):
    import envs
    import launcher
    monkeypatch.setattr(envs, "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", True)
    server = type("S", (), {"should_exit": False})()
    launcher.terminate_if_errored(server, FakeEngine(errored=True))
    assert server.should_exit is False


def test_terminate_no_op_when_healthy(monkeypatch):
    import envs
    import launcher
    monkeypatch.setattr(envs, "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", False)
    server = type("S", (), {"should_exit": False})()
    launcher.terminate_if_errored(server, FakeEngine(errored=False))
    assert server.should_exit is False


# --- 10: build_async_engine_client always shuts engine down ---

def test_build_async_engine_client_shuts_down():
    import api_server

    engine = FakeEngine()
    vllm_config = type("VC", (), {"_engine_client": engine})()
    args = type("A", (), {"vllm_config": vllm_config})()

    async def run():
        async with api_server.build_async_engine_client_from_engine_args(
            api_server.AsyncEngineArgs.from_cli_args(args)
        ) as eng:
            assert eng is engine
        # 退出上下文后 finally 应已 shutdown
    asyncio.run(run())
    assert engine.shutdown_calls == 1


# --- 11: with_cancellation returns handler result ---

def test_with_cancellation_returns_handler_result():
    from api_router import with_cancellation
    from _framework import Request

    @with_cancellation
    async def handler(request, raw_request):
        return "OK"

    raw = Request(headers={})
    assert asyncio.run(handler(object(), raw_request=raw)) == "OK"
