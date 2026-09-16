# 断连竞速（F5 HTTP 端）+ 四方取 min 输出预算 + usage 两值解析。
# 行为基准：vllm/entrypoints/serve/utils/api_utils.py:L37-L206（v0.27.1 现核）。
import asyncio

import pytest
from fastapi.exceptions import RequestValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.serve.utils.api_utils import (
    get_max_tokens,
    listen_for_disconnect,
    should_include_usage,
    validate_json_request,
    with_cancellation,
)
from vllm.sampling_params import RequestOutputKind


# ---------------------------------------------------------------------------
# get_max_tokens：四方取 min（api_utils.py:L170-L206）
# ---------------------------------------------------------------------------
class TestGetMaxTokens:
    def test_model_len_minus_input_when_no_request_token(self):
        # model_max_tokens = 100 - 30 = 70；platform(None)/fallback(None) 不设限
        assert get_max_tokens(100, None, 30, {}) == 70

    def test_request_token_wins_when_smaller(self):
        assert get_max_tokens(100, 10, 30, {}) == 10

    def test_default_sampling_params_max_tokens_as_fallback(self):
        assert get_max_tokens(100, None, 30, {"max_tokens": 50}) == 50

    def test_override_clamps_everything(self):
        assert get_max_tokens(100, 50, 30, {}, override_max_tokens=5) == 5

    def test_input_longer_than_model_len_raises(self):
        with pytest.raises(ValueError, match="exceeds model's maximum"):
            get_max_tokens(10, None, 30, {})

    def test_truncate_prompt_tokens_minus1_clamps_to_model_len(self):
        # limit == -1 → input_length 截到 max_model_len → model_max_tokens = 0
        assert get_max_tokens(100, None, 300, {}, truncate_prompt_tokens=-1) == 0

    def test_truncate_prompt_tokens_explicit_limit(self):
        assert get_max_tokens(100, None, 300, {}, truncate_prompt_tokens=20) == 80


# ---------------------------------------------------------------------------
# should_include_usage：stream_options 两值解析（api_utils.py:L276-L288）
# ---------------------------------------------------------------------------
class TestShouldIncludeUsage:
    def test_no_stream_options(self):
        assert should_include_usage(None, False) == (False, False)

    def test_include_usage_only(self):
        so = StreamOptions(include_usage=True)
        assert should_include_usage(so, False) == (True, False)

    def test_include_continuous_requires_usage(self):
        so = StreamOptions(include_usage=True, continuous_usage_stats=True)
        assert should_include_usage(so, False) == (True, True)

    def test_continuous_without_usage_is_false(self):
        so = StreamOptions(include_usage=False, continuous_usage_stats=True)
        assert should_include_usage(so, False) == (False, False)

    def test_force_include_usage_wins(self):
        assert should_include_usage(None, True) == (True, True)


# ---------------------------------------------------------------------------
# listen_for_disconnect + with_cancellation：双任务竞速（api_utils.py:L37-L94）
# ---------------------------------------------------------------------------
class _FakeReceive:
    def __init__(self, messages):
        self.messages = list(messages)
        self.awaited = asyncio.Event()

    async def __call__(self):
        self.awaited.set()
        if self.messages:
            return self.messages.pop(0)
        # 无更多消息：挂起直到被取消（模拟长连接 receive 通道）
        await asyncio.Event().wait()


class _FakeApp:
    class state:
        pass


class _FakeRequest:
    def __init__(self, messages=()):
        self.receive = _FakeReceive(list(messages))
        self.app = _FakeApp()
        self.headers = {}


class TestWithCancellation:
    async def test_handler_completes_first_returns_result(self):
        # handler 立即完成 → wrapper 返回其结果（api_utils.py:L90-L91）
        req = _FakeRequest([{"type": "http.disconnect"}])

        async def handler(request, raw_request):
            return "handler-result"

        wrapped = with_cancellation(handler)
        out = await wrapped(request=None, raw_request=req)
        assert out == "handler-result"

    async def test_disconnect_wins_cancels_handler_and_returns_none(self):
        # handler 阻塞、断连先到 → handler 被取消，wrapper 返回 None
        req = _FakeRequest([{"type": "http.disconnect"}])

        handler_cancelled = asyncio.Event()

        async def handler(request, raw_request):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise

        wrapped = with_cancellation(handler)
        out = await wrapped(request=None, raw_request=req)
        assert out is None
        assert handler_cancelled.is_set()

    async def test_listen_for_disconnect_skips_non_disconnect_messages(self):
        # 先收到非断连消息继续循环，直到 http.disconnect 才返回
        req = _FakeRequest(
            [{"type": "http.request"}, {"type": "http.disconnect"}]
        )
        await listen_for_disconnect(req)
        assert not req.receive.messages


# ---------------------------------------------------------------------------
# validate_json_request：content-type 门（api_utils.py:L348-L354）
# ---------------------------------------------------------------------------
class _HeadersReq:
    def __init__(self, ct):
        self.headers = {"content-type": ct} if ct is not None else {}


class TestValidateJsonRequest:
    async def test_json_ok(self):
        await validate_json_request(_HeadersReq("application/json"))

    async def test_json_with_charset_ok(self):
        await validate_json_request(_HeadersReq("application/json; charset=utf-8"))

    async def test_other_media_type_rejected(self):
        with pytest.raises(RequestValidationError):
            await validate_json_request(_HeadersReq("text/plain"))

    async def test_missing_content_type_rejected(self):
        with pytest.raises(RequestValidationError):
            await validate_json_request(_HeadersReq(None))


# ---------------------------------------------------------------------------
# to_sampling_params 的 output_kind 入口声明经 ChatCompletionRequest 驱动
# （WC2：stream 二值映射）——放此处一并覆盖 api_utils 邻近的协议定型面。
# ---------------------------------------------------------------------------
class TestOutputKindMapping:
    def _params(self, **kw):
        from conftest import make_request

        return make_request(**kw).to_sampling_params(64, {})

    def test_stream_true_maps_to_delta(self):
        assert self._params(stream=True).output_kind is RequestOutputKind.DELTA

    def test_stream_false_maps_to_final_only(self):
        assert self._params(stream=False).output_kind is RequestOutputKind.FINAL_ONLY
