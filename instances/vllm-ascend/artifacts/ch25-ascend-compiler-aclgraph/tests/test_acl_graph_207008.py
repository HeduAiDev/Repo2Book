"""ch25 — stream-resource 耗尽错误码 207008 的兜底识别与改写（acl_graph.py 真实行为）。

NPUGraph 捕获在 stream 资源不足时报错码 207008（或两个英文标志串）。裸抛对用户不可读，
_is_stream_resource_capture_error 负责识别、_raise_stream_resource_capture_error 改写成含
具体调参指引的 RuntimeError。这两个函数是纯字符串逻辑，host 可直接真跑。
"""
import pytest

import _ch25_acl_graph as ag


def test_error_code_constant_is_207008():
    assert ag._STREAM_RESOURCE_ERROR_CODE == "207008"


def test_detect_by_error_code_with_stream_resource_phrase():
    exc = RuntimeError("capture failed: error 207008, stream resource not available")
    assert ag._is_stream_resource_capture_error(exc) is True


def test_detect_by_marker_phrase_insufficient():
    exc = RuntimeError("RuntimeError: insufficient_stream_resources during capture")
    assert ag._is_stream_resource_capture_error(exc) is True


def test_detect_by_marker_phrase_are_insufficient():
    exc = RuntimeError("stream resources are insufficient")
    assert ag._is_stream_resource_capture_error(exc) is True


def test_bare_207008_without_stream_phrase_is_not_matched():
    # 207008 出现但无 'stream resource' 上下文、也无标志串 → 不误判
    exc = RuntimeError("unrelated failure code 207008 elsewhere")
    assert ag._is_stream_resource_capture_error(exc) is False


def test_unrelated_error_not_matched():
    exc = RuntimeError("some other runtime error")
    assert ag._is_stream_resource_capture_error(exc) is False


def test_raise_rewrites_with_guidance_and_chains_original():
    original = RuntimeError("error 207008 stream resource exhausted")
    with pytest.raises(RuntimeError) as ei:
        ag._raise_stream_resource_capture_error(original)
    msg = str(ei.value)
    # 改写后的报错含可操作指引
    assert "cudagraph_capture_sizes" in msg
    assert "stream-resource exhaustion" in msg
    assert "Original error:" in msg
    # 原异常被 chain（from exc）
    assert ei.value.__cause__ is original
