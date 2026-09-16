# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""serving 层的 P/D 责任：请求若**根本没进引擎**，替引擎把远端钉住的块销账。

# SOURCE: vllm/entrypoints/generate/base/serving.py:L113-L259
# SUBTRACTED: 全量 OpenAI 端点实现（chat/completions/embeddings/rerank 的渲染与
#   流式应答、模型名校验、LoRA 解析、trace header、指纹/度量）——本章只需要
#   「拒收检测包装」这一个切面；`_get_data_parallel_rank` 经 `generate_stream`/
#   `_with_kv_transfer_rejection_cleanup` 被它消费，故一并保留。
"""

from collections.abc import Awaitable
from typing import Any, TypeVar

from vllm.logger import init_logger

logger = init_logger(__name__)

_T = TypeVar("_T")


# SOURCE: vllm/entrypoints/openai/engine/protocol.py:L70-L90
class ErrorResponse:
    """本章的拒收判据：create_* 返回它就是「请求没进引擎」。"""

    # SOURCE: vllm/entrypoints/openai/engine/protocol.py:L70-L90
    def __init__(self, message: str = "") -> None:
        # SOURCE: vllm/entrypoints/openai/engine/protocol.py:L70-L90
        self.message = message


# SOURCE: vllm/entrypoints/generate/base/serving.py:L113-L200
class GenerateBaseServing:
    # SOURCE: vllm/entrypoints/generate/base/serving.py:L122-L150
    def __init__(
        self,
        engine_client: Any,
        *,
        has_kv_connector: bool | None = None,
    ) -> None:
        # SUBTRACTED: BaseServing/BeamSearchOnlineMixin 初始化、renderer /
        #   input_processor / request_logger / 系统指纹（L120-L200）——本章不跑
        #   真前端，只需要 engine_client 与「有没有 KV connector」这一个事实。
        # SOURCE: vllm/entrypoints/generate/base/serving.py:L122-L150
        self.engine_client = engine_client
        if has_kv_connector is None:
            vllm_config = getattr(engine_client, "vllm_config", None)
            kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
            has_kv_connector = kv_transfer_config is not None
        self.has_kv_connector = has_kv_connector

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L203-L217
    @staticmethod
    def _get_data_parallel_rank(raw_request: Any | None) -> int | None:
        # SOURCE: vllm/entrypoints/generate/base/serving.py:L203-L217
        """Pulls the data parallel rank from a header, if provided"""
        if raw_request is None:
            return None

        rank_str = raw_request.headers.get("X-data-parallel-rank")
        if rank_str is None:
            return None

        try:
            return int(rank_str)
        except ValueError:
            return None

    # SOURCE: vllm/entrypoints/generate/base/serving.py:L218-L252
    async def _with_kv_transfer_rejection_cleanup(
        self,
        awaitable: Awaitable[_T],
        request: Any,
        raw_request: Any | None,
    ) -> _T:
        """Wrap a `create_*` coroutine so that, if it raises or returns an
        ErrorResponse (i.e. the request never reached the engine), the KV
        connector is notified to free any pinned remote-prefill blocks."""
        kv_transfer_params = self.has_kv_connector and request.kv_transfer_params
        if not kv_transfer_params or not kv_transfer_params.get("do_remote_prefill"):
            return await awaitable

        notify = True
        try:
            result = await awaitable
            if not isinstance(result, ErrorResponse):
                notify = False
            return result
        finally:
            if notify:
                try:
                    await self.engine_client.notify_kv_transfer_request_rejected(
                        request.request_id,
                        kv_transfer_params,
                        data_parallel_rank=self._get_data_parallel_rank(raw_request),
                    )
                except Exception:
                    logger.warning(
                        "Failed to notify KV connector about rejected request %s",
                        request.request_id,
                        exc_info=True,
                    )


__all__ = ["GenerateBaseServing", "ErrorResponse"]
