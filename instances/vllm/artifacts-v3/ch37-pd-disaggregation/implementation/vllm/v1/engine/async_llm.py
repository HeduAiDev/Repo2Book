# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""异步前端：拒绝回执的发起端。

D 侧 serving 层拒收（请求根本没进引擎）时，这里构造一条**占位请求**发给引擎——
它唯一的作用是触发 connector 的 `request_finished` 钩子，让 P 释放被钉住的块。

# SOURCE: vllm/v1/engine/async_llm.py:L735-L769（notify_kv_transfer_request_rejected）
# SUBTRACTED: 整条前端（请求生命周期/输出流/中止/暂停恢复/度量）——本章只需要
#   这一根「拒收回执」的通道。
"""

import time
from typing import Any

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/async_llm.py:L72-L120
class AsyncLLM:
    # SOURCE: vllm/v1/engine/async_llm.py:L72-L120
    def __init__(self, engine_core: Any, vllm_config: Any = None) -> None:
        # SOURCE: vllm/v1/engine/async_llm.py:L72-L120
        self.engine_core = engine_core
        # SUBTRACTED: output_processor / engine_core_client 双通道 / 请求表——
        #   本章只需要 engine_core 的 add_request_async 入口。
        self.vllm_config = vllm_config

    # SOURCE: vllm/v1/engine/async_llm.py:L743-L769
    async def notify_kv_transfer_request_rejected(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any],
        *,
        data_parallel_rank: int | None = None,
    ) -> None:
        """Submit a pre-aborted request so the connector's request_finished
        hook runs to free any pre-admission KV-transfer resources (e.g. NIXL
        prefill blocks pinned on the P node)."""
        request = EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=[0],
            mm_features=None,
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"kv_transfer_params": dict(kv_transfer_params)},
            ),
            pooling_params=None,
            arrival_time=time.time(),
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=data_parallel_rank,
            abort_immediately=True,
        )
        await self.engine_core.add_request_async(request)


__all__ = ["AsyncLLM"]
