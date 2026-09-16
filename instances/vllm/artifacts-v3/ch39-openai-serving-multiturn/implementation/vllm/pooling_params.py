# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/pooling_params.py —— HOST SEAM（最小承载）：本章消费面
# （EngineCoreRequest.pooling_params 注解与 EngineClient.encode 签名）只需
# PoolingParams 形状本身；pooler 激活/维度裁剪等参数语义归 pooling 域
# （vllm/entrypoints/pooling/，m18 平行 API 面之一，不展开）。
from typing import Any

import msgspec

from vllm.sampling_params import RequestOutputKind


# SOURCE: vllm/pooling_params.py:L31-L72 —— HOST SEAM：PoolingParams 字段面
class PoolingParams(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    # SOURCE: vllm/pooling_params.py:L54 —— use_activation
    use_activation: bool | None = None
    # SOURCE: vllm/pooling_params.py:L58 —— dimensions
    dimensions: int | None = None
    # SUBTRACTED: vllm/pooling_params.py:L60-L70 step pooling 三字段与
    # late_interaction——本章纯 generate 主线不消费。
    # SOURCE: vllm/pooling_params.py:L71 —— output_kind（池化默认 FINAL_ONLY）
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY
    # SOURCE: vllm/pooling_params.py:L73 —— extra_kwargs
    extra_kwargs: dict[str, Any] | None = None

    # SOURCE: vllm/pooling_params.py:L76-L77
    @property
    def all_parameters(self) -> list[str]:
        return ["dimensions", "use_activation"]
