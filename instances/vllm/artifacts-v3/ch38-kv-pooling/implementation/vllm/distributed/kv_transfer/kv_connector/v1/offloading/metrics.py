# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""offloading 遥测（本章只保契约签名——减法计划删除项 7）。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:L1-L504
# SUBTRACTED: _TransferMetricName/_ConnectorMetricName 双指标族、
#   OffloadingConnectorStats 的 observe/increase/aggregate 记录体、
#   OffloadPromMetrics 的 Prometheus 注册与 build_metric_definitions 链、
#   get_connector_metric_definitions——纯观测旁路，传输与准入控制流不依赖
#   （保留契约签名与默认 None/空返回）。
"""

from typing import Any


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:OffloadingConnectorStats 类位
class OffloadingConnectorStats:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:OffloadingConnectorStats 类位
    """Connector-side stats container（契约位；记录点已删）。"""

    def __init__(self, data: dict[str, Any] | None = None):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:KVConnectorStats 基座
        self.data: dict[str, Any] = data or {}

    def is_empty(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:is_empty 段
        return not self.data

    def aggregate(self, other: "OffloadingConnectorStats") -> "OffloadingConnectorStats":
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:aggregate 段
        return OffloadingConnectorStats()


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:OffloadPromMetrics 类位
class OffloadPromMetrics:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:OffloadPromMetrics 类位
    """Prometheus metrics for OffloadingConnector（契约位）。"""


__all__ = ["OffloadingConnectorStats", "OffloadPromMetrics"]
