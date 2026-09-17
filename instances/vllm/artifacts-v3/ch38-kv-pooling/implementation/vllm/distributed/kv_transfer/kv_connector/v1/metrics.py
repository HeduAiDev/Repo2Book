# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""connector 遥测基类（本章只保契约签名——减法计划删除项 7）。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L1-L120
# SUBTRACTED: KVConnectorLogging / KVConnectorPromMetrics 的 Prometheus 注册体
#   与 reduce/log 链——纯观测旁路，控制流不依赖（保留契约签名与默认空返回）。
"""

from dataclasses import dataclass, field
from typing import Any


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L19-L61
@dataclass
class KVConnectorStats:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L19-L26
    """
    Base class for KV Connector Stats, a container for transfer performance
    metrics or otherwise important telemetry from the connector.
    """

    data: dict[str, Any] = field(default_factory=dict)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L28-L31
    def reset(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L28-L31
        """Reset the stats, clear the state."""
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L33-L39
    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L33-L39
        """Aggregate stats with another `KVConnectorStats` object."""
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L41-L51
    def reduce(self) -> dict[str, int | float]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L41-L51
        """Reduce the observations to representative values."""
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L53-L61
    def is_empty(self) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L53-L61
        """Return True if the stats are empty."""
        return not self.data


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L103-L120
# SUBTRACTED: Prometheus 注册体（Gauge/Counter/Histogram 构造）——契约位保留。
# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L103-L120
class KVConnectorPromMetrics:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L103-L120
    """Per-connector Prometheus metric registration（签名位）。"""


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:L15-L18
PromMetric = Any
PromMetricT = Any


__all__ = ["KVConnectorStats", "KVConnectorPromMetrics", "PromMetric", "PromMetricT"]
