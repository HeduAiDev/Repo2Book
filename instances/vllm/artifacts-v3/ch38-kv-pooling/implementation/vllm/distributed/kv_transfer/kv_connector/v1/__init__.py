# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v1 connector 包出口（本章消费面）。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/__init__.py
# SUBTRACTED: 其余重导出（metrics/example 等）——按需 import 即可。
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
    supports_hma,
)

__all__ = [
    "KVConnectorBase_V1",
    "KVConnectorHandshakeMetadata",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "KVConnectorWorkerMetadata",
    "SupportsHMA",
    "supports_hma",
]
