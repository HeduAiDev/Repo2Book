# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v0/v1 connector 公共类型位。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/base.py:L1-L10
# SUBTRACTED: KVConnectorBase（v0 基类）——本章只有 v1 面。
"""

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

# SOURCE: vllm/distributed/kv_transfer/kv_connector/base.py:L8
KVConnectorBaseType = KVConnectorBase_V1

# SOURCE: vllm/distributed/kv_transfer/kv_connector/base.py:L7-L9
KVConnectorBase = KVConnectorBase_V1

__all__ = ["KVConnectorBase", "KVConnectorBaseType"]
