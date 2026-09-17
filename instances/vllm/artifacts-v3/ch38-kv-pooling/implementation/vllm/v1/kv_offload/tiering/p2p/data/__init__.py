# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SOURCE: vllm/v1/kv_offload/tiering/p2p/data/__init__.py:L1-L10
# SUBTRACTED: NixlTransport 重导出（data/nixl.py 依赖真 NIXL 包
#   vllm.distributed.nixl_utils——删除项 9 不进精简版；契约面 base.py 保留）。
from vllm.v1.kv_offload.tiering.p2p.data.base import DataTransport, PollResult

__all__ = [
    "DataTransport",
    "PollResult",
]
