# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""执行器抽象：TP 各 worker 的输出聚合器在这里装配。

# SOURCE: vllm/v1/executor/abstract.py:L280-L284（init_kv_output_aggregator）
# SUBTRACTED: 执行器家族（单进程/Ray/多进程/worker 健康检查/集体 RPC 族）——
#   本章只保留「装配聚合器」这一行挂钩（EngineCore 启动时调用）。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator


# SOURCE: vllm/v1/executor/abstract.py:L37-L120
class Executor:
    """本章的切面：只有聚合器装配点。"""

    # SOURCE: vllm/v1/executor/abstract.py:L280-L284
    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        """Init KVOutputAggregator"""
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator

        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )

    # SOURCE: vllm/v1/executor/abstract.py:L204-L210
    def get_kv_connector_handshake_metadata(self):
        raise NotImplementedError

    # SOURCE: vllm/v1/executor/abstract.py:L276-L285
    def shutdown(self) -> None:
        # SOURCE: vllm/v1/executor/abstract.py:L276-L285
        """Shutdown the executor."""
        raise NotImplementedError


__all__ = ["Executor"]
