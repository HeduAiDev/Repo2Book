# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 传输配置：P/D 部署的角色门（kv_role / engine_id / 连接器名）。

# SOURCE: vllm/config/kv_transfer.py:L1-L121（整文件）
# SUBTRACTED: 无（本文件是本章 m1 的落点，逐字保留；仅 compute_hash 里
#   safe_hash 的 usedforsecurity 参数按宿主实现收敛）。
"""

import uuid
from dataclasses import field
from typing import Any, Literal, get_args

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

# SOURCE: vllm/config/kv_transfer.py:L11-L13
KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]


# SOURCE: vllm/config/kv_transfer.py:L16-L19
def kv_buffer_device_default_factory() -> str:
    from vllm.platforms import current_platform

    return current_platform.device_type


# SOURCE: vllm/config/kv_transfer.py:L22-L121
@config
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    kv_connector: str | None = None
    """The KV connector for vLLM to transmit KV caches between vLLM instances.
    """

    engine_id: str | None = None
    """The engine id for KV transfers."""

    kv_buffer_device: str = field(default_factory=kv_buffer_device_default_factory)
    """The device used by kv connector to buffer the KV cache. Choices are
    'cuda', 'cpu' and 'xpu'."""

    kv_buffer_size: float = 1e9
    """The buffer size for TorchDistributedConnector. Measured in number of
    bytes. Recommended value: 1e9 (about 1GB)."""

    kv_role: KVRole | None = None
    """Whether this vLLM instance produces, consumes KV cache, or both. Choices
    are 'kv_producer', 'kv_consumer', and 'kv_both'."""

    kv_rank: int | None = None
    """The rank of this vLLM instance in the KV cache transfer. Typical value:
    0 for prefill instance, 1 for decode instance.
    Currently only 1P1D is supported."""

    kv_parallel_size: int = 1
    """The number of parallel instances for KV cache transfer."""

    kv_ip: str = "127.0.0.1"
    """The KV connector ip, used to build distributed connection."""

    kv_port: int = 14579
    """The KV connector port, used to build distributed connection."""

    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    kv_connector_module_path: str | None = None
    """The Python module path to dynamically load the KV connector from.
    Only supported in V1."""

    enable_permute_local_kv: bool = False
    """Experiment feature flag to enable HND to NHD KV Transfer"""

    kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
    """Policy for handling KV cache load failures.
    'recompute': reschedule the request to recompute failed blocks
    'fail': immediately fail the request with an error finish reason (default)"""

    # SOURCE: vllm/config/kv_transfer.py:L74-L90
    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    # SOURCE: vllm/config/kv_transfer.py:L92-L106
    def __post_init__(self) -> None:
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role}. "
                f"Supported roles are {get_args(KVRole)}"
            )

        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(
                "Please specify kv_role when kv_connector "
                f"is set, supported roles are {get_args(KVRole)}"
            )

    # SOURCE: vllm/config/kv_transfer.py:L108-L110
    @property
    def is_kv_transfer_instance(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L112-L114
        return self.kv_connector is not None and self.kv_role in get_args(KVRole)

    # SOURCE: vllm/config/kv_transfer.py:L116-L118
    @property
    def is_kv_producer(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L116-L118
        return self.kv_connector is not None and self.kv_role in get_args(KVProducer)

    # SOURCE: vllm/config/kv_transfer.py:L116-L118
    @property
    def is_kv_consumer(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L116-L118
        return self.kv_connector is not None and self.kv_role in get_args(KVConsumer)

    # SOURCE: vllm/config/kv_transfer.py:L120-L121
    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)


__all__ = ["KVTransferConfig", "KVRole", "KVProducer", "KVConsumer"]
