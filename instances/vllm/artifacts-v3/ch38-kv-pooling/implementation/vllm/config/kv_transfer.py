# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVTransferConfig：kv_connector / kv_role / extra_config 三件（本章消费面）。

# SOURCE: vllm/config/kv_transfer.py:L1-L130
# SUBTRACTED: kv_buffer_device/size、kv_rank/parallel/ip/port（P/D 传输面，ch37）、
#   enable_permute_local_kv、kv_load_failure_policy（失败面归 ch16）、compute_hash
#   （本章兼容 hash 不算它）——减法计划把生态对照收敛到注册名 + extra_config。
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]


# SOURCE: vllm/config/kv_transfer.py:L23-L108
@dataclass
class KVTransferConfig:
    # SOURCE: vllm/config/kv_transfer.py:L23-L108
    """Configuration for distributed KV cache transfer."""

    kv_connector: str | None = None
    engine_id: str | None = None
    kv_role: KVRole | None = None
    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    kv_connector_module_path: str | None = None

    # SOURCE: vllm/config/kv_transfer.py:L84-L106（__post_init__ 校验）
    def __post_init__(self) -> None:
        # SOURCE: vllm/config/kv_transfer.py:L84-L106
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

    # SOURCE: vllm/config/kv_transfer.py:L110-L118
    @property
    def is_kv_transfer_instance(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L110-L112
        return self.kv_connector is not None and self.kv_role in get_args(KVRole)

    # SOURCE: vllm/config/kv_transfer.py:L114-L116
    @property
    def is_kv_producer(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L114-L116
        return self.kv_connector is not None and self.kv_role in get_args(KVProducer)

    # SOURCE: vllm/config/kv_transfer.py:L117-L118
    @property
    def is_kv_consumer(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L117-L118
        return self.kv_connector is not None and self.kv_role in get_args(KVConsumer)

    # SOURCE: vllm/config/kv_transfer.py:L120-L121
    def get_from_extra_config(self, key, default) -> Any:
        # SOURCE: vllm/config/kv_transfer.py:L120-L121
        return self.kv_connector_extra_config.get(key, default)


__all__ = ["KVTransferConfig", "KVRole", "KVProducer", "KVConsumer"]
