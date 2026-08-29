# SOURCE: vllm/config/kv_transfer.py
# KVTransferConfig——配置门（m16）：kv_role 三态（producer/consumer/both，
# 既设 connector 则必设 role）、kv_load_failure_policy（recompute|fail，
# 默认 fail——m10 双策）、is_kv_producer/consumer/transfer_instance 三谓词
# （驱动 requires_kv_delivery/defer_block_free/worker 装配门）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   kv_ip/kv_port/kv_rank/kv_parallel_size/kv_buffer_device/kv_buffer_size
#     （P/D 拓扑与握手参数 → ch36；本章单引擎语义下不消费）；
#   enable_permute_local_kv（HND/NHD 布局变换 → ch36）；
#   compute_hash（配置图哈希面 → ch03/19）；@config pydantic 装配链
#     （切面 dataclass 直供——post_init 校验逐字保留）。
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, get_args


# SOURCE: vllm/config/kv_transfer.py:L11-L13 角色三态的类型账
KVProducer = Literal["kv_producer", "kv_both"]
KVConsumer = Literal["kv_consumer", "kv_both"]
KVRole = Literal[KVProducer, KVConsumer]


# SOURCE: vllm/config/kv_transfer.py:L22 KVTransferConfig（@config 装配链
#   删——dataclass 直供）
@dataclass
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    # SOURCE: vllm/config/kv_transfer.py:L26-L28 kv_connector
    kv_connector: str | None = None
    """The KV connector for vLLM to transmit KV caches between vLLM instances.
    """

    # SOURCE: vllm/config/kv_transfer.py:L30-L31 engine_id
    engine_id: str | None = None
    """The engine id for KV transfers."""

    # SUBTRACTED: kv_buffer_device/kv_buffer_size（L33-L39——NIXL 面）。

    # SOURCE: vllm/config/kv_transfer.py:L41-L43 kv_role 三态
    kv_role: KVRole | None = None
    """Whether this vLLM instance produces, consumes KV cache, or both. Choices
    are 'kv_producer', 'kv_consumer', and 'kv_both'."""

    # SUBTRACTED: kv_rank/kv_parallel_size/kv_ip/kv_port（L45-L57——P/D
    #   拓扑与握手 → ch36）。

    # SOURCE: vllm/config/kv_transfer.py:L59-L60 kv_connector_extra_config
    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    # SOURCE: vllm/config/kv_transfer.py:L62-L64 kv_connector_module_path
    kv_connector_module_path: str | None = None
    """The Python module path to dynamically load the KV connector from.
    Only supported in V1."""

    # SUBTRACTED: enable_permute_local_kv（L66-L67——HND/NHD → ch36）。

    # SOURCE: vllm/config/kv_transfer.py:L69-L72 kv_load_failure_policy
    kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
    """Policy for handling KV cache load failures.
    'recompute': reschedule the request to recompute failed blocks
    'fail': immediately fail the request with an error finish reason (default)"""

    # SOURCE: vllm/config/kv_transfer.py:L92 __post_init__（校验面逐字）
    def __post_init__(self) -> None:
        # SOURCE: vllm/config/kv_transfer.py:L93-L94（engine_id 缺省生成）
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        # SOURCE: vllm/config/kv_transfer.py:L96-L100（非法 role 拒绝）
        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role}. "
                f"Supported roles are {get_args(KVRole)}"
            )

        # SOURCE: vllm/config/kv_transfer.py:L102-L106（既设 connector 则
        #   必设 role——配置门的硬规则）
        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(
                "Please specify kv_role when kv_connector "
                f"is set, supported roles are {get_args(KVRole)}"
            )

    # SOURCE: vllm/config/kv_transfer.py:L108-L110 is_kv_transfer_instance
    #   ——worker 装配门（ensure_kv_transfer_initialized 的判定）
    @property
    def is_kv_transfer_instance(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L110
        return self.kv_connector is not None and self.kv_role in get_args(KVRole)

    # SOURCE: vllm/config/kv_transfer.py:L112-L114 is_kv_producer——
    #   requires_kv_delivery 的驱动谓词
    @property
    def is_kv_producer(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L114
        return self.kv_connector is not None and self.kv_role in get_args(KVProducer)

    # SOURCE: vllm/config/kv_transfer.py:L116-L118 is_kv_consumer——
    #   defer_block_free 的驱动谓词
    @property
    def is_kv_consumer(self) -> bool:
        # SOURCE: vllm/config/kv_transfer.py:L118
        return self.kv_connector is not None and self.kv_role in get_args(KVConsumer)

    # SOURCE: vllm/config/kv_transfer.py:L120-L121 get_from_extra_config
    def get_from_extra_config(self, key, default) -> Any:
        return self.kv_connector_extra_config.get(key, default)
