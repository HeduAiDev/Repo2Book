# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""连接器工厂：命令行里那个 `NixlConnector` 字符串在这里落到类。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L1-L242
# SUBTRACTED: 其余后端注册行（LMCache/Mooncake/MoRIIO/Offloading/FlexKV/HF3FS/
#   DecodeBench/ExampleHiddenStates 等，L152-L175 与 L196-L242）——本章只讲 NIXL，
#   其余后端归 ch36（池化）；MultiConnector 的 HMA 传递分支一并删（减法计划删除项 9）。
"""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from vllm.config.kv_transfer import KVTransferConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L21-L25
class KVConnectorFactory:
    _registry: dict[str, Callable[[], type]] = {}

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L25-L36
    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L31-L34
        def loader() -> type:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L38-L72
    # SUBTRACTED: HMA 支持性检查（supports_hma_config，L74-L140）——本章的 NIXL
    #   connector 实现了 SupportsHMA，该检查恒过。
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L38-L72
    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L38-L72
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        connector_cls = cls.get_connector_class(kv_transfer_config)

        logger.info(
            "Creating v1 connector with name: %s and engine_id: %s",
            connector_cls.__name__,
            kv_transfer_config.engine_id,
        )
        # NOTE(Kuntai): v1 connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        return connector_cls(config, role, kv_cache_config)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L74-L94
    @classmethod
    def get_connector_class_by_name(cls, connector_name: str) -> type:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L74-L94
        """Get a registered connector class by name."""
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        return cls._registry[connector_name]()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L96-L140
    # SUBTRACTED: 外部模块路径加载（kv_connector_module_path）与旧式两参构造
    #   签名告警（L112-L135）——本章只走内建注册表。
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L96-L140
    @classmethod
    def get_connector_class(cls, kv_transfer_config: "KVTransferConfig") -> type:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L96-L140
        connector_name = kv_transfer_config.kv_connector
        if connector_name is None:
            raise ValueError("Connector name is not set in KVTransferConfig")
        if connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(f"Unsupported connector type: {connector_name}")
        return cast(type, connector_cls)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L143-L193
# SUBTRACTED: 除 NIXL 三条以外的全部注册（L146-L175 / L196-L242）——见模块头。
KVConnectorFactory.register_connector(
    "NixlConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlConnector",
)

KVConnectorFactory.register_connector(
    "NixlPullConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlPullConnector",
)

KVConnectorFactory.register_connector(
    "NixlPushConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlPushConnector",
)


__all__ = ["KVConnectorFactory"]
