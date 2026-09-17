# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""connector 工厂：注册表 + role-split 构建（生态面孔）。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L1-L243
# SUBTRACTED: Example/ExampleHiddenStates/LMCacheMP/NixlPull/NixlPush/MoRIIO/
#   DecodeBench/FlexKV/SimpleCPUOffload/HF3FS 十条注册行——减法计划删除项 6：
#   生态对照保留六条注册（OffloadingConnector/MultiConnector/MooncakeConnector/
#   MooncakeStoreConnector/LMCacheConnectorV1/NixlConnector），注册机制本身
#   （懒加载 loader + out-of-tree module path + HMA 支持）逐字保留。
"""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase,
    KVConnectorBaseType,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorRole,
    supports_hma,
)
from vllm.logger import init_logger
from vllm.utils.func_utils import supports_kw

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L27-L145
class KVConnectorFactory:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L28
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L30-L40
    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L31-L40
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[KVConnectorBase]:
            # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L36-L39
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L42-L75
    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> KVConnectorBase:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L48-L60
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        connector_cls = cls.get_connector_class(kv_transfer_config)

        # check if the connector supports HMA
        hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
        if hma_enabled and not cls.supports_hma_config(kv_transfer_config):
            raise ValueError(
                f"Connector {connector_cls.__name__} does not support HMA but "
                f"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`."
            )

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

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L77-L93
    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBaseType]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L78-L93
        """Get a registered connector class by name.

        Raises ValueError if the connector is not registered.
        """
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        return cls._registry[connector_name]()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L95-L128
    @classmethod
    def get_connector_class(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> type[KVConnectorBaseType]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L99-L128
        connector_name = kv_transfer_config.kv_connector
        if connector_name is None:
            raise ValueError("Connector name is not set in KVTransferConfig")
        connector_module_path = kv_transfer_config.kv_connector_module_path
        if connector_module_path is not None and not connector_module_path:
            raise ValueError("kv_connector_module_path cannot be an empty string.")
        if connector_module_path:
            # External module path takes priority over internal registry.
            connector_module = importlib.import_module(connector_module_path)
            try:
                connector_cls = getattr(connector_module, connector_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Class {connector_name} not found in {connector_module_path}"
                ) from e
            connector_cls = cast(type[KVConnectorBaseType], connector_cls)
            if not supports_kw(connector_cls, "kv_cache_config"):
                msg = (
                    f"Connector {connector_cls.__name__} uses deprecated "
                    "2-argument constructor signature. External v1 KV "
                    "connectors must accept kv_cache_config as the third "
                    "constructor argument and pass it to super().__init__()."
                )
                logger.error(msg)
                raise ValueError(msg)
        elif connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(f"Unsupported connector type: {connector_name}")
        return connector_cls

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L130-L145
    @classmethod
    def supports_hma_config(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L131-L145
        """Return whether this KV transfer config supports HMA.

        MultiConnector is a special case: the wrapper class implements
        SupportsHMA, but effective support depends on every configured child.
        """
        connector_cls = cls.get_connector_class(kv_transfer_config)
        if kv_transfer_config.kv_connector != "MultiConnector":
            return supports_hma(connector_cls)

        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )

        return MultiConnector.all_children_support_hma(kv_transfer_config)


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L176-L198
KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1",
)

KVConnectorFactory.register_connector(
    "NixlConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.nixl",
    "NixlConnector",
)

KVConnectorFactory.register_connector(
    "MultiConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector",
    "MultiConnector",
)

KVConnectorFactory.register_connector(
    "OffloadingConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
    "OffloadingConnector",
)

# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L218-L227
KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
    "MooncakeConnector",
)
KVConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector",
    "MooncakeStoreConnector",
)
# SUBTRACTED: ExampleConnector / ExampleHiddenStatesConnector / LMCacheMPConnector /
#   NixlPullConnector / NixlPushConnector / MoRIIOConnector / DecodeBenchConnector /
#   FlexKVConnectorV1 / SimpleCPUOffloadConnector / HF3FSKVConnector 十条注册行
#   ——减法计划删除项 6：实现体未进精简版（懒加载，注册名仅在按名取类时触）。
