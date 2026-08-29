# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py
# KVConnectorFactory——role-split 构造 + 懒加载注册表（m1）：同一个类按
# role 分别实例化（NOTE 原话『We build separately to enforce strict
# separation』——零共享状态的构造保证）；supports_hma_config 把 HMA 门。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 2 条具体后端注册行（L158-L242）——只留 ExampleConnector 一条示范
#     懒加载机制（后端本体 → ch36/37）；
#   supports_hma_config 的 MultiConnector 特例分支（L138-L145——子连接器
#     包装归 ch36）；
#   get_connector_class 的 2 参旧构造告警段（L115-L123——废弃面）。
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from .base import KVConnectorBase_V1, KVConnectorRole, supports_hma
from .kv_transfer import KVTransferConfig

if TYPE_CHECKING:
    from .config import VllmConfig
    from .kv_cache_interface import KVCacheConfig

import logging

logger = logging.getLogger(__name__)  # LOGGER SEAM：vllm.logger.init_logger → stdlib

# SOURCE: vllm/distributed/kv_transfer/kv_connector/base.py（KVConnectorBase =
#   KVConnectorBase_V1 的别名面——v1 时代唯一形态，本章直用本体）
KVConnectorBase = KVConnectorBase_V1
KVConnectorBaseType = KVConnectorBase_V1


# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L27 KVConnectorFactory
class KVConnectorFactory:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L28 懒加载
    #   注册表（import 只发生在 create 时——后端重依赖不进主路径）
    _registry: dict[str, Callable[[], type[KVConnectorBase]]] = {}

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L30
    #   register_connector——注册机制本体
    @classmethod
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L33-L34
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L36-L40
        def loader() -> type[KVConnectorBase]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L42
    #   create_connector——role-split 构造主入口（NOTE 原话逐字保留）
    @classmethod
    def create_connector(
        cls,
        config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> KVConnectorBase:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L49-L52
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        connector_cls = cls.get_connector_class(kv_transfer_config)

        # check if the connector supports HMA
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L54-L60
        #   （HMA 门：混合内存分配器开而 connector 不支持 → 拒）
        hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
        if hma_enabled and not cls.supports_hma_config(kv_transfer_config):
            raise ValueError(
                f"Connector {connector_cls.__name__} does not support HMA but "
                f"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`."
            )

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L62-L66
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
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L75
        #   ——同一个类、两个进程、零共享状态
        return connector_cls(config, role, kv_cache_config)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L77
    #   get_connector_class_by_name——按名取类（未注册 → ValueError）
    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBaseType]:
        """Get a registered connector class by name.

        Raises ValueError if the connector is not registered.

        Args:
            connector_name: Name of the registered connector.

        Returns:
            The connector class.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L91-L93
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        return cls._registry[connector_name]()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L95
    #   get_connector_class——外部模块路径优先于内置注册表
    @classmethod
    def get_connector_class(
        cls, kv_transfer_config: "KVTransferConfig"
    ) -> type[KVConnectorBaseType]:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L99-L104
        connector_name = kv_transfer_config.kv_connector
        if connector_name is None:
            raise ValueError("Connector name is not set in KVTransferConfig")
        connector_module_path = kv_transfer_config.kv_connector_module_path
        if connector_module_path is not None and not connector_module_path:
            raise ValueError("kv_connector_module_path cannot be an empty string.")
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L105-L114
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
            # SUBTRACTED: 旧 2 参构造的 supports_kw 告警段（L115-L123——
            #   废弃面；本章 connector 均为新 3 参签名）。
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L124-L127
        elif connector_name in cls._registry:
            connector_cls = cls._registry[connector_name]()
        else:
            raise ValueError(f"Unsupported connector type: {connector_name}")
        return connector_cls

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L130
    #   supports_hma_config——HMA 门判定
    @classmethod
    def supports_hma_config(cls, kv_transfer_config: "KVTransferConfig") -> bool:
        """Return whether this KV transfer config supports HMA.

        MultiConnector is a special case: the wrapper class implements
        SupportsHMA, but effective support depends on every configured child.
        """
        # SUBTRACTED: MultiConnector 特例分支（L137-L145——子连接器包装
        #   归 ch36；单 connector 判定即本体）。
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L137-L139
        connector_cls = cls.get_connector_class(kv_transfer_config)
        return supports_hma(connector_cls)


# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.

# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L152-L156
#   （唯一保留的注册行——懒加载机制的示范）
KVConnectorFactory.register_connector(
    "ExampleConnector",
    "implementation.example_connector",
    "ExampleConnector",
)

# SUBTRACTED: 其余 13 个后端注册行（L158-L242——第 2 条：NIXL/Mooncake/
#   LMCache/MoRIIO/Offloading/MultiConnector/FlexKV/HF3FS/SimpleCPUOffload/
#   DecodeBench/ExampleHiddenStates → ch36/37）。
