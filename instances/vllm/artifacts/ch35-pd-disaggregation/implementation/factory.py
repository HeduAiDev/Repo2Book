# SPDX-License-Identifier: Apache-2.0
# Subtract-only companion for ch29《PD 分离的抽象与调度器集成》.
# 只做减法：与 vLLM 同名/同结构/同控制流，只删不增。
#
# 本文件是 vllm/distributed/kv_transfer/kv_connector/factory.py 的子集：
# 保留懒加载注册表 + 按 role 单独构造两份实例的 create_connector。
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import KVConnectorBase_V1, KVConnectorRole

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

# SUBTRACTED: supports_hma / supports_kw import 及兼容旧两参签名的 compat 路径
# （_get_connector_class_with_compat 中外部 module_path 优先、supports_kw 探测）
# 与本章 role-split 主线无关；精简版只走『内部注册表 + 新三参签名』一条路
# （原 factory.py:L8-17, L102-134）。


# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L27
class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBase_V1]]] = {}

    @classmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L30
    def register_connector(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L36
        def loader() -> type[KVConnectorBase_V1]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L42
    def create_connector(
        cls,
        config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> KVConnectorBase_V1:
        kv_transfer_config = config.kv_transfer_config
        if kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set to create a connector")
        connector_cls = cls.get_connector_class_by_name(kv_transfer_config.kv_connector)

        # SUBTRACTED: HMA 支持校验（disable_hybrid_kv_cache_manager 取反后断言
        # supports_hma）+ logger.info；HMA 与 PD 分离正交（原 L56-68）。

        # NOTE(Kuntai): v1 connector is explicitly separated into two roles.
        # Scheduler connector:
        # - Co-locate with scheduler process
        # - Should only be used inside the Scheduler class
        # Worker connector:
        # - Co-locate with worker process
        # - Should only be used inside the forward context & attention layer
        # We build separately to enforce strict separation
        return connector_cls(config, role, kv_cache_config)
        # SUBTRACTED: compat_sig 旧两参签名分支（connector_cls(config, role)）；
        # 精简版只演示新三参签名（原 L77-82）。

    @classmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L77
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBase_V1]:
        """Get a registered connector class by name."""
        if connector_name not in cls._registry:
            raise ValueError(f"Connector '{connector_name}' is not registered.")
        return cls._registry[connector_name]()


# SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:L135
# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.
KVConnectorFactory.register_connector(
    "ExampleConnector",
    "implementation.example_connector",
    "ExampleConnector",
)
# SUBTRACTED: 真实 vLLM 还注册十余种重依赖 connector（Nixl/LMCache/Mooncake/
# Offloading/MoRIIO/...）；懒加载注册表的意义正在于只 import 选中的那一个，
# 精简版保留 ExampleConnector 一项即可演示注册+懒加载（原 L155-228）。
