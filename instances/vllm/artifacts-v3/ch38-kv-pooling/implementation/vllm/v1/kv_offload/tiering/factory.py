# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SecondaryTierFactory：tier 类型注册（fs/p2p 示范）。

# SOURCE: vllm/v1/kv_offload/tiering/factory.py:L1-L79
# SUBTRACTED: obj（S3 对象存储）与 example 两条注册行——减法计划删除项 3：
#   分层语义由 fs（线程池文件 I/O）+ p2p（网络）两种 secondary 完整示范，
#   注册机制由这两条示范。
"""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.v1.kv_offload.tiering.base import SecondaryTierManager

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec


# SOURCE: vllm/v1/kv_offload/tiering/factory.py:L13-L54
class SecondaryTierFactory:
    # SOURCE: vllm/v1/kv_offload/tiering/factory.py:L14
    _registry: dict[str, Callable[[], type[SecondaryTierManager]]] = {}

    # SOURCE: vllm/v1/kv_offload/tiering/factory.py:L16-L25
    @classmethod
    def register_tier(cls, tier_type: str, module_path: str, class_name: str) -> None:
        # SOURCE: vllm/v1/kv_offload/tiering/factory.py:L17-L25
        if tier_type in cls._registry:
            raise ValueError(f"Tier '{tier_type}' is already registered.")

        def loader() -> type[SecondaryTierManager]:
            # SOURCE: vllm/v1/kv_offload/tiering/factory.py:L21-L24
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[tier_type] = loader

    # SOURCE: vllm/v1/kv_offload/tiering/factory.py:L27-L42
    @classmethod
    def create_secondary_tier(
        cls,
        tier_config: dict,
        primary_kv_view: memoryview,
        offloading_spec: "OffloadingSpec",
    ) -> SecondaryTierManager:
        # SOURCE: vllm/v1/kv_offload/tiering/factory.py:L28-L42
        tier_cls = cls.get_tier_class(tier_config)
        config = tier_config.copy()
        tier_type = config.pop("type")
        return tier_cls(
            offloading_spec=offloading_spec,
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
            **config,
        )

    # SOURCE: vllm/v1/kv_offload/tiering/factory.py:L44-L54
    @classmethod
    def get_tier_class(cls, tier_config: dict) -> type[SecondaryTierManager]:
        # SOURCE: vllm/v1/kv_offload/tiering/factory.py:L45-L54
        tier_type = tier_config.get("type")
        if not tier_type:
            raise ValueError("Secondary tier configuration must include 'type'")
        if tier_type not in cls._registry:
            raise ValueError(
                f"Unknown secondary tier type: {tier_type!r}. "
                f"Supported types: {list(cls._registry)}"
            )
        return cls._registry[tier_type]()


# SUBTRACTED: example 层注册行（L57-L61）——删除项 3。
# SOURCE: vllm/v1/kv_offload/tiering/factory.py:L63-L67
SecondaryTierFactory.register_tier(
    "fs",
    "vllm.v1.kv_offload.tiering.fs.manager",
    "FileSystemTierManager",
)

# SOURCE: vllm/v1/kv_offload/tiering/factory.py:L69-L73
SecondaryTierFactory.register_tier(
    "p2p",
    "vllm.v1.kv_offload.tiering.p2p.manager",
    "P2PSecondaryTierManager",
)

# SUBTRACTED: obj 层注册行（L75-L79）——删除项 3（S3 对象存储第三种介质，
#   协议与 fs/p2p 同构）。
