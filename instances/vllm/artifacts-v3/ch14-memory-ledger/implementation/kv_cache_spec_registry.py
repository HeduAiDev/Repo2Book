# SOURCE: vllm/v1/kv_cache_spec_registry.py
# KVCacheSpecRegistry——spec 类型 → manager 类 / 分组基类的注册表（m11
# 装配点）：register_all_kvcache_specs 在 engine core 进程先注册内置族
# （engine/core.py:L254），get_manager_for_kv_cache_spec 按类型查表建
# manager；uniform_type_base_spec 决定「同型层可并桶」（is_uniform_with_
# collection 的判定基准）。
# SUBTRACTED: @register_kv_cache_spec 装饰器（L188-L209——平台自定义 spec
#   的外挂注册面，current_platform.register_custom_kv_cache_specs 的
#   调用点在 register_all_kvcache_specs 内一并删——平台域）。
from dataclasses import dataclass


@dataclass(frozen=True)
# SOURCE: vllm/v1/kv_cache_spec_registry.py:L24 KVCacheSpecMetadata
class KVCacheSpecMetadata:
    """Metadata for a registered KVCacheSpec."""

    # SOURCE: vllm/v1/kv_cache_spec_registry.py:L28-L33
    kvcache_spec_cls: type
    manager_class: type
    # The base spec class for grouping compatibility checks.
    # KVCacheSpecs with the same uniform_type_base_spec will be
    # grouped into one kvcache group
    uniform_type_base_spec: type


# SOURCE: vllm/v1/kv_cache_spec_registry.py:L36 注册表本体
_REGISTRY_KVCACHESPEC_LIST: dict[type, KVCacheSpecMetadata] = {}


# SOURCE: vllm/v1/kv_cache_spec_registry.py:L39 KVCacheSpecRegistry
class KVCacheSpecRegistry:
    """Global registry for KVCacheSpec types and their associated managers."""

    # SOURCE: vllm/v1/kv_cache_spec_registry.py:L44 _ensure_registered
    @classmethod
    def _ensure_registered(cls, vllm_config=None) -> None:
        """
        Run full KVCacheSpec registration if the registration is not done.
        """
        # SOURCE: vllm/v1/kv_cache_spec_registry.py:L49-L50
        if _REGISTRY_KVCACHESPEC_LIST:
            return

        # SUBTRACTED: get_current_vllm_config_or_none 的 contextvar 取数
        #   （L52-L55——ch03 装配域；engine core 已先显式注册，此路兜底）
        # lazy import to avoid circular dependency
        from .single_type_kv_cache_manager import (
            register_all_kvcache_specs,
        )

        # SOURCE: vllm/v1/kv_cache_spec_registry.py:L62-L63
        register_all_kvcache_specs(vllm_config)

    # SOURCE: vllm/v1/kv_cache_spec_registry.py:L65 register
    @classmethod
    def register(
        cls,
        kvcache_spec_cls: type,
        manager_class: type | None = None,
        uniform_type_base_spec: type | None = None,
    ) -> None:
        """
        Register a KVCacheSpec class with its manager and base spec.

        Args:
            kvcache_spec_cls: The KVCacheSpec subclass to register
            manager_class: The SingleTypeKVCacheManager to use for this spec
            uniform_type_base_spec: The base spec class for grouping compatibility.
                instead of being grouped to different kvcache group, `kvcache_spec_cls`
                and `uniform_type_base_spec` will be trated as uniform type.
                If None, defaults to kvcache_spec_cls itself (for built-in base specs).
        """
        # SOURCE: vllm/v1/kv_cache_spec_registry.py:L85-L86
        assert manager_class is not None, "manager_class is required"
        if uniform_type_base_spec is None:
            uniform_type_base_spec = kvcache_spec_cls
        # SOURCE: vllm/v1/kv_cache_spec_registry.py:L88-L105（同键重注册
        #   幂等 / 冲突 assert）
        assert issubclass(kvcache_spec_cls, uniform_type_base_spec), (
            f"{kvcache_spec_cls.__name__} must inherit from its declared "
            f"uniform_type_base_spec {uniform_type_base_spec.__name__}."
        )

        if kvcache_spec_cls in _REGISTRY_KVCACHESPEC_LIST:
            registered_spec = _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls]
            is_same_registration = (
                manager_class == registered_spec.manager_class
                and uniform_type_base_spec == registered_spec.uniform_type_base_spec
            )
            assert is_same_registration, (
                f"Conflicting registration for KVCacheSpec "
                f": {kvcache_spec_cls.__name__}"
            )

        _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls] = KVCacheSpecMetadata(
            kvcache_spec_cls=kvcache_spec_cls,
            manager_class=manager_class,
            uniform_type_base_spec=uniform_type_base_spec,
        )

    # SOURCE: vllm/v1/kv_cache_spec_registry.py:L109 get_manager_class
    @classmethod
    def get_manager_class(
        cls, kv_cache_spec
    ) -> type | None:
        """
        Get the single type kvcache manager class for a given KVCacheSpec instance.

        Args:
            kv_cache_spec: A KVCacheSpec instance

        Returns:
            The SingleTypeKVCacheManager class to use for this kvcache_spec
        """
        # SOURCE: vllm/v1/kv_cache_spec_registry.py:L122-L129
        cls._ensure_registered()
        kvcache_spec_cls = type(kv_cache_spec)

        # Walk up the MRO to find a registered base class
        for base in kvcache_spec_cls.__mro__:
            if base in _REGISTRY_KVCACHESPEC_LIST:
                return _REGISTRY_KVCACHESPEC_LIST[base].manager_class

        return None

    # SOURCE: vllm/v1/kv_cache_spec_registry.py:L132 get_uniform_type_base_spec
    @classmethod
    def get_uniform_type_base_spec(
        cls,
        kv_cache_spec,
    ) -> type | None:
        """
        Get the base kvcache spec class for grouping compatibility checks.
        KVCacheSpecs with uniform_type_base_spec will be trated as one group.

        Args:
            kv_cache_spec: A KVCacheSpec instance to get the base spec class for

        Returns:
            The base KVCacheSpec class for checking uniform type kvcache specs
        """
        # SOURCE: vllm/v1/kv_cache_spec_registry.py:L152-L161
        cls._ensure_registered()
        kvcache_spec_cls = type(kv_cache_spec)

        # Walk up the MRO to find a registered base spec
        for base in kvcache_spec_cls.__mro__:
            if base in _REGISTRY_KVCACHESPEC_LIST:
                return _REGISTRY_KVCACHESPEC_LIST[base].uniform_type_base_spec

        return None

    # SOURCE: vllm/v1/kv_cache_spec_registry.py:L164 check_kv_cache_spec_registry
    @classmethod
    def check_kv_cache_spec_registry(
        cls,
        kv_cache_spec: dict,
    ) -> None:
        """
        Check if the KVCacheSpecs of each layer are registered as expected.
        """
        # SOURCE: vllm/v1/kv_cache_spec_registry.py:L172-L186
        cls._ensure_registered()
        for layer_name, spec in kv_cache_spec.items():
            # use raise instead of assert to make it effective in production environment
            if cls.get_uniform_type_base_spec(spec) is None:
                raise ValueError(
                    f"Unsupported KV cache spec type for layer {layer_name}: "
                    f"{type(spec)}. Please register it using "
                    f"@register_kv_cache_spec decorator."
                )
            if cls.get_manager_class(spec) is None:
                raise ValueError(
                    f"No manager found for KV cache spec type for layer "
                    f"{layer_name}: {type(spec)}. Please register it using "
                    f"@register_kv_cache_spec decorator."
                )
