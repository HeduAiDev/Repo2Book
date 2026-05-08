# SPDX-License-Identifier: Apache-2.0
"""
OffloadingSpecFactory — registry of offload backends.

REFERENCE: vllm/v1/kv_offload/factory.py:L17-L58

This module mirrors the lazy-loading registry pattern used in vLLM.
Each spec is registered by name + (module_path, class_name) tuple;
modules are imported only when the corresponding spec is created.

Why lazy: vLLM ships ~18 connector backends (LMCache, Mooncake, Nixl,
hf3fs, p2p, etc.) and each has heavy native dependencies (CUDA RDMA,
LMCache server, Mooncake transports). Importing all of them upfront
would slow startup by seconds and would crash if any optional
dependency is missing. Lazy loading keeps startup cheap.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from .offload_spec import OffloadingSpec


class OffloadingSpecFactory:
    """Registry of offload spec backends.

    Use `register_spec(name, module_path, class_name)` at module-load time;
    use `create_spec(name, **kwargs)` at runtime to instantiate.

    REFERENCE: vllm/v1/kv_offload/factory.py:L17-L58
    """

    # name → loader callable that returns the OffloadingSpec subclass
    _registry: dict[str, Callable[[], type[OffloadingSpec]]] = {}

    @classmethod
    def register_spec(
        cls, name: str, module_path: str, class_name: str
    ) -> None:
        """Register a spec with a lazy-loading module + class name.

        REFERENCE: vllm/v1/kv_offload/factory.py:L20-L30
        """
        if name in cls._registry:
            raise ValueError(f"Spec {name!r} is already registered.")

        def loader() -> type[OffloadingSpec]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def unregister_spec(cls, name: str) -> None:
        """Test helper — not in production vLLM but useful for fixtures."""
        cls._registry.pop(name, None)

    @classmethod
    def known_specs(cls) -> list[str]:
        return list(cls._registry)

    @classmethod
    def create_spec(cls, name: str, **kwargs: Any) -> OffloadingSpec:
        """Instantiate a registered spec.

        REFERENCE: vllm/v1/kv_offload/factory.py:L32-L52
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown spec {name!r}. Known: {cls.known_specs()}"
            )
        spec_cls = cls._registry[name]()
        assert issubclass(spec_cls, OffloadingSpec), (
            f"Registered class {spec_cls!r} is not an OffloadingSpec."
        )
        return spec_cls(**kwargs)


# REFERENCE: vllm/v1/kv_offload/factory.py:L55-L58 — canonical registration
# We register the CPUOffloadingSpec at module-load time, mirroring vLLM's
# bootstrap sequence. Any test can call `factory.create_spec("CPUOffloadingSpec",
# hash_block_size=16, ...)` and get a real instance.
OffloadingSpecFactory.register_spec(
    "CPUOffloadingSpec",
    "instances.vllm.artifacts.12-kv-offload.implementation.offload_spec",
    "CPUOffloadingSpec",
)
