"""Tests for OffloadingSpecFactory — lazy-loading registry."""

from __future__ import annotations

import pytest

from implementation.factory import OffloadingSpecFactory
from implementation.offload_spec import CPUOffloadingSpec, OffloadingSpec


# Re-register the canonical spec under the package path that resolves
# in the test environment (factory module's auto-registration uses the
# project's fully-qualified path which is not on the test sys.path).
OffloadingSpecFactory.unregister_spec("CPUOffloadingSpec")
OffloadingSpecFactory.register_spec(
    "CPUOffloadingSpec",
    "implementation.offload_spec",
    "CPUOffloadingSpec",
)


@pytest.fixture
def fresh_registry_name():
    """Yield a unique registry name; clean up after the test."""
    name = "TestSpec_xyz"
    yield name
    OffloadingSpecFactory.unregister_spec(name)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestRegistration:
    def test_canonical_cpu_spec_registered(self):
        """CPUOffloadingSpec auto-registered at module-load time."""
        known = OffloadingSpecFactory.known_specs()
        assert "CPUOffloadingSpec" in known

    def test_double_registration_raises(self):
        """Registering an existing name raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            OffloadingSpecFactory.register_spec(
                "CPUOffloadingSpec",
                "instances.vllm.artifacts.12-kv-offload.implementation.offload_spec",
                "CPUOffloadingSpec",
            )

    def test_known_specs_returns_list(self):
        """known_specs returns a list of registered names."""
        names = OffloadingSpecFactory.known_specs()
        assert isinstance(names, list)
        assert len(names) >= 1

    def test_register_new_spec(self, fresh_registry_name):
        """register_spec adds a new entry to the registry."""
        OffloadingSpecFactory.register_spec(
            fresh_registry_name,
            "implementation.offload_spec",
            "CPUOffloadingSpec",
        )
        assert fresh_registry_name in OffloadingSpecFactory.known_specs()


# ---------------------------------------------------------------------------
# create_spec
# ---------------------------------------------------------------------------
class TestCreateSpec:
    def test_unknown_spec_raises(self):
        """Creating an unknown spec raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown spec"):
            OffloadingSpecFactory.create_spec("NoSuchSpec")

    def test_create_cpu_offloading_spec(self):
        """Create CPUOffloadingSpec via factory; verify type."""
        spec = OffloadingSpecFactory.create_spec(
            "CPUOffloadingSpec",
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=1024,
            cpu_bytes_to_use=1024 * 50,
        )
        assert isinstance(spec, CPUOffloadingSpec)
        assert isinstance(spec, OffloadingSpec)

    def test_create_with_arc_policy(self):
        """Factory passes kwargs through; eviction_policy='arc' wires ARC."""
        spec = OffloadingSpecFactory.create_spec(
            "CPUOffloadingSpec",
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=1024,
            cpu_bytes_to_use=1024 * 50,
            eviction_policy="arc",
        )
        mgr = spec.get_manager()
        assert mgr.policy_name() == "ARCCachePolicy"


# ---------------------------------------------------------------------------
# Lazy loading — module imported only on first create
# ---------------------------------------------------------------------------
class TestLazyLoading:
    def test_loader_invoked_only_on_create(self, fresh_registry_name):
        """register_spec stores a closure; module import happens at create_spec."""
        # Register a spec pointing to a non-existent module — register succeeds,
        # create fails. This proves laziness.
        OffloadingSpecFactory.register_spec(
            fresh_registry_name,
            "no.such.module",
            "ClassName",
        )
        # Registration should NOT have raised
        assert fresh_registry_name in OffloadingSpecFactory.known_specs()
        # Creation should now fail at import time
        with pytest.raises((ImportError, ModuleNotFoundError)):
            OffloadingSpecFactory.create_spec(fresh_registry_name)

    def test_unregister_cleanup(self, fresh_registry_name):
        """unregister_spec removes the entry."""
        OffloadingSpecFactory.register_spec(
            fresh_registry_name,
            "implementation.offload_spec",
            "CPUOffloadingSpec",
        )
        OffloadingSpecFactory.unregister_spec(fresh_registry_name)
        assert fresh_registry_name not in OffloadingSpecFactory.known_specs()
