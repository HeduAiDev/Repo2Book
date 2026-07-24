#!/usr/bin/env python3
"""
Driver for ch31 (two-frameworks strategy registry).

Loads the REAL third_party/ascend/backend/backend_register.py on host and
exercises the actual BackendStrategyRegistry:
  - m1: register (two-level dict build) + execute_func (lookup+call) + all raise branches
  - m2: dispatch via the SAME backend_policy resolution logic as utils.py:get_backend_func
        (that resolution is reproduced verbatim below because utils.py pulls heavy
         triton imports unavailable on host; the registry used is the real one and
         execute_func lookup is byte-identical to utils.py:L53).
  - m4: duplicate-def name rebind — module name `version_hash` vs the two distinct
        function objects the registry holds under ('mindspore'|'torch_npu','version_hash').

Only framework-free implementations are actually *called* (mindspore/cxx_abi -> 0,
and the pure-f-string capabilities header_file / allocate_memory / async_launch).
Capabilities whose bodies `import torch`/`import mindspore` (version_hash, type_convert)
are registered and looked up but not invoked on host — noted explicitly in output.
"""
import importlib.util
import json
import os

SRC = "/mnt/e/Laboratory/Repo2Book/instances/triton-ascend/source/third_party/ascend/backend/backend_register.py"

spec = importlib.util.spec_from_file_location("backend_register", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

reg = mod.backend_strategy_registry        # the _LazyBackendStrategyRegister singleton
inst = reg._get_instance()                 # the real BackendStrategyRegistry

out = {}

# ---- m1: registry shape -------------------------------------------------
out["m1_categories"] = inst.list_categories()
out["m1_methods_mindspore"] = inst.list_methods("mindspore")
out["m1_methods_torch_npu"] = inst.list_methods("torch_npu")
out["m1_n_categories"] = len(inst.list_categories())
out["m1_n_methods_per_cat"] = {c: len(inst.list_methods(c)) for c in inst.list_categories()}
out["m1_total_cells"] = sum(len(inst.list_methods(c)) for c in inst.list_categories())

# execute_func hit: mindspore/cxx_abi is framework-free -> real return
out["m1_execute_hit_mindspore_cxx_abi"] = inst.execute_func("mindspore", "cxx_abi")

# raise branch A: missing category
try:
    inst.execute_func("jax", "cxx_abi")
    out["m1_raise_missing_category"] = "NO RAISE (bug)"
except ValueError as e:
    out["m1_raise_missing_category"] = str(e)

# raise branch B: missing method under existing category
try:
    inst.execute_func("mindspore", "nonexistent_cap")
    out["m1_raise_missing_method"] = "NO RAISE (bug)"
except ValueError as e:
    out["m1_raise_missing_method"] = str(e)

# raise branch C: duplicate registration under existing (category, method)
try:
    @reg.register("mindspore", "cxx_abi")
    def _dup():
        return 999
    out["m1_raise_duplicate"] = "NO RAISE (bug)"
except ValueError as e:
    out["m1_raise_duplicate"] = str(e)

# ---- m2: dispatch = resolve backend_policy then execute_func -------------
def resolve_backend_policy(cache):
    """Byte-identical to utils.py:L42-L52 (backend_policy resolution)."""
    if cache["backend_policy"] is None:
        backend_policy_env = os.getenv("TRITON_BACKEND", "default").lower()
        if backend_policy_env == "torch_npu" or backend_policy_env == "mindspore":
            cache["backend_policy"] = backend_policy_env
        if cache["backend_policy"] is None:
            try:
                import torch          # noqa
                import torch_npu      # noqa
                cache["backend_policy"] = "torch_npu"
            except ImportError:
                cache["backend_policy"] = "mindspore"
    return cache["backend_policy"]

def get_backend_func(name, *args, cache=None, **kwargs):
    policy = resolve_backend_policy(cache)
    return reg.execute_func(policy, name, *args, **kwargs)

# Case 1: env forces mindspore, method=header_file (pure f-string, real return)
os.environ["TRITON_BACKEND"] = "mindspore"
cache_ms = {"backend_policy": None}
out["m2_case_ms_resolved_policy"] = resolve_backend_policy(cache_ms)
out["m2_case_ms_header_file"] = get_backend_func("header_file", True, cache=cache_ms)

# Case 2: env forces torch_npu, same method -> different real C++ string
os.environ["TRITON_BACKEND"] = "torch_npu"
cache_tn = {"backend_policy": None}
out["m2_case_tn_resolved_policy"] = resolve_backend_policy(cache_tn)
out["m2_case_tn_header_file"] = get_backend_func("header_file", True, cache=cache_tn)

# Case 3: cache stickiness — once resolved, env change is ignored
os.environ["TRITON_BACKEND"] = "torch_npu"
out["m2_case_cache_sticky_before"] = cache_ms["backend_policy"]
_ = get_backend_func("header_file", False, cache=cache_ms)  # env now torch_npu but cache=mindspore
out["m2_case_cache_sticky_after"] = cache_ms["backend_policy"]
out["m2_case_cache_sticky_header_file"] = get_backend_func("header_file", False, cache=cache_ms)

# also show allocate_memory divergence (both pure f-string) for richness
out["m2_allocate_memory_mindspore"] = reg.execute_func("mindspore", "allocate_memory", 4096, "stream_ptr")
out["m2_allocate_memory_torch_npu"] = reg.execute_func("torch_npu", "allocate_memory", 4096, "stream_ptr")
del os.environ["TRITON_BACKEND"]

# ---- m4: duplicate-def name rebind --------------------------------------
ms_vh = inst.strategies["mindspore"]["version_hash"]
tn_vh = inst.strategies["torch_npu"]["version_hash"]
module_name_vh = mod.version_hash   # the module-level name after both defs executed
out["m4_registry_ms_vh_id"] = id(ms_vh)
out["m4_registry_tn_vh_id"] = id(tn_vh)
out["m4_module_name_vh_id"] = id(module_name_vh)
out["m4_ms_vs_tn_distinct"] = (ms_vh is not tn_vh)
# module name `version_hash` == the LAST def (torch_npu one), not the mindspore one
out["m4_module_name_is_torch_version"] = (module_name_vh is tn_vh)
out["m4_module_name_is_mindspore_version"] = (module_name_vh is ms_vh)
# registry keeps BOTH; each carries its own __wrapped body (co_consts shows the import target)
out["m4_ms_vh_consts_has_mindspore"] = ("mindspore" in [c for c in ms_vh.__code__.co_names])
out["m4_tn_vh_consts_has_torch_npu"] = ("torch_npu" in [c for c in tn_vh.__code__.co_names])

print(json.dumps(out, indent=2, ensure_ascii=False))
