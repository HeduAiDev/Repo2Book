#!/usr/bin/env python3
"""
ch26 explainer driver — manual/host reproduction of the real hash & parse logic.

The real NPUOptions dataclass (third_party/ascend/backend/compiler.py:L705-810) has
~80 fields and pulls in ascend-only imports (get_libdevice / is_compile_on_910_95 from
triton.tools.get_ascend_devices), so it cannot be imported on a plain host. But the
teaching mechanisms are pure control flow:

  - NPUOptions.hash (compiler.py:L810-812):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
  - AscendBackend.hash (compiler.py:L971-974):
        return str(self.target)          # GPUTarget frozen dataclass
  - parse_options (compiler.py:L888-903): keep only kwargs whose key is a NPUOptions
        field name; setdefault("arch", target.arch).

This script reproduces each byte-for-byte with a SMALL illustrative field set so the
digests are REAL sha256 outputs the reader could recompute, while staying hand-checkable.
Digests differ from the real ~80-field object (which we cannot instantiate); that is
noted in explainer.json. The MECHANISM (join → sha256 avalanche; str(target)) is exact.
"""
import hashlib
import json
from dataclasses import dataclass


# ---- 1. NPUOptions.hash: exact algorithm, reduced illustrative field set ----------
def npuoptions_hash(fields: dict) -> str:
    # byte-for-byte the real line: compiler.py:L811-812
    key = "_".join([f"{name}-{val}" for name, val in fields.items()])
    return key, hashlib.sha256(key.encode("utf-8")).hexdigest()


# a tiny, hand-checkable subset of the real NPUOptions defaults (values verbatim from
# compiler.py:L706-728). Order = insertion order = dataclass field order.
base = {
    "debug": False,
    "num_warps": 32,
    "num_stages": 1,
    "arch": "Ascend910B",
    "compile_mode": "simd",
}
# same options except num_warps flipped 32 -> 16 (a single knob change)
warps16 = dict(base, num_warps=16)
# same options except target arch 910B -> 950
arch950 = dict(base, arch="Ascend950")

k_base, h_base = npuoptions_hash(base)
k_warps16, h_warps16 = npuoptions_hash(warps16)
k_arch950, h_arch950 = npuoptions_hash(arch950)


# ---- 2. AscendBackend.hash = str(GPUTarget) ---------------------------------------
@dataclass(frozen=True)
class GPUTarget:  # mirrors python/triton/backends/compiler.py:L217-223
    backend: str
    arch: str
    warp_size: int


t_910b = GPUTarget("npu", "Ascend910B", 0)
t_950 = GPUTarget("npu", "Ascend950", 0)
backend_hash_910b = str(t_910b)   # AscendBackend.hash body: return str(self.target)
backend_hash_950 = str(t_950)


# ---- 3. parse_options filtering ---------------------------------------------------
# real field-name gate: {k: opts[k] for k in NPUOptions.__dataclass_fields__ if k in opts}
NPUOPTIONS_FIELDS = set(base.keys()) | {"num_ctas", "cluster_dims", "enable_fp_fusion"}
user_kwargs = {
    "num_warps": 8,        # valid  -> kept
    "num_stages": 2,       # valid  -> kept
    "enable_fp_fusion": False,  # valid -> kept
    "block_size": 128,     # NOT a NPUOptions field -> dropped
    "foo": "bar",          # junk    -> dropped
}
kept = {k: user_kwargs[k] for k in NPUOPTIONS_FIELDS if k in user_kwargs}
# arch not in kwargs -> setdefault from target.arch
kept.setdefault("arch", t_910b.arch)
dropped = [k for k in user_kwargs if k not in NPUOPTIONS_FIELDS]


# ---- emit trace -------------------------------------------------------------------
def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


out = {
    "npuoptions_hash": {
        "base": {"key_string": k_base, "digest": h_base},
        "num_warps_32to16": {"key_string": k_warps16, "digest": h_warps16},
        "arch_910B_to_950": {"key_string": k_arch950, "digest": h_arch950},
        "avalanche": {
            "base_vs_warps16_prefix_hexchars_shared": common_prefix_len(h_base, h_warps16),
            "base_vs_arch950_prefix_hexchars_shared": common_prefix_len(h_base, h_arch950),
            "digest_len_hexchars": len(h_base),
        },
    },
    "backend_hash": {
        "target_910B": backend_hash_910b,
        "target_950": backend_hash_950,
        "equal": backend_hash_910b == backend_hash_950,
    },
    "parse_options": {
        "user_kwargs": user_kwargs,
        "kept": kept,
        "dropped": dropped,
        "arch_via_setdefault": kept["arch"],
    },
}
print(json.dumps(out, indent=2, default=str))
