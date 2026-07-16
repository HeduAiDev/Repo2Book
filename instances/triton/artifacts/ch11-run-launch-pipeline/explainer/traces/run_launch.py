#!/usr/bin/env python
"""
ch11 explainer trace driver — headless pin-compile of a real Triton kernel.

Host has NO GPU. pip-installed triton==3.2.0 is byte-identical to pinned v3.2.0
Python front-end (INSTANCE.md). We stub `driver.active` so JITFunction.run(...,
warmup=True) runs the ENTIRE run() body EXCEPT the `if not warmup` tail (grid
canonicalization + C++ launcher emission — both need a real device). That yields,
on a REAL compile:
  - miss vs hit end-to-end wall time  (run-orchestration-spine / runtime-cache-lookup
    / launch-overhead-anatomy)
  - the binder 5-tuple + cache key string  (lazy-binder-invocation)
  - compile slow-path products  (compile-slowpath-orchestration)
The None->*i8 fixup and grid canonicalization are pure Python; we exercise them with
the SAME code the source uses (jit.py:L604-606 / L640-649). The real cross-language
launcher emission needs a GPU and is NOT run (reported honestly, module/function stay
None post-compile — proof the device handles are lazy).
"""
import os, time, json, shutil
os.environ.setdefault("TRITON_DEBUG", "0")

# Clear the on-disk FileCacheManager cache so a memory-cache MISS also misses the
# disk layer and triggers a TRUE full compile (make_ir -> lowering -> ptxas -> cubin).
# The disk cache is ch14/ch12 territory; here we want "miss" == real compile cost.
_cache_dir = os.path.expanduser(os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache"))
if os.path.isdir(_cache_dir):
    shutil.rmtree(_cache_dir, ignore_errors=True)

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.runtime import driver as _drivermod

REAL_TARGET = GPUTarget("cuda", 80, 32)  # sm_80 (A100), warp_size=32

class FakeDriver:
    """Stand-in for driver.active so run(warmup=True) works headless."""
    def get_current_device(self):      return 0
    def get_current_stream(self, dev): return 0
    def get_current_target(self):      return REAL_TARGET

_drivermod.active = FakeDriver()   # install BEFORE any run()

class FDtype:
    def __init__(self, name): self.name = name
    def __str__(self):        return "torch." + self.name
class FakePtr:
    """Duck-types a tensor for the binder: data_ptr() (alignment) + dtype (mangle)."""
    def __init__(self, addr, dtype): self._a = addr; self.dtype = dtype
    def data_ptr(self):              return self._a

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

fp32 = FDtype("float32")
x = FakePtr(0x1000, fp32)   # 0x1000 % 16 == 0 -> aligned -> spec key "D"
y = FakePtr(0x2000, fp32)
o = FakePtr(0x3000, fp32)
N = 1024                     # 1024 % 16 == 0 -> spec key "D"; int -> "i32"
BLOCK = 256
grid = (triton.cdiv(N, BLOCK),)   # (4,)

rec = {"target": f"{REAL_TARGET.backend} sm={REAL_TARGET.arch} warp={REAL_TARGET.warp_size}",
       "N": N, "BLOCK": BLOCK, "grid": list(grid)}

jf = add_kernel
rec["binder_before_first_run"] = "None (lazy)" if jf.binder is None else "built"

# ---------- Phase A: first warmup run = cache MISS (triggers real compile) ----------
t0 = time.perf_counter()
k1 = jf.run(x, y, o, N, BLOCK=BLOCK, grid=grid, warmup=True)
miss_ms = (time.perf_counter() - t0) * 1e3
rec["miss_wall_ms"] = round(miss_ms, 3)
rec["binder_after_first_run"] = "built" if jf.binder is not None else "None"
dev = 0
rec["cache_entries_after_miss"] = len(jf.cache[dev])
rec["cache_keys"] = list(jf.cache[dev].keys())

# ---------- Phase B: repeated warmup runs, same args = cache HIT (no compile) ----------
REP = 2000
t0 = time.perf_counter()
for _ in range(REP):
    k2 = jf.run(x, y, o, N, BLOCK=BLOCK, grid=grid, warmup=True)
hit_us = (time.perf_counter() - t0) / REP * 1e6
rec["hit_wall_us_avg"] = round(hit_us, 3)
rec["hit_rep"] = REP
rec["hit_returns_same_object"] = (k2 is k1)
rec["miss_over_hit_ratio"] = round((miss_ms * 1e3) / hit_us, 1)

# ---------- Phase C: binder 5-tuple + cache key ----------
# run() injects kwargs["debug"]=False before calling the binder (jit.py:L564),
# so debug lands in excess_kwargs and the stored key carries {'debug': False}.
bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = jf.binder(
    x, y, o, N, BLOCK=BLOCK, debug=False)
rec["binder_bound_args_names"] = list(bound_args.keys())
rec["binder_sig_and_spec"] = list(sig_and_spec)
rec["binder_constexpr_vals"] = list(constexpr_vals)
rec["binder_non_constexpr_count"] = len(non_constexpr_vals)
rec["binder_excess_kwargs"] = dict(excess_kwargs)
key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
rec["cache_key_string"] = key
rec["cache_key_matches_stored"] = (key in jf.cache[dev])

# A DIFFERENT key: change BLOCK constexpr -> new cache entry = a second MISS that
# also misses the (freshly cleared) disk cache, so it is a TRUE full compile. The
# process is already warm (ptxas/LLVM loaded), so this isolates steady-state compile
# cost from the one-time first-compile-in-process warmup baked into miss_wall_ms.
t0 = time.perf_counter()
_ = jf.run(x, y, o, N, BLOCK=128, grid=grid, warmup=True)
rec["second_miss_wall_ms"] = round((time.perf_counter() - t0) * 1e3, 3)
rec["cache_entries_after_second_block"] = len(jf.cache[dev])
_, s2, ce2, _, ek2 = jf.binder(x, y, o, N, BLOCK=128, debug=False)
rec["cache_key_block128"] = ''.join(s2) + str((ce2, ek2))
rec["second_miss_over_hit_ratio"] = round((rec["second_miss_wall_ms"] * 1e3) / hit_us, 1)

# ---------- Phase D: None->*i8 signature fixup (source-faithful, jit.py:L604-606) ----------
@triton.jit
def opt_kernel(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    v = tl.load(a_ptr + pid, mask=pid < n)
    tl.store(a_ptr + pid, v, mask=pid < n)

_ = opt_kernel.run(x, None, N, BLOCK=BLOCK, grid=(1,), warmup=True)
ba2, sig2, cev2, ncv2, ek2b = opt_kernel.binder(x, None, N, BLOCK=BLOCK)
non_ce_idx = opt_kernel.non_constexpr_indices
sigkeys = [opt_kernel.params[i].name for i in non_ce_idx]
sigvals = sig2[:len(sigkeys)]
signature = {kk: ('*i8' if (v == 'none') else v) for (kk, v) in zip(sigkeys, sigvals)}
rec["none_fixup_sig_and_spec"] = list(sig2)
rec["none_fixup_signature_dict"] = signature

# ---------- Phase E: grid canonicalization (source-faithful, jit.py:L640-649) ----------
def canon(g, bargs):
    if callable(g):
        g = g(bargs)
    gs = len(g)
    return (g[0], g[1] if gs > 1 else 1, g[2] if gs > 2 else 1)
rec["grid_1tuple_in"] = [4];        rec["grid_1tuple_out"] = list(canon((4,), bound_args))
rec["grid_2tuple_in"] = [4, 2];     rec["grid_2tuple_out"] = list(canon((4, 2), bound_args))
rec["grid_3tuple_in"] = [4, 2, 3];  rec["grid_3tuple_out"] = list(canon((4, 2, 3), bound_args))
rec["grid_callable_in"] = "lambda meta: (meta['n_elements'] // meta['BLOCK'] + 1,)"
rec["grid_callable_out"] = list(canon(lambda m: (m['n_elements'] // m['BLOCK'] + 1,), bound_args))

# ---------- Phase F: compile products + lazy device handles (proof) ----------
md = k1.metadata
rec["compiled_kernel_name"] = k1.name
rec["metadata_num_warps"] = md.num_warps
rec["metadata_num_stages"] = md.num_stages
rec["metadata_shared_bytes"] = md.shared
rec["asm_stages"] = list(k1.asm.keys())
rec["ttir_len_chars"] = len(k1.asm.get("ttir", ""))
# Lazy device handles: module/function must still be None after compile (no GPU touched)
rec["module_after_compile"] = repr(k1.module)
rec["function_after_compile"] = repr(k1.function)

print(json.dumps(rec, indent=2, default=str))
