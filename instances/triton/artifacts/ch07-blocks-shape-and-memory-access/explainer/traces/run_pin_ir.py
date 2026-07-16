#!/usr/bin/env python3
"""Pin-compile driver (triton==3.2.0, byte-identical to pin 9641643) — trace-period IR.

Captures TRACE-PERIOD IR (ASTSource.make_ir, before ANY pass) for ch07 mechanisms:
program_id / arange / full, legacy load (ptr+mask), block pointer (make_block_ptr),
atomic_add, and the fp16-non-add type-check error. Headless, no GPU.

Run:  instances/triton/v32/bin/python run_pin_ir.py
"""
import json
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend
from triton.backends.compiler import GPUTarget
from triton._C.libtriton import ir

OUT = {}


def trace_ir(fn, signature, constants):
    """Return trace-period TTIR text (make_ir, before ANY optimization pass).

    Mirrors triton.compiler.compiler.compile() lines L267-L273 at the pin, but
    stops right after src.make_ir — i.e. the AST->TTIR translation, before
    add_stages()'s passes (add_inliner is the first make_ttir pass, not run here).
    """
    src = ASTSource(fn=fn, signature=signature, constants=constants)
    target = GPUTarget("cuda", 90, 32)  # Hopper sm_90, warp=32; trace IR is target-agnostic
    backend = make_backend(target)
    opts = backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    codegen_fns = backend.get_codegen_implementation()
    module_map = backend.get_module_map()
    mod = src.make_ir(opts, codegen_fns, module_map, ctx)
    return str(mod)


# ---- m1: grid coords -> block tensor (program_id / arange / full) ----
@triton.jit
def k_grid_to_block(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    npg = tl.num_programs(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # block-inner coordinate axis
    ones = tl.full((BLOCK,), 1, dtype=tl.int32)   # constant block via splat
    tl.store(out_ptr + offs, ones + npg)


OUT["m1_grid_to_block"] = trace_ir(
    k_grid_to_block, {"out_ptr": "*i32", "BLOCK": "constexpr"}, {"BLOCK": 8})


# ---- m6: legacy per-element load (ptr tensor + mask + other) ----
@triton.jit
def k_legacy_load(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n                               # tail-block boundary mask
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


OUT["m6_legacy_load"] = trace_ir(
    k_legacy_load, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"}, {"BLOCK": 8})


# ---- m7: block pointer (make_block_ptr / boundary_check) ----
@triton.jit
def k_block_ptr(x_ptr, out_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bp = tl.make_block_ptr(
        base=x_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    x = tl.load(bp, boundary_check=(0, 1), padding_option="zero")
    ob = tl.make_block_ptr(
        base=out_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(ob, x, boundary_check=(0, 1))


OUT["m7_block_ptr"] = trace_ir(
    k_block_ptr,
    {"x_ptr": "*fp32", "out_ptr": "*fp32", "M": "i32", "N": "i32", "BM": "constexpr", "BN": "constexpr"},
    {"BM": 16, "BN": 16})


# ---- m8: coalescing — contiguous vs strided address expression (trace IR) ----
@triton.jit
def k_coalesced(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)                     # stride 1: lane i -> base + i
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x)


@triton.jit
def k_strided(x_ptr, out_ptr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK) * STRIDE            # stride S: lane i -> base + i*S
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x)


OUT["m8_coalesced"] = trace_ir(
    k_coalesced, {"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "constexpr"}, {"BLOCK": 8})
OUT["m8_strided"] = trace_ir(
    k_strided, {"x_ptr": "*fp32", "out_ptr": "*fp32", "STRIDE": "constexpr", "BLOCK": "constexpr"},
    {"STRIDE": 4, "BLOCK": 8})


# ---- m10/m11: atomic_add (RMW op) + fp16-non-add type-check error ----
@triton.jit
def k_atomic_add(ptr, val_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(val_ptr + offs)
    tl.atomic_add(ptr + offs, v, sem="relaxed", scope="gpu")


OUT["m10_atomic_add"] = trace_ir(
    k_atomic_add, {"ptr": "*fp32", "val_ptr": "*fp32", "BLOCK": "constexpr"}, {"BLOCK": 8})


@triton.jit
def k_atomic_max_fp16(ptr, val_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(val_ptr + offs)
    tl.atomic_max(ptr + offs, v)   # fp16 + op != 'add' -> should raise


try:
    trace_ir(k_atomic_max_fp16, {"ptr": "*fp16", "val_ptr": "*fp16", "BLOCK": "constexpr"}, {"BLOCK": 8})
    OUT["m9_fp16_max_error"] = "NO ERROR (unexpected)"
except Exception as e:
    OUT["m9_fp16_max_error"] = f"{type(e).__name__}: {e}"


# ---- m9: atomic_add on fp16 IS allowed (only non-add rejected) ----
@triton.jit
def k_atomic_add_fp16(ptr, val_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    v = tl.load(val_ptr + offs)
    tl.atomic_add(ptr + offs, v)


try:
    OUT["m9_fp16_add_ok"] = "OK: " + trace_ir(
        k_atomic_add_fp16, {"ptr": "*fp16", "val_ptr": "*fp16", "BLOCK": "constexpr"}, {"BLOCK": 8}).splitlines()[0]
except Exception as e:
    OUT["m9_fp16_add_ok"] = f"{type(e).__name__}: {e}"


for k, v in OUT.items():
    print("=" * 70)
    print("###", k)
    print(v)

with open("pin_ir.json", "w") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False)
print("\n[written pin_ir.json]")
