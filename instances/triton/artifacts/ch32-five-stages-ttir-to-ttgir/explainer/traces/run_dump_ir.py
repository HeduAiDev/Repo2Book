#!/usr/bin/env python3
"""ch32 explainer trace driver — pin-compile (triton==3.2.0, byte-identical to pin).

Dumps TTIR (after make_ttir) and TTGIR (after the first hop make_ttgir /
add_convert_to_ttgpuir) for a small matmul kernel that uses BLOCK POINTERS
(tt.make_tensor_ptr) and tt.dot, so we can observe on a real IR:

  1. block pointer降解 by RewriteTensorPointer (tt.make_tensor_ptr present in
     the raw traced IR, GONE by the .ttir dump) -> m8
  2. every layoutless tensor贴上 default #blocked in TTGIR -> m3
  3. tt.dot's two operands forced to #ttg.dot_op (DotOperand) + convert_layout
     glue inserted around it -> m5 / m4
  4. the module attrs num-warps / threads-per-warp / num-ctas -> m4

Runs headless on host CPU path (no kernel launch needed; we only need the
compiler front-end + MLIR passes, which run on CPU).
"""
import json, os, sys

# make the compile deterministic / headless
os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")

import triton
import triton.language as tl
from triton.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget

OUT = os.path.join(os.path.dirname(__file__), "dump_ir.json")

BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16


@triton.jit
def matmul_bp(a_ptr, b_ptr, c_ptr, M, N, K,
              sm: tl.constexpr, sn: tl.constexpr, sk: tl.constexpr):
    # block pointers -> exercised RewriteTensorPointer at TTIR level
    a_bp = tl.make_block_ptr(a_ptr, shape=(M, K), strides=(sk, 1),
                             offsets=(0, 0), block_shape=(sm, sk),
                             order=(1, 0))
    b_bp = tl.make_block_ptr(b_ptr, shape=(K, N), strides=(sn, 1),
                             offsets=(0, 0), block_shape=(sk, sn),
                             order=(1, 0))
    a = tl.load(a_bp)
    b = tl.load(b_bp)
    c = tl.dot(a, b)                          # tt.dot -> forces DotOperand
    c_bp = tl.make_block_ptr(c_ptr, shape=(M, N), strides=(sn, 1),
                             offsets=(0, 0), block_shape=(sm, sn),
                             order=(1, 0))
    tl.store(c_bp, c)


def main():
    NUM_WARPS = 4
    THREADS_PER_WARP = 32
    NUM_CTAS = 1
    CAPABILITY = 80  # sm_80 (A100 class); layout inference is capability-agnostic here

    sig = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32",
           "M": "i32", "N": "i32", "K": "i32",
           "sm": "constexpr", "sn": "constexpr", "sk": "constexpr"}
    constexprs = {"sm": BLOCK_M, "sn": BLOCK_N, "sk": BLOCK_K}

    src = ASTSource(fn=matmul_bp, signature=sig, constants=constexprs)

    target = GPUTarget("cuda", CAPABILITY, THREADS_PER_WARP)
    backend = make_backend(target)
    options = backend.parse_options({
        "num_warps": NUM_WARPS,
        "num_ctas": NUM_CTAS,
    })

    # ---- 1. raw traced IR (before ANY pass) : make_ir ----
    from triton.compiler.compiler import ir as _ir
    context = _ir.context()
    _ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation()
    module_map = backend.get_module_map()
    traced = src.make_ir(options, codegen_fns, module_map, context)
    traced_ttir = str(traced)

    # ---- 2. after make_ttir ----
    stages = {}
    backend.add_stages(stages, options)
    metadata = {}
    ttir_mod = stages["ttir"](traced, metadata)
    ttir_txt = str(ttir_mod)

    # ---- 3a. FIRST HOP ONLY: run just add_convert_to_ttgpuir on a fresh
    #          clone of the TTIR, mirroring the FIRST line of make_ttgir.
    #          This isolates the ch32 subject (layout injection) from the
    #          downstream ttgir opt passes (coalesce/accelerate_matmul = ch33+).
    from triton.backends.nvidia.compiler import ir as nir  # same libtriton ir
    from triton._C.libtriton import passes as _passes  # noqa
    # re-parse the ttir text into a fresh module so we don't mutate ttir_mod
    hop_mod = ttir_mod  # first-hop pass runs in place; ttir_txt already captured
    pm = _ir.pass_manager(hop_mod.context)
    pm.enable_debug()
    from triton._C.libtriton import passes as P
    P.ttir.add_convert_to_ttgpuir(pm, f"cuda:{CAPABILITY}", NUM_WARPS, 32, NUM_CTAS)
    pm.run(hop_mod)
    ttgir_first_hop = str(hop_mod)

    # ---- 3b. after full make_ttgir (first hop + downstream ttgir opt = ch33+) ----
    #      recompute from a fresh trace so the full pipeline isn't polluted by 3a
    context2 = _ir.context()
    _ir.load_dialects(context2)
    backend.load_dialects(context2)
    traced2 = src.make_ir(options, codegen_fns, module_map, context2)
    stages2 = {}
    backend.add_stages(stages2, options)
    md2 = {}
    ttir2 = stages2["ttir"](traced2, md2)
    ttgir_mod = stages2["ttgir"](ttir2, md2)
    ttgir_txt = str(ttgir_mod)

    result = {
        "params": {
            "kernel": "matmul_bp (BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d)" % (BLOCK_M, BLOCK_N, BLOCK_K),
            "num_warps": NUM_WARPS, "threads_per_warp": THREADS_PER_WARP,
            "num_ctas": NUM_CTAS, "capability": CAPABILITY,
            "triton_version": triton.__version__,
        },
        "traced_ttir": traced_ttir,
        "ttir_after_make_ttir": ttir_txt,
        "ttgir_first_hop_only": ttgir_first_hop,
        "ttgir_after_make_ttgir": ttgir_txt,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print("triton", triton.__version__)
    print("=== traced (make_tensor_ptr present?) ===",
          "make_tensor_ptr" in traced_ttir)
    print("=== ttir (make_tensor_ptr gone?) ===",
          "make_tensor_ptr" not in ttir_txt)
    print("=== ttgir has #blocked? ===", "#blocked" in ttgir_txt or "blocked" in ttgir_txt)
    print("=== ttgir has dot_op / convert_layout? ===",
          "dot_op" in ttgir_txt, "convert_layout" in ttgir_txt)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
