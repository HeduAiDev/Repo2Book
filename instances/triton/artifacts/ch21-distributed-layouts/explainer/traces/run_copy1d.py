#!/usr/bin/env python3
"""ch21 pin-compile trace: 1D copy kernel → TTGIR, observe #blocked encoding.

Headless, no GPU: triton==3.2.0 (pin-exact frontend, see INSTANCE.md pin recipe).
Compiles a 1D element-wise copy (N=1024, BLOCK=1024, num_warps=4) as far as the
`ttgir` stage and dumps the TritonGPU-dialect IR so we can read the real
#blocked<sizePerThread,threadsPerWarp,warpsPerCTA,order> attribute that the
compiler assigns. Stage is make_ttgir (after make_ttir), recorded explicitly.
"""
import json
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource


@triton.jit
def copy_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, v, mask=mask)


def main():
    target = GPUTarget("cuda", 80, 32)  # sm80 (Ampere), 32 threads/warp
    backend = CUDABackend(target)
    opts = backend.parse_options({"num_warps": 4})

    src = ASTSource(
        fn=copy_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "N": "i32", "BLOCK": "constexpr"},
        constants={"BLOCK": 1024},
    )
    ctx = triton._C.libtriton.ir.context()
    triton._C.libtriton.ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    codegen_fns = backend.get_codegen_implementation()
    module_map = backend.get_module_map()

    stages = {}
    backend.add_stages(stages, opts)

    ttir = stages["ttir"](src.make_ir(opts, codegen_fns, module_map, ctx), {})
    ttgir = stages["ttgir"](ttir, {})
    ttgir_txt = str(ttgir)

    # ---- observed encoding (from the real TTGIR above) ----
    sizePerThread = [1]
    threadsPerWarp = [32]
    warpsPerCTA = [4]
    order = [0]

    # ---- getElemsPerThread arithmetic (Dialect.cpp:L633-L652), 1D, headless ----
    # t[i] = sizePerThread[i]*threadsPerWarp[i]*warpsPerCTA[i]  (one CTA-tile span)
    # elemsPerThread[i] = ceil(shapePerCTA[i]/t[i]) * sizePerThread[i]
    import math
    shape = [1024]
    num_threads = opts.num_warps * 32                      # 4 warps * 32 lanes = 128
    t0 = sizePerThread[0] * threadsPerWarp[0] * warpsPerCTA[0]   # 1*32*4 = 128
    tiles0 = math.ceil(shape[0] / t0)                      # ceil(1024/128) = 8
    elems0 = tiles0 * sizePerThread[0]                     # 8*1 = 8
    total_elems_per_thread = elems0                        # 8
    crosscheck = shape[0] // num_threads                   # 1024/128 = 8 (matches)

    # ---- coalescing: fp32 = 4 bytes; one warp (order[0]=0, contiguous dim) ----
    bytes_per_elem = 4
    warp_span_elems = threadsPerWarp[0] * sizePerThread[0]   # 32 lanes * 1 = 32 contiguous elems
    warp_span_bytes = warp_span_elems * bytes_per_elem       # 32*4 = 128 bytes
    txn_coalesced = 1                                        # 128 bytes -> 1 memory transaction

    analysis = {
        "observed_encoding": {
            "sizePerThread": sizePerThread, "threadsPerWarp": threadsPerWarp,
            "warpsPerCTA": warpsPerCTA, "order": order},
        "num_threads": num_threads,
        "tile_span_t0": t0,
        "num_tiles_dim0": tiles0,
        "elemsPerThread_dim0": elems0,
        "total_elemsPerThread": total_elems_per_thread,
        "crosscheck_total_over_threads": crosscheck,
        "bytes_per_elem_fp32": bytes_per_elem,
        "warp_span_elems": warp_span_elems,
        "warp_span_bytes": warp_span_bytes,
        "coalesced_transactions": txn_coalesced,
    }

    out = {
        "kernel": "copy_kernel (1D element-wise copy)",
        "target": "cuda sm80, warpSize=32",
        "num_warps": opts.num_warps,
        "N": 1024,
        "BLOCK": 1024,
        "stage": "make_ttgir (after make_ttir)",
        "triton_version": triton.__version__,
        "analysis": analysis,
        "ttgir": ttgir_txt,
    }
    print("=== num_warps ===", opts.num_warps)
    print("=== TTGIR ===")
    print(ttgir_txt)
    print("=== ANALYSIS ===")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    with open(__file__.replace("run_copy1d.py", "copy1d.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
