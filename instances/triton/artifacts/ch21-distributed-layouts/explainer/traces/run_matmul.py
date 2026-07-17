#!/usr/bin/env python3
"""ch21 pin-compile trace: 64x64 fp16 matmul → TTGIR, observe #mma / #dot_op / #slice.

Headless, no GPU: triton==3.2.0 (pin-exact frontend, see INSTANCE.md pin recipe).
Compiles a single-tile fp16 matmul (M=N=K=64, one program, num_warps=4) as far as
the `ttgir` stage and dumps the TritonGPU-dialect IR so we can read the real
NvidiaMmaEncodingAttr, DotOperandEncodingAttr(kWidth) and SliceEncodingAttr
attributes the compiler assigns for a Tensor-Core dot on sm80 (Ampere).
Stage is make_ttgir (after make_ttir), recorded explicitly.
"""
import json
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(a, b)
    c = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c, acc)


def main():
    target = GPUTarget("cuda", 80, 32)  # sm80 (Ampere)
    backend = CUDABackend(target)
    opts = backend.parse_options({"num_warps": 4})

    src = ASTSource(
        fn=matmul_kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
                   "M": "i32", "N": "i32", "K": "i32",
                   "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
        constants={"BM": 64, "BN": 64, "BK": 64},
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

    print("=== num_warps ===", opts.num_warps)
    print("=== TTGIR ===")
    print(ttgir_txt)
    print("=== ENCODING-DECLS ===")
    for line in ttgir_txt.splitlines():
        s = line.strip()
        if s.startswith("#") and ("mma" in s or "dot_op" in s or "slice" in s or "blocked" in s) and "=" in s:
            print(s)

    import math

    # ---- (A) auto-derive builder simulation (TritonGPUAttrDefs.td:L692-L728) ----
    # Reproduce the clamp loop for the operand-load #blocked the compiler emitted:
    # shape=[64,64], sizePerThread=[1,1], order=[1,0], numWarps=4, numThreadsPerWarp=32.
    def derive_blocked(shape, sizePerThread, order, numWarps, numThreadsPerWarp):
        rank = len(sizePerThread)
        threadsPerWarp = [0] * rank
        warpsPerCTA = [0] * rank
        shapePerCTA = list(shape)  # CTASplitNum=[1,1] -> shapePerCTA == shape
        remainingLanes = numThreadsPerWarp
        remainingThreads = numWarps * numThreadsPerWarp
        remainingWarps = numWarps
        prevLanes = 1
        prevWarps = 1
        steps = []
        for d in range(rank - 1):
            i = order[d]
            threadsPerCTA = max(1, min(remainingThreads,
                                       max(1, shapePerCTA[i] // sizePerThread[i])))
            threadsPerWarp[i] = max(1, min(threadsPerCTA, remainingLanes))
            warpsPerCTA[i] = max(1, min(threadsPerCTA // threadsPerWarp[i], remainingWarps))
            remainingWarps //= warpsPerCTA[i]
            remainingLanes //= threadsPerWarp[i]
            remainingThreads //= threadsPerCTA
            prevLanes *= threadsPerWarp[i]
            prevWarps *= warpsPerCTA[i]
            steps.append({"d": d, "axis_i": i, "threadsPerCTA": threadsPerCTA,
                          "threadsPerWarp_i": threadsPerWarp[i],
                          "warpsPerCTA_i": warpsPerCTA[i]})
        last = order[rank - 1]
        threadsPerWarp[last] = numThreadsPerWarp // prevLanes
        warpsPerCTA[last] = numWarps // prevWarps
        steps.append({"d": "last", "axis_i": last,
                      "threadsPerWarp_i": threadsPerWarp[last],
                      "warpsPerCTA_i": warpsPerCTA[last]})
        return threadsPerWarp, warpsPerCTA, steps

    tpw2d, wpc2d, builder_steps = derive_blocked([64, 64], [1, 1], [1, 0], 4, 32)

    # ---- (B) coalescing analysis for the operand-load #blocked ----
    # order=[1,0] -> fastest axis is dim1 (columns). Row-major fp16 tile (2 bytes).
    bytes_per_elem = 2
    # coalesced (as compiled): 32 lanes span 32 contiguous columns of one row
    coal_lanes = 32
    coal_span_elems = coal_lanes          # stride 1 along columns
    coal_span_bytes = coal_span_elems * bytes_per_elem     # 32*2 = 64 bytes
    coal_txn = 1
    # counterfactual order=[0,1]: 32 lanes span 32 rows (stride = N columns)
    row_stride_elems = 64                 # N = 64 columns per row
    strided_stride_bytes = row_stride_elems * bytes_per_elem   # 64*2 = 128 bytes apart
    strided_txn = 32                      # 32 disjoint addresses -> up to 32 transactions

    # ---- (C) dot-operand sizePerThread (Dialect.cpp:L2145-L2159) + kWidth ----
    bitwidth_fp16 = 16
    kWidth = 32 // bitwidth_fp16          # 32/16 = 2  (matches dot_op kWidth=2 in IR)
    # opIdx=0 (a, M x K): [M=2, K=2*kWidth];  opIdx=1 (b, K x N): [K=2*kWidth, N=1]
    sizePerThread_op0 = [2, 2 * kWidth]   # [2, 4]
    sizePerThread_op1 = [2 * kWidth, 1]   # [4, 1]
    a_contig_along_k = 2 * kWidth         # 4 fp16 contiguous along K per thread

    analysis = {
        "operand_blocked_encoding": {
            "sizePerThread": [1, 1], "threadsPerWarp": [1, 32],
            "warpsPerCTA": [2, 2], "order": [1, 0]},
        "mma_encoding": {"versionMajor": 2, "versionMinor": 0,
                         "warpsPerCTA": [2, 2], "instrShape": [16, 8]},
        "dot_op_kWidth": kWidth,
        "builder_derived_threadsPerWarp": tpw2d,   # [1, 32]
        "builder_derived_warpsPerCTA": wpc2d,      # [2, 2]
        "builder_steps": builder_steps,
        "coalescing": {
            "bytes_per_elem_fp16": bytes_per_elem,
            "coalesced_lanes": coal_lanes,
            "coalesced_span_bytes": coal_span_bytes,
            "coalesced_transactions": coal_txn,
            "strided_row_stride_bytes": strided_stride_bytes,
            "strided_transactions": strided_txn,
        },
        "dot_operand": {
            "bitwidth_fp16": bitwidth_fp16,
            "kWidth": kWidth,
            "sizePerThread_opIdx0_MK": sizePerThread_op0,   # [2, 4]
            "sizePerThread_opIdx1_KN": sizePerThread_op1,   # [4, 1]
            "a_contig_elems_along_K": a_contig_along_k,     # 4
            "instr": "mma.16816",
        },
        "slice_from_arange": {
            "arange_M_axis1_encoding": "slice<dim=1,parent=#blocked>",
            "expand_dims_axis1_restores": "tensor<64x1xi32,#blocked>",
            "arange_N_axis0_encoding": "slice<dim=0,parent=#blocked>",
            "expand_dims_axis0_restores": "tensor<1x64xi32,#blocked>",
        },
    }

    print("=== ANALYSIS ===")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))

    out = {
        "kernel": "matmul_kernel (single-tile fp16 matmul, tl.dot)",
        "target": "cuda sm80 (Ampere), warpSize=32",
        "num_warps": opts.num_warps,
        "M": 64, "N": 64, "K": 64,
        "dtype": "fp16",
        "stage": "make_ttgir (after make_ttir)",
        "triton_version": triton.__version__,
        "analysis": analysis,
        "ttgir": ttgir_txt,
    }
    with open(__file__.replace("run_matmul.py", "matmul.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
