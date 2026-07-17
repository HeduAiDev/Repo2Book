#!/usr/bin/env python3
"""ch30 pin 取证驱动：headless 编译一段 num_stages=3 的 fp16 matmul kernel，
dump make_ttir 之后 / make_ttgir 之后的 IR，观测软件流水线 pass 的落地：
  - load -> AsyncCopyGlobalToLocalOp 替换
  - 多 buffer memdesc 环形缓冲分配（[numBuffers, tile...]）
  - prologue / 稳态 / (谓词化) epilogue 三段结构
  - iter_args 模变量扩展

pin: triton==3.2.0（与 instances/triton/source 逐字节同，见 INSTANCE.md）。
headless、无 GPU：只编译到 ttgir（不发射 cubin / 不执行 kernel）。
用法：v32/bin/python run_pipeline_dump.py  （从 instances/triton 目录跑）
"""
import json
import sys
import triton
import triton.language as tl
from triton.compiler.compiler import make_backend, ASTSource
from triton.backends.compiler import GPUTarget


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)


def main(num_stages, cap, out_prefix):
    from triton._C.libtriton import ir
    from triton.backends.compiler import AttrsDescriptor

    sig = {"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
           "M": "i32", "N": "i32", "K": "i32",
           "stride_am": "i32", "stride_ak": "i32", "stride_bk": "i32",
           "stride_bn": "i32", "stride_cm": "i32", "stride_cn": "i32",
           "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr"}
    constants = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}
    # Specialization hints identical to what a real launch infers for a
    # row-major A[M,K] x B[K,N] matmul: base pointers 16B-aligned; the
    # inner-contiguous strides (stride_ak, stride_bn) == 1; the major
    # strides 16B-divisible. Without these the loads have contiguity 1
    # (16-bit width < cp.async min 32) and the pipeliner skips them.
    #   idx: 0 a_ptr 1 b_ptr 2 c_ptr | 6 stride_am 7 stride_ak 8 stride_bk
    #        9 stride_bn 10 stride_cm 11 stride_cn
    attrs = AttrsDescriptor.from_hints(
        {0: 16, 1: 16, 2: 16, 6: 16, 7: 1, 8: 16, 9: 1, 10: 16})
    src = ASTSource(fn=matmul_kernel, signature=sig, constants=constants, attrs=attrs)
    tgt = GPUTarget("cuda", cap, 32)
    be = make_backend(tgt)
    opts = be.parse_options({"num_warps": 8, "num_stages": num_stages})

    context = ir.context()
    ir.load_dialects(context)
    be.load_dialects(context)
    codegen_fns = be.get_codegen_implementation()
    module_map = be.get_module_map()
    metadata = {}

    module = src.make_ir(opts, codegen_fns, module_map, context)

    stages = {}
    be.add_stages(stages, opts)
    # mirror compile(): first_stage is src.ext ("ttir"); run ttir then ttgir
    ttir = stages["ttir"](module, metadata)
    ttir_str = str(ttir)
    with open(f"{out_prefix}.ttir.mlir", "w") as f:
        f.write(ttir_str)

    # ttgir (this runs the TritonGPU pipeline incl. software pipeliner)
    ttgir = stages["ttgir"](ttir, metadata)
    ttgir_str = str(ttgir)
    with open(f"{out_prefix}.ttgir.mlir", "w") as f:
        f.write(ttgir_str)

    # counts for a compact machine-readable summary
    summary = {
        "num_stages": num_stages,
        "sm": cap,
        "ttir_lines": ttir_str.count("\n") + 1,
        "ttgir_lines": ttgir_str.count("\n") + 1,
        "ttir_tt_load": ttir_str.count("tt.load"),
        "ttgir_tt_load": ttgir_str.count("tt.load"),
        "ttgir_async_copy": ttgir_str.count("async_copy_global_to_local"),
        "ttgir_async_commit": ttgir_str.count("async_commit_group"),
        "ttgir_async_wait": ttgir_str.count("async_wait"),
        "ttgir_local_alloc": ttgir_str.count("local_alloc"),
        "ttgir_memdesc_subview": ttgir_str.count("memdesc_subview"),
        "ttgir_local_load": ttgir_str.count("local_load"),
        "ttgir_scf_for": ttgir_str.count("scf.for"),
        "ttgir_scf_if": ttgir_str.count("scf.if"),
    }
    return summary, ttgir_str


def _accepts_context(fn):
    try:
        return "context" in fn.__code__.co_varnames
    except Exception:
        return False


if __name__ == "__main__":
    results = {}
    for ns, cap in [(3, 90), (3, 80), (2, 90), (4, 90)]:
        tag = f"sm{cap}_ns{ns}"
        try:
            summ, _ = main(ns, cap, f"matmul_{tag}")
            results[tag] = summ
            print(f"OK {tag}: {json.dumps(summ)}")
        except Exception as e:
            import traceback
            results[tag] = {"error": repr(e)}
            print(f"FAIL {tag}: {e!r}")
            traceback.print_exc()
    with open("pipeline_dump_summary.json", "w") as f:
        json.dump(results, f, indent=2)
