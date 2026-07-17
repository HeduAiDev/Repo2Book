#!/usr/bin/env python3
"""ch19 识字课佐证：出一段真 TTIR dump，对照 .td 三元组认字。

注意：本章 pin = triton v3.2.0（.td 定义即源码真相）。宿主机装的是 triton 3.6.0，
但本例覆盖的 tt.* 算子（make_range/splat/addptr/load/store）其 arguments/results/
assemblyFormat 在 3.2.0→3.6.0 之间逐字未变，dump 长相完全一致，可作『.td 三元组决定
dump 长相』的佐证。表格权威数字仍以 3.2.0 的 TritonOps.td file:Lxxx 为准，故 explainer
的 trace_source 标 manual（见 manual_reason）。

无 GPU：TTIR 是编译第一阶段（前端 AST→MLIR），不触碰 target codegen，headless 可出。
输出存 raw_ttir.txt（带 loc）与 ttir_clean.txt（剥掉 loc 便于阅读）。
"""
import re
from pathlib import Path

import triton
import triton.language as tl
from triton.compiler import ASTSource, compile

OUT = Path(__file__).parent


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def main():
    src = ASTSource(
        fn=add_kernel,
        signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
                   "n": "i32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 4},
    )
    k = compile(src)
    ttir = k.asm["ttir"]
    (OUT / "raw_ttir.txt").write_text(f"# triton {triton.__version__}\n{ttir}\n",
                                      encoding="utf-8")
    # strip loc(...) trailers + the #locN definitions for a clean teaching view
    lines = []
    for ln in ttir.splitlines():
        if re.match(r"^#loc", ln.strip()):
            continue
        # loc trailers may nest one level: loc("x_ptr"(#loc)) — strip iteratively
        prev = None
        while prev != ln:
            prev = ln
            ln = re.sub(r"\s*loc\((?:[^()]|\([^()]*\))*\)", "", ln)
        lines.append(ln.rstrip())
    clean = "\n".join(l for l in lines if l.strip())
    (OUT / "ttir_clean.txt").write_text(f"# triton {triton.__version__}\n{clean}\n",
                                        encoding="utf-8")
    print(f"triton {triton.__version__}")
    print(clean)


if __name__ == "__main__":
    main()
