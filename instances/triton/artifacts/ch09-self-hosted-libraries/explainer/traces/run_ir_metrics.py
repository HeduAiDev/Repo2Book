#!/usr/bin/env python3
"""ch09 IR 膨胀取证 —— triton==3.2.0 headless 精确编译, 数 TTIR 里内联展开的 op。
pin v3.2.0 (instances/triton/source == v3.2.0)。CUDA 环境可用, 只取 TTIR(前端AST->IR)。"""
import triton, triton.language as tl, re, json
from triton.backends.compiler import GPUTarget, AttrsDescriptor
def cc(fn, sig, cst, stage="ttir"):
    src = triton.compiler.ASTSource(fn=fn, signature=sig, constants=cst, attrs=AttrsDescriptor())
    return triton.compile(src, target=GPUTarget("cuda", 80, 32)).asm[stage]
def C(pat, s): return len(re.findall(pat, s))

@triton.jit
def ksort(X, Y, N: tl.constexpr):
    off = tl.arange(0, N); tl.store(Y + off, tl.sort(tl.load(X + off)))
@triton.jit
def ksm(X, Y, N: tl.constexpr):
    off = tl.arange(0, N); tl.store(Y + off, tl.softmax(tl.load(X + off)))
@triton.jit
def kplain(X, Y, N: tl.constexpr):
    off = tl.arange(0, N); tl.store(Y + off, tl.load(X + off) + 1.0)
@triton.jit
def khint(X, Y, N: tl.constexpr):
    off = tl.arange(0, N); off = tl.multiple_of(off, 128)
    tl.store(Y + off, tl.load(X + off) + 1.0)

out = {"triton_version": triton.__version__, "pin": "v3.2.0", "stage": "TTIR (make_ttir 后)"}

sort_rows = []
for N in (16, 64, 1024):
    t = cc(ksort, {"X": "*fp32", "Y": "*fp32", "N": "constexpr"}, {"N": N})
    open(f"ttir_sort_{N}.mlir", "w").write(t)
    sort_rows.append({"block_n": N, "log2_n_stages": N.bit_length() - 1,
        "ttir_lines": t.count("\n"),
        "arith_select": C(r"arith\.select", t),      # = compare-and-swap 数
        "arith_cmpf": C(r"arith\.cmpf", t),
        "arith_xori": C(r"arith\.xori", t),
        "tt_reshape": C(r"tt\.reshape", t),
        "tt_call": C(r"tt\.call", t)})               # 0 => 完全内联
out["sort_inlining"] = sort_rows

t = cc(ksm, {"X": "*fp32", "Y": "*fp32", "N": "constexpr"}, {"N": 128})
open("ttir_softmax_128.mlir", "w").write(t)
out["softmax_inlining"] = {"block_n": 128, "ttir_lines": t.count("\n"),
    "tt_call": C(r"tt\.call", t), "tt_reduce": C(r'"tt\.reduce"', t),
    "arith_maxnumf": C(r"arith\.maxnumf", t), "arith_subf": C(r"arith\.subf", t),
    "math_exp": C(r"math\.exp", t), "arith_divf": C(r"arith\.divf", t),
    "separate_softmax_func": ("@softmax" in t)}

tp = cc(kplain, {"X": "*fp32", "Y": "*fp32", "N": "constexpr"}, {"N": 128})
th = cc(khint, {"X": "*fp32", "Y": "*fp32", "N": "constexpr"}, {"N": 128})
open("ttir_plain_128.mlir", "w").write(tp); open("ttir_hint_128.mlir", "w").write(th)
def divis(s):
    m = re.search(r"tt\.divisibility = dense<(\d+)>", s); return int(m.group(1)) if m else None
out["multiple_of_hint"] = {"block_n": 128,
    "plain_has_divisibility_attr": "tt.divisibility" in tp,
    "hint_has_divisibility_attr": "tt.divisibility" in th,
    "hint_divisibility_value": divis(th),
    "note": "tl.multiple_of(off,128) -> make_range 上打 tt.divisibility=dense<128> 标记"}

json.dump(out, open("ir_metrics.json", "w"), indent=1)
print(json.dumps(out, indent=1))
