#!/usr/bin/env python3
"""ch17 控制流下降取证：pin v3.2.0 追踪期 TTIR（任何 pass 之前）。

配方同 ch01（INSTANCE.md『pin 精确 IR 验证配方』）：在 triton==3.2.0 的 venv 里跑，
其 Python 前端与 pin v3.2.0 逐字节相同；ASTSource(...).make_ir(...) 出追踪期 TTIR
（add_inliner 等 make_ttir pass 之前）。观测 scf.for / scf.if / cf.cond_br / scf.while
与 tt.num_stages / tt.loop_unroll_factor 属性、负步长 subi/addi、static_range 展开、
循环内 return 编译期报错。

用法：<repo>/instances/triton/v32/bin/python run_ch17_control_flow.py
输出：stdout 逐 kernel 的追踪期 TTIR；同时写 ch17_traces.json 供 lint 溯源。
"""
import json
import re
import triton
import triton.language as tl
from triton.compiler.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget

print(f"# triton {triton.__version__}  ({triton.__file__})")

target = GPUTarget("cuda", 90, 32)
backend = make_backend(target)
options = backend.parse_options({})
from triton._C.libtriton import ir


def trace_ir(fn, signature, constants):
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    src = ASTSource(fn, signature=signature, constants=constants)
    mod = src.make_ir(options, backend.get_codegen_implementation(),
                      backend.get_module_map(), ctx)
    txt = str(mod)
    txt = re.sub(r"\s*loc\(#loc\d*\)", "", txt)
    txt = re.sub(r"^#loc.*$", "", txt, flags=re.M)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt


results = {}


# ============ K1: scf.for + loop-carried + num_stages/loop_unroll_factor 属性 ============
@triton.jit
def k_for_range(out_ptr, n):
    acc = tl.zeros((16,), dtype=tl.float32)
    for k in tl.range(0, n, num_stages=3, loop_unroll_factor=2):
        acc = acc + 1.0
    tl.store(out_ptr + tl.arange(0, 16), acc)


t = trace_ir(k_for_range, {"out_ptr": "*fp32", "n": "i32"}, {})
results["K1_for_range_attrs"] = t
print("\n================ K1: tl.range(0,n,num_stages=3,loop_unroll_factor=2) ================")
print(t)


# ============ K2: 无 return 的 if → scf.if + yield ============
@triton.jit
def k_if_scf(out_ptr, c, BLOCK: tl.constexpr):
    x = tl.load(out_ptr + tl.arange(0, BLOCK))
    if c > 0:
        x = x + 1.0
    tl.store(out_ptr + tl.arange(0, BLOCK), x)


t = trace_ir(k_if_scf, {"out_ptr": "*fp32", "c": "i32"}, {"BLOCK": 8})
results["K2_if_scf"] = t
print("\n================ K2: if c>0: x=x+1  (无 return → scf.if) ================")
print(t)


# ============ K3: 带 return 的 if → cf.cond_br + no predecessors 死块 ============
@triton.jit
def k_if_return(out_ptr, c):
    if c > 0:
        tl.store(out_ptr, 1)
        return
    tl.store(out_ptr, 2)


t = trace_ir(k_if_return, {"out_ptr": "*i32", "c": "i32"}, {})
results["K3_if_return_cfg"] = t
print("\n================ K3: if c>0: store;return  (带 return → cf.cond_br) ================")
print(t)


# ============ K4: 负步长 range(10,0,-1) → scf.for 正着数 + iv=ub-iv+lb 反算 ============
@triton.jit
def k_for_negstep(out_ptr):
    acc = tl.zeros((4,), dtype=tl.int32)
    for k in range(10, 0, -1):
        acc = acc + k
    tl.store(out_ptr + tl.arange(0, 4), acc)


t = trace_ir(k_for_negstep, {"out_ptr": "*i32"}, {})
results["K4_for_negstep"] = t
print("\n================ K4: for k in range(10,0,-1)  (负步长翻转) ================")
print(t)


# ============ K5: scf.while → before(condition)/after(yield) 双区域 ============
@triton.jit
def k_while(out_ptr, n):
    i = 0
    acc = tl.zeros((4,), dtype=tl.int32)
    while i < n:
        acc = acc + i
        i = i + 1
    tl.store(out_ptr + tl.arange(0, 4), acc)


t = trace_ir(k_while, {"out_ptr": "*i32", "n": "i32"}, {})
results["K5_while"] = t
print("\n================ K5: while i<n  (scf.while 双区域) ================")
print(t)


# ============ K6: static_range 编译期整体展开（不生成 scf.for）============
@triton.jit
def k_static_range(out_ptr):
    acc = tl.zeros((4,), dtype=tl.int32)
    for k in tl.static_range(0, 3):
        acc = acc + k
    tl.store(out_ptr + tl.arange(0, 4), acc)


t = trace_ir(k_static_range, {"out_ptr": "*i32"}, {})
results["K6_static_range"] = t
print("\n================ K6: tl.static_range(0,3)  (编译期展开，无 scf.for) ================")
print(t)


# ============ K7: 循环内 return → 编译期 raise ============
@triton.jit
def k_return_in_for(out_ptr, n):
    for k in range(0, n):
        if k > 5:
            tl.store(out_ptr, k)
            return


raise_msg = None
try:
    trace_ir(k_return_in_for, {"out_ptr": "*i32", "n": "i32"}, {})
    raise_msg = "NO RAISE (unexpected)"
except Exception as e:
    # 取最内层业务报错文本
    msg = str(e)
    raise_msg = msg
results["K7_return_in_for_error"] = raise_msg
print("\n================ K7: return inside for  (编译期 raise) ================")
print(raise_msg)


# ============ K8: if/else 两侧各改不同变量 → then/else local_defs 求并（m7）============
# x 两侧都改（对称）；y 只在 else 改（非对称→then 用 livein 原值补齐 = φ 另一入边）。
@triton.jit
def k_if_else_union(out_ptr, c, BLOCK: tl.constexpr):
    x = tl.load(out_ptr + tl.arange(0, BLOCK))
    y = tl.load(out_ptr + tl.arange(0, BLOCK))
    if c > 0:
        x = x + 1.0
    else:
        x = x - 1.0
        y = y + 10.0
    tl.store(out_ptr + tl.arange(0, BLOCK), x)
    tl.store(out_ptr + tl.arange(0, BLOCK), y)


t = trace_ir(k_if_else_union, {"out_ptr": "*fp32", "c": "i32"}, {"BLOCK": 8})
results["K8_if_else_union"] = t
print("\n================ K8: if/else 两侧各改不同变量（then/else 求并） ================")
print(t)


# ---- op 计数辅助（供表格溯源）----
def count(txt, op):
    return len(re.findall(r"(?<![\w.])" + re.escape(op) + r"(?![\w])", txt))


summary = {}
for name, txt in results.items():
    if name.endswith("_error"):
        continue
    summary[name] = {
        op: count(txt, op)
        for op in ["scf.for", "scf.if", "scf.while", "scf.yield", "scf.condition",
                   "cf.cond_br", "cf.br", "ub.poison", "arith.subi", "arith.addi",
                   "tt.num_stages", "tt.loop_unroll_factor", "no predecessors"]
    }
print("\n================ OP COUNTS ================")
print(json.dumps(summary, indent=2))

# ============ 数值仿真：负步长诱导变量重构（执行 IR 里 iv=ub-j+lb 的算术）============
# range(10,0,-1): 前端 negative_step 分支交换后 lb'=0, ub'=10, step'=1（源 L935-L937）。
# scf.for 计数器 j=0..9；体首 iv=ub-j+lb（源 L1019-L1021 的 subi/addi）= 10-j。
NEG_lb_orig, NEG_ub_orig, NEG_step_orig = 10, 0, -1
lb2, ub2, step2 = NEG_ub_orig, NEG_lb_orig, -NEG_step_orig   # 交换 + 取正
neg_rows = []
for j in range((ub2 - lb2) // step2):     # scf.for 正着数的计数器
    iv = ub2 - j + lb2                     # iv = ub - j + lb（IR: subi 然后 addi）
    neg_rows.append({"scf_counter_j": j, "iv_reconstructed": iv})
neg_user_seq = [r["iv_reconstructed"] for r in neg_rows]
neg_python_seq = list(range(NEG_lb_orig, NEG_ub_orig, NEG_step_orig))
print("\n================ NEG-STEP RECONSTRUCTION (executed iv=ub-j+lb) ================")
print(f"lb'={lb2} ub'={ub2} step'={step2}")
for r in neg_rows:
    print(f"  j={r['scf_counter_j']}  ->  iv={r['iv_reconstructed']}")
print(f"  reconstructed seq = {neg_user_seq}")
print(f"  python range(10,0,-1) = {neg_python_seq}")
print(f"  MATCH = {neg_user_seq == neg_python_seq}")

# ============ 数值仿真：static_range 编译期展开的每次迭代累加（每 lane）============
# tl.static_range(0,3): 前端把循环拆成 3 次 visit（源 L904-L909），k 绑成 constexpr(0/1/2）。
# 体 acc = acc + k，acc 初值 0（每 lane 同）。
sr_acc = 0
sr_rows = []
for k in range(0, 3):
    sr_acc = sr_acc + k
    sr_rows.append({"unroll_index": k, "k_constexpr": k, "acc_after": sr_acc})
print("\n================ STATIC_RANGE UNROLL ACC (per lane) ================")
for r in sr_rows:
    print(f"  unroll#{r['unroll_index']}  k={r['k_constexpr']}  acc_after={r['acc_after']}")

out = {"triton_version": triton.__version__, "stage": "追踪期 / make_ir 之前",
       "ir": results, "op_counts": summary,
       "neg_step_reconstruction": {
           "orig": {"lb": NEG_lb_orig, "ub": NEG_ub_orig, "step": NEG_step_orig},
           "swapped": {"lb": lb2, "ub": ub2, "step": step2},
           "rows": neg_rows, "reconstructed_seq": neg_user_seq,
           "python_seq": neg_python_seq, "match": neg_user_seq == neg_python_seq},
       "static_range_unroll": {"rows": sr_rows}}
with open(__file__.rsplit("/", 1)[0] + "/ch17_traces.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n# wrote ch17_traces.json")
