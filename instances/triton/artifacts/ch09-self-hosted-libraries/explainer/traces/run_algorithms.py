#!/usr/bin/env python3
"""ch09 教学素材 trace 生成器 —— 纯 Python 忠实复刻 v3.2.0 源码算法。
数值权威 = triton v3.2.0 源码常量与控制流；每个机制产一份 JSON raw trace。
host 无需 CUDA(纯控制流)。IR 膨胀数由 tv32(triton==3.2.0) 精确编译另出。"""
import json, numpy as np

MASK32 = 0xFFFFFFFF
def u32(x): return int(x) & MASK32
def umulhi32(a, b): return (u32(a) * u32(b)) >> 32   # math.umulhi: 高 32 位

out = {}

# ---------- m1: cdiv (standard.py:L40  (x+div-1)//div) ----------
def cdiv(x, d): return (x + d - 1) // d
out["cdiv"] = {"formula": "(x + div - 1) // div  [standard.py:L40]",
    "cases": [{"x": x, "div": d, "floor": x // d, "result": cdiv(x, d)}
              for x, d in [(10, 3), (9, 3), (1, 4)]]}

# ---------- m3: softmax 减 max 数值稳定 (standard.py:L54) ----------
def softmax_naive(x):
    e = np.exp(np.array(x, dtype=np.float64)); return e / e.sum()
def softmax_stable(x):
    x = np.array(x, dtype=np.float64); z = x - x.max()
    e = np.exp(z); return e / e.sum()
xs = [1000.0, 1001.0, 1002.0]
en = np.exp(np.array(xs))                 # overflow to inf
out["softmax"] = {"x": xs,
    "naive_exp": [None if np.isinf(v) else v for v in en],
    "naive_exp_is_inf": [bool(np.isinf(v)) for v in en],
    "naive_result": [None if np.isnan(v) else v for v in softmax_naive(xs)],
    "max": max(xs),
    "z_shifted": list(np.array(xs) - max(xs)),          # [-2,-1,0]
    "stable_exp": list(np.exp(np.array(xs) - max(xs))),  # [e^-2,e^-1,1]
    "stable_exp_rounded": [round(float(v), 3) for v in np.exp(np.array(xs) - max(xs))],
    "stable_result": list(softmax_stable(xs)),
    "stable_result_rounded": [round(float(v), 3) for v in softmax_stable(xs)],
    "note": "naive: exp(1000)=inf -> inf/inf = nan; stable: max(z)=0 -> exp<=1"}

# ---------- m5: compare-and-swap XOR 条件交换 (standard.py:L334-339) ----------
def cas_pair(a, b, flip=0):
    # ints -> bitcast identity; ix ^ where((left>right)!=flip, ileft^iright, 0)
    left, right = a, b
    cond = (left > right) != bool(flip)
    x = a ^ (a ^ b if cond else 0)   # element that held 'a'
    y = b ^ (a ^ b if cond else 0)   # element that held 'b'
    return {"left": left, "right": right, "flip": flip, "cond_swap": cond,
            "xor_delta": a ^ b, "out_left": x, "out_right": y}
out["compare_and_swap"] = {"ascending_pairs": [
    cas_pair(3, 1, 0),   # 3>1 -> swap -> (1,3)
    cas_pair(1, 3, 0),   # 1>3 false -> keep -> (1,3)
    cas_pair(2, 0, 0)]}  # swap -> (0,2)

# ---------- m4 + m2: bitonic sort 忠实复刻 + CAS 计数 ----------
CALL_LOG = []   # 每个 _compare_and_swap 调用 = 一个向量化 op = IR 里一个 arith.select
def _log2(i):
    l = 0; n = i
    while n > 1: n >>= 1; l += 1
    return l
def _compare_and_swap(x, flip, i, n_dims):
    # 复刻 reshape 取偶/取奇 + 条件交换; 一次调用 = 整块一个向量化 CAS op
    n = len(x); stride = 2 ** (n_dims - i - 1)
    before = list(x); x = list(x)
    for base in range(0, n, 2 * stride):
        for off in range(stride):
            li = base + off; ri = base + stride + off
            l, r = x[li], x[ri]
            f = flip[li] if isinstance(flip, list) else flip
            if (l > r) != bool(f): x[li], x[ri] = r, l
    CALL_LOG.append({"call": len(CALL_LOG) + 1, "i": i,
                     "before": before, "after": list(x)})
    return x
def _bitonic_merge(x, stage, order, n_dims):
    n = len(x)
    if order == 2:
        block = 2 ** stage
        flip = [(idx // block) % 2 for idx in range(n)]
    else:
        flip = order
    for i in range(stage):
        x = _compare_and_swap(x, flip, i + (n_dims - stage), n_dims)
    return x
def bsort(x, descending=0):
    n_dims = _log2(len(x))
    x = list(x)
    for i in range(1, n_dims + 1):
        x = _bitonic_merge(x, i, 2 if i < n_dims else descending, n_dims)
    return x
CALL_LOG.clear()
arr = [3, 1, 2, 0]
res4 = bsort(arr)
out["bitonic_sort"] = {"input": arr, "sorted": res4, "n_dims": _log2(len(arr)),
    "num_cas_calls": len(CALL_LOG),
    "note": "每个 CAS 调用 = 整块一个向量化 op = 内联 IR 里一个 arith.select",
    "call_log": CALL_LOG}
# stage/CAS counts for representative block sizes (matches inlined IR select count)
counts = []
for n in [4, 8, 16, 64, 1024]:
    nd = _log2(n); cas = nd * (nd + 1) // 2
    counts.append({"n": n, "n_dims": nd, "num_compare_and_swap": cas})
out["bitonic_counts"] = counts

# ---------- m7 + m8: Philox 无状态 RNG + 环绕算术 ----------
PHILOX_KEY_A = 0x9E3779B9; PHILOX_KEY_B = 0xBB67AE85
PHILOX_ROUND_A = 0xD2511F53; PHILOX_ROUND_B = 0xCD9E8D57
def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds=10, log=None):
    c0, c1, c2, c3, k0, k1 = map(u32, (c0, c1, c2, c3, k0, k1))
    for rnd in range(n_rounds):
        A, B = PHILOX_ROUND_A, PHILOX_ROUND_B
        _c0, _c2 = c0, c2
        c0 = u32(umulhi32(B, _c2) ^ c1 ^ k0)
        c2 = u32(umulhi32(A, _c0) ^ c3 ^ k1)
        c1 = u32(B * _c2)          # sanitize_overflow=False -> mod 2^32
        c3 = u32(A * _c0)
        k0 = u32(k0 + PHILOX_KEY_A)
        k1 = u32(k1 + PHILOX_KEY_B)
        if log is not None:
            log.append({"round": rnd + 1, "c0": c0, "c1": c1, "c2": c2, "c3": c3,
                        "k0": k0, "k1": k1})
    return c0, c1, c2, c3
def randint4x(seed, offset, n_rounds=10, log=None):
    seed = seed & 0xFFFFFFFFFFFFFFFF
    seed_lo = u32(seed & MASK32); seed_hi = u32((seed >> 32) & MASK32)
    return philox_impl(offset, 0, 0, 0, seed_lo, seed_hi, n_rounds, log)
SEED = 0x2A  # 42
rlog0 = []
r_off0 = randint4x(SEED, 0, log=rlog0)
r_off0_again = randint4x(SEED, 0)                 # statelessness: identical
r_off1 = randint4x(SEED, 1)                       # different counter -> different
r_off7 = randint4x(SEED, 7)
out["philox"] = {"seed": SEED, "n_rounds": 10,
    "rounds_offset0": rlog0,
    "randint4x_offset0": list(r_off0),
    "randint4x_offset0_recompute": list(r_off0_again),
    "stateless_identical": list(r_off0) == list(r_off0_again),
    "randint4x_offset1": list(r_off1),
    "randint4x_offset7": list(r_off7),
    "offset0_vs_offset1_differ": list(r_off0) != list(r_off1)}

# m8: 环绕算术单步放大镜 (c1 = B*_c2 mod 2^32)
B = PHILOX_ROUND_B  # 0xCD9E8D57 = 3449720151
def wrap_row(m):
    full = B * m
    return {"B_ROUND_B": B, "operand__c2": m, "full_product": full,
            "two_pow_32": 2**32, "overflows": full >= 2**32, "wrapped_mod_2_32": u32(full)}
out["wraparound"] = {"rows": [wrap_row(1), wrap_row(3)],
    "note": "_c2=1: 积<2^32 不溢出, 结果不变; _c2=3: 积>2^32, sanitize_overflow=False 让它 mod 2^32 环绕; ch07 默认 True 会拦截"}

# ---------- m10: extern dispatch 按 dtype 选符号 (libdevice.py mulhi) ----------
arg_type_symbol_dict = {
    ("int32", "int32"): ("__nv_mulhi", "int32"),
    ("uint32", "uint32"): ("__nv_umulhi", "uint32"),
    ("int64", "int64"): ("__nv_mul64hi", "int64"),
    ("uint64", "uint64"): ("__nv_umul64hi", "uint64")}
def dispatch(arg_types):
    if arg_types not in arg_type_symbol_dict:
        return {"arg_types": list(arg_types), "error": "input arg type does not match"}
    sym, ret = arg_type_symbol_dict[arg_types]
    return {"arg_types": list(arg_types), "symbol": sym, "ret_dtype": ret}
out["extern_dispatch"] = {"table": [
    dispatch(("int32", "int32")), dispatch(("uint32", "uint32")),
    dispatch(("uint64", "uint64")), dispatch(("float16", "float16"))]}

json.dump(out, open("algorithms.json", "w"), indent=1, default=str)
print(json.dumps({k: (v if not isinstance(v, dict) else "...") for k, v in out.items()}, indent=1))
print("WROTE algorithms.json")
# spot summary
print("cdiv 10/3 =", out["cdiv"]["cases"][0]["result"])
print("sort", arr, "->", res4, "CAS calls=", out["bitonic_sort"]["num_cas_calls"])
for c in CALL_LOG: print("  call", c["call"], "i=", c["i"], c["before"], "->", c["after"])
print("philox stateless identical:", out["philox"]["stateless_identical"],
      "off0!=off1:", out["philox"]["offset0_vs_offset1_differ"])
print("randint4x(42,0)=", out["philox"]["randint4x_offset0"])
print("wrap:", B, "*", _c2, "=", full, "-> mod2^32 =", wrapped)
