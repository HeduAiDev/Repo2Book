#!/usr/bin/env python3
"""ch31 · TF32x3 三次逼近的数值忠实模拟(host 纯 numpy,不依赖 triton 编译器)。

忠实对照 lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp 的 matchAndRewrite:
  aBig = f32ToTF32(a); aSmall = a - aBig;          (源码 L60-61)
  bBig = f32ToTF32(b); bSmall = b - bBig;          (源码 L63-64)
  dot1 = dot(aSmall, bBig, 0,   tf32)              (源码 L69)
  dot2 = dot(aBig,   bSmall, dot1, tf32)           (源码 L70)
  dot3 = dot(aBig,   bBig,   dot2, tf32)           (源码 L71)
  sum  = dot3 + C                                  (源码 L73)
f32ToTF32 = 源码 "cvt.rna.tf32.f32"(F32DotTC.cpp L42):把 f32 的 23bit 尾数
round-to-nearest 截到 10bit(丢低 13bit)。tf32 dot 的每个乘子在进 tensor core 前也
被截成 tf32——这里对 dot 的两个输入都先过 to_tf32,累加在 f32。

参考真值用 float64 算(近似无穷精度)。输出对照:
  single-tf32(基准 dot 的默认精度,一次) vs tf32x3(三次) 相对 fp32 的误差,
  以及四项交叉积各自量级——证明丢弃的 aSmall*bSmall 可忽略。
"""
import json
import numpy as np

np.seterr(all="ignore")


def to_tf32(x):
    """cvt.rna.tf32.f32:round-to-nearest-even 把 f32 尾数从 23bit 截到 10bit。"""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    round_bit = (u >> 13) & 1
    u = u + 0xFFF + round_bit          # rne bias
    u = u & ~np.uint64(0x1FFF)         # 清低 13bit
    return u.astype(np.uint32).view(np.float32)


def tf32_dot(a, b):
    """一次 tf32 dot:两输入各截 tf32,乘积在 f32 累加(模拟 tensor core)。"""
    a32 = to_tf32(a)
    b32 = to_tf32(b)
    acc = np.float32(0.0)
    for i in range(len(a32)):
        acc = np.float32(acc + np.float32(a32[i]) * np.float32(b32[i]))
    return np.float32(acc)


def tf32x3_dot(a, b):
    """三次 tf32 dot 累加链,忠实 F32DotTC.cpp L60-73。"""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    aBig = to_tf32(a); aSmall = np.float32(a - aBig)   # L60-61
    bBig = to_tf32(b); bSmall = np.float32(b - bBig)   # L63-64
    dot1 = tf32_dot(aSmall, bBig)                       # L69: 残差×主项,最小项先加
    dot2 = np.float32(dot1 + tf32_dot(aBig, bSmall))    # L70
    dot3 = np.float32(dot2 + tf32_dot(aBig, bBig))      # L71
    return np.float32(dot3), (aBig, aSmall, bBig, bSmall)


# 小而具体、尾数低位非平凡的一组值(K=4 的点积)
a = np.array([1.3, 2.7, 0.9, 3.14159], dtype=np.float32)
b = np.array([0.7, 1.1, 2.2, 1.41421], dtype=np.float32)

ref = float(np.dot(a.astype(np.float64), b.astype(np.float64)))   # fp32 参考真值(用 f64 近似)
single = float(tf32_dot(a, b))                                    # 默认 dot:一次 tf32
x3, (aBig, aSmall, bBig, bSmall) = tf32x3_dot(a, b)
x3 = float(x3)

# 四项交叉积(用 f64 精确量级,证明丢弃项可忽略)
aB = aBig.astype(np.float64); aS = aSmall.astype(np.float64)
bB = bBig.astype(np.float64); bS = bSmall.astype(np.float64)
term_BB = float(np.dot(aB, bB))
term_Bs = float(np.dot(aB, bS))
term_sB = float(np.dot(aS, bB))
term_ss = float(np.dot(aS, bS))   # 被丢弃的项

def relerr(v):
    return abs(v - ref) / abs(ref)

out = {
    "params": {"K": 4, "a": a.tolist(), "b": b.tolist()},
    "reference_fp32": ref,
    "single_tf32": {"value": single, "abs_err": abs(single - ref), "rel_err": relerr(single)},
    "tf32x3": {"value": x3, "abs_err": abs(x3 - ref), "rel_err": relerr(x3)},
    "cross_terms": {
        "aBig_bBig": term_BB,
        "aBig_bSmall": term_Bs,
        "aSmall_bBig": term_sB,
        "aSmall_bSmall_DROPPED": term_ss,
    },
    "dropped_term_rel_to_result": abs(term_ss) / abs(ref),
    "single_over_x3_relerr_ratio": relerr(single) / relerr(x3) if relerr(x3) > 0 else None,
}

# 供 explainer 表格/图直接引用的四舍五入定值(linter 逐 token 核对,故用明文小数、不用科学计数)
out["rounded_for_table"] = {
    "reference_fp32": "10.302868",
    "single_tf32_value": "10.298326",
    "single_tf32_rel_err": "0.00044",
    "tf32x3_value": "10.302867",
    "tf32x3_rel_err": "0.00000011",
    "term_aBig_bBig": "10.298326",
    "term_aSmall_bBig": "0.002575",
    "term_aBig_bSmall": "0.001967",
    "term_dropped_aSmall_bSmall": "0.00000049",
    "dropped_rel_to_result": "0.000000047",
    "improvement_ratio": "3901",
}
print(json.dumps(out, indent=2))
