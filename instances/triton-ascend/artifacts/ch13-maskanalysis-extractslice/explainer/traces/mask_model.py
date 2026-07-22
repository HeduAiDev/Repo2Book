#!/usr/bin/env python3
"""ch13 MaskAnalysis 数值验证器(host 纯 Python 复刻 pin 源码公式)。

WHY manual/model:本章 skip_impl,是纯 C++ MLIR pass;宿主无 CANN,无法跑
triton-opt 产真实编译器 dump。此脚本**不是**精简版运行——它按 pin 源码
(MaskAnalysis.cpp @2badfc89e)里逐行的整数公式手工复刻,用来自校验 explainer
表格里的每个手算数字。公式出处以 file:Lxxx 标注在各函数 docstring。
真实 IR 前后对照另取自 pin 内 lit 夹具(见 explainer.json figure numbers 的 provenance)。
"""
import json


def parse_make_range(start, end, shape0):
    """parseMakeRange @ MaskAnalysis.cpp:L560-L582.
    stride=(end-start+shape-1)/shape,必须==1;填 start/end/dims[0]=shape/offsets[0]=0。"""
    stride = (end - start + shape0 - 1) // shape0
    ok = (stride == 1)
    return {"ok": ok, "stride": stride, "start": start, "end": end,
            "dims": [shape0], "offsets": [0]}


def clamp(v):
    """clampToNonNegativeIndex @ L68-L79(仅常量分支;非常量原样返回)。"""
    return max(0, v)


def parse_cmp(rng, bound, pred):
    """parseCmp @ L440-L558。rng={start,end,dims,offsets};一维 range,cmpDim=0。
    返回 (offsets,dims)。bound=rhs 标量。"""
    start, end = rng["start"], rng["end"]
    off = list(rng["offsets"])
    dims = list(rng["dims"])
    if pred == "slt":  # L501-L510
        realBound = max(start, bound)
        newEnd = min(end, realBound)
        newDim = newEnd - start
        dims[0] = clamp(newDim)
    elif pred == "sle":  # L512-L521  (lhs<=rhs <=> lhs<rhs+1)
        realBound = max(start, bound + 1)
        newEnd = min(end, realBound)
        newDim = newEnd - start
        dims[0] = clamp(newDim)
    elif pred == "sge":  # L523-L532  (唯一改 offset)
        realBound = max(start, bound)
        newStart = min(end, realBound)
        off[0] = newStart - start
        newDim = end - newStart
        dims[0] = clamp(newDim)
    elif pred == "eq":  # L534-L541
        off[0] = bound - start
        dims[0] = 1
    elif pred == "ne":  # L544-L553  (only lhs!=0;保持整维)
        assert bound == 0
        # offsets/dims 保持 lhs 原值(此处即 range 的 dims/offsets)
    return {"offsets": off, "dims": dims}


def min_states(a, b):
    """minStates @ L280-L306。逐维:newOff=max(off),newEnd=min(off+dim),newDim=clamp(end-off)。"""
    off, dims = [], []
    for i in range(len(a["dims"])):
        lo, ro = a["offsets"][i], b["offsets"][i]
        newOff = max(lo, ro)
        lend = lo + a["dims"][i]
        rend = ro + b["dims"][i]
        newEnd = min(lend, rend)
        dims.append(clamp(newEnd - newOff))
        off.append(newOff)
    return {"offsets": off, "dims": dims}


def add_state_scalar(rng, scalar):
    """addStateScalar @ L209-L225。start/end += scalar;dims/offsets 透传。"""
    return {"start": rng["start"] + scalar, "end": rng["end"] + scalar,
            "dims": list(rng["dims"]), "offsets": list(rng["offsets"])}


out = {}

# ---- m5 parseMakeRange ----
out["m5"] = {
    "range_0_16_s16": parse_make_range(0, 16, 16),
    "range_0_128_s128": parse_make_range(0, 128, 128),
    "range_0_32_s16_bad": parse_make_range(0, 32, 16),  # stride=2 → fail(defensive)
}

# ---- m3 parseCmp 5 谓词,range=[0,16), bound=10 ----
rng = parse_make_range(0, 16, 16)
out["m3"] = {p: parse_cmp(rng, 10, p)
             for p in ("slt", "sle", "sge", "eq")}
out["m3"]["ne"] = parse_cmp(rng, 0, "ne")  # ne 只支持 !=0

# ---- m4 parseAnd→minStates,16x16,行 slt<10,列 slt<12 ----
# 行掩码 A:dim0<10 broadcast → offsets[0,0] dims[10,16]
A = {"offsets": [0, 0], "dims": [10, 16]}
# 列掩码 B:dim1<12 broadcast → offsets[0,0] dims[16,12]
B = {"offsets": [0, 0], "dims": [16, 12]}
out["m4"] = {"A": A, "B": B, "intersection": min_states(A, B)}

# ---- m7 clamp:常量夹 / 非常量原样 ----
# 不相交 AND:A=[0,4) B=[10,16) → dim0 newDim=4-10=-6 → clamp 0
disjoint = min_states({"offsets": [0], "dims": [4]}, {"offsets": [10], "dims": [6]})
out["m7"] = {
    "disjoint_neg6_clamped": {"raw_newDim": 4 - 10, "clamped": clamp(4 - 10),
                              "result": disjoint},
    "normal_10": {"raw": 10, "clamped": clamp(10)},
    "nonconst": "getConstantIntValue 失败 → 原样返回(不发 max),atomic UT 回归缘由(L76-L78)",
}

# ---- m9 scalar 分支:parseAdd → addStateScalar,range[0,16)+5 ----
out["m9"] = {"shifted": add_state_scalar(parse_make_range(0, 16, 16), 5)}

# ---- m2 full parse:make_range(0,16) slt const(10) → extract_slice[0:10] ----
m2rng = parse_make_range(0, 16, 16)
out["m2"] = {"leaf_range": m2rng, "after_slt_10": parse_cmp(m2rng, 10, "slt")}

# ---- m10 select→ne roundtrip:cond=(range slt 10) → dims[10] 复现 ----
cond = parse_cmp(parse_make_range(0, 16, 16), 10, "slt")  # condState offsets[0] dims[10]
# parseSel 复制 cond 的 offsets/dims;cmpi ne(!=0)保持不变
out["m10"] = {"cond_mask": cond, "after_select_ne0": cond}

print(json.dumps(out, indent=2, ensure_ascii=False))
