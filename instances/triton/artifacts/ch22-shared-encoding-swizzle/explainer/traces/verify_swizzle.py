#!/usr/bin/env python3
"""ch22 explainer 交叉验证 — 本章无精简版(skip_impl:.td/C++ 无法做 Python 只删不增)。

用途:把 SharedEncodingAttr 的 swizzle *定义*(.td 描述里的统一公式)用 Python 逐位复现,
与 TritonGPUAttrDefs.td 里五个手算例子的表逐格比对(必须完全一致),再把 mma builder 的
参数反推算术(MFMA/Ampere/Hopper 三分支)照 .td 源码逐行代入具体参数算出 (vec,perPhase,maxPhase)。
所有硬件常量取自 pin=v3.2.0 源码,行号标在注释里。输出 JSON 存 swizzle_trace.json。

真值口径:.td 里五个例子的表 = 源码作者手写的 ground truth;本脚本复现它 => 公式正确。
参数反推分支:照 .td L282-L456 的算术逐行搬,给具体输入算输出;headless 无需编译器。
"""
import json

# ---- 统一 swizzle 公式(由 .td L166-L231 五例归纳) ----
# out[r][c] = in[r][ (floor(c/vec) ^ phase(r)) * vec + (c % vec) ],  phase(r)=(floor(r/perPhase)) % maxPhase
def swizzle_col(r, c, vec, perPhase, maxPhase):
    phase = (r // perPhase) % maxPhase
    group = c // vec
    return (group ^ phase) * vec + (c % vec)

def build_table(nrows, ncols, vec, perPhase, maxPhase):
    """返回 out[r][c] 上摆放的 *逻辑元素值*(输入张量 = 0..n-1,行主序 => in[r][c]=r*ncols+c)。
    out 存的是 in 的哪个逻辑元素:out_phys[r][c] = in[r][ swizzle_col ] = r*ncols + swizzle_col."""
    tab = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            sc = swizzle_col(r, c, vec, perPhase, maxPhase)
            row.append(r * ncols + sc)
        tab.append(row)
    return tab

# ---- .td 里五个例子的 ground-truth 表(逐字抄自 TritonGPUAttrDefs.td L168-L225) ----
GT = {
    "ex1_basic":            {"params": (1,1,4), "shape": (4,4),
        "table": [[0,1,2,3],[5,4,7,6],[10,11,8,9],[15,14,13,12]]},
    "ex2_perPhase2":        {"params": (1,2,4), "shape": (4,4),
        "table": [[0,1,2,3],[4,5,6,7],[9,8,11,10],[13,12,15,14]]},
    "ex3_maxPhase2":        {"params": (1,1,2), "shape": (8,4),
        "table": [[0,1,2,3],[5,4,7,6],[8,9,10,11],[13,12,15,14],
                  [16,17,18,19],[21,20,23,22],[24,25,26,27],[29,28,31,30]]},
    "ex4_perPhase2_maxPhase2":{"params": (1,2,2), "shape": (8,4),
        "table": [[0,1,2,3],[4,5,6,7],[9,8,11,10],[13,12,15,14],
                  [16,17,18,19],[20,21,22,23],[25,24,27,26],[29,28,31,30]]},
    "ex5_vec2":             {"params": (2,1,4), "shape": (4,8),
        "table": [[0,1,2,3,4,5,6,7],[10,11,8,9,14,15,12,13],
                  [20,21,22,23,16,17,18,19],[30,31,28,29,26,27,24,25]]},
}

def verify_examples():
    out = {}
    for name, g in GT.items():
        vec, per, mx = g["params"]
        nr, nc = g["shape"]
        got = build_table(nr, nc, vec, per, mx)
        match = (got == g["table"])
        out[name] = {
            "params": {"vec": vec, "perPhase": per, "maxPhase": mx},
            "shape": [nr, nc],
            "phase_per_row": [ (r // per) % mx for r in range(nr) ],
            "formula_table": got,
            "td_ground_truth": g["table"],
            "bit_exact_match": match,
        }
    return out

# ---- bank 冲突量化(把 swizzle 消冲突讲成具体倍数) ----
# 场景:一片 32x32 的 4B-word 共享内存(rowstride=32 word => bank=(r*32+physcol)%32=physcol%32)。
# 一个 warp 沿 order[0] 读逻辑列 c=0 的 32 行。bank = 物理列 % 32。
def bank_conflict(nrows, vec, perPhase, maxPhase, logical_col=0):
    banks = {}
    for r in range(nrows):
        physcol = swizzle_col(r, logical_col, vec, perPhase, maxPhase)
        b = physcol % 32
        banks.setdefault(b, []).append(r)
    distinct = len(banks)
    max_rows_same_bank = max(len(v) for v in banks.values())  # n-way conflict
    return {"distinct_banks": distinct, "way_conflict": max_rows_same_bank,
            "banks": {str(k): v for k, v in sorted(banks.items())}}

def bank_conflict_demo():
    # 无 swizzle: 等价 maxPhase=1(phase 恒 0),读列 0 => 全落 bank 0
    noswz = bank_conflict(32, vec=1, perPhase=1, maxPhase=1)
    # maxPhase=8 的软件 swizzle(Hopper 128B 档 / Ampere inner 常见)
    swz8 = bank_conflict(32, vec=1, perPhase=1, maxPhase=8)
    # maxPhase=4 档
    swz4 = bank_conflict(32, vec=1, perPhase=1, maxPhase=4)
    return {"no_swizzle_maxPhase1": noswz, "swizzle_maxPhase8": swz8,
            "swizzle_maxPhase4": swz4}

# ---- MFMA 参数反推(照 .td L282-L311 逐行) ----
def mfma_derive(typeBitWidth, innerDimLength, kWidth, mDim=32):
    numBanks = 32          # td:L288
    bankBitWidth = 32      # td:L289
    SIMDWidth = 16         # td:L290
    elemsPerOneBanksRow = (numBanks * bankBitWidth) // typeBitWidth   # td:L294
    perPhase = max(1, elemsPerOneBanksRow // innerDimLength)          # td:L296
    vecSize = kWidth                                                  # td:L298
    maxPhase = min(SIMDWidth // perPhase, innerDimLength // vecSize)  # td:L299
    if mDim == 4:          # td:L302-303
        maxPhase = 4
    return {"inputs": {"typeBitWidth": typeBitWidth, "innerDimLength": innerDimLength,
                       "kWidth": kWidth, "mDim": mDim},
            "elemsPerOneBanksRow": elemsPerOneBanksRow,
            "vec": vecSize, "perPhase": perPhase, "maxPhase": maxPhase}

# ---- Ampere 参数反推(照 .td L364-L392 逐行, opIdx=0 A operand, 非转置) ----
def ampere_derive(typeBitWidth, shape_inner, kWidth, opIdx=0, needTrans=False):
    perPhase = 128 // (shape_inner * 4 // kWidth)      # td:L366
    perPhase = max(perPhase, 1)                         # td:L367
    matShape = [8, 8, 4 * kWidth]                        # td:L368  目标 mma 指令 tile M,N,K
    vecWidth = 32 // typeBitWidth                        # td:L369
    rank = 2
    inner = rank - 1
    # td:L370-372: vecWidth != kWidth 且 order[0]==inner 时 perPhase 抬高
    order0_is_inner = True  # 取 order[0]==inner(K 最内圈,最常见走 swizzle 的情形)
    if vecWidth != kWidth and order0_is_inner:
        perPhase = max(perPhase, 2 * vecWidth)
    order0_is_last = True   # order[0]==rank-1
    if opIdx == 0:
        m = matShape[2] if needTrans else matShape[0]   # td:L376-377
        k = matShape[0] if needTrans else matShape[2]
        vec = k if order0_is_last else m                # td:L378
        mmaStride = m if order0_is_last else k          # td:L379
        maxPhase = mmaStride // perPhase                # td:L380
    else:
        n = matShape[2] if needTrans else matShape[1]
        k = matShape[1] if needTrans else matShape[2]
        vec = n if order0_is_last else k
        mmaStride = k if order0_is_last else n
        maxPhase = mmaStride // perPhase
    return {"inputs": {"typeBitWidth": typeBitWidth, "shape_inner": shape_inner,
                       "kWidth": kWidth, "opIdx": opIdx},
            "matShape_MNK": matShape, "vecWidth": vecWidth,
            "vec": vec, "perPhase": perPhase, "maxPhase": maxPhase, "mmaStride": mmaStride}

# ---- Hopper/MMAv3 by-eltTy 三档(照 .td L437-L455 逐行) ----
def hopper_derive(eleBitWidth, shape_inner):
    vec = 128 // eleBitWidth                             # td:L438
    contigByte = shape_inner * eleBitWidth // 8          # td:L441
    if contigByte >= 128 and contigByte % 128 == 0:      # td:L442-444
        perPhase, maxPhase = 1, 8
    elif contigByte >= 64 and contigByte % 64 == 0:      # td:L445-447
        perPhase, maxPhase = 2, 4
    elif contigByte >= 32 and contigByte % 32 == 0:      # td:L448-450
        perPhase, maxPhase = 4, 2
    else:
        perPhase, maxPhase = None, None                  # td:L452 llvm_unreachable
    return {"inputs": {"eleBitWidth": eleBitWidth, "shape_inner": shape_inner},
            "contigDimSizeInByte": contigByte, "vec": vec,
            "perPhase": perPhase, "maxPhase": maxPhase,
            "perPhase_x_maxPhase": (perPhase * maxPhase) if perPhase else None,
            "hasLeadingOffset": True}

def main():
    result = {
        "pin": "triton v3.2.0",
        "note": "manual cross-check: swizzle 公式逐位复现 .td 五例 + mma builder 参数反推逐行代入",
        "examples": verify_examples(),
        "bank_conflict": bank_conflict_demo(),
        "mfma": {
            # fp16, K=32, kWidth=4  => perPhase=2(非平凡), vec=4, maxPhase=8
            "fp16_K32_kW4": mfma_derive(16, 32, 4),
            # fp16, K=64, kWidth=4  => perPhase=1, vec=4, maxPhase=16
            "fp16_K64_kW4": mfma_derive(16, 64, 4),
        },
        "ampere": {
            # fp16, inner=32, kWidth=2, A operand => perPhase=2(非平凡), vec=8, maxPhase=4
            "fp16_inner32_kW2_opA": ampere_derive(16, 32, 2, opIdx=0),
            # fp16, inner=64, kWidth=2, A operand => perPhase=1, vec=8, maxPhase=8
            "fp16_inner64_kW2_opA": ampere_derive(16, 64, 2, opIdx=0),
        },
        "hopper": {
            # I8, inner=128 => 128B, (perPhase,maxPhase)=(1,8), vec=16
            "i8_inner128": hopper_derive(8, 128),
            # I8, inner=64  => 64B,  (2,4), vec=16
            "i8_inner64": hopper_derive(8, 64),
            # I8, inner=32  => 32B,  (4,2), vec=16
            "i8_inner32": hopper_derive(8, 32),
            # fp16, inner=64 => 128B, (1,8), vec=8
            "fp16_inner64": hopper_derive(16, 64),
        },
    }
    all_match = all(v["bit_exact_match"] for v in result["examples"].values())
    result["all_five_examples_bit_exact"] = all_match
    with open("swizzle_trace.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("=== SUMMARY (full data in swizzle_trace.json) ===")
    for name, v in result["examples"].items():
        print(f"{name}: bit_exact_match={v['bit_exact_match']}  phase_per_row={v['phase_per_row']}")
    bc = result["bank_conflict"]
    print(f"bank conflict col-read(32 rows): no_swizzle={bc['no_swizzle_maxPhase1']['way_conflict']}-way "
          f"({bc['no_swizzle_maxPhase1']['distinct_banks']} bank) -> "
          f"maxPhase8={bc['swizzle_maxPhase8']['way_conflict']}-way "
          f"({bc['swizzle_maxPhase8']['distinct_banks']} banks)")
    print(f"MFMA fp16 K=32 kW=4 -> {result['mfma']['fp16_K32_kW4']['vec']},"
          f"{result['mfma']['fp16_K32_kW4']['perPhase']},{result['mfma']['fp16_K32_kW4']['maxPhase']}")
    print(f"Ampere fp16 inner=32 kW=2 opA -> vec={result['ampere']['fp16_inner32_kW2_opA']['vec']},"
          f"perPhase={result['ampere']['fp16_inner32_kW2_opA']['perPhase']},"
          f"maxPhase={result['ampere']['fp16_inner32_kW2_opA']['maxPhase']}")
    print(f"ALL FIVE EXAMPLES BIT-EXACT: {all_match}")

if __name__ == "__main__":
    main()
