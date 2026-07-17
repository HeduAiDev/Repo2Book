#!/usr/bin/env python3
"""ch28 数值验证:把 AccelerateMatmul 的三个纯算术函数手工移植到 Python,
用来核对 explainer 里的 worked-example 表格数字(v3.2.0 源码 1:1 逐分支移植)。

这不是官方 subtract-only 精简版(本章 kind=skip_impl,无精简版);仅作 explainer
手工推演的**算术校验器**,故 explainer 的 trace_source 仍标 manual。移植锚点:
  getMMAVersionSafe / supportMMA(DotOp) / supportMMA(Value)
      lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:L26-L45
      lib/Analysis/Utility.cpp:L481-L537
  warpsPerTileV2  AccelerateMatmul.cpp:L47-L104
  warpsPerTileV3  AccelerateMatmul.cpp:L106-L132
  mmaVersionToInstrShape  Utility.cpp:L26-L75
"""

# ---- dtype 位宽 & 集合谓词(只覆盖本章示例用到的 f16/bf16/f32/fp8/int8) ----
BITWIDTH = {"f16": 16, "bf16": 16, "f32": 32, "fp8": 8, "int8": 8}
FP8 = {"fp8"}
def is_f32(t): return t == "f32"
def is_int8(t): return t == "int8"


def supportMMA_value(elem, version):
    # lib/Analysis/Utility.cpp:L523-L537
    isFP8 = elem in FP8
    return (isFP8 or elem in ("f16", "bf16")
            or (is_f32(elem) and version >= 2)
            or (is_int8(elem) and version >= 2))


def supportMMA_dot(aElem, bElem, K, retShape, numWarps, inputPrecision, version):
    # lib/Analysis/Utility.cpp:L481-L521
    if version == 3:
        if K < 256 // BITWIDTH[aElem]:      # K 小于原生 mma 尺寸
            return False
        rank = len(retShape)
        if rank == 3:                       # batched 退 v2
            return False
        okDtype = aElem in ("fp8", "int8", "f16", "bf16", "f32")
        if not (numWarps % 4 == 0 and retShape[rank - 2] % 64 == 0
                and retShape[rank - 1] % 8 == 0 and okDtype):
            return False
        # (fp8 累加边界 case 略)
    if is_f32(aElem) and is_f32(bElem):
        return inputPrecision == "tf32" and version >= 2
    return supportMMA_value(aElem, version) and supportMMA_value(bElem, version)


def getMMAVersionSafe(cap, aElem, bElem, K, retShape, numWarps, inputPrecision):
    # lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:L26-L45
    if cap < 75:
        versions = [1]
    elif cap < 90:
        versions = [2]
    elif cap < 100:
        versions = [3, 2]
    else:
        raise AssertionError("cap not supported")
    trace = []
    for v in versions:
        ok = supportMMA_dot(aElem, bElem, K, retShape, numWarps, inputPrecision, v)
        trace.append((v, ok))
        if ok:
            return v, versions, trace
    return 0, versions, trace


def warpsPerTileV2(shape, numWarps, hasChainedDot=False):
    # lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:L47-L104 (rank==2 路径)
    if hasChainedDot:
        return ([numWarps, 1] if shape[0] >= shape[1] else [1, numWarps]), []
    ret = [1, 1]
    shapePerWarp = [16, 8]      # rank-2 -> 16, rank-1 -> 8
    steps = []
    while True:
        if ret[0] * ret[1] >= numWarps:
            steps.append((list(ret), None, None, "break"))
            break
        lhs = shape[0] // shapePerWarp[0] // ret[0]
        rhs = shape[1] // (shapePerWarp[1] * 2) // ret[1]
        before = list(ret)
        if lhs >= rhs:
            if ret[0] < shape[0] // shapePerWarp[0]:
                ret[0] *= 2; branch = "M*2"
            else:
                ret[1] *= 2; branch = "N*2(M满)"
        else:
            ret[1] *= 2; branch = "N*2"
        steps.append((before, lhs, rhs, branch))
    return ret, steps


def mmaVersionToInstrShape(version, shape, elt, numWarps):
    # lib/Dialect/TritonGPU/Transforms/Utility.cpp:L26-L75
    if version == 1:
        return [16, 16], {}
    if version == 2:
        return [16, 8], {}
    # version == 3
    k = 256 // BITWIDTH[elt]
    validN_float = [256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,
                    136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8]
    validN_int8 = [224,208,192,176,160,144,128,112,96,80,64,48,32,24,16,8]
    validN = validN_int8 if elt == "int8" else validN_float
    m = 16
    mWarps = max(shape[0] // m, 1)
    nWarps = max(numWarps // mWarps, 1)
    maxN = max(shape[1] // nWarps, 8)
    for n in validN:
        if shape[1] % n == 0 and n <= maxN:
            return [m, n, k], {"k": k, "mWarps": mWarps, "nWarps": nWarps, "maxN": maxN, "n": n}
    raise AssertionError("type not supported")


def warpsPerTileV3(shape, numWarps, instrShape, hasChainedDot=False):
    # lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:L106-L132
    if hasChainedDot:
        return [numWarps, 1]
    ret = [4, 1]
    shapePerWarp = [16, instrShape[1]]
    while True:
        if ret[0] * ret[1] >= numWarps:
            break
        if shape[0] > shapePerWarp[0] * ret[0]:
            ret[0] *= 2
        else:
            ret[1] *= 2
    return ret


def main():
    print("=" * 68)
    print("[mma-version-select] getMMAVersionSafe(cap, dot) 逐场景")
    print("cap  A/B    K   retShape  nW  prec  | candidates -> per-version -> chosen")
    scen = [
        (70, "f16", "f16", 64, (128,128), 4, "ieee"),
        (80, "f16", "f16", 64, (128,128), 4, "ieee"),
        (90, "f16", "f16", 64, (128,128), 4, "ieee"),
        (90, "f16", "f16",  8, (128,128), 4, "ieee"),   # K 太小 -> 退 v2
        (90, "f32", "f32", 64, (128,128), 4, "tf32"),   # TF32 -> v3
        (90, "f32", "f32", 64, (128,128), 4, "ieee"),   # 非 TF32 -> 0(退回 FMA)
    ]
    for cap, a, b, K, rs, nw, prec in scen:
        chosen, cands, tr = getMMAVersionSafe(cap, a, b, K, rs, nw, prec)
        print(f"{cap:3} {a}/{b} {K:3} {str(rs):9} {nw:2} {prec:5} | {cands} -> {tr} -> chosen=v{chosen}")

    print("=" * 68)
    print("[warps-per-tile v2] shape=(128,128) numWarps=8 单 dot 贪心")
    ret, steps = warpsPerTileV2((128,128), 8)
    for i, (before, lhs, rhs, branch) in enumerate(steps, 1):
        print(f"  iter{i}: ret={before} prod={before[0]*before[1]} LHS={lhs} RHS={rhs} -> {branch}")
    print(f"  => warpsPerTile = {ret} (total {ret[0]*ret[1]}), per-warp tile = "
          f"[{128//ret[0]},{128//ret[1]}]")
    ret_c, _ = warpsPerTileV2((128,64), 8, hasChainedDot=True)
    print(f"  chained dot shape=(128,64) nW=8 (M>=N) => {ret_c}")

    print("=" * 68)
    print("[instr-shape] mmaVersionToInstrShape")
    for (v, shp, elt, nw) in [(1,(128,128),"f16",4), (2,(128,128),"f16",4),
                              (3,(128,128),"f16",4), (3,(32,128),"f16",8)]:
        ins, dbg = mmaVersionToInstrShape(v, shp, elt, nw)
        print(f"  v{v} elt={elt} shape={shp} nW={nw} -> instrShape={ins}  {dbg}")

    print("=" * 68)
    print("[warps-per-tile v3]")
    ins,_ = mmaVersionToInstrShape(3,(128,128),"f16",8)
    print(f"  v3 f16 shape=(128,128) nW=8 instrShape[1]={ins[1]} -> "
          f"{warpsPerTileV3((128,128),8,ins)}")
    print(f"  v3 chained (flash-attn) nW=8 -> {warpsPerTileV3((128,128),8,ins,hasChainedDot=True)}")


if __name__ == "__main__":
    main()
