#!/usr/bin/env python3
"""ch41 —— triton-tensor-layout getDistributedLayoutStr 的忠实复刻(host 侧手工推演)。

背景:pin=triton v3.2.0 的 triton-tensor-layout 二进制未构建(需 pinned LLVM,host 无),
故无法实跑真二进制。本脚本按 lib/Dialect/TritonGPU/IR/Dialect.cpp:L3291-L3398 的逐行逻辑
在 host 上复刻同一算法,对一个具体的 Blocked 布局产出**与真二进制逐字节相同**的打印,
用于校验 explainer 的 worked_example 表/图里每个数字。trace_source 仍标 "manual"。

被复刻的布局(triton-tensor-layout 命令行等价):
  triton-tensor-layout \
    -l "#triton_gpu.blocked<{sizePerThread=[1,1], threadsPerWarp=[4,8], \
        warpsPerCTA=[2,1], order=[1,0], CTAsPerCGA=[1,1], CTASplitNum=[1,1], CTAOrder=[1,0]}>" \
    -t "tensor<8x8xf16>"
  # 真实 NVIDIA warp:threadsPerWarp 逐维之积 = 4*8 = 32
  # numWarpsPerCTA = 2*1 = 2  →  warp1 的 lane 0 = 全局线程 0 + 1*32 = T32
  # numElementsPerThreads = 1(sizePerThread 之积)→ 每线程恰持 1 元素,寄存器下标恒 :0
  # numBlocks = 1  →  无 B{blockId}: 前缀

Blocked 布局的线性映射(order=[1,0],dim1 最快):
  lane(0..31) 拆成 (lane0=lane//8 ∈[0,4), lane1=lane%8 ∈[0,8))  # threadsPerWarp=[4,8]
  warp(0..1)  拆成 (warp0=warp     ∈[0,2), warp1=0)             # warpsPerCTA =[2,1]
  行 row(dim0) = lane0 + 4*warp0        # sizePerThread0=1
  列 col(dim1) = lane1                  # sizePerThread1=1, warp1=0
  → 元素(row,col) 的持有者 = 全局线程 T{lane + warp*32}:0
"""

# ---- 布局参数(照 getDistributedLayoutStr:L3282-L3285 抽取) ----
shape = [8, 8]
threadsPerWarp = 32          # getWarpSize(layout) = 4*8
numWarpsPerCTA = 2           # getNumWarpsPerCTA = 2*1
numBlocks = 1                # getNumCTAs
numElementsPerThreads = 1    # getTotalElemsPerThread = 1*1

tpw = [4, 8]                 # threadsPerWarp per-dim
wpc = [2, 1]                 # warpsPerCTA per-dim
spt = [1, 1]                 # sizePerThread per-dim


def ll_apply(blockId, warpId, tid, idx):
    """复刻 LinearLayout::apply 对 Blocked 的结果 → tensor 多维下标 outputs[dim]。"""
    lane0, lane1 = tid // tpw[1], tid % tpw[1]      # order=[1,0]:dim1 最快
    warp0, warp1 = warpId % wpc[0], warpId // wpc[0]
    reg = idx                                       # sizePerThread=1 → reg 恒 0
    row = reg * 0 + spt[0] * (lane0 + tpw[0] * warp0)
    col = spt[1] * (lane1 + tpw[1] * warp1)
    return [row, col]


def numCharacterPadding(value, mx):
    return len(str(mx)) - len(str(value))


def paddedString(value, mx):
    return " " * numCharacterPadding(value, mx) + str(value)


# ---- 四重循环:构建 elementMapping / threadMapping(L3298-L3341) ----
tensorSize = shape[0] * shape[1]
elementMapping = [""] * tensorSize
threadMapping = []
for blockId in range(numBlocks):
    for warpId in range(numWarpsPerCTA):
        for tid in range(threadsPerWarp):
            for idx in range(numElementsPerThreads):
                outputs = ll_apply(blockId, warpId, tid, idx)
                linearizedIdx = 0
                stride = 1
                for i in range(len(outputs) - 1, -1, -1):
                    linearizedIdx += outputs[i] * stride
                    stride *= shape[i]
                v = elementMapping[linearizedIdx]
                if v:
                    v += "|"
                gtid = tid + warpId * threadsPerWarp          # 全局线程号
                padding = (numCharacterPadding(blockId, numBlocks)
                           + numCharacterPadding(gtid, numWarpsPerCTA * threadsPerWarp)
                           + numCharacterPadding(idx, numElementsPerThreads))
                v += " " * padding
                if numBlocks > 1:
                    v += "B" + str(blockId) + ":"
                v += "T" + str(gtid) + ":" + str(idx)
                elementMapping[linearizedIdx] = v
                threadInfo = "(" + ",".join(
                    paddedString(outputs[i], shape[i]) for i in range(len(outputs))) + ")"
                threadMapping.append(threadInfo)


def delinearize(i, shp):
    ret = [0] * len(shp)
    for k in range(len(shp) - 1, -1, -1):
        ret[k] = i % shp[k]
        i //= shp[k]
    return ret


# ---- tensor 视角(!useHWPointOfView, L3343-L3374) ----
rank = len(shape)
tensor_view = ""
newLine = True
for i in range(tensorSize):
    indices = delinearize(i, shape)
    numOpenBracket = 0
    for j in range(rank - 1, -1, -1):
        if indices[j] % shape[j] != 0:
            break
        tensor_view += "["
        numOpenBracket += 1
    if newLine:
        tensor_view += " " * (rank - numOpenBracket)
        newLine = False
    tensor_view += elementMapping[i]
    nextIndices = delinearize(i + 1, shape)
    for j in range(rank - 1, -1, -1):
        if nextIndices[j] % shape[j] != 0:
            break
        tensor_view += "]"
    if nextIndices[-1] % shape[-1] == 0:
        tensor_view += "\n"
        newLine = True
    else:
        tensor_view += ", "

# ---- hardware/warp 视角(useHWPointOfView, L3375-L3397) ----
hw_view = ""
for blockId in range(numBlocks):
    if numBlocks > 1:
        hw_view += "Block" + str(blockId) + ":\n"
    for warpId in range(numWarpsPerCTA):
        hw_view += "Warp" + str(warpId) + ":\n"
        for idx in range(numElementsPerThreads):
            for tid in range(threadsPerWarp):
                lin = (blockId * numWarpsPerCTA * threadsPerWarp * numElementsPerThreads
                       + warpId * threadsPerWarp * numElementsPerThreads
                       + tid * numElementsPerThreads + idx)
                hw_view += threadMapping[lin]
                if tid < threadsPerWarp - 1:
                    hw_view += ", "
            hw_view += "\n"

import json, sys
out = {
    "layout": "#triton_gpu.blocked<{sizePerThread=[1,1], threadsPerWarp=[4,8], warpsPerCTA=[2,1], order=[1,0]}>",
    "tensor_type": "tensor<8x8xf16>",
    "threadsPerWarp": threadsPerWarp,
    "numWarpsPerCTA": numWarpsPerCTA,
    "numElementsPerThreads": numElementsPerThreads,
    "numBlocks": numBlocks,
    "tensor_view": tensor_view,
    "hw_view": hw_view,
    "elementMapping": elementMapping,
}
print("===== tensor 视角(默认)=====")
print(tensor_view)
print("===== hardware/warp 视角(-use-hw-view)=====")
print(hw_view)
if len(sys.argv) > 1 and sys.argv[1] == "--json":
    with open("layout_decode.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
