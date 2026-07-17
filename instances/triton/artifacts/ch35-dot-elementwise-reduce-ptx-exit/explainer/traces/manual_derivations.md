# ch35 手工推演记录（trace_source=manual）

本章 kind=skip_impl，**无精简版可跑**；且四个 worked_example 机制（fp8 转换 / reduce
蝶形树 / scan Kogge-Stone / break-phi-struct）的"实测轨迹"须在 CUDA 容器里 dump make_llir
产物才能真观测（见 dossier.pin_forensics_plan，该取证是 illustrator/tester 的活，archivist
落 trace）。作为 explainer，此处按**源码常量 + 算法语义手工推演**，每个引用源码常量的数字
标 `file:Lxxx`；纯算术（蝶形和、前缀和、指令计数）读者可心算复核。

---

## 1. fp8-conversion —— fp16↔fp8e4m3 打包 cvt

源码常量（`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp`）：

- `Fp8E4M3Nv_to_Fp16`（fp8→fp16）ElementwiseOpToLLVM.cpp:L180-L185
  = `{ptx="cvt.rn.f16x2.e4m3x2 $0,$1", inVecWidthBits=16, outVecWidthBits=32, numElements=2}`
- `Fp16_to_Fp8E4M3Nv`（fp16→fp8）ElementwiseOpToLLVM.cpp:L187-L191
  = `{ptx="cvt.rn.satfinite.e4m3x2.f16x2 $0,$1", inVecWidthBits=32, outVecWidthBits=16, numElements=2}`
- 约束选择 makeConverterFromPtx: `outConstraint = outVecWidthBits==16 ? "=h" : "=r"`；
  `inConstraint = inVecWidthBits==16 ? "h" : "r"` ElementwiseOpToLLVM.cpp:L288-L289
- 查表入口 srcMap: `{F8E4M3TyID,F16TyID,undef}→Fp8E4M3Nv_to_Fp16`（L417）；
  `{F16TyID,F8E4M3TyID,RTNE}→Fp16_to_Fp8E4M3Nv`（L420）

推演（位宽守恒校验）：
- fp16→fp8：输入 2 元素 × 16bit = 32bit（字段 inVecWidthBits=32 ✓）；输出 2 × 8bit = 16bit
  （字段 outVecWidthBits=16 ✓）→ in 约束 "r"、out 约束 "=h"；一条 cvt 转 2 元素。
- fp8→fp16：输入 2 × 8bit = 16bit（inVecWidthBits=16 ✓）；输出 2 × 16bit = 32bit
  （outVecWidthBits=32 ✓）→ in 约束 "h"、out 约束 "=r"；一条 cvt 转 2 元素。
- 两方向位宽互为镜像 (16,32) ↔ (32,16)，与 dossier 命门修正值逐字一致。

跨代对比（theory §5）：sm_89+ 上述一条 cvt 转 2 元素；pre-sm_89 的 e5m2 转换（hasNativeFP=false
分支 ElementwiseOpToLLVM.cpp:L54-L118）退化为 ~20 条 prmt/lop3/shr/mul 位操作序列模拟。

---

## 2. reduce-shfl-tree —— warp 内蝶形（shfl.bfly）求和

源码：`lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp` warpReduce
`for (N = numLaneToReduce/2; N>0; N>>=1) { shfl = shuffleXor(acc, N*interleave); accumulate }`
ReduceOpToLLVM.cpp:L166-L172。

参数：warp=32 车道，combine=sum，interleave=1，车道 i 初值 = i（0..31），全和 = Σ0..31 = 496。
N 序列 = 16,8,4,2,1（5 步 = log2(32)）。每步 lane j 与 lane (j XOR N) 交换并相加。

追踪 lane 0 的累加（每步覆盖车道数翻倍）：

| 步 | N  | 伙伴 lane(0^N) | 本步新并入车道集         | 本步并入和 | lane0 累积 acc | 覆盖车道数 |
|----|----|----------------|--------------------------|-----------|---------------|-----------|
| 0  | -  | -              | {0}                      | 0         | 0             | 1         |
| 1  | 16 | 16             | {16}                     | 16        | 16            | 2         |
| 2  | 8  | 8              | {8,24}                   | 32        | 48            | 4         |
| 3  | 4  | 4              | {4,20,12,28}             | 64        | 112           | 8         |
| 4  | 2  | 2              | {2,18,10,26,6,22,14,30}  | 128       | 240           | 16        |
| 5  | 1  | 1              | 其余 16 个奇偶混合车道   | 256       | 496           | 32        |

复核：16 / 48 / 112 / 240 / 496 累积；覆盖 1→2→4→8→16→32 严格翻倍；5 步后 = 全和 496。
每个数字均可心算：例如 {8,24}=32、{4,20,12,28}=64、{2,18,10,26,6,22,14,30}=128。

---

## 3. scan-kogge-stone —— warp 内前缀和（shfl.up）

源码：`lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp` warpScan
`for (i=1; i<=scanDim/2; i<<=1) { shfl=shuffleUp(acc, i*threadStride);
mask=icmp_sge(lane,i); tempAcc=accumulate(shfl,acc,mask); acc=select(mask,tempAcc,acc) }`
ScanOpToLLVM.cpp:L67-L78。

参数（缩小到 scanDim=8 便于心算）：8 车道，combine=sum，各 lane 初值 = 1，
期望前缀和 = [1,2,3,4,5,6,7,8]。i 序列 = 1,2,4（i<=scanDim/2=4，3 步 = log2(8)）。
每步：lane j 收到 shuffleUp(i) = lane(j-i) 的值；仅当 lane>=i（mask）才累加，否则保持。

逐步车道向量 acc[0..7]：

| 步 | i | mask 生效车道 | acc[0..7]（8 车道向量）      |
|----|---|---------------|-----------------------------|
| 0  | - | -             | [1,1,1,1,1,1,1,1]           |
| 1  | 1 | lane>=1       | [1,2,2,2,2,2,2,2]           |
| 2  | 2 | lane>=2       | [1,2,3,4,4,4,4,4]           |
| 3  | 4 | lane>=4       | [1,2,3,4,5,6,7,8]           |

复核第 3 步：lane4 = 4 + acc[0]=1 → 5；lane5 = 4 + acc[1]=2 → 6；lane6 = 4 + acc[2]=3 → 7；
lane7 = 4 + acc[3]=4 → 8。lane0..3 因 lane<4（mask 假）保持 [1,2,3,4]。得正确前缀和。
第 k 步后 lane j 持有 min(2^k, j+1) 个前驱之和：如第 2 步 lane3 = min(4,4)=4 个前驱 → 4 ✓。

---

## 4. break-phi-struct —— struct phi 拆标量 phi

源码：`lib/Target/LLVMIR/LLVMIRBreakPhiStruct.cpp` processPhiStruct
外层 `for (i=0; i<numScalarEl; i++)` 建标量 phi；内层 `for (j=0; j<numOperands; ++j)`
在每条 incoming 边 terminator 处 `CreateExtractValue(operand,i)`；末尾 `CreateInsertValue`
重组、`replaceAllUsesWith(newStruct)`。LLVMIRBreakPhiStruct.cpp:L21-L36。

参数：一个 struct 类型 `{i32, i32}` → numScalarEl=2（LLVMIRBreakPhiStruct.cpp:L18
`STy->getNumElements()`）；循环头 phi 有 2 条 incoming 边（preheader + latch）→ numOperands=2
（L17 `getNumIncomingValues()`）。

逐元素处理（外层循环 i）：

| 元素 i | 新建标量 phi | extractvalue（每 incoming 各 1 条） | 累积 insertvalue | 该 phi 的 incoming 数 |
|--------|-------------|-------------------------------------|-----------------|----------------------|
| 0      | phi0:i32    | 2（op0→e0, op1→e0）                 | 1（装入槽 0）   | 2                    |
| 1      | phi1:i32    | 2（op0→e1, op1→e1）                 | 2（装入槽 1）   | 2                    |

出口计数：标量 phi 数 = numScalarEl = 2；extractvalue 总数 = numScalarEl × numOperands
= 2×2 = 4；insertvalue 数 = numScalarEl = 2；每个标量 phi 的 incoming 数 = numOperands = 2
（与原 struct phi 一致）。语义等价：insertvalue 重组的 struct 与原 struct phi 逐元素相等，
replaceAllUsesWith 保证下游使用点不变。
