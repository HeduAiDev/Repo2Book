# 手工推演：ShallowCVScheduler 在 @forward（3 层 MLP）上的二次拆分

源：`third_party/ascend/AscendNPU-IR/bishengir/test/Dialect/HFusion/AutoSchedule/test-shallow-cv.mlir` @forward
RUN 行：`bishengir-opt %s -hfusion-auto-schedule="block-dim=40" -split-input-file`
func 属性：`hfusion.fusion_kind = #hfusion.fusion_kind<SHALLOW_CV>`（已是一个 ShallowCV 融合核）

## 夹具 IR 的算子清单（可直接数，L7-L28）
- matmul_transpose_b（cube/kMatmul）：%1, %8, %15 → **3 个 cube 段**
- broadcast（vector）：%broadcasted, %broadcasted_6, %broadcasted_7 → 3
- elemwise_binary add（vector，bias 加）：%4, %11, %18 → 3
- elemwise_binary max_signed（vector，relu）：%6, %13 → 2（末层无 relu）

结构：`mm1 →[bcast+add+max] mm2 →[bcast+add+max] mm3 →[bcast+add]`
三条 **vector 链**（层间/末层各一条），三个 **cube 段**（matmul）。

## applySchedule 侧的 blockDim（AutoScheduleBase.cpp:L1221-L1231）
tryGetFusionKind(func)=ShallowCV ∈ {MixCV, SingleCube, ShallowCV} →
`options.blockDim = max(blockDim/2, 1) = max(40/2,1) = 20`（cube:vector=1:2，cube 侧核数取半）。
输入 blockDim=40 来自 RUN 行；结果 20 = 40/2。

## ShallowCVScheduler::runOnOperation 三步（ShallowCVSchedule.cpp:L40-L65）
- **Step 1** applyOpFusionOutline(shallowCVFunc, {fusionMode=LastAxisPBR, alwaysInline=true, moveOutToParam=false})：
  对 ShallowCV 核**再跑一遍 LastAxisPBR 融合**，把纯 vector 段外提成独立 device 子核。
  三条 vector 链 → 各成一个 LastAxisPBR 子核（broadcast+add[+max] 融进一个核）。cube 段（matmul）留在原核。
- **Step 2** 逐外提子核 applySchedule(funcOp)：子核带 LastAxisPBR 标 →
  applySchedule switch（AutoScheduleBase.cpp:L579-L611）里 LastAxisPBR ∈ PBR 家族 → **AnyPBRScheduler** 真正切 tile。
  cube（matmul）段另由 cube 路径处理。
- **Step 3** applyTensorResultToOutParamsPass(shallowCVFunc)：对原 ShallowCV 核做结果转出参。

**cube/vector 分工的落点**：一个 ShallowCV 核被拆成「cube 子核（matmul）+ 若干 vector 子核（LastAxisPBR）」，
各归各的 scheduler；这就是 “shallow（浅）配合” 在 pass 层的具体形态。

注：Step1 精确外提出的子核个数需实跑 bishengir-opt 才能逐一点名（host 无工具链）；
本推演只断言 IR 里可数的结构量（3 cube 段 / 3 vector 链）与 blockDim=20，均有 file:Lxxx / RUN 行出处。
