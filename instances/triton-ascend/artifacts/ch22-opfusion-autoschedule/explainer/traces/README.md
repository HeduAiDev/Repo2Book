# ch22 traces — 手工推演记录（skip_impl，trace_source=manual）

本章是纯 C++ pass 章（bishengir submodule 内），**无精简版**，无法在 host 上跑出数值 dump：
- OpFusion / AutoSchedule 都在编译期改写 MLIR，运行它们需要构建 `bishengir-opt`（bishengir submodule + LLVM/MLIR 全量构建），host 无此工具链。
- 因此所有 worked example 的“轨迹”不是运行 dump，而是**对 pin 内 lit 夹具逐步手工推演**，且每一步结论都能对到夹具的 `// CHECK` 断言或源码常量（`file:Lxxx`）。这不是伪造 dump——lit 夹具的 FileCheck 行本身就是 pin 亲自断言的期望输出。

## 夹具与源码出处

| 用途 | 文件 | 关键锚点 |
|---|---|---|
| OpFusion 融合决策（fuseBlock）worked example | `third_party/ascend/AscendNPU-IR/bishengir/test/Dialect/HFusion/OpFusion/test_static_shallow_cv.mlir` (@testA, 7x7 静态) | FileCheck: @testA_0 恰含 {elemwise_unary, elemwise_binary, elemwise_unary, matmul, elemwise_unary} 5 op；@testA 调 @testA_0 |
| ShallowCV 二次拆分 worked example | `third_party/ascend/AscendNPU-IR/bishengir/test/Dialect/HFusion/AutoSchedule/test-shallow-cv.mlir` (@forward, 3 层 MLP) | RUN: `-hfusion-auto-schedule="block-dim=40"` |
| isFusible 分派 | `.../OpFusion/FusibleHelper.cpp:L557-L582` | switch(fusionKind_) |
| isShallowCVFusible 兼容表 | `.../OpFusion/FusibleHelper.cpp:L673-L712` | matmul↔全 vector 互融 |
| 五道关卡 | `.../OpFusion/FusibleBlockAnalyzer.cpp:L86-L147` | verifyRulesAndJoin |
| reduceRank/reduceDim 关卡守卫 | `.../OpFusion/FusibleHelper.cpp:L294-L312` | 仅 LastAxisPBR/AnyPBR 生效 |
| nodeType 关卡守卫 | `.../OpFusion/FusibleHelper.cpp` isRestrictedByNodeType | 仅 MixCV 生效 |
| 出组约束 | `.../OpFusion/FusibleBlockAnalyzer.cpp:L149-L173` | ShallowCV/MixCV 组必含 matmul |
| union-find join / find | `.../OpFusion/FusibleBlockAnalyzer.cpp:L294-L431` | 路径压缩 + union-by-size |
| applySchedule 选 scheduler | `.../AutoSchedule/AutoScheduleBase.cpp:L579-L611` | switch(fusionKind) |
| blockDim 减半 (cube:vector=1:2) | `.../AutoSchedule/AutoScheduleBase.cpp:L1221-L1231` | max(blockDim/2,1) |
| ShallowCVScheduler 三步 | `.../AutoSchedule/ShallowCVSchedule.cpp:L40-L65` | LastAxisPBR 外提→逐子核调度→TensorResultToOutParam |
