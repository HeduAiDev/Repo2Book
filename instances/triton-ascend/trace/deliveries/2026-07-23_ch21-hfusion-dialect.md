# HFusion 方言：Linalg 之上的张量级融合 IR 与算子上抬（deep+skip_impl）

- **Type**: delivery
- **Chapter**: ch21
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T14:49:55Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-5, deep, skip_impl, hivm-hfusion, hfusion-dialect, linalg-superset, fusion-kind, uplift-pass

## What happened

Part 5「硬件 IR HIVM」第二站，hivm-hfusion 子系统承 ch20：《HFusion 方言：Linalg 之上的张量级融合 IR 与算子上抬》，deep+skip_impl（纯 `.td` 方言 + C++ pass 章，AscendNPU-IR/bishengir submodule 内，无精简版；上一轮 Dossier 站曾崩于 API 错误 Connection closed、0 产物，本轮干净重发）。

**方言身份**：`HFusionBase.td:L31` `let name = "hfusion"` → 全方言 op 一律打印 `hfusion.<助记符>`；`dependentDialects` 里 `linalg::LinalgDialect` 是『建在 Linalg 之上』第一处硬证据。**词汇表**：函数化 elementwise——`elemwise_unary`/`elemwise_binary` 各携一个枚举属性 `fun`（`UnaryFn` 18 例/`BinaryFn` 18 例/`CompareFn` 10 例/`TernaryFn` 1 例/`TypeFn` 3 例）参数化整族函数，而非每函数一个 op。**结构化基类**：`HFusionStructuredBase_Op` 直接实现上游 `LinalgStructuredInterface`+`DestinationStyleOpInterface`+`ReifyRankedShapedTypeOpInterface`，是『HFusion 是 Linalg 超集』的代码层硬证据；`hfusion.gather` 三重循环等价语义 + gather 轴不可 tile 判据是本节数值推演重点。**专属 op 目录**：19 个非结构化 op（`HFusionOps.td`），深挖 `atomic_rmw`（AtomicKind 11 值参数化）与 `matmul_mx`（微缩放矩阵乘，MX/FP8/FP4）两例。HFusion 自身实定义 op 逐口径数实 = 19+5+9 = **33** 个（继承 Linalg 部分文档不给计数，正文守住口径边界）。**FusionKind**：10 种融合意图枚举（`PureElemwise=1` 起无 0，到 `Unknown=10`），func 级属性，由 `InferFuncFusionKind` 推断、驱动 `AutoSchedule` 分派——本章只讲『是什么/谁产/谁消费』，十种调度差异留给下一章 ch22。**LinalgToHFusion 上抬**：4 个 `OpRewritePattern` + `applyPartialConversion`，`linalg.map`/`linalg.generic` 全 illegal、`linalg.reduce` 仅带 `reduce_mode` 才 illegal；elementwise 上抬边界——NPU 扩展词汇（relu/rsqrt/tan/tanh/atan/ilogb/log1p/ldexp/powf/powi…）上抬 hfusion，Linalg 原生词汇（abs/exp/log/div）留 linalg。5 图（m2/m3/m4/m6/m7 + chapter-map）全 blind PASS，16 门禁全绿。

**评审**：APPROVED，8 条 issue 全 non-blocking（1 条 fidelity 层 must_keep 覆盖度——sort/histogram 裸列名字无语义说明；1 条 formulas 咨询性密度告警；1 条 anchors 跨书链接正则盲区误判；5 条 reader-comprehension——`hasIndexSemantics` 未解读、`RoundModeAttr` 悬空未呼应、FusionKind 十值修饰词区分度不足、`MemoryEffectsOpInterface` 两处跳过未交代、正文 `maxf` vs 配图 `maxnumf` 拼写不一致）。write↔review 2 轮收敛，blind 1 轮 PASS，map 1 轮 PASS。

## Why it matters

ch21 是全书讲清 HFusion 方言本身的权威章——此前 ch20 只把 `tt.histogram`/`ascend.mod` 等 op「送去了 hfusion」当作逃生舱去向提了一句，本章把这个方言摊开讲：它凭什么算 Linalg 超集（接口继承的代码证据）、扩展了哪些词汇（函数化枚举）、补了哪些 Linalg 没有的专属算子、以及它独有的融合意图（FusionKind）机制。同时把 ch12《AtomicRMWConverter：硬件原子算子》埋下的伏笔在这里坐实——当时说的『硬件原子算子』正是这里的 `hfusion::AtomicRMWOp`/`AtomicXchgOp`。为下一章 ch22（OpFusion/AutoSchedule 深挖 FusionKind 驱动的调度决策）铺好全部词汇基础。

## What to remember

- **IR 名权威延续**：`hfusion.<助记符>` 严格来自 `HFusionBase.td` 的 `let name` + 各 `def XxxOp` 助记符，不从 C++ 类名倒推（承 ch20 命门）。`UnaryFn`/`BinaryFn`/`FusionKind` 是枚举/属性，不是 op——分清界线是本章反复强调的纪律。
- **计数口径钉死**：HFusion 自身实定义 op = 33（19+5+9），继承 Linalg 全部算子**无计数**（`architecture.md` 只有一句自述，不给数字）——正文严禁宣称『HFusion 共 N 个算子』这种混淆两个口径的说法。枚举计数逐条数实（UnaryFn18/BinaryFn18/CompareFn10/TernaryFn1/TypeFn3/AtomicKind11/FusionKind10 等），不靠 brief。
- **新埋伏笔 f5**（ch21→ch22）：本章只讲 FusionKind 是什么、谁产谁消费，十种 kind 的调度差异（Cube/Vector 分工、tile 策略）留给 ch22（OpFusion + AutoSchedule）。
- **Bible 回写**：glossary +6 词条（FusionKind / 函数化 elementwise / HFusionStructuredBase_Op / hfusion.gather / matmul_mx / LinalgToHFusion）+ 补强既有『HFusion（hfusion）』词条的 op 计数权威（现 239 键）；concepts +6（现 252）；figures +6（5 机制图 + chapter-map，现 128，均 blind PASS）；interfaces 不新增（skip_impl 无精简版）；arc-map 新埋 f5（ch21→ch22）。
- **诚实边界**：host 无 `bishengir-opt` 昇腾编译器，跑不了 LinalgToHFusion pass 本身；『上抬后长这样』的 IR 对照取自项目自带 lit 夹具 `test/Conversion/LinalgToHFusion/{linalg-to-hfusion.mlir,arange.mlir}` 的 `// CHECK` 期望（pass 合约输出，非真机 emit）；gather 数值表按 `.td` 三重循环语义纯 host Python 复算坐实。交叉验证走 pin 精确源码 + 真实 lit 夹具，不伪造编译器 dump。
