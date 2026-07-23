# ch19 交付：不规则访存的驯服——离散掩码拆分与交错访存优化

- **Type**: delivery
- **Chapter**: ch19
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T00:00:00Z
- **Agents involved**: writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-4, deep, skip_impl, discrete-mask, interleave, mask-analysis, ascend-opt

## What happened

Part 4「异构双核」第五站，deep+skip_impl（纯 C++ MLIR pass 章，无精简版，deps=ch13）：《不规则访存的驯服：离散掩码拆分与交错访存优化》。dossier 与 dossier-verify 均已在此前完成（skip_dossier=true），本轮从 Explain 站开工——上一轮 Illustrate 站因 API 错误 Connection closed 中断（非内容问题），本轮先清掉半成品图再重跑。

本章讲结构化下降链遇到不规则访存时的两条驯服路径，机制上彼此独立但同属一个主题。**离散掩码**：总闸 `isDiscreteMask`（`DiscreteMaskAccessConversionPass.cpp:L59-L72`）复用 [ch13 建立的](../../artifacts/ch13-maskanalysis-extractslice/narrative/chapter.md) `MaskState::parse` 判连续 vs 离散（判据零重复，parse 成功走结构化 DMA、失败本 pass 接管）；`collectAndLeaves` 把 `andi` 掩码树递归拍平，遇 `broadcast(andi)` 用分配律下推；`decomposeAndMask` 把混合掩码拆成 `contMask`（收窄安全范围防越界）+ `discMask`（逐元素选择）；三条改写各有招——`DiscreteMaskLoadConversion` 安全全载 + select 屏蔽、`DiscreteMaskStoreConversion` 读-改-写 + `sync_block_lock` 加解锁、`DiscreteMaskAtomicConversion` 按 RMW 类型选幺元填充；离散写的代价定量为一次逻辑散点写付两趟全量 DMA 带宽。改写完成的 Store/Atomic 打上 `DiscreteMask` 属性——**这正是 [ch14](../../artifacts/ch14-unstructure-fallback/narrative/chapter.md)（`UnstructureConversionPass`）与 ch17（`UseAnalysis`）消费的那个跨章标签，本章是打上处**。第二块**交错步长**（相对独立）：`expandInterleaveMemRefType` 把末维 stride=2 视图翻倍成连续 2N 段供一次 DMA 搬完；`IndexMode::EVEN_MODE`/`ODD_MODE` 靠 offset 的「+1」判偶奇（互斥且完备）；`DeinterleaveStatusOptimization`（load 侧）用 `extract_slice(stride 2)` 隔一取一拆偶奇，`InterleaveStatusOptimization`（store 侧）是它的逆运算，两条独立算出的偶/奇 `materialize` 结果经 `insert_slice` 交织回一次落盘。

8 张图（fig-m1-gate/fig-m2-flatten/fig-m3-split/fig-m5-rmw/fig-m9-expand/fig-m11-deinterleave/fig-m12-interleave + 本章地图），全部自检 6 项通过 + 独立盲审 PASS。write↔review 2 轮，独立盲审 1 轮 0 failure，chapter-map 盲审 1 轮 PASS。verdict=**APPROVED**，0 blocking / 5 non-blocking。

## Why it matters

ch19 是「结构化下降链遇到不规则访存怎么办」这条线的收官两问之一：连续掩码/规整步长早在 ch13（MaskAnalysis）建立了处理路径，本章补上「parse 失败之后怎么办」和「stride≠1 怎么办」这两个此前一直悬着的问题，并且显式接上了此前两章（ch14/ch17）提前消费、但一直没交代来源的 `DiscreteMask` 属性——读者读完本章会第一次看到这枚标签「是谁打上去的」。它同时是全书交叉验证纪律最严的一章之一：无精简版，全靠 pin 精确源码逐段核对 + 2 处真实 lit 夹具（`loadstore.mlir`/`atomic.mlir`）逐字比对，不伪造任何编译器 dump。

## What to remember

- **本章心脏**：`isDiscreteMask`（总闸，判据=parse 成败）+ `decomposeAndMask`（contMask 护栏/discMask 选择的拆分）+ 三条改写（Load 全载+select / Store 读改写+锁 / Atomic 幺元）是离散掩码一半的支点；`expandInterleaveMemRefType`（末维翻倍换切法）+ `IndexMode::EVEN_MODE/ODD_MODE`（偶奇判定）+ `DeinterleaveStatusOptimization`/`InterleaveStatusOptimization`（一对互逆搬运）是交错步长一半的支点。
- **评审结论**：APPROVED，0 blocking，5 条 non-blocking，全在 reader-comprehension/pedagogy 层面：① m1 总闸缺一段呼应 m10（EVEN⊕ODD=1）写法的显式「不变量」论证（三分支互斥穷尽在代码结构上自明，非正确性缺陷）；② 开篇选读指引把「总闸」一词错误地连带指向 §二（§二讲的是 andi 树拍平，与「总闸」及「DMA 效率」无关，真正讲透代价的是 §一+§四/§五）；③ §七 `UseAnalysis`/「结构化偏移」首现只有半句话、无最小上下文；④ §十提前使用了 §十二才第一次解释的术语「materialize」；⑤ §十一「2N 能被 2 整除」是恒真命题，未真正解释「末维为偶数」这个前置条件在防什么。均记入 review-report.json，留待后续小修窗口处理，不阻断本次交付。
- **Bible 回写**：glossary 新增 7 条（`isDiscreteMask`、`contMask`/`discMask`、`hivm::SyncBlockLockOp`/`SyncBlockUnlockOp`、`IndexMode::EVEN_MODE`/`ODD_MODE`、`DeinterleaveStatusOptimization`/`InterleaveStatusOptimization`、`expandInterleaveMemRefType`、`UseAnalysis`（最小定义/前瞻，详解留后续 TritonToLinalg 章节））；concepts 新增 11 条（对应本章 13 个机制里的 11 个核心/支撑机制）；figures 新增 8 条（7 机制图 + chapter-map，均 blind_review=PASS）；interfaces **不新增**（skip_impl 无精简版）；arc-map **f4 回收**（ch18→ch19：不规则访存兑现，status 改 resolved）；本章**未埋新伏笔**（dossier.foreshadow_due.due_to_plant 为空）。`DiscreteMask`/`is_discrete_mask` 属性词条本身此前已在 ch14 首现登记（glossary 里标 `NEW(ch14)`），本章只是其打上处，未重复登记、也未改写既有词条。
- 诚实边界：host 无 CANN 工具链，本章无可运行精简版，交叉验证仅走 pin 精确源码 `@2badfc89e` 逐段核对 + 2 处真实 lit 夹具（`unittest/Conversion/General/DiscreteMaskAccess/{loadstore,atomic}.mlir`）逐字比对（含 initMap 幺元全表与夹具 `dense<>` 常量逐一对上），不伪造任何运行时 dump。对位基座《Triton 源码解读》里 Coalesce/AxisInfo 相关章节——基座靠 `AxisInfo` 分析指针连续性决定能否向量化，triton-ascend 换到 MLIR 层用 `MaskState` 连续性做同一件事，载体不同、判据同源。下一站：ch20《TritonAscend 方言与 NPU 专属语义注入》。
