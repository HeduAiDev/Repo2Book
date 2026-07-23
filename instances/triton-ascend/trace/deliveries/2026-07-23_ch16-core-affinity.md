# ch16 交付：Cube 还是 Vector——AI Core 异构双核与核亲和定点传播

- **Type**: delivery
- **Chapter**: ch16
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T00:47:00Z
- **Agents involved**: analyst, writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-4, deep, skip_impl, core-affinity, coretype-lattice, fixpoint, ascend-opt

## What happened

Part 4「异构双核」第二站，deep+skip_impl（纯 C++ MLIR pass 章，无 .py、宿主无 CANN 编不动，deps=ch02+ch15）：《Cube 还是 Vector：AI Core 异构双核与核亲和定点传播》。上一章 AutoBlockify 定的是**执行粒度**（多少逻辑实例挤一个物理块），本章定的是**执行载体**（每个 op 落哪类单元）——同属 ascend-opt 子系统，互不相扰。

术语坑首先钉死：子系统名 `TritonAffinityOpt` 的 `Affinity` 是核亲和（core affinity），与 MLIR `affine` 方言/多面体调度毫无关系。

10 个机制全覆盖：**m1 异构双核建模**（Cube 只做矩阵乘、Vector 做逐元素/规约，两颗判核种子来自最短的两条分支）；**m2 能力 vs 放置两套枚举**（`OpAbility`/`CoreType` 位编码故意对齐，`toCoreType` 只做位重解释）；**m3 CoreType 四态格**（`operator!`/`toHivm`/`exactlyOneType` 三个格上工具函数，`operator!` 是完整位补而非仅补单核态）；**m4 canRunOn 静态判核**（scf 早返回 + 4 条 TypeSwitch 臂，`Default` 兜底做到全覆盖，非枚举穷举）；**m5 数据流图建模**（`OpNode`/`ValueNode` 二部图，`absorb` 沿消费者反向回吸）；**m6 absorbCommon 传递函数**（能力硬钉/WRITE 跟数据源/outputs 按位或三出口，WRITE 分支覆盖式返回故非纯单调 join）；**m7 isUpstreamOfCubeMem 传染**（喂 cube 的读链反向染色，taint 单调只升不降，属 Kildall 1973 单调数据流分析谱系但不满足严格单调）；**m8 getWriteDataSource**（store 核跟数据源走，跳过 i1 mask，case3「存 bool 数据」是诚实标注的边界）；**m9 diffuse 两遍不动点**（worklist 迭代 + 残留 UNDETERMINED 兜底 VECTOR_ONLY + `threshold=节点数×5` 安全阀双重终止保证）；**m10 结果落地**（`getValueTypes`/`toHivm` 交下一章 `DAGScope`/`DAGSync`）。

7 张机制图 + 本章地图共 8 图，独立盲审 2 轮：round 1 `fig-ch16-taint-propagation` 命中 1 条 failure（图只画了被染色的操作数链前半段，未呈现 claim 里「epilogue 保持 VECTOR_ONLY 未被传染」这半个对比论点），补齐 epilogue 对照分支后 round 2 PASS(0 failure)；map 站 1 轮 PASS。write↔review 3 轮收敛，`lint_trace_consistency` 全绿零漂移。

## Why it matters

ch16 是全书第二次系统展示「数据流分析框架」的具体应用（第一次是 ch11 PtrAnalysis 的正向推导），但方向相反——本章是**后向**（从消费者回吸约束到生产者）+ **worklist 驱动到不动点**，且诚实标注了它与 Kildall 1973 经典单调框架的偏离点（WRITE 分支覆盖式返回破坏严格单调，靠 `threshold` 硬上限兜底而非理论证明）。这是一处很有教学价值的「像但不完全像教科书」案例，读者能借此建立起「不是所有 worklist 迭代都严格单调，但仍可能终止」这一更细致的心智模型。

## What to remember

- **本章心脏**：`absorbCommon` 三出口 + `isUpstreamOfCubeMem` 反向染色——一个 `PREFER_VECTOR` 的 op（如 `load`）若下游是 cube 或已被传染，会被拉成 `CUBE_ONLY`；染色单调不退，传播必收敛。
- **两道终止防线务必分清**：① 核在四元格上按位或只升，每节点至多升两步（不是三步——评审曾抓到一处手误，已订正）；② `diffuse` 的 `threshold=节点数×5` 是独立于①的硬上限安全阀，专门应对 WRITE 分支破坏单调性时的病态输入。
- **评审结论**：4 维评审 APPROVED，0 blocking 遗留（初始评审曾有 2 条 blocking：m3 对 `operator!` 的行为描述有算术错误「对顶/底不作用」实为完整位补；m5/m9 依赖的 `const`/`return` 两节点未在开篇 kernel 示例中交代来源——均已由 writer 定点修复，chapter.md 现已订正）；其余 non-blocking（1 处行号笔误 L379→L381、「至多三步」应为「两步」、m7 不变量论证漏引第三处赋值、m8/m9 缺独立「源码」过渡段风格小不一致、m10 无三段模板、`%bias` 参数来源/`OpNode`/`worklist` 三处前向引用未加括注）均已定点修复。
- **Bible 回写**：glossary 新增 9 条（`TritonAffinityOpt`/`OpAbility`/`CoreType`/`CoreType` 四态格/`canRunOn`/`absorbCommon`/`isUpstreamOfCubeMem`/`diffuse`/`toHivm`）；concepts 新增 9 条（对应 m1-m10 核心机制摘要）；figures 新增 8 条（7 机制图 + chapter-map）；interfaces **不新增**（skip_impl 无精简版）；arc-map **新埋伏笔 f2**（本章只求解 `Value→CoreType` 标注不落 IR，下一章 `DAGScope`/`DAGSync` 按标注切 scope、插同步搬运——plant=ch16, payoff=ch17, status=open）；`bible.py due ch16` 无应埋/应回收项，f2 为本章新立。
- 诚实边界：host 无 CANN 工具链，交叉验证走 pin 精确源码 `@2badfc89e`（`DAG.cpp`/`DAG.h` ~534 行）逐行手算，triton-ascend 树内无该 pass 的编译器测试夹具，不伪造编译器 dump，每处数字均标 `文件:Lxxx` 供对眼。对位基座 ch27/ch28（GPU 选 mma 指令 vs 昇腾放 op 到核，同为矩阵乘硬件强相关但抽象层不同）。下一站：ch17《把双核落到 IR：Scope 切分与 cube↔vector 同步搬运》，回答核标注怎么落进 IR。
