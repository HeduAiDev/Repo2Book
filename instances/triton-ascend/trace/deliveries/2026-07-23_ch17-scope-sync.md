# ch17 交付：把双核落到 IR——Scope 切分与 cube↔vector 同步搬运

- **Type**: delivery
- **Chapter**: ch17
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T00:00:00Z
- **Agents involved**: analyst, writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-4, deep, skip_impl, scope, sync, dagsync, dagscope, ascend-opt

## What happened

Part 4「异构双核」第三站，deep+skip_impl（纯 C++ MLIR pass 章，DAGScope.cpp~1139 行 + DAGSync.cpp~1333 行合计 2472 行，无 .py、宿主无 CANN 编不动，deps=ch16）：《把双核落到 IR：Scope 切分与 cube↔vector 同步搬运》。ch16 算出的是**每个 op 想在哪个核**（数据流不动点），本章把这个结果**物化到 IR**——不重新推导核标注，只消费。

两 pass 分工与先后：`DAGSyncPass`（IR 标识 `dag-sync`）先跑——在还没切 scope 的扁平 IR 上，`LegalizeDot` 把带非零累加器的 `dot` 拆成 `dot(零累加)+arith.addf`（制造干净的 cube→vector 边）→ 主 walk 逐条跨核数据边判 `needVectorCubeSync`（仅 `CUBE_ONLY↔VECTOR_ONLY` 触发）→ 插 `sync_block_set/wait` + 数据搬运（CUBE→VECTOR 走 `fixpipe(NZ2ND)→UB→to_tensor`；VECTOR→CUBE 走 `to_memref→CBUF(L1) nz alloc→copy→convert_layout(ND)`，多一层 32 字节对齐 nz 重排）→ `processScfForSync` 补循环迭代参数的跨核依赖 → `addMemEffectsSync` 靠别名分析补 SSA 看不见的 WAR/WAW 同步。`DAGScopePass`（IR 标识 `dag-scope`）后跑——`encapsulateWithScope` 建两个 `scope.scope`（`aivScope` 挂 VECTOR、`aicScope` 挂 CUBE 属性）→ `collectOpsToMove` 给每个 op 定路由（`copy`→aiv、`fixpipe`→cube、`scf`/`scope` 结构 op→两边都要且按 `MoveType` 过滤重建）→ `SplitScope` 两遍分发、逆序 erase 原 op → `addSyncOpsForBufferWait` 对 `fixpipe`/`to_memref` 补 pipeline 级 buffer-wait set/wait。

16 个机制全覆盖（m1 两 pass 分工先后 / m2 跨核为什么必须同步 / m3 LegalizeDot / m4 主遍历去重触发 / m5 CUBE→VECTOR 搬运+同步 / m6 VECTOR→CUBE 搬运+同步（多一层 CBUF 对齐）/ m7 32B 对齐 nz 布局量化 / m8 scf.for 迭代参数跨核 / m9 set/wait 落点微调 FindEarliest/LastestPosition / m10 addMemEffectsSync 别名分析补同步 / m11 encapsulateWithScope 建两 scope / m12 collectOpsToMove 算子路由 / m13 SplitScope 裁剪重建 / m14 addSyncOpsForBufferWait 缓冲就绪握手 / m15 flag 事件旗池与死锁 / m16 对位基座 scope+事件 vs warp+mbarrier）。交叉验证走 ① pin `@2badfc89e` 精确源码 ② bishengir 子模块真实 lit 夹具（`test/Dialect/Scope/ops.mlir`、`test/Dialect/HIVM/IR/sync-ops.mlir`）展示 `scope.scope`/`sync_block_set`/`sync_block_wait` 的真实 IR 文本形态；dag-sync/dag-scope 两 pass 本身无 lit 夹具，前后 IR dump 均如实标注为「示意，非真实 dump」，不伪造。

10 张机制图（m1/m2/m5/m6/m7/m11/m12/m13/m14/m16）+ 本章地图共 11 图。write↔review 1 轮收敛，独立盲审 1 轮 0 failure，chapter-map 盲审 1 轮 PASS。

## Why it matters

ch17 是 ch16 数据流分析结果的「落地」环节，也是全书第一次系统展示「同一份分析结果如何被拆成两个先后 pass 分别消费」——`DAGSync` 先在扁平 IR 上用它判同步、`DAGScope` 后用它切 scope，二者共享同一份 `AffinityDAG::Graph`（经 `registerGraph`/`GraphManager::getGraph` 传递，不重算）。这也是本书里离 GPU 侧 warp specialization 概念最近的一次对照点：昇腾用「两颗异构物理核 + scope 容器 + block 级事件」，GPU 用「同一 SM 内 warp 分组 + mbarrier」，读者能借这组对照建立起「同一个『生产者/消费者协作』问题在不同硬件抽象层级上长成什么样」的心智模型。

## What to remember

- **本章心脏**：`needVectorCubeSync`（何时插同步）+ 一对搬运链（`insertCubeToVectorDataMovement`/`insertVectorToCubeDataMovement`）+ `encapsulateWithScope`/`SplitScope`（scope 怎么建、怎么切）——同步管时序、搬运管位置/格式、scope 切分把两者都归位到各自的容器里。
- **评审结论**：APPROVED，0 blocking，13 条 non-blocking：1 条 fidelity（`insertCubeToVectorDataMovement` 代码块把 `hivm::FixpipeOp` 尾部实参的省略注释放在 `dma_mode` 之前，字面读来像 `dma_mode` 是最后一个参数，实际其后还有 5 个被省略的可选属性——按项目惯例应显式声明重排或恢复原序）+ 2 条 algorithm-pedagogy（m9/m6 缺显式「不变量」收尾段，素材已备好只是正文未采用）+ 1 条格式一致性（m5/m6/m16 三节直觉段缺「**直觉**。」标签）+ 1 条图文行号落差（fig-m1-pass-order）+ 1 条配图缺口建议（m15 死锁机制无图，仅供参考）+ 7 条 reader-comprehension（`PIPE_MTE2`/`PIPE_S` 突然出现无旁注——查证这两个 pipe 值实际已在 ch07 的 `CORE/PIPE/MODE` 词条里登记过全部 8 档，故不需要新开 glossary 词条，只是正文局部缺一句提示；第四节「见第十一节」应为「见第十四节」的错误跨节指引；`convert_layout(ND)` 与「cube 需要 nz」表面矛盾未消歧；`aic/aiv` 与 `CUBE/VECTOR` 两套命名对应关系已在 ch02 `mix_mode` 词条立过，正文可以但非必须再点一句；`WAR/WAW/RAW` 先用后定义；`fig-m13-split-rebuild` 图上 yield 值数量与迭代参数数量不对应）。均记入 review-report.json，留待后续小修窗口处理，不阻断本次交付。
- **Bible 回写**：glossary 新增 9 条（`DAGSyncPass/DAGScopePass`、`LegalizeDot`、`needVectorCubeSync`、`hivm::FixpipeOp`(编译器 pass 侧构造，区别于 ch05 `al.fixpipe` 语言前端)、`hivm::CopyOp/ConvertLayoutOp`、`CBUF`、`aivScope/aicScope` 命名、flag 事件旗池与 `syncFlag%14`、对位基座 scope+事件 vs warp+mbarrier）；concepts 新增 11 条（对应 m1-m14 核心机制摘要）；interfaces **不新增**（skip_impl 无精简版）；arc-map **f2 回收**（ch16→ch17：核标注怎么落进 IR，status 改 resolved）+ **新埋伏笔 f3**（ch17→ch18：切好的 scope 里跨核 buffer 只是单块分配，真机要榨干双核并行度需要 UB 多缓冲——管线第三趟 `add_dag_ssbuffer`/`DAGSSBuffer`，下一章开讲）。
- 诚实边界：host 无 CANN 工具链，交叉验证走 pin 精确源码 `@2badfc89e` 逐行手算 + bishengir 真实 lit 夹具展示 IR 文本语法，dag-sync/dag-scope 两 pass 本身无 lit 夹具、不伪造前后 dump。对位基座《Triton 源码解读》ch31-prefetch-warp-specialization（cube/vector 分域+scope/事件 ≈ producer/consumer warp+mbarrier，但跨的是物理核而非同 SM 内 warp）。下一站：ch18《DAGSSBuffer：UB 多缓冲与昇腾的软件流水线》。
