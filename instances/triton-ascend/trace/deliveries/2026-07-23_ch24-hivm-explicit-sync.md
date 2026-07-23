# ch24 交付：HIVM 显式同步——set_flag/wait_flag 流水线同步与 Cube↔Vector 核间同步

- **Type**: delivery
- **Chapter**: ch24
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist, **Lead（Review round3 逃逸后手动接管）**
- **User present**: False
- **Tags**: triton-ascend, part-5, deep, skip_impl, hivm-hfusion, explicit-sync, set-flag, wait-flag, pipe-barrier, event-id, cross-core, sync-block, GraphSyncSolver, MemAlias, review-escape, lead-takeover

## What happened

Part 5「硬件 IR HIVM」第五站，hivm-hfusion 子系统承 ch23（HIVM 方言）。kind=deep，纯 C++ MLIR pass 章，三条 pass（`-hivm-inject-sync` / `-hivm-inject-block-sync` / `-hivm-cross-core-gss`）活在 `bishengir-opt` 里，无精简版（无 `implementation/` 目录，skip_impl）。ch23 把达芬奇内存墙写进 IR 类型系统，本章接着把**同步**也摆进 IR——因为达芬奇一个核内 6+ 条异步引擎各跑各的指令流、硬件不自动检测跨引擎数据依赖，编译器必须显式插同步。

12 机制，**两层显式同步**：

- **核内同步**（单核内跨引擎）：`set_flag`/`wait_flag` 是跨引擎一对信号（生产者 pipe 置位、消费者 pipe 等待、共享一个 event id），`pipe_barrier` 是同引擎单参串行屏障。同引擎依赖插 1 个 barrier、异引擎依赖插 1 对 flag（决策二分，§24.5）。注入 pass `AutoInjectSync` 是六道工序流水线（IRTranslator→Plan→MoveState→RemoveRedundant→EventIdAllocation→Codegen，≤1 条指令早退，§24.3），核心是 `GraphSyncSolver` 求解事件图定同步边、`EventIdSolver` 按 (生产者 pipe,消费者 pipe) 分池 + 池内生命周期重叠着色复用编号（每池上限 8，§24.7）。依赖图由 `MemAlias` 内存别名分析建（RAW/WAR/WAW，§24.4），并经**最小同步集传递归约**归约（可达即冗余不插直连、断路即补插 MTE2→MTE3 直连，§24.6），循环内不变同步外提（§24.8）。

- **跨核同步**（Cube↔Vector 两颗物理核）：`sync_block_set`/`sync_block_wait` 经 FFTS + global memory 握手——Cube 结果经 FIX 落 gm 后置位、Vector 等到再从 gm 读（Cube 侧默认等 `PIPE_FIX`、Vector 侧默认等 `PIPE_MTE3`，§24.9）。注入只对 MIX 核触发，两条融合路径 `InjectBlockSync` + `CrossCoreGSS`（§24.10）；比跨核旗更重的是**块间锁** sync block lock（块间互斥而非单向 happens-before，§24.11）。§24.12 收束成「两层同步、一套分析」。

贯穿全章两个真实 lit 夹具 worked example：`inject-sync.mlir`（核内）、`inject-block-sync.mlir`（跨核）。

**取证边界**：host 无昇腾 NPU、无预构建 `bishengir-opt`、无 CUDA，三条 pass 跑不起来。地面真值取自上游 lit 夹具的 `// CHECK:` 行（上游 CI 每次用 FileCheck 对着它校验，等价于 pass 可复现真实产出），抽取脚本 `traces/extract_fixture_checks.py`→`traces/fixture_checks.json`；同名函数去重盲区由 `traces/cross_core_manual_notes.md` 手抄补齐；所有 `trace_source=manual`，数字均标夹具锚点或 `file:Lxxx`——非编造 dump。

**交付曲折（review-escape + Lead 接管，如实）**：多维评审进到 **Review round3** 的 revise-fig 阶段时 **API 崩溃（Connection closed）**，workflow 逃逸未到 Archive。崩前 writer 已完成 #47 补写与 #22/#30/#23/#35/#18/#16/#28 多条正文核实，但**两图的 gen 脚本已改却没重渲染**（盘上是陈旧 SVG），且正文一处 fixture **张冠李戴（#30）**。Lead 手动接管：①两图**重渲染**（gen 脚本已改的），blind 因内容变更**重置 PENDING→独立盲审 PASS**（0 failure）；②正文 fixture #30 张冠李戴已纠；③writer 崩前的补写/核实由 Lead 复核在案；④chapter-map 生成→Lead 核 PASS→writer 插图引用 + 补 2 处半角标点。终审 algorithm-pedagogy / formula-structure 两维 **PASS、0 issue**。16 门全绿。verdict **APPROVED**。

**工具改进（连带）**：`lint_diagram_scaffolding` 增补抓 `fixture_checks.json` 裸名——防内部产物名泄漏进图面。

## Why it matters

ch23 把「达芬奇内存墙」写进 IR 类型系统，ch24 把「达芬奇没有硬件依赖检测」这条同样残酷的物理事实写进 IR——读者第一次看到 GPU 上被 warp scheduler/scoreboard 隐式兜住的跨引擎/跨核依赖，在昇腾上必须由编译器**显式**用 `set_flag`/`wait_flag`/`sync_block_set` 逐条插出来，而且要经依赖图（MemAlias）+ 传递归约求最小同步集、event id 分池复用有限的 8 个物理信号位。承 ch17（DAGScope/DAGSync 在 ascend-opt 侧的 scope 切分与跨核搬运）——ch17 是 Triton 下降链早段的同步雏形，本章是 HIVM 硬件 IR 层同步的**完全体**（事件图求解 + 分池着色 + 块间锁），两者是同一「异构双核必须显式同步」母题在下降链两端的现身。

对位基座《Triton 源码解读》：GPU 靠 `cp.async` + `mbarrier` + warp 硬件调度隐式重叠与同步；昇腾把这套全摊到编译器可见的 IR 里逐 op 显式物化——同一「让并行安全」母题在有/无硬件依赖检测两种硬件哲学上的对应物。

## What to remember

- **两层同步别混**：核内（跨引擎，`set_flag`/`wait_flag`/`pipe_barrier`，GraphSyncSolver+EventIdSolver）vs 跨核（Cube↔Vector 两颗物理核，`sync_block_set`/`sync_block_wait` 经 FFTS+显存，InjectBlockSync+CrossCoreGSS，只对 MIX 核）。event id 每池上限 8、按 (生产者 pipe,消费者 pipe) 分池——同号跨池是不同物理信号位。
- **决策二分口诀**：同引擎依赖→1 个 `pipe_barrier`；异引擎依赖→1 对 `set_flag`/`wait_flag`。最小同步集 = 依赖图传递归约（可达即冗余、断路即补插 MTE2→MTE3 直连）。
- **无新伏笔**：`python3 scripts/bible.py due ch24` 两清单皆空；dossier `foreshadow_due.应埋伏笔`/`应回收`均为空。本章承 ch23（已交付），未埋新伏笔、无回收动作。
- **Bible 回写**：figures **+8**（本章全部 8 图均登记，claim 取自 explainer figure_spec；chapter-map 登记为 `fig-ch24-chapter-map` 防跨章撞 id——现 148 条；吸取 ch23 归档漏登记教训，逐一核对目录名）；glossary **+11**（set_flag/wait_flag、pipe_barrier、event id、sync_block_set/wait、sync block lock、GraphSyncSolver、EventIdSolver、MemAlias、最小同步集(传递归约)、AutoInjectSync、InjectBlockSync/CrossCoreGSS，现 262 键）；concepts **+12**（现 283）；interfaces 不新增（deep+skip_impl 无精简版，同 ch20-23 先例）；arc-map 无变化。
- **skip_impl 交叉验证口径**：无精简版；靠 pin 源码（`@2badfc89e`）+ 上游 lit 夹具 `inject-sync.mlir` / `inject-block-sync.mlir` 的 `// CHECK` 行（CI FileCheck 地面真值），host 无 NPU/`bishengir-opt`/CUDA 跑不起来，标『非真机 dump』；同名函数去重盲区由手抄 notes 补齐。
- **恢复史（review-escape）**：Review round3 revise-fig 崩（API Connection closed）→Lead 接管——两图 gen 脚本已改但没重渲染（陈旧 SVG），Lead 重渲染 + 独立盲审（blind 因内容变更重置 PENDING→PASS）；正文 fixture 张冠李戴 #30 已纠；writer #47 补写 + #22/#30/#23/#35/#18/#16/#28 核实（崩前已做、Lead 复核）；chapter-map 生成+Lead 核+writer 插引+补 2 半角标点。连带修 `lint_diagram_scaffolding` 补抓 `fixture_checks.json` 裸名。
