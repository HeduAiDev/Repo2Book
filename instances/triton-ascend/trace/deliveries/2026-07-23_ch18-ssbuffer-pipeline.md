# ch18 交付：DAGSSBuffer——UB 多缓冲与昇腾的软件流水线

- **Type**: delivery
- **Chapter**: ch18
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T00:00:00Z
- **Agents involved**: analyst, writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-4, deep, skip_impl, dagssbuffer, double-buffer, software-pipelining, ub, ascend-opt

## What happened

Part 4「异构双核」第四站，deep+skip_impl（纯 C++ MLIR pass 章，`DAGSSBuffer.cpp` ~5534 行——本书至今最大单文件，无 .py、宿主无 CANN 编不动，deps=ch17）：《DAGSSBuffer：UB 多缓冲与昇腾的软件流水线》。ch16 判核、ch17 把核决策物化成 scope+跨核同步，本章是**另一条正交优化**——在**单核循环内部**做 UB 多缓冲流水，重叠访存与计算，与跨核同步无关。

`DAGSSBuffer`（方言 pass 名 `dag-ssbuf`）挂在自动调度链第三站（先 `dag-sync`/`dag-scope`，中间夹 cse/canonicalizer，再 `dag-ssbuffer`），只处理 `isCube` 为假的 vector scope。`runOnOperation` 五段编排里前四段（`AddIfCondition`/`FlowSssbuf`/`ControlSsbufV2`/`ChangeAdvanceOpForm`）本章点到为止，第五段 `WalkAIVNestedForAndProcess` 才是双缓冲变换主体，按三步驱动：① `addDoubleBuffForArgs` 把要多缓冲的 buffer 在 `scf.for` 的 `iter_args` 里扩成 N 份（`bufferNum` 写死为 2）+ 2 个计数器 `frontCnt`/`postCnt`，每个 buffer 净增 `(N-1)+2=N+1` 个 iter_args；② `buildNBufferProducer`/`buildNBufferConsumer` 按 `cnt % N` 选写/读哪份 buffer，N=2 时全退化成 `arith.select`、零 `scf.if`（producer 因要同时更新两份 buffer 故 2 条 select、consumer 只需取一份故 1 条）；③ `addMultiBuffCaculate` 把 producer/consumer 接到已有的「搬运 if / 计算 if」上，回填 `yield` 让缓冲轮转——其中 buffer0 来自「搬运 if」的 else-yield（历史包袱：变换前那份原始 buffer）、buffer1..N-1 直接来自新增 iter_args。核心不变量：`frontCnt` 恒领先 `postCnt` 一位（奇偶相反）⇒ 写侧与读侧永远落在不同 buffer ⇒ DMA 搬运与 Vector 计算可以并行；退回单计数器则写读同块、必须串行。一个 4-tile 演示例（`T_load=3`、`T_compute=2`）从单缓冲 20 个单位降到双缓冲 14 个单位（约 1.43×，渐近上界 1.67×）。收尾对位基座 GPU：昇腾无 `cp.async`+硬件异步拷贝，必须在 IR 里显式物化同一件事（`num_stages` 隐式声明 vs `DAGSSBuffer` 显式复制 buffer+手写计数器）。

3 张机制图（m1-iterarg-expand/m2-counter-mod-select/m3-load-compute-overlap）+ 本章地图共 4 图。write↔review 1 轮收敛，独立盲审 1 轮 0 failure，chapter-map 盲审 1 轮 PASS。verdict=**APPROVED**，0 blocking / 5 non-blocking。

## Why it matters

ch18 补上了 ch17 结尾留的尾巴——「切好的两个 scope 里跨核 buffer 只是单块分配」——但同时把边界钉死：本章的重叠是**单核循环内部**的访存↔计算重叠，与跨核并行正交，不能和「cube 算第 N 块、vector 算第 N-1 块」那类跨核流水混为一谈（此点在小结明确口径校准，也是 f3 回收时需要留意的措辞）。它也是全书第三次系统展示「同一个软件流水线目标，GPU 隐式（`num_stages`+`cp.async`）vs 昇腾显式（`DAGSSBuffer` 手写 buffer 复制+计数器）」这组对照——把 GPU 侧一条指令背后悄悄做的事情，在昇腾侧摊开成看得见的 IR 变换。

## What to remember

- **本章心脏**：`addDoubleBuffForArgs`（buffer 扩容+计数器新增）+ `buildNBufferProducer`/`buildNBufferConsumer`（cnt%N 选 buffer，N=2 免 scf.if）+ `addMultiBuffCaculate`（接线回填轮转）——三步对应 iter_args 扩容、写读分叉、时序错位三层，`frontCnt` 领先 `postCnt` 一位是贯穿全章的核心不变量。
- **评审结论**：APPROVED，0 blocking，5 条 non-blocking：1 条 algorithm-pedagogy（m1-m3 三个 core 机制直觉/数值表/invariant/量化四层齐备，仅建议 §五 trace 表前加一句回指 §二 的量化数字）+ 1 条图注质量（`fig-ch18-m2-counter-mod-select` 图注只描述画面未落结论，建议补一句「producer 多出的 1 条 select 正是同时更新两份 buffer 的代价」）+ 3 条 reader-comprehension（「一条 `arith.select`」在直觉段/图 alt/小结与「机制」段的精确计数 producer 2 条、consumer 1 条前后不一致，需统一措辞；驱动循环代码里的 `level`/`maxLevels` 原样保留却未解释；`addMultiBuffCaculate` 里 buffer0 取自 else-yield、buffer1..N-1 取自新增 iter_args 的来源不对称未加说明）。均记入 review-report.json，留待后续小修窗口处理，不阻断本次交付。
- **Bible 回写**：glossary 新增 7 条（`DAGSSBuffer`(dag-ssbuf)、`AIV` 缩写、`iter_args`、`frontCnt`/`postCnt`、`bufferNum`、软件流水线（显式 vs 隐式）、`num_stages`/`cp.async` 对照项）；concepts 新增 6 条（对应 addDoubleBuffForArgs/producer-consumer-select/frontCnt-postCnt 不变量/addMultiBuffCaculate 接线/GPU-昇腾对照）；figures 新增 4 条（m1/m2/m3 + chapter-map，均 blind_review=PASS）；interfaces **不新增**（skip_impl 无精简版）；arc-map **f3 回收**（ch17→ch18：UB 多缓冲，status 改 resolved）+ **新埋伏笔 f4**（ch18→ch19：本章双缓冲/软流水只覆盖结构化的规整 load-compute 循环，真实 kernel 里离散掩码/交错步长的不规则访存怎么处理留给下一章）。
- **旁记（非本章范围）**：核对时发现 Bible `figures.json` 里 ch17 一条记录都没有（该章 10 张机制图 + chapter-map 疑似归档时漏登记），已如实记录、未在本次任务中回填——留给下次接触 ch17 或批次体检时处理。
- 诚实边界：host 无 CANN 工具链，本章无专用 lit 夹具，交叉验证仅走 pin 精确源码 `@2badfc89e` 逐行手算 + 一段忠实重写 producer/consumer 逻辑的 Python 交叉核对；正文 IR 片段按变换前后语义手工构造最小示例，标注「非编译器 dump」，不伪造真实 dump。对位基座《Triton 源码解读》`num_stages`/软件流水线相关章节。下一站：ch19《离散掩码拆分与交错优化》。
