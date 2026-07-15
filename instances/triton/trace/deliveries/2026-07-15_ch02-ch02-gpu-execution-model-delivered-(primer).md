# ch02 gpu-execution-model delivered (primer)

- **Type**: delivery
- **Chapter**: 02
- **Date**: 2026-07-15
- **Timestamp**: 2026-07-15T17:05:00Z
- **Agents involved**: analyst, implementer, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch02, primer, occupancy, coalescing, register-spill, epiphany-figure, paper-grounding

## What happened

第二章《GPU 执行模型》primer 交付。kind=primer(论文忠实小型参考实现,非 subtract-only,门禁为 lint_paper_grounding)。锚点复用 ch01 已见的 tutorial 01-vector-add,给读者三把贯穿全书的性能判据尺:①occupancy(活跃 warp/上限,受寄存器与共享内存双闸压制,取两者较小者)②合并访存 coalescing(相邻 lane 访问相邻地址才能合并成一次事务,T(A)=|{floor(a/128)}| 公式化论证)③寄存器溢出 register spill(spill=max(0,need-budget)单调非降,访存代价跳变)。开篇顿悟图把 grid->block(CTA)->warp->lane 层次 + 内存延迟金字塔(reg~1/SMEM~20-30/L2~200/HBM~400-800 cycle)+ coalescing 直觉一图打通。9 个机制(m01-m09)分 core/supporting 两档,6 张图经 illustrator 自查+2 轮盲审(round1 抓出 fig-m03-latency-pyramid 的三处编造统一'~10x'倍数标注并勒令改为按层真实倍数或直接删除,round2 通过)。本章地图 1 轮 PASS。write-review 2 轮收口,reviewer 多维度评审(algorithm-pedagogy/reader-comprehension/derivation-audit)汇总后 APPROVED,共 6 条 negotiable/non-blocking issue(m03 缺显式单调性收束句、公式密度建议、SM 全称首现晚于使用、顿悟图'线程'与正文'lane'措辞不统一、'尾块'专名误用于泛指、37.5%/38%精确度可更精确),均判定不阻断交付。论文根基:GPU 执行模型本体无学术论文(厂商 PTX/CUDA 编程指南),tile 立论引 Triton MAPL 2019(仅引题名/立论)。

## Why it matters

本章是全书性能判据的地基——后续布局/访存合并/共享内存分配/后端占用率等章节的性能决策都要拿 occupancy/coalescing/register-spill 这套模型来量;顿悟图method首次在本书validate:头图设计过的数学表达+真实延迟数量级,不是源码走读。

## What to remember

ch02 done(kind=primer)。review APPROVED,6 条 negotiable issue 均未阻断(留给后续任一次 retrofit 顺手打磨即可,非必须)。已登记 11 条精简版接口签名进 book/bible/interfaces.json(spmd_tile.*/simt_hierarchy.*/memory_hierarchy.*/coalescing.count_transactions/occupancy.occupancy/register_spill.spilled_registers)。bible.py due ch02 为空(无应埋/应回收伏笔——ch01 埋的 5 条伏笔 payoff 章都不是 ch02)。blind round1 抓到 fig-m03-latency-pyramid 编造的'~10x'倍数标注问题,round2 修复后过。
