# ch21-distributed-layouts-delivered-(skip_impl)

- **Type**: delivery
- **Chapter**: 21
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T04:36:42Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch21, part-5, skip_impl, distributed-layout, blocked-encoding, slice-encoding, nvidia-mma, dot-operand, coalescing

## What happened

第二十一章《Distributed 布局：Blocked、Slice、MMA 与 DotOperand 编码》交付（Part V「IR 与布局」第三站，skip_impl 章；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 archivist 串行补归档）。承 ch20「布局=函数 𝓛」的抽象，把 distributed 布局的具体 encoding 形态走一遍：①4 级计算层级公共骨架(CTAs Per CGA→Warps Per CTA→Threads Per Warp→Values Per Thread，`DistributedEncodingTrait`)；②`BlockedEncodingAttr` 三元组(`sizePerThread`/`threadsPerWarp`/`warpsPerCTA`+`order`)把张量切成「每线程一块连续元素」，`getContigPerThread()`恒等于`getSizePerThread()`；③coalescing 编码充要条件——`order[0]`指向连续维+元素块连续⇒warp 32 lane 覆盖连续地址段⇒合并成一笔事务(回指 ch07/f9)；④自动推导 builder 从 shape+numWarps 起由 order[0] 逐级 clamp 反解各级 tile；⑤`getElemsPerThread`布局算术(t=sizePerThread·threadsPerWarp·warpsPerCTA，headless 可算)；⑥`SliceEncodingAttr`——对 parent 挤掉一维的降维投影(expand_dims 的逆)，靠`paddedShape`回填+`erase(dim)`实现；⑦`NvidiaMmaEncodingAttr`(versionMajor 区分 Volta/Ampere/Hopper，instrShape 携带 MMA 指令尺寸，点到即止深化留 ch27)；⑧`DotOperandEncodingAttr`(opIdx+parent+kWidth，Ampere 下 kWidth=32/bitwidth，命中 Tensor Core 所需的操作数排布)；⑨backend_seam——AMD/Nvidia 矩阵乘布局同继承`DistributedEncoding`、并排同一 .td，后端新增布局族的现成样板。全部由 TritonGPUAttrDefs.td/Dialect.cpp 真实源码逐段内嵌驱动；无精简版(kind=skip_impl，.td/C++ 定义即源码真相，交叉验证由 explainer 的 pin v3.2.0 headless 精确编译承担，实测 1D copy 与 64×64 fp16 matmul@sm80 的真实 #blocked/#mma/#dot_op encoding)。9 机制(6 core+3 supporting，6 个带图)；7 图(chapter-map+6 机制图：fig-distributed-hierarchy/fig-blocked-triple/fig-coalescing-order/fig-slice-squeeze/fig-nvidia-mma-version/fig-dot-operand-sizes)全 blind PASS(round1 zero failures)。write-review 2 轮、无 escalation。review APPROVED(7 issues 全 negotiable/non-blocking，均 reader-comprehension/文档同步维度：dossier elide 字段留白未登记/slice-squeeze 量化深度弱于其余 core 机制/fig-blocked-triple 图注漏收尾结论句/instrShape 引文未列出却被文字暗示已引用/worked-example 表格先于推导依据出现/sizePerThread 与四级层级 ValuesPerThread 未打通名称等)。

## Why it matters

本章把 ch07 的合并访存判据、ch20 的抽象布局函数，第一次落到 GPU 张量类型尾巴上真正打印的 encoding 参数——读懂`#triton_gpu.blocked<{...}>`/`#mma<{...}>`/`#dot_op<{...}>`是后续所有性能 pass 章(ch25 起)的读图前提。`BlockedEncodingAttr`三元组是 ch25(AxisInfo 与 Coalesce)把人工判据自动化的对象；`NvidiaMmaEncodingAttr`/`DotOperandEncodingAttr`是 ch27(mma 布局深化)、ch28(AccelerateMatmul)的直接前置知识；`SliceEncodingAttr`呼应 reduce/broadcast 结果布局，是 ch22(shared/swizzle 对照)前的最后一块 distributed 拼图。

## What to remember

ch21 done（kind=skip_impl，Part V 第三站，7 图全 PASS，write-review 2 轮/blind 1 轮/无 escalation）。**glossary.json 215→226**（新增 11 条：`BlockedEncodingAttr`/`sizePerThread`/`threadsPerWarp`/`warpsPerCTA`/`order`/`getDefaultBlockedEncoding`/`coalescing 编码条件`/`getElemsPerThread`/`SliceEncodingAttr`/`NvidiaMmaEncodingAttr`/`DotOperandEncodingAttr`；同时扩充既有`Blocked 三元组`词条，标注 ch21 完整展开三元组+order+布局算术+兄弟成员）。**concepts.json 159→163**（新增 4 条→ch21：BlockedEncoding 三元组切法决定访存合并、自动推导 builder 反解三元组、SliceEncoding 降维投影、NvidiaMma/DotOperand 是 Tensor Core 强制的输入输出布局）。**interfaces.json 新增 ch21 键**（`BlockedEncodingAttr`/`getDefaultBlockedEncoding`/`getElemsPerThread`+`squeeze`/`SliceEncodingAttr`/`NvidiaMmaEncodingAttr`/`DotOperandEncodingAttr`/`DistributedEncodingTrait`+`DistributedEncoding`模板，逐条标注供 ch23(LinearLayout)/ch27(mma 布局深化)/ch28(AccelerateMatmul)回指的源码锚点）。

**arc-map.json**：**未新开正式伏笔**——dossier.json 的`foreshadow_due.should_plant`为空，且`bible.py due ch21`应埋/应回收均为空；判断本章「NvidiaMmaEncoding/DotOperandEncoding 深化留 ch27」「toLinearLayout 接口留 ch23」均为已有伏笔覆盖(f17 已 plant ch20→payoff ch23 覆盖 toLinearLayout 一句带过)或泛化的「下一章见」式过渡语句(mma 深化留 ch27 属于 ch27 自身主题的自然延续，非本章埋下的具体技术悬念钩子)，比照 ch20 处理 distributed/shared 分野的先例，从简不登记以防 bloat。**一致性核验**：全部 resolved 伏笔(f4→ch16/f5→ch13/f7→ch06/f11→ch12/f12→ch14/f13→ch17/f14→ch20)均 payoff==resolved_in 且 payoff≤已交付章节，无异常；f15(plant ch19→payoff ch24)/f16(plant ch17→payoff ch30)/f17(plant ch20→payoff ch23)仍 open，均未被误动，payoff 均>ch21。

trace：本条 delivery 已建；`state.json`已加`ch21`条目；`trace/INDEX.md`已刷新（保留最近 10 条，自检确认 ch21 在列）。**注**：本次归档过程中曾误跑无参`archivist.py record`探针两次，产生`decisions/2026-07-17_untitled.md`与`deliveries/2026-07-17_untitled.md`两个垃圾文件（及被自动写入 INDEX.md 的两条 Untitled 行）——已发现并清理（删除垃圾文件、手工修复 INDEX.md 移除对应行），未遗留。`reviews/review-report.json`、`reviews/run-ledger.json`、`narrative/chapter.md`、`diagrams/`、`dossier/dossier.json`均未触碰（按指示，Lead 已预写评审文件，writer 并行修 ch06）。
