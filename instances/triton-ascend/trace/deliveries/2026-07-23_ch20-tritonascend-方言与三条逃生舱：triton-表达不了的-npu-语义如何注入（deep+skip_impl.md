# TritonAscend 方言与三条逃生舱：Triton 表达不了的 NPU 语义如何注入（deep+skip_impl）

- **Type**: delivery
- **Chapter**: ch20
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T12:55:58Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-5, deep, skip_impl, hivm-hfusion, ascend-dialect, escape-hatch, ir-opname, authority

## What happened

Part 5「硬件 IR HIVM」——**hivm-hfusion 子系统开篇**，deep+skip_impl（纯 C++/`.td` 方言 + pass 章，无精简版）：《TritonAscend 方言与三条逃生舱：Triton 表达不了的 NPU 语义如何注入》。本章 workflow 在 **Review 站因图文矛盾逃生、未跑到 Archive**，矛盾由 Lead 处置完后由 Archivist 补归档。

**方言**：`TritonAscendDialect.td:L15` `let name = "ascend"` → 本方言 op 一律打印 `ascend.<助记符>`；`TritonAscendOps.td` 实定义 **11 个 op**（`grep -c 'TT_Ascend_Op<"' = 11`），是主链 TritonToLinalg 装不下的 NPU 专属语义（显式 GM/UB 索引搬运、双核同步、片上算子等）的共享容器。**§二是全书 ascend.* 命名的权威章**：给出 IR 名读法表（IR 名 = 方言 let name + ODS 助记符，**绝不从 C++ 类名倒推**）+ 错形反例 `ascend.indexput`（丢下划线，正确 `ascend.index_put`）/`triton.ascend.mod`（三段）/`tt.mod`（错方言），并证明 11 个里 6 个是 snake_case、机械倒推必错。

**三条逃生舱**（compiler.py 挂载序 **hivm → hfusion → llvm**，L148/149/150，全排在主链 add_triton_to_linalg L157 之前）：TritonToHFusion（贪婪 applyPatternsAndFoldGreedily，3 pattern：ascend.mod→hfusion elemwise、tt.histogram→histogram、tt.fp_to_fp 非 RTNE→cast，RTNE 默认返 failure 留主链）；TritonToHIVM（partial conversion，1 pattern：ascend.custom→hivm SyncBlock*，落核翻转，且这条舱只服务已废弃裸 sync_block_* 窄路径——新一代前端直建 hivm sync op 不经此）；TritonToLLVM（partial conversion，1 pattern：tt.elementwise_inline_asm→LLVM inline asm + 32 位打包）。三舱合计 **5 pattern**；11 个 ascend op 里三舱只消费 2 个（ascend.mod→HFusion、ascend.custom→HIVM），其余走主链其它 pass。核心判据：**『走逃生舱』标准是『主链吞不下』，不是『属不属于 ascend 方言』**（5 类源 op 里 3 类是核心 tt.*）。

**交付曲折（如实记）**：workflow Review 站 figure-integration 维抓出正文小节序数词『第一/二/三条舱』与图/管线序（hivm→hfusion→llvm）矛盾 → 逃生。Lead 处置：illustrator **正确判定图无错**（图按真实挂载序 hivm→hfusion→llvm）、退回 writer；派 writer 改中性标题 + 补『讨论序≠挂载序』说明句；派 illustrator 补本章地图并**独立复核 PASS**；Archivist 补归档。3 图（fig-ch20-ascend-dialect-container 状态表 / fig-ch20-pipeline-position 流程图 / chapter-map）全 blind PASS，16 门禁全绿。承 ch10 分水岭 18 趟管线里的三趟；对位基座《Triton 源码解读》ch24 的 ttng.* 硬件方言。

## Why it matters

ch20 是**全书 ascend.* 算子命名的权威来源章**：此前 ch06 曾把 `ascend.indirect_load` 误写成 `tt.indirect_load`（传播 6 处后修），ch10 盲审也查出正文 IR 名错 `triton.ascend.annotation`——命名规则一直分散、易错。本章 §二把『方言前缀 + 助记符、绝不从 C++ 类名倒推』讲透并证明 6/11 是 snake_case（机械倒推必错），配确定性门禁 `lint_ir_opname.py --all`，把这条全书通用约定钉死在一处权威出处。它同时是 hivm-hfusion 子系统的开篇，交代了主链 TritonToLinalg 为什么容不下这些窄情形、要单开三条窄逃生舱治——为 Part 5 后续 HIVM/HFusion 硬件 IR 章节建立坐标系。

## What to remember

- **权威口径**：ascend 方言 11 个 op 的正确 IR 名已登记进 Bible『IR 算子名的写法约定（全书通用）』词条（补强 ch20 为权威出处）——尤其 `ascend.index_put`（**带下划线**，最易错）；后续任何章讲到 ascend.* op 复用此表、别再从类名倒推。历史错名 ch06 `tt.indirect_load` 已修。
- **三舱事实钉死**：挂载序 hivm→hfusion→llvm（compiler.py:L148-150）、全在主链 add_triton_to_linalg（L157）之前；合计 5 pattern（HFusion 3 贪婪 / HIVM 1 / LLVM 1）；11 个 ascend op 三舱只消费 2 个。**正文讨论序（HFusion→HIVM→LLVM）≠ 管线挂载序**——这正是 Review 站图文矛盾的根因，图按真实挂载序无错、由 writer 补说明句消解。
- **Review 站逃生（如实）**：figure-integration 维抓小节序数词与图/管线序矛盾 → 逃生；illustrator 判图无错退回 writer；Lead 派 writer 改中性标题+补『讨论序≠挂载序』、派 illustrator 补本章地图并独立复核 PASS、Archivist 补归档。lead_brief_errors=无（本章 brief 无误）。
- **Bible 回写**：glossary +2 词条（『TritonAscend 方言（ascend）』『三条逃生舱（TritonToHIVM/HFusion/LLVM）』）+ 补强命名约定词条（现 233 键）；concepts +10（现 246）；figures +3（现 111，均 blind PASS）；interfaces **不新增**（skip_impl 无精简版）；arc-map **无埋无回收**（bible.py due ch20 两清单皆空）。
- **诚实边界**：host 无 CANN 工具链，skip_impl 无精简版；交叉验证走 pin `@2badfc89e` 精确源码（TritonAscendDialect.td / TritonAscendOps.td / TritonToHFusion|HIVM|LLVM.cpp / compiler.py:L148-157）逐段核对，不伪造编译器 dump。承 ch10；对位基座《Triton 源码解读》ch24（ttng.*）。
