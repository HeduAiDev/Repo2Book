# ch11 指针算术的逆向工程PtrAnalysis把addptr链还原成stride-offset

- **Type**: delivery
- **Chapter**: ch11
- **Date**: 2026-07-22
- **Timestamp**: 2026-07-22T18:46:44Z
- **Agents involved**: analyst, writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: ptranalysis, addstate, mlir, skip_impl, part3

## What happened

Part 3 第 3 章·deep+skip_impl(纯 C++ MLIR pass 章，承 ch10 分水岭总览，skip_dossier=true 直接吃已由 Lead 逐条订正的 dossier)：《指针算术的逆向工程：PtrAnalysis 把 addptr 链还原成 stride/offset》。深挖 TritonToStructured::PtrAnalysis 核心算法上半——visitOperand 递归分派器(14 个 defining-op 分支，3 快门在前)如何沿 tt.addptr 定义链向上问、把散落在 make_range/splat/broadcast/expand_dims/mul/add/rem/div 里的地址算术前向还原成 PtrState(source+offset+逐维 stateInfo)。11 个机制全部覆盖：m1 状态词汇(StateInfo/PtrState)、m2 分派器全貌、m3 visitOperandAddptr(拆 ptr/offset 双子状态再 addState 合并)、m4 addState 完整代数(dimIndex 归并+isMultiple 校验+维拆分)、m5 mulState/subState(仅一侧标量)、m6 make_range 精确公式 stride=(end-start+n-1)/n、m7 三个纯形状算子、m8 一条真实 matmul b_ptrs 2D 指针链的完整前向传播 worked example(14 个算子、演化表 13 行)、m9 失败/保守路径、m10 normalizeState 规范化、m11 rem/div 按块折叠。**dossier-verify 两处关键订正均已落地**：①block-arg 指针(!tt.ptr)是 visitOperand 的成功入口而非失败(只有非指针非标量裸 block-arg 才触发 failure)；②make_range 叶子的 size 是 shape[0](结果长度 n)，不是 end。6 图+本章地图共 7 图，独立盲审首轮 PASS(0 failure)、chapter-map 盲审首轮 PASS。lint_trace_consistency 全绿，零数值漂移。Review 3 轮收敛：algorithm-pedagogy 维复核 APPROVED 无 issue；另一份综合 review-report 记 7 条 issue(6 条 non-blocking 可读性/术语先用后定义类，1 条 figure-integration 维标 blocking——fig-m7-shape-evolution 图内脚注『splat 是唯一引入 source 的形状算子』与正文订正后口径(source 来自 initStateByPointer、splat 不产生新 source)字面矛盾，需 writer/illustrator 后续定点改 gen_fig-m7-shape-evolution.py:L117 脚注措辞并重渲染)。

## Why it matters

ch11 是 triton-to-linalg 子系统(ch10-14)deep+skip_impl 系列的第二章，把 ch10 分水岭总览里一句话带过的 PtrAnalysis 逆向算法拆到可验证的代数细节——addState 的 dimIndex 归并/isMultiple 校验/维拆分是本章代数核心，m8 的完整 2D 指针链 worked example 是全章唯一走完递归树全貌的示范。两处 dossier-verify 订正(block-arg 指针是成功入口、make_range size≠end)若不落实会在正文与 worked example 之间自相矛盾，已核实两处正文表述一致、互相呼应。埋伏笔 f1 指向 ch12：PtrState 终态的 strides/sizes/offset 三元组 + shouldLinearize 标志要在下一章铸成 memref.reinterpret_cast。

## What to remember

ch11 APPROVED(skip_impl 无精简版/无测试，交叉验证走 pin 精确源码+lit 夹具)；7 图+本章地图盲审全 PASS；遗留 1 条 figure-integration blocking 级图注口径问题(fig-m7 脚注『splat 引入 source』与正文订正矛盾)待 writer/illustrator 后续小修 gen_fig-m7-shape-evolution.py:L117，非本次交付阻断项；已登记 glossary+8/concepts+9/figures+7；埋伏笔 f1→ch12(三元组+shouldLinearize 落 memref)。
