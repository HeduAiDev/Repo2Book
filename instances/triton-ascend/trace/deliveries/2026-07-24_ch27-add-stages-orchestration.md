# ch27-add-stages-orchestration

- **Type**: delivery
- **Chapter**: ch27
- **Date**: 2026-07-24
- **Timestamp**: 2026-07-24T02:00:00Z
- **Agents involved**: writer, reviewer, archivist
- **User present**: False
- **Tags**: backend-runtime, part-6, add_stages, ttir_to_linalg, ttadapter, pass-orchestration, force_simt_only, ttgir-contrast

## What happened

Part 6「后端与运行时」第二站(承 ch26 装配站，本章深入 ch26 登记表的三段内部)·kind=meta(综合/编排章，非新机制首现，站在后端 POV 把 ch10-24 逐章讲过的 pass 串成一条真实流水线)·deps=ch10(triton-to-linalg 分水岭)/ch26(add_stages 装配机制)。9 个机制(4 core+5 supporting)全覆盖：**three-stage-registration**(add_stages 两个 if 只改「登记哪个实现」——force_simt_only 为真只登记 ttir+npubin 两段直接 return，为假额外登记 ttadapter，两分支互斥穷尽)；**make-ttir-shared-passes**(8 个 TTIR 前端优化 pass，与基座逐字共享，源码注释自证)；**ttadapter-pass-orchestration**(本章主脊：ttir_to_linalg 按拓扑序挂 11 个 ascend.passes.ttir.add_* 主线——auto_blockify→structure→discrete_mask→annotation→unstructure→hivm→hfusion→llvm→bubble_up→structure 第二遍→triton_to_linalg 收口，逐个对上 ch10-24 章号)；**auto-scheduling-conditional-block**(可选 dag_sync/dag_scope/ssbuffer+cse/canonicalizer 清理，默认关，开启后 11→18 趟)；**pass-params-from-metadata**(编排与旋钮解耦：所有开关从 NPUOptions/metadata 取出后逐个喂给 add_*)；**structure-pass-twice**(triton_to_structure 跑两遍，pass 迭代收敛手法)；**auto-blockify-size-gate**(未开自动并行块映射时强制归 1)；**force-simt-fast-path**(force_simt_only 快路径：装配层少登记 ttadapter 一段而非在 pass 链里加分支，11→0)；**three-vs-five-stage-contrast**(昇腾三段 vs 基座五段，对位基座 ch26 CUDABackend，因无真实 warp/warp_size=0 而省掉整层 TTGIR)。write↔review 1 轮收敛，blind 1 轮 0 failure，map 1 轮 PASS。多维评审 APPROVED，0 blocking+7 non-blocking(1 条 fidelity 精确性：ttir_to_npubin 代码块头部行区间标 L824-L868 比函数真实结尾 L874 少 6 行，dossier code_spine 与 embed_excerpts 两处行号本就不一致，writer 忠实抄了错的那处；2 条 algorithm-pedagogy：three-stage-registration/force-simt-fast-path 两个 core 机制的『互斥穷尽』不变量停留在隐含成立、未显式点破；1 条 anchors warn：标题裸文字章号，全书统一良性模式非本章缺陷；3 条 reader-comprehension：表格里两个开关缺中文译名破坏密度一致性/add_dag_sync 括号引用归属歧义/表格「对应章」列三格描述性文字与其余真实章号混排造成短暂困惑)——均 negotiable 且 non-blocking，留存量回修批次，未做退回重写。**回收伏笔 f6**(ch26→ch27：ttadapter 段内部 pass 编排已在本章兑现，arc-map 状态改 resolved)；**新埋伏笔 f7**(ch27→ch28：npubin 段 bishengir-compile 命令行拼接、compile_on_910_95 两候选实现差异、闭源边界，本章只讲到 force_simt_only 快路径 ttir_to_npubin 的开头几行开关)。kind=meta(非 primer 非 deep 常规码章)，无 implementation/tests 目录(该章无精简版产物，纯编排综合章，交叉验证走 pin 源码逐行核对 add_stages/ttir_to_linalg/ttir_to_npubin 三处真实函数体，不伪造运行 dump)。

## Why it matters

backend-runtime 子系统承上启下站：把 Part 3-5(ch10-24)逐章分别讲过的十来个 pass——auto_blockify(ch15)、triton_to_structure(ch10-13)、discrete_mask(ch19)、unstructure(ch14)、hivm(ch23)、hfusion(ch21)——第一次按真实拓扑序拼成读者能看见首尾相接的一条流水线,是全书『先见树木后见森林』方法论的收口站。同时首次把昇腾三段与基座五段并排对照(对位 ch26 讲的 CUDABackend),把『少两段』这个此前只在 ch01 鸟瞰图上出现过的事实,追溯到『无真实 warp 故无需 TTGIR 层』这一可推导的硬件根因,而非停留在『昇腾编译流程更短』的印象式陈述。

## What to remember

ch27 APPROVED,backend-runtime 子系统第二站(承 ch26)。4 core+5 supporting 共 9 机制全覆盖,主脊是 ttir_to_linalg 11 个 add_* 拓扑序编排与 ch10-24 各章的逐一对应。评审 0 blocking+7 non-blocking(1 条行号精确性问题源头在 dossier embed_excerpts,留存量回修;2 条不变量显式化;1 条 anchors warn 良性;3 条表格一致性打磨)。**回收伏笔 f6**(ttadapter 内部编排兑现),**新埋伏笔 f7**(→ch28:npubin 段 bishengir-compile 命令行构造与 compile_on_910_95 两实现差异,闭源边界)。Bible 回写:glossary+4(ttir_to_npubin/CTA/warp_size 两副面孔 opt.warp_size=32 vs GPUTarget.warp_size=0;force_simt_only/compile_on_910_95/named_ops/make_ttir·ttir_to_linalg/ttir_to_linalg pass 管线 18 趟均已在 ch02/ch09/ch10/ch20/ch26 首现,本章确认沿用未重复登记)/concepts+6(对应 6 条机制级要点);interfaces 不新增(无精简版)。下一站 ch28(npubin 段/bishengir-compile 命令行/闭源边界)。
