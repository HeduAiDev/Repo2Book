# ch15-ssa-and-structured-control-flow-delivered-(primer)

- **Type**: delivery
- **Chapter**: 15
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch15, primer, part-4, SSA, phi, block-argument, scf.for, scf.if, iter_arg, loop-carried, f13-plant

## What happened

第十五章《SSA 与结构化控制流：φ 节点、块参数与 loop-carried 变量》交付（Part IV「编译前端」理论台阶，kind=primer，skip_impl 无精简版；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 archivist 串行补归档）。三层递进：①SSA 不变量(∀xᵢ, |defs(xᵢ)|=1)+φ 节点记号(Cytron et al. 1991，汇合块按前驱选值)——红线首现：φ 只是语义记号，Triton 不跑其支配边界/最小 φ 插入算法；②MLIR 命门：块参数取代 φ（arXiv:2002.11054 §Regions and blocks 逐字引文），φ『拉』块参数『推』的等价式，region⊃block⊃Op 嵌套定义；③scf.for/scf.if 结构化控制流参数布局（归纳变量 arg(0)+loop-carried 变量 arg(i+1)，SCF 官方方言文档逐字引文两处）；④落地：Triton 前端 `enter_sub_region`+`visit_For`(L957-L1027)+`visit_if_scf`(L656-L681) 真实源码内嵌，loop-carried 判据 = `local_defs ∩ liveins`，红线第三遍收口——零支配边界计算。8 机制（4 有独立图：phi-merge-diamond/phi-vs-block-arg/scf-for-arg-layout/loop-carried-scope-diff）。5 图（chapter-map+4 机制图）全 blind PASS（round1 0 failures）。review APPROVED，PRIMER 维度 paper-fidelity PASSED、φ 红线全程持住；全部非阻断（1 处 dossier `paper_origin.sections` metadata 卫生问题已由 Lead 归档前修复，lint_paper_grounding 现零 WARN；其余为篇幅/直觉句锦上添花建议）。

## Why it matters

本章是全书读懂后续所有循环/控制流下降章节的语言地基：`num_stages` 软件流水线调度、循环体寄存器压力的一部分来源，都挂在本章建立的 loop-carried/iter_arg 值链上。术语精度红线（φ 是记号非算法、Triton 直接从结构化 AST 用作用域交集构造 SSA、零支配边界）已写入 glossary 供全书后续章节遵守，避免『Triton 计算 φ』这类误用扩散。

## What to remember

ch15 done（kind=primer，skip_impl，Part IV 理论台阶）。glossary.json 152→161（新增 9 条：φ 节点/块参数/loop-carried 变量/scf.for/scf.if/iter_arg/region-block-MLIR 嵌套/terminator/结构化控制流；同时更新已有『SSA』词条补齐三层落地摘要+红线）。concepts.json 131→135（新增 4 条→ch15：SSA 不变量、φ↔块参数拉推等价、loop-carried dry-run+交集判据、scf.for/scf.if+iter_arg 值链）。interfaces.json 新增 ch15 键（源码接口非精简版，理论记号不入表，只登记后续 ch16/ch17 会回指的真实锚点：`enter_sub_region`/`set_value`/`visit_For`/`visit_if_scf`）。arc-map.json 新开正式伏笔 **f13**（plant ch15 → payoff ch17：SSA/φ/块参数地基在 ch17 补全 visit_If 分流/负步长/visit_While/属性挂载，把 if/for/while 全部路径走通）；未动其它伏笔状态。一致性核验：全部 status=resolved 的伏笔（f5→ch13/f7→ch06/f11→ch12/f12→ch14）均满足 payoff==resolved_in 且 payoff≤ch15，无异常；f1/f6/f8/f9/f10 仍 open（payoff 均>ch15），f13 新开 open。reviews/review-report.json 与 run-ledger.json 由 Lead 预写，本次未改动；narrative/chapter.md 与 diagrams/ 由 writer/illustrator 定稿，archivist 未触碰；dossier/dossier.json 的 paper_origin.sections 由 Lead 修过，archivist 未再动。
