# ch19-tt-dialect-vocabulary-delivered-(code)

- **Type**: delivery
- **Chapter**: 19
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch19, part-5, tt-dialect, TTIR, td-triple, assemblyFormat, trait, encoding, f14-plant, f15-plant

## What happened

第十九章《tt.\* 方言词汇表：读懂任何一段 IR dump 的入门词典》交付（Part V「IR 与布局」开篇，全书第一次从 Python 前端切到 MLIR/C++ 层；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 archivist 串行补归档）。七机制递进：①`.td` 三元组(`arguments`/`results`/`assemblyFormat`)↔dump 行双射——以 `make_range`(无操作数)→`splat`(一操作数)→`addptr`(两操作数)→`load`(可选操作数+默认值属性)逐步加码为样本，归纳法证明模板↔dump 行无歧义可逆；②`tt.*` 16 算子词汇表，专挑三个"长得不一样"的认脸——带 region 的 `reduce`/`scan`、`trans` 的"给线程手里元素改名"语义（为布局章埋钩子）、`make_tensor_ptr` 块指针构造口（与 `addptr` 指针张量对照两种寻址模态）；③`tt` 层类型系统只有 shape+dtype——`PointerType` 两种嵌套 `TT_PtrTensor`(指针张量)/`TT_TensorPtr`(块指针)对称拼出，`MemDescType` 是唯一带 encoding 字段的类型但 `tt` 层构造时仍常传空；④trait 性能语义——`Pure`→获准 CSE/DCE、`SameOperandsAndResultEncoding`→布局沿数据流传播但 `tt` 层因 encoding 空而恒放行(`verifySameEncoding` 代码级证明)、`TensorSizeTrait`→2^20 元素上限+2 的幂约束(H100 256KB 寄存器动机)、`VerifyTensorLayoutsTrait`→布局合法性闸门，`TT_Op` 基类用 `!listconcat` 自动挂两条 trait 给全体算子；⑤方言注册口 `name="tt"`+`dependentDialects`(回指 ch14 后端契约)；⑥枚举属性打印字符串(`I32EnumAttr` 符号名/整数值/打印字符串三元组)。本章无 subtract-only 精简版(skip_impl)——`.td`/C++ 定义本身即源码真相，忠实性靠 embed_excerpts 逐字内嵌+must_keep 符号覆盖保证。7 机制(4 core+3 supporting)。5 图(chapter-map+4 机制图：td-triple-to-dump/op-vocabulary/type-system/trait-perf)全 blind PASS。review APPROVED。

## Why it matters

本章是全书读 IR dump 的识字课，Part V 后续布局章(ch20-24)、Part VI-VIII 的优化 pass 章都要求读者已认得 `tt.*` 算子长相、类型嵌套写法、trait 标签。特别是"`tt` 层 encoding 恒空"这一不变量，是理解下一章"布局是个函数"的入口悬念；`trans` 算子的 encoding 说明文档是全章唯一直接前瞻 `convertLayout`/布局搬运的钩子。这两处新埋伏笔（f14→ch20、f15→ch24）把本章识字课与后续布局机制正式挂钩，避免读者在 ch20 之后忘记本章铺垫的对照关系。

## What to remember

ch19 done（kind=code，Part V 开篇，skip_impl 无精简版）。glossary.json 161→176（新增 15 条：`.td 三元组`/`assemblyFormat`/`PointerType`/`MemDescType`/`TensorPtr`/`Pure`/`SameOperandsAndResultEncoding`/`TensorSizeTrait`/`VerifyTensorLayoutsTrait`/`TypesMatchWith`/`DefaultValuedAttr`/`OptionalAttr`/`dependentDialects`/`CSE`/`DCE`；同时扩充已有『tt. 方言』词条补上硬件无关+encoding 恒空的定位）。concepts.json 135→139（新增 4 条→ch19：`.td` 三元组↔dump 行双射方法论、tt 层 encoding 恒空的代码级证明、trait 的性能语义 Pure→CSE/DCE、方言注册口 dependentDialects）。interfaces.json 新增 ch19 键（源码接口非精简版，供 ch20-24 回指：`TT_Op` 基类、`PointerType`、`MemDescType`、`getPointeeType`/`isTensorPointerType`、`verifySameEncoding`、`TT_TransOp` encoding 说明文档）。arc-map.json 新开两条正式伏笔：**f14**(plant ch19→payoff ch20：encoding 恒空↔ttg 层带 layout 的对照，下一章正面回答布局是个函数)、**f15**(plant ch19→payoff ch24：trans 的 encoding 文档，真正数据搬运在 convertLayout，ttg.\*/ttng.\* 算子章兑现)；未动其它伏笔状态。

一致性核验：全部 status=resolved 的伏笔（f5→ch13/f7→ch06/f11→ch12/f12→ch14）均满足 payoff==resolved_in 且 payoff≤已交付章节，无异常。**发现并修复 Bible 完整性 bug**：伏笔 f4(plant ch01→payoff ch16)在本次会话读取之前的工作区里已被误标为 `resolved`/`resolved_in=ch16`，但 `instances/triton/artifacts/ch16-codegenerator-ast-visitor/` 当时尚无 `reviews/` 目录（未过 review、未归档），trace 无 ch16 交付记录，`git show HEAD` 确认上一次提交里 f4 仍是 `open`——疑似 ch16 pipeline 的 Dossier 站误写 bible 或上一位 archivist 会话遗留的半成品改动（此前 ch13 归档时也发现过同类 bug：ch12 提交曾误把 f12 一并标 resolved）。已先上报 Lead 未擅自处理；Lead 确认后（ch16 走 skip_archive 模式，其 Archive 站不会再跑、Dossier 站也已跑过不会再触 arc-map，此刻改动安全）按其指示把 f4 恢复为 `"status": "open"` 并删除 `resolved_in` 字段（对齐 git HEAD 状态）。修复后复核：`bible.py due ch16` 正确显示 f4 为应回收项；arc-map 全部 resolved 伏笔仅剩 f5/f7/f11/f12，payoff 均已交付，无异常。

reviews/review-report.json 与 run-ledger.json 由 Lead 预写，本次未改动；narrative/chapter.md 由 writer 并行修，diagrams/ 由 illustrator 并行修一张图，dossier/dossier.json 的 embed 行号由 Lead 刚修过——以上三者 archivist 均未触碰。state.json 已加 ch19 条目并刷新 updated 时间戳；trace/INDEX.md 已刷新（自检确认 ch19 已在列，保留最近 10 条）。
