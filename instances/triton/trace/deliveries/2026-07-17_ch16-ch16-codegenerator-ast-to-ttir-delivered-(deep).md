# ch16-codegenerator-ast-to-ttir-delivered-(deep)

- **Type**: delivery
- **Chapter**: 16
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch16, deep, part-4, CodeGenerator, ast.NodeVisitor, constexpr, tensor, visit_Call, f4-payoff

## What happened

第十六章《CodeGenerator：AST 访问者、两个世界与表达式/函数下降》交付（Part IV「编译前端」，kind=deep，skip_impl 无精简版；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 archivist 串行补归档）。打开 `CodeGenerator`（`ast.NodeVisitor` 子类）这台把 kernel 函数体 AST 翻成 `tt.*` IR 的核心机器：①`visit` 外壳（set_loc 挂源码位置 `#loc` + 异常就地包 `CompilationError` + `generic_visit` 白名单拒绝未实现节点）；②`gscope`/`lscope`/`local_defs` 三层作用域账本 + `name_lookup` 三级查找（local→global→builtin，global 只放行 constexpr，`visiting_arg_default_value` 求参数默认值时临时放宽）；③贯穿全器的 constexpr（编译期 Python 值）↔tensor（运行期 SSA 句柄）二分，在 `visit_FunctionDef` 参数下降 / `call_JitFunction` 调用下降 / `visit_BinOp` 运算符下降三处反复现身，同一条规则；④`visit_Call` 三分派（**f4 命门**）：static 编译期求值 / JITFunction 经 `call_JitFunction` 内联（constexpr 抽进 `constants` 进 `mangle_fn` 函数名、tensor 走 `handle` 进 `tt.call` 操作数，同 (arg_types, constants) 只生成一次）/ builtin 注入 `_builder` 建 op；⑤`visit_FunctionDef` 建 `tt.func`：constexpr 参数不占 IR 位（idx 跳位）+ `AttrsDescriptor` 的 `tt.divisibility` 经 `set_arg_attr` 落进 IR（第一性能命门，前瞻 ch25 AxisInfo 消费）。11 机制（6 core + 5 supporting，6 个带图）。7 图（chapter-map + 6 机制图：visit-shell-flow/three-scope-lookup/constexpr-tensor-worlds/visit-call-dispatch/constexpr-not-in-ir/divisibility-chain）全 blind PASS（round1 0 failures）。review APPROVED（6 negotiable/non-blocking issues，均 reader-comprehension 维度：§5 直觉段缺失、§2 图文字数不一致、`new_constants`/`visiting_arg_default_value` 未解释、TMA descriptor 分支未标省略、开篇钩子 API 缺回指；f4 三岔恢复经 Lead 核验源码准确）。无精简版（kind=deep 按 skip_impl 处理——编译器代码库无法干净抽取同名同结构精简版，`CodeGenerator` 离不开整条 `ir.builder`/`language` 栈；交叉验证支柱由 pin triton==3.2.0 headless `ASTSource(...).make_ir(...)` 精确编译出真·追踪期 TTIR 承担，观测 `tt.func` 签名/constexpr 不占位/`tt.divisibility` 属性）。write_review_rounds=1、blind_rounds=1（0 failures）、map_rounds=1（PASS）、无 escalation。

## Why it matters

本章是全书 f4 伏笔（ch01 埋：visit_Call 三岔分发 + tl @builtin/@jit 两层结构）的正式回收章——把 ch01 鸟瞰章点名但未展开的"追踪器怎么把 f(...) 调用分成三条路"讲透，也是"你标的 :constexpr 参数为何不占运行期签名一个位"这条全书性能主线（f3）在前端的第一个真实落点。`set_arg_attr`→`tt.divisibility` 这条属性链把 ch09 `multiple_of`/`max_contiguous` 打的提示真正接进 IR，是 ch25 AxisInfo 静态分析能消费到的前提——呼应已登记的 f9（不重开新伏笔）。

**f4 回收时点修正（重要）**：f4 此前曾被 ch16 writer 在 Write 阶段用旧流程 `bible.py payoff --resolve` **过早**标 resolved，又被 ch19 archivist 发现异常（当时 ch16 无 reviews/ 目录、未过 review）并回滚为 open（记录于 ch19 note）。现由本章 archivist 在真正定稿归档时合法回收，exp-0717-2 已把此漏洞堵上（writer 不得再直接调 `payoff --resolve`，回收权收归 archivist 归档步骤）。

## What to remember

ch16 done（kind=deep，skip_impl，Part IV）。glossary.json 176→184 条（新增 8 条：CodeGenerator/gscope-lscope-local_defs/name_lookup/constexpr↔tensor 二分/ast_to_ttir/visit_FunctionDef/set_arg_attr-tt.divisibility/CompilationError；更新既有『visit_Call 三岔』词条追加 ch16 完整展开+f4 回收标注；`mangle_fn` 未新登记——dossier/正文明确其"回指 ch10 mangle"，视为复用非新概念）。concepts.json 新增 4 条→ch16（CodeGenerator 逐节点分派、constexpr↔tensor 二分贯穿全器三处现身、visit_Call 三分派判据落地、constexpr 不占 IR 位+divisibility 落 tt.divisibility）。interfaces.json 新增 ch16 键（源码接口非精简版：`ast_to_ttir`/`visit_FunctionDef`/`name_lookup`/`visit_Call`/`call_JitFunction`/`visit`/`CompilationError._format_message`，均带 code_generator.py/errors.py 真实行号锚点，供 ch17/ch25 回指）。

arc-map.json：**f4 status open→resolved，resolved_in=ch16**（`python3 scripts/bible.py payoff --resolve f4 --in ch16`）。一致性核验：全部 resolved 伏笔（f4→ch16/f5→ch13/f7→ch06/f11→ch12/f12→ch14）均满足 `payoff==resolved_in` 且 `payoff≤ch16`，无异常；`bible.py due ch16` 已验证 f4 不再列出待回收项；f1/f2/f3/f6/f8/f9/f10/f13/f14/f15 仍 open（payoff 均>ch16），未动。本章未埋新伏笔（m11 前瞻 ch25 AxisInfo 消费 tt.divisibility 与既有 f9 重合，未重复登记）。

reviews/review-report.json 与 run-ledger.json 由 Lead 预写，本次未改动；narrative/chapter.md（writer 并行修 6 处 reader-comprehension 建议）与 diagrams/（illustrator 并行修 1 张图）由二者定稿，archivist 未触碰；dossier/dossier.json 未改动。
