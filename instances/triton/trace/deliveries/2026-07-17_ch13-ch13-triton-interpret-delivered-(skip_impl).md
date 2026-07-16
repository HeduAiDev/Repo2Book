# ch13-triton-interpret-delivered-(skip_impl)

- **Type**: delivery
- **Chapter**: 13
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch13, skip_impl, part-3, interpreter, TRITON_INTERPRET, InterpretedFunction, GridExecutor, f5-payoff

## What happened

第十三章《TRITON_INTERPRET：让整本书的核，在没有 GPU 的机器上跑给你看》交付（Part III，kind=skip_impl，pin triton==3.2.0；并行发车 skip_archive 模式，Review+Map 已 APPROVED，本次由 Lead 手工串行补归档）。全章讲透替身执行的完整链路：① 入口分叉——`@triton.jit` 装饰器里 `TRITON_INTERPRET==1` 时返回 `InterpretedFunction` 而非 `JITFunction`（jit.py:L834-L838），分叉发生在装饰器层、用户零改动切换。② AST 改写——`FunctionRewriter.rewrite_ast` 取源码→定位 def→dedent→`ast.parse`→`ASTTransformer.visit_Assign` 把每个赋值 `x=value` 改写为 `x=to_tensor(value, interpreter_builder, False)`→重排行号（报错指回原文件）→compile/exec 进原函数 globals；回答『为何不能原样跑 Python』——裸 int 无张量语义可依。③ `GridExecutor.__call__` 三重 for 串行遍历 grid（一次一个 program）：剔 `RESERVED_KWS`→`_init_args_hst` 拷参到 host→`_patch_lang` 把 tl.* builtin 整体重绑到 `InterpreterBuilder`→`_implicit_cvt` 把指针实参换成装 `data_ptr()`(uint64 地址)的 `TensorHandle`→逐点 `set_grid_idx` 调核体→`_restore_args_dev` 回拷副作用。④ `InterpreterBuilder` 与真 IR builder 同名 `create_*` 接口，用 numpy 直算数值、不建 IR 节点（program_id 从 grid_idx 取、load/store 经 C++ `_interpreter` 按地址读写 host 内存）。⑤ 点破边界：串行≠并行，查对错不查快慢——`RESERVED_KWS`（num_warps 等）与 `cache_modifier`/`eviction_policy` 等性能旋钮在解释器里被剔除或忽略，量不出合并访存/occupancy。8 机制(5 core+3 supporting)。7 张机制图+chapter-map 全 blind PASS。review APPROVED(全非阻断，含入口分叉段缺实机验证引用/m7 标签体例不统一/fig-m3 行号旁注误差/`self.constexprs` 首现未释等 reader-comprehension 小卡点)。

**本章正式回收 f5**（ch01 埋：TRITON_INTERPRET=1 返回 InterpretedFunction、FunctionRewriter 重写 AST、GridExecutor CPU 串行遍历 grid——本章逐段内嵌真源码正面回收，data_flow 九步完整对应）。

## Why it matters

本章是 ch01 埋下的 f5（替身执行旁路点名）唯一正式回收处，把「不是原样跑 Python」坐实成具体源码机制（AST 重写注入张量语义 + numpy 兜底 builder），也是全书唯一一处「无需 GPU、以 CPU 串行复现」的调试/验证路径，为后续任何提到"无卡验证逻辑正确性"的章节提供回指锚点。归档时发现并修复了一处 Bible 完整性 bug：ch12 交付提交（a74daad2）在回收 f11 时误连带把 f5（本应待 ch13 才回收）与 f12（payoff ch14，尚未交付）一并标记 resolved——f5 因巧合与本次任务对象一致而结果无误，但 f12 是真实的过早回收（ch14 未交付），已改回 open，`bible.py due ch14` 复核确认 f12 恢复正确显示待回收。

## What to remember

ch13 done（kind=skip_impl，Part III）。glossary.json 135→145（新增 10 条：TRITON_INTERPRET/InterpretedFunction/FunctionRewriter/to_tensor(解释器语境)/GridExecutor/_implicit_cvt/_patch_lang/InterpreterBuilder/TensorHandle/替身执行(interpreter 模式)）。concepts.json 新增 5 条→ch13（替身执行=入口分叉+AST 重写+CPU 串行、AST 改写把赋值包成 to_tensor、解释器不建 IR 直接 numpy 算、指针实参→uint64 地址、单线程串行≠GPU 并行的边界）。interfaces.json 新增 ch13 键（源码接口，非精简版，ch13 skip_impl 无精简版）：`InterpretedFunction`、`GridExecutor.__call__`、`InterpreterBuilder.create_*`。arc-map.json：**f5 回收**（status open→resolved，resolved_in=ch13，`bible.py due ch13` 确认无遗留）；同时修复 f12 的过早 resolved（改回 open，见上）；dossier `foreshadow_due.plant` 为空，本章未新开正式伏笔。reviews/review-report.json 与 run-ledger.json 由 Lead 预写，本次未改动；narrative/chapter.md 与 diagrams/ 由 writer/illustrator 并行定点修，archivist 未触碰。
