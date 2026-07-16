# ch14-compile-driver-loop-delivered-(skip_impl)

- **Type**: delivery
- **Chapter**: 14
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch14, skip_impl, part-4, compile, triton_key, ASTSource, IRSource, add_stages, BaseBackend, CUDABackend, f12-payoff

## What happened

第十四章《compile() 驱动主循环、编译入口与后端契约》交付（Part IV 开篇，kind=deep 但按 skip_impl 处理，pin triton==3.2.0；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 Lead 手工串行补归档）。全章打开 `compile()` 编排本体：①选后端（`make_backend(target)` 在已发现后端里筛 `supports_target` 为真者，要求恰好一个——是『target→backend』良定义函数、非单射，多个 target 可映同一 backend）；②内容寻址磁盘缓存键 = `triton_key()`（把 frontend+compiler/+backends/+libtriton.so+language/ 全部源码/二进制逐文件 sha256 拼成『编译器身份』单射指纹，@lru_cache 一进程一次）-`src.hash()`-`backend.hash()`-`options.hash()`-`env_vars`，sha256 后查缓存；③命中直接返回 `CompiledKernel`，未命中走慢路径：`stages=dict()` 交给 `backend.add_stages(stages, options)` 填出 `ir_name→pass` 有序字典（`BaseBackend` 六个抽象钩子契约之一，`CUDABackend.add_stages` 登记 ttir/ttgir/llir/ptx/cubin 五级即样例）；④`src.make_ir` 造起点 module（`ASTSource` 跑 `ast_to_ttir` 前端，`IRSource` 直接 `parse_mlir_module` 绕过前端，两入口 `hash` 口径不同：前者按 fn.cache_key+attrs+签名+常量寻址特化身份，后者按文件内容 sha256 寻址）；⑤主循环 `for ext, compile_ir in stages[first_stage:]` 逐级降级并落盘（IR 入口 `first_stage+=1` 跳过起点级——这正是『拿一份改过的 .ttgir 绕过前端直接从下一级起步做 IR 级实验』的机制本体）；⑥写回 metadata、`context.disable_multithreading()`、返回 `CompiledKernel`。9 机制（6 core+3 supporting）。6 图（chapter-map+5 个机制图：driver-loop/triton-key-invalidation/two-entrypoints/add-stages-skeleton/cache-key-composition）全 blind PASS。review APPROVED（7 条非阻断：1 处 `_path_to_binary`/docstring 静默删减未按惯例加省略标记+1 处 m3/m4 量化数字延后编排（合理，非漏做）+1 处图内文字跨框断裂+1 处 make_backend 判定表缺『2 个 actives』失败例+**1 处术语精度：正文误用『单射』描述 make_backend，应为『良定义函数（可多对一）』**+1 处 `self.capability` 赋值来源未交代+1 处 `AttrsDescriptor()` 默认构造未点破）。

**本章正式回收 f12**（ch11 埋：compile() 内部五段驱动主循环——ch11 只把 compile 当一次黑箱调用，本章打开其内容寻址缓存键/make_backend/add_stages/两入口/BaseBackend 六钩子全部展开）。回指 ch01 f1（后端接缝，payoff 仍在 ch36，本章只回指 BaseBackend 不动 f1）；回指 ch07（attrs 来源）、ch10（launch 特化缓存键三桶，与本章 triton_key 正交）；前瞻 ch36（CUDABackend 怎么填五段 add_stages 的完整实现）与 Part V-VIII（各 pass 专章）。

## Why it matters

本章是 ch11 埋下的 f12 唯一正式回收处，把『compile 是一次黑箱调用』坐实成具体的内容寻址缓存架构与后端契约机制，也是全书『改一行编译器源码，所有 kernel 缓存全失效』这一读者杠杆的建立处。review 指出的术语精度问题（单射 vs 良定义函数）已在 Bible glossary 中以正确措辞登记，避免误用词汇沿书扩散。

## What to remember

ch14 done（kind=deep 按 skip_impl 处理，Part IV 开篇）。glossary.json 145→152（新增 7 条：compile()/triton_key/ASTSource/IRSource/add_stages/BaseBackend/CUDABackend；同时更新 make_backend 词条补『良定义函数非单射』精度、更新已有『编译磁盘缓存键』词条标注 ch14 展开完整主循环）。concepts.json 新增 5 条→ch14（compile 五段驱动主循环/内容寻址编译缓存与内存 launch 缓存正交/add_stages 后端契约填降级链骨架/ASTSource-IRSource 两入口身份差异/IR 级实验机制）。interfaces.json 新增 ch14 键（源码接口，非精简版，ch14 skip_impl 无精简版）：`compile()`/`ASTSource`/`IRSource`/`BaseBackend.add_stages`/`BaseBackend.hash`·`parse_options`·`supports_target`/`make_backend`/`CUDABackend.add_stages`/`triton_key()`。arc-map.json：**f12 回收**（status open→resolved，resolved_in=ch14，`bible.py due ch14` 确认无遗留）；f1 未动（仍 open，payoff=ch36）。一致性核验：全部 status=resolved 的伏笔（f5→ch13/f7→ch06/f11→ch12/f12→ch14）均满足 payoff==resolved_in 且 payoff≤ch14，无异常。reviews/review-report.json 与 run-ledger.json 由 Lead 预写，本次未改动；narrative/chapter.md 与 diagrams/ 由 writer/illustrator 定稿，archivist 未触碰。
