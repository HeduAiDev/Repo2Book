# ch10-jitfunction-and-cache-keys-delivered-(skip_impl)

- **Type**: delivery
- **Chapter**: 10
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch10, skip_impl, part-3-opener, JITFunction, KernelParam, cache-key, compute_spec_key, mangle_type, create_function_from_signature

## What happened

第十章《@triton.jit、JITFunction 与缓存键》交付（Part III 开篇章，kind=source-reading/mode=skip_impl，pin triton==3.2.0 精确编译取证）。第一段（装饰期，10 机制中 3 个）：`@triton.jit` 把普通函数包成 `JITFunction`——`__init__` 只做纯登记（`inspect.signature` 解析参数成 `KernelParam`、切掉装饰器行留纯源码、按 device 分桶开空 `cache`），一行 IR 不生成；`KernelParam` 靠三个 `cached_property`（`is_constexpr`/`is_const`/`annotation_type`）分流参数走「constexpr 单列 / 静态签名 / 运行时 mangle」三条通道；`fn[grid]` 经 `KernelInterface.__getitem__` 两步语法糖（方括号记 grid，圆括号才发射）。第二段（发射期缓存键，7 机制）：`create_function_from_signature` 用 `exec` 把逐参数分派逻辑固化成一条直线 binder 函数（launch 快路径，摊薄发射开销大头）；`compute_spec_key` 把值域压成 D/1/N 三桶特化位（16 对齐/恰为1/其余）；`mangle_type` 给实参盖类型邮戳（签名项）；发射缓存键 = 签名 + 特化位 + `constexpr` 值——同源码不同特化各编一份，是本章性能落点（`add_kernel` 4 次发射示例：3 个不同键、1 次命中复用）；另一把 `cache_key` 源码哈希（`DependenciesFinder` 对 AST 遍历、递归混入被调 JITFunction 的 `cache_key`）与发射键正交，改代码即全域失效；`used_global_vals` 发射前核对兜底，全局量变了直接抛错不静默用旧产物。全篇 10 机制（5 core + 5 supporting）。禁区遵守：不重讲 ch01 的 visit_Call 三分派/@jit 内联心智模型（回指不重述）、不重讲 ch04 的 @builtin/constexpr 两层结构（回指其定义）、不展开 `run()` 完整 launch 流程（到「JITFunction 就绪+缓存键怎么算」为止，编译→发射留给下一章）。

## Why it matters

本章坐实了 ch03 早先点破但未展开的「缓存键=签名+特化位+constexpr(每次都在编译的病根)」——用 runtime 真源码把三个成分精确对应到 `mangle_type`/`compute_spec_key`/`constexpr_vals` 三处代码，并给出可直接套用的性能诊断法：认清什么进了缓存键就知道什么会触发重编译，尤其警惕给 `constexpr` 喂连续变化的值（会导致编译风暴）。review 提出的 8 条 issue 均 negotiable/non-blocking，其中 1 条是可核实的事实性瑕疵（「特化位」小节称 jit.py 的 `compute_spec_key` 与 backend 的 `AttrsDescriptor.get_property_key` 逻辑逐字相同，但二者在指针参数为 `None` 时给出不同特化码），6 条是 reader-comprehension 维度的术语先用后定义/近义词未打通类可读性小卡点（「特化位」「特化码」「特化项」「对齐位」四词未显式对齐、mangle 黑话先出现在表格里才后文解释等），1 条格式不统一（两个 supporting 机制缺「不变量」加粗标签）。均判定不阻断归档，留给 writer 后续顺手打磨。

## What to remember

ch10 done（kind=source-reading/mode=skip_impl，Part III 开篇章，无精简版接口）。dossier `foreshadow_due` 为空（should_plant/should_recycle 均空），本章未新开伏笔条目，也未标记回收——它是把 ch03 已建立的概念（非 arc-map 登记的 foreshadow，而是 concepts.json 条目）坐实到具体源码机制，`run()` 的完整 launch 编译→发射链接续下一章。glossary.json 新增 7 条术语（KernelParam、do_not_specialize/do_not_specialize_on_alignment、mangle_type、compute_spec_key、create_function_from_signature/binder、cache_key(源码哈希)、DependenciesFinder、fn[grid] 两步语法糖）。concepts.json 新增 10 条 → ch10。write_review_rounds=1、blind_rounds=1（0 failures）、map_rounds=1（PASS）、无 escalation。
