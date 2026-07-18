# ch04 交付：前端接缝——双 builder 与 Ascend 内建的分发路由

- **Type**: delivery
- **Chapter**: ch04
- **Date**: 2026-07-18
- **Timestamp**: 2026-07-18
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, ch04writer, reviewer, Lead, archivist
- **User present**: false
- **Tags**: triton-ascend, part-2, deep, language-layer, dual-builder, visit-call-fourth-branch, dual-builtin-marker, setup-unified-builder, with-dispatch, insertion-point, transient-failure, resume

## What happened

Part 2 语言层**首章**，kind=**deep**（含只做减法的 implementation/ 精简版 + tests）。verdict=**APPROVED**，全 linter green，5 图全部盲审 PASS。主题：昇腾在语言层的接入机关——fork 在 `python/triton/compiler/code_generator.py` 给前端 `CodeGenerator` 开的『接缝』。

**六机制主线**：
1. **m1 双 builder 构造**（code_generator.py:L215-L231）：同一 CodeGenerator 实例在同一 MLIR context 上并挂 `self.builder`(ir.builder，emit 标准 Triton IR) + `self.ascend_builder`(ascendnpu_ir_builder，emit 昇腾 hivm/ascend 方言)——不同对象、共享 context。构造末尾 `setup_unified_builder`(L228)反挂接线。
2. **m2 visit_Call 第四岔**（L1168-L1206）：基座三岔（常量折叠/JITFunction/统一 builtin 入口门 `language.core.is_builtin` 读 `__triton_builtin__` → 门外兜底 `return fn(*args,**kws)`）；fork 在进门后加第④岔 `_builder = ascend_builder if extension.is_builtin(fn) else builder`(L1179-L1183)。含 worked example：al.sub_vec_id→ascend_builder、tl_load→builder、plain_python→兜底裸调用。
3. **m3 双内建标记**（extension/core.py:L66-L90 + code_generator.py:L1179-L1183）：`@al.builtin` 一枚图章盖两印 `__triton_builtin__`+`__ascend_builtin__`（setattr 在 L82-83）；两谓词各读一个。集合关系 A(ascend builtin) ⊆ B(triton builtin)——进门才谈选路。
4. **m4 setup_unified_builder**（extension/builder.py:L32-L86）：把 ascend emit 方法作『插入点同步 wrapper』挂到主 builder。
5. **m5 WITH_DISPATCH 注册表**（L25-L31/L801-L814/L51）：模块级空 dict 被 `ASCEND_WITH_DISPATCH` update，`with al.scope` 查表分发 + `mangle_ty` override 钩子。
6. **m6 插入点/loc 接力**（L1180-L1193/L353-L365）：切 builder 前拷入插入点+loc、emit 后同步回，成对存恢复。

**瞬时故障经过（值得记）**：write r1 曾 `Connection closed`，resume 复跑成。Review round 2 部分评审 agent `Connection closed` → `review-agents-failed` 逃逸（**不假通过**）。Lead `resumeFromRunId`(wf_5e0f75da) 复跑：cached agents 秒回、失败的评审+Map 重跑、瞬时失败清除 → 正常 APPROVED。**教训：review-agents-failed 是瞬时基建失败，首选 resume 而非重发。**

**Lead 派 ch04writer 补 6 处**（non-blocking reader-comp）：L90 截短异常文本 fidelity、`_generator/generator` 命名打通、loc gloss、region/SSA gloss、simd/simt gloss、formulas 密度。

回环轮数：impl↔test 1 轮、write↔review 1 轮、blind 1 轮、map 1 轮。

## Why it matters

Part 2 语言层的**接缝基线**：后续语言层各章（al.copy/al.fixpipe 等昇腾内建、scope 语义）都从『双 builder + 第四岔选路 + 双标记 + WITH_DISPATCH』这套接线长出来。ch01 埋的『双 builder / OpBuilder→ch04』线索在此兑现。deep 章接口已登记 interfaces.json，勿改名。

## What to remember

- **诚实边界**：host 无 NPU/CANN，运行时轨迹标『需真机』；编译期锚点行号照读、已核对。
- **本章埋下**（→P2 ch08 scope 章）：`with al.scope` 引出的 **scope / region / SSA** 概念、`_generator` 回调机制——ch04 只 gloss 一句、系统展开留 ch08。本章**无 arc-map 正式伏笔埋/回收**（bible.py due ch04 空）。
- **事实校准点**（勿再回退）：双标记 setattr 在 extension/core.py **L82-83**（非 L86-87，m3 图曾误标已修）；is_builtin 入口门读 `__triton_builtin__`、extension.is_builtin 第四岔读 `__ascend_builtin__`；A(ascend)⊆B(triton)。
