# ch08 交付：作用域、核间同步与流水线提示——scope / sync_block / PIPE / compile_hint

- **Type**: delivery
- **Chapter**: ch08
- **Date**: 2026-07-21
- **Timestamp**: 2026-07-21
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, writer, reviewer, Lead, archivist
- **User present**: false
- **Tags**: triton-ascend, part-2, deep, language-layer, with-dispatch, handle-scope-with, two-pass-ssa, outline-scope, sync-block, getcore, pipe-narrowing, compile-hint, annotation-mark, part-2-finale

## What happened

Part 2 语言层第五章、也是**该 Part 的收官章**（物理章号 ch08），kind=**deep**，deps=ch02 + ch04。正文 1190 行，**11 项章级门禁全绿**（fidelity / source_grounding / structure / formulas / dossier / explainer / trace_consistency / diagrams / diagram_scaffolding / ir_opname / chapter_map --require）+ 全局四扫全绿，精简版 **63 tests passed**，**10 张图**（9 机制图 + chapter-map）blind_review 全 PASS。verdict=**APPROVED**（评审首轮 REVISE：3 blocking + 4 non-blocking，全部修完）。

**四条主线**：

**一、`with scope` 被编译器特判**。`scope` 类（`third_party/ascend/language/cann/extension/scope.py:L28-L71`）空得反常：`__enter__` 只 `return self`、`__exit__` 直接 `return False`，**一行 IR 都不发**。语义全在编译器对 `with` 的特判里——基座 `visit_With`（`python/triton/compiler/code_generator.py:L801-L813`）`assert len(node.items) == 1` 后，拿 with 项里被调用的 **`scope` 类对象本身**（不是字符串）当键查 `WITH_DISPATCH`（import 期被 `dispatch.py:L25-L34` 的 `ASCEND_WITH_DISPATCH = {scope: handle_scope_with, "mangle_ty": mangle_ty}` 注入），命中就把**整条 with 的 AST** 交给 `handle_scope_with`，没命中才当透明壳。`scope(...)` 这个调用**从未被求值**。与 ch04 的「按 `is_builtin` 选 builder」是同一路由思路的第二个入口。

**二、两趟 visit 的 SSA 穿线**（`third_party/ascend/language/cann/extension/code_generator.py:L137-L208`）。第 1 趟在 `create_block()` 造的 dummy 块里试跑块体，只为读走 `local_defs`（块内定义/改写的名字与类型），随后 `dummy.erase()`——这趟的 IR 连块带值一起作废，产物只有冻结的 `names` / `ret_types`；中间 `create_scope_op(mlir_attrs, ret_types)` 建带 region 的 `scope.scope`、`create_block_with_parent(region#0, [])` 建**无块参数**入口块；第 2 趟重置 `lscope = liveins.copy()` 后重跑块体，`scope_return(handles)` 封口，退出子区域再按同一 `names` 顺序把 `scope_op.get_result(i)` 回填外层符号表。**不变量**：设 k = `scope_defs` 大小，则结果数 = `scope.return` 操作数数 = 回填名字数 = k 且顺序一致（三处共用同一游标 `names`）。代价：块体生成两遍、丢弃一遍，嵌 N 层最内层走 $`2^N`$ 遍（纯编译期）。括号里的关键字则从 AST 直接揭（`_extract_scope_attributes` 只收 `ast.Constant`），四条翻译规则（`noinline` 默认开 / `core_mode` 走 cube-vector 两项白名单 / `disable_auto_sync` 加 `hivm.` 前缀 / 其余透传）之外有**三种静默失效**：拼错核名、用位置参数、传变量——都不报错，只是核类型声明悄悄消失（连 `list` 透传那条规则从 `with scope` 入口也够不着，`ast.List` 不是 `ast.Constant`）。`scope.scope` 活不到最后：ttadapter 段的 `outline-scope`（`.../Dialect/Scope/Transforms/Passes.td:L23-L68`）把 region 外提成带 `tcore_type` 的 `func.func`，原地只留一次 `call`——**核类型最终是函数级属性**，这也回答了 `noinline` 为何默认打开。

**三、核间同步有两代**。旧代（`aux_ops.py:L57-L96`）进门先 `DeprecationWarning`，经 `_utils.custom_op`（`_utils.py:L5-L16`，**十一行、只认三个 op 名的手写 if-elif 分发**）落成通用的 `ascend.custom`，**`receiver` 与流水线信息在语言层就丢了**；新代（`core.py:L202-L234`）补上 `sender_pipe` / `receiver_pipe`，经 `create_sync_block` 落成 `hivm.sync_block_set` / `hivm.sync_block_wait`。两代共守四条校验（核名白名单 cube/vector、`sender ≠ receiver` → `ValueError`、`0 ≤ event_id < 16`、pipe 必须是 `PIPE` 枚举实例），但**并非「新的更严」**：旧代丢信息，新代对 `event_id` 少调一次 `_constexpr_to_value`，`constexpr(99)` 因 `isinstance(int)` 不成立而绕过范围检查（同一个值反被旧代 `AssertionError` 拦住）。落核由 C++ `GetCore`（`ascend_ir.cc:L93-L113`）**按 op 名翻转**：set 落发方核、wait 落收方核，两端恒互补。pipe **两侧要么都不给、要么都给**——触发条件写死为 `and`，只给一边直接 `TypeError`；写死的缺省配对是 cube 发 `PIPE_FIX`/收 `PIPE_MTE2`、vector 发 `PIPE_MTE3`/收 `PIPE_MTE2`。一对多的 `sync_block_all` 四模式，模式名点到哪一侧哪一侧拿 `PIPE_ALL`，`all_sub_vector` 是新代独有。

**四、`compile_hint` 只贴条**（`aux_ops.py:L114-L151`）。五路类型分派后由 `annotation.mark`（`ascend_ir.cc:L597-L603`，`annotation::MarkOp`）旁挂到目标张量，原算子一个字节不动。**顺序即语义**：`bool` 必须排在假值判断前（源码注释写明 `handle False explicitly`），而整数 `0` 会掉进假值分支变成 unit 属性。**两个入口不等价**：公开入口 `compile_hint` 的 SIMT 门控（`L137-L139`）生效，`compile_hint_impl` 里同款检查**是注释掉的 FIXME**（`L115-L118`），所以直呼 impl 的 `multibuffer` 不受门控；且公开入口 `L141-L150` **按真值解包**（值为假时短路不解），已把 constexpr 拆成裸值 ⇒ impl 的 `constexpr` 分支**从公开入口不可达**。收尾点名第三个编排入口 `parallel(bind_sub_block=…)`（`aux_ops.py:L99-L111`，910B 最多 2 个 vector 核，对上 ch02 的 cube : vector = 1 : 2）。

**贯穿全章的结构性主题——收窄链第二次现形**（首见 ch05 地址空间「定义 7 → 导出 5」）：`PIPE` 在 `HIVMAttrs.td:L220-L253` 定义 **15 档**，`ascend_ir.cc:L420-L436` 的 `py::enum_` 只导出 **8 档**（掉队 7 档：`PIPE_MTE4` / `PIPE_MTE5` / `PIPE_V2` / 两个 `VIRTUAL_PIPE_MTE2_L1*` / `PIPE_NUM` / `PIPE_UNASSIGNED`，掉在 pybind 这一级；`UNASSIGNED` 取 99 是留白哨兵）；`TCoreType` 定义与导出都是 **4 档**，可 `scope(core_mode=…)` 白名单只 **2 档**（掉在语言层，`CUBE_OR_VECTOR` / `CUBE_AND_VECTOR` 从 `scope` 到不了、但注册自定义算子时填得进去）。第三种形态是 `compile_hint` 那条从公开入口不可达的 `constexpr` 分支——掉的不是枚举档位，而是一整条分派分支。纪律：**碰到枚举先数三遍，把数字记在正确的那一级上**（「PIPE 有 8 个」对 Python 与 pybind 成立，对 `.td` 不成立）。

**流水线两次逃生**：Dossier 站因环境隔离守卫（Write 被拒）逃生——analyst 已产出完整档案并自验（`lint_dossier` 绿、embed_verbatim 零不匹配），Lead 审读其 build 脚本（只读 pin、只写一个 JSON、无 subprocess）后执行落盘，再以 `skip_dossier` 续跑；Illustrate 站再次逃生——explainer.json 尚在中转区未落盘，**盲审拿不到 spec 时拒绝伪造判定**、9 张图全标 BLOCKED 并上报（处置正确），Lead 落盘四个中转目录后逐站手工推进（write / review / map 三站均由 Lead 单独派角色）。run_ledger：impl↔test 1 轮、write↔review 1 轮、**blind 6 轮**、map 1 轮。

## Why it matters

ch08 收束 Part 2：**达芬奇硬件模型里每一处与 GPU 不同的地方，最终都在语言表面长出了一个对应的关键字**——ch04 的双 builder 接缝、ch05 的显式内存层级与搬运边、ch06 的昇腾内建算子、ch07 的自带菜谱，到本章的「谁来干、何时干、顺带提醒编译器一句」。这条链是 ch02 硬件模型的语言层兑现。

同时它把 Part 3 的入口摆好了：`scope.scope` 的 region、`hivm.sync_block_*` 的核类型与流水线属性、`annotation.mark` 上的提示，统统是**半成品 IR**，要经 `outline-scope` 这类 pass 重排、外提、消化。ch09（MLIR 与 Linalg primer，kind=primer，走 `lint_paper_grounding` 成对门禁）从头讲这套 IR，随后的下降链章节要**接住**本章交出的每一样东西。

本章还外溢了一条全书级改进：writer 核 `ascend.custom` 时发现 ch07 正文与 Book Bible 三个文件同款病灶（`hivm.CustomOp` 这类「方言前缀 + C++ 类名」混写），顺藤查出全书五处、已全部订正，并新增确定性门禁 `scripts/lint_ir_opname.py` 与 Bible 词条「IR 算子名的写法约定（全书通用）」。

## What to remember

- **事实校准点（勿再回退）**：①`with scope` 的查表键是 **`scope` 类对象本身**，`__enter__`/`__exit__` 一行 IR 不发；②`aux_ops.py:L36` 的 `_utils.custom_op` 与 ch07 的 `register_custom_op` **同名不同物**（Lead brief 曾混淆，dossier 记入 `lead_brief_corrections`）；③`PIPE` 15 → 8 掉在 **pybind**，`TCoreType` 4 → 4 → 2 掉在**语言层白名单**；④缺省 pipe 触发条件是 `and`（单边给 = `TypeError`），配对是 cube: FIX/MTE2、vector: MTE3/MTE2；⑤`GetCore` 按 **op 名**翻转（set 落发方、wait 落收方）；⑥`compile_hint` 的 `constexpr` 分支从公开入口**不可达**，SIMT 门控只在公开入口生效（impl 那份是注释掉的 FIXME），故 `multibuffer` 照发不误；⑦C++ 侧 `create_annotation_mark` 的第三个形参叫 **`attrVal`**（Python 侧是 `hint_val`），**没有 `attr` 这个名字**——曾因自造此名多跑一轮盲审。
- **诚实边界**：host 无昇腾 NPU/CANN；数值表来自精简版 + FakeBuilder（枚举档数照 **pybind 真导出** 4/8/3 造，不照 `.td` 的 15 档——ch05 教训已固化进 conftest 约定）；C++ 侧未编译，`GetCore` / `GetSyncBlockModeAndPipes` 的表标注「由分支逐条读出」，`lint_explainer` 的 manual-trace warn 属预期。掉队 7 档 PIPE 与 `CUBE_OR_VECTOR`/`CUBE_AND_VECTOR` 的硬件含义**源码无依据、不编**。
- **图的两条本章规矩**（沿用价值高）：凡图上出现 IR 必须标编译阶段（ttir / ttadapter）；素材 trace 的内部 step 号不得上图（脚手架泄漏）。
- **本章无 arc-map 正式伏笔埋/回收**（`bible.py due ch08` 两清单皆空；本书 arc-map.json 至今为空数组，前向线索一律走 glossary 词条）。**已兑现的既往线索已回写 glossary**：ch03 的 `compile_hint` preview、ch04 的 `WITH_DISPATCH` / scope-region-SSA；另补写 ch06『annotation::MarkOp』与 ch07『CORE / PIPE / MODE』两条词条。新前向线索：`outline-scope` / `InlineScope` 实现、`hivm.sync_block_*` 与 `annotation.mark` 的 pass 侧消化、`parallel` 循环下降 → P3/P4/P5。
- Bible 回写：glossary +17 条（另更新 4 条既有词条）、concepts +19 条、figures +10 条、interfaces 登记 ch08 精简版签名（规范路径前缀，按真实包树）、voice-guide 取证口径补第 4 级「手工读 C++ 分支」。
