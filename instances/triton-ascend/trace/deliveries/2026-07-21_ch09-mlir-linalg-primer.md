# ch09 交付：MLIR 与 Linalg——结构化张量 codegen 的编译基础设施

- **Type**: delivery
- **Chapter**: ch09
- **Date**: 2026-07-21
- **Timestamp**: 2026-07-21T15:35:21Z
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, writer, reviewer, derivation-auditor, Lead, archivist
- **User present**: False
- **Tags**: triton-ascend, part-3, primer, mlir, linalg, progressive-lowering, structured-codegen, indexing-expression, iteration-domain, subset-by-image, tiling-fixed-point, destination-passing-style, bufferization, named-vs-generic, namedops-actual-value, hfusion, hivm, paper-grounded, part-3-opener

## What happened

Part 3 的**开篇原理章**（物理章号 ch09），kind=**primer**，deps=ch01 + ch08。正文 884 行，**11 项章级门禁全绿**（**paper_grounding `--expect-primer`——primer 章以它替代 `lint_fidelity`，本章不跑 fidelity** / source_grounding / structure / formulas / dossier / explainer / trace_consistency / diagrams 几何 / diagram_scaffolding / ir_opname / chapter_map --require）+ 全局四扫全绿，论文忠实 NumPy 参考实现 **24 tests passed**，**8 张图**（7 机制图 + chapter-map）blind_review 全 PASS。verdict=**APPROVED**。

**承重在两篇论文**，不在本仓源码：arXiv:2002.11054（MLIR: Scaling Compiler Infrastructure for Domain Specific Computation）与 arXiv:2202.03293（Composable and Modular Code Generation in MLIR）。论文包 `book/papers/ch09-mlir-linalg-primer/{meta.json,paper.md}`（459 行、222 处 §/Eq 锚点），正文用 `[MLIR §x]` / `[Linalg §x]` 引 **31 种共 117 处**。**本仓落地材料极薄且已在正文开头如实声明**：概述性文档两份共 111 行（`AscendNPU-IR/docs/.../architecture.md` 78 行 + `include/TritonToLinalg/Passes.td` 33 行），另有 `backend/compiler.py` 三处装配/透传行——只用于核实 `namedOps` 的实际取值，不算对位材料。

**两个答案。**

**一、MLIR 回答「怎么造一层 IR」**：①一刀切 IR 很成功但只有一层 → 一对孪生原则 **progressive lowering（渐进式下降）** 与「降之前别把结构丢了」；②IR 模型是 **Op → region → block → Op 的三层递归**——指令/函数/module 一律是 Op，block argument 取代 φ 节点，terminator 封口，`isolated-from-above` 是作用域屏障；③**attribute 把编译期静态信息做成 IR 里的一等数据**（仿射映射因此可以被直接读来推理，不必从低层分析里恢复）；④**方言只是命名空间分组，但允许不同方言的算子在任意层级共存**——这是渐进式下降的物理基础；⑤**ODS/`.td` 写声明、C++ 由生成器产出**（DRR 同理），本仓 `Passes.td` 即实例；⑥算子开放可扩展后 pass 靠四条复用路径写；pass manager 不绑定固定粒度，代价是没有全模块 use-def 链；文本形式完全可往返，故每趟 pass 都能单独跑（`triton-opt`）。

**二、Linalg 回答「张量计算该造成什么样的 IR」**：①**算子把索引表达式写在自己身上**（"配料单"贴在算子上）；②**迭代域是隐式的**——边界不写出来，由「迭代器必须扫过操作数全部数据」反解（本章的卷积例：5 个迭代维上界各自从 O/I/K 某一轴形状读出，合计 6,070,272）；③于是「这块循环碰哪片数据」退化成一次**求像**——**求像即子集**，卷积的 halo 就是「像宽 − tile 宽」这笔账（图上 25%(2/8) 与 50%(2/4) 两行算术）；④**tiling 的不动点**：切完循环体里仍是同一个 `linalg.conv_1d_nwc_wcf`，只是操作数变小 ⇒ 变换可以继续叠；⑤**padding/packing** 把动态边界磨平，正确性条件是一条**幺元条件**；⑥**向量化**搬运通用、计算体分五种情形，判定只读结构信息；⑦**bufferization** 把不可变张量落进内存，就地写由 **`outs`（destination-passing style）** 这条编译期约束兜底——论文从「结构化算子怎么和 `scf.for` 组合」第一性原理推出 DPS，`outs` 不是"顺手传个输出 buffer"；⑧**named op 与 `linalg.generic` 同源**——具名算子只是**省略了算子体**的写法（参考实现里 `to_generic` 只翻一个 `is_named` 标志位），论文并**刻意压小 named 算子范围**以反 ONNX/HLO 式算子增殖；⑨方法论收口：**合法性与可施加性从算子的性质与结构导出，而不是从低层 IR 的分析中恢复**；⑩**三问（Legality / Applicability / Profitability）依附于抽象层级，而层级是可以设计的**（phase-ordering 反例：融合提升时间局部性却破坏后续识别 BLAS-2/3 库实现的能力）。

**三、昇腾对位（本章唯一的落地节）**：自研方言按文档自述共 **5 个**——HFusion（**Linalg 方言的扩展集、文档明说只处理 named operation**，做硬件相对无关的优化）/ HIVM（细粒度感知 NPU 硬件细节、转低层指令）/ HACC（异构硬件抽象）/ Annotation 与 Scope（打编译提示标记，正是 ch08 那套提示信息在 IR 侧的落点）；「能加方言就不改上游」（增强优先放独立方言目录，无法隔离的才落 patch）；`bishengir-compile` **输入输出都是 MLIR**，到 `hivmc` 才转 LLVM IR 出算子二进制——这使 [Linalg §2.1]「每一步都物化在 IR 里」在昇腾侧成为可核对的事实（文档口径）。

**本章立的一条跨章硬口径——`namedOps` 的实际取值**：`Passes.td` 的 Option 默认与 `compiler.py:L96` 的签名默认**都是 `false`**，但**产出编译产物那条路只有一个装配点**（`compiler.py:L949-L951` 传 **`named_ops=True`**，经 `L157-L164` 下传到 `add_triton_to_linalg` 第三个位置），且 `ttir_to_linalg` 全仓 **7 处 Python 调用点无一例外传 `True`**（装配点 1 + `unittest/` 下 6 处）。`.td` 的默认值管的是**另一条路**——`triton-opt` 命令行单跑（9 个 lit 用例中 7 个显式写 `named-ops=True`、2 个走默认）。⇒ **全书禁写「昇腾侧默认不产 named op」**，与 HFusion 只吃 named op 的自述并不矛盾。这是经验 **exp-2026-07-21-10（声明的默认值 ≠ 路径上的实际取值）** 的正面案例。

**流水线**：workflow `wf_93c43c57-2cb` 跑到 Review 第 3 轮**逃生**——推导审计 agent 与 reader agent **双双崩在 API 错误 `Connection closed mid-response`**，primer 章不得因评审 agent 崩溃而免审通过，workflow 正确中止上报。其后 Lead 手工推进四站：多维评审 1 轮（PASS-with-fixes，11 条）→ 定点修订 → 单独派 illustrator 补绘本章地图（round 1 PASS）→ **独立**盲审（chapter-map PASS；`fig-ch09-image-is-the-subset` 首轮 FAIL——标注框标题印内部产物名 + 机制编号 = 违反 HARD RULE 3 零脚手架泄漏 → 改标题「halo 这笔账」重渲染、数字未动 → PASS）→ **补跑推导审计**（PASS-with-fixes，9 条含 **3 条 must-fix**）→ 定点修订 → 全绿。

## Why it matters

ch09 是 Part 3 的**地基**：从这章起全书进入下降链内部，而下降链的每一段都是 MLIR 的 pass 与方言。没有这一章，后面 ch10 的 `triton_adapter`、ch11 的指针算术逆向工程、以及 HFusion/HIVM 各章都会退化成「一堆看不懂的算子名」；有了这一章，[第 1 章](../artifacts/ch01-birdseye-ascend-backend/narrative/chapter.md)那张 `ttir → ttadapter → npubin` 下降链图上的每个箭头，都可以从「一次神秘的翻译」读成「一次有原则的渐进下降」。

它还把「为什么要抛弃指针模型换结构化 memref」讲成了**可推导的结论**而非口号：正是因为索引表达式写在算子身上，「碰哪片数据」才退化成求像；正是因为 tiling 有不动点，变换才叠得起来。ch01 埋下的「结构化 memref → ch09/ch10/ch11」这条前向线索在此兑现了理论侧的一半，实现侧的一半交给 ch10。

**方法论层面它留下两条教训**：①**workflow 在评审站因 API 崩溃而逃生时，绝不能把「没跑成的审计」当成「通过」**——Lead 补跑那一站即抓出 3 条 must-fix；②**声明的默认值 ≠ 路径上的实际取值**——把取值点数完（7/7）才敢下结论，这条纪律直接反证了 ch01 的一处旧表述。

## What to remember

- **`namedOps` 口径（跨章承重，勿回退）**：`.td` 默认 false、`compiler.py:L96` 签名默认 false，但编译产物路径上的唯一装配点 `compiler.py:L949-951` 传 `named_ops=True`，全仓 7/7 Python 调用点都传 True；`.td` 默认只管 `triton-opt` 命令行单跑（9 个 lit 用例 7 显式 True / 2 默认）。**禁写「昇腾侧默认不产 named op」**。
- **留给 ch10 的必答项（本章刻意不答）**：`namedOps` 的**实现语义**。已核实 `TritonToLinalgPass.cpp:L524`（namedOps 为真时张量上的 `arith` 保持合法）与 `L651`（`if (!namedOps)` 才加载 `populateElementwiseToLinalgConversionPatterns`）⇒ 真实语义是「**别把逐元素 `arith` 摊成 `linalg.generic`**」，**不是**「发射 linalg 具名算子」；全仓搜 `linalg::AddOp` / 产出 `linalg.add` **零命中**。ch09 因把实现语义留给 ch10 而未被带偏。
- **跨章缺陷（已立案）**：**ch01:L147 的 `(如 linalg.add)` 举例无据** → `artifacts/ch01-birdseye-ascend-backend/reviews/LEAD-PENDING-FIX.md`，待派 writer 小修。另：`outline-final.json` 6 处陈旧 `chNNb` 依赖 id（已修）；vllm-ascend ch37 三处图面脚手架泄漏（已立案，跨实例）。
- **数字锚点**：论文包 459 行 / 222 锚点，正文引 31 种共 117 处；本仓落地材料 111 行（78 + 33）；自研方言 5 个；迭代域例合计 6,070,272；halo = 像宽 − tile 宽（25% = 2/8、50% = 2/4）。
- **诚实边界**：昇腾侧对位材料极薄且已在节首挂「证据强度声明」，凡「昇腾为什么这样做」的因果一律标为类比或留给后章；三条明确不下结论的问题：`ttadapter` 怎么把指针张量变结构化（ch10/ch11）、`namedOps` 打开后改变哪些算子形态（ch10）、昇腾 tiling/融合是否与论文同一套机制（形式相似 ≠ 机制相同，待 HIVM 章据源码定论）。
- **Lead 派工失误两则（run-ledger `lead_brief_errors`）**：①M3（padding 幺元消歧）派工表述有误 → writer 把反例**归因写错**（把「补乘法幺元 1 会错 18」挂到 $K$ 侧残留值 3 那一行，按朴素规则实际错 6），由补跑的推导审计抓出并修正；②Lead 一度把图的 overflow 警告误判为 BLOCKING（真正的阻断项是孤儿图），已向 illustrator 更正。
- **本章无 arc-map 伏笔埋/回收**（`bible.py due ch09` 两清单皆空；本书 arc-map.json 至今空数组，前向线索一律走 glossary 词条）。已核对并更新两条前指词条：`triton_adapter`（"系统性展开在 ch10"——仍准确）、「结构化 Linalg / memref」（"数学根基见原理篇 ch09"——已兑现，改写为回指）。
- Bible 回写：glossary +14 条（另更新 3 条既有词条）、concepts +30 条、figures +8 条；interfaces **不新增**（primer 章无精简版接口，参考实现不进接口注册表）。
