# MLIR 与 Linalg：结构化张量 codegen 的编译基础设施——忠实精读

> 本文件是 ch09 原理章（kind=primer）的**真相源**，供 writer 叙事、illustrator 出图、implementer 写论文忠实参考实现。
>
> 出处只有两类，读者/writer 须能逐一回溯：
> - **[MLIR §x]** = *MLIR: A Compiler Infrastructure for the End of Moore's Law*，**arXiv:2002.11054**（v2，2020-03-01；共 21 页）。
> - **[Linalg §x]** = *Composable and Modular Code Generation in MLIR: A Structured and Retargetable Approach to Tensor Compiler Construction*，**arXiv:2202.03293**（v1，2022-02-07；共 43 页）。
> - **[src: <path>:Lxx]** = 本仓源码内文档/`.td` 的逐字引文（triton-ascend v3.2.1 pin @ `2badfc89e`）。规范路径前缀 `third_party/ascend/...`，**绝不**写 `instances/.../source/`。
>
> **两篇论文均由 analyst 于 2026-07-21 联网取回 PDF 全文并逐页读取**，不是凭记忆默写。英文引文取自 PDF 抽取文本，只做了两处无损整理：还原 PDF 连字（`ﬁ`→`fi`、`ﬂ`→`fl`）、修复跨行断字；其余逐字未改。
>
> **本章边界**：只讲编译基础设施本身。达芬奇硬件形态见 ch02，昇腾后端全景见 ch01，triton_adapter 的具体 pass 见 ch10 起。§10 的昇腾对位在本仓证据**很薄**（两份文件合计 111 行），已就地标注。

---

## §0 读法与出处约定

这一章是**先修课**，不是源码解读课。往下读 Part 3、Part 4 时，读者会看到 `ttadapter` 把 Triton IR 变成 `linalg` 算子、看到 `bishengir-compile` 把 Linalg 变成 HFusion 再变成 HIVM。如果没有 MLIR/Linalg 的心智模型，那些 pass 名字只是黑话；有了，它们会变成「同一套设计原则在昇腾硬件上的一次具体实例化」。

本章的承重是**两篇论文**，昇腾侧材料只做对位锚点。凡本包写下的因果、定义、断言，后面都跟着 `[MLIR §x]` 或 `[Linalg §x]`；找不到出处的话不写，拿不准的进 §11 的 open_question。**特别地：凡「共 N 个/N 种」这类计数，只在论文自己给出数字或自己列完编号清单时才写，并注明是哪个口径。**

---

## §1 为什么需要 MLIR：一刀切 IR 与 N 个自建 IR 的两难

MLIR 论文开篇的问题意识非常朴素：编译器领域已有 LLVM、JVM 这样成熟的复用平台，但它们都是**单一抽象层**的：

> [MLIR §1] "A common characteristic of these popular systems is their “one size fits all” approach—a single abstraction level to interface with the system: the LLVM Intermediate Representation (IR) is roughly “C with vectors”, and JVM provides an “object-oriented type system with a garbage collector” abstraction."

单层抽象的代价，是**每个领域各自造 IR**。论文点名：Swift、Rust、Julia、Fortran 各自发展了自己的 IR，用来做语言/库专属优化、流敏感类型检查，以及改善下降过程的实现 [MLIR §1]；机器学习系统则普遍拿「ML 图」当自己的领域专用抽象 [MLIR §1]。而造 IR 的工程成本高得离谱，基础设施质量往往不是首要目标，于是用户看到的就是编译慢、实现有 bug、诊断质量差、优化后代码难调试 [MLIR §1]。

MLIR 的解法写在 §1 里，是三件事：

> [MLIR §1] "MLIR does this by (1) standardizing the Static Single Assignment (SSA)-based IR data structures, (2) providing a declarative system for defining IR dialects, and (3) providing a wide range of common infrastructure (including documentation, parsing and printing logic, location tracking, multithreaded compilation support, pass management, etc)."

论文还交代了它的来处：MLIR 起于一个观察——现代 ML 框架其实是「一堆互不共享基础设施的编译器、图技术与运行时」拼起来的（[MLIR §1.2]，Fig.1 画的正是 TensorFlow 模型执行横跨 TensorFlow Graph / XLA HLO / TensorRT / nGraph / Core ML / TF Lite / LLVM IR / TPU IR 等一大片各行其是的系统），Fig.2 则画出另一半困境：Swift/Rust/Julia 等语言各自有 AST 与中间 IR，最后才汇到 LLVM IR。

**给 writer 的三层递进**：① 先讲「一刀切 IR 很成功，但只有一层」；② 再讲「于是每个领域自己造 IR，重复造轮子且质量参差」；③ 落到「MLIR 不提供一个更好的 IR，它提供**造 IR 的基础设施**」——这句是整章的题眼，也是读者理解「为什么昇腾能自己长出 HFusion/HIVM 两个方言」的钥匙。

---

## §2 七条设计原则：渐进式下降与「维持高层语义」

[MLIR §2] 依次给出设计原则。**通读 §2 全节所见为 7 条**（论文以加粗小标题分段，未自行编号，以下计数是本 analyst 通读 §2 后的清点，口径已注明）：

1. **Little builtin, everything customizable**——系统只保留极少的基本概念，其余全部可定制：「类型、操作、属性」这三个 IR 里最常见的抽象，要能用来表达其他一切 [MLIR §2]。成功判据是能表达从 ML 图、AST、多面体这类数学抽象，到 CFG 与 LLVM IR 这类指令级 IR，且**不把这些抽象的概念硬编码进系统** [MLIR §2]。
2. **SSA and regions**——SSA 让数据流分析简单而稀疏；但表达高层抽象需要把**嵌套 region** 作为 IR 的一等概念。论文明说这是有代价的取舍：「我们打算牺牲 LLVM 的规范化（normalization），有时还有正规形式（canonicalization）属性」，换来「同一个循环嵌套，按你这一趟 pass 的需要，既可以看成嵌套 region、也可以看成线性化控制流」的选择权 [MLIR §2]。
3. **Progressive lowering**（本章后面反复用到的那条）：

   > [MLIR §2] "The system should support progressive lowering, i.e. from the higher-level representation down to the lowest-level, with the lowering being performed in small steps along multiple abstraction levels."

   论文把它和既有做法对照：Open64 的 WHIRL 表示有五个层级，Clang 也是从 AST 逐级下到 LLVM IR → SelectionDAG → MachineInstr → MCInst——但**这些层级是写死的（in a rigid way）**，而可扩展性要求更灵活的设计 [MLIR §2]。它还有一个深远后果：pass 的角色被归为四类——优化变换、使能变换、下降、清理——而系统应当允许**在单个 operation 的粒度上**混搭这四类角色，而不是在整个编译单元上排 pass 顺序 [MLIR §2]。
4. **Maintain higher-level semantics**——系统必须保留分析与优化所需的高层语义与结构：

   > [MLIR §2] "the system should maintain structure of computation and progressively lower to the hardware abstraction. The loss of structure is then conscious and happens only where the structure is no longer needed to match the underlying execution model."

   论文举的例子就是循环结构：一旦丢掉它、退回 CFG 形式的控制流，「实质上意味着这一层不会再做任何变换了」[MLIR §2]。它的推论是**混合抽象层级**：同一份 IR 里，一部分保持高层、另一部分已经下降 [MLIR §2]——这一条直接为 §5.2 的「方言混合」和 Linalg 论文的整条流水线铺路。
5. **IR validation**——开放的生态必须配套广泛的验证机制，且要尽可能声明式、成为单一真相源 [MLIR §2]。
6. **Declarative rewrite patterns**——定义「表示的修改者」应当和定义新抽象一样简单；常见变换应能以声明式的、机器可分析的重写规则表达 [MLIR §2]。
7. **Source location tracking and traceability**——一个 operation 的来处（原始位置与所经变换）应当在系统内可追溯，用以对付复杂编译系统的「不透明」问题；论文特别提到安全攸关与密码学场景下，优化可能改变甚至完全废掉源程序里的防护（WYSINWYX 问题）[MLIR §2]。

**读到这里该记住的一句话**：MLIR 的设计原则里，**「渐进式下降」与「维持高层语义」是一对**——前者说「一步一步降」，后者说「降之前别把结构丢了」。Linalg 那篇论文（下文 §6 起）整篇就是这一对原则在张量计算上的兑现；昇腾的 `ttadapter → HFusion → HIVM` 三段，也是这一对原则的一次具体实例化。

---

## §3 IR 模型：op / region / block / attribute / type / dialect

[MLIR §3] 把 IR 的构件逐个定义。这一节是本章最基础、也最该让读者背下来的部分。

### 3.1 Operation：语义的唯一单位

> [MLIR §3] "The unit of semantics in MLIR is an “operation”, referred to as Op. Everything from “instruction” to “function” to “module” are modeled as Ops in this system. MLIR does not have a fixed set of Ops, but allows (and encourages) user-defined extensions—compiler passes treat unknown Ops conservatively"

一个 Op 有唯一的 opcode（文本上是「方言名.操作名」的点分前缀 [MLIR §3]）、零或多个 operand 与 result（都保持 SSA 形式、都有类型），此外还可以带 attribute、region、block argument 与 location 信息 [MLIR §3]。

**「未知算子按保守方式处理」这半句很重要**：它是「任何人都能加算子而不破坏既有 pass」的底座，也是昇腾能往 MLIR 生态里塞自研方言的前提。

### 3.2 Attribute：编译期静态信息

> [MLIR §3] "An MLIR attribute is structured compile-time static information, e.g., integer constant values, string data, or a list of constant floating point values. Attributes are typed, and each Op instance has an open key-value dictionary from string names to attribute values."

论文用 Fig.4 里的 `affine.for` 举例：循环上下界与步长是 attribute，写作 `{lower_bound = () -> (0), step = 1 : index, upper_bound = #map3}`；其中 `() -> (0)` 是内联的仿射式（affine form），`#map3` 是**属性别名**——先给某个属性值起个标签，之后凡需要属性值的地方都能用这个标签 [MLIR §3]。

**记住这条**：仿射映射在 MLIR 里是**属性**，不是某种特殊语法。Linalg 的索引映射之所以能被编译器直接读来推理，正因为它是 IR 里一等的、结构化的编译期数据（详见 §7）。

### 3.3 Region 与 Block：递归嵌套

> [MLIR §3] "An instance of an Op may have a list of attached regions. A region provides the mechanism for nested structure in MLIR: a region contains a list of blocks, and a block contains a list of operations (which may contain regions). As with attributes, the semantics of a region are defined by the operation they are attached to"

三层递归（Op → region → block → Op）正是 **[MLIR Fig.3]** 画的东西，也是本章 illustrator 要重绘的第一张关键图。要点：

- region 里的 block 之间构成一张 CFG；每个 block 以 **terminator** 结尾，terminator 自己定义控制流转移语义 [MLIR §3]。
- MLIR **不用 φ 节点**，而用 SSA 的函数式形态：terminator 把值传给后继 block 的 **block argument** [MLIR §3]。`affine.for` 就是拿入口 block 的参数当循环归纳变量 [MLIR §3]。
- **可见性**：region 内的 op 可以用「词法上位于该 region 之外且在其上方」定义的值；但被标为 **isolated from above** 的 op 是作用域屏障（如 `std.func`）——好处是「一个含 isolated-from-above 算子的 module 可以被 MLIR 并行处理，因为 use-def 链不会跨越隔离边界」[MLIR §3]，这条正是 §4.3 并行编译的前提。
- **符号表**：Op 可以挂 symbol table，把字符串名字关联到 IR 对象；符号不必遵守 SSA（可以先用后定义），所以才可能表达递归函数、全局变量、具名 module [MLIR §3]。

### 3.4 Dialect：分组机制，也是共存机制

> [MLIR §3] "MLIR manages extensibility using Dialects, which provide a logical grouping of Ops, attributes and types under a unique namespace. Dialects themselves do not introduce any new semantics but serve as a logical grouping mechanism"

论文强调方言只是**逻辑分组**，类比「设计一组模块化的库」；把所有东西塞进一个方言在技术上可行，但概念太多、名字冲突，很快会失控 [MLIR §3]。而最关键的一句是共存：

> [MLIR §3] "Although each Op, type and attribute belongs to exactly one dialect, MLIR explicitly supports a mix of dialects to enable progressive lowering. Ops from different dialects can coexist at any level of the IR at any time, they can use types defined in different dialects, etc."

**「不同方言的算子可以在任意时刻、任意层级共存」——这是渐进式下降在 IR 层面的物理基础。** 没有它，「一步一步降」就只能退化成「一次性全量翻译」。

### 3.5 类型系统与内建方言

类型是编译期语义：每个值都有类型，类型系统用户可扩展，甚至能引用外部类型系统（`llvm::Type`、`clang::Type`）；MLIR **强制严格类型相等检查、不提供隐式类型转换规则** [MLIR §3]。论文还给了一句常被忽略的说明：MLIR 只支持非依赖类型（trivial、参数化、函数、和类型、积类型）[MLIR §3]。

最后是一条对读者心智很有帮助的事实：**函数与 module 不是新概念，它们就是 builtin 方言里的 Op** [MLIR §3]——module 是「带一个 region、region 里一个 block」的 Op，function 是「带一个 region、region 参数即函数参数」的 Op [MLIR §3]。

---

## §4 声明式基础设施：ODS(TableGen)、DRR、pass manager、verifier、可往返文本形式

[MLIR §4] 讲的是「IR 之外的那一半」——定义方言/算子/重写/验证/pass 的工具。这些正是读者在本仓 `.td` 文件里会亲眼看到的东西。

### 4.1 ODS：用 TableGen 声明式地定义算子

> [MLIR §4.1] "MLIR uses TableGen-based [47] specification for Operation Descriptions (ODS), defining the structure of an Op and components of its verifier declaratively. TableGen is a data modeling tool intended to help define and maintain records of domain-specific information, used extensively in LLVM."

一条 ODS 定义包含：唯一名字、一串描述算子性质的 **trait**、一串 argument（operand 与 attribute）、一串 result；argument/result 各有名字与类型约束（如「静态形状的 float 或 int32 张量」）；还可以写人类可读的描述用于生成文档，以及（有限的）自定义文本形式 [MLIR §4.1]。当 ODS 表达力不够时，可以经 builder / printer / parser / verifier 子句注入额外 C++ 代码 [MLIR §4.1]。最后：

> [MLIR §4.1] "The ODS definition is ultimately translated into C++ code (including Op classes with named accessors, verification, etc.) which interoperate with the rest of the system."

**这就是读者在本仓看到 `.td` 文件的原因**——[src: third_party/ascend/AscendNPU-IR/docs/source/en/introduction/architecture.md:55] 逐字写道：「The include directory holds declaration files, including C++ headers (.h, .hpp) and TableGen definition files (.td); the include directory under the build directory also contains code generated by TableGen (.h.inc, .cpp.inc).」——`.td` 写声明、TableGen 生成 `.h.inc/.cpp.inc`，与 [MLIR §4.1] 描述的机制一致。

本仓最短的一份 `.td` 样本正好可以给读者当第一眼（33 行全文的核心两段）：

> [src: third_party/ascend/include/TritonToLinalg/Passes.td:6-15]
> ```tablegen
> def TritonToLinalg : Pass<"triton-to-linalg", "mlir::ModuleOp"> {
>     let summary = "Convert Triton to Linalg dialect";
>     let constructor = "triton::createTritonToLinalgPass()";
>     let options = [
>         Option<"globalKernel", "global-kernel",
>             "bool", /*default*/"true",
>             "generate a global kernel">,
>         Option<"namedOps", "named-ops",
>             "bool", /*default*/"false",
>             "use linalg named ops instead of linalg.generic">,
> ```

注意这里声明的是 **pass**（不是算子）——TableGen 在 MLIR 生态里被同一套「声明写在 `.td`、C++ 由生成器产出」的方式复用到 pass 定义上。这段的 `namedOps` 选项还会在 §7.3 再出现一次：它正是 Linalg 论文「named op 只是 generic op 的声明式配置」在本仓的落点。

### 4.2 DRR：声明式重写规则

DRR（Declarative Rewrite Rule）与 ODS 一样是嵌进 TableGen 的 DSL，用来表达源 DAG 与目标 DAG 的模式、约束（含动态约束）与优先级收益；概念上，**DRR 表达的是「在给定约束下两个 DAG 等价」** [MLIR §4.2]。论文的 Fig.6 就是把 `LeakyReluOp` 重写成「比较 + 选择」的一条规则。DRR 最终也翻成 C++，可以和直接用通用图重写框架写的复杂 pattern 混用 [MLIR §4.2]。

### 4.3 Pass manager：不绑定固定粒度，且能并行

> [MLIR §4.3] "Whereas pass management in existing systems is typically defined over a fixed granularity (e.g., module, function or loop pass managers), in MLIR modules and functions are not special—they are merely Ops with regions and there can be multiple variants of them. Therefore, the MLIR pass manager is also not specialized on a fixed set of ops, but instead works on arbitrary Ops at arbitrary levels of nesting."

并行编译由 §3.3 提到的 isolated-from-above 保证：这类 op 定义了一棵可并行处理的 region 树 [MLIR §4.3]。论文并点出代价：**正因如此，MLIR（与 LLVM 不同）没有全模块范围的 use-def 链**，全局对象要经符号表引用，常量则实现为带属性的 operation [MLIR §4.3]。

### 4.4 可往返的文本形式

IR 有完全反映内存表示的文本形式，对调试、理解变换过程、写测试都至关重要 [MLIR §4.4]。两种形式（通用形式与自定义形式）**完全可往返**，每个 pass 都能单独测试、以文本作输入输出；因为没有隐藏状态，「单跑一个 pass 的结果与在完整流水线里跑同一个 pass 的结果相同」[MLIR §4.4]。

**这条是本书读者的实操福利**：Part 3/4 里所有「dump 出中间 IR 看一眼」的做法，正建立在这个属性上。

### 4.5 文档与 4.6 验证器

文档由 ODS 描述生成，与验证代码同源，因而更容易与运行时行为保持同步 [MLIR §4.5]。验证器先查全局结构性质——类型必须精确匹配、值只定义一次且满足支配与可见性、符号名在符号表内唯一、所有 block 以 terminator 结尾——然后再跑各算子与属性自己的验证器；**验证失败被视为不变量被破坏，直接中止编译** [MLIR §4.6]。

---

## §5 可复用的 pass：trait / hook / interface / 方言专属，以及方言混合

[MLIR §6.1] 直面一个尖锐问题：算子和类型都开放可扩展，那 pass 还怎么写？论文原话是「我们发现了四条主要途径」（four major approaches），逐条如下：

1. **基础算子 trait**——DCE、CSE 这类「家常」pass 只依赖很简单的性质（「无副作用」「可交换」），这些性质定义为 Op trait，由 ODS 里的算子作者声明，pass 于是能跨抽象域通用 [MLIR §6.1]。
2. **特权算子 hook**——有些性质一个 bit 表达不了、需要 C++ 实现（如常量折叠逻辑）。比 folding 更有意思的是 `getCanonicalizationPatterns`：它让算子作者声明适用于自己的折叠模式，从而支撑一个**能施加到所有方言**的通用 “Canonicalization” pass——论文说这一个可扩展机制吃掉了 LLVM 生态里 InstCombine / DAGCombine / PeepholeOptimizer / SILCombine 等一堆专用 pass 所做的事，而那些正是众所周知的维护负担 [MLIR §6.1]。
3. **优化接口（Optimization Interfaces）**——论文举内联器为例：内联器想同时服务 TensorFlow 图、Flang 函数、函数式语言的闭包，可它根本不知道什么是调用点、什么是被调用者。解法是把它需要知道的两件事抽成接口（Fig.10 的 `DialectInlinerInterface`：某算子能否被内联进某 region、内联后落在块中间的 terminator 怎么处理），由各算子/方言自行注册实现；**没实现接口的算子，对应优化 pass 就保守对待** [MLIR §6.1]。
4. **方言专属 pass**——完全可以写只服务某个方言的 pass，由该方言算子的完整语义驱动；在不需要泛化时，这是新变换简单有用的起点 [MLIR §6.1]。

[MLIR §6.2] 补上另一半，也是论文自称「最深刻但也最难领会」的一点：**MLIR 允许并鼓励把不同方言的算子混在同一个程序里**。仿射方言就是例子——仿射控制流与仿射映射的定义，与 region 里那些算子的语义**无关**，于是仿射方言可以和表示目标无关算术的 “standard” 方言、以及多个目标专属机器指令方言组合使用 [MLIR §6.2]。论文说这种「用 op interface 拿到具体算子语义，从而复用通用多面体变换」的方式，是一种**在其他系统中没见过的复用形态** [MLIR §6.2]。

顺带一提论文给的**互操作范式** [MLIR §6.3]：要接一个外部系统，就定义一个尽可能直接对应它的方言，让格式能简单可预测地来回往返；IR 进了 MLIR 之后，再用 MLIR 的全套设施升/降到更方便的表示。——这一条读者在 Part 3 会直接用上：`ttadapter` 面对的正是「把另一个系统（Triton 的 IR）接进 MLIR 生态」这个问题。

### 5.1 论文当年的采纳度（数字口径注明）

[MLIR §5] 给了 2020 年那个时间点的社区快照，本包**照抄不外推**：一个 MLIR 用于 HPC 的学术研讨会有来自 16 所大学的与会者、涉及 4 个国家的 4 家国家实验室；MLIR 被 14 家跨国公司认可；LLVM 开发者大会上超过 100 位工业界开发者参加了 MLIR 圆桌；并且——

> [MLIR §5] "More than 26 dialects are in development in public or private and 7 projects across different companies are replacing custom infrastructure with MLIR."

**这些是 2020 年的数字，不代表今天**（今天 MLIR 上游方言远不止此数，但本包不给未经核实的当代数字）。

---

## §6 结构化 codegen：为什么把变换搬到张量层

第二篇论文接着 MLIR 往下走一层：有了造 IR 的基础设施，**张量计算该造成什么样的 IR**？

### 6.1 问题：传统 codegen 从循环开始，太晚了

> [Linalg §2.1] "Code generation approaches for numerical computing have traditionally focused on optimizing the performance of loop nests. Associated analyses focus on scalar elements as the body of a loop nest typically computes a single element. Such analyses must consider memory dependences and aliasing."

论文承认这条路很成熟，也承认它适用的前提：当输入语言是 C 或 Fortran 时，问题**本来就是**以「在预分配内存上跑的循环」形式给出的 [Linalg §2.1]。但在 ML 这类领域，我们有奢侈的起点——程序本来就定义在远高于循环的抽象层：

> [Linalg §2.1] "This opens up the opportunity to revisit classical loop optimizations like fusion, tiling or vectorization without the need for complicated analysis and heuristics. Advantages include reduced complexity and maintenance cost while also scaling naturally to extensions like sparse tensors, that are even more difficult to analyze at the loop level."

于是有了名字：

> [Linalg §2.1] "We refer to this approach as structured code generation since the compiler primarily leverages structural information readily available in the source code."

**一句话：结构化 codegen 不是「更聪明地分析循环」，而是「在还没退化成循环之前就把变换做完」。** 论文在 §1 里把病因说得更直白：过快跨越抽象鸿沟会 (1) 丢掉高层 IR 上本来就有的信息，(2) 因为要从低层 IR 重建高层语义而加剧 phase-ordering 问题 [Linalg §1]。

### 6.2 全景流水（Fig.1，本章第二张关键图）

[Linalg §2.1] 给出的整条流水（对应 **[Linalg Fig.1]**）：

| 层级 | 内容 | 论文原话要点 |
|---|---|---|
| Structured IR | 稠密/稀疏张量上的张量代数算子，组织成函数式程序 | "composed of tensor algebra operations, organized as a functional program over dense and sparse tensors" [Linalg §2.1] |
| Tiled structured | tiling 引入循环；**可多级、渐进地 tiling** | "tiling produces loops around structured operations similar to the original ones but on smaller tensors. We also perform fusion of tensor operations at this level." [Linalg §2.1] |
| 向量抽象 | 把小张量上的计算映射到可重定向的向量抽象 | "This mapping exploits high-level knowledge about the operations that we have carefully preserved. In particular, it is not required to analyze the loop control flow enclosing the finer-grained tensor operations." [Linalg §2.1] |
| Buffer 层 | 从不可变张量值下降到有副作用的 buffer，得到「嵌套循环 + 向量 + 副作用」 | "We lower this to a representation on side-effecting buffers in the next step." [Linalg §2.1] |
| 目标层 | 直接翻到 `llvm` 方言跑 CPU、或 offload GPU kernel、或切成异步块交给任务并行运行时 | [Linalg §2.1] |

两条**必须让读者带走**的判断：

- **tiling 的粒度选择服务于硬件映射**：论文给的原型例子是「先按 cache 层级 tiling 矩阵乘，再把切小后的矩阵乘直接下降到汇编写的超优化 microkernel」[Linalg §2.1]。
- **可组合性来自泛型**：「tiling 与 fusion 变换在它们所作用的算子与数据类型上都是完全泛型的……它们只假定计算与复合数据上存在一种泛型的、（就集合包含而言）单调的结构化分解模式」——稠密与稀疏张量代数都具备这种分块分解模式，于是同一套 codegen 抽象泛型地适用于两者 [Linalg §2.1]。

论文还坦白了这套流水的**可选性（optionality）**：对某些算子，跳过某些层级、甚至走一条完全不同的路，都是可行选项；而这之所以可能，正是因为「每一步都物化在 IR 里，几乎没有承重逻辑被藏在编译器内部复杂的 C++ 分析与启发式里」[Linalg §2.1]。论文并不声称覆盖 ML/HPC 的全部计算：「不是所有问题都要在同一个抽象里解决，而应为每一类问题用最合适的抽象」[Linalg §2.1]。

### 6.3 方言栈：linalg 在哪一层

[Linalg §2.3] 按抽象层级由低到高列出与 codegen 相关的方言，**共 7 个（§2.3.1–§2.3.7，论文自己的编号清单）**：`vector`、`gpu`、`memref`、`tensor`、`scf`、`linalg`、`sparse_tensor`。几个对本书后续最要紧的定义：

- **`memref`**：MLIR 里 n 维内存 buffer 的主表示，是进入「有副作用的内存操作」的入口。关键性质是**索引方案与底层存储解耦**：「与传统指针不同，memref 是带显式 layout 的多维 buffer」——论文给的例子是 `memref<10x10xf32, strides: [1,10]>`，存储是 row-major、访问却是 column-major [Linalg §2.3.3]。
- **`tensor`**：抽象的 n 维张量类型，**还没决定内存表示**；张量值不可变、遵守 def-use SSA 语义，于是 peephole、CSE、DCE、循环不变量外提这些经典变换可以无差别地施加到张量算子上 [Linalg §2.3.4]。因为不可变，「写入」由「值插入」类算子表达——产生一个新张量，其中某个值或某个子集被替换 [Linalg §2.3.4]。
- **`scf`**：结构化控制流，提供 `scf.for` / `scf.while`（无早退）与 `scf.if`，嵌进 MLIR 的 SSA+region 形态；比 CFG 高一层，且 **`scf` 循环可以 yield SSA 值** [Linalg §2.3.5]。
- **`linalg`**：

  > [Linalg §2.3.6] "The linalg dialect provides higher-level compute primitives that can operate on both tensor and memref containers. These primitives can decompose into versions of themselves operating on structured subsets of the original input data and producing similarly structured subset of their results. They also capture program invariants and structural information, such as independency of certain parts of computation or reduction patterns, that decreases the need for expensive analyses prior to transformation."

**把上面这段拆成三个可检验的性质，让读者记住 structured op 的定义**：① 同时作用于 `tensor` 与 `memref` 两种容器；② 能分解成「自己的小号版本」，作用在输入的结构化子集上、产出同样结构化的结果子集；③ 自带程序不变量与结构信息（哪些部分互相独立、哪些维度是归约），**从而减少变换前的昂贵分析**。

---

## §7 structured op 解剖：索引表达式、隐式迭代域、named vs generic、destination-passing style

这一节回答大纲的核心提问：**「结构化」到底结构在哪里，为什么它让编译器不必做模式匹配就能推理循环嵌套与访存。**

### 7.1 算子自带索引表达式

[Linalg §3] 一上来就以 `linalg.conv_1d_nwc_wcf` 为例，给出它对应的**索引记法**——一个 5 维矩形迭代域作用在 3 维张量上：

```math
O[n, w, f] \; = \; I[n, w+kw, c] \cdot K[kw, c, f]
```

（[Linalg §3]，PDF p.8 的未编号索引式；论文原文用 `.` 表示乘并省略了累加号，本包按论文写法照录，符号含义见下表。）

| 符号 | 含义（[Linalg §3]） |
|---|---|
| $`O`$ | 输出张量，此例类型 `tensor<1x988x64xf32>` |
| $`I`$ | 输入张量，`tensor<1x990x32xf32>` |
| $`K`$ | 卷积核张量，`tensor<3x32x64xf32>` |
| $`n, w, f`$ | 输出侧三个迭代维（batch、空间、输出通道） |
| $`kw, c`$ | 核宽与输入通道两个迭代维 |

### 7.2 迭代域是隐式的，而且可推导

紧接着的这句是整个 structured op 思想的**技术核心**：

> [Linalg §3] "The iteration domain is implicit in the operation description and is such that the iterators span the entire data of the operands."

也就是说，**没人手写循环边界**：迭代域由算子描述隐含地确定，且要求迭代器恰好扫过操作数的全部数据。此例的迭代域就是这组不等式：

```math
0 \le n < O.0, \qquad 0 \le w < O.1, \qquad 0 \le f < O.2, \qquad 0 \le kw < K.0, \qquad 0 \le c < K.1
```

其中 $`O.d`$ 表示 $`O`$ 的第 $`d`$ 维尺寸 [Linalg §3]。论文注明这些量的推导「遵循与 Tensor Comprehensions 相同的规则」，且**在稠密情形下可由连续施加 Fourier-Motzkin 消元过程得出** [Linalg §3]。

于是 tiling 需要的「这一块循环对应每个张量的哪一片数据」就不再是分析问题，而是一次**代数计算**：

> [Linalg §3.1] "The derivation of dense subsets is obtained by computing the image of the iteration domain by the indexing function for each tensor."

**用一句话讲给读者**：迭代域是定义域，每个张量的索引函数是一个映射；把（切过的）迭代域**打到**某个张量上取像，就得到该张量要访问的那一片。传统编译器要靠依赖分析与别名分析回答的问题，在这里退化成「求像」。这就是「无需模式匹配就能推理循环嵌套与访存」的确切含义。

向量化那一步同样吃这份索引信息：

> [Linalg §3.3] "The vector.transfer operations are indexed following the indexing expressions of the linalg operation. This behavior is generic for the data movement part of all linalg operations."

**术语诚实提示（重要，见 §11 open_question 1）**：论文里用的措辞是 “indexing function / indexing expressions”，以及作为 MLIR 属性的仿射映射（`affine_map<...>`，如 [Linalg §3] Fig.4 的 `#id3d = affine_map<(d0,d1,d2)->(d0,d1,d2)>`、[Linalg §3.3] Fig.6 的 `#proj_012 = affine_map<(d0,d1,d2,d3) -> (d0,d1,d3)>`）。而 `indexing_maps` / `iterator_types` **这对属性名，在两篇论文里只出现在 `vector.contract` 上**：

> [Linalg §3.3, Fig.6] `%17 = vector.contract{indexing_maps=[#proj_012,#proj_32,#proj_012], iterator_types=["parallel","parallel","parallel","reduction"]}`

——注意 `iterator_types` 的取值 `parallel` / `reduction`，正是 §6.3 说的「哪些维度独立、哪些是归约」这类结构信息的具体形态。writer 若要以 `linalg.generic` 的 `indexing_maps`/`iterator_types` 之名讲解，**须说明这是 MLIR 实现层的属性名**，论文口径是索引表达式与迭代器类型；语义有论文支撑，名字须另找出处。

### 7.3 named op 与 generic op：同一个东西的两种穿法

每个 linalg 算子都有一个**算子体（body region）**，表达该计算的标量形式：

> [Linalg §3.3] "Under the hood, every linalg operation has a body region that expresses the scalar form of the computation."
>
> [Linalg §3.3, 脚注 6] "The body may be printed explicitly (when expressed in linalg.generic form) or simply elided when expressed in \"named\" form (such as linalg.conv_xxx)."

也就是说 `linalg.matmul`、`linalg.conv_1d_nwc_wcf` 这些**具名算子只是省略了算子体的写法**，语义上仍归结到 `linalg.generic`。论文在与 ONNX 的对比里把这条设计意图写死：

> [Linalg §5.1] "We also minimize the range of \"named\" operations by making them simple declarative configurations of a small set of generic operations. Compared to a typical numerical library interface (the de facto standard until now), this greatly reduces maintenance burden and increases portability."

论文并点名批评了反面：ONNX 与 XLA 的 HLO 都出现了**算子增殖（operation proliferation）** 问题——HLO「演化成一大批语义互相交叠的算子」[Linalg §5.2]。

**落到本仓的两个真实锚点**（都很薄，别撑成主证据）：

- [src: third_party/ascend/AscendNPU-IR/docs/source/en/introduction/architecture.md:17] "The HFusion (Hybrid Fusion) dialect is an extension set based on the MLIR community Linalg dialect. The HFusion dialect inherits all operations of the Linalg dialect and extends with operations not yet supported by the Linalg community. **Note that the operations handled by the HFusion dialect are all named operations, so that high-level semantics are preserved as much as possible for the compiler to process.**"（粗体为本包所加）——昇腾在这里做了一个明确的选择：**HFusion 层只处理 named op**，理由与论文一致：尽量保住高层语义。
- [src: third_party/ascend/include/TritonToLinalg/Passes.td:13-15] `Option<"namedOps", "named-ops", "bool", /*default*/"false", "use linalg named ops instead of linalg.generic">`——`ttadapter` 侧把「产 named op 还是产 generic」做成了一个**默认关闭**的开关。这两条摆在一起有张力（一个说「HFusion 处理的都是 named op」，一个说「默认不产 named op」），但**本包不给因果解释**：该选项在实现里如何被尊重、默认路径下算子的真实形态如何，是 ch10 读 `TritonToLinalgPass.cpp` 的事（见 §11 open_question 3）。

### 7.4 Destination-passing style：为什么算子要带一个「输出张量」参数

张量不可变，那「就地更新」怎么办？答案是 bufferization 阶段的一条启发式，其前提是算子写成 **destination-passing style**：

> [Linalg §3.4] "In such operations, one of the tensor arguments is tied with the resulting tensor for in-place bufferization. Such a tensor argument is called an output tensor... Intuitively, output tensors are similar to C++ output parameters that are passed as non-const references and used for returning the result of a computation. Except these ties between an output tensor (argument) and the operation's result serve as a bufferization constraint with no observable impact on the functional semantics; in particular, output tensors still appear as immutable."

论文给的理由是从「结构化算子与 `scf.for` 如何组合」这一第一性原理推出来的：`scf.for` 要 yield 一个值，其嵌套 region 就必须 yield **完整的张量**而非任意子集；而 region 里的算子通常作用在张量子集上（多半是 tiling 的产物），于是自然要成对插入 `extract_slice`/`insert_slice`；这些算子与配套的 `scf.yield` 天然「消费掉」自己的张量参数（此后不会再被使用），因而是就地 bufferize 的理想候选 [Linalg §3.4]。论文最后落到一句：

> [Linalg §3.4] "linalg.generic itself is designed as a destination-passing style operation. This includes linalg.matmul and any other operation that reduces to linalg.generic."

**读者此处该记住**：`linalg.matmul ins(%a, %b) outs(%c)` 里那个 `outs`，不是「顺手传个输出 buffer」，而是一条**编译期的 bufferization 约束**——它不改变函数式语义（张量仍不可变），只告诉后面的 bufferization「结果可以写进这块」。

---

## §8 变换栈：tiling → padding/packing → vectorization → bufferization → 向量渐进下降

[Linalg §3] 用同一个卷积例子把整条变换链走了一遍。本节按论文小节顺序给 writer 铺好料。

### 8.1 Tiling：循环体里仍然是同一个结构化算子（Fig.4，本章第三张关键图）

> [Linalg §3.1] "Tiling the operation introduces scf.for loops as well as subset operations (tensor.extract_slice and tensor.insert_slice) to access the tiled data... **The tiled form of the operation is itself a linalg.conv_1d_nwc_wcf operating on the tiled subsets.**"（粗体为本包所加）

这是 structured op 最反直觉、也最关键的性质：**tiling 之后，循环体里不是标量语句，而是同一个算子的小号版本**。所以变换可以继续叠：再 tiling 一次、做融合、再向量化。

论文并诚实交代了 tiling 带来的麻烦：所选 tile size（此例 `1x8x32x1x8`）虽然是静态的，但有些除法除不尽，边界 tile 要按「满/不满」分类，于是**没有哪个静态张量类型对每次循环迭代都合法**，被切出来的张量类型必须放松成动态形状 `!tDyn`；后续单独的 canonicalization 再把能确定的部分重新静态化 [Linalg §3.1]。此外，「tiling on tensors」引入的 `scf.for` 会在每次迭代 yield 完整张量值，**避免多余的分配与拷贝是 bufferization 的责任** [Linalg §3.1]。

### 8.2 Padding 与 packing：把动态边界磨平

tile 内容变动态会妨碍向量化（向量化要求静态尺寸）。论文给出**三种缓解手段（§3.2 自列的 (1)(2)(3)）**：

1. **多级循环剥离（peeling）或版本化**：把静态已知的主体隔离到主循环，边界走 cleanup 循环；cleanup 仍是动态的，但总可以按 1 来 tiling、退化成尺寸为 1 的维度再细粒度向量化 [Linalg §3.2]。
2. **padding**：把动态 tile 补到更大的、静态已知的尺寸。**补的值必须是消费该 tile 的算子的幺元（neutral）**；代价是边界处多出拷贝与计算，但所有 tile 都变成满 tile [Linalg §3.2]。
3. **显式 masking**：论文明说「work in progress and outside the scope of this paper」[Linalg §3.2]——**本包不展开，writer 也不要展开。**

padding 还有一个性能侧的延伸：对存在时间复用的算子，可以把 pad 操作**外提（hoist）** 出 tile 循环，把补齐后的 tile 存进一个更高维的 packed 张量。好处有两层：摊薄拷贝成本，以及把这些 tile 在内存里**连续排布**——从而缩短短时间内被复用的 tile 之间的内存距离、减少 TLB miss [Linalg §3.2]。外提层数可按张量配置，在内存占用、拷贝成本与计算收益之间权衡 [Linalg §3.2]。

**读者提示（本书相关）**：「补齐 + 打包成连续 tile」这个动机，与 ch02 讲的昇腾末轴对齐/搬运代价是同一类问题的不同实例——但**论文没有讲 NPU**，两者的联系是本书的类比，不是论文断言。

### 8.3 Vectorization：数据搬运通用，计算体分五种情形

`linalg` 算子的向量化配方是：为每个操作数引入一个 `vector.transfer_read`，以向量形式完成计算，再经 `vector.transfer_write` 写回张量或 buffer；其中 `vector.transfer` 的索引跟随 linalg 算子的索引表达式，**这部分对所有 linalg 算子是通用的** [Linalg §3.3]。论文顺带称 `vector.transfer` 是「弥合内存与向量之间鸿沟的瑞士军刀」：它携带足够信息以编码广播、置换、掩码、补齐等多维向量访存模式，因而易于重定向到不同的内存子系统与向量 ISA [Linalg §3.3]。

**计算体的向量化则分情况（§3.3 自列的 5 条）** [Linalg §3.3]：

| # | 情形 | 处理 |
|---|---|---|
| 1 | 逐点算子（索引全为恒等） | 算子体里每条运算直接写成逐点的向量变体 |
| 2 | 低维操作数 | `vector.broadcast` 升到高维，归约到情形 1 |
| 3 | 索引表达式里有置换 | 用 `vector.transpose` 处理 |
| 4 | 有归约维 | 视对算子体的进一步分析，下降成一等的 `vector.contract` 或 `vector.multi_reduction` |
| 5 | 滑窗模式（如卷积） | 特殊处理：沿某些维度展开并抽取切片，进而归结为 `vector.contract` 或 `vector.fma`；论文称这一简单策略「在覆盖 strided 与 dilated 卷积的同时交付了高性能」 |

论文对这一整套的收尾很关键：**「所有这些变换都沿 SSA def-use 链实现，且按设计即合法（legal by design）」** [Linalg §3.3]。

### 8.4 Bufferization：把不可变张量落进内存

> [Linalg §3.4] "Bufferization is the process of materializing tensor values into memory (memref). It is necessary to make tensor programs concretely executable with a source of data residing in memory. In our current compilation pipeline, it is one of the last steps."

目标写得很直白：**尽可能少分配、尽可能少拷贝**；buffer 要尽量复用与就地更新，否则程序变换会带来意料之外的分配与拷贝，代价很大 [Linalg §3.4]。难点是 **read-after-write 冲突**：为每次写都新分配 buffer 永远安全但浪费；复用并就地写则可能非法——若被覆盖位置的原数据在之后还要被读 [Linalg §3.4]。论文把高效 bufferization 类比为寄存器合并（register coalescing）问题 [Linalg §3.4]。启发式的落点就是 §7.4 的 destination-passing style。

### 8.5 多维向量算子的渐进下降

最后一段（[Linalg §3.5]）是「渐进式下降」原则的教科书演示：从向量化后的矩阵乘出发，(a) 先做 vector **unrolling**——两个目的：把向量算子拆成目标已知支持的尺寸（如映射到 AMX 指令），以及提前把非 2 的幂尺寸拆成 2 的幂的组合（例：`vector<12xf32>` 拆成 3 个 `vector<4xf32>`）以免后端产出次优代码；(b) 把 `vector.transfer_read` 里的转置物化出来；(c) 生成 1 维 load 与广播；(d) 把 `vector.contract` 降成外积（论文注明也可选内积或 LLVM 矩阵 intrinsic）；(e) 进而映射到 SIMD 的 fused multiply-add [Linalg §3.5]。每一级下降都伴随折叠与 peephole，减少 IR 体量并使能后续变换；最终得到的向量 IR 作用在 `vector<8xf32>` 上（例如 AVX2 支持的宽度），相当紧凑 [Linalg §3.5]。

---

## §9 transformation-oriented IR design：合法性、可施加性、收益

[Linalg §3.6] 把前面所有变换背后的方法论提炼成一句：

> [Linalg §3.6] "These transformations are legal by design, in the sense that their legality and applicability derive from the operation's properties and structure. We refer to this philosophy as transformations-oriented IR design."

（论文此处写作 “transformations-oriented”，§3.6.1 标题与 §6 结论处写作 “transformation-oriented”；本包照录两处原样，不代论文统一。）

[Linalg §3.6.1] 把传统数值计算编译器的取舍拆成三问，这是给读者的**分析框架**：

- **合法性（Legality）**：哪些变换施加后不改变可观察语义？通常靠静态分析检查（例：支配性分析给出代码移动的必要条件——使用点必须仍被定义点支配）。
- **可施加性（Applicability）**：找到该施加变换的位置有多难？变换后 IR 会变得多复杂？还包含「丢了多少信息、IR 是否仍可分析、后续变换是否仍容易施加」。
- **收益（Profitability）**：按某个度量，哪些变换算有益？通常由启发式或性能模型决定（多面体编译器常聚焦于找一个目标函数去最小化，自动调优器则可能依赖学习到的性能模型）。

而全节的论点是：**这三问依附于哪一层抽象，是可以设计的**——

> [Linalg §3.6.1] "The finer-grained the IR, the more general and canonical the representation, but also the more intractable the analyses and transformations."

论文举了一个非常有说服力的 phase-ordering 例子：为提升时间局部性而做的循环融合，可能**破坏后续识别出高效 BLAS-2/BLAS-3 库实现的能力** [Linalg §3.6.1]。这正是「过早下降到循环层」的代价。

[Linalg §3.6.2] 则给出高层 IR 的第二重红利——**变换目标好指定**：绝大多数变换针对 IR 中的**单个算子**，而不是「循环」这种多算子构造；tiling、fusion、unrolling 都施加于高层算子而非循环（论文诚实注明：显式分布与软件流水这类变换仍天然附着于循环，见 §3.6.2 脚注 8）。论文进一步指出这条路通向**把变换本身写成 MLIR 方言**：变换序列可以被存储、分析、变换，并与主编译器分开发布 [Linalg §3.6.2]；并且这种直接控制多维结构化算子的能力，「在把循环当作变换句柄的 IR 里是没有的」[Linalg §4.2]。

**与多面体编译的关系**（读者常问，论文有正面回答）：[Linalg §5.6] 承认多面体模型几十年来处在循环嵌套优化的前沿，也列了它未被主流采纳的原因——表示多级 tiling/并行/数据搬运/展开时 IR 比仿射调度复杂得多、需要 isl schedule tree 这类复杂抽象；调度与代码生成依赖指数级算法；仿射表示与 SSA 形式不可组合，从而与归纳变量正规化、循环不变量外提、向量化等 pass 产生顺序冲突。MLIR 的 `affine` 方言缓解了其中一些长期问题，但——

> [Linalg §5.6] "Structured operations avoid these problems by operating on a higher level of abstraction, involving tensor-operation-specific optimizations and lowering strategies instead."

论文的结论段再收一次口：这套设计的特征是 **transformation-oriented IR design——「不再在低层 IR 上做合法性分析与可施加性检查，而是系统性地依赖精心设计的抽象的渐进分解」** [Linalg §6]。

---

## §10 落到本书：MLIR/Linalg 与 ttadapter → HFusion → HIVM 的对位

> ⚠️ **证据强度声明**：本节的昇腾侧材料在本仓**极薄**——`architecture.md` 78 行、`Passes.td` 33 行，合计 111 行，且都是概述性文字，没有算子清单、没有 pass 实现。**承重仍是两篇论文**；本节只做概念对位，凡涉及「昇腾为什么这样做」的因果，一律标为本书类比或留给下游章节，不冒充论文/文档断言。

### 10.1 三层对位表

| 本章的论文概念 | 昇腾侧的落点 | 证据与强度 |
|---|---|---|
| 方言 = 算子/属性/类型的命名空间分组 [MLIR §3] | AscendNPU-IR 自研方言：HFusion、HIVM、HACC、Annotation、Scope | [src: architecture.md:11] 逐字列出这五个自研方言及各自职责（文档口径，未核对 `.td`） |
| Linalg structured op 与 named op 设计 [Linalg §2.3.6][Linalg §5.1] | **HFusion = Linalg 的扩展集**，继承 Linalg 全部算子、只处理 named op | [src: architecture.md:17]，见 §7.3 逐字引文（文档自述，本包未核对算子清单） |
| 渐进式下降 + 方言共存 [MLIR §2][MLIR §3] | `ttadapter → HFusion → HIVM → NPU binary` 的多段下降 | 下降链本身见 ch01/ch10；[src: architecture.md:76] 说明 `bishengir-compile` 「输入与输出都是 MLIR」，`hivmc` 再把低层 MLIR 转成 LLVM IR 并做底层指令编译——**「每一步都物化在 IR 里」这条 [Linalg §2.1] 的原则在这里是可检验的** |
| 「维持高层语义、只在必要时丢结构」[MLIR §2] | HFusion 被定义为「硬件相对无关的优化层」，HIVM 才「细粒度感知 NPU 硬件细节」 | [src: architecture.md:11] 逐字：HFusion "responsible for hardware-relatively-independent optimization"；HIVM "responsible for fine-grained awareness of NPU hardware details and for converting high-level programming languages into NPU low-level instructions" |
| ODS/TableGen 声明式定义 + 生成 C++ [MLIR §4.1] | `.td` 写声明、build 目录下生成 `.h.inc/.cpp.inc`；`include`/`lib` 目录结构对齐 MLIR | [src: architecture.md:55]；样本见 [src: third_party/ascend/include/TritonToLinalg/Passes.td] 全文 33 行 |
| 「定义对应外部系统的方言以便互操作」[MLIR §6.3] | AscendNPU-IR 的 Conversion 目录既含生态转换（如 TorchToHFusion）也含内部方言转换（如 HFusionToHIVM） | [src: architecture.md:72] 逐字给出这两个例子 |
| 上游复用而非侵入式修改 [MLIR §1 的复用主张] | 对 MLIR 上游的增强优先放在 `bishengir/Dialect` 下的独立方言目录；实在无法隔离的改动以**独立 patch 文件**施加，每个 patch 有自己的提交信息以便未来回上游社区 | [src: architecture.md:37] |

### 10.2 三条留给下游章节、本章不许下结论的问题

1. **`ttadapter` 究竟怎么把「指针张量」变成结构化张量**——这是全书与基座最根本的 divergence，属 ch10/ch11（PtrAnalysis）。本章只提供「为什么值得变成 Linalg」的理由（§6–§9），不碰实现。
2. **默认产 generic 还是 named**——`Passes.td` 的 `namedOps` 默认 `false`（§7.3），与 HFusion「只处理 named op」的自述之间的关系，须 ch10 读实现后回答。
3. **昇腾的 tiling/融合与论文的 tiling/融合是否同一套机制**——论文的 tiling 面向 cache 层级与向量 ISA，昇腾面向 UB 容量/double-buffer/cube-vector 分工（ch02）。**形式相似不等于机制相同**，须 Part 4 的 HIVM 章据源码定论。

### 10.3 一句话收束（给 writer）

**MLIR 给了「怎么造一层 IR」的方法，Linalg 给了「张量计算该造成什么样的 IR」的答案；昇腾后端不是另起炉灶，而是在这两个答案上加了一层硬件专属的方言（HFusion/HIVM）。** 读者读完本章再回看 ch01 的下降链图，那条链上的每个箭头都会从「一次神秘的翻译」变成「一次有原则的渐进下降」。

---

## §11 open_question 与诚实边界清单

**必须留白的问题（writer 不得自行填平）**

1. **`indexing_maps` / `iterator_types` 的名字问题**：这对属性名在两篇论文正文里**只出现在 `vector.contract` 上**（[Linalg §3.3] Fig.6），linalg 侧论文的措辞是 indexing function / indexing expressions（[Linalg §3.1][Linalg §3.3]）与 OpDSL 的索引记法（[Linalg §4.1] Fig.11，如 `C[D.m, D.n] += cast(T3, A[D.m, D.k]) * cast(T3, B[D.k, D.n])`）。**语义有论文支撑，属性名属实现层口径**——要按名字讲，须另引上游文档或本仓 `.td`，或明说这是实现层名字。
2. **HFusion 「继承 Linalg 全部算子」只有一句文档自述**（[src: architecture.md:17]），本包未核对 `.td` 算子清单。**任何「共 N 个算子/N 种扩展」的计数，本包一律不给**，writer 也不要造。
3. **`namedOps` 默认 `false` 的实际效果**：本包只陈述 `.td` 的声明，不代 ch10 下结论（见 §10.2）。
4. **两篇论文都不涉及 NPU / cube-vector 双核异构**：「Linalg 的可重定向性覆盖昇腾」是本书的类比，不是论文断言。§10 的对位一律按类比措辞写。
5. **显式 masking**：[Linalg §3.2] 自称 work in progress 且在该文范围之外——本章同样不展开。
6. **论文的性能数据不进本章**：[Linalg §4] 的单线程 CPU 实验（带宽/矩阵乘/卷积/稀疏/autotuning）与本书主题（昇腾 NPU）无对位关系，且是 2022 年特定 CPU 上的数字。**除非 writer 明确只用来佐证「结构化 codegen 能达到可观性能」这一定性结论并注明硬件与年份，否则不引。**

**可以放心写死的硬事实（均有逐字引文，见正文对应锚）**

- MLIR 的 IR 构件与语义：Op 是语义单位、指令/函数/module 都是 Op（[MLIR §3]）；region→block→Op 三层递归（[MLIR §3]，Fig.3）；attribute 是编译期静态信息的开放键值字典（[MLIR §3]）；用 block argument 取代 φ 节点（[MLIR §3]）；isolated-from-above 使并行编译成为可能（[MLIR §3][MLIR §4.3]）；不同方言算子可在任意层级共存（[MLIR §3]）；严格类型相等、无隐式转换（[MLIR §3]）。
- MLIR 的基础设施：ODS 基于 TableGen、最终翻成 C++（[MLIR §4.1]）；DRR 表达 DAG 等价（[MLIR §4.2]）；pass manager 不绑定固定粒度、无全模块 use-def 链（[MLIR §4.3]）；文本形式完全可往返、单跑一个 pass 与在流水线里跑结果相同（[MLIR §4.4]）；验证失败即中止编译（[MLIR §4.6]）；可复用 pass 的四条途径（[MLIR §6.1] 原文 “four major approaches”）。
- Linalg 的结构化思想：structured op 的三性质（[Linalg §2.3.6]）；迭代域隐式且迭代器扫过操作数全部数据（[Linalg §3]）；稠密子集由「迭代域在索引函数下的像」求出（[Linalg §3.1]）；tiled 形态仍是同一个算子（[Linalg §3.1]）；named op 是 generic op 的声明式配置（[Linalg §5.1]）；destination-passing style 是 bufferization 约束而非语义变化（[Linalg §3.4]）；变换 legal by design（[Linalg §3.3][Linalg §3.6]）；合法性/可施加性/收益三问（[Linalg §3.6.1]）。
- 数字类：Open64 WHIRL 五个层级（[MLIR §2]）；2020 年社区快照 16 所大学 / 4 国 4 家国家实验室 / 14 家跨国公司 / 100+ 工业界开发者 / 26+ 方言 / 7 个项目（[MLIR §5]，**须注明是 2020 年口径**）；codegen 相关方言 7 个（[Linalg §2.3.1–§2.3.7] 自列清单）；padding 三种缓解手段（[Linalg §3.2] 自列 (1)(2)(3)）；算子体向量化五种情形（[Linalg §3.3] 自列 (1)–(5)）；例子里的 tile size `1x8x32x1x8` 与张量类型（[Linalg §3][Linalg §3.1]）。

**图**

- 三张 key_figures 见 `meta.json`：[MLIR Fig.3]（op/region/block 递归结构）、[Linalg Fig.1]（结构化 codegen 鸟瞰流水）、[Linalg Fig.4]（tiling 前后：循环体里仍是同一算子）。**一律原创重绘**，不得内嵌受版权保护的论文原图；图注须写成「重绘自 arXiv:<id> Fig.N …」（`lint_paper_grounding` 靠图注里的 Fig 号回连登记项）。术语译名统一由 Book Bible。
