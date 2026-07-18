# 论文包 · ch15《SSA 与结构化控制流：φ 节点、块参数与 loop-carried 变量》

> **这份文件是什么**：primer 原理章的**素材真相源**，不是论文翻译。它把本章要讲透的三层理论（SSA/φ 的定义 → MLIR 用块参数取代 φ → 结构化控制流的 iter_arg 语义）整理成 writer/analyst 直接可用的骨架，每个论点标出处、标来源层级。
>
> **⚠️ 全章写作红线（贯穿始终，反复对 writer 强调）**
> Triton 前端**并不运行** Cytron 的 φ 放置算法（支配边界 / 最小 φ 插入）。`code_generator.py` 是**直接从 Python AST 构造 SSA**：`visit_For` 先 dry-run 一遍循环体（`code_generator.py:958-975`），再用 `local_defs ∩ liveins` 的 scope 差集认出 loop-carried 变量。所以——
> **φ 在本章只能当「语义与记号」讲**（汇合处如何选出正确来源），**绝不可**讲成 Triton 要实现的算法。凡出现 φ，必须同时提醒：这是理解 IR 汇合语义的记号，MLIR 用块参数实现它，Triton 前端用 AST scope 差集直接构造它——三者是同一件事的不同层，不是同一段代码。

---

## 来源层级说明（诚实标注，防编造）

写作时每条事实按下面三档标注可信度，**不得越档**：

| 档 | 含义 | 本包内哪些属于此档 |
|---|---|---|
| **A｜逐字核到原文** | 已从 PDF/官方文档逐句读到，可逐字引用 | MLIR 论文全部引文（arXiv:2002.11054 PDF 21 页已抽取核对）；Triton 源码全部行号（本地 `code_generator.py` 已核） |
| **B｜权威转述** | 原文可能在付费墙，用 MLIR 论文对它的引用 + 已核实的书目事实 | Cytron 1991 的**内容**（TOPLAS 正文付费墙）——用 MLIR 论文正文与参考文献 [15] 的转述 + 已核卷期页 |
| **C｜官方文档为准（无学术论文）** | 确认不存在专属论文，以官方 dialect 文档措辞为准 | SCF dialect（scf.for/scf.if）的 iter_args 语义 |

> Cytron 1991 正文（支配边界算法细节）**读不到就不写**。本章只需要它的**定义性结论**（SSA 是什么、φ 是什么），这一层 MLIR 论文已完整转述并引用，属 B 档，足够；深入算法细节既超出本章需要，也无原文支撑。

---

## 参考文献总表（三篇 + 一处「确认无论文」）

| # | 文献 | 出处（已核） | 本章角色 | status |
|---|---|---|---|---|
| 1 | Cytron, Ferrante, Rosen, Wegman, Zadeck，*Efficiently Computing Static Single Assignment Form and the Control Dependence Graph* | ACM TOPLAS **13(4):451–490**, Oct 1991 · DOI:10.1145/115372.115320 | SSA/φ 的**定义性文献**（primer-core） | fetched（B 档，见下） |
| 2 | Lattner, Amini, Bondhugula, Cohen, Davis, Pienaar, Riddle, Shpeisman, Vasilache, Zinenko，*MLIR: A Compiler Infrastructure for the End of Moore's Law* | arXiv:**2002.11054** [cs.PL] · CGO 2021 pp.2–14 · DOI:10.1109/CGO51591.2021.9370308 | region/块 + **块参数取代 φ** + 结构化控制流（primer-core，**本章命门**） | fetched（A 档，PDF 逐句核） |
| 3 | Appel，*SSA is Functional Programming* | ACM SIGPLAN Notices **33(4):17–20**, April 1998 · DOI:10.1145/278283.278285 | φ↔块参数设计差异的**文献出处**（prereq-box） | fetched（B 档，见下） |
| — | MLIR **SCF dialect**（scf.for/scf.if） | 无专属学术论文，官方 dialect 文档为准 | iter_args 语义依据（citation） | no-paper（C 档，确认不存在） |

---

## 第 1 层 · 动机：为什么编译器 IR 要 SSA

**核心命题**：SSA（Static Single Assignment，静态单赋值）= **每个变量在程序文本里只被赋值一次**。一旦某个名字被再次赋值，就换一个新版本（`x`→`x₁`→`x₂`）。

**为什么要这样**（MLIR 论文对 SSA 的评价，A 档逐字，§"SSA and regions"）：

> "The Static Single Assignment (SSA) form [15] is a widely used representation in compiler IRs. It provides numerous advantages including making dataflow analysis **simple and sparse**, is widely understood by the compiler community for its relation with **continuation-passing style**, and is established in major frameworks."
> —— arXiv:2002.11054，§"SSA and regions"

翻成本章能用的直觉：**每个变量只赋值一次 → 「这个值从哪来」有唯一答案 → 数据流图直接显式化 → 优化（常量传播、死代码消除、公共子表达式）不必反复追问「此刻 x 是哪次赋值的结果」**。这就是 SSA 让 dataflow analysis「simple and sparse」的原因——use-def 关系一步到位，不用做到处的定值到达分析。

> **primer 记号**：把「变量 `$`x`$` 的第 `$`i`$` 个版本」写作 `$`x_i`$`。SSA 的不变量是：
> ```math
> \forall\, \text{值 } x_i:\quad \bigl|\{\text{给 } x_i \text{ 赋值的语句}\}\bigr| = 1
> ```
> 「只赋值一次」不是语言层面的 immutable 变量，而是 **IR 层面的表示约定**——同一个 Python 变量 `acc` 在源码里被 `acc += ...` 更新多次，下降到 IR 后会变成 `$`acc_0, acc_1, \dots`$` 一串版本。

**出处**：SSA 的定义归 Cytron 1991（DOI:10.1145/115372.115320）；「为什么好」的评价用 MLIR 论文 §"SSA and regions" 逐字（A 档）。

---

## 第 2 层 · φ 节点：汇合处如何选值（定义 + 记号，标红线）

### 2.1 问题：控制流一汇合，SSA 就「撞版本」

SSA 要求「一个变量一个版本」，但**控制流汇合点**天然打破它。经典例子：

```
if cond:
    x = a        # 这一支给 x 赋 a
else:
    x = b        # 那一支给 x 赋 b
use(x)           # 汇合后 use 到的 x 是哪一个？
```

下降到 SSA：then 支产出 `$`x_1 = a`$`，else 支产出 `$`x_2 = b`$`。汇合点 `use(x)` 用到的那个 `x`，**在编译期无法静态确定是 `$`x_1`$` 还是 `$`x_2`$`**——取决于运行时走了哪条路。SSA 的「单赋值」在这里似乎失效了。

### 2.2 φ 节点：Cytron 给的答案

**φ 节点**（phi function）就是补这个洞的记号。在汇合块的开头放一个伪操作：

```math
x_3 = \phi(x_1,\ x_2)
```

语义：**「若控制流从 then 支来，`$`x_3`$` 取 `$`x_1`$`；从 else 支来，取 `$`x_2`$`」**。φ 按「你从哪个前驱块进来」选出对应的来源版本，于是汇合点又得到了**唯一一个**新版本 `$`x_3`$`，SSA 不变量恢复。φ 的实参个数 = 该汇合块的前驱块个数，第 `$`k`$` 个实参对应第 `$`k`$` 个前驱。

**出处 / 来源层级（B 档）**：φ 与 SSA 的定义出自 Cytron et al. 1991（TOPLAS 13(4):451–490）。该文正文（含支配边界、最小 φ 插入算法）在 ACM 付费墙内，本包**未取到全文正文**；上述定义性结论取自 MLIR 论文对 [15] 的转述与 SSA 通识，卷期页码与 DOI 已联网核实。**深入的 φ 放置算法细节本章不需要、也不写**。

### 2.3 ⚠️ 红线：φ 在本章是「语义与记号」，不是 Triton 的算法

**必须对读者讲清的三件事**（否则整章会误导）：

1. **Cytron 的贡献里有两半**：一半是「φ 是什么」（定义 / 记号，本章要），另一半是「φ 该放在哪、怎么放最少」（支配边界 + 最小 φ 插入算法，本章**不要**）。
2. **Triton 前端不跑那个算法**。`code_generator.py` 直接从 Python AST 构造 SSA：`visit_For` dry-run 循环体后按 scope 差集认 loop-carried 变量（详见第 5 层），根本不计算支配边界。
3. 因此本章凡出现 φ，一律以「**理解汇合语义的记号**」身份出场：它帮读者想清楚「循环 / 分支汇合处，一个值到底从哪来」。至于这个语义在工程上怎么落地——**下一层给答案：MLIR 用块参数实现它**。

> writer 落笔提示：可以在 φ 定义处放一个「记号 vs 算法」的边注框，一句话钉死红线，避免读者以为后面要教 φ 插入算法。

---

## 第 3 层 · MLIR 的块参数取代 φ（本章命门，A 档逐字）

这是全章的枢纽：**φ 是想清楚汇合语义的记号，块参数（block argument）是把这个语义落地成 IR 的工程形式**。Triton 生成的就是 MLIR，所以读者最终看到的不是 φ，而是块参数。

### 3.1 region / block：MLIR 的嵌套结构（A 档逐字，§"Regions and blocks"）

> "An instance of an Op may have a list of attached regions. **A region provides the mechanism for nested structure in MLIR: a region contains a list of blocks, and a block contains a list of operations (which may contain regions).** ... For example, the `affine.for` operation in Figure 4 is a **loop with the single-block body attached as a region**, located between `({` and `})` delimiters."
> —— arXiv:2002.11054，§"Regions and blocks"

嵌套三层：**Op ⊃ region ⊃ block ⊃ Op**（可再嵌 region）。一个 block 内部若多于一块，就构成 CFG；每个 block 以 **terminator**（终结子，如 branch/switch/yield）结尾，terminator 决定控制流转到哪个后继 block（A 档，同节）：

> "The body of each region is a list of blocks, and each block ends with a **terminator operation**, that may have successor blocks to which the control flow may be transferred. ... The graph of successors defines a CFG, allowing standard SSA-based control flow within a region."

对本章的意义：`scf.for` 的循环体就是「单块 body 挂成一个 region」——`affine.for` 是它的同构前身，论文用的正是这个例子。

### 3.2 块参数取代 φ（**本章命门，A 档逐字**）

> "**Instead of using φ nodes, MLIR uses a functional form of SSA [2] where terminators pass values into block arguments defined by the successor block.** Each block has a (potentially empty) list of typed block arguments, which are regular values and obey SSA. The semantics of terminator Ops defines what values the arguments of the block will take after the control is transferred. For the first (entry) block of the region, the values are defined by the semantics of the enclosing Op. For example, **`affine.for` uses the entry block argument `%arg4` as loop induction variable.**"
> —— arXiv:2002.11054，§"Regions and blocks"

**这段是全章的锚**。把 φ 和块参数摆在一起看，是同一语义的两种写法：

| 视角 | φ 节点（记号） | 块参数（MLIR 落地） |
|---|---|---|
| 值在哪声明 | 汇合块开头 `$`x_3=\phi(x_1,x_2)`$` | 后继块的**参数列表**声明 `^bb(%x3: T)` |
| 来源怎么给 | φ 按「从哪个前驱来」选实参 | 每个前驱的 **terminator 跳转时传实参** |
| 语义 | 汇合处按前驱选值 | terminator 把值「传进」后继块参数 |

**一句话记法**：**φ「拉」（汇合块回头问各前驱要值），块参数「推」（各前驱主动把值推给后继块参数）**——同一件事，方向相反、都是 SSA。

> **primer 记号（φ ↔ 块参数等价）**：设汇合块 `$`B`$` 有前驱 `$`P_1,\dots,P_n`$`，φ 写法与块参数写法逐一对应：
> ```math
> \underbrace{x = \phi(v_1,\dots,v_n)}_{\text{φ：在 }B\text{ 头部选值}}
> \quad\Longleftrightarrow\quad
> \underbrace{B(x{:}\,T)\ \text{；}\ P_k \text{ 的 terminator: } \mathtt{br}\ B(v_k)}_{\text{块参数：各前驱传实参}}
> ```

### 3.3 结构化控制流是 MLIR 的明写设计目标（A 档逐字）

> "To support heterogeneous compilation, the system has to support the **expression of structured control flow**, concurrency constructs, closures in source languages, and many other purposes."
> —— arXiv:2002.11054，§"SSA and regions"

即：把循环 / 条件表达成**结构化的、带 region 的 Op**（而非扁平 CFG + goto），是 MLIR 的一等公民目标。这正是 `scf.for` / `scf.if` 存在的理由，也是第 4 层的入口。

**出处**：本层全部引文 A 档，均来自 arXiv:2002.11054 PDF 逐句核对（§"Regions and blocks" 与 §"SSA and regions"）。「块参数取代 φ」那句里的 **[2] 即 Appel 1998**（第 4.5 / prereq-box 展开）。

---

## 第 4 层 · 结构化控制流：scf.for / scf.if 与 iter_arg（C 档，官方文档为准）

### 4.1 诚实前提：SCF dialect 无专属论文

**确认结论（不是「没查到」，而是「不存在」）**：MLIR 的 SCF（Structured Control Flow）dialect **没有**自己的学术论文，官方 dialect 文档通篇不引任何学术参考。故本层以「MLIR 论文（region / 块参数 / 结构化控制流，第 3 层）+ 官方 SCF dialect 文档」为准，**不得为 scf.for/scf.if 杜撰论文出处**。官方文档措辞可直接作语义依据，且与 Triton 源码逐字对应。

### 4.2 scf.for 的 region 参数 = 归纳变量 + 每个 loop-carried 变量一个（C 档，官方文档措辞）

官方 SCF dialect 文档对 `scf.for` region 参数的措辞：

> "The operation region has **an argument for the induction variable, followed by one argument for each loop-carried variable**."
> —— MLIR 官方 SCF dialect 文档（scf.for）

这就是**归纳变量（induction variable）+ iter_arg（loop-carried 变量）** 的合成：region 入口块的参数列表 = `[归纳变量, iter_arg₀, iter_arg₁, …]`。归纳变量占 **arg(0)**，第 `$`i`$` 个 loop-carried 变量占 **arg(i+1)**。

**与 Triton 源码严丝合缝（A 档，本地已核 `code_generator.py:1002`）**：

```python
# code_generator.py:1002 —— dry-run 后正式 visit，把每个 loop-carried 变量绑到 for-body 的块参数
self.set_value(name, language.core.tensor(for_op.get_body(0).arg(i + 1), yields[i].type))
```

**`arg(i + 1)` 里的 `+1` 正是跳过归纳变量**（归纳变量是 arg(0)）——与官方文档「induction variable, followed by one argument for each loop-carried variable」逐字对上。这是块参数取代 φ 在 Triton 里的**具体落点**：iter_arg 就是 `scf.for` body 块的参数。

### 4.3 yield 把 loop-carried 值传给下一轮 / 传出循环（C 档 + A 档源码）

官方文档：

> "The region must terminate with a **`scf.yield`** that passes the current values of all loop-carried variables to the next iteration, or to the `scf.for` result, if at the last iteration."
> —— MLIR 官方 SCF dialect 文档（scf.for）

对应 Triton 源码（A 档，本地已核）：

```python
# code_generator.py:1011-1012 —— 循环体末尾把本轮 loop-carried 值 yield 出去（= terminator 传实参给后继/下一轮）
if len(yields) > 0:
    self.builder.create_yield_op([y.handle for y in yields])
```

循环结果再接回 `for_op.get_result(i)`（`code_generator.py:1027`，A 档已核）。`scf.yield` 就是第 3 层「terminator 把值推给块参数」的实例：**它把本轮的 loop-carried 新值推给下一轮的 iter_arg（块参数），末轮则推给 `scf.for` 的结果**。

### 4.4 scf.if 也能产出结果（C 档 + A 档源码）

> "`scf.if` may also produce results ..."
> —— MLIR 官方 SCF dialect 文档（scf.if）

对应 Triton 源码 `visit_if_scf`（A 档，本地已核 `code_generator.py:680` 附近）：两支各自 `create_yield_op`，汇合后 `if_op.get_result(i)` 取出结果。这正是 §2.1 那个 `if/else` 里 `x` 的 φ 问题的 MLIR 落地——**`scf.if` 的两支各 yield 一个值，`if_op` 的结果就是 φ 选出来的那个**。φ（记号）→ 块参数 / yield（MLIR）→ `if_op.get_result`（Triton），三层同一件事。

### 4.5 prereq-box：为什么块参数不是 MLIR 拍脑袋发明的（Appel 1998，B 档）

MLIR 论文那句「a functional form of SSA **[2]**」里的 **[2] = Appel 1998**（A 档已核 PDF 参考文献第 942 行）：

> "[2] A. W. Appel. **SSA is functional programming**. ACM SIGPLAN NOTICES, 33(4):17–20, 1998."
> —— arXiv:2002.11054 参考文献 [2]（逐字核到）

**Appel 的洞见（B 档，DOI:10.1145/278283.278285 已核书目，正文观点为通识转述）**：把 SSA 看成函数式程序——

- **基本块 = 函数**；
- **块参数 = 函数形参**；
- **φ 的各路来源 = 各调用点传入的实参**；
- **跳转 = 尾调用**。

于是 φ 的「并行拷贝」语义与「函数入参绑定」天然等价。**块参数不是临时发明，而是 SSA 的函数式重述**——这为第 3 层「块参数取代 φ」提供了理论正当性。

> 工程动机旁证（官方 MLIR Rationale «Block Arguments vs PHI nodes»，可作 C 档旁证，非学术论文）：φ 必须钉在块首（IR 变换要手工跳过）、函数参数沦为特例、φ 块的原子执行语义反直觉——改用块参数后这些特例一律消失。

**出处 / 层级**：Appel 书目 A 档（MLIR 论文 [2] 逐字核）；Appel 正文观点 B 档（DOI 已核，正文视付费墙情况按通识转述，**不逐字引未取到的原文**）。

---

## 第 5 层 · loop-carried 变量：Triton 前端如何识别（A 档源码，红线落点）

**这一层是红线的正面表述**：读者已从第 2–4 层知道「汇合处要 φ / iter_arg 选值」，本层告诉他们 **Triton 前端不靠支配边界算法，而是靠 AST dry-run + scope 差集**认出哪些变量是 loop-carried。

### 5.1 什么是 loop-carried 变量

循环体里被更新、且更新要「跨轮存活」的变量（如累加器 `acc`、running max `m`）——它必须**带进循环（作为 iter_arg 初值）、带出循环（作为结果）**。反之，纯粹循环内局部、每轮重算的临时量不是 loop-carried，不进 iter_arg。**认对这个集合，是把 Python for 正确下降到 `scf.for` 的关键。**

### 5.2 Triton 的做法：dry-run 循环体 + scope 差集（A 档，本地已核）

`visit_For` 的两趟结构（`code_generator.py:958-975`，A 档逐字核）：

```python
# code_generator.py:958-975（节选）—— 第一趟：dry-run 循环体，只为收集 local_defs
with enter_sub_region(self) as sr:
    liveins, insert_block = sr
    ...
    block = self.builder.create_block()
    self.builder.set_insertion_point_to_start(block)
    # dry visit loop body
    self.scf_stack.append(node)
    self.visit_compound_statement(node.body)   # 试跑一遍，填充 self.local_defs
    self.scf_stack.pop()
    block.erase()                              # 立刻擦掉——这趟只为「看看循环体定义了哪些名字」

    # If a variable (name) is defined in both its parent & itself, then it's
    # a loop-carried variable. (They must be of the same type)
```

判定规则（源码注释逐字）：**「若一个名字在父作用域（liveins）与循环体内（local_defs）都被定义，它就是 loop-carried 变量」**——即取交集 `$`\text{loop-carried} = \text{local\_defs} \cap \text{liveins}`$`。紧接着的循环里以 `if name in liveins` 落地这个交集判定。

```math
\text{loop-carried} \;=\; \{\, \text{name} \;\mid\; \text{name} \in \texttt{local\_defs} \ \wedge\ \text{name} \in \texttt{liveins} \,\}
```

直觉：**循环体里赋了值（`local_defs`）、循环前也已经存在（`liveins`）的名字**，就是那个「进来一个旧值、出去一个新值」的跨轮变量——它必须成为 `scf.for` 的 iter_arg。dry-run 的唯一目的就是**先探一遍循环体定义了哪些名字**，好在正式生成前把 iter_arg 列表定下来。

### 5.3 ⚠️ 红线再钉一次

- 这是 **Triton 的工程做法**（AST dry-run + `local_defs ∩ liveins` scope 差集），**不是 Cytron 的 φ 放置算法**（支配边界 / 最小 φ 插入）。
- 二者解决的问题层次不同：Cytron 算法回答「一般 CFG 上 φ 该放哪、放最少」；Triton 因为**直接从结构化的 Python AST 出发**（循环边界天然已知），根本不需要计算支配边界——它只要认出 loop-carried 集合，交给 `scf.for` 的 iter_arg 机制即可。
- writer 落笔时：第 5 层与第 2 层要**首尾呼应**——第 2 层说「φ 是想清楚汇合语义的记号」，第 5 层说「Triton 用 scope 差集 + iter_arg 落地这个语义，全程没碰 φ 插入算法」。红线闭环。

---

## 给下游（analyst / writer）的落笔清单

1. **叙事主线**：动机（SSA 为何存在）→ φ（汇合选值的记号，**标红线**）→ 块参数（MLIR 落地 φ，**逐字命门引文**）→ scf.for/scf.if 的 iter_arg（结构化控制流，源码逐字对应）→ Triton 前端如何识别 loop-carried（dry-run + scope 差集，**红线正面表述 + 闭环**）。
2. **红线出现三次**：φ 定义处（边注框）、第 3 层块参数处（φ 是记号、块参数是落地）、第 5 层（Triton 做法 ≠ Cytron 算法）。缺一处都可能误导。
3. **引文层级不得越档**：MLIR 引文可逐字（A），Cytron/Appel 正文按 B 档转述、**不逐字引未取到的原文**，SCF 语义按 C 档官方文档。
4. **数学记号**：φ↔块参数等价式、loop-carried 交集式、SSA 单赋值不变量——已按 primer 可用形式给出，行内一律 `` $`…`$ ``、块级 ```math 围栏。
5. **源码锚点（本地已核，A 档）**：`code_generator.py` 的 958-975（dry-run）、1002（`arg(i+1)`）、1011-1012（`create_yield_op`）、1027（`get_result`）、680 附近（`if_op.get_result`）。

---

## fetch 记录（诚实标注抓到 / 未抓到）

- **arXiv:2002.11054（MLIR）**：✅ **A 档全取**。ar5iv HTML 转换失败（LaTeXML fatal error）；改经 arXiv PDF 落盘后用 pypdf 抽取 21 页全文，§"Regions and blocks" / §"SSA and regions" / 参考文献 [2][15] **逐句逐字核到**，本包所有 MLIR 引文均来自此次抽取，非凭记忆。
- **Cytron 1991（DOI:10.1145/115372.115320）**：⚠️ **B 档**。TOPLAS 正文在 ACM 付费墙，**未取全文正文**；卷期页（13(4):451–490）与 DOI 已联网核实，定义性内容用 MLIR 论文 §"SSA and regions" 对 [15] 的转述。支配边界算法细节**本章不需要、未写**。
- **Appel 1998（DOI:10.1145/278283.278285）**：⚠️ **B 档**。书目经 MLIR 论文参考文献 [2] 逐字核到（33(4):17–20）；正文「基本块即函数 / φ 即形参」为函数式 SSA 通识转述，**未逐字引原文**。
- **SCF dialect**：⚠️ **C 档**。确认**无专属学术论文**；iter_args 语义引官方 dialect 文档措辞，与 Triton 源码逐字对应，**未杜撰论文出处**。
