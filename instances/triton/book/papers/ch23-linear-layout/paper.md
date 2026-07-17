# ch23 论文包 —《LinearLayout：一个抽象统一所有布局》

> 本章定位：**本子系统的思想高潮 primer**。第 20 章《布局即函数》已把 layout 讲成一个函数 $`\mathcal{L}`$——「张量索引 → 允许访问该处的线程集合」——并在结尾用一个 3–5 句的**前瞻框**点到一句：「$`\mathcal{L}`$ 其实是 GF(2) 上的线性映射，深化见第 23 章」。**本章就是那个深化。** ch20 只给结论一句，ch23 要把 **GF(2) 代数、bases、xor 线性律、RREF、Four Russians 全部讲透**——这是本章的**主体**，不是前瞻。
>
> 本章的**主真相源仍是源码（A 档），不是论文**：`include/triton/Tools/LinearLayout.h` 的**顶部大段注释**逐字给出了 4×4 swizzle 例子、bases 定义、xor 线性律与 GF(2) 数学背景；`lib/Tools/LinearLayout.cpp` 给出了把布局压成比特矩阵、用 `f2reduce` 做 RREF 的真实调用点。论文 arXiv:2505.23819 只作**学术出处**（B 档）；Four Russians 的复杂度与出身作**百科背景**（C 档）。**红线：只写已核实内容——A 档逐字可核，B/C 档标清出处，抓不到的如实标注、绝不编造。**

---

## 0. 来源层级表（防越档编造）

| 档 | 含义 | 本章用到的具体来源 | 用法 |
|---|---|---|---|
| **A** | 源码逐字 / 源码注释（最高权威，本章主真相源） | `include/triton/Tools/LinearLayout.h` 头注释（`:18–312`，含 4×4 swizzle 例 `:36–74`、bases 定义 `:31–47`、xor 线性律 `:49–74`、其它布局的 bases 例 `:106–148`、GF(2) 数学背景 `:183–277`、`compose`/`invertAndCompose` 契约 `:620–672`、`bases` 字段注释 `:314–320`）；`lib/Tools/LinearLayout.cpp`（`getMatrix` 把布局压成比特矩阵 `:65–113`、`getMatrixRank` 用 RREF 求秩 `:139–159`、`compose` `:813–841`、`invertAndCompose` `:843–922`、三处 `f2reduce::inplace_rref_strided` 调用 `:151` 求秩、`:912` 求逆复合、`:966` `getFreeVariableMasks` 求自由变量掩码）；`third_party/f2reduce/README.md`（**在仓库内**，逐字点名「Kronrod's algorithm ('method of four Russians')」`:7`、RREF over GF(2) 定位 `:1–5`、单函数签名 `:29`）；`third_party/f2reduce/f2reduce.h`（`inplace_rref_strided` 签名与 little-endian 位打包语义 `:7–24`）；`include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h`（`toLinearLayout` 统一入口签名与输入维注释 `:13–45`） | 所有核心论断——GF(2) 线性映射、bases、xor 线性律、4×4 填表、复合、比特矩阵、RREF——**逐字引 .h/.cpp/README**；这是本章基石 |
| **B** | 论文权威转述 | arXiv:2505.23819《Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $`\mathbb{F}_2`$》（Keren Zhou, Mario Lezcano, Adam Goucher, …, Phil Tillet, Thomas Raoux, Zahi Moudallal 等 11 人） | 为「layout = GF(2) 线性映射」「generic layout-to-layout 转换，消除二次爆炸」提供**学术出处**；不作为技术推导的第一依据（源码更权威） |
| **C** | 官方文档 / 百科 | Method of Four Russians（Wikipedia）：出身（Arlazarov, Dinitz, Kronrod, Faradžev, 1970）+ 分块查表把复杂度削一个对数因子 | **仅**用于 Four Russians 背景盒：解释 `f2reduce` 为何快，一段带过 |

> 红线：本包只登记**已核实**的内容。A 档所有引文都能在标注的 `.h` / `.cpp` / `README.md` 行号处逐字核对（这些是**本章基石**，源码注释比论文更权威）；B 档已 WebFetch arxiv.org/abs/2505.23819 核实摘要逐字；C 档已 WebFetch Wikipedia 核实。**未在源码/论文/百科核到的一律不写。**

---

## 1. 动机：ch20 给了函数 $`\mathcal{L}`$，但每种布局各写各的规则太乱

第 20 章的结论：TritonGPU 张量比普通张量多一个 `encoding`，它的正式定义是一个函数 $`\mathcal{L}`$——把张量索引映射到「允许访问该处的 CUDA 线程集合」。ch20 用逐格表把这个 $`\mathcal{L}`$ 讲得能对账，但停在了「$`\mathcal{L}`$ 是一张对照表」这一层。

问题随之而来：**表太多了。** Blocked 一种规则、Slice 一种、MMA（Tensor Core 操作数）又一种、Shared（带 swizzle）再一种……每加一种硬件排布，就得写一个新的布局类、一套新的 index 计算、以及**任意两种布局之间**的转换代码。转换的数量随布局种类**二次爆炸**。LinearLayout.h 的头注释把这个痛点讲得很直白（`LinearLayout.h:76–80`，A 档逐字）：

> *Indeed the whole point of LLs is that they allow us to specify transposed and swizzled layouts as a "general case".  Instead of a layout class for registers in a thread, and another layout for registers in a thread but in MMAv2 order, and so on, all of these can be represented by different LLs.  This gets rid of special cases and lets us write more general code.*

论文摘要用同一句话概括这个动机（arXiv:2505.23819，B 档，已 WebFetch 核实）：用 $`\mathbb{F}_2`$ 上的线性代数建模布局，实现 *generic layout-to-layout conversions, eliminating the quadratic explosion that plagues existing solutions*。

**本章的命题**：能不能用**一个抽象**统一 Blocked / Slice / MMA / Shared 全部布局？答案是 **LinearLayout（LL）**——把布局看成 **GF(2) 上的线性函数**。这个想法归功于 Adam P. Goucher（`LinearLayout.h:20`，A 档逐字：*The idea for linear layouts is due to Adam P. Goucher.*）。下面五节把这个抽象拆开讲透。

---

## 2. 第一步：LinearLayout 是「硬件位置 → 逻辑索引」的函数

先注意 LL 相对 ch20 的 $`\mathcal{L}`$ 换了**方向**。ch20 的 $`\mathcal{L}`$ 是「张量索引 → 线程集合」；LL 反过来，是「硬件位置 → 逻辑张量索引」（`LinearLayout.h:22–29`，A 档逐字）：

> *In Triton, a linear layout (LL) is a function that maps from a "hardware location" to a "logical tensor index".*
>
> *For example, suppose we have a 2D tensor T stored in GPU registers.  T's layout (i.e., L) is the function that, given a "hardware location" tuple of (thread-id, warp-id), returns an index (x,y) into T.  In other words, if L(t,w) = (x,y) is our linear layout func, then a register in thread t in warp w contains the value T[x,y].*

为什么选这个方向而不是反过来？源码专门解释了（`LinearLayout.h:169–181`，A 档逐字，精简）：同一个 `tensor[x,y]` 可能存在硬件的**多个**位置，若从「张量索引 → 硬件位置」出发就不再是一个**函数**（一个输入多个输出），会把一切复杂化。所以 LL 选「硬件 → 逻辑」这个方向，保证它是**函数**。（代价：LL 一般**不是单射**——多个硬件位置可以映到同一个逻辑元素，正对应「同一元素存多份」，`LinearLayout.h:164–167`。）

实际的 GPU 寄存器布局，输入维通常是 `(reg, thread-id, warp-id, block-id)` 四元组——`reg` 表示一个线程可能在多个寄存器里存该张量的多个值（`LinearLayout.h:82–85`，A 档逐字）。LL 是一般的 **MD → ND** 函数。

---

## 3. 关键事实：只需给出「2 的幂次输入」上的取值——bases（基向量）

LL 的**灵魂**在这一句（`LinearLayout.h:31–34`，A 档逐字）：

> *The key fact about LLs is, the mapping from (t,w) to (x,y) is not arbitrary.  We only need to specify the value of L(t,w) at certain special points (namely, the values L(t,0) and L(0,w) where t and w are powers of 2), and from those we can compute all the other values of L.*

也就是说：一个把 $`t \in \{0..3\}`$、$`w \in \{0..3\}`$ 映射的 2D→2D 布局，看起来要填 16 个格子；但**你只需要选 4 个值**——$`L(1,0), L(2,0), L(0,1), L(0,2)`$（输入是 2 的幂次的那些点），剩下 12 格全部能算出来。这 4 个特殊值有个名字（`LinearLayout.h:46–47`，A 档逐字）：

> *You only need to specify these four values to define the whole linear layout.  These special values are called the "basis vectors" or "bases" of the layout.*

源码里 `LinearLayout` 类就是这么存的（`LinearLayout.h:314–320`，A 档逐字）：

> *bases[inDim][i] = L(0, ..., inDim=2^i, ..., 0).  All other values of L are computed by xor'ing bases together, using the linearity rule.*

**这就是「统一」的来源**：无论 Blocked、MMA、Shared 还是 transpose，一个布局都被压缩成**一小组基向量**。头注释给了几个一眼看懂的例子（`LinearLayout.h:106–148`，A 档逐字摘要）：

| 布局 | bases | 出处 |
|---|---|---|
| 1D 恒等 $`L(x)=x`$（8 元素） | $`[L(1),L(2),L(4)] = [1,2,4]`$ | `:108–112` |
| 1D 全零 $`L(x)=0`$ | $`[0,0,0]`$ | `:114–117` |
| 2D→2D 恒等 | $`L(0,1){=}(0,1),\ L(0,2){=}(0,2),\ L(1,0){=}(1,0),\ L(2,0){=}(2,0)`$ | `:119–125` |
| 2D→2D 转置（transpose） | $`L(0,1){=}(1,0),\ L(0,2){=}(2,0),\ L(1,0){=}(0,1),\ L(2,0){=}(0,2)`$ | `:127–132` |
| 2D→1D broadcast $`L(x,y)=x`$ | $`L(0,1){=}0,\ L(0,2){=}0,\ L(1,0){=}1,\ L(2,0){=}2`$ | `:142–148` |

看第 4 行：**转置**——把每个 base 的两个分量对调——就自动实现了。broadcast（第 5 行）——把某维的 base 全设成 0——某维就被「压扁」。这些过去要写专门代码的布局操作，在 LL 里只是**换一组 bases**。

---

## 4. xor 线性律：$`L(a \oplus b) = L(a) \oplus L(b)`$——亲手填 4×4 swizzle 表

bases 只给了「2 的幂次输入」上的值，其余格子靠一条规则补全。头注释把这条规则和一个**可以亲手填的 4×4 例子**摆在一起（`LinearLayout.h:36–74`，A 档逐字，这是本章的**核心顿悟抓手**）。

**第一步：只选 4 个 base**（4 warps × 4 threads/warp，张量 T 形状 4×4；`LinearLayout.h:40–44` 逐字）：

```
               t/w    0     1     2    3
               0      ? (0,1) (0,2)    ?
    L(t,w) =   1  (1,1)     ?     ?    ?
               2  (2,2)     ?     ?    ?
               3      ?     ?     ?    ?
```

即选定 $`L(0,1){=}(0,1)`$、$`L(0,2){=}(0,2)`$、$`L(1,0){=}(1,1)`$、$`L(2,0){=}(2,2)`$。

**第二步：xor 线性律**（`LinearLayout.h:49–51` 逐字，`⊕` = xor）：

```math
L(t_1 \oplus t_2,\ w_1 \oplus w_2) = L(t_1, w_1) \oplus L(t_2, w_2)
```

**第三步：亲手填几格验证**（`LinearLayout.h:56–59`，A 档逐字，读者应拿笔逐位异或核对）：

```
L(0,0) = L(1 ⊕ 1, 0 ⊕ 0) = L(1,0) ⊕ L(1,0) = (1,1) ⊕ (1,1) = (0,0)
L(0,3) = L(0 ⊕ 0, 2 ⊕ 1) = L(0,2) ⊕ L(0,1) = (0,2) ⊕ (0,1) = (0,3)
L(3,0) = L(2 ⊕ 1, 0 ⊕ 0) = L(2,0) ⊕ L(1,0) = (2,2) ⊕ (1,1) = (3,3)
L(3,3) = L(3 ⊕ 0, 0 ⊕ 3) = L(3,0) ⊕ L(0,3) = (3,3) ⊕ (0,3) = (3,0)
```

（逐位核对 `L(0,3)`：$`w=3=0b11`$，拆成 $`2 \oplus 1`$，故 $`L(0,3)=L(0,2)\oplus L(0,1)=(0,2)\oplus(0,1)`$；分量各自异或：$`0\oplus0=0`$、$`0b10\oplus0b01=0b11=3`$，得 $`(0,3)`$。全部只用异或，无需任何乘法。）

**注意一个白送的推论**（`LinearLayout.h:61–62` 逐字）：无论 bases 怎么选，$`L(0,0)`$ 永远是 $`(0,0)`$——因为 $`L(0,0)=L(x\oplus x)=L(x)\oplus L(x)=0`$。线性函数必过原点。

**填满后的整张表**（`LinearLayout.h:66–70`，A 档逐字）：

```
              t/w   0     1     2     3
              0  (0,0) (0,1) (0,2) (0,3)
    L(t,w) =  1  (1,1) (1,0) (1,3) (1,2)
              2  (2,2) (2,3) (2,0) (2,1)
              3  (3,3) (3,2) (3,1) (3,0)
```

头注释点破它的身份（`LinearLayout.h:72–74` 逐字）：

> *Careful readers will recognize this as a classic "swizzled" layout where (t, w) -> (t, w ⊕ t).  To go from this formula to an LL, you only need to compute the results at input points (0,1), (0,2), (1,0), and (2,0).*

**这就是整章的顿悟**：一个经典的 swizzle 布局 $`(t,w)\mapsto(t,w\oplus t)`$，本来要写一个专门的公式/查表，现在只需 **4 个基向量 + 一条异或律**就完整表示，且**读者可以亲手把 16 格全部验算出来**。

**为什么线性律成立 = 为什么是 GF(2)。** 头注释的「数学背景」小节点明（`LinearLayout.h:183–277`，A 档逐字摘要）：把线性函数写成 $`L(a)=a_1 B_1 \oplus a_2 B_2 \oplus \dots \oplus a_M B_M`$，其中标量取自域 $`\mathbb{F}`$。普通线性代数里 $`\mathbb{F}`$ 是实数；LL 里 $`\mathbb{F}`$ 取 **GF(2)**——两元素域 $`\{0,1\}`$，**加法定义为 xor，乘法定义为按位 and**（`LinearLayout.h:218–221` 逐字）。于是每个输入 bit 要么「选进」对应的 base（该 bit=1）、要么不选（=0），把选中的 bases 异或起来就是结果。bases $`B_1..B_M`$ 正是 $`L`$ 在 $`F(1),F(2),F(4),F(8)`$ 这些 2 的幂次点上的值（`LinearLayout.h:255–259` 逐字）。**xor 线性律不是巧合，是「GF(2) 上线性」的直接后果。**

高维（MD→ND）时，源码把多维输入的 bit「摞成」1D，做常规 1D 计算，再「拆回」ND（`LinearLayout.h:261–275`，A 档逐字摘要），此时线性律退化成最朴素的 $`L(x\oplus y)=L(x)\oplus L(y)`$。

---

## 5. 复合与求逆 = GF(2) 上的比特矩阵行化简（RREF）

布局既然是 GF(2) 线性函数，就能写成**比特矩阵**：M 个输入 bit、N 个输出 bit，矩阵 $`B`$ 是 $`N\times M`$、每列是一个 base 向量，$`L(a)=Ba`$（矩阵-向量乘，加法是 xor、乘法是 and）。头注释给了 4×4 的逐位乘例（`LinearLayout.h:223–253`，A 档逐字），并指出更紧凑的写法：把每列 base 当成一个 N-bit 整数、输入当成 M-bit 整数，一次乘法就是「按输入的每个 1-bit 挑出对应列、全部 xor」。

源码把这套表示**落实**在 `getMatrix`——它把一个 LinearLayout 压成 $`\sum\text{outDimLog2} \times \sum\text{inDimLog2}`$ 的比特矩阵，每列对应一个 base（`LinearLayout.cpp:65–113`，A 档；注释 `:79–93` 给了一个把 4 个 base 摆成 3×4 比特矩阵的逐列例子）。矩阵每行打包进一个 `uint64_t`，因此 `assert(numCols <= 64)`（`LinearLayout.cpp:76–77`）——LL 不处理超过 64 bit 的巨型布局。

有了比特矩阵，**布局的复合与求逆就退化成矩阵运算**：

- **复合** `L.compose(outer)` = 先跑 `L` 再跑 `outer`，即 $`(O\circ L)(x)=O(L(x))`$（`LinearLayout.h:620–637` 契约逐字）。实现上就是把 `L` 的每个 base 喂给 `outer` 求值、得到新 bases（`LinearLayout.cpp:813–841`）——对应比特矩阵相乘。

- **求逆 + 复合** `A.invertAndCompose(B)`：求出 $`C`$ 使 $`A(x)=B(C(x))`$，若 $`B`$ 可逆则 $`C(x)=B^{-1}(A(x))`$（故得名，`LinearLayout.h:639–643` 逐字）。典型用途：寄存器布局 $`R`$、共享内存布局 $`S`$，要把寄存器里的张量值存进 shared memory，就需要「$`(\text{lane},\text{warp})\to`$ shmem offset」= $`R.\text{invertAndCompose}(S)`$（`LinearLayout.h:645–656` 逐字摘要）——这正是布局转换的核心操作。

**求逆怎么做？把两个布局的比特矩阵横向拼起来，做高斯消元（RREF）。** `invertAndCompose` 把 `matOuter` 和 `matThis` 水平拼接成一个大矩阵，然后（`LinearLayout.cpp:905–913`，A 档逐字摘要）：

> *Perform Gaussian elimination on `m`.  Because `outer` was modified to be bijective, the first half of the matrix should be the identity matrix.  The remaining half are the bases for the combined transformation.*

RREF 把左半化成单位阵（相当于对 `outer` 求逆），右半自动变成复合后的 bases。求秩（判断布局是否满射/可逆）同样靠 RREF——`getMatrixRank` 化简后数非零行即秩（`LinearLayout.cpp:139–159`，A 档逐字）。

**执行 RREF 的引擎**是仓库内 `third_party/f2reduce` 的唯一函数（三处调用：`LinearLayout.cpp:151` 求秩、`:912` 求逆复合、`:966` `getFreeVariableMasks` 求自由变量掩码）：

```cpp
f2reduce::inplace_rref_strided(m.get(), numRows, numCols, /*stride=*/1);
```

`f2reduce.h:7–24`（A 档逐字）定义它：「Converts a matrix over F_2 into row-reduced echelon form」，原地覆盖，little-endian 位打包（第 i 行第 j 列 = `(matrix[i*stride + k] >> j) & 1`）。**布局代数的全部重活——复合、求逆、求秩——最终都归到这一个 GF(2) RREF 调用上。**

---

## 6. Four Russians：f2reduce 为什么快（背景盒）

> **[背景盒 · Four Russians]** `f2reduce` 的 README（**在仓库内**，`third_party/f2reduce/README.md:1–7`，A 档逐字）自陈它是「a MIT-licenced library for Gaussian elimination over GF(2)」，把二进制矩阵化成 RREF，并采用「**Kronrod's algorithm ('method of four Russians')**」等优化。
>
> Method of Four Russians（C 档，Wikipedia 已 WebFetch 核实）：由苏联学者 **Arlazarov、Dinitz、Kronrod、Faradžev 于 1970 年**提出，是「加速涉及布尔矩阵（或每格取值有限的矩阵）的算法」的技术。核心手法：把矩阵切成小的 $`t\times t`$ 块、对每块**预建查表**，用查表替代逐格运算——从而把标准高斯消元/布尔矩阵乘法的复杂度**削去约一个对数因子**（取 $`t=\log n`$ 时，处理量从 $`n^2`$ 级降到 $`n^2/(\log n)^2`$ 级，整体消元由 $`O(n^3)`$ 量级降到约 $`O(n^3/\log n)`$）。README 还注明：不用 Strassen，故大满秩矩阵会被 M4RI 反超，但在**小/宽/低秩矩阵上更快**（`README.md:17–25`，A 档逐字摘要）——而 LL 的比特矩阵正好是**小矩阵**（≤64 列），命中 f2reduce 的强项。
>
> 一句话：**布局复合/求逆 = GF(2) 上比特矩阵的 RREF；f2reduce 用 Four Russians 的分块查表法把这个 RREF 做得又小又快。** 你不需要懂这套加速就能用 LL（`LinearLayout.h:185–186` 逐字：*You shouldn't need to understand this math to use linear layouts, but it helps with the implementation.*），但它解释了「统一抽象」为何在编译期也**跑得起**。

---

## 7. 收束：为什么一个抽象能统一 transpose / swizzle / broadcast / 全部布局

把前六节串起来，「统一」不是口号而是机制：

1. **任意布局 = 一组 bases**（§3）——Blocked、MMA、Shared、transpose、broadcast 全被压成同一种数据：几个基向量。
2. **求值 = xor 线性律**（§4）——补全任何输入都只用异或，无需每种布局各写一套 index 公式。
3. **transpose = 对调 base 分量；broadcast = 把某维 base 置零；swizzle = base 里带上 $`t\oplus w`$ 那样的交叉项**（§3–4 的例子逐字可核）——过去的「特殊情况」变成「换一组 bases」。
4. **布局之间的转换 = 比特矩阵复合 / 求逆 = GF(2) 上 RREF**（§5）——$`N^2`$ 种「A→B」转换代码坍缩成一个 `invertAndCompose`，消除二次爆炸（论文 B 档语）。
5. **RREF 由 f2reduce/Four Russians 快速完成**（§6）——统一抽象在编译期可承受。

这正是 LinearLayout.h 头注释的收尾承诺（`LinearLayout.h:76–80` 逐字，已在 §1 引）：**用「一般情况」表示转置与 swizzle，消掉所有特殊布局类，写更通用的代码。** 论文把同一件事写成 $`\mathbb{F}_2`$ 上的鲁棒代码生成（B 档）。ch20 说「布局是函数」，ch23 说清了**这个函数是 GF(2) 上的线性函数**——于是它能被几个基向量压缩、被异或求值、被矩阵行化简复合求逆。**这就是「一个抽象统一所有布局」。**

---

## 8. key_figures（每张标 grounding 层级；已写入 meta.json）

1. **`fig-4x4-swizzle-fill`（核心顿悟图，A 档）** — 4×4 swizzle 填表：左边只标出 4 个 base 格（`L(0,1)=(0,1)`、`L(0,2)=(0,2)`、`L(1,0)=(1,1)`、`L(2,0)=(2,2)`），中间用异或律逐格推导（高亮 §4 的 4 条 `L(0,0)/L(0,3)/L(3,0)/L(3,3)` 演算），右边是填满的整表并标注其身份 $`(t,w)\mapsto(t,w\oplus t)`$。**这是全章顿悟图，读者应能照图亲手验算。** grounding：A（`LinearLayout.h:36–74`）。
2. **`fig-bases-xor-linear`（bases + xor 线性律机制图，A 档）** — 把「输入的每个 1-bit 挑选对应 base、全部 xor」画成流程：输入整数的比特分解 → 选中的 base 列 → 异或求和 = 输出。配 1D 恒等 `[1,2,4]` 与 transpose 两组 bases 对比，点明「换一组 bases = 换一种布局」。grounding：A（`LinearLayout.h:106–148, 242–259`）。
3. **`fig-compose-invert-rref`（复合=矩阵乘 / 求逆=RREF 比特矩阵图，A 档）** — 上半：布局 → 比特矩阵（每列一个 base，`getMatrix` 的 3×4 例），复合 = 两矩阵相乘；下半：`invertAndCompose` 把 `matOuter|matThis` 横向拼接、经 `f2reduce` RREF 后左半成单位阵、右半是复合 bases。标注两处 `inplace_rref_strided` 调用点。grounding：A（`LinearLayout.cpp:79–113, 887–913`）。
4.（可选）**`fig-four-russians-block-lookup`（Four Russians 背景图，C 档）** — 矩阵切成 $`t\times t`$ 小块 + 预建查表替代逐格消元的示意，标出「削一个对数因子」。grounding：C（Wikipedia Method of Four Russians）+ A（`README.md:7`）。

---

## 附：A 档引用锚点清单（逐字可核）

| 论断 | 文件:行 |
|---|---|
| LL = 硬件位置 → 逻辑张量索引 | `LinearLayout.h:22–29` |
| 想法归功 Adam P. Goucher | `LinearLayout.h:20` |
| key fact：只需 2 的幂次输入上的取值 | `LinearLayout.h:31–34` |
| 4×4 swizzle 选 4 个 base | `LinearLayout.h:40–44` |
| 这些特殊值叫 bases / 基向量 | `LinearLayout.h:46–47` |
| xor 线性律 $`L(a\oplus b)=L(a)\oplus L(b)`$ | `LinearLayout.h:49–51` |
| 亲手填 4 格演算 | `LinearLayout.h:56–59` |
| $`L(0,0)=(0,0)`$ 必然 | `LinearLayout.h:61–62` |
| 填满的整表 + swizzle 身份 $`(t,w)\mapsto(t,w\oplus t)`$ | `LinearLayout.h:66–74` |
| 消除特殊情况、写通用代码 | `LinearLayout.h:76–80` |
| 输入维 (reg, thread, warp, block) | `LinearLayout.h:82–85` |
| 其它布局的 bases 例（恒等/零/transpose/broadcast） | `LinearLayout.h:106–148` |
| 为何映射方向是硬件→逻辑（保证是函数） | `LinearLayout.h:169–181` |
| GF(2)：加法=xor、乘法=and；bases=F 在 2 幂次点取值 | `LinearLayout.h:183–259` |
| 高维摞成 1D、1D 线性律 | `LinearLayout.h:261–275` |
| `bases[inDim][i] = L(...2^i...)`，其余 xor 补全 | `LinearLayout.h:314–320` |
| `compose` = $`O\circ L`$ 契约 | `LinearLayout.h:620–637` |
| `invertAndCompose` = $`B^{-1}(A(x))`$ 契约 + R/S 例 | `LinearLayout.h:639–672` |
| `getMatrix`：布局压成比特矩阵（每列一 base） | `LinearLayout.cpp:65–113` |
| `getMatrixRank`：RREF 后数非零行 | `LinearLayout.cpp:139–159` |
| RREF 调用（求秩） | `LinearLayout.cpp:151` |
| `compose` 实现（base 逐个求值） | `LinearLayout.cpp:813–841` |
| `invertAndCompose`：拼接矩阵 + RREF，左半=单位阵、右半=复合 bases | `LinearLayout.cpp:843–922` |
| RREF 调用（求逆复合） | `LinearLayout.cpp:912` |
| `getFreeVariableMasks`：RREF 求自由变量掩码 | `LinearLayout.cpp:958–966` |
| RREF 调用（自由变量掩码） | `LinearLayout.cpp:966` |
| f2reduce = GF(2) RREF；Four Russians（Kronrod） | `third_party/f2reduce/README.md:1–7` |
| `inplace_rref_strided` 签名 + 位打包语义 | `third_party/f2reduce/f2reduce.h:7–24` |
| `toLinearLayout` 统一入口签名与输入维注释 | `LinearLayoutConversions.h:13–45` |

## 附：B/C 档核实记录

- **B 档** arXiv:2505.23819（WebFetch 核实）：标题《Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $`\mathbb{F}_2`$》，作者 Keren Zhou, Mario Lezcano, Adam Goucher, Akhmed Rakhmati, Jeff Niu, Justin Lebar, Pawel Szczerbuk, Peter Bell, Phil Tillet, Thomas Raoux, Zahi Moudallal。摘要逐字确认：*models tensor layouts using linear algebra over $`\mathbb{F}_2`$*、*binary matrices acting on the bits of the hardware representation*、*generic layout-to-layout conversions, eliminating the quadratic explosion*、*integrate linear layouts with Triton*。（与 ch20 已抓事实一致，复用。）
- **C 档** Method of Four Russians（Wikipedia，WebFetch 核实）：Arlazarov, Dinitz, Kronrod, Faradžev，1970；分块 $`t\times t`$ + 查表；取 $`t=\log n`$ 时查表构建 $`O(n(\log n)^2)`$、处理 $`O(n^2/(\log n)^2)`$，整体较标准消元削去一至两个对数因子。M4RI 库实现其 F₂ 变体。
