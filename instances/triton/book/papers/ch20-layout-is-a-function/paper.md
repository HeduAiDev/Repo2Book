# ch20 论文包 —《布局即函数：GPU 张量凭什么和普通张量不同》

> 本章定位：**概念地基 primer**。核心命题只有一句——TritonGPU 张量比普通张量多出的 `encoding` 属性，其**正式定义就是一个函数 $`\mathcal{L}`$**，把张量的多维索引映射到「允许访问该处数据的 CUDA 线程集合」。
> 这句定义**白纸黑字写在源码 `TritonGPUAttrDefs.td`**，因此本章的**主真相源是源码（A 档），不是论文**。论文只用于结尾一个 3–5 句的「前瞻框」，把「$`\mathcal{L}`$ 其实是 GF(2) 上的线性映射」这条深线索引到第 23 章。**GF(2) 的线性律、bases、RREF、Four Russians 全部是 ch23 的内容，本章不展开。**

---

## 0. 来源层级表（防越档编造）

| 档 | 含义 | 本章用到的具体来源 | 用法 |
|---|---|---|---|
| **A** | 源码逐字 / 源码定义（最高权威，本章主真相源） | `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`（layout=函数 $`\mathcal{L}`$ 定义、distributed/shared 分野、4 级层次、wrap-around/broadcast、Blocked 三元组示例、CTALayout）；`include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td`（模块契约 num-warps / num-ctas / threads-per-warp）；`include/triton/Tools/LinearLayout.h` 头注释、`lib/Tools/LinearLayout.cpp`（前瞻框的 GF(2)/f2reduce 事实，均为**源码注释**级证据） | 正文核心定义与所有结构性论断**逐字引 .td**；前瞻框的技术事实引 LinearLayout.h/.cpp 注释 |
| **B** | 论文权威转述 | arXiv:2505.23819《Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using $`\mathbb{F}_2`$》（Zhou, Lezcano, Goucher, …, Tillet, Raoux 等） | **仅**用于前瞻框：为「layout = GF(2) 线性映射」这句提供论文级出处；**不在 ch20 展开** |
| **C** | 官方文档 / 百科 | Method of Four Russians（Wikipedia / M4RI 库）；f2reduce 为 Triton `third_party` 中的 GF(2) RREF 实现 | **仅**用于前瞻框末句点名 RREF 引擎，一句带过 |

> 红线：本包只登记**已核实**的内容。A 档所有引文都能在下方标注的 .td / .h / .cpp 行号处逐字核对；B/C 档只服务前瞻框，越档展开 GF(2) 代数即侵占 ch23。

---

## 1. 动机：普通张量只有 shape + dtype，GPU 张量为什么还要 `encoding`

一个普通张量（NumPy / PyTorch CPU 张量）回答两个问题就完整了：**形状**（shape，几维、每维多长）和**元素类型**（dtype，fp16 / i32…）。数据躺在一段连续内存里，「谁来读它」不是张量类型的一部分——CPU 上就是那一个（或几个）线程顺序访问。

GPU 不是这样。同一个张量的元素被**成百上千个 CUDA 线程同时持有**：某个元素落在哪个线程的寄存器里、或落在共享内存里被哪些线程可见，直接决定了访存是否合并（coalescing）、是否有 bank conflict、MMA 指令能否喂到正确的数据。**「数据如何在线程间切分」本身就是必须固化进类型的信息。** 这就是 TritonGPU 张量比普通张量多出的第三样东西——`encoding`（布局属性）。

.td 开篇第一段就把这层动机点破（`TritonGPUAttrDefs.td:35–36`，A 档逐字）：

> *TritonGPU tensors differ from usual tensors in that they contain a `_layout_` attribute which determines how the data should be partitioned across CUDA threads.*

**教学锚点**：整章要把这个抽象的「partition across threads」变成一张**可以逐格核对的表**——把张量画成格子，每格填上「持有该格的线程号」。

---

## 2. 核心定义：$`\mathcal{L}`$ —— 索引 → 线程集合（A 档，本章基石）

.td 用一句话给出正式定义（`TritonGPUAttrDefs.td:36–38`，A 档逐字）：

> *Formally speaking, we define a layout as a function $`\mathcal{L}`$ that maps a multi-dimensional tensor index $`i \in \mathbb{Z}^d`$ to a set of integers $`T`$ corresponding to the indices of the CUDA threads allowed to access some data at index $`i`$.*

写成数学：

```math
\mathcal{L}:\ \mathbb{Z}^d \longrightarrow \mathcal{P}(\{\text{thread ids}\}), \qquad i \longmapsto \mathcal{L}(i) \subseteq \{\text{允许访问 }T[i]\text{ 的线程集合}\}
```

注意**值域是线程的集合**（不是单个线程）——一个元素可以同时被多个线程持有（见 §5 broadcast）。

.td 紧接着给了一个可以逐格核对的例子（`TritonGPUAttrDefs.td:40–50`，A 档逐字）：

```
L(0, 0) = {0, 4}
L(0, 1) = {1, 5}
L(1, 0) = {2, 6}
L(1, 1) = {3, 7}
```

含义（.td 原文）：`T[0,0]` 同时归线程 0 和 4；`T[0,1]` 归线程 1 和 5；依此类推。**这就是本章的「顿悟图」原型**——把张量的每个索引位置画成格子，格子里填 $`\mathcal{L}(i)`$ 这个线程集合，抽象函数就变成一张看得见、能对账的表。

`encoding` 属性存的正是「如何构造这个 $`\mathcal{L}`$」的参数。定义完立刻分野（`TritonGPUAttrDefs.td:52`，A 档逐字）：

> *Right now, Triton implements two main classes of layouts: shared, and distributed.*

---

## 3. 两大类分野：distributed（寄存器） vs shared（共享内存）

| | **Distributed 布局** | **Shared 布局** |
|---|---|---|
| 数据住在哪 | 分散在各线程的**寄存器**里 | **共享内存**（shared memory） |
| $`\mathcal{L}(i)`$ 的形态 | 由 4 级层次计算得出（见 §4），一般是小集合 | 对**所有**索引 $`i`$，$`\mathcal{L}(i)`$ = block 内全部线程 |
| 源码定义 | `DistributedEncodingTrait` / `BlockedEncodingAttr` … | `SharedEncodingAttr` |
| 典型用途 | 寄存器级计算、访存合并、MMA 操作数 | 跨线程共享、swizzle 消 bank conflict |

**Shared 的 $`\mathcal{L}`$ 极简**（`TritonGPUAttrDefs.td:158–161`，A 档逐字）：

> *An encoding for tensors whose elements may be simultaneously accessed by different cuda threads in the programs, via shared memory. In other words, for all indices $`i \in \mathbb{Z}^d`$, $`\mathcal{L}(i) = \{0, 1, ..., 32*\text{num\_warps} - 1\}`$.*

也就是说 shared 布局里**每个元素对 block 内所有线程可见**（`32*num_warps` = 每 warp 32 线程 × warp 数 = block 线程总数，与 §6 模块契约对齐）。shared 布局额外携带 swizzle 参数（vec / perPhase / maxPhase / order）用来打乱列序、避免 bank conflict——.td:163–237 给了 5 个 xor swizzle 的逐格示例（本章只需点到，swizzle 细节非核心命题）。

**Distributed 的 $`\mathcal{L}`$ 由层次结构算出**——这是下一节的主线。

---

## 4. Distributed 的四级层次：CTA → Warp → Thread → Value（A 档核心）

`DistributedEncodingTrait` 的描述给了四级层次的**白纸黑字定义**（`TritonGPUAttrDefs.td:470–471`，A 档逐字）：

> *The Distributed encoding describes the layout $`\mathcal{L}`$ with the 4-level compute hierarchy on GPU. It is abstracted from the top to the bottom as CTAs Per CGA -> Warps Per CTA -> Threads Per Warp -> Values Per Thread.*

四级从粗到细（顶 → 底）：

| 级 | 名字 | 参数（Blocked 中） | 语义 |
|---|---|---|---|
| 1（顶） | **CTAs Per CGA** | `CTAsPerCGA` | 张量在一个 CGA（thread group cluster，Hopper 引入）里如何分给多个 CTA（thread block） |
| 2 | **Warps Per CTA** | `warpsPerCTA` | 一个 CTA 内如何分给各 warp |
| 3 | **Threads Per Warp** | `threadsPerWarp` | 一个 warp 内如何分给 32 个 lane（线程） |
| 4（底） | **Values Per Thread** | `sizePerThread` | 每个线程自己持有多少个（连续）元素 |

上两级（CTA、Warp）的 id 分配规则 .td 给了确定式（`TritonGPUAttrDefs.td:473–481`，A 档逐字）：

> *For CTAs Per CGA and Warps Per CTA level, the linear id is distributed contiguously with the shape and order.*

并给出 shape=[4,4]、order=[0,1] 的逐格 linear-id 示例（列优先填号）：

```
layout = [0  4  8  12]
         [1  5  9  13]
         [2  6  10 14]
         [3  7  11 15]
```

底两级（Thread、Value）的具体 id 分布**因子类而异**（.td:483：*variant for each sub-class encoding*）——`BlockedEncodingAttr` 是最直白的一种：三元组 `sizePerThread` / `threadsPerWarp` / `warpsPerCTA` 直接给出「每个线程 / 每个 warp / 每个 CTA 各拥有多少元素」（`TritonGPUAttrDefs.td:596–599`，A 档逐字）。

.td:601–619 的 Blocked 示例可逐格核对——一个 16×16 张量、2 warp（64 线程）、`sizePerThread={2,2}`、`threadsPerWarp={8,4}`、`warpsPerCTA={1,2}`：

```
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...
```

每格填的是线程号，`sizePerThread={2,2}` 让每个线程占一个连续 2×2 小块（故每个号出现 2×2 次）——**这正是 §2 顿悟图在真实 Blocked 布局上的落地**。

---

## 5. broadcast 与 wrap-around：当张量与布局形状不匹配

Distributed 的 $`\mathcal{L}`$ 由一个 $`d`$ 维张量 $`T`$（布局张量，形状可与被编码张量不同）完全刻画。.td 给出精确的映射公式（`TritonGPUAttrDefs.td:540–557`，A 档逐字）：

> *Distributed encodings have a layout function $`\mathcal{L}`$ that is entirely characterized by a d-dimensional tensor $`T`$. … when the tensor dim size `T.shape[d]` is larger than the layout dim size `L.shape[d]`, on that particular dim, we distribute values … in a **"wrapped around"** manner, with each thread owning multiple values. OTOH, when the tensor dim size … is smaller … we distribute … in a **"broadcasted"** manner, with each value owned by multiple threads.*

两种语义对称：

- **wrap-around**（张量比布局大）：布局线程号在该维**循环复用**，**一个线程持有多个元素**。
- **broadcast**（张量比布局小 / 布局比张量大）：**一个元素被多个线程同时持有**——正好解释了 §2 里 $`\mathcal{L}(0,0)=\{0,4\}`$ 那种「集合而非单点」的值。

.td:559–569 的示例逐格可核对——张量 $`T`$ 是 2×8、布局 $`L`$ 是 4×4：

```
T = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

L(T) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]
```

- 列方向：$`T`$ 宽 8、$`L`$ 宽 4 → wrap-around，前 4 列的线程号在后 4 列**重复出现**。
- 行方向：$`T`$ 高 2、$`L`$ 高 4 → broadcast，每格是**两个线程的集合**（如 `{0,8}` = 线程 0 与 8 都持有该元素）。

**教学锚点**：broadcast / wrap-around 是「$`\mathcal{L}`$ 值为集合」这件事的两种来源，画在格子表里一眼可辨——broadcast 让一格里出现多个号，wrap-around 让同一个号在多格出现。

---

## 6. 模块契约：谁决定线程总数

$`\mathcal{L}`$ 的值域是「线程 id 集合」，那**总共有多少线程**？答案不在张量类型里，而在**模块级属性**上——TritonGPU module 携带三个契约属性（`TritonGPUDialect.td:24–46`，A 档逐字）：

| 属性 | 取值方法 | 缺省 | 含义 |
|---|---|---|---|
| `triton_gpu.num-warps` | `getNumWarps`：缺失则 `report_fatal_error`（**必须存在**） | 无（强制） | 每个 program 的 warp 数 |
| `triton_gpu.num-ctas` | `getNumCTAs`：缺失返回 1 | 1 | 每个 CGA 的 CTA 数（Hopper 多 CTA 特性） |
| `triton_gpu.threads-per-warp` | `getThreadsPerWarp`：缺失返回 32 | 32 | 每 warp 线程数 |

于是一个 block 的线程总数 = `num_warps × threads_per_warp`——这正是 §3 shared 布局里 `32*num_warps` 的来历（threads_per_warp 缺省 32）。**模块契约（num-warps / num-ctas / threads-per-warp）与四级层次（warpsPerCTA / CTAsPerCGA / threadsPerWarp）必须自洽**：布局参数描述「怎么分」，模块属性锁定「一共有多少个」。CTA 层的切分则由 `CTALayoutAttr`（`CTAsPerCGA` / `CTASplitNum` / `CTAOrder`，`TritonGPUAttrDefs.td:67–105`）承担，缺省为全 1（单 CTA），多 CTA 目前是实验特性。

---

## 7. 前瞻框：$`\mathcal{L}`$ 是 GF(2) 上的线性映射（3–5 句，深化见第 23 章）

> **[前瞻 · 第 23 章展开]** 到这里 $`\mathcal{L}`$ 还只是一张「索引 → 线程集合」的对照表；但它其实不是任意函数，而是 **GF(2)（二元域）上的线性映射**——Triton 源码把这套统一模型叫 **LinearLayout**（思路归功于 Adam P. Goucher，见 `LinearLayout.h` 头注释）。关键事实：你**只需给出 $`\mathcal{L}`$ 在「2 的幂次输入」上的取值**（称为 bases / 基向量），其余全部输入都能靠**异或线性律** $`\mathcal{L}(a \oplus b) = \mathcal{L}(a) \oplus \mathcal{L}(b)`$ 推出来（`LinearLayout.h` 逐字给了此律与推导示例）。因为是线性的，两个布局的**复合与求逆**就退化成 **GF(2) 上的矩阵行化简（RREF）**——源码里由 `third_party/f2reduce`（`inplace_rref_strided`，`LinearLayout.cpp:151`）用 **Four Russians** 分块查表法快速完成。学术出处见 arXiv:2505.23819《Linear Layouts … Using $`\mathbb{F}_2`$》。**这套 GF(2) 线性代数——bases、xor 线性律、RREF、Four Russians——是第 23 章的主角，本章到此为止，只需记住：布局不是杂乱的表，而是一个可以用几条基向量 + 异或就压缩表示的线性对象。**

---

## 8. 建议 key_figures（每张标 grounding 层级）

1. **`fig-layout-as-function-table`（核心顿悟图，A 档）** — 把张量画成格子、每格填 $`\mathcal{L}(i)`$ 线程集合。用 .td:40–50 的 $`\mathcal{L}(0,0)=\{0,4\}`$… 例子，把抽象函数 $`\mathcal{L}`$ 变成一张可核对的表。**这是全章的顿悟图。** grounding：A（`TritonGPUAttrDefs.td:40–50`）。
2. **`fig-distributed-vs-shared`（分野图，A 档）** — 左：distributed=元素分散在各线程寄存器（小集合）；右：shared=同一元素对 block 内全部 `32*num_warps` 线程可见。grounding：A（`.td:158–161` + `52`）。
3. **`fig-four-level-hierarchy`（四级层次图，A 档）** — 自顶向下 CTA→Warp→Thread→Value 的嵌套矩形，配 Blocked 三元组 `sizePerThread/threadsPerWarp/warpsPerCTA`，叠在 .td:601–619 的 16×16 逐格线程号表上。grounding：A（`.td:470–471, 596–619`）。
4.（可选）**`fig-broadcast-wraparound`（映射语义图，A 档）** — 用 .td:559–569 的 2×8 张量 / 4×4 布局，标出「列方向 wrap-around（号重复）」与「行方向 broadcast（格内多号 `{0,8}`）」。grounding：A（`.td:540–569`）。

---

## 附：A 档引用锚点清单（逐字可核）

| 论断 | 文件:行 |
|---|---|
| 张量多出 layout 属性、决定跨线程切分 | `TritonGPUAttrDefs.td:35–36` |
| $`\mathcal{L}`$: 索引 → 线程集合（正式定义） | `TritonGPUAttrDefs.td:36–38` |
| $`\mathcal{L}(0,0)=\{0,4\}`$… 顿悟例 | `TritonGPUAttrDefs.td:40–50` |
| 两大类：shared / distributed | `TritonGPUAttrDefs.td:52` |
| shared: 所有 $`i`$，$`\mathcal{L}(i)=\{0..32*\text{num\_warps}-1\}`$ | `TritonGPUAttrDefs.td:158–161` |
| 4 级层次 CTA→Warp→Thread→Value | `TritonGPUAttrDefs.td:470–471` |
| CTA/Warp 级 linear-id contiguous 分配 | `TritonGPUAttrDefs.td:473–481` |
| Blocked 三元组语义 | `TritonGPUAttrDefs.td:596–619` |
| wrap-around / broadcast 定义 | `TritonGPUAttrDefs.td:540–557` |
| wrap/broadcast 2×8 逐格示例 | `TritonGPUAttrDefs.td:559–569` |
| CTALayout（CTAsPerCGA/SplitNum/Order） | `TritonGPUAttrDefs.td:67–105` |
| 模块契约 num-warps/num-ctas/threads-per-warp | `TritonGPUDialect.td:24–46` |
| GF(2) 线性律 + bases（前瞻框） | `LinearLayout.h:16–90` |
| RREF via f2reduce（前瞻框） | `LinearLayout.cpp:151` |
