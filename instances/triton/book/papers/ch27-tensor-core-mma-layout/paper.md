# ch27 论文包 —《Tensor Core 与 MMA 布局：mma/dot-operand 编码为什么长这样》

> 本章定位：**AccelerateMatmul 的前置原理章（primer）**。第 20 章《布局即函数》把 layout 讲成一个函数 $`\mathcal{L}`$——「张量索引 → 允许访问该处的线程集合」；第 23 章把它证成 GF(2) 上的线性映射。到本章要回答一个更具体的问题：**Tensor Core 的 MMA 布局（`NvidiaMmaEncodingAttr` / `DotOperandEncodingAttr`）为什么长成现在这个怪样子？** 答案不是「Triton 设计者拍脑袋」，而是**被 NVIDIA warp 级 MMA 指令（`mma.sync.m16n8k16` 等）对操作数在寄存器 / 共享内存里的分布要求（fragment 布局）倒逼出来的**——warp 的 32 个线程必须各自持有 A/B/C 矩阵的一小片固定切块（fragment），编码里的每个字段（`warpsPerCTA` / `instrShape` / `versionMajor` / `opIdx` / `kWidth`）都是在**对齐这个硬件契约**。没命中 Tensor Core 的性能根因，大多就在这些字段没和 fragment 对齐。
>
> **本章的主真相源是源码（A 档）**：`include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td` 逐字给出 `NvidiaMmaEncodingAttr` / `DotOperandEncodingAttr` 的字段定义与文档注释（**包括一张逐字印在注释里的 MMAv2 accumulator 线程布局矩阵**）；`lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp` 逐段给出「MMA 版本怎么选、`warpsPerTile` 怎么定、`kWidth` 怎么取、MMAv3 为何把操作数搬进共享内存」的真实决策点；`lib/Dialect/TritonGPU/IR/Dialect.cpp` 给出 `getMMAv2RepForOperand` / `getSizePerThreadForOperand` 把 fragment 元素数**算死**的实现。NVIDIA PTX ISA 的 mma 指令文档只作**厂商规范出处（C 档）**——它没有 arXiv/DOI，是工程规范不是论文。
>
> **红线：只写已核实内容。** A 档一切引文都能在标注的 `.td` / `.cpp` 行号处逐字核对；C 档 PTX ISA 的 URL 与小节名**逐字印在源码注释里**（`.td:1066`、`:1102`、`AccelerateMatmul.cpp:505`、`:509`），可据此核对出处。**本包在组装时无联网 / WebFetch 能力**，故 PTX ISA 中 A/B 操作数 fragment 的**逐 lane（线程）→ 逐寄存器元素坐标表**未能在线逐字复制——凡涉及该细粒度坐标，一律**标「待核」并回指 PTX ISA 对应小节**，绝不凭记忆编造线程 / 寄存器对应关系。可从源码**逐字或逐算式**核到的（accumulator 线程矩阵、每线程元素数、每行线程数、`kWidth` 取值规则）照写，其余留白。

---

## 0. 来源层级表（防越档编造）

| 档 | 含义 | 本章用到的具体来源 | 用法 |
|---|---|---|---|
| **A** | 源码逐字 / 源码注释（最高权威，本章主真相源） | `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`：`NvidiaMmaEncodingAttr` 定义（`:1046`）、`versionMajor` / `versionMinor` 语义（`:1053–1059`）、MMAv1 warpTileSize=[16,16]（`:1064`）、**MMAv2 warpTileSize=[16,8] + 逐字 accumulator 线程矩阵**（`:1100–1126`）、参数列表 `versionMajor/versionMinor/warpsPerCTA__/CTALayout/instrShape`（`:1130–1137`）、`isVolta/isTuring/isAmpere/isHopper`（`:1204–1207`）、`DotOperandEncodingAttr` 定义（`:1306`）、`opIdx`(a=0,b=1) / `parent` / `kWidth` 语义与 MMAv3 操作数「几乎总在 shared」注释（`:1309–1322`）、参数列表（`:1324–1329`）、Ampere 下 `kWidth=32/bitwidth` 的默认 builder（`:1339–1341`）、`getContigPerThread` 按 `opIdx` 把 `kWidth` 摆到 K 维（`:1348–1359`）；`lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`：MMA 版本按算力选（`getMMAVersionSafe` `:26–45`）、`warpsPerTileV2`（`:47–104`）、`warpsPerTileV3` 最小单元 (4,1)（`:106–132`）、`getSharedMemoryMMAOperand` 把操作数搬进 shared（`:136–166`）、主重写 `matchAndRewrite` 版本分派（`:250–303`）、**MMAv3 → 操作数放 shared + `WarpGroupDotOp`**（`:313–321`）、fp8/mxfp 显式选 `kWidth=4/8`（`:474`、`:481`）、**「4 threads per row」+ PTX `#mma-16816` 出处注释**（`:504–516`）；`lib/Dialect/TritonGPU/IR/Dialect.cpp`：`getMMAv2RepForOperand`（`:2016–2040`，`shapePerWarp={1,16,8,4*64/bitwidth}` `:2021`）、`getSizePerThreadForOperand`（`:2145–2159`）、`kWidth` 校验器（`:1076–1092`）、`getThreadsPerWarp` 在 opIdx=1 交换 M/N（`:2164–2175`）；`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp`：mma 指令名表——s8/fp8 低精度走 `m16n8k32`（`:265` `...m16n8k32...satfinite.s32.s8.s8.s32`、`:271` `...m16n8k32...f32.e5m2.e5m2.f32`，佐证 §2.2『位宽越小 K 维越大』） | 所有核心论断——fragment 元素数、accumulator 线程矩阵、字段↔fragment 对应、`warpsPerTile` / `kWidth` 取法、MMAv3 搬 shared、低精度 K 维——**逐字引 `.td` / `.cpp`**；这是本章基石 |
| **C** | 厂商官方文档（NVIDIA PTX ISA，工程规范，无 arXiv/DOI） | NVIDIA PTX ISA，*Warp-level matrix multiply-accumulate*（`mma.sync`）：`docs.nvidia.com/cuda/parallel-thread-execution/index.html`——URL 与小节名（`mma.884` / `mma.16816` / `#mma-16816-a-f16` / `#mma-16816-c`）**逐字印在源码注释里**（`.td:1066/:1102`、`AccelerateMatmul.cpp:505/:509`） | 为「fragment 布局是硬件契约」提供**规范出处**；A/B 操作数逐 lane→逐寄存器坐标表**本包未能联网复制，标「待核」**，只写源码已逐字 / 逐算式给出的部分 |

> 红线：本包只登记**已核实**内容。A 档所有引文都能在标注的 `.td` / `.cpp` 行号处逐字核对（**源码里那张 accumulator 线程矩阵与 `getSizePerThreadForOperand` 的算式比任何二手叙述都权威**）；C 档 PTX ISA 的 URL / 小节名逐字取自源码注释。**PTX A/B fragment 的细粒度坐标表标「待核」，绝不编造。**

---

## 1. 动机：MMA 布局不是「设计」出来的，是被 fragment「倒逼」出来的

在 ch20 / ch23 之后，我们已经接受一件事：TritonGPU 张量比普通张量多一个 `encoding`，它规定「张量的每个元素，物理上落在哪个线程的哪个寄存器里」。对绝大多数布局（Blocked / Slice），这个规定是 Triton 自己**为了访存效率**挑的，有自由度。

**但 Tensor Core 布局没有自由度。** Tensor Core 只认一条 warp 级指令——`mma.sync`（如 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`）。这条指令由一个 warp 的 **32 个线程协同**执行一次 $`16\times8\times16`$ 的矩阵乘累加。硬件规定：执行前，A、B、C 三个矩阵的元素必须**已经**按一张固定的表分散在这 32 个线程的寄存器里——线程 $`t`$ 持有 A 的哪几个元素、B 的哪几个、累加器 C 的哪几个，**一位都不能错**。这张「线程 → 元素」的固定表，就是 **fragment 布局**。

于是 Triton 的处境是：要想让编译器发出 `mma.sync`，它**必须**先把参与 `tt.dot` 的张量，重排成 Tensor Core 要求的那张 fragment 表。`NvidiaMmaEncodingAttr` 与 `DotOperandEncodingAttr` 就是「把这张硬件 fragment 表编码成一个 layout」的产物——**它们长得怪，是因为它们在忠实抄写一份硬件契约，而不是在做优雅的软件设计。**

源码把这个「抄写关系」写得很直白。`NvidiaMmaEncodingAttr` 的文档注释开门见山（`TritonGPUAttrDefs.td:1049–1050`，A 档逐字）：

> *An encoding for tensors that have been produced by tensor cores.*

它由两个「代际参数」刻画（`:1052–1059`，A 档逐字，精简）：

> *- A 'versionMajor' which specifies the generation the tensor cores whose output is being partitioned: 1 for first-gen tensor cores (Volta), and 2 for second-gen tensor cores (Turing/Ampere).*
> *- A 'versionMinor' which indicates the specific layout of a tensor core generation ...*

关键词是 **partitioned**——这个 encoding 描述的是「tensor core 的输出**如何在 warp 的线程间切分**」。而切分规则由哪一代 tensor core 决定：不同代（Volta / Ampere / Hopper）的 `mma` 指令，fragment 表**不一样**，所以要用 `versionMajor` 区分。注释甚至把 PTX ISA 的 URL 和小节名**逐字钉在源码里**（`:1064–1067`，A 档逐字）：

> *For first-gen tensor cores, the implicit warpTileSize is [16, 16]. Note: the layout is different from the recommended in PTX ISA https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (mma.884 section, FP32 accumulator).*

以及 MMAv2（`:1100–1103`，A 档逐字）：

> *For second-gen tensor cores, the implicit warpTileSize is [16, 8]. Information about this layout can be found in the official PTX documentation https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (mma.16816 section, FP32 accumulator).*

**这两段注释就是本章的「源码 → PTX 契约」的实锤**：Triton 自己声明它的 MMA 布局对应 PTX ISA 的 `mma.884` / `mma.16816` 小节。下面先把这个 fragment 契约讲清（§2），再看 Triton 用哪些字段抄它（§3），最后看 Hopper 为什么把契约的一半搬到共享内存（§4）。

---

## 2. 核心：warp 级 mma 指令的操作数分布要求（fragment 布局）

以第二代 tensor core（Turing / Ampere）的主力指令 `mma.sync.m16n8k16` 为例。一次运算如下，由 32 个线程协同完成：

```math
C_{16\times 8} \mathrel{+}= A_{16\times 16}\, B_{16\times 8}
```

三块矩阵各自的 fragment 要求如下——**能从源码逐字 / 逐算式核到的照写，PTX 细粒度坐标表标「待核」**。

### 2.1 累加器 C（`m16n8`）：源码里逐字印着线程矩阵（A 档，可直接重绘）

最幸运的是 C 的 fragment 布局**逐字印在 Triton 源码注释里**。`NvidiaMmaEncodingAttr` 的注释给出「MMAv2、`blockTileSize=[32,16]`」时的线程矩阵 $`L`$（`TritonGPUAttrDefs.td:1105–1126`，A 档逐字，取 warp 0 左上 $`8\times8`$ 块）：

```
                warp 0
-----------------/\-------------
[ 0   0   1   1   2   2   3   3
[ 4   4   5   5   6   6   7   7
[ ..............................
[ 28  28  29  29  30  30  31  31
[ 0   0   1   1   2   2   3   3       ← 第 8 行起，线程 id 从 0 重复
[ 4   4   5   5   6   6   7   7
[ ..............................
[ 28  28  29  29  30  30  31  31
```

矩阵里的数字是**线程 id（lane，0–31）**，格子位置是 C 元素坐标 `(row, col)`。逐格读这张源码表（全部 A 档，无任何外部依赖）：

- **行 0**：线程 0 持有 `C(0,0)`、`C(0,1)`；线程 1 持有 `C(0,2)`、`C(0,3)`；……线程 3 持有 `C(0,6)`、`C(0,7)`。即**一行 8 列由 4 个线程分担，每线程 2 个连续列**。
- **行 1**（"4 4 5 5 6 6 7 7"）：由线程 4–7 分担 → **相邻行就换下一组** 4 个线程（行 2 是 lane 8–11、行 3 是 lane 12–15……行 7 是 lane 28–31）；8 行（行 0–7）正好用满 lane 0–31。
- **行 8 起线程 id 从 0 重复**（+8 行偏移）：说明线程 0 除了 `C(0,0)/C(0,1)`，还持有 `C(8,0)/C(8,1)`。

把这些拼起来：**一个 lane 在 `m16n8` 的 C 里持有 4 个 fp32 累加元素**，坐标为（其中 `g = lane>>2` 是组号、`h = lane&3` 是组内序）：

```math
(g,\,2h),\quad (g,\,2h{+}1),\quad (g{+}8,\,2h),\quad (g{+}8,\,2h{+}1)
```

这不是从 PTX 记忆里搬来的，而是**从源码那张矩阵一格一格读出来的**。它和 `NvidiaMmaEncodingAttr::getContigPerThread` 的断言一致（`:1240–1246`，A 档逐字）：最内维 `contigPerThread=2`（就是那「2 个连续列」）。

> **[fragment 抓手 · C]** 16×8 的 C 由 32 线程分持，每线程 4 个 fp32。**这正是 key_figure (a) 中 C-accumulator 半张图的逐字依据**（源码矩阵 `.td:1105–1126`）。

### 2.2 操作数 A / B（`m16n16` / `m16n8` 沿 K）：元素数从源码算死，逐 lane 坐标待核

A、B 的 fragment，源码没有像 C 那样印出整张线程矩阵，但把**每线程元素数**用算式钉死了。`getMMAv2RepForOperand` 定义每个 warp tile 的形状（`Dialect.cpp:2016–2040`，A 档逐字关键行 `:2021`）：

```cpp
SmallVector<int> shapePerWarp = {1, 16, 8, 4 * 64 / bitwidth};
```

这四个数是 `{batch, M=16, N=8, K=4*64/bitwidth}`。对 `f16`（`bitwidth=16`）K 维每 warp 覆盖 $`4\times64/16 = 16`$——正好是 `m16n8k16` 的 K=16。每线程沿各维持有的元素数由 `getSizePerThreadForOperand` 算死（`Dialect.cpp:2145–2159`，A 档逐字）：

```cpp
if (opIdx == 0) {            // A
  sizePerThread[rank - 2] = 2;
  sizePerThread[rank - 1] = 2 * kWidth;
} else if (opIdx == 1) {     // B
  sizePerThread[rank - 2] = 2 * kWidth;
  sizePerThread[rank - 1] = 1;
}
```

对 `f16`（下一节会算出 `kWidth=2`）：**A 每线程持 M 方向 2、K 方向 $`2\cdot2=4`$，共 8 个 f16**；核对 $`16\times16`$ 的 A 共 256 元素、除以 32 线程正好得 8，一致。**B 每线程持 K 方向 4、N 方向 1，共 4 个 f16**；核对 $`16\times8`$ 的 B 共 128 元素、除以 32 得 4，一致。这些**元素数全是 A 档算式的直接后果，可逐一验算**。

至于 A/B **每个 lane 具体持哪几个 `(row,k)` 坐标**——源码只给了一个结构性事实（`AccelerateMatmul.cpp:504–506`，A 档逐字）：

> *// For bf16, we have 4 threads per row*
> *// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-a-f16*

即 A 的每一行（沿 K=16）由 **4 个线程**分担，每线程持 K 方向连续 4 个元素（$`4\times4=16`$ ✓）。**再细到「哪个 lane 持哪几个坐标」的完整表，权威在 PTX ISA `#mma-16816-a-f16` / `#mma-16816-b-f16`——本包无联网未能逐字复制，标「待核」，绝不凭记忆填坐标。** key_figure (a) 中 A/B 半张图应由 illustrator 据 PTX ISA 该小节忠实重绘（元素数 / 每行线程数已由本节 A 档锁定，坐标以 PTX 为准）。

**本节小结**：C 布局源码逐字可绘；A/B 的**每线程元素数、每行线程数** A 档算死，逐 lane 坐标待核 PTX。这就是 fragment 契约。下一节看 Triton 的编码字段如何逐一对上它。

---

## 3. 展开：编码字段逐一对应 fragment 要求

`NvidiaMmaEncodingAttr` 与 `DotOperandEncodingAttr` 的字段，不是随意命名的旋钮，而是 fragment 契约的**逐项落点**。逐个对上。

### 3.1 `versionMajor` / `versionMinor` → 哪一代 tensor core 的 fragment 表

字段定义（`TritonGPUAttrDefs.td:1130–1137`，A 档逐字）：

```
let parameters = (
  ins
  "unsigned":$versionMajor,
  "unsigned":$versionMinor,
  ArrayRefParameter<"unsigned">:$warpsPerCTA__,
  "CTALayoutAttr":$CTALayout,
  ArrayRefParameter<"unsigned">:$instrShape
);
```

`versionMajor` 选**哪一代**的 fragment 表：1=Volta（`mma.884`，warpTileSize [16,16]）、2=Turing/Ampere（`mma.16816`，warpTileSize [16,8]）、3=Hopper（WGMMA）。`isVolta/isTuring/isAmpere/isHopper` 就是对它的判定（`:1204–1207`，A 档逐字）。**选错版本 = 抄错 fragment 表 = 生成非法 `mma`**，所以它排在字段第一位。版本本身怎么定，见 §3.4。

### 3.2 `instrShape` → 一条 mma 指令的 tile 形状（fragment 的「单位砖」）

`instrShape` 就是**单条 `mma` 指令的 $`[M,N]`$**（MMAv2 是 `[16,8]`，见 §2 注释里的 warpTileSize=[16,8]）。它是 fragment 的「单位砖」：整块 `tt.dot` 的输出会被切成若干块 `instrShape` 大小的砖，每块砖用一条 `mma.sync` 算。§2 那张 C 线程矩阵，画的就是**一块砖内**的线程分布。

### 3.3 `warpsPerCTA` / `warpsPerTile` → 多个 warp 怎么平铺整块输出 tile

一条 `mma` 只算一块 `[16,8]` 砖，一个 warp 一次也只做一块砖。但 `tt.dot` 的输出往往是 $`128\times128`$ 这种大 tile，需要**多个 warp 各包一片**、每片内再切成多块砖迭代。`warpsPerCTA` 就规定「这些 warp 沿 M/N 怎么摆」。它由 `getWarpsPerTile` 算出（`AccelerateMatmul.cpp:219–231`，A 档逐字）：

```cpp
static SmallVector<unsigned, 3>
getWarpsPerTile(DotOp dotOp, const ArrayRef<int64_t> shape, int version,
                int numWarps, const SmallVector<unsigned, 3> &instrShape) {
  switch (version) {
  case 2:
    return warpsPerTileV2(dotOp, shape, numWarps);
  case 3:
    return warpsPerTileV3(dotOp, shape, numWarps, instrShape);
  ...
```

MMAv2 的 `warpsPerTileV2` 从每 warp 的最小覆盖 `shapePerWarp = [16, 8]`（`:84–85`，A 档逐字——**恰好等于 `instrShape`**）出发，按输出 shape 沿两维**贪心翻倍**分配 warp（`:90–102`）：

```cpp
shapePerWarp[rank - 1] = 8;
shapePerWarp[rank - 2] = 16;
do {
  if (ret[0] * ret[1] >= numWarps) break;
  if (shape[0] / shapePerWarp[0] / ret[0] >= shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
    if (ret[0] < shape[0] / shapePerWarp[0]) ret[0] *= 2;
    else ret[1] *= 2;
  } else {
    ret[1] *= 2;
  }
} while (true);
```

**为什么要 `warpsPerTile`（why）**：因为一条 mma 只覆盖 `instrShape` 一小砖，`warpsPerTile` 是「把这些砖沿 M/N 平铺、铺满整块输出、并让每个 warp 分到尽量方正的一片」的分配表——目标是让每个 warp 承担相近的砖数、K 维归约尽量落在同一 warp 内。注意一个专门为 flash-attention 式**链式 dot** 加的分支（`warpsPerTileV2:74–80`，A 档逐字）：若检测到后续还有 dot，就退化成把 warp 全压到单轴（`{numWarps,1}` 或 `{1,numWarps}`），让归约留在同一 warp。MMAv3 的 `warpsPerTileV3` 同理，但最小不可分单元是 `(4,1)`（`:119–120`，A 档逐字：*For MMAv3, the smallest indivisible unit of warp shape is (4, 1).*）——因为 Hopper 的 WGMMA 是 **一个 warpgroup（4 warps）** 协同的指令，4 个 warp 绑在一起是硬件下限。

### 3.4 版本怎么选：`getMMAVersionSafe`（A 档逐字）

`versionMajor` 不是用户填的，是按 GPU 算力选的（`AccelerateMatmul.cpp:26–45`，A 档逐字）：

```cpp
static int getMMAVersionSafe(int computeCapability, DotOp op) {
  SmallVector<int> versionsSupported;
  if (computeCapability < 75) {
    versionsSupported = {1};                 // Volta
  } else if (computeCapability < 90) {
    versionsSupported = {2};                 // Turing/Ampere
  } else if (computeCapability < 100) {
    versionsSupported = {3, 2};              // Hopper: 优先 v3，回退 v2
  }
  for (int baseVersion : versionsSupported)
    if (supportMMA(op, baseVersion)) return baseVersion;
  ...
```

主重写循环 `matchAndRewrite` 按此分派（`:250–303`，A 档逐字关键行）：`getMMAVersionSafe` 定版本 → `mmaVersionToInstrShape` 定 `instrShape` → v1 走 Volta 专用 builder（`:291–294`），v2/v3 走 `getWarpsPerTile` + 通用 builder（`:296–302`）。**这一段把 §3.1–3.3 的字段全部实例化出来。**

### 3.5 `opIdx` / `kWidth` → 每线程沿 K 一次持几个元素（`DotOperandEncodingAttr`）

A、B 操作数的布局用 `DotOperandEncodingAttr`。它的注释（`TritonGPUAttrDefs.td:1309–1322`，A 档逐字，精简）说明了每个字段的意义：

> *given `d = tt.dot a, b, c` tt.dot's operands a and b must be of DotOperandEncodingAttr layout, if the dot is MMA v1 or v2 (i.e. pre-Hopper). ... a's opIdx is 0, b's opIdx is 1. The parent field is the layout of d. kWidth defines number of consecutive elements stored by one thread along k dimension.*

三个字段（`:1324–1329`）：

```
ins
"unsigned":$opIdx,                              // 0=A, 1=B
"Attribute":$parent,                            // = d 的 NvidiaMma 布局
DefaultValuedParameter<"unsigned", "0">:$kWidth // K 方向每线程连续元素数
```

- **`opIdx`**：区分这是 A（0）还是 B（1）。它决定 `kWidth` 摆在哪一维——`getContigPerThread`（`:1348–1359`，A 档逐字）对 opIdx=0 把 `kWidth` 放最内维（K），opIdx=1 放次内维；`getThreadsPerWarp`（`Dialect.cpp:2164–2175`）在 opIdx=1 时交换 M/N。这对应 §2.2 里 A 沿 K 连续、B 沿 K 连续但转置摆放的 fragment。
- **`kWidth`（本章的关键 why）**：**「一个线程沿 K 维一次连续持有几个元素」**——直接对应 fragment 里每线程的寄存器打包宽度。取值不是随意的：Ampere parent 的默认 builder 用（`TritonGPUAttrDefs.td:1339–1341`，A 档逐字）：

```cpp
unsigned bitwidth = eltTy.getIntOrFloatBitWidth();
unsigned MMAv2kWidth = 32 / bitwidth;
return $_get(context, opIdx, parent, MMAv2kWidth);
```

**`kWidth = 32 / bitwidth`**：这就是「让每个线程沿 K 一次装满一个 32-bit 寄存器」——f16（16-bit）→ `kWidth=2`（一个寄存器塞 2 个 f16）、fp8（8-bit）→ 4。为什么对齐 32 bit？因为 `ldmatrix` / `mma` 的操作数寄存器是 32-bit 粒度，K 方向凑成 32-bit 的连续段才能一次搬进 fragment、不浪费寄存器带宽。§2.2 里 `sizePerThread` 的 K 维 `2*kWidth`——那个因子 2 是因为 `m16n8k16` = 沿 K **叠两块** `m16n8k8`，每块贡献 `kWidth` 个连续元素。**校验器把这个契约钉死**（`Dialect.cpp:1076–1092`，A 档逐字精简）：Ampere MMA parent 下 `kWidth==0` 报错（*mandatory for Ampere MMA parent*）、非 Ampere 的 Nvidia parent 下 `kWidth!=0` 报错——**`kWidth` 是且仅是 Ampere fragment 的必填契约**。

低精度路径会**显式**挑更大的 `kWidth` 塞更多 K（`AccelerateMatmul.cpp:474`、`:481`，A 档逐字）：mxfp 的 `E2M1`（4-bit）用 `kWidth=4`、fp8 的 `E5M2/E4M3` 用 `kWidth=8`——位宽越低、一个 32/64-bit 寄存器能塞的 K 元素越多，`kWidth` 就越大。**这就是「为什么 kWidth 这样取」：它 = 每线程沿 K 的寄存器打包宽度，由元素位宽对齐硬件寄存器粒度倒推。**

---

## 4. 落地：MMAv3（Hopper WGMMA）为什么把操作数放共享内存

前三节都在讲 pre-Hopper（v1/v2）：A、B 以 `DotOperandEncodingAttr` 布局**待在寄存器里**，喂给 `mma.sync`。到 Hopper（`versionMajor==3`，WGMMA）**契约变了**：操作数**不再全从寄存器取，而是从共享内存取**。`DotOperandEncodingAttr` 的注释点破这一点（`TritonGPUAttrDefs.td:1312–1313`，A 档逐字）：

> *For MMA v3, the operands are \*almost always\* in a regular shared encoding, but sometimes the LHS is also a dot-operand encoding.*

`matchAndRewrite` 的 v3 分支就是这个「搬进 shared」的落点（`AccelerateMatmul.cpp:313–321`，A 档逐字）：

```cpp
if (versionMajor == 3) {
  auto eltType = dotOp.getA().getType().getElementType();
  // In MMAV3 tranpose is only supported for f16 and bf16.
  bool allowTranspose = eltType.isF16() || eltType.isBF16();
  a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
  b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
  newDot = taskIdRewriter.create<triton::nvidia_gpu::WarpGroupDotOp>(
      dotOp.getLoc(), newRetType, a, b, newAcc, nullptr,
      dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc(), false);
}
```

对比 v2 分支（`:322` 起）是把 A/B 转成 `DotOperandEncodingAttr`（寄存器 fragment），**v3 分支两个操作数都过 `getSharedMemoryMMAOperand`**——把它们分配进共享内存（`:136–166`，A 档逐字：`SharedEncodingAttr::get(...)` + `LocalAllocOp`），产物不是 `DotOp` 而是 **`WarpGroupDotOp`**。

**为什么 MMAv3 要搬 shared（why）**：Hopper 的 WGMMA 是一条 **异步、warpgroup 级（4 warps=128 线程协同）** 的指令，它直接从**共享内存**读操作数矩阵（可选从寄存器读 A），这样单条指令能吞下远大于 `m16n8k16` 的 tile（如 `m64nNk16`），且与访存**异步重叠**。因此编码层面：v3 的 A/B 不再需要「逐 lane 寄存器 fragment」那套 `DotOperandEncodingAttr`，而是常规 `SharedEncodingAttr`；warp 平铺的最小单元也从 v2 的单 warp 变成 v3 的 `(4,1)`=一个 warpgroup（§3.3 已述）。**这条 `WarpGroupDotOp` 正是第 24 章点名的 warp-group 级异步 dot——本章只讲清「为什么它的操作数在共享内存」这一 fragment 契约层面的根因，指令本身的异步流水与 lowering 由 ch24 展开，此处回指、不重讲。**

---

## 5. 收束：编码字段 = fragment 契约的逐项落点

把四节串起来，「MMA 布局为什么长这样」就有了机械答案：

1. **Tensor Core 只认 `mma.sync`，它要求操作数预先按固定 fragment 表分散在 32 个 lane 的寄存器里**（§2）——C 的线程矩阵源码逐字可绘，A/B 元素数 A 档算死。
2. **`versionMajor/Minor` 选哪一代的 fragment 表，`instrShape` 是单条指令的砖，`warpsPerCTA/warpsPerTile` 把砖平铺成整块输出 tile**（§3.1–3.4）——`warpsPerTile` 存在是因为一条 mma 只覆盖一小砖、要多 warp 铺满且归约留在 warp 内。
3. **`opIdx` 分 A/B，`kWidth = 32/bitwidth` = 每线程沿 K 的寄存器打包宽度**（§3.5）——低位宽塞更多 K，`kWidth` 就更大；这是对齐硬件 32-bit 寄存器粒度的直接后果。
4. **Hopper 的 WGMMA 换了契约：操作数从共享内存取，产物是 `WarpGroupDotOp`，warp 最小单元变 `(4,1)` 一个 warpgroup**（§4）——回指 ch24。

一句话：**`NvidiaMmaEncodingAttr` / `DotOperandEncodingAttr` 的每个字段，都是在把一份 NVIDIA fragment 硬件契约逐项抄进 layout。** 它们长得怪，是因为硬件本来就长这样。命中 Tensor Core、让 `tt.dot` 真正降级成 `mma.sync` 而不是回退到慢速 FMA 路径，前提就是这些字段全部对齐了 fragment——这也是本章的性能收益落点。

---

## 6. key_figures（每张标 grounding 层级；已写入 meta.json）

1. **`fig-m16n8k16-fragment`（核心 fragment 图，A+C 档，混合 grounding）** — `mma.sync.m16n8k16` 的 warp fragment 线程→寄存器映射：**C-accumulator 半张图逐字重绘源码矩阵**（`TritonGPUAttrDefs.td:1105–1126`，lane 0 持 `(g,2h)/(g,2h+1)/(g+8,2h)/(g+8,2h+1)`，A 档可核）；**A/B 半张图**标注每线程元素数（A=8×f16、B=4×f16，A 档由 `getSizePerThreadForOperand`/`getMMAv2RepForOperand` 算死）+「4 threads per row」结构（`AccelerateMatmul.cpp:504`），**逐 lane→逐寄存器坐标由 illustrator 据 PTX ISA `#mma-16816-a-f16`/`-b`/`-c` 忠实重绘（本包标「待核」，坐标以 PTX 为准，不得凭记忆填）**。grounding：A（`.td:1105–1126`、`Dialect.cpp:2016–2040/2145–2159`）+ C（PTX ISA `#mma-16816-*`，待 illustrator 联网核）。
2. **`fig-encoding-to-fragment`（编码字段↔fragment 对应图，A 档）** — 左列 `NvidiaMmaEncodingAttr{versionMajor/instrShape/warpsPerCTA}` 与 `DotOperandEncodingAttr{opIdx/kWidth}` 五个字段，右列 fragment 契约的对应项（版本→哪代 fragment 表、instrShape→单指令砖 [16,8]、warpsPerCTA→砖平铺、opIdx→A/B、kWidth=32/bitwidth→每线程 K 寄存器打包宽度），逐箭头连线并标源码锚点。grounding：A（`.td:1130–1137/1324–1341`、`AccelerateMatmul.cpp:47–104/219–231`、`Dialect.cpp:1076–1092`）。
3.（可选）**`fig-mmav3-operand-to-shared`（MMAv3 操作数搬 shared 对比图，A 档）** — 上半 v2：A/B 转 `DotOperandEncodingAttr`（寄存器 fragment）→ `DotOp`；下半 v3：A/B 过 `getSharedMemoryMMAOperand` → `SharedEncodingAttr`（共享内存）→ `WarpGroupDotOp`，warp 最小单元 (4,1)=warpgroup；标注回指 ch24。grounding：A（`AccelerateMatmul.cpp:136–166/313–321`、`.td:1312–1313`）。

---

## 附：A 档引用锚点清单（逐字可核）

| 论断 | 文件:行 |
|---|---|
| `NvidiaMmaEncodingAttr` = tensor core 输出的分区编码 | `TritonGPUAttrDefs.td:1046–1050` |
| `versionMajor`(1=Volta/2=Turing-Ampere) / `versionMinor` 语义 | `TritonGPUAttrDefs.td:1052–1059` |
| MMAv1 warpTileSize=[16,16] + PTX `mma.884` URL | `TritonGPUAttrDefs.td:1064–1067` |
| MMAv2 warpTileSize=[16,8] + PTX `mma.16816` URL | `TritonGPUAttrDefs.td:1100–1103` |
| **MMAv2 accumulator 线程矩阵（逐字，C fragment 依据）** | `TritonGPUAttrDefs.td:1105–1126` |
| `NvidiaMma` 参数列表 versionMajor/Minor/warpsPerCTA/CTALayout/instrShape | `TritonGPUAttrDefs.td:1130–1137` |
| `getContigPerThread` 最内维=2（C 每线程 2 连续列） | `TritonGPUAttrDefs.td:1240–1246` |
| `isVolta/isTuring/isAmpere/isHopper` 判定 | `TritonGPUAttrDefs.td:1204–1207` |
| `DotOperandEncodingAttr` 定义 + opIdx(a=0,b=1)/parent/kWidth 语义 | `TritonGPUAttrDefs.td:1306–1322` |
| MMAv3 操作数「几乎总在 shared」注释 | `TritonGPUAttrDefs.td:1312–1313` |
| `DotOperand` 参数列表 opIdx/parent/kWidth | `TritonGPUAttrDefs.td:1324–1329` |
| Ampere 默认 `kWidth = 32/bitwidth` builder | `TritonGPUAttrDefs.td:1339–1341` |
| `getContigPerThread` 按 opIdx 把 kWidth 摆到 K 维 | `TritonGPUAttrDefs.td:1348–1359` |
| `getMMAVersionSafe`：按算力选版本(<75→1,<90→2,<100→{3,2}) | `AccelerateMatmul.cpp:26–45` |
| `warpsPerTileV2`：shapePerWarp=[16,8]+贪心翻倍；链式 dot 压单轴 | `AccelerateMatmul.cpp:47–104` |
| `warpsPerTileV3`：最小不可分单元 (4,1)=warpgroup | `AccelerateMatmul.cpp:106–132` |
| `getSharedMemoryMMAOperand`：操作数→SharedEncoding+LocalAlloc | `AccelerateMatmul.cpp:136–166` |
| `getWarpsPerTile` 版本分派 (v2/v3) | `AccelerateMatmul.cpp:219–231` |
| `matchAndRewrite` 版本分派 + 字段实例化 | `AccelerateMatmul.cpp:250–303` |
| **MMAv3 分支：A/B 搬 shared + `WarpGroupDotOp`** | `AccelerateMatmul.cpp:313–321` |
| fp8/mxfp 显式选 kWidth=4(E2M1)/8(E5M2,E4M3) | `AccelerateMatmul.cpp:474, 481` |
| 「4 threads per row」+ PTX `#mma-16816-a-f16`/`-c` URL | `AccelerateMatmul.cpp:504–516` |
| `getMMAv2RepForOperand`：shapePerWarp={1,16,8,4*64/bitwidth} | `Dialect.cpp:2016–2040` |
| `getSizePerThreadForOperand`：A=[2,2*kWidth]/B=[2*kWidth,1] | `Dialect.cpp:2145–2159` |
| `kWidth` 校验器：Ampere MMA 必填、非 Ampere Nvidia 禁填 | `Dialect.cpp:1076–1092` |
| `getThreadsPerWarp`：opIdx=1 交换 M/N | `Dialect.cpp:2164–2175` |

## 附：B/C 档核实记录

- **B 档**：无。本章 MMA/fragment 布局是 NVIDIA **厂商工程规范**，无 arXiv/DOI 论文（区别于 ch23 的 LinearLayout 有正式论文）。meta.json 里挂的 arXiv:2209.05433（FP8 E4M3/E5M2）与 arXiv:2310.10537（OCP MXFP）是**元素格式**的先修框（供 §3.5 低位宽 `kWidth` 与后续类型章用），非 MMA 布局本身的出处。
- **C 档 · PTX ISA（部分待核）**：NVIDIA PTX ISA *Warp-level matrix multiply-accumulate*（`mma.sync`），`docs.nvidia.com/cuda/parallel-thread-execution/index.html`，小节 `mma.884` / `mma.16816`（`#mma-16816-a-f16` / `#mma-16816-b-f16` / `#mma-16816-c`）。**URL 与小节名逐字取自源码注释**（`TritonGPUAttrDefs.td:1066/1102`、`AccelerateMatmul.cpp:505/509`），出处可核。**本包组装时无联网 / WebFetch 能力**：C 累加器 fragment 已由源码那张线程矩阵（`.td:1105–1126`）逐字给出、无需 PTX；**A/B 操作数的逐 lane→逐寄存器元素坐标表未能在线逐字复制，全程标「待核」**——每线程元素数（A=8/B=4 f16）与「4 threads per row」结构已由 A 档算式 / 注释锁定，但**具体坐标须由 illustrator 在渲染 `fig-m16n8k16-fragment` 时据 PTX ISA `#mma-16816-*` 联网忠实重绘**，本 analyst 不凭记忆填任何坐标。
- **姊妹篇提示**（不展开）：`instances/triton-ascend/` 的对应 primer 会把本章的「NVIDIA PTX mma tile 模型」换成 **Ascend cube 单元的 tile 模型**——本章是「硬件 MMA tile 原理」这个通用章位的 **NVIDIA 实例**，fragment→编码字段的对应关系是配对脊柱的共享骨架。
