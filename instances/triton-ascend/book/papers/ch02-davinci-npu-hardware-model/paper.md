# 达芬奇（DaVinci）NPU 硬件模型：忠实综述

> 本文件是 ch02 原理章（kind=primer）的**真相源**，供 writer 叙事、illustrator 出图。
> 引用两类出处，读者/writer 须能逐一回溯：
> - **[HPCA'21] / [HotChips'19]** = 达芬奇架构论文（架构 why）。存在性/元数据 web-verified；
>   **论文正文本机未获取**（host 无网络 + DOI 版权），故凡具体 micro-arch 数值均标
>   `[paper-attributed, host 未复核]`——writer 联网时收口或按级别软化，**不得当作硬事实写死**。
> - **[src: <path>:Lxx]** = triton-ascend v3.2.1 源码内文档**逐字**引文（pin @ `2badfc89e`）。
>   这是**落地细节的 source-cited 真相源**，可在 pin blob 逐字复核，优先级高于记忆。
>
> 规范路径前缀 `docs/...`、`third_party/ascend/...`（勿写 `instances/.../source/`）。

---

## §1 为什么 NPU 不是 GPU：领域专用 vs 通用 SIMT

读者写下的 `tl.load` / `tl.dot` / `tl.store` 在 GPU 上跑和在昇腾 NPU 上跑，**底层硬件模型截然不同**。理解这本书后半部分每一个 pass（AutoBlockify、PlanMemory、CVPipeline、HIVM lowering）之前，必须先建立达芬奇的硬件心智模型——否则「后端为什么要这么切分、这么同步、这么融合」全部悬空。

**GPU（SIMT）心智模型**：成百上千个同构 SM，每个 SM 跑大量 warp，靠**超额订阅（oversubscription）**用线程级并行掩盖访存延迟；片上有硬件管理的 L1/L2 cache 与程序员部分管理的 shared memory，但 global→shared 的搬运在很多路径上是**隐式**的；grid 是**逻辑任务维**，`[n, m, l]` 等价于 `n×m×l` 个线程，每个线程对应一次 kernel 执行、只执行一次，物理映射交给硬件调度器。

**昇腾（达芬奇）心智模型**：少量**异构**的 AI Core，每个 AI Core 内部是**领域专用**的功能单元分工（矩阵单元 + 向量单元 + 标量单元），片上是**显式管理的多级 scratchpad**（不是 cache——没有自动换入换出），数据搬运必须由 kernel **显式写出**。grid 不再是逻辑维，而是**物理核拓扑的映射**：

> [src: docs/en/migration_guide/architecture_difference.md:5] "NPUs are strongly bound to physical cores in Triton multi-core parallelism. This represents a core difference from GPUs' logical dimension parallelism + automatic physical mapping in hardware."

> [src: docs/en/migration_guide/architecture_difference.md:15] "In NPUs, vector cores and cube cores belong to multiple physical cores. The number of cores varies with the generation of hardware. Each core executes only one block and can schedule the block execution repeatedly."

一句话对照（源码原表，[src: architecture_difference.md:9-12]）：GPU 的 grid 是「逻辑任务维（与物理核解耦）」，昇腾的 grid 是「物理核组映射（绑定 AI Core 拓扑）」；GPU 对 grid 维度/尺寸无硬限，昇腾要求 `Grid size ≤ 总 AI Core 数`、2D 还需拓扑匹配。

**架构动机（论文侧）**：达芬奇被设计为一个「可伸缩、统一」的深度神经网络计算架构——用领域专用的矩阵/向量单元换取远高于通用 SIMT 的能效比，代价是把访存与并行的管理责任显式交给编译器/程序员 `[HPCA'21]`。这正是为什么昇腾后端不是「把 GPU kernel 原样搬过来」，而要重写一整套结构化下降（Triton→Linalg→HFusion→HIVM）——把显式搬运、tiling、CV 分工在编译期做出来。

**读者此刻该记住的一句话**：GPU 靠「海量线程 + 隐式 cache + 逻辑 grid」；达芬奇靠「异构功能单元 + 显式 scratchpad + 物理 grid」。本书后面所有「为什么」都从这句话生长出来。

---

## §2 达芬奇 AI Core 结构：cube + vector + scalar 与 1:2 配比

### 2.1 三类功能单元

一个达芬奇 **AI Core** 内部有三类计算单元，各司其职 `[HPCA'21][HotChips'19]`：

- **Cube（矩阵）单元**：做矩阵乘累加。达芬奇的标志性设计是一个 **3D-Cube 脉动阵列（systolic array）**，一拍完成一批 `A×B` 的乘累加——这是把 matmul/卷积做到高能效的核心。`[paper-attributed, host 未复核：阵列的具体维度（常见公开口径为 16×16×16 FP16 MAC）与每拍 MAC 数须联网核 HPCA'21 后落笔，本机不写死具体数字]`
- **Vector（向量）单元**：做逐元素（elementwise）与归约（reduce）类计算——激活函数、加减乘除、softmax 的指数/求和、layernorm 的均值方差等。
- **Scalar（标量）单元**：做标量运算、地址计算、循环控制与流程调度（相当于核内的「小 CPU」）。

源码文档对分工的**逐字**口径：

> [src: docs/en/migration_guide/migrate_from_gpu.md:108] "Ascend NPU | Multiple AI cores, categorized into cube cores (for matrix multiplication) and vector cores (for vector computation)"

对照 GPU：[src: migrate_from_gpu.md:109] GPU 侧是「CUDA cores（标量/向量）+ Tensor cores（矩阵）」，且「并发度由编译器与硬件自动决定」。达芬奇把「矩阵 vs 向量」的分工**显式暴露**到编程模型，正是后面 `mix_mode` 的由来。

### 2.2 1:2 配比与两个核数

关键的硬件事实——**一个 AI Core = 1 个 cube 核 + 2 个 vector 核**，即 **cube:vector = 1:2**：

> [src: docs/en/programming_guide.md:16] "on an NPU card, a computing core (AI Core) consists of one cube core, with each cube core paired with two vector cores."

> [src: docs/en/programming_guide.md:14] "For CV fusion operators, the number of cores is equal to the **number of cube cores** (usually half of the number of vector cores). During operator execution, vector cores are called at a ratio of 1:2."

这个 1:2 不是随口比喻，它决定了：
- **两个核数**：`num_aicore`（= cube 核数）与 `num_vectorcore`（= vector 核数，≈ cube 核数的两倍），运行期可查（[src: programming_guide.md:16,25-26]）：

  > [src: docs/en/programming_guide.md:25-26]
  > ```python
  > vectorcore_num = properties["num_vectorcore"]
  > aicore_num = properties["num_aicore"]
  > ```
- **核数怎么固定**：纯 vector 算子把核数固定到 vector 核数；CV 融合算子把核数固定到 cube 核数（[src: programming_guide.md:13-14]）。这直接落到「grid 怎么设」（见 §4.4）。
- **编译器侧的呼应**：HIVM 的 CV 编译有一个 `AutoSubTiling` pass「automatically implements the CV 1:2 subtiling ratio」（[src: third_party/ascend/AscendNPU-IR/docs/source/en/introduction/architecture.md:29]）——即 1:2 是从硬件一路贯穿到编译器自动分块比例的。

**给 writer 的三层递进锚点**：①先讲「一个核里有干矩阵的和干向量的两拨人」；②再点「人数是 1:2，且各有各的片上储物柜」；③最后落到「所以后端要为 CV 融合 kernel 专门做 1:2 subtiling 与核间同步」——把结构图接到后端 pass。

---

## §3 片上内存层级与显式搬运：UB/L1/L0/GM + double-buffer

### 3.1 多级 scratchpad

达芬奇的片上内存是**显式管理的多级 scratchpad**，不是自动 cache `[HPCA'21]`。从大到小、从远到近，读者需要建立的层级（职责为 paper-attributed，容量为 source-cited）：

- **GM（Global Memory）**：片外 DRAM，全局可见，容量大但延迟高。所有输入/输出张量初始都在这里。
- **UB（Unified Buffer）**：**vector 单元**的主片上缓冲区，逐元素/归约计算的数据在这里进出。**容量是本章最硬的一个数字**：
  > [src: docs/en/programming_guide.md:180] "the on-chip memory capacity of Atlas 800T/I A2 is 192 KB. After doublebuffer is enabled by default, the capacity is reduced to half of the original capacity."
  > [src: docs/en/programming_guide.md:272] "The UB size of the A2 series products is 192 KB (1,572,864 bits)."
- **L1 缓冲区**：cube 计算的较大片上缓冲，承接从 GM 搬入、供给 L0。
- **L0A / L0B / L0C**：**cube 单元**紧贴脉动阵列的小缓冲——通常 L0A 放左矩阵、L0B 放右矩阵、L0C 放累加结果。`[paper-attributed, host 未复核：L0A/L0B/L0C 的精确职责与容量以 HPCA'21 / CANN 文档为准，本机不写死]`

编译器侧确认「片上内存要显式推导」而非硬件自动管理：

> [src: third_party/ascend/AscendNPU-IR/docs/source/en/introduction/architecture.md:31] "Intra-core on-chip memory mapping: ... automatically implement on-chip memory space derivation, on-chip memory data layout derivation, on-chip memory access alignment, OP temporary space allocation, and on-chip memory address assignment."

——注意这些都是**编译器在做**的事（HIVM 层），因为硬件不替你做。这正是「显式」的含义。

### 3.2 显式搬运：tl.load / tl.store 是真的在搬数据

在 GPU 上 `tl.load` 大体是「读到寄存器/让 cache 生效」；在昇腾上，`tl.load` 是**把数据从 GM 显式搬进片上 UB**，`tl.store` 是**从 UB 显式写回 GM**。源码文档在 add_kernel 例子里逐字点明这层语义：

> [src: docs/en/programming_guide.md:100-101] "# Load data of x and y to the on-chip memory. \n x = tl.load(x_ptr + offsets, mask=mask)"

后端还提供一类专门优化——「先整块搬进 UB 再从 UB 里选值」（Transferring Data to the UB and Then Selecting，[src: programming_guide.md:122-124]），本质就是因为搬运是显式且有代价的，把离散小搬运换成一次大搬运能显著改善 MTE（搬运引擎）占用（[src: programming_guide.md:160-165] 的 aiv_mte2 前后对比）。这类优化在 GPU 上根本不存在——因为 GPU 的搬运是隐式的。

### 3.3 double-buffer：为什么它把可用 UB 减半

为了让「搬运」和「计算」流水并行（transfer 第 N+1 块的同时 compute 第 N 块），达芬奇用 **double-buffer（ping-pong）**：同一块逻辑缓冲区在物理上开两份，一份在算、一份在搬。代价是**可用容量减半**：

> [src: docs/en/programming_guide.md:180] "After doublebuffer is enabled by default, the capacity is reduced to half of the original capacity. Therefore, data needs to be tiled ..."

而且它**默认就开**：

> [src: docs/en/programming_guide.md:176] "the compiler is configured with **multiBuffer** set to **True** by default, and the parallel storage and computation are supported by default."

serial（搬完再算）vs parallel（边搬边算）的对照见 [src: programming_guide.md:169-176]——parallel 的前提是「合理设计 tiling，让下一批数据能在当前批计算时提前备好」（[src: programming_guide.md:175]）。**这条直接把 §3 的内存约束接到 §4 的 tiling 必然性。**

---

## §4 关键约束：对齐、tiling 硬件必然、mix_mode、grid 强绑物理核

这一节是把 §2/§3 的硬件形态**翻译成后端与用户必须遵守的规则**——本书后面反复出现的四条命门。

### 4.1 tiling 是硬件必然，不是优化选项

片上（192KB，double-buffer 后 ~96KB）远小于要处理的数据总量，所以**必须 tiling**——每次只搬一小块进 UB 算：

> [src: docs/en/programming_guide.md:180] "The on-chip memory space is usually much smaller than the total data volume ... Therefore, data needs to be tiled during operator computation, and only a small part of the data is loaded and processed each time."

超了会直接编译报错（这是读者一定会撞到的错误，[src: programming_guide.md:263-272]）：

> [src: docs/en/programming_guide.md:268] "error: ub overflow, requires 3072256 bits while 1572864 bits available!"

昇腾的 tiling 是**三级**的（[src: docs/en/migration_guide/architecture_difference.md:37-40]）：
> ```text
> ncore:      the number of cores in use (cross-core tiling)
> xblock:     the size of inter-core data blocks (inter-core tiling)
> xblock_sub: the granularity of intra-core tiling (fine-grained intra-core tiling)
> ```
——`ncore` 跨核切、`xblock` 核间切、`xblock_sub` 核内再切。核内那层 `for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB)` 的写法见 [src: docs/en/migration_guide/architecture_difference.md:97-110] 的 `triton_better_kernel`。这解释了为什么昇腾 kernel 常见「核内再套一层 for 循环」，而等价 GPU kernel 没有。

### 4.2 32B / 512B 末轴对齐

UB 对张量**末轴（tail axis）字节数**有对齐要求，不满足会**自动 padding**、拖慢性能：

> [src: docs/en/programming_guide.md:111] "For VV operators, ... the UB of the Ascend hardware requires that the size of the tail axis of the tensor be divisible by **32 bytes**. For CV operators, ... the size of the tail axis of the tensor must be divisible by **512 bytes**. If the tail axis length is insufficient, the tail axis length will be automatically padded."

文档给的反面案例：`(2048, 3)`、`(2048, 1)` 这种末轴极短的张量因自动 padding「性能显著劣化」，要用「借轴转置（borrowing axis for transpose）」规避（[src: programming_guide.md:111,116-119]）。这条是后面「为什么后端要做 layout/transpose 变换、为什么 `<Nx1xf32>` 会在硬件上膨胀」的根因。术语脚注（[src: programming_guide.md:113]）：**VV 算子 = 只用 Vector Core；CV 算子 = 同时用 AI Core（cube）与 Vector Core**。

### 4.3 mix_mode：aic / aiv / mix

由 §2 的 cube/vector 分工直接派生出 kernel 的三种执行模式：
- **aiv（纯 vector）**：kernel 只有向量/逐元素运算 → 只用 vector 核。
- **aic（cube）**：涉及矩阵乘。
- **mix（CV 融合 / Mix Kernel）**：一个 kernel 里**既有 cube 又有 vector** 运算（典型：flash-attention——`Q·Kᵀ`、`P·V` 是矩阵乘走 cube，softmax 走 vector）。

判据就是**有没有 `tl.dot`**：

> [src: docs/en/migration_guide/migrate_from_gpu.md:103] "For Triton operators that perform only vector core computations, the number of concurrent tasks should be equal to the number of vector cores. For other types of Triton operators (those using **tl.dot**), the number of concurrent tasks should be equal to the total number of AI cores."

CV 融合是达芬奇 cube/vector 物理分离硬件形态的**编译级后果**——HIVM 要「感知 CV 核分离架构、自动做 CV 融合编译」：把 Mix Kernel 拆成独立的 AIC/AIV kernel、在 CV 数据依赖处插入 store/load 与核间同步、推导中转 workspace 大小、用 CVPipeline 让 cube/vector 流水并行（[src: third_party/ascend/AscendNPU-IR/docs/source/en/introduction/architecture.md:29]）。整套 CV-fused kernel 专属的 autotune 旋钮（`enable_hivm_auto_cv_balance` / `tile_mix_vector_loop` / `tile_mix_cube_loop` 等，[src: docs/en/architecture_design_and_core_features.md §3.2.1]）之所以只对 CV kernel 有意义，正因为它们调的是 cube/vector 两拨核的协同。

### 4.4 grid 强绑物理核

回到 §1 的那句「grid 是物理核映射」，落到实践就是**推荐把逻辑核数固定到物理核数，核内用 for 循环分块**：

> [src: docs/en/programming_guide.md:11] "The most recommended method is to **fix the number of cores to the number of physical cores of the hardware** and perform more detailed data block division within the cores."

调用侧写法（[src: programming_guide.md:33-35]）就是 `NUM_CORE = vectorcore_num; grid = (NUM_CORE,)`，核内用 `for block_idx in range(pid, NUM_BLOCKS, NUM_CORE)` 做 step 式任务分配（[src: programming_guide.md:58]）。原因（[src: programming_guide.md:10]）：runtime 虽允许最多 65,535 并发任务，但超过物理核数的任务要「分新一轮下发」，带来核启动/初始化开销。`TRITON_ALL_BLOCKS_PARALLEL` 让编译器自动把逻辑核数收敛到物理核数（[src: migrate_from_gpu.md:104]）——这也是后面 `AutoBlockify` / `auto_blockify_size` 的动机来源。

---

## §5 这张硬件图挂着后面整本书（primer 收束）

把本章建立的硬件模型与后续章节的机制一一挂钩，读者就知道「为什么先修这一章」：

| 本章硬件事实 | 后面哪个机制/章挂在上面 |
|---|---|
| cube/vector 1:2 分工 | mix_mode、CV 融合、CVPipeline、AutoSubTiling（HIVM 章） |
| UB 192KB + double-buffer 减半 | tiling 必然、PlanMemory、UB overflow 诊断 |
| 32B/512B 末轴对齐 | layout/transpose 变换、`<Nx1>` 膨胀、借轴转置 |
| 显式搬运 GM↔UB | tl.load/store 下降、MTE 搬运优化、bind_buffer/multibuffer |
| grid 强绑物理核 | AutoBlockify、`auto_blockify_size`、TRITON_ALL_BLOCKS_PARALLEL |
| tl.dot ⇒ 用 AI 核 | mix kernel 拆分为 AIC/AIV、核间同步 |

一句话收尾：**读者写下的 `tl.*` 不是在 GPU 上跑，而是被 fork 后的昇腾后端逐层下降成达芬奇 NPU 的 cube/vector 指令**——本书接下来讲的每一个 pass，都是在把这张硬件图上的约束「编译」出来。

---

## 附：诚实边界清单（给 writer / illustrator）

- **source-cited 硬事实（可逐字复核，放心写死）**：1:2 配比、num_aicore/num_vectorcore、UB 192KB/1,572,864 bits、double-buffer 默认开且减半、32B/512B 对齐、三级 tiling（ncore/xblock/xblock_sub）、grid 强绑物理核、tl.dot⇒AI 核、CV 融合/CVPipeline/AutoSubTiling 1:2、UB overflow 报错样例。行号见正文 [src:] 锚。
- **paper-attributed（host 未联网复核，须软化或联网收口）**：脉动阵列具体维度与每拍 MAC 数、L0A/L0B/L0C 的精确职责与容量、达芬奇「3D-Cube」的微架构细节。正文已就地标 `[paper-attributed, host 未复核]`，**writer 勿把具体数字写成断言**。
- **不在本 primer 范围（交别处，勿画）**：MLIR 方言/op/pass 原理（→ MLIR primer，arXiv:2002.11054）、Linalg 结构化 codegen（→ Linalg primer，arXiv:2202.03293）、AscendNPU-IR 的 HFusion/HIVM 方言分层图与编译流图（→ ch01 鸟瞰）。
- **图**：两张 key_figures（AI Core 结构、片上内存层级）须 illustrator **原创重绘**，不得内嵌受版权保护的论文原图；术语译名统一由 Book Bible。
