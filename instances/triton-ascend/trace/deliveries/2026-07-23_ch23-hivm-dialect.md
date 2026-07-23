# ch23 交付：HIVM 方言——达芬奇硬件 IR

- **Type**: delivery
- **Chapter**: ch23
- **Date**: 2026-07-23
- **Timestamp**: 2026-07-23T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: triton-ascend, part-5, deep, skip_impl, hivm-hfusion, hivm-dialect, address-space, cube-vector, mmadL1, infer-mem-scope, flagship

## What happened

Part 5「硬件 IR HIVM」第四站，hivm-hfusion 子系统承 ch21（HFusion 方言）/ch22（FusionKind 双核调度视角）。kind=deep，纯 C++/TableGen MLIR 方言章，无精简版（无 `implementation/` 目录）。ch22 讲的是**为什么这么分**（调度策略），本章讲**在 IR 里怎么落地**（硬件 IR 编码）——互为表里。

三支柱全覆盖，10 个机制：①方言身份（`HFusionToHIVM` 入口 pass 把 `linalg`/`hfusion` 判 illegal、`hivm` 判 legal；算子助记符前缀 `hir.`，方言名 `hivm` ≠ 前缀，打印成 `hivm.hir.xxx`）；②六级显式内存层级（`HIVM_AddressSpaceEnum`：Zero 哨兵 + GM/L1/L0A/L0B/L0C/UB，`#hivm.address_space<>` 属性贴到 `memref` 类型尾）；③Cube/Vector 双核分工写死在算子级 Trait（`VectorCoreTypeTrait`/`CubeCoreTypeTrait`）+ 函数级 `func_core_type`（AIC/AIV/MIX 三态）两层判据；④DMA 算子族（load/store/copy/fixpipe）按方向绑定内存层级与流水引擎；⑤Cube 矩阵路径核心 `MmadL1Op`（`C=C+A×B`，DPS 语义，L0A/L0B 是宏算子内部缓冲不出现在操作数上）；⑥Vector 逐元素算子族（四十余个共享 `PIPE_V+VectorCoreTypeTrait`，一律落 UB）；⑦HFusion→HIVM 逐算子映射派发（linalg/hfusion 算子查表换成对应 `hivm.hir.vXxx`，矩阵/归约走并列 `populate*`）；⑧内存层级推断算法 `InferHIVMMemScope`（四步带优先级不覆盖：mmadL1 约束→func 参数 GM→PointerCastOp 传播→核类型兜底，外加沿 `scf.for`/yield/view 的 use-def 级联到不动点）；⑨贯穿全章的真实 lit 夹具 worked example（`test/Dialect/HIVM/infer-hivm-mem-scope.mlir` 的 `test_infer_mem_scope_complicated`，一个矩阵核降到 HIVM 后每个 `memref` 该带的地址空间被 CHECK 断言逐一钉死）。

**取证边界**：host 无昇腾 NPU、无 `bishengir-opt` 可执行文件，全部 IR 落位取自仓库已提交的 lit 回归夹具（CI 逐字核对，权威性等价真机 trace），非编造 dump。

**评审**：多维评审 1 轮 APPROVED，6 条 issue 全 non-blocking（均 reader-comprehension/figure-integration 维度，无 blocking）——核心是「论断先出现、证据滞后 5 节」「插图副标题引入正文未打通的 `TCoreType` 底层枚举」「§23.7 步骤编号三处不一致（use-def 级联 vs PointerCastOp 传播被同一个『步③』混用）」「DPS 缩写代码块先出现、两节后才展开」「`AscendC` 术语零解释首现即弃用」。这些属于机械/措辞层面的小修，本轮记录在案，留存量回修批次处理，不阻塞本章交付。blind_review 1 轮 PASS（0 failure），chapter-map 1 轮 PASS。verdict APPROVED。

## Why it matters

ch23 是全书第一个把「达芬奇内存墙」写进 IR 类型系统的章节——此前章节（ch16 核标注、ch17 Scope 切分、ch18 UB 多缓冲）都在**隐式**处理内存位置，本章把这套隐式约定坐实成显式的 `#hivm.address_space<>` 类型属性 + 静态 Trait 编码，读者第一次看到「Cube/Vector 二选一」「六级内存」不再是叙事概念、而是可以从 TableGen 定义与 lit 断言里直接核验的类型系统事实。与 ch22 互为表里：ch22 回答调度策略「怎么分」，ch23 回答硬件 IR「分完落在哪」。

对位基座《Triton 源码解读》ch19-24（TTGIR 布局）+ ch27（MMA 布局）——GPU 把并行度摆在 warp/lane 上的 layout attribute，NPU 摆的是数据搬进哪级 buffer、走哪条流水、哪种核、矩阵靠 nZ/zN 分形，是同一「把物理执行细节编码进类型系统」母题在异构双核硬件上的对应物。

## What to remember

- **地址空间四档口径（承 ch05 纪律，本章新场景）**：`.td` 定义 7 个（Zero+GM+L1+L0A+L0B+L0C+UB），本章 worked example 实际用到 3 级（gm/cbuf/cc）——L0A/L0B 藏在 `mmadL1` 宏算子内部，不显式出现在 IR 里；这与 ch05 立的「.td 七档 vs pybind 五档」口径是同一枚举在编译器 IR 层与 Python 语言层的两次不同现身，不要混淆。
- **算子级 vs 函数级核归属两套判据**：`VectorCoreTypeTrait`/`CubeCoreTypeTrait`（算子级静态标记）与 `func_core_type`（AIC/AIV/MIX，函数级属性）是不同层级的两套判据，`InferHIVMMemScope` 兜底步读的是后者。
- **`InferHIVMMemScope` 四步优先级不覆盖**：mmadL1 约束＞func 参数 GM＞PointerCastOp 传播＞核类型兜底，高优先级先定的地址空间后续步骤不再覆盖；外加沿 use-def 的级联传播到不动点（这是附加过程，不是第五个优先级——正文 §23.7 步骤编号有三处轻微不一致，已记入 issue 待存量回修）。
- **无新伏笔**：dossier `foreshadow_due.should_plant`/`should_payoff` 均为空；本章依赖 ch21/ch22（均已回收），未埋新伏笔。
- **Bible 回写**：glossary +6（`hir.` 助记符前缀 / `VectorCoreTypeTrait`+`CubeCoreTypeTrait` / `func_core_type`(TFuncCoreType) / `MmadL1Op` / `InferHIVMMemScope` / `PointerCastOp`，现 251 键；`AddressSpace`/`TCoreType`/DPS 等既有词条本章沿用未重复登记）；concepts +9（现 271）；interfaces 不新增（deep+skip_impl 无精简版，同 ch20-22 先例）；arc-map 无变化（本章无伏笔动作）。
- **遗留待办（非阻塞，留存量回修批次）**：6 条 reader-comprehension issue——§23.3/§23.5 论断证据滞后至 §23.8、插图副标题 `TCoreType` 枚举与正文未搭桥、§23.7「步③」三处指向不一致（PointerCastOp 传播 vs use-def 级联）、DPS 缩写代码先于释义两节、`AscendC` 零解释首现。
