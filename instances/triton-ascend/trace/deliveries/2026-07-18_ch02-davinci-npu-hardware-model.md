# ch02 交付：达芬奇 NPU 硬件模型——cube/vector 双核、片上内存层级与显式搬运

- **Type**: delivery
- **Chapter**: ch02
- **Date**: 2026-07-18
- **Timestamp**: 2026-07-18
- **Agents involved**: analyst, ch02paperprep, explainer, ch02ill, ch02writer, reviewer, Lead, archivist
- **User present**: false
- **Tags**: triton-ascend, part-1, primer, davinci-npu, ai-core, cube-vector, systolic-array, onchip-memory, double-buffer, tiling, mix-mode, grid-physical-core, honesty-boundary, dossier-verify-escape, blind-review, paper-package

## What happened

Part 1 **flagship primer**（kind=**primer**，无精简版接口，替代门禁 **lint_paper_grounding**）。verdict=**APPROVED**，全 linter green，**3 图全部盲审 PASS**。主题：下降链最终落地的目标硬件——达芬奇 NPU 硬件模型三支柱，对位基座《Triton 源码解读》ch02（GPU SIMT/占用率 primer），同为『目标硬件执行模型 primer』。

**十一机制 / 三支柱主线**（source-cited 硬事实 vs paper-attributed 软事实严格分层）：
1. **为什么 NPU 不是 GPU**：领域专用异构核（cube/vector）vs 通用 SIMT。
2. **AI Core 结构**：1 cube（脉动阵列，专啃矩阵乘）+ 2 vector（逐元素/规约）+ scalar（标量/控制/地址），**cube:vector=1:2**（source-cited，programming_guide.md:L14,16）→ 落 `num_aicore`/`num_vectorcore`（L16,25-26）。
3. **片上内存层级**：GM(片外 DRAM) / UB(**192KB=1,572,864 bits**，服务 vector) / L1 / L0A(左矩阵 A)-L0B(右矩阵 B)-L0C(累加 C，服务 cube) 多级显式 scratchpad（非 GPU 自动 cache）。
4. **显式搬运**：`tl.load`(GM→UB)/`tl.store`(UB→GM) 是真的在搬数据。
5. **double-buffer(multiBuffer)** 默认开、把可用 UB **减半**（192KB→约 96KB）。
6. **tiling 是硬件必然**（非优化选项）：三级 `ncore`(inter-core)/`xblock`(intra-core)/`xblock_sub`(核内 sub-block)——规范 worked example 用 architecture_difference.md:97-124 的 `triton_better_kernel`（ncore=32/xblock=32768/xblock_sub=8192），`masked_fill_kernel` 作核内 sub-block 补例。
7. **32B(VV)/512B(CV) 末轴对齐**、自动 padding、借轴转置。
8. **mix_mode(aic/aiv/mix)**：判据**双向且唯一**——有 `tl.dot` 一定用 cube、无 `tl.dot` 一定不用；mix 态触发 HIVM 的 CV 融合（拆 AIC/AIV + 核间同步 + AutoSubTiling CV 1:2 + CVPipeline）。1:2 从硬件配比一路贯穿到编译器子分块。
9. **grid 强绑物理核**：核数固定到物理核 `grid=(NUM_CORE,)` + 核内 step for 循环；`TRITON_ALL_BLOCKS_PARALLEL` 触发 AutoBlockify 自动收敛。
10. **收束表**（primer 收尾『为什么先修』）：每条硬件事实前挂到下游 pass/Part（见下 What to remember 的前向线索）。

**诚实边界（本章最吃劲处，严守）**：
- **source-cited 硬事实写死**（可在 pin@2badfc89e 逐字复核）：1:2 配比 / num_aicore·num_vectorcore / UB 192KB=1,572,864 bits / double-buffer 默认开且减半 / 32B·512B 对齐 / 三级 tiling 参数 / grid 强绑物理核 / tl.dot⇒AI 核 / CV 融合·CVPipeline·AutoSubTiling 1:2 / UB overflow 报错样例。
- **paper-attributed 软化措辞、勿写断言、禁杜撰数字**（HPCA'21 / HotChips'19，host 未联网复核）：脉动阵列具体维度与每拍 MAC 数、L0A/L0B/L0C 精确职责与容量、3D-Cube 微架构。
- **图面同步降确定性**：`davinci-ai-core-structure` 图的 **16×16×16** 脉动阵列维度已降小字斜体灰（与 paper-attributed caveat 同级），1:2 徽标为唯一粗体强调（source-cited）——**视觉确定性与文字确定性对齐**。Lead Read-PNG 亲验，blind_review=PASS。

**dossier-verify escape 经过（值得记）**：dossier-verify 站抓出 `ch02paperprep` 把三级 tiling 规范 worked example `triton_better_kernel` **引错文件名**——paper.md §4.1 原写 `[src: migrate_from_gpu.md:97-104]`，经 pin 复核该函数（def L97、`for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB)` L99、调用侧 ncore=32/xblock=32768/xblock_sub=8192 L107-110）实际在 **architecture_difference.md:97-124**（是文件名引错、非函数不存在）。Lead 修 **paper.md + dossier + skip_dossier 复跑**清除。**教训：paper-prep 的 [src:] 锚点须逐条核到文件+行号，dossier-verify 是 primer 章的关键防线。**

**blind round 1→2 修图**：`davinci-ai-core-structure` 首轮盲审就 16×16×16 维度视觉确定性过高（与文字 paper-attributed 软化不一致）打回；Lead 派 `ch02ill` 修图（降小字斜体灰）后 round 2 PASS。

**Lead 另修/补**：修 dossier `paper_origin.sections`（对齐 paper.md 真标题）清 3 条 WARN；派 `ch02writer` 补 4 处 reader-comp。

**论文包**：`instances/triton-ascend/book/papers/ch02-davinci-npu-hardware-model/`——HPCA'21（Ascend, pp.789-801，DOI 10.1109/HPCA51647.2021.00071，web-verified）+ HotChips'19（DaVinci，DOI 10.1109/HOTCHIPS.2019.8875654，web-verified）两篇一手论文 + **5 份 source-docs**（programming_guide.md / architecture_difference.md / migrate_from_gpu.md / AscendNPU-IR architecture.md / architecture_design_and_core_features.md）。meta.json + paper.md 齐备。

回环轮数：blind 2 轮（round 1 打回 16×16×16 → round 2 PASS）、map 1 轮、dossier-verify 1 次 escape+复跑。

## Why it matters

全书**硬件地基 primer**：后面每一个 pass 章都是在把这张硬件图上的约束「编译」出来。ch01 埋的『达芬奇双核 1:2→ch02』『显式内存→ch05』线索在此兑现并量化。收束表把六条硬件命门前挂到 P3/P4/P5，是后续章节的『为什么这么设计』总索引。primer 章**无精简版接口**（interfaces 跳过），门禁走 lint_paper_grounding（# PAPER 全覆盖 + 正文有出处）。

## What to remember

- **诚实边界是本章招牌**（勿回退）：source-cited 写死 / paper-attributed 软化 / 图面 16×16×16 降视觉确定性——三层一致。任何后续复用达芬奇脉动阵列维度时同样 paper-attributed，别升格为断言。
- **本章无 arc-map 正式伏笔埋/回收**（`bible.py due ch02` 空，primer 硬件先修章）。收束表的『硬件事实→下游 pass』是**前向线索（delivery 记录，非正式伏笔）**：
  - cube/vector 1:2 分工 → `mix_mode`、CV 融合、CVPipeline、AutoSubTiling（**P5 HFusion/HIVM**）
  - UB 192KB + double-buffer 减半 → tiling 必然、**PlanMemory**、UB overflow 诊断（**P4 优化 / P5 HIVM**）
  - 32B/512B 末轴对齐 → layout/transpose 变换、`<Nx1>` 膨胀、借轴转置（**P4 优化**）
  - 显式搬运 GM↔UB → `tl.load`/`tl.store` 下降、MTE 搬运优化、**bind_buffer/multibuffer** 缓冲绑定（**P3 分水岭 / P5 HIVM**）
  - grid 强绑物理核 → **AutoBlockify**、`TRITON_ALL_BLOCKS_PARALLEL`、`auto_blockify_size` 收敛粒度（**P4 优化**）
  - `tl.dot` ⇒ 用 AI 核 → Mix Kernel 拆 AIC/AIV、核间同步（**P5 HIVM**）
- **事实校准点**：UB=192KB=1,572,864 bits；double-buffer 默认开→可用约 96KB；三级 tiling 规范例 `triton_better_kernel` 在 **architecture_difference.md:97-124**（非 migrate_from_gpu.md，dossier-verify 已修）；1:2 是硬件配比、也是 AutoSubTiling 子分块比，同一个 1:2 贯穿硬件到编译器。
- **边界**：MLIR 方言/op/pass 原理→ch09 MLIR primer；Linalg 结构化 codegen→Linalg primer；昇腾方言分层图/编译流图→ch01 鸟瞰。本章不重画方言栈图。
