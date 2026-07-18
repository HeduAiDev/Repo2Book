# Triton-Ascend 源码解读 — 架构地图（cartography）

> 2026-07-18 Lead 综合自 6 子系统 digest。基线 triton-ascend v3.2.1 @ 2badfc89e + AscendNPU-IR submodule 47a0229。

## 心智模型（一句话）

同一份 Triton 前端,昇腾后端如何抛弃 GPU 的 SIMT 指针张量模型、走一条 Triton→Linalg→HFusion→HIVM→AscendC 的结构化下降链,把 tl.* 核落到达芬奇 cube/vector 双核。

**fork 而非插件**：上游 Triton 3.2.0 整树在内（`python/triton/`、`lib/`、`include/`），昇腾增量在 `third_party/ascend/`（+ AscendNPU-IR submodule，全开源）。与 vllm-ascend 的『注册表顶替/monkey-patch』不同，本书主线是**同一 Triton 前端、一条与 GPU 完全不同的 NPU 后端下降链**。

## 下降链（与基座对照）

```
基座 triton (GPU):   TTIR → TTGIR → LLVM → PTX → cubin        （SIMT，tensor-of-pointers，layout 编码）
triton-ascend (NPU): TTIR → ttadapter(Linalg) → HFusion → HIVM → Standard → AscendC 库调用 → (闭源 CCE) NPU 二进制
                          └ 抛弃指针模型，还原成结构化 memref     └ 全开源可读到底（比 ptxas 边界更深）
```

## 子系统地形（6 块）

- **后端与运行时**（`backend/`）：AscendBackend 契约 + 三段 add_stages + bishengir 闭源边界 + NPU 驱动/发射器 + 一后端两框架
- **语言层 CANN 扩展**（`language/cann/`）：双 builder 分发 + UB/GM 显式搬运 buffer 语言 + 内建/向量/自定义算子 + scope/核间同步
- **分水岭 Triton→Linalg**（`lib/TritonTo{Linalg,Structured,Unstructure,Annotation}/`）：triton-shared 血缘:PtrAnalysis/BlockPtrAnalysis/MaskAnalysis 把指针算术还原成结构化 memref
- **昇腾优化 pass**（`lib/{AutoBlockify,TritonAffinityOpt,DiscreteMaskAccessConversion}/`）：异构双核编排:核亲和定点→scope 切分→跨核同步→UB 多缓冲软流水（affine=核亲和，非多面体）
- **HFusion/HIVM 硬件 IR**（`AscendNPU-IR/bishengir/`）：HFusion 融合层 + HIVM 达芬奇硬件方言 + 显式同步 + 降到 AscendC 库调用（全开源）
- **度量与实战**（`tutorials/ + unittest/`）：flash-attention CV 融合 capstone + 测试套件揭示的诚实能力边界

## 逐 Part 大纲

### 鸟瞰与达芬奇硬件模型（part-1）
*建立心智模型:fork 而非插件、同前端异后端、达芬奇 cube/vector 双核 + 显式内存搬运;vector-add on-ramp。*

- **ch01** 【概览】从 Triton 到昇腾:后端全景与三段下降流水线（鸟瞰） ★
- **ch02** 【原理】达芬奇 NPU 硬件模型:cube/vector 双核、片上内存层级与显式搬运（原理篇）
- **ch03** 【概览】上手第一课:vector-add 与 GPU→NPU 的最小改写

### 语言层:CANN 扩展（part-2）
*昇腾把 DaVinci 编程模型显式暴露到 tl.* 之上:双 builder、UB/GM 显式搬运、内建算子、自定义算子、scope/核间同步。*

- **ch04** 前端接缝:双 builder 与 Ascend 内建的分发路由
- **ch05** 显式内存层级:UB/GM/L1/L0C、buffer 语言与 copy/fixpipe
- **ch06** 昇腾内建算子:索引搬运、向量算子与定制 cast
- **ch07** 自定义算子框架与 Ascend libdevice:register_custom_op 与数学库
- **ch08** 作用域、核间同步与流水线提示:scope/sync_block/PIPE/compile_hint

### 分水岭:Triton→Linalg（part-3）
*全书与基座最根本的 divergence:抛弃 tensor-of-pointers SIMT 模型,把指针算术逆向还原成结构化 memref/Linalg(triton-shared 血缘)。*

- **ch09** 【原理】MLIR 与 Linalg:结构化张量 codegen 的编译基础设施（原理篇）
- **ch10** 分水岭:从指针张量到结构化张量——triton_adapter 总览 ★
- **ch11** 指针算术的逆向工程:PtrAnalysis 把 addptr 链还原成 stride/offset
- **ch12** 落到 memref:BlockPtrAnalysis、reinterpret_cast 与 load/store→linalg
- **ch13** 边界的语义:MaskAnalysis 把 mask 还原成 extract_slice
- **ch14** 结构化装不下时:Unstructure 兜底路径与 gather/scatter 标量化

### 昇腾优化 pass:异构双核编排（part-4）
*NPU 编译第一性问题:每个 op 放 cube 还是 vector 核(核亲和定点)+ 跨核显式同步 + UB 多缓冲软流水 + 网格合并 + 离散访存驯服。*

- **ch15** AutoBlockify:把多个网格实例折成一条 blockify 循环
- **ch16** Cube 还是 Vector:AI Core 异构双核与核亲和定点传播
- **ch17** 把双核落到 IR:Scope 切分与 cube↔vector 同步搬运
- **ch18** DAGSSBuffer:UB 多缓冲与昇腾的软件流水线
- **ch19** 不规则访存的驯服:离散掩码拆分与交错访存优化

### HFusion/HIVM 硬件 IR 与下降（part-5）
*NPU 的硬件 IR(TTGIR+PTX 对位物):HFusion 融合层 + HIVM 达芬奇硬件方言 + 显式同步,一路降到 AscendC 库调用(全开源可读到底)。*

- **ch20** TritonAscend 方言与三条逃生舱:Triton 表达不了的 NPU 语义如何注入
- **ch21** HFusion 方言:Linalg 之上的张量级融合 IR 与算子上抬
- **ch22** 算子融合与自动调度:FusionKind 分类与 Cube/Vector 分工的 tile 策略
- **ch23** HIVM 方言:达芬奇硬件 IR——显式内存层级、流水线与 Cube/Vector 双核 ★
- **ch24** HIVM 显式同步:set_flag/wait_flag 流水线同步与 Cube↔Vector 核间同步
- **ch25** 下降链收官:HFusion→HIVM→Standard——从融合张量 op 到 AscendC 库调用 ★

### 后端与运行时（part-6）
*把各层收进后端框架:AscendBackend 契约、三段下降编排、bishengir 闭源边界、NPU 驱动装载、动态发射器、一后端两框架。*

- **ch26** 昇腾后端如何挂进 Triton:AscendBackend 契约、NPUOptions 与 hacc.target 注入
- **ch27** 三段下降链:add_stages 与 Triton-MLIR→Linalg 的编排
- **ch28** 闭源边界 bishengir-compile:编译选项、子进程调用与元数据回收
- **ch29** NPU 运行时驱动与二进制装载:NPUDriver、NPUUtils 与 npu_utils.cpp
- **ch30** 动态生成的发射器:generate_npu_wrapper_src、rtKernelLaunch 与 taskqueue/msprof
- **ch31** 一套后端，两个框架:torch_npu / mindspore 策略注册表

### 度量与实战（part-7）
*flash-attention CV 融合 capstone 把全链串一遍;测试套件揭示的诚实能力边界。*

- **ch32** 【概览】实战收官:flash-attention 融合注意力在昇腾的 CV 融合落地 ★
- **ch33** 【概览】昇腾 Triton 的能力边界:测试套件揭示的支持/未支持谱系

## 配对脊柱（每章 ⇄ 基座 triton 章）

| 章 | 本书 | ⇄ 基座《Triton 源码解读》 |
|----|------|--------------------------|
| ch01 | 从 Triton 到昇腾:后端全景与三段下降流水线（鸟瞰） | ch01(Triton 是什么) |
| ch02 | 达芬奇 NPU 硬件模型:cube/vector 双核、片上内存层级 | ch02(GPU 执行模型) |
| ch03 | 上手第一课:vector-add 与 GPU→NPU 的最小改写 | ch03() |
| ch04 | 前端接缝:双 builder 与 Ascend 内建的分发路由 | ch04() |
| ch05 | 显式内存层级:UB/GM/L1/L0C、buffer 语言与 cop | ch07(块/形状/访存) |
| ch06 | 昇腾内建算子:索引搬运、向量算子与定制 cast | ch07(块/形状/访存), ch08(dot/reduce/scan), ch05(类型系统) |
| ch07 | 自定义算子框架与 Ascend libdevice:register | ch09(自托管库/extern) |
| ch08 | 作用域、核间同步与流水线提示:scope/sync_block/PI | ch04() |
| ch09 | MLIR 与 Linalg:结构化张量 codegen 的编译基础设 | ch02(GPU 执行模型) |
| ch10 | 分水岭:从指针张量到结构化张量——triton_adapter 总览 | ch32(五段 TTIR→TTGIR) |
| ch11 | 指针算术的逆向工程:PtrAnalysis 把 addptr 链还原 | — none-new(GPU 侧无对应) |
| ch12 | 落到 memref:BlockPtrAnalysis、reinter | ch33(类型塌缩), ch34(共享内存降级) |
| ch13 | 边界的语义:MaskAnalysis 把 mask 还原成 extr | — none-new(GPU 侧无对应) |
| ch14 | 结构化装不下时:Unstructure 兜底路径与 gather/s | — none-new(GPU 侧无对应) |
| ch15 | AutoBlockify:把多个网格实例折成一条 blockify  | ch25(AxisInfo/Coalesce) |
| ch16 | Cube 还是 Vector:AI Core 异构双核与核亲和定点传 | ch27(Tensor Core MMA), ch28(AccelerateMatmul) |
| ch17 | 把双核落到 IR:Scope 切分与 cube↔vector 同步搬 | ch31(warp specialization) |
| ch18 | DAGSSBuffer:UB 多缓冲与昇腾的软件流水线 | ch29(软件流水primer), ch30(流水线落地) |
| ch19 | 不规则访存的驯服:离散掩码拆分与交错访存优化 | ch25(AxisInfo/Coalesce) |
| ch20 | TritonAscend 方言与三条逃生舱:Triton 表达不了的 | ch24(TTG/TTNG op), ch19(TT 方言词汇表) |
| ch21 | HFusion 方言:Linalg 之上的张量级融合 IR 与算子上 | ch19(TT 方言词汇表) |
| ch22 | 算子融合与自动调度:FusionKind 分类与 Cube/Vect | ch29(软件流水primer), ch30(流水线落地), ch31(warp specialization) |
| ch23 | HIVM 方言:达芬奇硬件 IR——显式内存层级、流水线与 Cube | ch19(TT 方言词汇表), ch23(LinearLayout), ch27(Tensor Core MMA) |
| ch24 | HIVM 显式同步:set_flag/wait_flag 流水线同步 | ch26(共享内存屏障), ch31(warp specialization) |
| ch25 | 下降链收官:HFusion→HIVM→Standard——从融合张量 | ch32(五段 TTIR→TTGIR), ch33(类型塌缩), ch34(共享内存降级), ch35(dot/reduce→PTX) |
| ch26 | 昇腾后端如何挂进 Triton:AscendBackend 契约、N | ch14(编译驱动) |
| ch27 | 三段下降链:add_stages 与 Triton-MLIR→Lin | ch36(CUDABackend) |
| ch28 | 闭源边界 bishengir-compile:编译选项、子进程调用与 | ch37(ptxas→cubin→发射) |
| ch29 | NPU 运行时驱动与二进制装载:NPUDriver、NPUUtils | ch37(ptxas→cubin→发射) |
| ch30 | 动态生成的发射器:generate_npu_wrapper_src、 | ch37(ptxas→cubin→发射) |
| ch31 | 一套后端，两个框架:torch_npu / mindspore 策略 | — none-new(GPU 侧无对应) |
| ch32 | 实战收官:flash-attention 融合注意力在昇腾的 CV  | ch43(fused-attention capstone) |
| ch33 | 昇腾 Triton 的能力边界:测试套件揭示的支持/未支持谱系 | — none-new(GPU 侧无对应) |

## 关键裁决（见 SYNTHESIS-NOTES.md）
- **primer 去重**：达芬奇硬件 primer（ch02，HPCA'21+HotChips'19 web-verified）+ MLIR/Linalg primer（ch09，arXiv:2002.11054/2202.03293）各一次；triton-shared『raise pointer→structured』折进 ch10 分水岭总览不单列。
- **事实性陷阱**：`TritonAffinityOpt` 的 affine = **核亲和（core affinity）**，非多面体/仿射调度——ch16 dossier 须硬注禁引 polyhedral 论文。AutoBlockify = 网格实例合并，非 UB tiling。
- **审批闸**：本大纲须用户审批后才逐章发车（与基座 triton 的豁免不同）。