# triton-ascend 大纲综合笔记（Lead，2026-07-18）

从 6 子系统 digest 综合 outline-final.json 时**必须**遵守的裁决/事实（防写作期误引）：

## ⚠️ 事实性陷阱（会毁一整章，务必写进相关章 dossier 注记）
1. **`TritonAffinityOpt` 的 "affine" = 核亲和（core affinity），不是多面体/仿射调度（polyhedral/affine scheduling）**。
   DAG.cpp 的 absorb/diffuse 定点传播判每个 op 跑 cube 还是 vector 核。**writer 绝不许引 polyhedral 论文**；
   数据流不动点可用 Kildall 1973 背书，别碰 Feautrier/多面体。（cartoOpt 实读源码确认。）
2. **AutoBlockify ≠ UB/cube 容量 tiling**。它是**网格实例合并**：把 autoBlockifySize 个逻辑 program 折成一个
   前导张量维批处理（UnrealizedConversionCast 载体逐 op 下推），摊薄启动 + 提向量化。**真正的 UB 容量 tiling
   在下游闭源 bishengir/HIVM**，不在 third_party/ascend/lib。（我发车 brief 里把 AutoBlockify 说成 UB tiling 是错的。）
3. 昇腾专属优化的真命门 = **异构双核编排**：TritonAffinityOpt（核亲和定点）→ DAGScope（物化 AIC/AIV 两 scope）
   → DAGSync（跨核 set/wait 事件 + 共享缓冲搬运）→ DAGSSBuffer（5.5k 行，UB double-buffer 软流水）。

## Primer 去重（读者别读两遍方言栈图）
- **1× 达芬奇 NPU 硬件 primer**（cartoHw 持，Part 1）：cube/vector 1:2、UB 192KB/L1/L0/GM、double-buffer 减半 UB→tiling 是硬件必然、32B/512B 末轴对齐、mix_mode(aic/aiv)、grid 强绑物理核。
  论文根 web-verified：HPCA'21「Ascend…」DOI 10.1109/HPCA51647.2021.00071 + HotChips'19「DaVinci」DOI 10.1109/HOTCHIPS.2019.8875654（见 instances/triton/book/cartography/papers-recon-ascend.json，2026-07-15 核真）。
- **1× MLIR/Linalg 编译基础设施 primer**（Triton→Linalg Part 早段）：arXiv:2002.11054(MLIR) + 2202.03293(Linalg indexing_map)，均 base recon web-verified。放一次，下游 opt/hivm 章交叉引用不重设。
- **triton-shared「raise pointer-arithmetic to structured」**：PtrAnalysis copyright=Huawei+Microsoft，RFC 有、无 arXiv。**倾向折进「Triton→Linalg 分水岭总览」章作智识框架，不单列 primer**（external-source 溯源）。
- 脉动阵列（Kung 1982 / TPU Jouppi 2017）：cube 原理，与基座 ch27 交叉引用即可，不单设。

## 归属/接缝
- 前端**双 builder**（cartoLang 发现）：code_generator.py `self.builder + self.ascend_builder`，visit_Call 用 `extension.is_builtin` 路由（L1183），`__triton_builtin__ + __ascend_builtin__` 双标记 → 归 **语言层 Part**，只讲 Python 表面怎么 emit 进去；**hivm op 语义归 hivm Part**，不重复。
- ascendnpu_ir_builder / hivm dialect C++ + Fractal NZ 布局 + fixpipe NZ2ND → 归 **cartoHivm/hivm Part**。
- `ttir_to_linalg` Python 编排（含 `ascend.passes.ttir.add_*` 调用点）归 backend Part；C++ pass 内部归 linalg Part。
- `triton-to-structure` 两次出现(compiler.py:131/:152)第二遍收敛 + force-simt-template 归属 → **施工期实跑 dump 坐实，勿测绘期臆测**（记进该章 open_question）。
- AscendNPU-IR submodule 现已 populate（47a0229，1522 文件）→ cartoHivm 读 bishengir 内部。

## 规范路径前缀（答 cartoHw 占位问题）
`ascend/...`、`third_party/ascend/...`、改动过的上游 `python/triton/...`；AscendNPU-IR 内部带 `third_party/ascend/AscendNPU-IR/...`。**绝不带 instances/.../source/**。

## 诚实边界章（Part-9）
unittest 323 .py 揭示能力谱系：支持面广（math/reduce/scan/dot/matmul/attention/atomic/block_ptr + 昇腾扩展 compile_hint/sync_block/multibuffer/npu_indexing/fixpipe/paged_kvcache）；**38 skip/xfail = 诚实未支持**（waiting bishengir/TA、NPUIR 四月更新回退、UB overflow、attn_cp 整批跳）。

## 章数预算
各子系统慷慨提议合计 backend6+linalg5+opt5+lang6+hw5+hivm? ≈ 30-33。目标 ~28-34 章、~7 Part。
综合期按 Part 平衡裁（发射器拆合、PtrAnalysis/BlockPtrAnalysis 拆合、language 6→4 可压等，各 digest open_questions 已给压缩版）。

## Part 骨架（草案，待 hivm digest + 综合定稿）
- P1 鸟瞰 + 达芬奇硬件 primer（+ vector-add on-ramp）
- P2 语言层（CANN 扩展：双 builder / UB-GM buffer 语言 / 索引搬运 / 向量算子 / custom_op+libdevice / scope+sync）
- P3 分水岭 Triton→Linalg（+ MLIR/Linalg primer；PtrAnalysis / BlockPtr→memref / MaskAnalysis / Unstructure 兜底）
- P4 昇腾优化 pass（AutoBlockify / 核亲和双核 / scope+跨核同步 / DAGSSBuffer 软流水 / 离散掩码）
- P5 HFusion/HIVM 硬件 IR 与下降（待 hivm digest）
- P6 后端与运行时（挂载 / 三段下降 / bishengir 边界 / NPU 驱动 / 动态发射器 / 多框架）
- P7 度量与实战（vector-add / flash-attention CV 融合 capstone / 诚实能力边界）

## 综合后复核项（synth 可能读到旧版 digest）
- **hivm-dialect digest 后更新为 6 章**（新增「算子融合与自动调度:FusionKind + Cube/Vector tile 策略」，
  锚 lib/Dialect/HFusion/Transforms/{OpFusion,AutoSchedule}/，对位基座 ch29-31）。若 synth 读的是旧 5 章版，
  **审核 outline 时须补这一章**（P5 HFusion-HIVM Part）。cartoHivm 给的合并选项:HIVM IR+同步(④⑤)可合、
  或方言逃生舱(①)并入 cartoLinalg——综合定夺。
- FusionKind/AutoSchedule 归 HFusion Part（cartoOpt 的 opt-pass Part 只讲 TritonAffinityOpt 核亲和,不碰 FusionKind）。
