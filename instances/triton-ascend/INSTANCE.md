# 实例：triton-ascend（《Triton-Ascend 源码解读》）

> 本文件 = 本实例的「源码版本 + 当前状态 + 专属规则」。通用方法论见仓库根 `CLAUDE.md`；配置见 `instances/triton-ascend/repo2book.json`。

## 源码版本（行号基线）
- 仓库：**官方主仓（triton-lang 组织）** `https://github.com/triton-lang/triton-ascend.git` →
  `instances/triton-ascend/source/`（分支 `book-baseline`）。
  ⚠️ 仓库考古：Gitee 已冻结（2025-10 迁移公告）→ GitCode（Ascend org）→ 现归入 triton-lang
  官方组织（main 活跃至 2026-07）；三处的 v3.2.1 为同一 commit，**以 triton-lang 官方仓为准**（2026-07-15 用户定）。
- **钉死 v3.2.1 @ `2badfc89e70a9b7a5e88463a116c2feddce4b101`**（2026-04-25 正式版，
  README 自述「当前版本」，配套 CANN 9.0.0）。
- 选版依据（2026-07-15 用户定）：triton-ascend 最新正式版 + 配套 Triton（v3.2.0）。
  main 分支是「升级到 Triton 3.5」的开发线（version.txt=3.5.0，2026 年计划），不作书基线。
- 读者：advanced（zh-CN）。

## 架构形态（与 vllm-ascend 不同，写作时刻意区分）
**本仓是 Triton 的 fork（整树内嵌），不是 vllm-ascend 那种 OOT 插件**：上游 Triton 3.2.0
全量在树内（`python/triton/`、`lib/`、`include/`），昇腾增量在 `ascend/`、`third_party/ascend/`
（AscendNPU-IR submodule）与对上游文件的**原位修改**。叙事主线是「fork 改了哪里/为什么改」：
逐章 `pairs_with` 指回基座书（instances/triton），`git diff` 基座 v3.2.0（9641643da）即昇腾
增量的真相源。（vllm-ascend 的主线是"注册表顶替/monkey-patch"；本书预计是"后端接入点 +
原位 diff + AscendNPU-IR 下沉"——cartography 时核实。）

## 规范路径约定
仓库根相对路径：`ascend/...`、`third_party/ascend/...`、改动过的上游文件如
`python/triton/backends/...`；AscendNPU-IR 内部须带 submodule 前缀
`third_party/ascend/AscendNPU-IR/...`。

## 实例专属硬规则
- 运行验证需昇腾 NPU/CANN 工具链，宿主无此环境——tester/explainer 以编译期产物
  （IR dump/linalg 降级结果）与 interpreter 模式为主，trace_source 如实标注。

## 当前状态（2026-07-18）
- ✅ scaffold + clone（GitCode）+ 钉版 v3.2.1（main `2badfc89e`）。
- ✅ **基座书《Triton 源码解读》43 章已全部完本**（gate 已开）；active_instance 已切到 triton-ascend。
- ✅ **AscendNPU-IR submodule 已 populate**（此前为空）：`third_party/ascend/AscendNPU-IR`
  @ `47a0229060e37f92a49cfb82d81c756628e6c7ae`（1522 文件；含 bishengir 的 HFusion/HIVM
  方言定义与 Linalg→HFusion→HIVM→Standard 下降链——**开源可读**，不是纯闭源 blob）。
- 🔄 **cartography 进行中（2026-07-18 起）**：6 子系统 analyst 并行测绘中——backend-pipeline /
  triton-to-linalg / ascend-opt-passes（AutoBlockify）/ hivm-dialect / language-cann /
  tutorials-hwmodel。digest 落 `book/cartography/digests/`，Lead 综合成 outline + ARCHITECTURE。
- **下降链已探明**：`ttir（与基座共享）→ ttadapter(triton_adapter：Triton-MLIR→Linalg)→
  npubin(bishengir-compile：Linalg→HFusion→HIVM→NPU binary)`。3 阶段（基座 GPU 是 5 段
  ttir→ttgir→llir→ptx→cubin）；根本 divergence = **换模型**（SIMT 指针张量 → 结构化 Linalg/
  仿射 tiled dataflow），bishengir 部分闭源（边界≈基座的 ptxas）。
- ✅ **大纲已获用户审批（2026-07-18「同意」）**：33 章 / 7 Part（outline-final.json）+ ARCHITECTURE.md，
  开始逐章发车。roadmap.py 生成器已建并验（7 Part 窄长条 5.17:1，4 调用/几何/Read-PNG 通过）。
- 🚗 **施工中**：ch01 鸟瞰先行（skip_impl/meta，含 book-map 详细全书地图）→ 逐 Part 推进。
  发车须 scriptPath（workflow-byname-stale-snapshot）+ 显式 instance:"triton-ascend"（护栏）。
