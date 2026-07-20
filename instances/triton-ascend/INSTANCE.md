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
- 🚗 **施工中（2026-07-20 状态）**：ch01/ch02/ch03/ch04/ch05 已定稿提交；ch06 在跑；
  ch07 无正文（早前 test-exhausted 逃生舱，需重发）；ch08-ch33 未开工。
  发车须 scriptPath（workflow-byname-stale-snapshot）+ 显式 instance:"triton-ascend"（护栏）。

## 本书已踩过的坑（发车/写作前先看这条）

- **地址空间「有几级」是本书最容易数错的数**（ch05 连错三轮，exp-2026-07-20-04）：
  `.td`（HIVMAttrs.td:L188-L194）定义 **7** 级，但 `ascend_ir.cc:L412-L418` 的 `py::enum_`
  **只 `.value()` 导出 5 级**（L1/UB/L0A/L0B/L0C）——**Zero 与 GM 不进 Python**。
  语言层能写出哪些门牌号，取决于那几行 `.value()`，不取决于 `.td`。
  ⇒ **凡断言「共 N 个 / N 级」，必须追到最窄的那一层**（pybind `.value()` / `__all__` / 注册表白名单），
  不能停在「整体反射 / 遍历 `__dict__`」这类措辞上。
- **写不出 `space=GM` 的 buffer**：GM↔UB 那一跳不由门牌号表达，走 ch02 的显式搬运与 ch06 的
  索引搬运算子（后者吃的是基座 Triton 的**裸指针**，不是 buffer）。禁写「GM 是 address_space 之一」。
- **精简版替身也会编码错模型**：ch05 的 conftest 曾照 `.td` 造出 7 个假枚举成员，测试遂「自洽地通过」。
  替身的成员清单必须对齐**真正的绑定导出**，并优先写**反向断言**（某名字不存在）——那才是承重的。
- **linter 批量扫要用退出码**，别 grep 输出里的 `BLOCKING`：`🟢 仅警告（无 BLOCKING）` 会被误判成红。
- 跨实例 linter 已修（exp-2026-07-20-03）：显式传章节路径时按路径定实例，**不必**再带
  `REPO2BOOK_INSTANCE=triton-ascend`。
