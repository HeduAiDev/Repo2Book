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
- 🚗 **施工中（2026-07-22 状态）**：**ch01–ch15 已全部定稿提交并归档**（Part 1 完 + Part 2 语言层完
  + **Part 3 triton-to-linalg 子系统核心 ch09–ch14 全完** + **ascend-opt 子系统开篇 ch15 AutoBlockify 完**）；ch16–ch33 未开工。
  下一章 **ch16**（Cube 还是 Vector:核亲和定点传播，DAG.cpp，ascend-opt 子系统 → **发车须 skip_impl:true**，纯 C++ pass 已核）。
  ⚠️ ch16 术语坑:`affine`=**核亲和(core affinity)**、**不是**多面体/仿射调度;数据流不动点属 Kildall 一路。
  ✅ ch12/ch13 零逃生;ch14 零逃生但评审 12 issue/3 blocking(完整性『四类』漏第5条 + 两图 vs 源码矛盾)全经 Lead 核源码处置。
  ⚠️ **图 blocking 错终检必须 vs 源码/正文,不能只信 blind-vs-spec**(exp-2026-07-22-03);改 gen 后必重渲+重置 PENDING;
  读 review-report 前先看现盘是否已被 revise 循环改掉(别照单重复施工)。
  发车须 scriptPath（workflow-byname-stale-snapshot）+ 显式 instance:"triton-ascend"（护栏）。
  ⚠️ triton-to-linalg 子系统章交付普遍经逃生：dossier-verify 拦计数/分析错（ch10 pass 数、ch11 m9 失败规则）、
  Implement 拦非 skip_impl、Review 偶发 API 崩溃靠 Lead 补跑缺维。**别信 brief 快照的任何计数**、
  内嵌 C++ 若重排须首块前声明、IR 名警惕三段点分错形（ch10 L549）、splat 不产生 source（ch11 核心易错点）。
- ⚠️ **triton-to-linalg 子系统（ch10–ch14+）是纯 C++ MLIR pass 章，发车须 `skip_impl:true`**
  （2026-07-22 定，对齐姊妹篇《Triton 源码解读》ch25/28/30/32/33 的 skip_impl 先例）。
  这些章的 must_keep 全是 C++ 符号（PtrAnalysis/BlockPtrAnalysis/MaskAnalysis/Unstructure 等
  `lib/*/*.cpp`），**.py 零命中、宿主无 CANN/NPU 编不动、用 Python 重写违反 HARD RULE 2（deep 不享 primer 豁免）**。
  ⚠️ **outline-final.json 里这些章标 `mode=code` 是模板默认、对本子系统不适用**——照它发车会在 Implement 站逃生（ch10 已踩，2026-07-22）。
  **交叉验证不走精简版，走两样**：① pin 精确源码逐段解读；② `unittest/Conversion/**/*.mlir` 的 lit 夹具（真实、pin 内、带 RUN+CHECK 前后对照）作 IR-dump 素材，explainer/illustrator 阶段用，trace_source 标『pin 内 lit 夹具』，**不伪造编译器 dump**。
- ⚠️ **ch10 发车必答项（务必注入 dossier brief）**：`namedOps` 的**实现语义**。
  实测 `TritonToLinalgPass.cpp:L524`（namedOps 为真时张量上的 `arith` 保持合法）与 `L651`
  （`if (!namedOps)` 才加载 `populateElementwiseToLinalgConversionPatterns`）⇒ 真实语义是
  「**别把逐元素 `arith` 摊成 `linalg.generic`**」，**不是**「发射 linalg 具名算子」；
  全仓搜 `linalg::AddOp` / 产出 `linalg.add` **零命中**。Bible 已立两条词条，但
  `bible.py due ch10` 走 arc-map、**不会自动提醒**（本书 arc-map 至今为空数组）。
- 📌 **待回修（已立案，勿遗忘）**：
  - `ch01/reviews/LEAD-PENDING-FIX.md` — ch01:L147 的 `(如 linalg.add)` 举例无据（Bible 侧已订正，正文待 writer）。
  - `ch09/reviews/LEAD-PENDING-FIX-forward-refs.md` — ch09 有 7 处裸 `chXX` 指向未写章，
    ch10/ch11 定稿后改成 markdown 链接；建议与上一条**合并成一次派工**，少动老章。
  - `instances/vllm-ascend/artifacts/ch37-primer-dspark/reviews/LEAD-PENDING-FIX.md` — 三处图面脚手架泄漏。

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
- **IR 算子名带方言前缀 `ascend.`,不是 `tt.`**：`TT_Ascend_Op` 绑定 `TritonAscend_Dialect`，
  其 `let name = "ascend"` → 打印出来是 `ascend.indirect_load` / `ascend.index_select_simd`。
  ch06 曾把 `tt.indirect_load` 写进正文+两张图+explainer+traces 共 6 处文件；正文改对后，
  `lint_chapter_map` 的「杜撰符号」检查才把图里的错名暴露出来（此前靠正文那一格互相兜底）。
- **「同名同签名、返回值语义被悄改」是 lint_fidelity 的盲区**：ch06 内嵌的 `gather.py` 被改了一行
  （`* K + k_offs` → `* N + n_offs`，把对的改成错的）；ch07 精简版 `constexpr.__eq__` 标着「节选」
  却把 `constexpr(...)` 包装改成裸 bool。**两次都是人工评审抓出来的**——`# SOURCE:` 标注只保证
  「出处存在」，不保证「逐字未改」。写/改精简版与内嵌引文时，请拿 `sed -n 'a,bp' <pin 文件>` 对眼。
- **归档不要漏章**：ch05 曾正文提交（9e5a02bd）却没派归档，Bible/trace 里一直没有它，
  直到写 ch06/ch07 归档时对照 `state.json` 的章号序列（ch01–04 后直接跳 ch06）才发现。
  **每章提交前顺手核一眼 `python3 -c "import json;print(sorted(json.load(open('instances/triton-ascend/trace/state.json'))['chapters']))"`。**
