# 实例：triton（《Triton 源码解读》）

> 本文件 = 本实例的「源码版本 + 当前状态 + 专属规则」。通用方法论见仓库根 `CLAUDE.md`；配置见 `instances/triton/repo2book.json`。

## 源码版本（行号基线）
- 仓库：`https://github.com/triton-lang/triton.git` → `instances/triton/source/`（blobless clone）。
- **钉死 v3.2.0 @ `9641643da6c52000c807b5eeed05edaec4402a67`**（2025-01-22）。
- 选版依据（2026-07-15 用户定）：与姊妹篇 triton-ascend **v3.2.1** 配套——triton-ascend 是
  Triton **fork**（整树内嵌，树内 `__version__='3.2.0'`），其旧 master 的 third_party/triton
  submodule 钉的正是本 commit（三重验证：树内版本号 / submodule 钉点 / README 配套声明）。
  官方最新 v3.7.1；日后可整体重基（vllm 书 v0.21.0 重基有 SOP 先例）。
- 读者：advanced（zh-CN）。

## 规范路径约定
正文引用一律**仓库根相对路径**：`python/triton/runtime/jit.py:L123`、
`lib/Dialect/TritonGPU/Transforms/...`、`include/triton/...`、`third_party/nvidia/...`
（绝不写 `instances/triton/source/`——零脚手架泄漏）。

## 与姊妹篇的关系
`instances/triton-ascend/`（v3.2.1）是本仓的 fork：上游文件几乎全量在场 + 新增 `ascend/`、
`third_party/ascend/`（含 AscendNPU-IR submodule）与对上游文件的原位修改。姊妹篇逐章
`pairs_with` 指回本书章；本书是配对脊柱的基座端。

## 实例专属硬规则

### ★ pin 精确 IR 验证配方（2026-07-15 ch01 终审员实证，全书 43 章验 IR 都靠它）
**`pip install triton==3.2.0` 装出的 wheel，其 Python 前端与本 pin 逐字节相同**（已 diff：
core/standard/semantic/code_generator/compiler/jit/interpreter/backends/__init__ + nvidia
compiler/driver，全部 IDENTICAL），且自带 `libtriton.so`。**因此可以 headless、无 GPU 做真·pin 实机编译**：
```
python -m venv v32 && v32/bin/pip install triton==3.2.0
# ASTSource(fn, signature, constants).make_ir(...)        → 追踪期 IR（任何 pass 之前）
# backend.add_stages()["ttir"](...)                        → make_ttir 之后
```
- **凡给 IR 事实必须标明取自哪个阶段**（追踪期 / make_ttir 之后 / 更后）——两者差异极大：
  `add_inliner` 是 make_ttir 的**第一个 pass**，追踪期能看到的 `tt.call`/被调 `tt.func`
  在 `.ttir` dump 里**已被内联抹平**。让读者用 `TRITON_KERNEL_DUMP` 去 `.ttir` 找 `tt.call`
  = 承诺一个不可复现的证据（ch01 曾踩，见 exp-0715-1）。
- **禁止拿环境里装的新版 triton（3.5/3.6）或记忆中的上游代码当准**：上游改过 `cdiv` 体、
  加过 `BoundJITMethod` 等，拿新版"验证"会得出**错误**结论。一律以 pin 为准。
- 无法编译的部分（真跑 kernel 需 GPU）：用 `TRITON_INTERPRET=1` 与编译期产物替代；
  explainer 的 `trace_source` 如实标注，不许假装跑了真核。

### ★ 读者收益主线（2026-07-15 用户定）
全书围绕一条读者收益主线组织：**「这一章的内容如何帮我优化 Triton 算子的性能」**。
每章 hook 点破本章解锁的性能杠杆、小结回扣。结构主线（一路降级成 PTX）定顺序，
性能主线定 stakes。落点=outline 每章 `perf_payoff` 字段 + voice-guide 已注入。

### ★ 「可运行交叉验证」= pin 精确编译，非 subtract-only 精简版（2026-07-16 Lead 定）
Triton 是编译器代码库:core.py/semantic.py 等是庞大且互相缠绕的 DSL 库,对它做「同名同结构
只删不增」的精简版既不自然、也跑不起来(切一片 core.py 离不开整条编译栈)。因此本书的
**「运行起来看数值」支柱由 pin 精确编译承担**:explainer 用 `pip install triton==3.2.0`(与
pin 逐字节同)headless 编译真 kernel,观测**真实** IR/dtype/报错(比 subtract-only 重实现更真)。
- 多数章 **skip_impl**(无精简版);explainer 的 `trace_source` 记「pin-compile」并标 IR 阶段。
- 仅当某机制能干净抽取并独立跑(编译器里少见)才用精简版。cartography 的 mode 以此为准。

### 其他
- 双语栈：Python DSL 层（python/triton）与 C++/MLIR 层（lib/include）。精简版（implementer）
  预计只对 Python 层可行，MLIR pass 层多数章走 `skip_impl` 轻流程——cartography 逐章标注。

## 当前状态（2026-07-18）
- ✅ **全书 43 章定稿归档收官**（ch01–ch43 连续、无洞；HEAD `6095b803`，vllm-book-v2-rebuild 已推送）。
  - 18 条伏笔全部闭合（末伏笔 f8：ch07 block-ptr advance 滑窗守恒 → ch43 tutorial 06 真实兑现，跨 36 章）。
  - Bible 终态：glossary 516 / concepts 359 / figures 95；trace state 43 章全 `done`、INDEX 置顶 ch43。
  - primer 论文包（ch02/15/20/23/27/29/42）paper.md+meta 全部入库（07-18 补齐 ch15/20/29 的 paper.md 入库缺口）。
- 🔄 批次收尾：book-retro（ch36–ch43 最终批）挖经验候选中；ledger 已有 exp-0716-1（figure-only
  review-exhausted，5 样本，标「下次 book-retro 优先」）与 exp-0718-1（pin-vs-host arch 挑明）待落笔。
- ⚠️ git add 教训（2026-07-18）：多 pathspec `git add` 任一 typo 即**原子全败**且易被 `2>/dev/null`
  吞掉——ch40 曾因 `tritriton` typo 漏提交正文/素材两轮（9012327f→34228331 才补齐）。
  **提交后必查 `git status --short` 验没有漏网**，勿静音 git add 的 stderr。
- **审批闸豁免（2026-07-15 用户令）**：「查明所有引用论文和 gap 点，查明后无需经我审批，直接开工」
  ——论文清单+gap 点核清、primer 章并入大纲后，直接逐章发车，不等大纲审批。
