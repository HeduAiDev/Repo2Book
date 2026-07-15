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

### 其他
- 双语栈：Python DSL 层（python/triton）与 C++/MLIR 层（lib/include）。精简版（implementer）
  预计只对 Python 层可行，MLIR pass 层多数章走 `skip_impl` 轻流程——cartography 逐章标注。

## 当前状态（2026-07-15）
- ✅ scaffold + blobless clone + 钉版 v3.2.0；已设为 active_instance。
- ✅ cartography 收官：43 章 / 9 Part / 7 primer；论文清单 25 条全部核真；roadmap 生成器就绪。
- 🔄 ch01 施工中（dossier 已过 5 轮对抗性自核——开篇章定义全书心智模型，从严）。
- **审批闸豁免（2026-07-15 用户令）**：「查明所有引用论文和 gap 点，查明后无需经我审批，直接开工」
  ——论文清单+gap 点核清、primer 章并入大纲后，直接逐章发车，不等大纲审批。
