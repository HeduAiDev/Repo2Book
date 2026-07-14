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
- 双语栈：Python DSL 层（python/triton）与 C++/MLIR 层（lib/include）。精简版（implementer）
  预计只对 Python 层可行，MLIR pass 层多数章走 `skip_impl` 轻流程——cartography 逐章标注。
- 运行验证无 GPU 时用 `TRITON_INTERPRET=1` interpreter 模式与编译期 IR dump 替代；
  explainer 的 trace_source 如实标注，不许假装跑了真核。

## 当前状态（2026-07-15）
- ✅ scaffold + blobless clone + 钉版 v3.2.0；已设为 active_instance。
- ⏳ cartography 测绘中（RUNBOOK §0.6：fan-out → synthesis → 覆盖交叉核对 → 路径核对 →
  **用户审批闸**）。大纲未获批前不发车任何章。
