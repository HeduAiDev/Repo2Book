# 实例：vLLM（vLLM v1 源码解读：从真实源码读懂推理引擎）

> 本文件 = 当前实例的「源码版本 + 当前状态 + 实例专属规则」。**通用方法论/工厂运转见仓库根 `CLAUDE.md`**；本实例配置见 `instances/vllm/repo2book.json`。
> 中文、高级读者，解读 vLLM v1 推理引擎；异步三段式解耦为旗舰。

> **2026-06-21 重大转向**：旧的"从零简化重写 + 理论推导"产出太抽象、脱离代码，**全部废弃**。新书 = **直接解读真实 vLLM v1 源码**，按真实模块组织。旧 agent 提示词/经验仅作批判参考。

## 源码版本（行号基线）
- 书锚定 vLLM **v0.21.0**（发布提交 `ad7125a4`，`instances/vllm/source/` 工作树即此版）。
- 全书 ~3000 处精确行号引用已由 `scripts/remap_lines_v021.py` 确定性重映射到 v0.21.0（difflib 行级对齐：平移类自动改号、内容真改处定点重抽 v0.21.0 片段）；v0.21.0 引入的面向读者新特性另以「v0.21.0 更新」注织入。
- `f3fef123`（v0.20.1 线、v0.21.0 前 245 提交）= 升级前基线，仅作历史 diff（remapper 据此工作，可复用于未来升级）。少数被 v0.21.0 删除的代码示例（如 RMSNorm `forward_static`）显式标注为 `(基线 f3fef123)`。

## 实例专属硬规则
- **vLLM 相关代码调试一律进 Docker 容器**（host 无 CUDA/vLLM）：`scripts/vllm_docker.sh ...`，镜像 `vllm/vllm-openai:latest`。容器内 vLLM 版本可能与 v0.21.0 有行号差，仅用于观察行为。
- 正文规范路径前缀 `vllm/…`（**绝不** `instances/vllm/source/…`）。
- 架构地图 + 大纲在 `instances/vllm/book/cartography/`（`ARCHITECTURE.md` 全量、`outline-final.json` 8 Part/33 章、`map.json` 结构化）。章节用 `ch`-前缀 slug，置于 `instances/vllm/artifacts/`。

## 当前状态（2026-06-28）：📕 全书 33 章完成 + 已全量重基到 v0.21.0
- **ch01–ch33 全部完成**（31 源码解读章 + ch01/ch02/ch26 三个 meta 概览章，meta 走 `skip_impl` 轻流程无精简版）。全 APPROVED、已推 `vllm-book-v2-rebuild` 远程。规模 ≈24.5k 行正文 + 142 图。
- **v0.21.0 升级（2026-06-28）**：① 16 章自然织入「v0.21.0 更新」内容（ch03/08/09/17/18/19/23/24/25/26/27/28/29/31/32 + ch27 XPU 分发图）；② `source/` 工作树 checkout 到 v0.21.0，~3000 处行号引用确定性重映射（1458 identity + 1531 shift + 41 content-resync）；③ bible glossary+interfaces 登记 v0.21.0 增量。整理稿见 `instances/vllm/book/_v021-update/`。
- **连贯性干净**：26/26 伏笔全回收；glossary/interfaces 登记 30 章；全书 0 断裂章内锚点 / 0 半角标点 / 0 图几何问题。
- 体系经实战加固：archive 注入完整 reviewV+崩溃重试、防假通过 escape hatch、dossier-verify 对抗自核（实战拦下 ch31 SyncMPClient / ch01 CompilationMode 等事实错误）、off-spine 分层 roadmap、git push 须前台。
- **润色已大体完成**：断锚/半角/图标签重叠/术语对齐 glossary、算法维度增补 24 章均已做并推远程。剩余仅最低价值声线微调 + lint_formulas 内联密度软噪声。

## 实例专属坑
1. 别写脱离代码的抽象——正文以真实 vllm 源码为主线、自包含内嵌。
2. implementer 别过度删减/误删——只删 `delete` 批准项，`must_keep` 必保留。
3. 标记完成前跑全部 linter。
4. vLLM 相关运行进容器；行号以 v0.21.0（`ad7125a4`）为准，升级前基线 `f3fef123` 仅作历史 diff。
5. 别赌自己的上下文——决策/状态写进 trace、Bible、本文件。
