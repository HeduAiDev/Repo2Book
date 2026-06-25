# repo2book — vLLM 源码解读书工厂（v2，2026-06-21 重建）

把真实代码仓库变成**源码解读型**技术书。当前实例：vLLM v1 推理引擎，中文，高级读者。

> **2026-06-21 重大转向**：旧的"从零简化重写 + 理论推导"产出太抽象、脱离代码，**全部废弃**。新书 = **直接解读真实 vLLM v1 源码**，按真实模块组织，异步三段式解耦为旗舰。旧 agent 提示词/经验仅作批判参考。

## 📍 先读这些（架构师/编排者的持久文档）

你（主 session）是 **Team Lead / 架构师**，也会被上下文压缩。**不要靠记忆运转工厂**——靠这些持久文档：

- **操作手册**：`docs/superpowers/ARCHITECT-RUNBOOK.md` ← 怎么发车/监控/处理逃生舱/续跑（**运转工厂先读它**）
- **设计依据**：`docs/superpowers/specs/2026-06-21-vllm-source-reading-book-system.md`（方法论/角色/workflow/连贯性/协同/逃生舱）
- **实施计划**：`docs/superpowers/plans/2026-06-21-vllm-book-system-rebuild.md`
- **架构地图 + 大纲**：`instances/vllm/book/cartography/`（`ARCHITECTURE.md` 全量、`outline-final.json` 8 Part/33 章、`map.json` 结构化）
- **Book Bible**：`instances/vllm/book/bible/`（术语/接口/伏笔回收/声线——跨章连贯性真相源）
- **源码 pin**：`f3fef123`，根 `instances/vllm/source/`（blobless clone）

## ⛔ HARD RULES

1. **叙事守护**：主编排者**不得**直接写/编 `artifacts/*/narrative/chapter.md`——只有 Writer 角色可写。质量不对就**改提示词，不改章节内容**。
2. **只做减法不做加法**：implementer 产出的精简版与 vLLM 同名/同结构/同控制流，**只删不增**，不杜撰任何 vLLM 没有的东西。
3. **零脚手架泄漏**：正文是正式出版物——规范 `vllm/...` 路径（**绝不** `instances/vllm/source/`）、自然标题（**绝不** "Cell N"）、不提内部文件（impl-notes.md/dossier）。
4. **vLLM 相关代码调试一律进 Docker 容器**（host 无 CUDA/vLLM）：`scripts/vllm_docker.sh ...`，镜像 `vllm/vllm-openai:latest`。
- 可改：`.claude/agents/`、`.claude/workflows/`、`scripts/`、`docs/`、`CLAUDE.md`、`repo2book.json`、`instances/vllm/book/`。

## 核心方法论（修复旧体系脱节）

三支柱 + 写作规则：
- **A 档案即唯一真相源**：analyst 深读真实源码产出 `dossier.json`（含**要内嵌的真实源码片段** + 减法计划）。implementer 和 writer 都吃这份，**不以对方产物为准** → 结构性根除"writer 花篇幅讲 implementer 杜撰代码"的脱节。
- **B 只做减法的可运行精简版**：忠实子集，`# SOURCE:`/`# SUBTRACTED:` 全标注。**防过度删减**：只删 dossier `delete` 批准项、`must_keep` 符号必保留（`lint_fidelity` 校验）。
- **C 自包含、内嵌真源码**：不指望读者开着源码——正文直接内嵌真实源码片段（裁剪无关分支），逐段解读。精简版作"跑起来看数值"的交叉验证，不是主角。
- **每章开场 Roadmap**：复用 `instances/vllm/book/assets/roadmap/roadmap.py` 出"你在这里"图 + 前后衔接。

## 运转工厂：混合编排（Workflow + 少量持久角色）

- **per-chapter workflow** `.claude/workflows/chapter-pipeline.js`：6 阶段 `Dossier→Implement→Test→Write→Review→Archive`，含 impl↔test / write↔review 有界回环、多维并行评审、**逃生舱**（任一阶段返回 `status=BLOCKED` → 立即中止升级 Lead）、dossier 对抗性自核。
- **发车**（详见 RUNBOOK）：`Workflow({name:"chapter-pipeline", args:{chapter_id, slug, source_root, focus, highlight, paths}})`。后台跑，完成/逃生舱触发会通知我；可 `/workflows` 看进度、`TaskStop` 急停、`resumeFromRunId` 续跑。
- **6 角色 = 持久提示词**（`.claude/agents/{analyst,implementer,tester,writer,reviewer,archivist}.md`），workflow 经 agentType 调用 / 或 agent 自读契约。**持久分两层**：提示词+经验持久（文件），进程按任务 spawn（迭代靠 dossier+ledger+SendMessage 续接）。
- **活体双向迭代 / 升级**：workflow 做不到的，由 Lead（我）+ 命名 agent + SendMessage 编排。

## 质量闸门（确定性 linter，前置于 LLM 评审）

```
python3 scripts/lint_fidelity.py {chapter_dir}            # 保真度：# SOURCE 全覆盖/无杜撰/不喧宾夺主/无过度删减(must_keep)
python3 scripts/lint_chapter_structure.py {chapter}/narrative/chapter.md  # Roadmap + 内嵌真源码 + 零脚手架泄漏
python3 scripts/lint_formulas.py {chapter}/narrative/chapter.md           # 公式可渲染
python3 scripts/lint_source_grounding.py {chapter}/                        # 源码根基
```
机械问题让 writer 定点小修，不退整章。

## 跨章连贯性（不赌易失记忆 → 显式持久工件）

- **Book Bible**（`instances/vllm/book/bible/`，Archivist 持有）：术语译名、精简版接口注册表、**伏笔/回收登记**、声线指南。
- `python3 scripts/bible.py due {chapter_id}` → 本章应埋/应回收项；写后回写。
- **伏笔是自顶向下设计的**（大纲+依赖图开局即全知），注入每章 dossier。
- **连贯性审计**：每完成一个 Part 跑并行审计（未回收伏笔/术语漂移/接口不符）。

## 记忆体系
- **Archivist**（唯一全书持久角色）：trace 长期记忆 + Book Bible + 再水化简报。`scripts/archivist.py`。
- **knowledge/**（仓库特定事实，TTL）+ **wisdom/**（跨实例模式）：写前查、收工后 `scripts/learn.py extract`。
- **架构师自身连贯性**：本 CLAUDE.md + RUNBOOK + trace 决策记录——保证失忆/换会话后仍能运转工厂。

## 公式规则（NON-NEGOTIABLE，auto-REJECT）
`\text{}`→`\mathrm{}`；`\boxed{}`→markdown 粗体标题；`\tag*{}`→`$$` 外；inline `\frac`→提升为 `$$` 块；`$$` 与内容分行。inline 仅限单符号/简单式。

## 当前状态（2026-06-25）：📕 全书 33 章草稿全部完成
- **ch01–ch33 全部完成**（31 源码解读章 + ch01/ch02/ch26 三个 meta 概览章，meta 走 `skip_impl` 轻流程无精简版）。全 APPROVED、已推 `vllm-book-v2-rebuild` 远程。规模 ≈24.5k 行正文 + 142 图。
- **连贯性干净**：26/26 伏笔全回收；glossary/interfaces 登记 30 章；全书 0 断裂章内锚点（`scripts/lint_anchors.py --all` 校验）。
- 旧 `NN-*` 文章已从 main 与 vllm-book-v2-rebuild 两分支删除。
- 体系经实战加固：archive 注入完整 reviewV+崩溃重试、防假通过 escape hatch（评审 agent 全失败不静默 APPROVE）、dossier-verify 对抗自核（实战拦下 ch31 SyncMPClient/ch01 CompilationMode 等事实错误）、off-spine 分层 roadmap、写手提示词强化、git push 须前台（后台 SSH 失败）。
- **润色已大体完成**：断锚(10, `lint_anchors.py`)、半角标点(11, `lint_punct.py`)、图标签重叠(3)、术语对齐 glossary(6)、**算法维度增补 24 章**(Wave1 ch17-33 按 reviewer 点名 + Wave2 ch09-16 按 dossier.theory：逐拍 tick 表/归纳证明/复杂度量化, 全锚定 dossier 数字无杜撰)均已做并推远程。复验全书 structure/formulas/fidelity 无 BLOCKING、0 断锚、0 半角。
- **剩余仅余最低价值可选项**：个别长句拆分/过渡句/`本章只…`模板等纯声线微调、lint_formulas 内联密度软噪声。ch03–16 的 on-disk review-report 为 archivist 有损重建(约 3 条/章)，完整 issues 在工作流通知里。
- 新书章节用 `ch`-前缀 slug，置于 `instances/vllm/artifacts/`。

## 常见坑
1. 别写脱离代码的抽象——正文以真实 vllm 源码为主线、自包含内嵌。
2. implementer 别过度删减/误删——只删 `delete` 批准项，`must_keep` 必保留。
3. 标记完成前跑全部四个 linter。
4. vLLM 相关运行进容器；行号以 `f3fef123` 为准（容器 vllm 0.15.1 可能有行号差，仅用于观察行为）。
5. 别赌自己的上下文——决策/状态写进 trace、Bible、本文档。
