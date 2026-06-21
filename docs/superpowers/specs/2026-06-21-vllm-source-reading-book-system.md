# 设计文档 — vLLM 源码解读书 · 重建系统

> 日期: 2026-06-21 · 作者: 架构师 (Claude) · 状态: 待评审
> 关联: `instances/vllm/book/cartography/`（架构地图 + 最终大纲）

## 1. 背景：为什么重建

旧体系是"从零简化重写 + 理论推导"。实测旧 ch04（连续批处理）发现**致命脱节**：

- implementer 写了简化**且杜撰**的玩具代码（如 Phase 3 自动生成 token 的伪模拟）；
- writer 把 implementer 的代码当"教材"，花 150+ 行讲 implementer 的结构，真实 vLLM 复杂度被降级成脚注式引用（"vLLM 还做了 X/Y/Z"）；
- 结果：读者学到的是 implementer 的玩具，不是 vLLM 的真实设计。

用户裁定：**废弃全部旧文章产出**，转为**直接解读真实 vLLM v1 源码**的书。旧 agent 提示词/经验仅作批判性参考。

## 2. 定位与读者

- **书名方向**：vLLM v1 源码解读（按真实模块/子系统，异步解耦为主线）。
- **源码 pin**：`f3fef123`，位于 `instances/vllm/source/`。
- **读者**：高级（精通 PyTorch / CUDA / Transformer），中文，大白话 + 技术深度。
- **范围**：8 Part / 33 章，见 `instances/vllm/book/cartography/outline-final.{json,md}`。

## 3. 核心方法论（修复脱节的关键）

三根支柱 + 两条硬性写作规则：

### 支柱 A — 档案即唯一真相源 (Dossier as the single source of truth)
旧流水线**串联传递工件**（implementer 的代码成了 writer 的教材）。新流水线让 implementer 和 writer **共享同一份对真实 vLLM 源码的深度分析（档案）**，谁都不以对方产物为准。这从结构上根除脱节。

### 支柱 B — 只做减法的可运行精简版 (subtract-only companion)
implementer 产出的精简版**与 vLLM 同名、同结构、同控制流，只删不增**：
- 每处删除标 `# SUBTRACTED: <删了什么·为什么·原 file:Lxxx>`；
- 每函数标 `# SOURCE: vllm/...:Lxxx`；
- **禁止**：杜撰 vLLM 没有的抽象/数据结构、改名、无注释简化、加玩具模拟。
- 验收判据：**"把 vLLM 删掉所有 SUBTRACTED 分支，应当就得到精简版。"**
- **防过度删减/误删**：analyst 在 `subtraction_plan.delete` 给出**唯一批准删除清单**、在 `must_keep` 列出可检测符号；implementer 只能删 delete 项、must_keep 必保留、不得按己见删其他细节；`lint_fidelity` 自动校验 must_keep 符号都在精简版（缺 = BLOCKING）。writer 缺料可要求 implementer 补回。

### 支柱 C — 内嵌真实源码、自包含 (self-contained, real source embedded)　★新增
**不指望读者开着源码读书。** 叙事必须**直接内嵌真实 vLLM 源码片段**（逐字、带 file:Lxxx），删去无关分支用 `# … 省略：<什么> …` 标记。读者只读书即可看懂真实实现。
- 内嵌真源码（裁剪无关）= 展示**完整结构**，是被解读的权威对象；
- 精简版 = **可运行最小骨架**，用于跑通/打断点/数值追踪，交叉验证理解。
- 两者**互补不冗余**：典型节奏是「内嵌真源码片段 → 逐行解读设计决策 → "剥掉 X/Y 分支，本质就是精简版这几行" → 运行精简版看数值」。

### 写作规则 1 — 每章 Roadmap（"你在这里"）　★新增
每章开头必有 Roadmap：复用全书统一的**请求生命周期主线（19 步）+ 子系统依赖图**地图，高亮当前章位置，一句话说明「上一章立了什么 / 本章解决什么 / 下一章接什么」。地图用 svg-diagram 出一张母版，各章高亮不同节点。

### 写作规则 2 — 源码手撕，源码即正文
延续旧体系唯一正确的内核：每章必须**走读真实源码**（现在是内嵌呈现）+ 推导背后理论 + 差异分析。只有理论没有源码走读 = 废章。

### 写作规则 3 — 零脚手架泄漏（读者视角）　★新增
正文是**正式出版物**，绝不泄漏生产脚手架：
- 引用源码用**规范 vLLM 路径**（`vllm/v1/engine/async_llm.py:L280`），**绝不**出现 `instances/vllm/source/...` 这种本仓库目录结构；
- 章节用**自然标题**（如「输入预处理：从 Prompt 到 EngineCoreRequest」），**绝不**出现 `Cell 1/Cell 2` 这类内部脚手架名——读者完全不懂；
- **绝不**提及内部文件（`impl-notes.md`、`dossier`、「详见 xxx.md」「这里只截取部分」）——出版物里这些都不存在。

## 4. 章节模板（内部脚手架，按本书重定义）

| 段 | 名称 | 要点 |
|---|---|---|
| 0 | **Roadmap** ★ | 你在这里（主线高亮图）+ 前后衔接 + 本章依赖 |
| 1 | Hook | 从真实入口文件切入，给出本章要害问题 |
| 2 | Problem / 设计动机 | 用具体数字/场景说明 vLLM 为何这么设计 |
| 3 | 源码走读 ★ | **内嵌真实源码片段（裁剪无关）**，逐段解读控制流与设计决策 |
| 4 | 理论深挖 | 推导/证明/复杂度量化（算法章必备：图示+数值追踪+归纳证明） |
| 5 | 精简版交叉验证 | subtract-only 精简版 + 运行输出，"剥掉哪些分支得到它" |
| 6 | 差异分析 | 精简版 vs 真实 vLLM：删了什么、为什么安全 |
| 7 | Source Map 表 | ≥5 行：精简版↔真实 file:Lxxx↔改动↔原因 |
| 8 | 小结 | 关键要点 + 通向下一章 |

> 上表的「段」是**内部脚手架**（供写作/评审确保覆盖），**绝不作为章节可见标题**——成稿用自然标题。Cell 是脚手架不是牢笼；写作者可为讲清而调整，但 Roadmap / 内嵌源码（规范 vLLM 路径）/ 源码走读 / 精简版交叉验证 / 零脚手架泄漏为强制项。

## 5. 新 agent 体系

混合编排：**Workflow 编排 + 少量持久角色**。角色 = 自定义 agent 类型（`.claude/agents/*.md` 精炼提示词），由 per-chapter workflow 通过 `agentType` 调用。

| 角色 | 持久? | 职责 | 关键契约 |
|---|---|---|---|
| **Analyst**（新） | 否（workflow 扇出） | 深读真实源码产出**档案**：代码主干 file:Lxxx + **要内嵌的源码片段** + 设计决策 + 数据流 + 数学 + **减法计划** + 配图计划 | 档案是 implementer/writer 共同真相源 |
| **Implementer** | **章节级**（迭代期续接） | 按减法计划产出 subtract-only 精简版；**TDD 先写测试** | 只删不增；`# SOURCE:`/`# SUBTRACTED:` 全标注 |
| **Tester** | 否（确定性闸门） | 跑通 + 测试过 + 行为对齐档案记录的 vLLM 行为 | 反压闸门；verification-before-completion |
| **Writer** | **章节级**（迭代期续接） | 叙事主线=真实源码；内嵌真源码逐段解读；精简版作交叉验证；出图 | 不以精简版为主角；若大篇幅讲精简版=解读不够 |
| **Reviewer** | **章节级**（协作回环） | **协作式**评审：保真度/可读性/算法可理解性/Roadmap/自包含/公式 | 提改法不死卡；新增首要维度=**vLLM 保真度** |
| **Archivist** | **全书持久** | 记录/备份/state/上下文再水化 | 跨会话长期记忆 |

**两层持久**：Archivist 全书常驻（长期记忆）；implementer/writer/reviewer **按章持久**——在本章迭代期间作为**命名 agent 用 SendMessage 续接、上下文原样保留**，章末释放。既非全书常驻 tmux 守护（旧体系脆弱点），又保证迭代连续性。

## 6. Per-chapter 混合 Workflow

`scripts` 提供一个 `chapter-pipeline` workflow（可恢复、内建并行）：

```
Phase A 档案（扇出并行 → 汇总 barrier）
  a1 代码主干分析（真实 file:Lxxx 范围 + 内嵌片段）
  a2 理论/数学分析（WHY、推导、复杂度）
  a3 减法计划（删什么/为什么/必须保留什么）
  a4 配图计划（svg-diagram 规格 + Roadmap 高亮）
  → 合并 dossier.json

Phase B 实现+测试（pipeline，失败重试回 implementer）
  implementer(TDD) → tester 闸门

Phase C 叙事
  writer 消费 dossier + 精简版 → chapter.md + 图（内嵌真源码、Roadmap）

Phase D 评审（多维并行 → 协作汇总；REVISE 有界回环）
  fan out: 保真∥可读∥算法可理解∥公式/lint → 汇总成单一 verdict + 协作反馈

Phase E 归档
  archivist 记录/备份/更新 state
```

并行点：① Phase A 四路分析扇出 ② Phase D 多维评审扇出 ③ **章间**：无依赖章节多条 pipeline 并行（worktree 隔离）。

## 7. 编排与记忆

- **Workflow 为骨架**：确定性、可恢复（`resumeFromRunId`）、内建并发上限。
- **持久角色**：仅 Archivist（+ trace/state.json 长期记忆），跨会话再水化，规避旧 tmux 团队的脆弱/脱节。
- **角色 = 持久提示词文件**：`.claude/agents/{analyst,implementer,tester,writer,reviewer,archivist}.md`。workflow 实际用 `agentType:'general-purpose'` spawn，并在每段 prompt **首行强制"先读 `.claude/agents/{role}.md`"** 注入角色契约（不赌自定义 agentType 解析） ⇒ 这就是"aiteam + workflow"的落地。
- **知识/智慧**：`knowledge/`（仓库特定事实，TTL）、`wisdom/`（跨实例模式）沿用，写作前查、收工后抽（learn.py）。

## 7.5 迭代与重做：非持久化下如何保证收敛

**核心：不持久化 ≠ 重做时失忆。** 三层保证：

- **第 1 层 · 让重做本来就少**：档案为真相源 → 一次成型质量更高（不再串联猜对方）；确定性闸门前置——linter（formula/fidelity/self-contained/source-grounding）+ tester 在 LLM 评审**之前**跑，机械问题由"定点小修"处理，不占完整重做轮次；reviewer 协作式给**具体改法+建议方案**，单轮命中率高。
- **第 2 层 · 重做带全上下文**：每章维护 **revision ledger**（dossier + 每次产出 + 每次评审的具体反馈 + 改了什么/还差什么），任何重做都注入完整 ledger，fresh agent 也带着"第 2 次试过 X、reviewer 说仍有 Y"的记忆；对 implementer/writer/reviewer 用**章节级命名 agent + SendMessage 续接**，上下文原样保留。
- **第 3 层 · 有界 + 升级**：同一问题 >K 轮（默认 3）→ 升级 Team Lead（我）→ 必要时拉上用户；loop-detection 防止 thrash；workflow `resumeFromRunId` 可恢复，迭代被打断已完成阶段命中缓存。
- **第 4 层 · 逃生舱（agent 主动拉闸）**：每个阶段 agent 返回 `status=OK|BLOCKED`。发现路线/档案错（源码与计划不符、subtraction_plan 破坏正确性、缺料）→ 返回 `BLOCKED`+reason，**不硬着头皮做错**；脚本立即 `return` 中止并把 reason 交给 Lead，我修正后 `resumeFromRunId` 续跑。**dossier 阶段还有对抗性自核**（第二个 analyst 校验档案忠于源码），在 implementer 动工**之前**就拦住错误路线。Lead 也可随时 `TaskStop` 正在跑的 workflow。
- **跨章学习**：反复出现的毛病 → wisdom/knowledge DB → 后续章节 agent 开工前被预警 → 重做越来越少。

## 7.6 跨章节连贯性、伏笔与回收

**关键认知**：33 章的连贯性**不能赌单个常驻 agent 的对话记忆**——Claude Code 约 50 轮压缩上下文，写到第 30 章早已丢失 ch04 的伏笔细节。把连贯性**外化为持久工件**不是退而求其次，而是即便有持久化也该这么做的正解。非持久化反而逼我们做对。

- **A · Book Bible（全书圣经，Archivist 持有，持久）**：术语/译名表；精简版**代码接口注册表**（各章建立的类/方法签名，保后续章节兼容）；**伏笔登记**`{埋什么·埋于 chN·将于 chM 回收}`；**回收/承诺登记**；voice/style 指南；跨章依赖（来自依赖图）。
- **B · 自顶向下弧线设计**：因为大纲 + 依赖图 + 19 步主线**开局即全知**，写作前由架构师设计**弧线图**（哪些概念贯穿全书、何处埋伏笔、何处回收），注入每章 dossier。例（与 `bible/arc-map.json` seed 一致）：「ch04 埋 per-request 队列 → ch08 回收」「ch04 埋 EngineCore 进程边界/IPC → ch07 回收」。伏笔是**设计出来的**，不是碰运气涌现。
- **C · 每章 read-before / update-after 协议**：写前读 圣经（本章**应埋/应收**的条目）+ Archivist 再水化简报 + dossier；写后更新圣经（新术语/接口/已埋伏笔/已回收/新承诺）。
- **D · 连贯性审计**：reviewer 的 cross-chapter 维度对照圣经校验；每完成一个 Part 跑 **continuity-audit workflow（待实现/后续，当前由 Lead 手工或命名 agent 触发）** 并行扫全 Part（未回收的伏笔 / 断裂的回调 / 术语漂移 / 接口不符）→ 标记修复。这比常驻 writer 更强——它不会回头重审旧章。
- **E · Archivist 是持久守护者**：唯一全书常驻角色，持有圣经 + trace，发再水化简报，触发审计。**持久化用在真正该用的地方（记忆），而非脆弱的逐角色守护进程。**

## 7.7 agent 间协同：workflow 的能与不能

- **workflow 不做的**：workflow 里的 agent 是**一次性、相互隔离**的——彼此之间**无直接 SendMessage、无活体来回对话、不能在任务中途互相提问**。
- **workflow 做的是"编排者中介式协同"**：脚本就是协调者，把一个 agent 的**结构化输出**喂进另一个 agent 的 prompt。这覆盖绝大多数协同：
  - **顺序交接**（dossier → implementer → tester → writer）：`pipeline`。
  - **并行交叉校验 / pair / panel / swarm**：`parallel()` 同输入多 agent + 一个**合并/仲裁 agent**（旧体系的 pair/panel/swarm 拓扑天然映射于此）。
  - **修订回环（writer↔reviewer）**：脚本中介的往返 + revision ledger。reviewer 结构化输出含 `{问题, 建议改法, 理由, 可商榷点}`，writer 重做输出含 `{对每条的采纳/反驳 + 理由}`（receiving-code-review：不表演式同意）→ **协同被显式记录，确定性收敛**。
  - **角色间 Q&A**：agent 在 schema 里返回 `questions` 字段 → 脚本路由给另一角色 → 答案再喂回。
- **需要真·活体双向对话 → Lead 主导的命名 agent + SendMessage**（单个 workflow 内做不到）：硬骨头的协作修订、升级场景，双方都保留**活上下文**。由我（Team Lead）编排。
- **混合边界正按此切**：并行 + 确定性 + 结构化交接 → **Workflow**；活体双向迭代 / 升级 → **Lead + 命名 agent + SendMessage（章节级持久角色）**。

## 8. superpowers 集成映射

| skill | 用在哪 |
|---|---|
| brainstorming | 全书/各 Part 立项（本文档即产物） |
| writing-plans / executing-plans | 每个 Part 的实施计划 + 受检执行 |
| test-driven-development | implementer/tester：先写测试再实现精简版 |
| verification-before-completion | tester + 所有"标记完成"前的实证 |
| requesting/receiving-code-review | reviewer↔writer 协作回环（不表演式同意，技术严谨） |
| systematic-debugging | 精简版行为与档案不符时 |
| dispatching-parallel-agents / subagent-driven-development | workflow 扇出编排 |
| using-git-worktrees | 章间并行隔离 |
| svg-diagram | Roadmap 母版 + 算法图示 + ch26 代码→架构图 |

## 9. 质量闸门与 linter

- **fidelity linter（新，`lint_fidelity.py`）**：精简版每 def/class 有 `# SOURCE:`；无杜撰符号；叙事真实 `vllm/` 引用数 ≥ 精简版引用数；**over_subtraction**——dossier.must_keep 符号必须都在精简版（缺=BLOCKING）。
- **chapter-structure linter（新，`lint_chapter_structure.py`）**：一个脚本**三查**——Roadmap 段存在、内嵌真实源码块（```python + vllm/ 路径）、**零脚手架泄漏**（无 instances/vllm/source、无 Cell N、无内部文件引用）。
- **formula linter**（沿用）：`\text{}`/`\boxed{}`/`\tag*{}`/inline `\frac` 等阻断项。
- **source-grounding linter**（沿用）：Cell 引用、`# REFERENCE`、Source Map ≥5 行。
> 共 **4 个 linter 脚本**：`lint_fidelity` / `lint_chapter_structure` / `lint_formulas` / `lint_source_grounding`。Roadmap 与自包含检查**已并入** `lint_chapter_structure`，不是独立脚本。

> **运行环境（硬约束）**：vLLM 相关代码/测试一律在 vLLM 容器（`vllm/vllm-openai:latest`，CUDA）内跑，用 `scripts/vllm_docker.sh`；精简版纯单元测试（不 `import vllm`）可在 host 跑。容器 vllm 0.15.1 与源码 pin `f3fef123` 行号可能略差——**行号引用以 `f3fef123` 源码为准，容器仅用于观察/验证行为**；`test-report.json` 记录 docker 命令 + 镜像 tag + vllm 版本。

## 10. 旧体系处置

- **废弃**：全部旧章节叙事产出；脆弱的持久 tmux 团队编排与心跳/信号脚本（monitor/heartbeat/signal/spawn_pipeline）。
- **保留/改造**：archivist.py、learn.py、decide.py、guard_narrative.py、lint_formulas.py、lint_source_grounding.py、trace/knowledge/wisdom 体系。
- **新增**：chapter-pipeline workflow、analyst 角色、fidelity/self-contained linter、Roadmap 母版。
- 旧 artifacts 章节目录暂留作批判参考，不复用。

## 11. ch04 试点计划（旗舰：AsyncLLM 三段式）

> 依赖 ch03（配置）；试点时档案自带必要前置背景，先打通新流水线全链路。

1. 跑 `chapter-pipeline ch04`：档案聚焦 `async_llm.py`（generate/add_request/output_handler）、三段式解耦、per-request `RequestOutputCollector` 队列。
2. implementer 产出 subtract-only 的 AsyncLLM 精简版（同名 generate/add_request/_run_output_handler，删 multiproc/DP/异常恢复，标注）。
3. tester 验证异步队列行为。
4. writer 内嵌真实 `async_llm.py` 片段逐段解读 + Roadmap（主线高亮"Stage1/IPC/Stage3"）+ 精简版交叉验证。
5. 多维协作评审 → 归档。
6. 复盘：是否根除脱节、自包含与 Roadmap 是否到位 → 据此再调提示词（架构师持续迭代职责）。

## 12. 下一步

经你评审本文档后 → 用 writing-plans 出**实施计划**（建 analyst/精炼各角色提示词、写 chapter-pipeline workflow、写两个新 linter、做 Roadmap 母版、跑 ch04 试点）。

## 13. 开放项

- 全书统一 Roadmap 母版的视觉形态（线性主线条 vs 分层依赖图）——出图时定。
- 内嵌真源码的体量上限（单块行数）与省略粒度——试点 ch04 校准。
- 是否将 ch26"代码→架构图"出图法回灌为各章图示规范附录。
