# repo2book — 把代码仓变成「源码解读型」技术书（v2 工厂）

把任意真实代码仓库变成**源码解读型**技术书：直接解读真实源码、按真实模块组织、正文自包含内嵌真源码逐段讲。

> 本文件 = **通用操作手册（仓库无关）**。**当前在写哪本书**由 `repo2book.json` 的 `active_instance` 决定 → 看 `instances/<active>/INSTANCE.md`（该实例的源码版本 / 当前状态 / 专属硬规则）。
> 新开一本：`python3 scripts/new_instance.py <name> --repo <git-url> [--title …] [--activate]`，再按 RUNBOOK §0 出地图/大纲。

## 📍 先读这些（架构师/编排者的持久文档）

你（主 session）是 **Team Lead / 架构师**，也会被上下文压缩。**不要靠记忆运转工厂**——靠这些持久文档：

- **操作手册**：`docs/superpowers/ARCHITECT-RUNBOOK.md` ← 怎么发车/监控/处理逃生舱/续跑（**运转工厂先读它**）
- **设计依据**：`docs/superpowers/specs/2026-06-21-vllm-source-reading-book-system.md`（方法论/角色/workflow/连贯性/协同/逃生舱——以 vLLM 为首例，方法论通用）
- **当前实例**：`repo2book.json.active_instance` → 实例配置 `instances/<active>/repo2book.json`、状态与专属规则 `instances/<active>/INSTANCE.md`、架构地图 `…/book/cartography/`、Book Bible `…/book/bible/`。
- **实例解析**：脚本统一经 `scripts/instance.py` 定位活动实例（或环境变量 `REPO2BOOK_INSTANCE` 覆盖）；linter 的 `--all` 自动扫活动实例。

## ⛔ HARD RULES

1. **叙事守护**：主编排者**不得**直接写/编 `artifacts/*/narrative/chapter.md`——只有 Writer 角色可写。质量不对就**改提示词，不改章节内容**。
2. **只做减法不做加法**：implementer 产出的精简版与源码**同名/同结构/同控制流，只删不增**，不杜撰源码没有的东西。
3. **零脚手架泄漏**：正文是正式出版物——规范源码路径（如 `vllm/...`，**绝不** `instances/<x>/source/`）、自然标题（**绝不** "Cell N"）、不提内部文件（dossier/impl-notes.md）。
4. **实例专属硬规则**见 `instances/<active>/INSTANCE.md`（如某栈调试须进容器、运行环境约束、源码版本/行号基线）。
- 可改：`.claude/agents/`、`.claude/workflows/`、`scripts/`、`docs/`、`CLAUDE.md`、`repo2book.json`、`instances/<active>/`。

## 核心方法论（修复"脱离代码"的脱节）

三支柱 + 写作规则：
- **A 档案即唯一真相源**：analyst 深读真实源码产出 `dossier.json`（含**要内嵌的真实源码片段** + 减法计划）。implementer 和 writer 都吃这份，**不以对方产物为准** → 结构性根除"writer 花篇幅讲 implementer 杜撰代码"的脱节。
- **B 只做减法的可运行精简版**：忠实子集，`# SOURCE:`/`# SUBTRACTED:` 全标注。**防过度删减**：只删 dossier `delete` 批准项、`must_keep` 符号必保留（`lint_fidelity` 校验）。
- **C 自包含、内嵌真源码**：不指望读者开着源码——正文直接内嵌真实源码片段（裁剪无关分支），逐段解读。精简版作"跑起来看数值"的交叉验证，不是主角。
- **每章开场 Roadmap**：复用 `instances/<active>/book/assets/roadmap/roadmap.py` 出"你在这里"图 + 前后衔接。

## 运转工厂：混合编排（Workflow + 少量持久角色）

- **per-chapter workflow** `.claude/workflows/chapter-pipeline.js`：6 阶段 `Dossier→Implement→Test→Write→Review→Archive`，含 impl↔test / write↔review 有界回环、多维并行评审、**逃生舱**（任一阶段返回 `status=BLOCKED` → 立即中止升级 Lead）、dossier 对抗性自核。
- **发车**（详见 RUNBOOK）：`Workflow({name:"chapter-pipeline", args:{chapter_id, slug, source_root, focus, highlight, paths}})`。后台跑，完成/逃生舱触发会通知我；可 `/workflows` 看进度、`TaskStop` 急停、`resumeFromRunId` 续跑。
- **6 角色 = 持久提示词**（`.claude/agents/{analyst,implementer,tester,writer,reviewer,archivist}.md`，已去 vLLM 化、仓库无关），workflow 经 agentType 调用 / 或 agent 自读契约。**持久分两层**：提示词+经验持久（文件），进程按任务 spawn（迭代靠 dossier+ledger+SendMessage 续接）。
- **活体双向迭代 / 升级**：workflow 做不到的，由 Lead（我）+ 命名 agent + SendMessage 编排。

## 质量闸门（确定性 linter，前置于 LLM 评审）

```
python3 scripts/lint_fidelity.py {chapter_dir}                            # 保真度：# SOURCE 全覆盖/无杜撰/不喧宾夺主/无过度删减(must_keep)
python3 scripts/lint_chapter_structure.py {chapter}/narrative/chapter.md  # Roadmap + 内嵌真源码 + 零脚手架泄漏
python3 scripts/lint_formulas.py {chapter}/narrative/chapter.md           # 公式可渲染
python3 scripts/lint_source_grounding.py {chapter}/                        # 源码根基
python3 scripts/lint_anchors.py --all   # 章内锚点    python3 scripts/lint_punct.py --all   # 半角标点
python3 scripts/lint_diagram_geometry.py --all   # 图：文字越界/相撞/压框/箭头悬空（--all 走活动实例）
```
机械问题让 writer 定点小修，不退整章。

## 跨章连贯性（不赌易失记忆 → 显式持久工件）

- **Book Bible**（`instances/<active>/book/bible/`，Archivist 持有）：术语译名、精简版接口注册表、**伏笔/回收登记**、声线指南。
- `python3 scripts/bible.py due {chapter_id}` → 本章应埋/应回收项；写后回写。（bible.py 经 `instance.py` 自动定位活动实例的 bible。）
- **伏笔是自顶向下设计的**（大纲+依赖图开局即全知），注入每章 dossier。
- **连贯性审计**：每完成一个 Part 跑并行审计（未回收伏笔/术语漂移/接口不符）。

## 记忆体系
- **Archivist**（唯一全书持久角色）：trace 长期记忆 + Book Bible + 再水化简报。`scripts/archivist.py`（按 `repo2book.json.active_instance` 定位实例）。
- **knowledge/**（仓库特定事实，TTL，每实例）+ **wisdom/**（跨实例模式，2+ 实例出现才提升）：写前查、收工后 `scripts/learn.py extract`。
- **架构师自身连贯性**：本 CLAUDE.md + RUNBOOK + 实例 INSTANCE.md + trace 决策记录——保证失忆/换会话后仍能运转工厂。

## 公式规则（NON-NEGOTIABLE，auto-REJECT）
`\text{}`→`\mathrm{}`；`\boxed{}`→markdown 粗体标题；`\tag*{}`→`$$` 外；inline `\frac`→提升为 `$$` 块；`$$` 与内容分行。inline 仅限单符号/简单式。

## 当前实例 → 看 INSTANCE.md
- 活动实例、源码版本/行号基线、当前进度、实例专属坑：**`instances/<active>/INSTANCE.md`**（`<active>` = `repo2book.json` 的 `active_instance`，当前为 `vllm`）。
- **新建一本书**：`python3 scripts/new_instance.py <name> --repo <git-url> --title "…" --prefix <规范路径前缀> --activate` → scaffold 实例骨架 + blobless clone 源仓 → 按 RUNBOOK §0 出架构地图 + 大纲 → 逐章发车。

## 常见坑（通用）
1. 别写脱离代码的抽象——正文以真实源码为主线、自包含内嵌。
2. implementer 别过度删减/误删——只删 `delete` 批准项，`must_keep` 必保留。
3. 标记完成前跑全部 linter（含 `--all` 锚点/半角/图几何）。
4. 别赌自己的上下文——决策/状态写进 trace、Bible、本文档 / `INSTANCE.md`。
5. **git push 须前台**（后台 shell SSH 鉴权失败）；只在用户要求时提交/推送。
