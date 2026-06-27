# repo2book

> 把**任意代码仓库**变成一本**源码解读型**技术书 —— 多 agent 工作流，直接解读真实源码、按真实模块组织、正文自包含内嵌真源码逐段讲。

不是"从零重写个玩具"，也不是"贴一堆源码链接"。repo2book 的书 = 拿真实仓库当教材，**内嵌真实源码片段**逐段解读，配一个"只做减法"的可运行精简版交叉验证，外加每章 Roadmap、公式推导、数值追踪、架构图。

**首本实例**：[vLLM v1 源码解读](instances/vllm/)（8 Part / 33 章，≈24.5k 行正文 + 142 图，锚定 vLLM v0.21.0）。

---

## 快速开始：给任意仓库开一本书

```bash
# 1. 新建实例（scaffold 骨架 + blobless 克隆目标仓）
python3 scripts/new_instance.py redis \
    --repo https://github.com/redis/redis.git \
    --title "Redis 源码解读" --prefix src --lang C --activate

# 2. 在 instances/redis/source/ pin 一个 commit，填好 instances/redis/INSTANCE.md
# 3. 出架构地图 + 大纲（见 docs/superpowers/ARCHITECT-RUNBOOK.md §0）
#    → instances/redis/book/cartography/{ARCHITECTURE.md, outline-final.json}
# 4. 逐章发车（主 session 即 Team Lead，有 Workflow 工具）：
#    Workflow({ name:"chapter-pipeline", args:{ chapter_id, slug, focus, highlight, paths, source_root }})
```

主 session（你对话的 AI）就是 **Team Lead / 架构师**：它读 `CLAUDE.md` + RUNBOOK，调度 6 个角色 agent，按工作流逐章产书。

---

## 它怎么工作

**三支柱方法论**（根除"正文脱离代码"）：
- **A 档案即唯一真相源** — analyst 深读真实源码产出 `dossier.json`（含要内嵌的真源码片段 + 减法计划）；implementer 与 writer 都吃这份，不以对方产物为准。
- **B 只做减法的精简版** — 与源码同名/同结构/同控制流，`# SOURCE:`/`# SUBTRACTED:` 全标注，`lint_fidelity` 防过度删减。
- **C 自包含内嵌真源码** — 正文直接内嵌真实源码片段逐段讲，精简版只作"跑起来看数值"的交叉验证。

**编排** = per-chapter 工作流 `.claude/workflows/chapter-pipeline.js`，6 阶段 `Dossier→Implement→Test→Write→Review→Archive`，含有界回环、多维并行评审、**逃生舱**（任一阶段 `BLOCKED` 即早停升级）、dossier 对抗性自核。

**6 个持久角色**（`.claude/agents/`，仓库无关）：analyst / implementer / tester / writer / reviewer / archivist。

**质量闸门**（确定性 linter，前置于 LLM 评审）：保真度、章节结构、公式可渲染、源码根基、章内锚点、半角标点、图形几何（文字越界/相撞/压框/箭头悬空）。

**跨章连贯性**：每实例一本 **Book Bible**（术语/接口/伏笔回收/声线），archivist 持有；**wisdom/** 收跨实例通用模式。

---

## 目录结构

```
CLAUDE.md                       通用操作手册（仓库无关，每会话自动加载）
repo2book.json                  顶层注册表：active_instance + 实例清单 + 共享资源
README.md                       本文件
docs/superpowers/               设计 spec / 实施 plan / ARCHITECT-RUNBOOK（发车手册）
.claude/agents/                 6 个角色提示词（去仓库化）
.claude/workflows/              chapter-pipeline.js（单章工作流）
.claude/skills/                 svg-diagram 等技能
schemas/                        book_outline / chapter_context JSON schema
wisdom/                         跨实例通用模式
scripts/                        实例解析 + linter + bible/archivist/learn + new_instance
instances/<name>/               一本书 = 一个实例
  ├── repo2book.json            实例配置（源仓 URL / 书名 / 读者画像 / 版本）
  ├── INSTANCE.md               实例的当前状态 + 专属硬规则
  ├── source/                   目标仓真实源码（blobless clone，gitignored）
  ├── book/{cartography,bible,assets}   架构地图 / Bible / Roadmap 资产
  ├── knowledge/                仓库特定事实（TTL）
  ├── trace/                    项目长期记忆（archivist 持有）
  └── artifacts/chNN-slug/      每章：dossier/implementation/tests/narrative/diagrams/reviews
```

**实例无关性**：脚本统一经 `scripts/instance.py` 定位「活动实例」（`repo2book.json.active_instance`，或环境变量 `REPO2BOOK_INSTANCE` 覆盖）。linter 的 `--all` 自动扫活动实例，不写死任何仓库。

---

## 脚本一览

| 脚本 | 作用 |
|---|---|
| `new_instance.py` | 新建一本书：scaffold 实例骨架 + blobless clone 源仓 |
| `instance.py` | 解析活动实例的目录/配置/glob（去仓库化的核心） |
| `lint_fidelity / lint_chapter_structure / lint_formulas / lint_source_grounding` | 内容质量闸门 |
| `lint_anchors / lint_punct / lint_diagram_geometry / lint_diagrams` | 锚点 / 半角标点 / 图形几何 / 图有效性 |
| `bible.py` | 跨章连贯性 CLI（伏笔 due/回收、术语、接口注册） |
| `archivist.py` | 长期记忆 / trace / 状态 |
| `learn.py` | 收工后抽取 knowledge / wisdom |
| `remap_lines_v021.py` | 源码升级时把行号引用确定性重映射到新版本（difflib 行级对齐，可复用于任意实例升级） |

---

## 权威文档

- **`CLAUDE.md`** — 通用操作手册（每会话自动加载）。
- **`docs/superpowers/ARCHITECT-RUNBOOK.md`** — 发车 / 监控 / 逃生舱 / 续跑。
- **`instances/<active>/INSTANCE.md`** — 当前在写这本书的源码版本、状态、专属规则。
- **`docs/superpowers/specs/…`** — 方法论与设计依据（以 vLLM 为首例，方法论通用）。
