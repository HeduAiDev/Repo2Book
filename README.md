# repo2book

> 将任意代码仓库转化为源码级深度技术书籍的多智能体写作系统。

当前实例：**vLLM 从零到专家**（28章，5部分，已发布13章）。

---

## 快速开始

```bash
# 1. 一次性初始化（克隆仓库后执行）
python3 scripts/setup.py

# 2. 启动 Claude Code（tmux 内），主 session 即 Team Lead
#    主 session 有 Agent 工具，可 spawn 全部子 agent

# 3. 对 Team Lead（主 session）说："开始写第5章"
#    Team Lead 会: spawn agents → 创建 tasks → 分发 → 监控进度
```

> **关键架构：** 主 session（你正在对话的 AI）即 Team Lead。只有主 session 有 Agent 工具可以 spawn/kill agent。子 agent 负责执行具体任务，通过 SendMessage 横向通信，通过 hook 自动交接。

---

## 核心理念

### 源码根基 + 理论深度

每一章必须同时满足两个要求：
- **源码溯源**：每个概念必须追溯到具体源码文件和行号（如 `vllm/v1/core/sched/scheduler.py:L352`）
- **理论推导**：从第一性原理出发，包含形式化证明、数值追踪、直觉引导

只堆源码引用 → "源码 dump"（读者学不到原理）。只讲理论 → "教科书复读"（不是关于这本源码的书）。

### 持久化 agent — 项目的身份根基

**每个 agent 只启动一次，在整个项目生命周期中持续运行。** 每个 agent 在自己的 tmux 窗口中保持连续会话。写完第 1 章后进入空闲状态（不会终止），第 2 章开始时同一个 agent 醒来继续工作，带着所有累积的上下文。

```
tmux 会话: repo2book
├── 窗口0 (紫色): book-editor   ← 总调度，与用户交互
├── 窗口1 (蓝色): implementer   ← 持续存在，第1章→第2章→...→第28章
├── 窗口2 (黄色): tester        ← 持续存在，积累测试模式
├── 窗口3 (绿色): writer        ← 持续存在，保持叙事一致性
├── 窗口4 (红色): reviewer      ← 持续存在，积累审查经验
└── 窗口5 (青色): archivist     ← 持续存在，项目永久记忆
```

**为什么这很重要**：
- 第14章的 writer 记得第3章怎么解释 FlashAttention 的——叙事风格、公式约定、图表风格保持一致
- 第6章的 tester 记得在第4章发现的前置条件边界情况
- reviewer 的 auto-REJECT 触发条件随着章节积累越来越精准
- archivist 的追踪记录是连续的，看到的是整个项目的时间线，不是碎片化的章节快照

---

## 目录结构

```
repo2book/
├── README.md                           ← 你在这里
├── CLAUDE.md                           ← 完整框架文档
├── repo2book.json                      ← 主配置（流水线、团队、拓扑、质量规则）
├── .gitmodules                         ← 源码仓库 submodule 追踪
│
├── .claude/                            ← 项目级 Claude Code 配置
│   ├── agents/                         ← 6个 agent 定义（自动发现，完整提示词）
│   │   ├── book-editor.md              # 意图解析、拓扑决策、团队调度
│   │   ├── implementer.md              # 源码分析→重写实现
│   │   ├── tester.md                   # 反压质量闸门
│   │   ├── writer.md                   # 平衡法则、五步节奏、公式规范
│   │   ├── reviewer.md                 # 九维审查、auto-REJECT 触发
│   │   └── archivist.md                # 追踪系统、会话备份、上下文恢复
│   ├── teams/book-factory.json         # 团队配置（setup.py 部署到全局）
│   ├── settings.local.json             # Agent Teams 开关 + 流水线钩子
│   └── skills/svg-diagram/             # Python→SVG→xmllint→PNG 图表流水线
│
├── instances/vllm/                     ← vLLM 书籍实例（完整、自包含）
│   ├── repo2book.json                  # 实例配置
│   ├── artifacts/                      # 全29章内容（实现、测试、叙事、审查）
│   ├── book/book-outline.json           # 28章5部分大纲
│   ├── source/                         # Git submodule: vllm-project/vllm（锁定提交）
│   ├── knowledge/                      # 仓库专属知识（TTL过期、按模块组织）
│   │   ├── INDEX.md                    # 模块→章节索引
│   │   └── modules/                    # scheduler、attention 等模块知识点
│   └── trace/                          # 长期项目记忆（archivist 维护）
│       ├── INDEX.md                    # 全部事件索引
│       ├── state.json                  # 项目状态唯一真相来源
│       ├── chapters/                   # 按章节组织的追踪（决策、交付、会话备份）
│       ├── cross-chapter/              # 跨章节决策、用户偏好、大纲变更日志
│       └── sessions/                   # 会话总结（跨会话连续性）
│
├── wisdom/                             ← 通用智慧（所有 repo2book 实例共享）
│   ├── INDEX.md                        # 模式目录，含角色关联
│   ├── debugging.md                    # F.linear 张量形状、CUDA 不匹配、SVG 裁剪
│   ├── testing.md                      # 驱逐测试设计、OOM 路径、Docker 命令
│   ├── writing.md                      # 公式规范、代码走读、大白话谱系
│   └── architecture.md                 # 反压闸门、横向通信、拓扑模式
│
├── scripts/                            ← CLI 工具链（8个脚本）
│   ├── setup.py                        # 克隆后一次性初始化
│   ├── team_orchestrator.py            # 创建任务、启动 agent、监控流水线
│   ├── hook_pipeline.py                # TaskCompleted → 下一 agent 收件箱通知
│   ├── archivist.py                    # 追踪记录、上下文恢复、会话备份、大纲差异
│   ├── learn.py                        # 知识提取、分类、压缩
│   ├── decide.py                       # 投票、讨论、升级、拓扑提案
│   ├── lint_formulas.py                # LaTeX 公式校验
│   ├── lint_source_grounding.py        # 源码引用覆盖率校验
│   └── guard_narrative.py              # 防止直接编辑叙事文件的锁系统
│
└── schemas/                            ← 新实例的 JSON Schema 模板
    ├── book_outline.json               # 书籍大纲结构
    └── chapter_context.json            # 每章状态结构
```

---

## 流水线架构

### 五阶段流水线 + 一个常驻角色

```
implementer → tester → writer → reviewer → archivist → 发布
     │           │         │           │           │
     │      反压闸门        │      反压闸门      终端阶段
     │                      │                      │
     └── 横向通信 (Writer→Implementer) ────────────┘
     └── 横向通信 (Reviewer→Writer) ───────────────┘
```

| 阶段 | Agent | 职责 | 闸门 |
|------|-------|------|------|
| 1 | **implementer** | 阅读目标仓库源码，写出带 `# REFERENCE` 注释的重实现 | `implementation_exists` |
| 2 | **tester** | 编写并运行 pytest。任何测试失败→流水线停止（反压） | `tests_pass` |
| 3 | **writer** | 阅读实现，写教育性叙事：代码走读 + 数学证明 + 数值追踪 + 源码差异 | `narrative_complete` |
| 4 | **reviewer** | 零基础读者视角九维审查。运行公式和源码覆盖 lint | `review_approved` |
| 5 | **archivist** | 备份全部 agent 会话记录、创建交付记录、更新项目状态。终端阶段 | `delivery_recorded` |
| — | **book-editor** | 常驻调度者。与用户交互，调度流水线，决定拓扑，追踪状态 |

### 横向通信（不通过 Leader）

- **reviewer → writer**：REVISE 指令直接发送，无需 Leader 中转
- **writer → implementer**：代码修改请求直接发送
- **循环检测**：同一问题 > 3 轮 → 升级到 Leader

### 章间自动传递（batch 模式）

```
第 N 章 archivist 完成
  → hook 触发
  → book-editor 收件箱收到："第 N 章已发布"
  → book-editor 检查大纲，找到第 N+1 章
  → 创建 N+1 章任务
  → SendMessage 给 implementer（同一个 implementer！）
  → 流水线级联
  → 重复直到全部完成或被中断
```

---

## 两种工作模式

### 模式一：规划→执行（推荐新书）

```bash
# 研究遍历——分析全部28章
python3 scripts/team_orchestrator.py plan

# 输出：
#   - 复杂度估算（medium/complex）
#   - 推荐拓扑（linear/pair/panel/writer_editor）
#   - 依赖警告
#   - 计划保存到 trace/cross-chapter/plan.json

# 与用户一起审查计划，调整拓扑，确认顺序，然后一章一章推进：
python3 scripts/team_orchestrator.py pipeline 14-triton-primer
# 用户审查输出，给反馈，然后：
python3 scripts/team_orchestrator.py pipeline 15-llama-model-architecture
```

### 模式二：全自动（成熟流水线）

```bash
# 从当前位置自动写完所有剩余章节
python3 scripts/team_orchestrator.py batch

# book-editor 自动处理一切：
#   → 创建任务 → 派发 implementer → 流水线级联 → 发布
#   → 检查下一章 → 重复 → 直到全部完成
#
# 仅在以下情况中断：
#   - 测试失败（反压闸门）
#   - >3轮审查循环（循环检测）
#   - 需要变更拓扑
#   - 用户主动查询
```

---

## Agent 记忆系统

Agent 通过经验自我提升。同一错误不被发现两次，项目知识不断累积。

### 双层记忆

| | 知识 (Knowledge) | 智慧 (Wisdom) |
|---|---|---|
| **范围** | 当前仓库专属事实 | 跨项目通用模式 |
| **位置** | `instances/{repo}/knowledge/` | `wisdom/` |
| **增长** | 线性（每次遇到一个知识点） | 对数级（必须跨项目重复出现） |
| **生命周期** | 30天 TTL，自动归档 | 永久，持续精炼 |
| **防膨胀** | 每模块最多15条，自动压缩 | 严格门槛（需2+仓库确认） |

### Agent 工作协议

每个 agent 在每次任务前后必须执行：

```
任务前（BEFORE WORK）：                    任务后（AFTER WORK）：
  archivist.py brief（上下文恢复）           learn.py extract（保存经验）
  learn.py query（相关知识查询）              archivist.py record（记录事件）
  wisdom/ 阅读（通用模式）                   lint 检查（公式/源码覆盖）
```

角色特定的查询优先级：
- implementer: debugging > architecture > testing > writing
- tester: testing > debugging > architecture > writing
- writer: writing > architecture > debugging > testing
- reviewer: writing > architecture > testing > debugging
- book-editor: architecture > 全部

---

## 组织拓扑

并非所有章节难度相同。流水线支持5种拓扑模式：

| 模式 | 人员配置 | 适用场景 |
|------|---------|---------|
| **linear** | 每阶段1人（默认） | 标准章节，概念成熟 |
| **pair** | 2个 implementer | 复杂算法（如 FlashAttention 级别） |
| **panel** | 2个 reviewer | 新部分的第一章（为后续定模式） |
| **writer_editor** | writer + 风格编辑 | 公式密集、风格关键的章节 |
| **swarm** | 所有阶段2+人 | 全新概念，书中没有前例 |

```bash
# 为复杂章节切换拓扑（需绝对多数投票通过）
python3 scripts/decide.py propose-topology 14-triton-primer pair "Triton 语言陌生，需要交叉检查"
```

---

## Archivist — 长期记忆系统

Archivist 解决长期 AI 项目的根本问题：**上下文丢失**。Claude Code 会话在~50轮后压缩上下文。数周后，agent 会忘记当初为什么做了某个决定。Archivist 是项目的永久记忆。

### 职责
- **记录**每个重要事件（决策、交付、用户反馈、bug）
- **备份**原始 agent 会话记录（完整可重现性）
- **恢复** agent 上下文（工作前提供简报）
- **检测**大纲变更并与 book-editor 通信
- **警报**上下文丢失风险（>50轮、>24小时会话）

### 终端流水线阶段
Archivist 是**第五流水线阶段**——reviewer 批准后，archivist 备份全部 agent 会话记录、创建交付记录、更新项目状态。只有完成这些，章节才算**已发布**。

---

## 团队工作流

### 架构原则

**主 session（你正在对话的 AI）即 Team Lead。** 只有主 session 有 Agent 工具。子 agent 没有 Agent 工具——它们负责执行具体任务。

### 启动

```bash
python3 scripts/setup.py    # 一次性初始化
```

然后主 session 用 Agent 工具 spawn 全部 6 个 agent（后台运行）。

### 日常交互

```
用户: "写第5章"
  → 主 session (Team Lead) 通知 book-editor
  → book-editor 创建 tasks，SendMessage 给 implementer
  → Pipeline hook 自动级联
  → 主 session 报告: "第5章完成 ✓"

用户: "全书进度"
  → 主 session 读 state.json → 汇总报告
```

### Agent 生命周期

```
主 session (Team Lead) spawn → agent 后台运行
  → 接收任务 → 完成 → 空闲 → 下一任务 → 完成 → ...
  → agent 挂了 → 主 session 重新 spawn
  → 全书完成 → 主 session shutdown 所有 agent

---

## 创建新书实例

```bash
# 1. 克隆 repo2book 作为模板
git clone <repo2book> my-book

# 2. 将目标仓库克隆为 submodule
cd my-book/instances/my-book
git submodule add https://github.com/target/project.git source

# 3. 配置
#   编辑 repo2book.json → 更新 source、book、instances 部分
#   编写 book/book-outline.json
#   编写 instances/my-book/knowledge/（知识库）

# 4. 初始化
python3 scripts/setup.py
python3 scripts/archivist.py init --instance my-book

# 5. 规划全书
python3 scripts/team_orchestrator.py plan

# 6. 启动团队并开始写作
python3 scripts/team_orchestrator.py start-team
python3 scripts/team_orchestrator.py pipeline 01-first-chapter
```

---

## 质量规则（不可商榷）

### 公式规则（reviewer auto-REJECT）
- `\text{}` → `\mathrm{}`
- `\boxed{}` → markdown 粗体标题
- `\tag*{}` → 标注放在 `$$` 块外
- `\frac` 在行内 `$...$` 中 → 提升为独立块
- `$$` 与公式内容同在一行 → 分两行

### 源码溯源（reviewer auto-REJECT）
- Cell 2-7 每节至少1个源码文件:行号引用
- 每个实现函数必须有 `# REFERENCE:` 注释
- 源码映射表至少5行
- 3段以上无源码引用 → auto-REJECT

---

## 脚本速查

| 脚本 | 用途 |
|------|------|
| `setup.py` | 一次性项目初始化 |
| `team_orchestrator.py` | 创建流水线任务、启动 agent、监控进度 |
| `hook_pipeline.py` | 阶段间自动传递（PostToolUse hook 触发） |
| `archivist.py` | 追踪管理、会话备份、大纲差异、上下文恢复 |
| `learn.py` | 知识提取、分类、防膨胀压缩 |
| `decide.py` | 投票、讨论、升级、拓扑提案 |
| `lint_formulas.py` | LaTeX 公式校验 |
| `lint_source_grounding.py` | 源码引用覆盖率校验 |
| `guard_narrative.py` | 防止调度者直接编辑叙事文件 |

---

## 当前实例：vLLM 从零到专家

- **源码**: [vllm-project/vllm](https://github.com/vllm-project/vllm) @ `f3fef1235`
- **章节**: 28章，5部分
- **已发布**: 13章（第1-2部分完成）
- **剩余**: 15章（第3-5部分：Triton 算子、Prefill-Decode 分离、模型专属优化）
- **读者水平**: 进阶（HPC 学生/运维，有 Python + PyTorch + CUDA/Triton 基础）

完整框架文档见 `CLAUDE.md`。

---

> *repo2book — 让任意代码仓库成为一本不仅能读懂，还能跟着走源码的书。*
