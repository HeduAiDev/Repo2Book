# 架构师 / 编排者操作手册（ARCHITECT RUNBOOK）

> 你（主 session）= **Team Lead / 架构师 / 编排者**。你不是持久 agent，会被上下文压缩。
> **失忆或换会话后：按本手册 + CLAUDE.md 运转工厂，不要靠记忆。**

## 0. 上手顺序（冷启动/压缩后先读）
1. `CLAUDE.md`（自动加载）——通用方法论 + HARD RULES（仓库无关）。
2. 本手册——具体操作。
3. `repo2book.json.active_instance` → `instances/<active>/INSTANCE.md`——当前在写哪本书的源码版本/状态/专属规则。
4. `instances/<active>/book/cartography/ARCHITECTURE.md` + `outline-final.json`——架构地图 + 大纲。
5. `python3 scripts/archivist.py state` + `instances/<active>/book/bible/`——当前状态 + 连贯性。
6. `docs/superpowers/specs/2026-06-21-vllm-source-reading-book-system.md`——设计与为什么（以 vLLM 为首例，方法论通用）。

### 0.5 新建一本书（换实例）
`python3 scripts/new_instance.py <name> --repo <git-url> --title "…" --prefix <规范路径前缀> --activate`
→ scaffold `instances/<name>/` 骨架（含**继承的 voice-guide**，承袭既往实例约定，不退白板）+ blobless clone 源仓 + 置为 active → 在 source/ pin commit、填 INSTANCE.md → **按 §0.6 出架构地图 + 大纲** → 回到 §3 逐章发车。脚本统一经 `scripts/instance.py` 认活动实例，无需改 linter。

### 0.6 出架构地图 + 大纲（cartography playbook —— 别临场拍脑袋）
这步定全书骨架，**经验沉淀在这张清单里**（vLLM 当年的一次性 cartography workflow 已散失，故写成 playbook，别再裸手重走弯路）：
1. **子系统测绘（fan-out）**：按源码顶层目录/子系统分组，每组派一个 analyst 读真实源码（姊妹篇还要对照基座实例 `instances/<base>/source`），产 digest：可成章单元、`key_source_paths`、`pairs_with`、教学价值、该子系统"怎么接入/改写"的主线。
2. **综合（synthesis）**：1 个 agent 汇总成 `outline-final.json`（**遵从 `schemas/book_outline.json` v2**：`book` + `parts[]` + `chapters[]`，每章 `chapter_id/slug/title/focus/part/key_source_paths/pairs_with/deps/est_size/mode`）+ `ARCHITECTURE.md`（心智模型 + 子系统地形 + 逐 Part 大纲 + 配对脊柱）。
3. **⚠️ 强制：子系统覆盖交叉核对**（最易漏，vLLM-ascend 试点连栽四次：PD 分离 / 池化 / kv_offload / 310P 都是用户事后揪出的）：列源码**每个顶层子系统**，逐一确认"被某章 `key_source_paths` 覆盖 / 或显式点名入横切"，**未覆盖即漏章**。死盯易被低估的：PD 分离（proxy 调度 + **KV 亲和/命中路由**）、KV 池化/外存储、KV 卸载（host/CPU 分层）、芯片/硬件分代变体（如 310P，常是整套子类化）、网络加载——这些常被压成一章或漏掉。
4. **路径核对**：每个 `key_source_paths`（及 `pairs_with` 的基座路径）在 source/ 真实存在。
5. **配对脊柱**（姊妹篇）：每章钉一个对位基座章，正文对照基座说"顶替/扩展了哪一站"。
6. **用户审批闸**：把 Part/章列表 + 覆盖核对结论给用户，**批准后**才逐章发车——别跳。

## 1. 心智模型（一句话）
真实源码是教材；analyst 把它读成 **dossier（唯一真相源）**；implementer 据此**只删不增**做可运行精简版；writer 以**真实源码为主线**写自包含章节；reviewer 协作式把关；archivist 持久化记忆。编排靠 **chapter-pipeline workflow**（并行+确定性+逃生舱），活体迭代靠我 + 命名 agent + SendMessage。

## 2. 目录地图
```
.claude/agents/{analyst,implementer,tester,writer,reviewer,archivist}.md  ← 6 角色持久提示词
.claude/workflows/chapter-pipeline.js                                     ← 单章流水线
scripts/lint_fidelity.py  lint_chapter_structure.py  lint_formulas.py  lint_source_grounding.py
scripts/instance.py       ← 活动实例解析（去仓库化核心；linter --all 据它扫）
scripts/new_instance.py   ← 新建一本书（scaffold 实例 + 克隆源仓）
scripts/bible.py          ← 跨章连贯性 CLI（due/foreshadow/payoff/term/iface）
scripts/archivist.py learn.py   ← 长期记忆 / 自学习
scripts/remap_lines_v021.py     ← 源码升级时行号确定性重映射（可复用于任意实例）
instances/<active>/repo2book.json + INSTANCE.md         ← 实例配置 + 当前状态/专属规则
instances/<active>/source/                              ← 目标仓真实源码（blobless clone）
instances/<active>/book/{cartography,bible,assets}/     ← 架构地图 + 大纲 / Book Bible / Roadmap 母版
instances/<active>/artifacts/chNN-slug/                 ← 每章产物（ch- 前缀 slug）
instances/<active>/{knowledge,trace}/                   ← 仓库事实(TTL) / 项目长期记忆
docs/superpowers/{specs,plans}/                         ← 设计 + 计划
（当前 active = vllm；其源码 @ v0.21.0，调试进容器 scripts/vllm_docker.sh，详见 instances/vllm/INSTANCE.md）
```

## 3. 发车：跑一章
```
Workflow({ name: "chapter-pipeline", args: {
  chapter_id: "ch04",
  slug:       "ch04-async-llm",
  focus:      "AsyncLLM 三段式异步解耦",
  highlight:  "async-engine",                 // Roadmap 高亮键（见 roadmap.py STAGES）
  source_root:"/mnt/e/Laboratory/Repo2Book/instances/vllm/source",
  paths:      ["vllm/v1/engine/async_llm.py","vllm/v1/engine/__init__.py"]
}})
```
- 后台跑；完成或逃生舱触发会 task-notification 通知我。
- args 来源（精确字段）：`outline-final.json` 本章的 `subsystem` 把章映射到 `cartography/map.json` 的对应子系统条目；`paths` = 该条目的 `key_files[].path`；`focus` = 本章 `scope`/`title`；`highlight` = 该 `subsystem`（即 `roadmap.py` STAGES 的键）。
- 章节目录不存在时 workflow 内 agent 会按绝对路径 Write 创建（dossier/implementation/tests/narrative/reviews/diagrams）。

### 风险高/首跑：分段发车
先只跑到 dossier 审一眼再放行：把 workflow 临时改成 dossier 后 `return`，或直接让 analyst 角色单独产 dossier，我审"路线/减法计划/must_keep"对不对，再跑完整 pipeline。

## 4. 监控
- `/workflows` 看实时阶段进度。
- `TaskOutput`/读 `/tmp/.../tasks/<id>.output` 看结果。
- 跑偏了：`TaskStop` 急停。

## 5. 逃生舱：处理 BLOCKED / 升级
任一阶段 agent 返回 `status="BLOCKED"` → workflow **早停**（不跑到底）、返回 `{escalated:<stage>, ...}` 并通知我。**共 6 个 stage：**
| escalated | 含义 | 返回字段 | 我的动作 |
|---|---|---|---|
| `dossier` | analyst 产档案时源码与计划不符/无法忠实产出 | `reason` | 修 dossier 输入或 analyst 提示词 → 续跑 |
| `dossier-verify` | 对抗性自核判定档案不可放行 | `problems`（数组） | 按 problems 修 dossier → 续跑 |
| `implement` | 减法计划会破坏正确性/缺料 | `reason` | 修 dossier.subtraction_plan 或 implementer 提示词 → 续跑 |
| `write` | writer 缺要讲清的细节 | `reason` | 让 implementer 补 must_keep；或命名 agent+SendMessage 活体迭代 → 续跑 |
| `review-revise` | 评审回环中 writer 再次 BLOCKED | `reason` | 同上 → 续跑 |
| `review-exhausted` | 评审 3 轮仍有 blocking（兑现">3 轮升级"） | `issues`（数组） | 我介入：修提示词/dossier，或命名 agent+SendMessage 活体迭代 → 续跑 |
- **agent 不能自己联系我或杀 workflow**，只能返回 BLOCKED 拉闸；我也可随时 `TaskStop`。
- **续跑**：`Workflow({scriptPath:".claude/workflows/chapter-pipeline.js", resumeFromRunId:"<上次 runId>"})`，已完成阶段命中缓存。
- **Workflow 入参**：发车可用 `name:"chapter-pipeline"`（已注册）或 `scriptPath`；**续跑必须 `scriptPath` + `resumeFromRunId`**。
- 同一问题 >3 轮自动升级到我；必要时拉用户。

## 6. 质量闸门（手动复核）
```
D=instances/vllm/artifacts/ch04-async-llm
python3 scripts/lint_fidelity.py $D
python3 scripts/lint_chapter_structure.py $D/narrative/chapter.md
python3 scripts/lint_formulas.py $D/narrative/chapter.md
python3 scripts/lint_source_grounding.py $D
jq -r '.overall_verdict' $D/reviews/review-report.json
```
全部无 BLOCKING + verdict=APPROVED 才算过。

## 7. 跨章连贯性 + 读者视角理解检查（每章/每 Part/全书）
- 写前：`python3 scripts/bible.py due {chapter_id}`（应埋/应回收）。
- 写后：archivist 回写 bible（新接口/已埋/已回收）。
- **每章（自动）**：chapter-pipeline 的 Review 阶段含一维 **Haiku 读者视角理解检查**（小模型当"没读过源码的读者"，book-only、不上网，扫局部读不懂处：术语首现未释/逻辑跳跃/引入未建立的概念/只有结论无例子）——顾问性、不门控，issue 进 review-report 由 writer 顺手清。
- 每完成一个 Part：跑连贯性审计（未回收伏笔/术语漂移/接口不符）+ **journey 理解审计**：派 Haiku 读者**按顺序连读该 Part 全部章节**（book-only、不上网），找只有顺读才暴露的问题——前向引用缺口、术语在更晚章才定义、节奏/铺垫断层。
- **全书完成后**：再跑一次全书级 Haiku 连读理解审计（vLLM 书即如此做），把"一个读者能否只靠本书从头学懂"作为终验；发现的卡点派 writer 定点补（改正文不改结论）。
- 何以分层：局部可读性是每章属性（inline 早抓便宜修）；journey 理解是跨章属性（必须顺读才暴露，放 Part/全书边界）。只放全书=攒满一书的困惑才发现；只放每章=漏跨章缺口。

## 8. 架构师的持续职责（用户明确要求"在工程中迭代"）
- 试点/每章复盘 → **改提示词不改章节**（HARD RULE）。fidelity 阈值不合适 → 改 `scripts/lint_fidelity.py` 常量 + 测试。
- 重大决策/转向 → `python3 scripts/archivist.py record --type decision ...` 存进 trace，并更新本手册 + CLAUDE.md。
- superpowers skills 落点见 spec §8（brainstorming/writing-plans/TDD/verification/receiving-code-review/...）。

## 9.5 全书批量循环（goal: 完成全书编写）

被压缩/换会话后，照此续跑批量，**不靠记忆**：

1. **进度真相** = `ls instances/vllm/artifacts/`（有目录=已写）。队列与参数 = `instances/vllm/book/cartography/chapter-queue.json`（每章 slug/focus/highlight/paths/mode/deps）。
2. **选下一章**：chapter-queue 里 mode=code 且无 artifacts 目录的、依赖已满足的最前一章（数字序）。**ch01/ch02 是 mode=meta（概览，无精简版），留到所有 code 章之后，用定制轻量流写**。
3. **发车**：把 `.claude/workflows/chapter-pipeline.js` 顶部的 `CFG` 改成该章参数（本机 args 注入不可靠，靠 CFG），`node --check` 后 `Workflow({scriptPath, args:{同 CFG}})`。
3b. **挂看门狗（必做，别盲等）**：workflow **崩溃是静默的**——只等完成通知会永远等不到。发车后立刻 `Bash(run_in_background)` 一个 for-loop：每 60s 检 `{chapter_dir}/reviews/review-report.json`，出现即报"完成"、逾期(~70min)报"逾期可能崩溃"。崩了就 `TaskStop {taskId}` 再 `Workflow({scriptPath, resumeFromRunId})`（缓存命中已完成阶段，从崩溃点重跑）。判活/判崩：resume 报 "still running" = 活着（别 stop）；"started 无 result" 只是进行中，不等于崩溃。
4. **验收**（流水线完成后，逐条亲跑）：5 linter（fidelity/chapter_structure/formulas/source_grounding/diagrams）全过 + pytest 过 + 脱节体检（叙事引真 vllm/ ≫ 引精简版 implementation/）+ **亲眼看 1 张图确认中文渲染**（lint 查不出 rsvg 与否）+ review verdict=APPROVED + 无 negotiable=false 未修项。
5. **提交**（事故教训：通过即提交）：`git add` 该章 artifacts + bible + trace，commit（带 Co-Authored-By）。
6. **回到 2**，直到 ch01-ch33 全 done；其间**每完成一个 Part** 跑一次连贯性审计 + 批量润色（读各章 review-report.json 的 negotiable 项，派 writer 批量定点修）。
- 串行（整章级，避免 bible 竞争）；逃生舱触发则按 §5 处理后续跑。
- 进度（2026-06-25）：✅ **全书 ch01–ch33 草稿全部完成**（全 APPROVED + 推远程，26/26 伏笔回收，0 断锚）。循环已跑完；剩余为全书润色（清各章 negotiable）。详见 CLAUDE.md「当前状态」。
  - 实战经验补：① 会话用量上限约每 6 章触发一次→escape hatch 防假通过 + 限额重置后 `resumeFromRunId` 续跑；② git push 必须前台（后台 shell SSH 鉴权失败）；③ 监控在 review-report.json 出现即报 DONE，但 archive 的 bible/trace 回写稍后→提交前确认 bible interfaces 有本章；④ meta/概览章用 CFG `skip_impl:true` 走轻流程（无精简版、不跑 fidelity）；⑤ off-spine 章 highlight 用子系统键（roadmap 自动高亮父阶段+「本章深入」框）。

## 9. 当前状态 & 下一步
- 系统重建完成（地基 12/12 测试、6 角色、Roadmap、Bible、workflow+逃生舱、架构师文档）。**冷启动 Team Lead 文档考 v2 已 PASS 60/60**。
- 首跑前 `instances/vllm/artifacts/` 不存在属正常——workflow 内 agent 会按绝对路径自建章节目录；`state.json` 已 bootstrap。
- **下一步：发 ch04 试点**（§3），复盘是否根除脱节 → 迭代提示词 → 再推进 outline 其余章节。
- 未做（后续）：continuity-audit workflow、批量并行（worktree 隔离）、旧 artifacts 实际清理、repo2book.json pipeline 接线。
