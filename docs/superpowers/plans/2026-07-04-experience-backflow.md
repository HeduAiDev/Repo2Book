# 经验回流系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec `docs/superpowers/specs/2026-07-04-experience-backflow-design.md`——run-ledger 信号持久化、book-retro 复盘 workflow、curator 落笔角色、经验台账,并退役 wisdom/knowledge/learn.py。

**Architecture:** 经验的家 = 下次必然被读的地方(linter>契约>skill/RUNBOOK>INSTANCE);自动发现(retro)+人把方向(Lead 批准)+agent 落笔(curator)+台账验证生效(复发即升级落点)。

**Tech Stack:** Workflow JS(既有约定)、markdown 台账、无新 Python 依赖(spec §7:不新增 linter,YAGNI)。

## Global Constraints

- workflow 语法检查用 async-wrapper 法(裸 node --check 必失败);全角标点;每 agent 调用显式 model。
- **≥2 章重复才成经验候选;reader 顾问 issue ≥4 章**(spec §3);retro workflow **不做任何修改动作**(spec §3 末)。
- curator **只许 Edit 批准清单内文件;linter 类不直接改,产 SDD 简报;禁动 workflow 编排逻辑**(spec §5 P5)。
- git:定点 add(工作区有章节产出等未提交内容,严禁 -A);绝不 push;commit 尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 旧行为零回归:两个 workflow 的既有 agent 提示词除注入 run-ledger 外不得变(缓存续跑友好:改动集中在 Archive 段与新增变量行)。

## File Structure

```
.claude/workflows/chapter-pipeline.js     # 改:过程计数器 + Archive 写 run-ledger
.claude/workflows/chapter-retrofit.js     # 改:同上(kind=retrofit)
.claude/workflows/book-retro.js           # 新:复盘 workflow(挖掘→聚类→报告)
.claude/agents/curator.md                 # 新:落笔角色契约
docs/superpowers/experience-ledger.md     # 新:经验台账(表头 scaffold)
.claude/agents/{analyst,implementer,tester,explainer,illustrator,writer,reviewer,archivist}.md  # 改:删 learn.py 行
scripts/attic/learn.py                    # 移:退役
knowledge/ + instances/*/knowledge/       # 删(分拣后);.gitignore 移除该行
wisdom/ → docs/attic/wisdom/              # 移(有效条款先出候选清单)
CLAUDE.md / README.md / docs/superpowers/ARCHITECT-RUNBOOK.md  # 文档同步
```

---

### Task 1: chapter-pipeline.js 写 run-ledger

**Files:**
- Modify: `.claude/workflows/chapter-pipeline.js`

**Interfaces:**
- Produces: `{chapter_dir}/reviews/run-ledger.json` schema(Task 2/3 同款):`{chapter_id, kind: "code|primer|meta|retrofit", impl_test_rounds, impl_test_ledger[], write_review_rounds, blind_rounds, blind_failures[{round, failures[]}], escalated: null|{stage,reason}}`。

先 Read 全文件。五处精确 Edit(锚点为当前文件唯一串):

- [ ] **Step 1: 过程计数器。** `let ledger = []` 行(Phase B/C 前)后追加一行:

```js
let implTestRounds = 0
```

impl/test 循环体内(锚:`phase('Implement')` 行之前的 `for (let r = 1; r <= 3; r++) {`,即 `let testV = null` 之后那个循环)第一行加 `implTestRounds = r`。

- [ ] **Step 2: 盲审历史。** `let blindLedger = []` 后追加:

```js
let blindHistory = []
```

`blindLedger = ((blindV && blindV.failures) || []).map(...)` 行(line≈199)之前插入:

```js
  blindHistory.push({ round: b, failures: (blindV && blindV.failures) || [] })
```

注意插入点在 `if (blindV && blindV.all_pass) break` **之前**,让 PASS 轮也记录(failures 为空数组)——即放在 blind agent 调用之后、break 判断之前。

- [ ] **Step 3: 评审轮数。** Review 循环(`const DIMS = [` 之后的 `for (let r = 1; r <= 3; r++) {`)前加 `let reviewRounds = 0`,循环体 `phase('Review')` 行后加 `reviewRounds = r`。

- [ ] **Step 4: Archive 注入。** `const reviewJson = JSON.stringify(...)` 行后插入:

```js
const runLedger = JSON.stringify({
  chapter_id: A.chapter_id,
  kind: PRIMER ? 'primer' : (A.skip_impl ? 'meta' : 'code'),
  impl_test_rounds: implTestRounds, impl_test_ledger: ledger,
  write_review_rounds: reviewRounds,
  blind_rounds: blindHistory.length, blind_failures: blindHistory,
  escalated: null,
})
```

archiveTask 的 `reviewJson + '\n' +` 行后插入:

```js
  '任务一b：把这个 run-ledger 对象**原样**写入 ' + CH + '/reviews/run-ledger.json（经验回流的信号源，不要改写）：\n' + runLedger + '\n' +
```

- [ ] **Step 5: 验证。** async-wrapper node --check(分割点 `// ⚠️ 本环境实测`,scratchpad 代实际路径)→ SYNTAX-OK-WRAPPED;`grep -c 'run-ledger.json\|runLedger' .claude/workflows/chapter-pipeline.js` ≥3;确认既有 agent 提示词串除 archiveTask 外零改动(`git diff` 逐 hunk 核)。

- [ ] **Step 6: 提交。**

```bash
git add .claude/workflows/chapter-pipeline.js
git commit -m "feat(retro): chapter-pipeline 落盘 run-ledger——回环轮数/盲审史成为经验信号(Task 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: chapter-retrofit.js 写 run-ledger

**Files:**
- Modify: `.claude/workflows/chapter-retrofit.js`

同款套路,四处 Edit:

- [ ] **Step 1:** `let blindLedger = []` 后加 `let blindHistory = []`;盲审 agent 调用后、`if (blindV && blindV.all_pass) break` 前插 `blindHistory.push({ round: b, failures: (blindV && blindV.failures) || [] })`。
- [ ] **Step 2:** PatchWrite/Review 循环(`for (let r = 1; r <= 2; r++) {`)前加 `let reviewRounds = 0`,`phase('PatchWrite')` 行后加 `reviewRounds = r`。
- [ ] **Step 3:** Archive 段 `const arch = await agent(` 之前插入:

```js
const runLedger = JSON.stringify({
  chapter_id: A.chapter_id, kind: 'retrofit',
  flagged: diag.flagged_count,
  impl_test_rounds: 0, impl_test_ledger: [],
  write_review_rounds: reviewRounds,
  blind_rounds: blindHistory.length, blind_failures: blindHistory,
  escalated: null,
})
```

archive 提示词 `'任务一：…retrofit-review.json：\n' + JSON.stringify(reviewV) + '\n' +` 之后插入:

```js
  '任务一b：把这个 run-ledger 对象**原样**写入 ' + CH + '/reviews/run-ledger.json：\n' + runLedger + '\n' +
```

- [ ] **Step 4:** async-wrapper 检查(分割点 `const CFG`)→ SYNTAX-OK-WRAPPED;git diff 核既有提示词零漂移。
- [ ] **Step 5:** 提交(消息 `feat(retro): chapter-retrofit 落盘 run-ledger(Task 2)` + 尾注)。

---

### Task 3: book-retro.js 复盘 workflow(新)

**Files:**
- Create: `.claude/workflows/book-retro.js`

**Interfaces:**
- Consumes: `args {instance, chapters:[slug]|null, date}`;各章 `reviews/{review-report,run-ledger,retrofit-review}.json`;`book/audits/*.json`;台账 `docs/superpowers/experience-ledger.md`(Task 5 scaffold)。
- Produces: `instances/<x>/book/retro/retro-<date>.json`(spec §3 schema);返回候选摘要。**只读挖掘,不做修改**。

- [ ] **Step 1: 写完整文件:**

```js
export const meta = {
  name: 'book-retro',
  description: '复盘:从评审 issue/run-ledger/审计报告挖经验候选——≥2 章重复才算,产出含落点建议与 patch 草案的清单,不做任何修改',
  phases: [
    { title: 'Mine', detail: '每章一个挖掘 agent,只读 reviews/+audits' },
    { title: 'Cluster', detail: 'opus 聚类:≥2 章重复,对照台账标复发' },
  ],
}

const CFG = { instance: 'vllm-ascend', chapters: null, date: 'undated', repo_root: '/mnt/e/Laboratory/Repo2Book' }
const A = (typeof args !== 'undefined' && args && args.instance) ? args : CFG
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance
const ARTS = REPO + '/instances/' + INST + '/artifacts'
const BOOK = REPO + '/instances/' + INST + '/book'
const LEDGER = REPO + '/docs/superpowers/experience-ledger.md'
const OUT = BOOK + '/retro/retro-' + (A.date || 'undated') + '.json'

const MINE_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['signals'],
  properties: { signals: { type: 'array', items: { type: 'object', additionalProperties: false,
    required: ['signal', 'evidence', 'candidate_rule', 'source'],
    properties: { signal: { type: 'string' }, evidence: { type: 'string' },
      candidate_rule: { type: 'string' }, source: { type: 'string' } } } } },
}
const CLUSTER_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['candidates', 'stats_note'],
  properties: {
    candidates: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['id', 'pattern', 'occurrences', 'root_cause', 'target', 'draft_patch', 'expected_effect', 'recurrence'],
      properties: { id: { type: 'string' }, pattern: { type: 'string' },
        occurrences: { type: 'array', items: { type: 'object', additionalProperties: false,
          required: ['chapter', 'evidence'],
          properties: { chapter: { type: 'string' }, evidence: { type: 'string' } } } },
        root_cause: { type: 'string' }, target: { type: 'string' },
        draft_patch: { type: 'string' }, expected_effect: { type: 'string' },
        recurrence: { type: 'string' } } } },
    stats_note: { type: 'string' },
  },
}

phase('Mine')
let slugs = A.chapters
if (!slugs || !slugs.length) {
  const LIST_SCHEMA = { type: 'object', additionalProperties: false, required: ['slugs'],
    properties: { slugs: { type: 'array', items: { type: 'string' } } } }
  const ls = await agent('列出目录 ' + ARTS + ' 下所有 chNN-* 子目录名（Bash ls），按章号排序返回 slugs。',
    { schema: LIST_SCHEMA, label: 'list-chapters', phase: 'Mine', model: 'haiku', agentType: 'general-purpose' })
  slugs = (ls && ls.slugs) || []
}
log('挖掘 ' + slugs.length + ' 章的经验信号')

const mined = await parallel(slugs.map(function (slug) {
  return function () {
    return agent(
      '你是经验信号挖掘员。**只读**：' + ARTS + '/' + slug + '/reviews/ 下的 review-report.json、run-ledger.json、retrofit-review.json（存在哪个读哪个）+ ' + BOOK + '/audits/ 里涉及本章的条目（目录可能不存在）。\n' +
      '任务：挑出**指向流程/提示词/门禁缺陷**的信号——不是本章内容问题本身，而是「什么样的问题反复让 agent 犯错」。看：blocking issue 的成因类型、回环 >1 轮的原因、盲审 FAIL 的原因、escalated 记录。\n' +
      '每条返回 {signal(一句话概括问题类型), evidence(引用原文≤60字), candidate_rule(若要根除，一句话规则该写成什么), source(文件名+定位)}。\n' +
      '章级一次性内容错误（如某行号写错）不算信号，返回时跳过。没有信号返回 signals=[]。',
      { schema: MINE_SCHEMA, label: 'mine:' + slug.slice(0, 12), phase: 'Mine', model: 'sonnet', agentType: 'general-purpose' }
    ).then(function (r) { return { slug: slug, signals: (r && r.signals) || [] } })
  }
}))

phase('Cluster')
const all = mined.filter(Boolean)
const payload = JSON.stringify(all)
const clustered = await agent(
  '你是复盘聚类员。输入是各章挖掘出的经验信号（JSON 见文末）。另外 Read 台账 ' + LEDGER + '（已落地经验清单，可能只有表头）。\n' +
  '任务：按 root cause 聚类。**硬门槛：同类信号 ≥2 章出现才成候选；来源为 reader 顾问维度的信号需 ≥4 章**。\n' +
  '每个候选给：id（exp-' + (A.date || 'undated') + '-序号）、pattern、occurrences（逐章+证据）、root_cause、target（五选一：linter:<脚本名> | contract:<角色> | skill:<名> | runbook | instance）、draft_patch（可直接落笔的条款文字或 linter 规则描述，全角标点）、expected_effect（下批次哪个可数指标应下降）、recurrence（台账已有同 pattern 则写"复发:<台账id>,建议升级为<新落点>"，否则写 "new"）。\n' +
  'target 选择原则：可机检的优先 linter，角色行为进 contract，操作方法进 skill/runbook，仓库事实进 instance。\n' +
  'stats_note 写一句总量统计。**你不做任何文件修改**。\n\n' + payload,
  { schema: CLUSTER_SCHEMA, label: 'cluster', phase: 'Cluster', model: 'opus', agentType: 'general-purpose' }
)
const cands = (clustered && clustered.candidates) || []
const report = { date: A.date, instance: INST, chapters_mined: slugs.length,
  stats_note: (clustered && clustered.stats_note) || '', candidates: cands }
await agent('把下面 JSON **原样** Write 到 ' + OUT + '（目录不存在先建），写完返回 "written"。\n' + JSON.stringify(report),
  { label: 'write-report', phase: 'Cluster', model: 'haiku', agentType: 'general-purpose' })
log('复盘完成：' + cands.length + ' 个经验候选 → ' + OUT)
return { report: OUT, candidates: cands.map(function (c) { return { id: c.id, pattern: c.pattern, target: c.target, recurrence: c.recurrence } }) }
```

- [ ] **Step 2:** async-wrapper 检查(分割点 `const CFG`)→ SYNTAX-OK-WRAPPED。
- [ ] **Step 3:** 提交(`feat(retro): book-retro 复盘 workflow——挖掘→聚类→候选清单,只读不改(Task 3)` + 尾注)。

---

### Task 4: curator.md 角色契约 + 经验台账 scaffold

**Files:**
- Create: `.claude/agents/curator.md`
- Create: `docs/superpowers/experience-ledger.md`

- [ ] **Step 1: 写 `.claude/agents/curator.md`**(完整内容):

```markdown
---
name: curator
description: 经验落笔员——把 Lead 批准的经验候选定点写进契约/skill/RUNBOOK/INSTANCE;linter 类不直接改,产 SDD 简报;经验回流的最后一步
tools: Read, Edit, Write, Grep, Glob, Bash
model: inherit
color: teal
---

# Curator — 经验落笔员

你把 **Lead 已批准**的经验候选写进它的家。你是回流的手,不是脑:**不评判、不扩写、不夹带**——
批准清单说什么落什么。

## 输入
Lead 给你的批准清单:每条 {id, target, draft_patch(最终文字), 落点文件}。
落点文件之外的任何文件**禁止修改**;workflow 编排逻辑(.claude/workflows/ 的控制流)禁止触碰。

## 按 target 落笔
- `contract:<role>` / `skill:<name>` / `runbook` / `instance` → Read 目标文件,把 draft_patch
  **定点 Edit** 进最贴切的小节(全角标点,融入原文格式与语气;插入位置在报告里写明行号)。
- `linter:<脚本名>` → **不直接改代码**。产出 SDD 任务简报写入
  docs/superpowers/plans/briefs/lint-<id>.md(含:规则描述、blocking/warn 定级、测试用例草案),
  返回时提醒 Lead 走 TDD 小任务。
- 每条落地后,向 docs/superpowers/experience-ledger.md 追加一行
  `| <id> | <日期> | <pattern> | <落点文件> | <针对指标> | active |`。

## 铁律
- 一次只处理清单内条目;发现 draft_patch 与目标文件已有内容冲突/重复 → 不落笔,回报 Lead 裁决。
- 落笔后逐条回报:文件+行号+插入内容摘要,便于 Lead 抽查 diff。
```

- [ ] **Step 2: 写 `docs/superpowers/experience-ledger.md`**(scaffold):

```markdown
# 经验台账(experience ledger)

> 经验回流系统的生效验证账本:每条 = 一次"发现→批准→落笔"。retro 复盘时对照本表——
> pattern 复发 = 沉淀无效 → 升级落点(契约→linter);连续两次复盘未复发 → 标 proven。
> 详见 docs/superpowers/specs/2026-07-04-experience-backflow-design.md。

| id | 日期 | pattern | 落点(文件) | 针对指标 | 状态 |
|---|---|---|---|---|---|
```

- [ ] **Step 3:** 核 frontmatter 完整(`head -8`);提交(`feat(retro): curator 落笔角色 + 经验台账 scaffold(Task 4)` + 尾注)。

---

### Task 5: 退役 learn.py/knowledge/wisdom(契约清理 + 分拣)

**Files:**
- Modify: `.claude/agents/{analyst,implementer,tester,explainer,illustrator,writer,reviewer}.md`(删 learn.py 行;archivist 无则跳过——先 grep)
- Move: `scripts/learn.py` → `scripts/attic/learn.py`
- Delete: 仓库根 `knowledge/`、`instances/vllm/knowledge/`、`instances/vllm-ascend/knowledge/`;`.gitignore` 移除 `knowledge/` 行
- Move: `wisdom/` → `docs/attic/wisdom/`
- Create: `instances/vllm-ascend/book/retro/wisdom-candidates.json`(wisdom 有效条款 → 首批经验候选,供发车阶段 Lead 批)

本任务有判断成分,执行者须逐步核验:

- [ ] **Step 1: 契约清理。** `grep -n 'learn.py\|wisdom' .claude/agents/*.md`;逐文件 Edit:删除"收工后 `python3 scripts/learn.py extract …`"子句(保留句子其余部分,句子只剩空壳则整句删);删除"读 `wisdom/…`"子句(writer 的 voice-guide 引用**保留**——那是 bible 不是 wisdom)。改后 `grep -c 'learn.py' .claude/agents/*.md` 全 0、`grep -c 'wisdom' .claude/agents/*.md` 全 0。
- [ ] **Step 2: learn.py 入 attic。** `mkdir -p scripts/attic && git mv scripts/learn.py scripts/attic/learn.py`;文件头加一行注释 `# RETIRED 2026-07-04:经验回流系统替代(见 specs/2026-07-04-experience-backflow-design.md);仅存档。`;`grep -rn 'learn.py' scripts/ .claude/ CLAUDE.md docs/superpowers/ARCHITECT-RUNBOOK.md --include='*.py' --include='*.js' --include='*.md' | grep -v attic | grep -v plans/ | grep -v specs/` 应仅剩历史 spec/plan 文档(不改历史文档)。
- [ ] **Step 3: knowledge 分拣。** 读根 `knowledge/modules/*.md`(约 12 文件):凡**仓库事实**(源码行为/版本坑)→ 按实例归并进 `instances/<x>/INSTANCE.md` 的坑段(判断实例:内容涉 vllm_ascend 路径→ascend,否则 vllm);过程性/复述性内容 → 丢弃。分拣表(文件→去向)写进报告。然后 `rm -rf knowledge/ instances/vllm/knowledge/ instances/vllm-ascend/knowledge/`;`.gitignore` 删 `knowledge/` 行;`grep -rn 'knowledge/' scripts/*.py | grep -v attic` 应为 0(instance.py 若有 knowledge 辅助函数,删除该函数并确认无调用方)。
- [ ] **Step 4: wisdom 盘点。** 读 `wisdom/{architecture,debugging,testing,writing}.md`,逐条判断:已被现行契约/skill/RUNBOOK 覆盖 → 弃;已过时(针对 v2 流程)→ 弃;仍有效且未覆盖 → 写入 `instances/vllm-ascend/book/retro/wisdom-candidates.json`,格式同 retro 候选(id: exp-wisdom-<n>,occurrences 填 `[{"chapter":"wisdom 存量","evidence":"<原文≤60字>"}]`,recurrence:"new")。然后 `mkdir -p docs/attic && git mv wisdom docs/attic/wisdom`。
- [ ] **Step 5: 回归。** `python3 -m pytest scripts/tests -q` 全过(67);`python3 scripts/bible.py due ch20` 正常(instance.py 未被破坏);两 workflow wrapped-check 过。
- [ ] **Step 6: 提交**(一次,含 mv/rm):`refactor(retro): 退役 learn.py/knowledge/wisdom——经验回流替代;存量分拣归位(Task 5)` + 尾注。git add 用明确路径:`.claude/agents/ scripts/attic scripts/learn.py .gitignore docs/attic instances/vllm/INSTANCE.md instances/vllm-ascend/INSTANCE.md instances/vllm-ascend/book/retro/ wisdom knowledge instances/vllm/knowledge instances/vllm-ascend/knowledge`(git 对已删目录用 `git add -u <path>`)。

---

### Task 6: 文档同步 + 全量回归

**Files:**
- Modify: `CLAUDE.md`、`README.md`、`docs/superpowers/ARCHITECT-RUNBOOK.md`

- [ ] **Step 1: CLAUDE.md「记忆体系」段整段替换**为:

```markdown
## 记忆体系
- **Archivist**(唯一全书持久角色):trace 长期记忆 + Book Bible + concepts.json。`scripts/archivist.py`。
- **经验回流**(替代已退役的 wisdom/knowledge):每章落盘 `reviews/run-ledger.json`(回环轮数/盲审史);批次收尾跑 `book-retro` workflow 挖经验候选(≥2 章重复才算)→ Lead 批准 → curator 落笔进 linter/契约/skill/RUNBOOK/INSTANCE → 台账 `docs/superpowers/experience-ledger.md` 记录并在下次复盘验证生效(复发即升级落点)。
- **架构师自身连贯性**:本 CLAUDE.md + RUNBOOK + 实例 INSTANCE.md + trace 决策记录。
```

- [ ] **Step 2: README 两处。** 目录结构块删 `wisdom/` 行、`knowledge/` 行(实例树内),`scripts/` 行的 `learn` 改为 `retro 台账见 docs/`(措辞融入);脚本表删 `learn.py` 行;「跨章连贯性」句的"**wisdom/** 收跨实例通用模式"改为"经验回流:复盘 workflow 把反复出现的问题沉淀进 linter/契约/skill(台账验证生效)"。
- [ ] **Step 3: RUNBOOK。** 加小节:

```markdown
## 复盘发车(book-retro,经验回流)

批次/Part 收尾时:`Workflow({name:"book-retro", args:{instance, chapters:[slug]|null, date:"YYYY-MM-DD"}})`
→ 报告在 `instances/<x>/book/retro/`。Lead 逐条批(改落点/措辞/驳回)→ 派 curator(.claude/agents/curator.md)
按批准清单落笔 → 台账 `docs/superpowers/experience-ledger.md` 自动追加。
linter 类候选:curator 产 SDD 简报,Lead 另走 TDD 小任务。
复发判定:retro 对照台账,active 条目 pattern 再现 = 沉淀无效 → 升级落点(契约→linter)。
```

逃生舱表末尾加一行提醒:`处理任何升级时,顺手补写该章 reviews/run-ledger.json 的 escalated 字段(早退章不经过 Archive,信号靠 Lead 补记)。`

- [ ] **Step 4: 回归。** pytest 全过;`grep -rn 'wisdom/\|learn.py' CLAUDE.md README.md docs/superpowers/ARCHITECT-RUNBOOK.md | grep -v attic | grep -v 已退役` 复核无残留引用(历史 spec/plan 不算);三 workflow wrapped-check 过。
- [ ] **Step 5: 提交**(`docs(retro): 记忆体系段改写/README/RUNBOOK 同步——经验回流上线(Task 6)` + 尾注)。

---

## 发车阶段(运营,Task 1-6 完成后)

1. **首跑复盘**:`Workflow({name:"book-retro", args:{instance:"vllm-ascend", chapters:[Part VIII 已完成章 + 回修过的 9 章 slug], date:"<当日>"}})`。**挖掘质量基准**:候选里应能自动再发现已手工沉淀过的模式(图注写法/公式锚/半角标点类)——没挖到则调挖掘提示词再跑。
2. Lead 逐条批(含 wisdom-candidates.json 的存量候选)→ 派 curator 落笔 → 抽查 diff → 台账建档。
3. linter 类候选走 SDD 小任务(TDD)。
4. 下一批次(如 vllm 书回修/新书)收尾再跑 retro,验证台账条目不复发。

## 计划自检记录

- **Spec 覆盖**:§2→Task 1+2(+RUNBOOK 逃生舱提醒在 Task 6);§3→Task 3;§4(curator)→Task 4+发车 2/3;§5(台账)→Task 4 scaffold + curator 契约追加行为 + Task 3 聚类对照;§6 退役→Task 5 + Task 6 文档;§7 验收→Task 各 Step 回归 + 发车 1 基准;§8 风险→门槛/批准制/只读挖掘/一行一条均落于对应任务。
- **占位符**:无 TBD;所有代码/契约/文档块完整。
- **类型一致性**:run-ledger schema 键名 Task 1/2/3(挖掘提示词)一致;候选 schema 键名 Task 3/4(curator 输入)/Task 5(wisdom-candidates)一致;台账列序 Task 4 scaffold 与 curator 追加行一致;分割点(pipeline `// ⚠️`,retrofit/retro `const CFG`)与既有惯例一致。
- **顺序**:Task 1/2 独立;3 依赖 4 的台账路径(仅读,可先行,scaffold 缺失时聚类提示词已写"可能只有表头");4 独立;5 依赖 4 完成后再删 wisdom(候选文件格式引用 Task 3 schema);6 收尾。建议执行序:1→2→4→3→5→6。

