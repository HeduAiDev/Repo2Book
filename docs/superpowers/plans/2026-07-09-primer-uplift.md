# primer 降台阶体系 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 9 章 primer 原理章降认知台阶——符号速查+首现解释、论文精髓图忠实重绘、子论文先修分级、reader 台阶四问硬门禁;试点 ch21-primer-mla → 用户验收 → 8 章铺开;四机制固化进 pipeline/契约/lint。

**Architecture:** 两组 lint 新检查(符号覆盖 warn+key_figures↔重绘图注 blocking)先行;契约五处入约;chapter-pipeline reader 维对 primer 转 blocking;新 primer-uplift workflow 两段式(Phase A 诊断→用户批→Phase B 施工)。Spec: `docs/superpowers/specs/2026-07-09-primer-uplift-design.md`。

**Tech Stack:** Python linter+pytest;JS workflow;既有 svg-diagram skill/figure-manifest 盲审/# PAPER 锚体系。

## Global Constraints

- primer 判定=dossier 顶层 `kind:"primer"`(lint_paper_grounding 已用);码章一律不受本计划影响。
- 重绘图图注固定句式:「重绘自 arXiv:xxxx Fig.N:<一句话>」或降级「按 arXiv:xxxx §y 描述重绘」;产物 `diagrams/paper-fig-*.{py,svg,png}`+figure-manifest 登记。
- 画布预算沿用 lint_chapter_map 同款(宽 ≤1500、比例 ≤2.6:1——重绘图不受 chapter-map lint 管,但 illustrator 契约同预算);几何 lint 必过。
- writer 施工**禁整章重写**,只许定点 Edit;HARD RULE:主编排者不写 chapter.md。
- workflow 语法检查 async-wrapper 法;提交带 Co-Authored-By 尾注;不 push。
- 用户批准门 ×2:Phase A 诊断后(先修分级+key_figures 清单)、试点 ch21 完工后(降台阶效果验收)。

---

### Task 1: lint_paper_grounding 两组新检查(TDD)

**Files:**
- Modify: `scripts/lint_paper_grounding.py`
- Test: `scripts/tests/test_lint_paper_grounding.py`(既有文件,先读其 fixture 风格)

**Interfaces:**
- Produces: ①`symbol_context`(**warn**,不计退出码):primer 章 `$$` 块符号候选(独立单字母、`\alpha` 类希腊命令、`\mathrm{word}`、下标基名如 `q_t`→`q`;白名单豁免 max/min/exp/log/softmax/argmax/mathrm 内已豁免词与纯数字)须满足:首现公式块 ±3 个非空 prose 行内被提及(`$sym$`/反引号/裸字均可),或出现在任一 markdown 表格行。②`key_figure_missing`(**blocking**):论文包 `meta.json` 的 `key_figures[]` 每条须有章内图注含「重绘自 …Fig.<N>」或「按 …描述重绘」的对应(按 Fig 号匹配);反向:章内「重绘自」图注的 Fig 号须在 key_figures 中登记。meta 无 key_figures 字段 → warn(包策展缺口)。码章(无 kind:primer)两检查均空跑。

- [ ] **Step 1: failing tests**(适配既有 fixture helper;语义如下)

```python
def test_symbol_without_context_warns(tmp_path):
    # $$ 块引入 \delta,前后 3 行 prose 无 δ/delta 提及、无表格行 → warn 1 条
    ch = _mk_primer(tmp_path, body="推导如下。\n\n$$\n\\delta = j - t\n$$\n\n继续。\n")
    res = run_lint(ch)
    assert any("delta" in w or "δ" in w for w in res["symbol_context"])

def test_symbol_with_nearby_prose_ok(tmp_path):
    ch = _mk_primer(tmp_path, body="其中 $\\delta$ 是相对位置偏移。\n\n$$\n\\delta = j - t\n$$\n")
    assert run_lint(ch)["symbol_context"] == []

def test_symbol_in_table_ok(tmp_path):
    ch = _mk_primer(tmp_path, body="| 符号 | 含义 |\n|---|---|\n| $\\delta$ | 相对偏移 |\n\n$$\n\\delta = j - t\n$$\n")
    assert run_lint(ch)["symbol_context"] == []

def test_key_figure_registered_and_redrawn_ok(tmp_path):
    ch = _mk_primer(tmp_path, meta_key_figures=[{"fig": "Fig.2", "arxiv": "arXiv:2405.04434", "shows": "x", "why_essential": "y", "target_section": "z"}],
                    body="![重绘自 arXiv:2405.04434 Fig.2:MLA 压缩路径](../diagrams/paper-fig-2.png)\n")
    assert run_lint(ch)["key_figure_missing"] == []

def test_key_figure_not_redrawn_fail(tmp_path):
    ch = _mk_primer(tmp_path, meta_key_figures=[{"fig": "Fig.2", "arxiv": "a", "shows": "x", "why_essential": "y", "target_section": "z"}], body="没图。\n")
    assert len(run_lint(ch)["key_figure_missing"]) == 1

def test_orphan_redraw_caption_fail(tmp_path):
    ch = _mk_primer(tmp_path, meta_key_figures=[], body="![重绘自 arXiv:x Fig.9:孤儿](../diagrams/paper-fig-9.png)\n")
    assert len(run_lint(ch)["key_figure_missing"]) == 1

def test_code_chapter_skipped(tmp_path):
    ch = _mk_code_chapter(tmp_path, body="$$\n\\delta=1\n$$\n")
    res = run_lint(ch)
    assert res["symbol_context"] == [] and res["key_figure_missing"] == []
```

- [ ] **Step 2: 确认红**:`python3 -m pytest scripts/tests/test_lint_paper_grounding.py -q` → 新增 7 FAIL
- [ ] **Step 3: 实现**(读现文件结构:kind 判定/pack 定位已有;符号提取正则 `\\[a-zA-Z]+` 中希腊命令集+`(?<![A-Za-z])[A-Za-z](?:_[A-Za-z0-9{\\]+)?(?![A-Za-z])`+`\\mathrm\{(\w+)\}`;warn 通道仿 lint_anchors 不计退出码,blocking 进退出码)
- [ ] **Step 4: 全绿**;对 9 章真实 primer 各跑一次,报告 warn/blocking 基线计数(此时 key_figures 均未登记,应全为 meta 缺字段 warn,无 blocking)
- [ ] **Step 5: Commit** `feat(lint): paper_grounding 增符号上下文 warn+key_figures↔重绘图注对应门禁`

### Task 2: 契约五处入约

**Files:**
- Modify: `.claude/agents/analyst.md`(primer 建包节:key_figures 登记职责)
- Modify: `.claude/agents/explainer.md`(primer 节:symbol_table 产出)
- Modify: `.claude/agents/illustrator.md`(新「论文精髓图重绘」节)
- Modify: `.claude/agents/writer.md`(primer 节:符号速查表/首现解释/直觉→公式→数值例/先修框)
- Modify: `docs/superpowers/ARCHITECT-RUNBOOK.md`(primer 发车节:key_figures 建包要求+uplift workflow 用法)

**Interfaces:**
- Consumes: Task 1 检查语义(图注句式/符号覆盖口径,逐字对齐)。
- Produces: illustrator 重绘节含**取原图流程**(ar5iv/arXiv HTML 抓图 URL→curl 下载→Read 亲眼看→重绘;失败降级按包描述+图注如实);writer 先修框样式(blockquote+出处 arXiv 号);explainer `symbol_table: [{symbol, meaning, first_use, source}]` 字段定义。

- [ ] **Step 1: 五处 Edit**(各 ≤15 行,正反例各一;措辞与 Task 1 lint 报错文案对齐)
- [ ] **Step 2: 自检** `grep -n 'key_figures\|symbol_table\|重绘自\|先修框\|台阶四问' .claude/agents/*.md docs/superpowers/ARCHITECT-RUNBOOK.md` 各处命中
- [ ] **Step 3: Commit** `docs(contracts): primer 降台阶四机制入约`

### Task 3: chapter-pipeline reader 门控(primer 转 blocking)

**Files:**
- Modify: `.claude/workflows/chapter-pipeline.js`(readerThunk 附近,约 L259-265)

**Interfaces:**
- Consumes: 文件内既有 `PRIMER` 布尔与 DIM_SCHEMA;write↔review 回环的 blocking 聚合逻辑(先读懂再改)。
- Produces: PRIMER 时 reader prompt 换「第一次读这篇论文的工程师」人格+台阶四问(①符号都认识吗②公式前有直觉铺垫吗③跳步了吗④需要先读别的论文吗),允许 blocking=true(卡回 writer,并入既有回环轮数上限);非 PRIMER 保持顾问性原文不动。

- [ ] **Step 1: 读 readerThunk 与 verdict 聚合**,确认 blocking issue 如何触发 revise 轮
- [ ] **Step 2: 实现**(PRIMER 三元切换 prompt 与 blocking 许可;注意 resume 缓存——非 primer 路径 prompt 逐字不动)
- [ ] **Step 3: wrapped 语法检查**(`// ⚠️ 本环境实测` 切分法)过
- [ ] **Step 4: Commit** `feat(pipeline): reader 维对 primer 转硬门禁(台阶四问)`

### Task 4: primer-uplift workflow(两段式)

**Files:**
- Create: `.claude/workflows/primer-uplift.js`

**Interfaces:**
- Produces: `Workflow({scriptPath, args:{instance, chapters:[slug…], phase:"diagnose"|"apply", approvals?, repo_root?}})`。
  - **diagnose**:每章一 agent(sonnet)读 chapter.md+论文包+explainer.json,写 `{chapter}/reviews/uplift-diagnosis.json`:`{symbols_uncovered:[{symbol, first_use, suggested_meaning}], key_figures_candidates:[{fig, arxiv, shows, why_essential, target_section}], cliff_points:[{section, formula_hint, which_of_four_questions, suggested_fix}], prerequisites:[{concept, cited_paper, where_used, load: "light|heavy|critical", proposal}]}`;返回各类计数+critical 清单。**只读不改正文**。
  - **apply**(args.approvals={chapters:{slug:{key_figures:[…], prerequisites:[…已批级别]}}}):pipeline(章, ①素材 agent:explainer.json 补 symbol_table+论文包 meta.json 写 key_figures+heavy 先修的包扩容(新 md+# PAPER 锚) → ②illustrator:逐 key_figure 取原图亲眼看→重绘→manifest 登记→lint_paper_grounding+lint_diagram_geometry 自跑 → ③writer 定点:符号速查表(本章地图后)/首现解释/直觉垫(cliff_points 逐条)/light 先修框/图嵌入(target_section),自跑 lint_paper_grounding+lint_formulas+lint_chapter_structure → ④reader 门禁 agent(haiku,台阶四问,fail 则 writer 微修回环 ≤2 轮,竭尽 BLOCKED) → 返回 {slug, counts, reader_verdict, rounds})。
- 结构范本:`.claude/workflows/chapter-map-rollout.js`(args 护栏/schema/pipeline 两 stage);回环范本:chapter-pipeline Map 站。

- [ ] **Step 1: 写 workflow**(meta.phases: Diagnose 或 Materials/Illustrate/Write/ReaderGate 按 phase 分支;agent() 的 opts.phase 显式标注)
- [ ] **Step 2: wrapped 语法检查**过
- [ ] **Step 3: Commit** `feat(workflow): primer-uplift 两段式(诊断→批→施工+reader 门禁回环)`

### Task 5: 试点 ch21 Phase A 诊断 → 🛑 用户批准门

- [ ] **Step 1: 发车** `Workflow({scriptPath:".claude/workflows/primer-uplift.js", args:{instance:"vllm-ascend", chapters:["ch21-primer-mla"], phase:"diagnose", repo_root:"/mnt/e/Laboratory/Repo2Book"}})`
- [ ] **Step 2: 汇总诊断呈用户**:先修分级清单(逐条 load+proposal)+key_figures 候选(逐张 why_essential)+台阶点计数;用户批/改
- [ ] **Step 3: 🛑 等批准**(critical 级先修同时决策)

### Task 6: 试点 ch21 Phase B 施工 → 🛑 用户验收门

- [ ] **Step 1: 带批准 args 发 apply**(approvals 内嵌用户裁决)
- [ ] **Step 2: 电池**:lint_paper_grounding(blocking 0/warn 报告)+lint_formulas+structure+map --require+anchors/punct(全书)
- [ ] **Step 3: 验收材料**:重绘图 PNG(SendUserFile)+符号速查表/先修框/直觉垫的 before→after 摘录+reader 门禁轮数
- [ ] **Step 4: Commit**(ch21 产物+诊断存档)`feat(book): ch21 MLA 降台阶试点`
- [ ] **Step 5: 🛑 用户验收**(效果不合意→调契约/模板重跑试点)

### Task 7: 8 章铺开 + 收口(验收后)

- [ ] **Step 1: 批量 Phase A**(两发:ascend 5 章 ch09/23/26/31/34;vllm 3 章 ch24/26/30)→ 汇总一次呈批(含 critical 升级项)
- [ ] **Step 2: 批量 Phase B**(带批准;并发自然排队)
- [ ] **Step 3: 全书电池**(Task 6 同款 ×9 章)+ BLOCKED 处置
- [ ] **Step 4: 记账**:INSTANCE×2/experience-ledger(本轮新病种)/memory;papers-map prerequisite_papers 回填核对
- [ ] **Step 5: Commit** `feat(books): 9 章 primer 降台阶收口`

## Self-Review

- Spec 覆盖:§1→Task 1①+2(explainer/writer 契约)+4③;§2→Task 1②+2(analyst/illustrator)+4②;§3→Task 4(diagnose prerequisites+apply 分级处置)+5/7 批准门;§4→Task 3+4④+2(writer 模式);§5→Task 4-7;§6 风险各有落点(warn 级/降级图注/回环上限/试点先行)。无缺口。
- 占位扫描:无 TBD;Task 4 引用范本文件属结构指引,接口与 schema 已给全。
- 命名一致:`uplift-diagnosis.json`/`key_figures`/`symbol_table`/`paper-fig-*`/`phase:"diagnose"|"apply"` 全文一致。
