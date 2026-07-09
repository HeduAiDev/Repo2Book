# 「本章地图」(源码剖面图) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 每章开篇一张「源码剖面图」——真实源码走线(入口→模块→出口)挂 §N.M 讲解站牌+阅读路线,配确定性门禁;试点 4 章→用户验收→全量 72 章。

**Architecture:** illustrator 逐章手绘(svg-diagram skill 新参考模板定视觉语言),lint_chapter_map 确定性核"§徽标↔实际标题、代码符号↔dossier/正文";chapter-pipeline 在评审收敛后加 Map 站;存量用专项 rollout workflow。Spec: `docs/superpowers/specs/2026-07-08-chapter-map-design.md`。

**Tech Stack:** Python(linter+SVG 模板)/pytest;JS workflow(.claude/workflows);既有 svg-diagram skill、figure-manifest 盲审机制。

## Global Constraints

- 图节点=真实符号名(dossier.mechanisms 锚点可核),≤12 节点,宽 ≤1500 且宽高比 ≤2.6:1;primer 章节点=论文概念(# PAPER 可核)。
- 图位置:开篇 hook 段后、第一个 `##` 前;图后 1–2 句选读指引;禁脚手架措辞。
- HARD RULE:主编排者不写 chapter.md——正文插引由 writer/rollout agent 做;linter/模板/契约/工作流可直接改。
- workflow 语法检查用 async-wrapper 法(`const CFG` 处切分),不用裸 node --check。
- 提交信息带 Co-Authored-By 尾注;不 push。

---

### Task 1: lint_anchors 防复发 warn(行文节号前缀异常)

**Files:**
- Modify: `scripts/lint_anchors.py`(check_cross 附近加 warn 类)
- Test: `scripts/tests/test_lint_anchors.py`

**Interfaces:**
- Produces: warn 类 `stale_section_prefix`——行文 `N.M` 的 N≠本章目录号、且该行无指向 `chNN-` 的链接时 warn(不计退出码,同既有裸章号 warn 通道)。

- [ ] **Step 1: failing tests**(追加到既有测试文件;fixture 风格仿文件内已有 tmp 章结构)

```python
def test_stale_section_prefix_warns(tmp_path):
    ch = _mk_chapter(tmp_path, "ch20-foo", "# 第 20 章 X\n\n正文见 19.5 揭晓。\n")
    warns = run_lint_collect_warns(tmp_path)   # 按文件内既有 helper 命名对齐
    assert any("19.5" in w for w in warns)

def test_section_prefix_with_link_ok(tmp_path):
    body = "# 第 20 章 X\n\n见[第 19 章](../../ch19-bar/narrative/chapter.md) 19.5 节。\n"
    ch = _mk_chapter(tmp_path, "ch20-foo", body); _mk_chapter(tmp_path, "ch19-bar", "# 第 19 章 Y\n")
    warns = run_lint_collect_warns(tmp_path)
    assert not any("19.5" in w for w in warns)

def test_own_section_and_versions_ok(tmp_path):
    body = "# 第 20 章 X\n\n本章 20.3 讲;v0.21.0 与 3.5 倍不受影响。\n"
    _mk_chapter(tmp_path, "ch20-foo", body)
    warns = run_lint_collect_warns(tmp_path)
    assert warns == []
```

实现要点:正则 `(?<![\dv.])(\d{1,2})\.(\d{1,2})(?![\d.])`,N 须命中实例现存章号集合且 ≠ 本章号;跳过代码围栏/标题/图行;同行含 `ch{N:02d}-` 链接则豁免。helper 若不存在,读现测试文件用其真实结构改写以上骨架(断言语义不变)。

- [ ] **Step 2: 跑新测试确认红**:`python3 -m pytest scripts/tests/test_lint_anchors.py -q` → 新增 3 条 FAIL
- [ ] **Step 3: 实现**(lint_anchors.py 内新函数 `check_stale_section_prefix(chapter_dir, all_chapter_nums)`,并入 --all 的 warn 输出通道)
- [ ] **Step 4: 全绿**:同命令 PASS;`REPO2BOOK_INSTANCE=vllm python3 scripts/lint_anchors.py --all` 退出码 0(今日修后 warn 应≈0,若>0 逐条人核是否真残留)
- [ ] **Step 5: Commit** `feat(lint): 行文节号前缀异常 warn——交错残留防复发(exp 台账见 ledger)`

### Task 2: lint_chapter_map.py(新 linter,TDD)

**Files:**
- Create: `scripts/lint_chapter_map.py`
- Test: `scripts/tests/test_lint_chapter_map.py`

**Interfaces:**
- Produces: CLI `python3 scripts/lint_chapter_map.py {chapter_dir} [--require]`。无 `diagrams/chapter-map.svg` 且无 --require → exit 0(豁免期);--require 时缺图/缺开篇引用/缺选读指引 → exit 1。有图必核:①SVG 文本 `§?N.M` 徽标 ⊆ 正文 `## N.M` 标题集且 N=目录号;②代码 token(含 `_`/`()`/`.` 的标识符,len≥4)须为 dossier.json 原文子串或 chapter.md 子串(dossier `kind:primer` 时改核 `book/papers/<slug>/*.md`);违规 exit 1。

- [ ] **Step 1: failing tests**

```python
import json, subprocess, sys
from pathlib import Path
LINT = Path(__file__).resolve().parents[1] / "lint_chapter_map.py"

def _mk(tmp_path, svg_texts, headings="## 20.1 入口\n## 20.2 分流\n", dossier=None, body_extra=""):
    ch = tmp_path / "ch20-foo"; (ch / "diagrams").mkdir(parents=True); (ch / "narrative").mkdir()
    tspans = "".join(f'<text x="0" y="0">{t}</text>' for t in svg_texts)
    (ch / "diagrams" / "chapter-map.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg">{tspans}</svg>', encoding="utf-8")
    (ch / "narrative" / "chapter.md").write_text(
        f"# 第 20 章 X\n\nhook。\n\n![本章地图](../diagrams/chapter-map.png)\n\n只想看结论,跳 §20.2。\n\n{headings}\n{body_extra}", encoding="utf-8")
    (ch / "dossier.json").write_text(json.dumps(dossier or
        {"mechanisms": [{"anchors": ["forward_impl", "_get_fia_params"]}]}), encoding="utf-8")
    return ch

def _run(ch, *flags):
    return subprocess.run([sys.executable, str(LINT), str(ch), *flags], capture_output=True, text=True)

def test_badge_matches_headings_pass(tmp_path):
    assert _run(_mk(tmp_path, ["§20.1", "forward_impl"])).returncode == 0

def test_badge_not_in_headings_fail(tmp_path):
    r = _run(_mk(tmp_path, ["§20.9"]))
    assert r.returncode == 1 and "20.9" in r.stdout

def test_fabricated_symbol_fail(tmp_path):
    r = _run(_mk(tmp_path, ["§20.1", "totally_fake_fn()"]))
    assert r.returncode == 1 and "totally_fake_fn" in r.stdout

def test_no_map_no_require_ok(tmp_path):
    ch = tmp_path / "ch21-bar"; (ch / "narrative").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text("# 第 21 章\n\n## 21.1 a\n", encoding="utf-8")
    assert _run(ch).returncode == 0

def test_no_map_with_require_fail(tmp_path):
    ch = tmp_path / "ch21-bar"; (ch / "narrative").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text("# 第 21 章\n\n## 21.1 a\n", encoding="utf-8")
    assert _run(ch, "--require").returncode == 1

def test_require_checks_position_and_guidance(tmp_path):
    ch = _mk(tmp_path, ["§20.1"])
    md = ch / "narrative" / "chapter.md"
    md.write_text(md.read_text(encoding="utf-8").replace("![本章地图](../diagrams/chapter-map.png)\n\n只想看结论,跳 §20.2。\n\n", "") +
                  "\n![本章地图](../diagrams/chapter-map.png)\n", encoding="utf-8")   # 图挪到第一个 ## 之后
    assert _run(ch, "--require").returncode == 1
```

- [ ] **Step 2: 确认红**:`python3 -m pytest scripts/tests/test_lint_chapter_map.py -q` → 6 FAIL(文件不存在)
- [ ] **Step 3: 实现** `scripts/lint_chapter_map.py`:xml.etree 解析 SVG、`"".join(el.itertext())` 聚 text 节点(tspan 拼接);徽标正则 `§?(\d{1,2})\.(\d{1,2})`;标题正则 `^##\s+(\d{1,2})\.(\d{1,2})`;符号 token 正则 `[A-Za-z_][A-Za-z0-9_.]{2,}\(?\)?` 过滤含 `_` 或 `(` 者;primer 判定读 dossier 顶层 `kind`;--require 位置检查=第一个 `## ` 行之前存在 `chapter-map.png` 引用行、其后 5 个非空行内含 `§` 或 `节`。输出风格仿 lint_formulas(分类+行号+修法)。
- [ ] **Step 4: 全绿** 同命令 6 PASS
- [ ] **Step 5: Commit** `feat(lint): lint_chapter_map——本章地图门禁(徽标↔标题/符号防杜撰/--require 位置与指引)`

### Task 3: renumber 引擎纳入 diagrams/*.py(TDD)

**Files:**
- Modify: `scripts/renumber_chapters.py`(`_rewrite_targets`)
- Test: `scripts/tests/test_renumber_chapters.py`

**Interfaces:**
- Produces: 重编号时 `artifacts/*/diagrams/*.py` 内 `§N.M`/`第 N 章`/`chNN-slug` 同批重写(gen 脚本是 SVG 的真相源;PNG/SVG 由补章 SOP 重跑再生)。`renumber-*.json` 豁免保持不变。

- [ ] **Step 1: failing test**(仿文件内既有 fixture 造 tmp 实例,diagrams 下放 `gen_x.py` 含 `"§19.2"` 与 `ch19-foo` 字符串,跑 --plan 后断言变为 `§20.2`/`ch20-foo`)
- [ ] **Step 2: 确认红**
- [ ] **Step 3: 实现**:`_rewrite_targets` 的 glob 集合追加 `artifacts/*/diagrams/*.py`(保持 `renumber-` 前缀豁免逻辑在后)
- [ ] **Step 4: 全套回归**:`python3 -m pytest scripts/tests/test_renumber_chapters.py -q` 全绿
- [ ] **Step 5: Commit** `feat(renumber): diagrams gen 脚本纳入重写目标——章图不再漂移`

### Task 4: svg-diagram 模板 example-chapter-map.py

**Files:**
- Create: `.claude/skills/svg-diagram/references/example-chapter-map.py`
- Modify: `.claude/skills/svg-diagram/SKILL.md`(模板索引表加一行)

**Interfaces:**
- Produces: 可直接运行的参考实现(以假想章数据):横向泳道=代码层(如 调度层/执行层/算子层),圆角节点=真实符号名+一行短语,右上角 §徽标胶囊(样式:圆角矩形、fill #eef2ff、stroke #6366f1、文字 `§20.4`),入口/出口箭头从画布左右边缘进出,底部路线条(快通道/全通道,§号列表)。视觉语义写死在注释里:**不可变**=徽标样式/路线条/配色语义(入口绿 #22c55e、出口橙 #f97316、主线蓝 #3b82f6);**可变**=泳道数/分组/节点排布。

- [ ] **Step 1: 写模板**(与 references/ 既有 example-*.py 同构:纯 stdlib 生成 SVG,数据表在文件头)
- [ ] **Step 2: 渲染验证**:`python3 example-chapter-map.py && rsvg-convert …` + `python3 scripts/lint_diagram_geometry.py <svg>` 无问题 + Read PNG 亲眼看
- [ ] **Step 3: SKILL.md 索引加行**(何时用:每章开篇本章地图;不可变项列出)
- [ ] **Step 4: Commit** `feat(skill): chapter-map 参考模板——源码剖面图视觉语言定调`

### Task 5: 契约与 RUNBOOK

**Files:**
- Modify: `.claude/agents/illustrator.md`(新「本章地图」职责节)
- Modify: `.claude/agents/writer.md`(开篇结构条款)
- Modify: `docs/superpowers/ARCHITECT-RUNBOOK.md`(Map 站+rollout 说明)
- Modify: `docs/superpowers/ARCHITECT-RUNBOOK.md` 补章 SOP §4(重编号后重跑 diagrams gen 脚本再生 SVG/PNG)

**Interfaces:**
- Consumes: Task 2 的 lint CLI、Task 4 的模板路径。
- Produces: illustrator 条款=输入(定稿 chapter.md+dossier+primer 论文包)/节点预算 ≤12/超长章聚合/自查项(§徽标逐一对标题、符号逐一对 dossier、Read PNG)/模板路径;writer 条款=hook 段后插 `![本章地图:…](../diagrams/chapter-map.png)`+1–2 句选读指引(自然措辞示例给一条)。

- [ ] **Step 1: 三处 Edit**(条款各 ≤15 行,给正反例各一)
- [ ] **Step 2: 自检**:`grep -n 'chapter-map' .claude/agents/*.md docs/superpowers/ARCHITECT-RUNBOOK.md` 每处命中
- [ ] **Step 3: Commit** `docs(contracts): 本章地图职责入约(illustrator/writer/RUNBOOK/补章 SOP)`

### Task 6: chapter-pipeline Map 站

**Files:**
- Modify: `.claude/workflows/chapter-pipeline.js`

**Interfaces:**
- Consumes: 评审收敛(APPROVED)后的定稿 chapter.md;Task 2/4/5 产物。
- Produces: Archive 前新 phase `Map`:单 illustrator agent(prompt 引契约+模板+lint 自跑),站内 ≤2 轮自检回环(lint_chapter_map 无 --require + lint_diagram_geometry),两轮不过 → status=BLOCKED 走既有逃生舱;通过后 writer 微 agent 插图引+指引句,末尾门禁串 `lint_chapter_map --require`。meta.phases 加 `{title:'Map'}`。

- [ ] **Step 1: 读文件定插入点**(write↔review 回环收敛判定之后、Archive agent 之前;沿用 MODELS 映射:illustrator=sonnet、writer=sonnet)
- [ ] **Step 2: 实现**(agent prompt 写明:先读 illustrator.md 契约「本章地图」节+模板;产出 diagrams/chapter-map.{py,svg,png}+figure-manifest 登记+盲审复用既有盲审 agent 形状;null 护栏与 run-ledger 记录仿相邻站)
- [ ] **Step 3: 语法检查**(async-wrapper 法,`// ⚠️ 本环境实测` 处切分):node --check 过
- [ ] **Step 4: Commit** `feat(pipeline): Map 站——评审收敛后产本章地图,门禁 lint_chapter_map --require`

### Task 7: rollout workflow + 试点 4 章(用户验收门)

**Files:**
- Create: `.claude/workflows/chapter-map-rollout.js`

**Interfaces:**
- Produces: `Workflow({scriptPath|name, args:{instance, chapters:[slug…], repo_root}})`;pipeline(chapters, illustrator 画图+登记+自检回环 ≤2, writer 插引+指引, 返回 {slug, map_lint, geometry, blind_verdict});汇总 totals。试点即用它跑 4 章:`vllm-ascend ch20-ascend-attention-mha`、`vllm-ascend ch03-two-stage-monkey-patch`、`vllm ch24-primer-flash-attention`、`vllm ch36-engine-core`。

- [ ] **Step 1: 写 workflow**(meta phases: Draw/Insert;两 stage pipeline;illustrator prompt 与 Task 6 同源措辞——从契约引用而非复制;schema 强制返回计数与 verdict)
- [ ] **Step 2: 语法检查**(async-wrapper)
- [ ] **Step 3: 试点发车**(两次 Workflow 调用,ascend 2 章 + vllm 2 章;args 显式)
- [ ] **Step 4: 验收材料**:4 张 chapter-map.png 用 SendUserFile 发给用户,附每章 lint/盲审结论
- [ ] **Step 5: Commit**(workflow 脚本+4 章产物)`feat(rollout): chapter-map-rollout+试点 4 章`
- [ ] **Step 6: 🛑 用户验收门**——风格/信息密度定稿,拿到"继续"再进 Task 8;返工则调模板/契约后重跑试点

### Task 8: 全量铺开(验收后)+ 收口

**Files:**
- Modify: 68 章 artifacts(rollout workflow 产出)
- Modify: `CLAUDE.md`(质量闸门清单加 lint_chapter_map 一行)、`docs/superpowers/experience-ledger.md`、两书 `INSTANCE.md`

**Interfaces:**
- Consumes: Task 7 定稿的模板/契约。

- [ ] **Step 1: 全量发车**(rollout workflow,两书各一次,章列表=全部减试点 4 章;并发上限自然排队)
- [ ] **Step 2: 电池**:逐章 `lint_chapter_map --require` + `lint_diagram_geometry --all` + `lint_anchors/punct --all` + 36/36×2 structure 全绿
- [ ] **Step 3: pipeline --require 生效确认**(Task 6 已串,此处跑一遍 wrapped check 防漂移)
- [ ] **Step 4: 记账**:INSTANCE.md×2、ledger(若 rollout 暴露 ≥2 章重复问题按回流流程记)、CLAUDE.md 闸门行
- [ ] **Step 5: Commit** `feat(books): 72 章本章地图全量铺开+门禁转 blocking`

## Self-Review

- Spec 覆盖:§1(形态/位置)→Task 4/5/7;§2(生产)→Task 4/6/7;§3(门禁 4 条)→Task 2(1-3)/既有 geometry(4)+防复发→Task 1;§4(pipeline/契约/rollout)→Task 5/6/7;§5(试点→全量)→Task 7/8;§6(renumber 漂移)→Task 3+SOP(Task 5)。无缺口。
- 占位扫描:无 TBD;Task 1 测试骨架标明"按既有 helper 对齐"属适配性说明,语义完整。
- 类型/命名一致:`chapter-map.{py,svg,png}`、`lint_chapter_map.py --require`、workflow args `{instance, chapters, repo_root}` 全文一致。
