# repo2book v3 素材先行流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec `docs/superpowers/specs/2026-07-02-material-first-pipeline-v3-design.md`——新增 explainer/illustrator 两角色与 Explain/Illustrate 两流水线阶段、三个新 linter、svg-diagram skill v2、存量回修 workflow,使 opus/sonnet 全流水线可稳定产出"深算法讲解 + 语义准确插图"。

**Architecture:** 图与数值轨迹先于写作、经运行验证产出(explainer.json 素材真相源);插图强制"渲染→Read PNG 亲眼看→自查→盲审"双重验收;writer 减负拿素材自由叙事;质量约束全部落在类型化工件 + 确定性 linter,不落在行文。

**Tech Stack:** Python 3(stdlib only,与现有 scripts/ 一致)、pytest(scripts/tests/ 现有约定)、Claude Code Workflow JS(ESM)、SVG + rsvg-convert。

## Global Constraints

- 全流水线目标执行模型是 opus/sonnet:所有 agent 提示词 ≤60 行、程序化、每步给确定性验收命令(spec §2 P3)。
- 只做减法契约、零脚手架泄漏、公式规则等 CLAUDE.md 既有硬规则全部继续有效。
- 新 linter 风格与现有一致:stdlib-only、`lint_x(chapter_dir)->dict` + `print_report(res, cd)->int`、阻断 exit 1、中文报告、`scripts/tests/test_lint_x.py` 用 tmp_path 构造 fixture(参照 `scripts/tests/test_lint_diagrams.py`)。
- 不动:implementer/tester 契约、bible.py/archivist.py、现有 5 个 linter 的既有行为(lint_diagrams 只增不改旧判定)、两本书已 APPROVED 的章节内容。
- git:每任务用 `git add <明确文件>` 定点提交(不 `git add -A`,工作区有本计划无关的脏文件);**绝不 push**(push 须用户前台操作)。
- 提交信息格式沿用仓库习惯(`feat(...)`/中文摘要),尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 章节工件目录约定(已存在,勿改名):`{chapter_dir}/dossier/dossier.json`、`implementation/`、`tests/`、`narrative/chapter.md`、`diagrams/`、`reviews/`;v3 新增 `explainer/explainer.json`、`explainer/traces/`、`diagrams/figure-manifest.json`、`retrofit/retrofit-plan.json`。

## File Structure(全景)

```
scripts/lint_dossier.py                      # 新:dossier.mechanisms 账本校验
scripts/lint_explainer.py                    # 新:explainer 素材校验(数字可溯源)
scripts/lint_trace_consistency.py            # 新:正文数值表 vs 素材,数字不漂移
scripts/lint_diagrams.py                     # 改:v3 章增查 figure-manifest(自查全真+盲审 PASS)
scripts/tests/test_lint_{dossier,explainer,trace_consistency}.py   # 新
scripts/tests/test_lint_diagrams.py          # 改:补 manifest 用例
.claude/skills/svg-diagram/SKILL.md          # 重写 v2:figure-spec 先行/设计规则/双重验收
.claude/skills/svg-diagram/references/example-{swimlane,layout,before-after,state-machine}.py  # 新模板
.claude/agents/explainer.md                  # 新角色
.claude/agents/illustrator.md                # 新角色
.claude/agents/{analyst,writer,reviewer,archivist}.md  # 改
.claude/workflows/chapter-pipeline.js        # 改:插 Explain/Illustrate,重写 Write/Review 契约
.claude/workflows/chapter-retrofit.js        # 新:存量外科回修
CLAUDE.md / docs/superpowers/ARCHITECT-RUNBOOK.md      # 文档同步
```

各 JSON 工件的权威 schema 以 spec §4.1/§4.3/§4.4/§5.1 为准;linter 代码即可执行定义。

---

### Task 1: scripts/lint_dossier.py(TDD)

**Files:**
- Create: `scripts/lint_dossier.py`
- Test: `scripts/tests/test_lint_dossier.py`

**Interfaces:**
- Consumes: `{chapter_dir}/dossier/dossier.json` 的 `mechanisms[]` 字段(见测试 fixture 即 schema)。
- Produces: `lint_dossier(chapter_dir: str) -> dict`(keys: `invalid, mechanism, anchor, warn`;`warn` 非阻断)+ CLI `python3 scripts/lint_dossier.py <chapter_dir>`,阻断 exit 1。Task 8/9 的 workflow 会调用此 CLI。

- [ ] **Step 1: 写失败测试** `scripts/tests/test_lint_dossier.py`:

```python
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_dossier import lint_dossier

GOOD_MECH = {
    "id": "m1", "name": "抢占回退循环", "kind": "algorithm",
    "source_anchors": ["pkg/sched.py:L2-L4"], "needs_figure": True,
    "needs_worked_example": True, "difficulty": "core",
}


def _mk(tmp, mechanisms, with_source=True):
    """构造 instances 形状的树:<inst>/artifacts/ch01 + <inst>/source/pkg/sched.py(5 行)。"""
    inst = tmp / "inst"
    ch = inst / "artifacts" / "ch01"
    (ch / "dossier").mkdir(parents=True)
    if with_source:
        (inst / "source" / "pkg").mkdir(parents=True)
        (inst / "source" / "pkg" / "sched.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    (ch / "dossier" / "dossier.json").write_text(
        json.dumps({"mechanisms": mechanisms}), encoding="utf-8")
    return str(ch)


def test_valid_mechanisms_pass(tmp_path):
    r = lint_dossier(_mk(tmp_path, [GOOD_MECH]))
    assert not r["invalid"] and not r["mechanism"] and not r["anchor"]


def test_missing_mechanisms_blocking(tmp_path):
    r = lint_dossier(_mk(tmp_path, []))
    assert r["invalid"]


def test_algorithm_without_worked_example_blocking(tmp_path):
    m = dict(GOOD_MECH, needs_worked_example=False)
    assert lint_dossier(_mk(tmp_path, [m]))["mechanism"]


def test_anchor_line_out_of_range_blocking(tmp_path):
    m = dict(GOOD_MECH, source_anchors=["pkg/sched.py:L2-L99"])
    assert lint_dossier(_mk(tmp_path, [m]))["anchor"]


def test_anchor_bad_format_blocking(tmp_path):
    m = dict(GOOD_MECH, source_anchors=["sched.py 第2行"])
    assert lint_dossier(_mk(tmp_path, [m]))["anchor"]


def test_missing_source_dir_warns_only(tmp_path):
    r = lint_dossier(_mk(tmp_path, [GOOD_MECH], with_source=False))
    assert r["warn"] and not r["anchor"]


def test_duplicate_id_blocking(tmp_path):
    assert lint_dossier(_mk(tmp_path, [GOOD_MECH, dict(GOOD_MECH)]))["mechanism"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/e/Laboratory/Repo2Book && python3 -m pytest scripts/tests/test_lint_dossier.py -q`
Expected: FAIL/ERROR `ModuleNotFoundError: No module named 'lint_dossier'`

- [ ] **Step 3: 实现 `scripts/lint_dossier.py`**(完整内容):

```python
#!/usr/bin/env python3
"""Dossier 机制清单 linter — 校验 dossier.json 的 mechanisms[](v3 素材先行流水线的账本)。

mechanisms 是"一图讲一机制、一例讲一算法"的覆盖度账本:explainer 按它产素材、
illustrator 按它配图、reviewer 按它对账。

阻断项:JSON 不合法/缺 mechanisms;机制缺必填字段、枚举非法、id 重复;
        kind=algorithm 但 needs_worked_example!=true;source_anchors 格式非法/文件不存在/行号越界。
警告项:实例 source/ 不在(跳过锚点行号核验)。
用法:python3 lint_dossier.py <chapter_dir>   阻断项存在则 exit 1。
"""
import json
import re
import sys
from pathlib import Path

KINDS = {"algorithm", "dataflow", "layout", "protocol", "config"}
DIFF = {"core", "supporting"}
ANCHOR = re.compile(r'^([\w./-]+\.\w+):L(\d+)(?:-L?(\d+))?$')


def _source_root(chapter_dir: Path):
    for p in chapter_dir.resolve().parents:
        if p.name == "artifacts":
            return p.parent / "source"
    return None


def lint_dossier(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    res = {"invalid": [], "mechanism": [], "anchor": [], "warn": []}
    f = d / "dossier" / "dossier.json"
    if not f.exists():
        res["invalid"].append("  dossier/dossier.json 缺失")
        return res
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except ValueError as e:
        res["invalid"].append(f"  JSON 不合法: {e}")
        return res
    mechs = doc.get("mechanisms")
    if not isinstance(mechs, list) or not mechs:
        res["invalid"].append("  缺 mechanisms[](v3 机制清单——覆盖度/配图/深度的账本)")
        return res
    src = _source_root(d)
    if src is None or not src.exists():
        res["warn"].append("  找不到实例 source/,跳过锚点行号核验")
        src = None
    seen = set()
    for i, m in enumerate(mechs):
        mid = m.get("id") or f"#{i}"
        if m.get("id") in seen:
            res["mechanism"].append(f"  {mid}: id 重复")
        seen.add(m.get("id"))
        for k in ("id", "name", "kind", "source_anchors", "difficulty"):
            if not m.get(k):
                res["mechanism"].append(f"  {mid}: 缺 {k}")
        if m.get("kind") not in KINDS:
            res["mechanism"].append(f"  {mid}: kind={m.get('kind')!r} 非法(应为 {sorted(KINDS)})")
        if m.get("difficulty") not in DIFF:
            res["mechanism"].append(f"  {mid}: difficulty={m.get('difficulty')!r} 非法(core|supporting)")
        if m.get("kind") == "algorithm" and m.get("needs_worked_example") is not True:
            res["mechanism"].append(f"  {mid}: kind=algorithm 必须 needs_worked_example=true")
        for a in m.get("source_anchors") or []:
            am = ANCHOR.match(a)
            if not am:
                res["anchor"].append(f"  {mid}: 锚点格式非法 {a!r}(应为 path:Lnnn[-Lnnn])")
                continue
            if src is None:
                continue
            fp = src / am.group(1)
            if not fp.exists():
                res["anchor"].append(f"  {mid}: 文件不存在 {am.group(1)}")
                continue
            n = sum(1 for _ in fp.open(encoding="utf-8", errors="replace"))
            end = int(am.group(3) or am.group(2))
            if end > n:
                res["anchor"].append(f"  {mid}: 行号越界 {a}(文件共 {n} 行)")
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Dossier Lint: {cd}\n{'=' * 60}")
    blocking = sum(len(v) for k, v in res.items() if k != "warn")
    for k, issues in res.items():
        for i in issues:
            print(("⚠️ " if k == "warn" else "❌ ") + f"{k}: {i}")
    if blocking == 0:
        print("✓ dossier 机制清单检查通过")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_dossier.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_dossier(sys.argv[1]), sys.argv[1]))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest scripts/tests/test_lint_dossier.py -q`
Expected: `7 passed`

- [ ] **Step 5: 提交**

```bash
git add scripts/lint_dossier.py scripts/tests/test_lint_dossier.py
git commit -m "feat(v3): lint_dossier——mechanisms 机制清单账本校验(素材先行流水线 Task 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: scripts/lint_explainer.py(TDD)

**Files:**
- Create: `scripts/lint_explainer.py`
- Test: `scripts/tests/test_lint_explainer.py`

**Interfaces:**
- Consumes: `{chapter_dir}/dossier/dossier.json`(Task 1 的 mechanisms 账本)+ `{chapter_dir}/explainer/explainer.json` + `{chapter_dir}/explainer/traces/*`。
- Produces: `lint_explainer(chapter_dir: str) -> dict`(keys: `invalid, mechanism, trace, figure, warn`)+ CLI 同 Task 1 约定。explainer.json 的可执行 schema 就在本 linter 与测试 fixture 中,Task 6 的 explainer 角色契约引用它。

- [ ] **Step 1: 写失败测试** `scripts/tests/test_lint_explainer.py`:

```python
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_explainer import lint_explainer

DOSSIER = {"mechanisms": [{
    "id": "m1", "name": "抢占回退循环", "kind": "algorithm",
    "source_anchors": ["pkg/sched.py:L1-L2"], "needs_figure": True,
    "needs_worked_example": True, "difficulty": "core"}]}

GOOD_ENTRY = {
    "mechanism_id": "m1",
    "intuition": "像叠盘子,永远从最上面拿",
    "worked_example": {
        "params": {"queue": 3},
        "trace_source": "run",
        "trace_ref": "traces/m1.json",
        "table": {"columns": ["轮次", "队列长", "返回"],
                  "rows": [["1", "3", "-"], ["2", "2", "req9"]]},
    },
    "invariant": {"claim": "队列长每轮严格减 1",
                  "argument": "每轮必 pop 一次,非负整数单调递减必有限步触底"},
    "quantified": "3 个请求 2 轮完成,O(len(running))",
    "figure_specs": [{
        "figure_id": "fig-m1", "claim": "抢占按 LIFO 弹出尾部,队列长每轮减 1",
        "template": "state-table",
        "numbers": [{"value": "3", "provenance": "traces/m1.json"}],
        "elements": ["逐轮状态表"], "caption_draft": "队列长 3→2→1:LIFO 抢占每轮恰弹出一个",
    }],
}
TRACE = '{"rounds": [{"round": 1, "qlen": 3}, {"round": 2, "qlen": 2, "victim": "req9"}]}'


def _mk(tmp, entry, trace=TRACE, dossier=DOSSIER):
    ch = tmp / "inst" / "artifacts" / "ch01"
    (ch / "dossier").mkdir(parents=True)
    (ch / "explainer" / "traces").mkdir(parents=True)
    (ch / "dossier" / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
    (ch / "explainer" / "explainer.json").write_text(
        json.dumps({"mechanisms": [entry] if entry else []}), encoding="utf-8")
    if trace is not None:
        (ch / "explainer" / "traces" / "m1.json").write_text(trace, encoding="utf-8")
    return str(ch)


def test_good_entry_passes(tmp_path):
    r = lint_explainer(_mk(tmp_path, GOOD_ENTRY))
    assert not r["invalid"] and not r["mechanism"] and not r["trace"] and not r["figure"]


def test_missing_mechanism_entry_blocking(tmp_path):
    assert lint_explainer(_mk(tmp_path, None))["mechanism"]


def test_table_number_not_in_trace_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["worked_example"]["table"]["rows"][0][1] = "777"   # trace 里没有 777
    assert lint_explainer(_mk(tmp_path, e))["trace"]


def test_single_row_table_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["worked_example"]["table"]["rows"] = [["1", "3", "-"]]
    assert lint_explainer(_mk(tmp_path, e))["mechanism"]


def test_manual_without_reason_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["worked_example"]["trace_source"] = "manual"
    del e["worked_example"]["trace_ref"]
    assert lint_explainer(_mk(tmp_path, e, trace=None))["trace"]


def test_manual_with_reason_warns_only(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["worked_example"]["trace_source"] = "manual"
    e["worked_example"]["manual_reason"] = "本章 skip_impl,无精简版可跑"
    del e["worked_example"]["trace_ref"]
    r = lint_explainer(_mk(tmp_path, e, trace=None))
    assert not r["trace"] and r["warn"]


def test_needs_figure_without_spec_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["figure_specs"] = []
    assert lint_explainer(_mk(tmp_path, e))["figure"]


def test_figure_number_without_provenance_blocking(tmp_path):
    e = json.loads(json.dumps(GOOD_ENTRY))
    e["figure_specs"][0]["numbers"] = [{"value": "3"}]
    assert lint_explainer(_mk(tmp_path, e))["figure"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest scripts/tests/test_lint_explainer.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'lint_explainer'`

- [ ] **Step 3: 实现 `scripts/lint_explainer.py`**(完整内容):

```python
#!/usr/bin/env python3
"""Explainer 素材 linter — 校验 explainer/explainer.json(illustrator/writer 的素材真相源)。

对照 dossier.mechanisms 账本逐机制对账:
阻断项:explainer.json 缺失/不合法;needs_worked_example 机制无条目;intuition/quantified/
        invariant 空;逐轮表 <2 行;trace_source 非法;run 无 trace_ref 或表格数字在 trace 原始
        输出里找不到(数字不许编);manual 无 manual_reason;needs_figure 机制无 figure_spec;
        figure_spec 缺 claim/template/caption_draft、numbers 缺 value/provenance。
警告项:manual trace 未经运行验证。
用法:python3 lint_explainer.py <chapter_dir>   阻断项存在则 exit 1。
"""
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r'-?\d+(?:\.\d+)?')
TEMPLATES = {"state-table", "swimlane", "layout", "tensor-flow",
             "before-after", "state-machine", "flow", "tiling"}


def _nums(text: str) -> set:
    return {float(t) for t in NUM.findall(text)}


def lint_explainer(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    res = {"invalid": [], "mechanism": [], "trace": [], "figure": [], "warn": []}
    ef = d / "explainer" / "explainer.json"
    if not ef.exists():
        res["invalid"].append("  explainer/explainer.json 缺失")
        return res
    try:
        doc = json.loads(ef.read_text(encoding="utf-8"))
    except ValueError as e:
        res["invalid"].append(f"  JSON 不合法: {e}")
        return res
    dossier = {}
    df = d / "dossier" / "dossier.json"
    if df.exists():
        try:
            dossier = json.loads(df.read_text(encoding="utf-8"))
        except ValueError:
            pass
    want_we = {m["id"] for m in dossier.get("mechanisms", []) if m.get("needs_worked_example")}
    want_fig = {m["id"] for m in dossier.get("mechanisms", []) if m.get("needs_figure")}
    got = {m.get("mechanism_id"): m for m in doc.get("mechanisms", [])}
    for mid in sorted(want_we - set(got)):
        res["mechanism"].append(f"  {mid}: dossier 要求 worked example,explainer 无条目")
    for mid, m in got.items():
        we = m.get("worked_example") or {}
        for k in ("intuition", "quantified"):
            if not m.get(k):
                res["mechanism"].append(f"  {mid}: {k} 为空")
        inv = m.get("invariant") or {}
        if not (inv.get("claim") and inv.get("argument")):
            res["mechanism"].append(f"  {mid}: invariant.claim/argument 为空(要单调量或基例+归纳步,不是断言)")
        rows = ((we.get("table") or {}).get("rows")) or []
        if len(rows) < 2:
            res["mechanism"].append(f"  {mid}: 逐轮表不足 2 轮(rows={len(rows)})")
        ts = we.get("trace_source")
        if ts == "run":
            tr = d / "explainer" / (we.get("trace_ref") or "")
            if not we.get("trace_ref") or not tr.exists():
                res["trace"].append(f"  {mid}: trace_source=run 但 trace_ref 缺失/文件不存在")
            else:
                have = _nums(tr.read_text(encoding="utf-8", errors="replace"))
                for row in rows:
                    for cell in row:
                        for v in _nums(str(cell)):
                            if v not in have:
                                res["trace"].append(
                                    f"  {mid}: 表格数字 {v:g} 在 {we['trace_ref']} 里找不到(数字不许编)")
        elif ts == "manual":
            if not we.get("manual_reason"):
                res["trace"].append(f"  {mid}: trace_source=manual 必须写 manual_reason(降级原因)")
            else:
                res["warn"].append(f"  {mid}: manual trace 未经运行验证")
        else:
            res["trace"].append(f"  {mid}: trace_source={ts!r} 非法(run|manual)")
        specs = m.get("figure_specs") or []
        if mid in want_fig and not specs:
            res["figure"].append(f"  {mid}: dossier 要求配图,figure_specs 为空")
        for s in specs:
            fid = s.get("figure_id") or "?"
            for k in ("claim", "caption_draft"):
                if not s.get(k):
                    res["figure"].append(f"  {fid}: {k} 为空")
            if s.get("template") not in TEMPLATES:
                res["figure"].append(f"  {fid}: template={s.get('template')!r} 非法({sorted(TEMPLATES)})")
            for n in s.get("numbers") or []:
                if not (n.get("value") and n.get("provenance")):
                    res["figure"].append(f"  {fid}: numbers 项缺 value/provenance(图中数字皆有出处)")
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Explainer Lint: {cd}\n{'=' * 60}")
    blocking = sum(len(v) for k, v in res.items() if k != "warn")
    for k, issues in res.items():
        for i in issues:
            print(("⚠️ " if k == "warn" else "❌ ") + f"{k}: {i}")
    if blocking == 0:
        print("✓ explainer 素材检查通过(数字可溯源/逐轮表/不变量/figure-spec 齐)")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_explainer.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_explainer(sys.argv[1]), sys.argv[1]))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest scripts/tests/test_lint_explainer.py -q`
Expected: `8 passed`

- [ ] **Step 5: 提交**

```bash
git add scripts/lint_explainer.py scripts/tests/test_lint_explainer.py
git commit -m "feat(v3): lint_explainer——素材真相源校验,表格数字必须在 trace 里可寻(Task 2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: scripts/lint_trace_consistency.py(TDD)

**Files:**
- Create: `scripts/lint_trace_consistency.py`
- Test: `scripts/tests/test_lint_trace_consistency.py`

**Interfaces:**
- Consumes: `narrative/chapter.md`(含 `<!-- trace: <mechanism_id> -->` 标记约定,标记必须紧邻 markdown 表格之前,空行允许)、`explainer/explainer.json`、`dossier/dossier.json`。
- Produces: `lint_trace_consistency(chapter_dir) -> dict`(keys: `invalid, drift, coverage, warn`)+ CLI。**标记约定由本 linter 定义**,Task 7 writer 契约与 Task 8 workflow 引用:"数值推演表前一行放 `<!-- trace: mX -->`(HTML 注释,读者不可见)"。无 explainer.json 的章(v2 旧章)只 warn 不阻断——保证旧书兼容。

- [ ] **Step 1: 写失败测试** `scripts/tests/test_lint_trace_consistency.py`:

```python
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_trace_consistency import lint_trace_consistency

EXPLAINER = {"mechanisms": [{
    "mechanism_id": "m1",
    "worked_example": {"table": {"columns": ["轮次", "队列长"],
                                 "rows": [["1", "3"], ["2", "2"]]}},
}]}
DOSSIER = {"mechanisms": [{"id": "m1", "needs_worked_example": True}]}

GOOD_MD = """# 第 X 章

<!-- trace: m1 -->

| 轮次 | 队列长 |
|---|---|
| 1 | 3 |
| 2 | 2 |
"""


def _mk(tmp, md, explainer=EXPLAINER, dossier=DOSSIER):
    ch = tmp / "inst" / "artifacts" / "ch01"
    (ch / "narrative").mkdir(parents=True)
    (ch / "explainer").mkdir(parents=True)
    (ch / "dossier").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(md, encoding="utf-8")
    if explainer is not None:
        (ch / "explainer" / "explainer.json").write_text(json.dumps(explainer), encoding="utf-8")
    (ch / "dossier" / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
    return str(ch)


def test_matching_table_passes(tmp_path):
    r = lint_trace_consistency(_mk(tmp_path, GOOD_MD))
    assert not r["invalid"] and not r["drift"] and not r["coverage"]


def test_drifted_number_blocking(tmp_path):
    md = GOOD_MD.replace("| 2 | 2 |", "| 2 | 99 |")   # 99 不在素材里
    assert lint_trace_consistency(_mk(tmp_path, md))["drift"]


def test_unknown_mechanism_mark_blocking(tmp_path):
    md = GOOD_MD.replace("trace: m1", "trace: m9")
    assert lint_trace_consistency(_mk(tmp_path, md))["invalid"]


def test_mark_without_table_blocking(tmp_path):
    md = "<!-- trace: m1 -->\n\n这里没有表格。\n"
    assert lint_trace_consistency(_mk(tmp_path, md))["invalid"]


def test_missing_mark_is_coverage_gap(tmp_path):
    md = "# 第 X 章\n\n正文完全没有数值推演表标记。\n"
    assert lint_trace_consistency(_mk(tmp_path, md))["coverage"]


def test_no_explainer_old_chapter_warns_only(tmp_path):
    r = lint_trace_consistency(_mk(tmp_path, GOOD_MD, explainer=None))
    assert r["warn"] and not r["invalid"] and not r["drift"] and not r["coverage"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest scripts/tests/test_lint_trace_consistency.py -q`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: 实现 `scripts/lint_trace_consistency.py`**(完整内容):

```python
#!/usr/bin/env python3
"""数值一致性 linter — 正文数值推演表 vs explainer 素材,数字不许漂移。

约定(writer 契约同款):数值推演表的 markdown 表格前放一行 `<!-- trace: <mechanism_id> -->`
(HTML 注释,读者不可见;标记与表格间允许空行)。writer 可自由改排版/措辞,数字不可改。

阻断项:标记的机制在 explainer 里不存在;标记后没有紧跟表格;表内出现素材之外的数字(漂移);
        dossier 里 needs_worked_example 的机制在正文无任何 trace 标记(覆盖缺口)。
警告项:无 explainer.json(v2 旧章,跳过检查)。
用法:python3 lint_trace_consistency.py <chapter_dir>   阻断项存在则 exit 1。
"""
import json
import re
import sys
from pathlib import Path

NUM = re.compile(r'-?\d+(?:\.\d+)?')
MARK = re.compile(r'<!--\s*trace:\s*([\w-]+)\s*-->')


def _nums(text: str) -> set:
    return {float(t) for t in NUM.findall(text)}


def _tables_after_marks(md: str):
    lines = md.splitlines()
    for i, ln in enumerate(lines):
        m = MARK.search(ln)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        rows = []
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            rows.append(lines[j])
            j += 1
        yield m.group(1), "\n".join(rows)


def lint_trace_consistency(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    res = {"invalid": [], "drift": [], "coverage": [], "warn": []}
    ef = d / "explainer" / "explainer.json"
    if not ef.exists():
        res["warn"].append("  无 explainer.json(v2 旧章?)——跳过一致性检查")
        return res
    nar = d / "narrative" / "chapter.md"
    if not nar.exists():
        res["invalid"].append("  narrative/chapter.md 缺失")
        return res
    try:
        doc = json.loads(ef.read_text(encoding="utf-8"))
    except ValueError as e:
        res["invalid"].append(f"  explainer.json 不合法: {e}")
        return res
    allowed = {}
    for m in doc.get("mechanisms", []):
        table = ((m.get("worked_example") or {}).get("table")) or {}
        allowed[m.get("mechanism_id")] = _nums(json.dumps(table))
    marked = set()
    for mid, table in _tables_after_marks(nar.read_text(encoding="utf-8")):
        marked.add(mid)
        if mid not in allowed:
            res["invalid"].append(f"  标记 trace:{mid} 在 explainer 里不存在")
            continue
        if not table:
            res["invalid"].append(f"  标记 trace:{mid} 后没有紧跟 markdown 表格")
            continue
        extra = _nums(table) - allowed[mid]
        if extra:
            res["drift"].append(
                f"  trace:{mid} 表内数字 {sorted(extra)} 不在 explainer 素材里(数字不可改,排版随意)")
    df = d / "dossier" / "dossier.json"
    if df.exists():
        try:
            need = {m["id"] for m in json.loads(df.read_text(encoding="utf-8")).get("mechanisms", [])
                    if m.get("needs_worked_example")}
        except ValueError:
            need = set()
        for mid in sorted(need - marked):
            res["coverage"].append(f"  机制 {mid} 的数值推演表未进正文(缺 <!-- trace: {mid} --> 标记)")
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Trace-Consistency Lint: {cd}\n{'=' * 60}")
    blocking = sum(len(v) for k, v in res.items() if k != "warn")
    for k, issues in res.items():
        for i in issues:
            print(("⚠️ " if k == "warn" else "❌ ") + f"{k}: {i}")
    if blocking == 0:
        print("✓ 正文数值推演表与 explainer 素材一致")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_trace_consistency.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_trace_consistency(sys.argv[1]), sys.argv[1]))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest scripts/tests/test_lint_trace_consistency.py -q`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add scripts/lint_trace_consistency.py scripts/tests/test_lint_trace_consistency.py
git commit -m "feat(v3): lint_trace_consistency——正文数值表数字不漂移+机制覆盖对账(Task 3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: lint_diagrams.py v2 —— figure-manifest 校验(TDD,只增不改旧判定)

**Files:**
- Modify: `scripts/lint_diagrams.py`(在 `lint_diagrams()` 返回前追加 manifest 检查;`print_report` 的 blocking 统计加上 manifest)
- Test: `scripts/tests/test_lint_diagrams.py`(追加用例,现有 4 个用例必须原样通过)

**Interfaces:**
- Consumes: `diagrams/figure-manifest.json`(schema 见下)+ `explainer/explainer.json`(有它才启用 manifest 检查——旧章无 explainer 不受影响)。
- Produces: `lint_diagrams()` 返回 dict 新增 key `manifest`(阻断)。manifest schema(Task 6 illustrator 契约、Task 8 盲审 agent 引用):`{"figures": [{"figure_id", "gen", "svg", "png", "selfcheck": {"claim_readable_10s", "numbers_match_spec", "no_overlap", "arrows_attached", "cjk_rendered", "reading_order_clear"}(全 bool), "blind_review": {"verdict": "PASS|FAIL|PENDING", "notes"}}]}`。

- [ ] **Step 1: 在 `scripts/tests/test_lint_diagrams.py` 追加失败用例**(文件末尾追加;并给 `_mk` 加 explainer/manifest 参数——修改 `_mk` 为下面版本,原 4 个用例不用改):

```python
# _mk 替换为(向后兼容,新增两个可选参数):
def _mk(tmp, svgs: dict, pngs: list, narrative: str = "", explainer: str = None, manifest: str = None):
    d = tmp / "ch"
    (d / "diagrams").mkdir(parents=True)
    (d / "narrative").mkdir(parents=True)
    for name, body in svgs.items():
        (d / "diagrams" / name).write_text(body, encoding="utf-8")
    for name in pngs:
        (d / "diagrams" / name).write_bytes(b"\x89PNG" + b"0" * 4000)
    (d / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    if explainer is not None:
        (d / "explainer").mkdir(parents=True)
        (d / "explainer" / "explainer.json").write_text(explainer, encoding="utf-8")
    if manifest is not None:
        (d / "diagrams" / "figure-manifest.json").write_text(manifest, encoding="utf-8")
    return str(d)


# 文件末尾追加:
EXPL = ('{"mechanisms": [{"mechanism_id": "m1", "figure_specs": '
        '[{"figure_id": "fig-x", "claim": "c", "template": "flow"}]}]}')
SELF_OK = ('{"claim_readable_10s": true, "numbers_match_spec": true, "no_overlap": true, '
           '"arrows_attached": true, "cjk_rendered": true, "reading_order_clear": true}')


def _man(verdict="PASS", selfcheck=SELF_OK):
    return ('{"figures": [{"figure_id": "fig-x", "gen": "gen_fig-x.py", "svg": "fig-x.svg", '
            '"png": "fig-x.png", "selfcheck": ' + selfcheck +
            ', "blind_review": {"verdict": "' + verdict + '", "notes": ""}}]}')


def test_v3_manifest_ok_passes(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG, "gen_fig-x.py": "#"}, ["fig-x.png"],
            "![](../diagrams/fig-x.png)", explainer=EXPL, manifest=_man())
    assert not lint_diagrams(d)["manifest"]


def test_v3_missing_manifest_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG}, ["fig-x.png"],
            "![](../diagrams/fig-x.png)", explainer=EXPL)
    assert lint_diagrams(d)["manifest"]


def test_v3_blind_review_not_pass_blocking(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG, "gen_fig-x.py": "#"}, ["fig-x.png"],
            "![](../diagrams/fig-x.png)", explainer=EXPL, manifest=_man(verdict="PENDING"))
    assert lint_diagrams(d)["manifest"]


def test_v3_selfcheck_false_blocking(tmp_path):
    bad = SELF_OK.replace('"no_overlap": true', '"no_overlap": false')
    d = _mk(tmp_path, {"fig-x.svg": SVG, "gen_fig-x.py": "#"}, ["fig-x.png"],
            "![](../diagrams/fig-x.png)", explainer=EXPL, manifest=_man(selfcheck=bad))
    assert lint_diagrams(d)["manifest"]


def test_old_chapter_without_explainer_unaffected(tmp_path):
    d = _mk(tmp_path, {"fig-x.svg": SVG}, ["fig-x.png"], "![](../diagrams/fig-x.png)")
    assert not lint_diagrams(d)["manifest"]
```

注意:gen 脚本以 `gen_fig-x.py` 名写进 svgs dict(会被当文本写入 diagrams/,足够文件存在性检查用);`_mk` 里 svg glob 不匹配 .py,不影响旧判定。

- [ ] **Step 2: 跑测试确认新用例失败、旧用例通过**

Run: `python3 -m pytest scripts/tests/test_lint_diagrams.py -q`
Expected: 4 passed(旧), 5 failed/error(新——`KeyError: 'manifest'`)

- [ ] **Step 3: 修改 `scripts/lint_diagrams.py`**:

3a. `lint_diagrams()` 里 res 初始化行加 `"manifest": []`:

```python
    res = {"svg_invalid": [], "png_missing": [], "orphan": [], "no_renderer": [], "overflow": [], "manifest": []}
```

3b. 文件头 import 区加 `import json`。docstring 阻断项一段追加一行:`v3 章(有 explainer.json):figure-manifest.json 缺失/未登记 spec 图/自查非全真/盲审非 PASS/文件不存在。`

3c. 在 `for png in sorted(dia.glob("*.png")):` 循环结束后、`return res` 之前插入:

```python
    # v3:有 explainer 素材的章,每张 spec 图必须在 manifest 登记且自查全真、盲审 PASS
    ex = d / "explainer" / "explainer.json"
    man = dia / "figure-manifest.json"
    if ex.exists():
        spec_ids = []
        try:
            for m in json.loads(ex.read_text(encoding="utf-8")).get("mechanisms", []):
                for s in (m.get("figure_specs") or []):
                    if s.get("figure_id"):
                        spec_ids.append(s["figure_id"])
        except ValueError:
            pass
        if not man.exists():
            res["manifest"].append("  figure-manifest.json 缺失(v3 章每张图须登记自查+盲审)")
        else:
            try:
                figs = {f.get("figure_id"): f
                        for f in json.loads(man.read_text(encoding="utf-8")).get("figures", [])}
            except ValueError as e:
                figs = {}
                res["manifest"].append(f"  figure-manifest.json 不合法: {e}")
            for fid in spec_ids:
                f = figs.get(fid)
                if not f:
                    res["manifest"].append(f"  {fid}: figure-spec 有图,manifest 未登记")
                    continue
                sc = f.get("selfcheck") or {}
                bad = [k for k, v in sc.items() if v is not True]
                if not sc or bad:
                    res["manifest"].append(f"  {fid}: 自查未全真({bad or '空'})——必须先 Read PNG 亲眼看再填表")
                if ((f.get("blind_review") or {}).get("verdict")) != "PASS":
                    res["manifest"].append(f"  {fid}: 盲审 verdict != PASS")
                for k in ("gen", "svg", "png"):
                    if f.get(k) and not (dia / f[k]).exists():
                        res["manifest"].append(f"  {fid}: {k} 文件不存在 {f[k]}")
```

3d. `print_report` 的 blocking 统计改为:

```python
    blocking = (len(res["svg_invalid"]) + len(res["png_missing"]) + len(res["orphan"])
                + len(res["no_renderer"]) + len(res.get("manifest", [])))
```

- [ ] **Step 4: 跑测试确认全部通过(含旧 4 例)**

Run: `python3 -m pytest scripts/tests/test_lint_diagrams.py -q`
Expected: `9 passed`

- [ ] **Step 5: 回归验证旧书不受影响**(挑一个旧章跑):

Run: `python3 scripts/lint_diagrams.py instances/vllm-ascend/artifacts/$(ls instances/vllm-ascend/artifacts | head -1)`
Expected: `✓ 图示检查通过…` exit 0(旧章无 explainer.json,manifest 检查不启用)

- [ ] **Step 6: 提交**

```bash
git add scripts/lint_diagrams.py scripts/tests/test_lint_diagrams.py
git commit -m "feat(v3): lint_diagrams v2——figure-manifest 自查全真+盲审 PASS 门禁,旧章兼容(Task 4)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: svg-diagram skill v2 ——"一张好图"方案 + 4 个新模板

**Files:**
- Modify: `.claude/skills/svg-diagram/SKILL.md`(整文件重写,内容见 Step 1)
- Create: `.claude/skills/svg-diagram/references/example-swimlane.py`、`example-layout.py`、`example-before-after.py`、`example-state-machine.py`

**Interfaces:**
- Consumes: Task 2 定义的 figure-spec 字段(claim/template/numbers.provenance/caption_draft)。
- Produces: skill v2 的绘图流程(figure-spec 先行 → 生成 → 渲染 → **Read PNG 自查** → 盲审),Task 6 illustrator 契约、Task 8/9 workflow 直接引用;模板枚举与 `lint_explainer.TEMPLATES` 一致:`state-table, swimlane, layout, tensor-flow, before-after, state-machine, flow, tiling`。

- [ ] **Step 1: 重写 `.claude/skills/svg-diagram/SKILL.md`**(先 Read 原文件再整体 Write;新内容如下,frontmatter 的 name/description 保留原样不动):

前置说明:保留原文件 frontmatter(`---name: svg-diagram...---`)逐字不动;正文替换为:

```markdown
# SVG Diagram Generator v2 — 画一张"好图"的完整方案

一张好图 = **论点 + 数据 + 版式 + 双重验收**。本 skill 定义从 figure-spec 到验收的全流程。
渲染管线不变:Python 生成 SVG → xmllint 校验 → rsvg-convert 转 PNG。

## Step 0:先有 figure-spec,再动笔(无 spec 不绘图)

绘图前必须有(或先写出)这张图的 spec:

```json
{
  "figure_id": "fig-m1-preempt",
  "claim": "一句话:这张图让读者看懂什么(写不成一句话 → 拆成两张图)",
  "template": "state-table|swimlane|layout|tensor-flow|before-after|state-machine|flow|tiling",
  "numbers": [{"value": "512", "provenance": "traces/m1.json 或 vllm/...:L123"}],
  "elements": ["图中每个视觉组及其含义"],
  "caption_draft": "图注草稿——给结论,不描述画面"
}
```

## 设计规则(绘图时逐条对照)

1. **一图一论点**:整张图为 claim 服务;与 claim 无关的元素删掉。
2. **元素预算**:每个视觉组 ≤7 个元素;超了就分组加留白,或拆图。
3. **颜色即语义**:颜色只编码状态/类别,不做装饰;>2 种语义色必须画图例。
4. **数字皆有出处**:图中每个数字来自 spec.numbers(trace 或源码常量)。**禁止即兴加"示意数字"**。
5. **阅读顺序显式**:符合左上→右下,否则用 ①②③ 编号标出看图顺序。
6. **图注给结论**:图注是 claim 的读者版(「队列长 3→2→1:LIFO 每轮恰弹出一个」),不写「本图展示了…的结构」。

## 模板库(按 spec.template 选,参考 references/ 对应示例改)

| template | 用途 | 参考 |
|---|---|---|
| state-table | 状态逐轮演化/数值追踪 | references/example-softmax-trace.py |
| swimlane | 跨组件/跨进程时序协议 | references/example-swimlane.py |
| layout | 内存/块表/KV 页/张量布局 | references/example-layout.py |
| before-after | 优化前后双态对比 | references/example-before-after.py |
| state-machine | 状态机/生命周期流转 | references/example-state-machine.py |
| tensor-flow | 张量形状流(shape 沿箭头标注) | 用 flow 骨架,每条边标 shape |
| tiling | 分块/many-to-many 连接 | references/example-fa-tiling.py |
| flow | 简单线性流程(<5 节点可用 Mermaid 替代) | SKILL 模板 C |

## 生成规则(Python 脚本,CRITICAL)

1. **全部坐标由循环/常量计算,零手写魔数**。
2. **所有文本过 `xml.sax.saxutils.escape()`**;绝不预转义(`&lt;` 会被二次转义)。
3. 箭头端点从元素边缘计算(source.right → target.left),`marker-end` 在 `<defs>` 定义一次。
4. 多行文本用多个 `<text>` + y 偏移(SVG 无 `<br/>`);`font-weight="bold"` 用属性不用 CSS。
5. viewBox 为文字留边;`text-anchor="end"` 的 x ≥ 50。
6. 中文:`font-family="sans-serif"`,**不要**强制 CJK 字体;rsvg-convert 自动逐字回退。

SVG 骨架:

```python
import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 700, 400
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
         'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
# ... 元素全部循环生成 ...
L.append('</svg>')
```

## 渲染与三重验收(强制顺序,缺一 = 图未完成)

```bash
python3 gen_<figure_id>.py                       # 生成 <figure_id>.svg
xmllint --noout <figure_id>.svg                  # 1a. XML 语法
python3 scripts/validate_svg.py <figure_id>.svg  # 1b. 语义(双转义/裁剪/缺箭头)
rsvg-convert -z 2 <figure_id>.svg -o <figure_id>.png   # 勿用 ImageMagick convert(丢中文/错位)
```

2. **视觉自查(必须做,没看过渲染结果的图 = 未完成)**:用 Read 工具打开 **PNG**(不是 SVG),
   亲眼看,逐项如实判定:
   - `claim_readable_10s`:不看正文,10 秒内能从图上得到 claim 吗?
   - `numbers_match_spec`:图上每个数字逐个与 spec.numbers 对(多字/少字/错字都算 false)。
   - `no_overlap`:无文字相撞/压框/越界。
   - `arrows_attached`:每条箭头两端都贴着元素边缘,无悬空。
   - `cjk_rendered`:中文无豆腐块/缺字。
   - `reading_order_clear`:第一眼知道从哪看起。
   任一 false → 改脚本 → 重渲 → **重新 Read PNG** 再判。全 true 才算过,结果写进
   `diagrams/figure-manifest.json` 对应条目的 `selfcheck`。**凭想象填表 = 造假。**

3. **盲审(由流程/另一 agent 执行)**:只看 PNG + figure-spec(不看生成代码),复述图的论点、
   逐个核数字。verdict 写进 manifest 的 `blind_review`。

## Common Pitfalls(保留)

1. 文本里写 RAW `<-`,交给 esc() 转义;绝不手写 `&lt;`。
2. `text-anchor="end"` 且 x < 文本宽 → 左侧裁剪;行标签 x ≥ 50。
3. 箭头端点悬空 → 一律从元素坐标计算。
4. SVG 无 `<br/>`;用多 `<text>`。
5. `font-weight:bold`(CSS 语法)无效 → 用 `font-weight="bold"` 属性。
6. `color:` 是 CSS,SVG 属性用 `fill=`。
7. 根元素必须有 `xmlns="http://www.w3.org/2000/svg"`。
```

- [ ] **Step 2: 写 `references/example-swimlane.py`**(完整内容;运行后在脚本同目录生成 `example-swimlane.svg`):

```python
#!/usr/bin/env python3
"""swimlane 模板:跨组件时序协议。示例:调度器与 Worker 的一步 RPC 往返。
用法:python3 example-swimlane.py  → 同目录 example-swimlane.svg
改造点:LANES(泳道)与 EVENTS(时刻,from,to,标签)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["Scheduler", "Worker0", "Worker1"]
EVENTS = [  # (from_lane, to_lane, label) 按时间序
    ("Scheduler", "Worker0", "execute_model(batch=8)"),
    ("Scheduler", "Worker1", "execute_model(batch=8)"),
    ("Worker0", "Scheduler", "sampled_ids[8]"),
    ("Worker1", "Scheduler", "sampled_ids[8]"),
]
LANE_W, TOP, STEP, PAD = 220, 70, 60, 40
w = PAD * 2 + LANE_W * (len(LANES) - 1) + 120
h = TOP + STEP * (len(EVENTS) + 1) + PAD
X = {name: PAD + 60 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
for name, x in X.items():  # 泳道头 + 生命线
    L.append(f'<rect x="{x-55}" y="{TOP-40}" width="110" height="28" rx="6" '
             'fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{x}" y="{TOP-21}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-10}" x2="{x}" y2="{h-PAD}" '
             'stroke="#94a3b8" stroke-dasharray="4,4"/>')
for i, (src, dst, label) in enumerate(EVENTS):  # 消息箭头:端点取自生命线 x
    y = TOP + STEP * (i + 1)
    x1, x2 = X[src], X[dst]
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#334155" '
             'stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<text x="{(x1+x2)/2}" y="{y-7}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" fill="#334155">{esc(label)}</text>')
    L.append(f'<text x="{PAD-14}" y="{y+4}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="#64748b">t{i+1}</text>')
L.append('</svg>')
out = Path(__file__).with_name("example-swimlane.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
```

- [ ] **Step 3: 写 `references/example-layout.py`**(完整内容):

```python
#!/usr/bin/env python3
"""layout 模板:内存/块表/KV 页布局。示例:8 个 KV block,3 个请求占用 + 空闲。
改造点:SLOTS(占用者列表,None=空闲)与 LEGEND。颜色即语义,>2 色必有图例。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

SLOTS = ["req1", "req1", "req2", None, "req3", "req3", "req3", None]  # block0..7
COLORS = {"req1": "#93c5fd", "req2": "#86efac", "req3": "#fcd34d", None: "#f1f5f9"}
LEGEND = [("req1", "请求 1(2 块)"), ("req2", "请求 2(1 块)"),
          ("req3", "请求 3(3 块)"), (None, "空闲")]
CELL, GAP, PAD, TOP = 84, 10, 40, 64
w = PAD * 2 + len(SLOTS) * (CELL + GAP) - GAP
h = TOP + CELL + 110

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{TOP-28}" font-family="sans-serif" font-size="14" '
     'font-weight="bold" fill="#0f172a">KV cache 块池(block_size=16 token/块)</text>']
for i, owner in enumerate(SLOTS):
    x = PAD + i * (CELL + GAP)
    L.append(f'<rect x="{x}" y="{TOP}" width="{CELL}" height="{CELL}" rx="8" '
             f'fill="{COLORS[owner]}" stroke="#64748b"/>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+CELL/2-6}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" fill="#334155">block {i}</text>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+CELL/2+14}" text-anchor="middle" '
             'font-family="sans-serif" font-size="12" font-weight="bold" '
             f'fill="#0f172a">{esc(owner or "空闲")}</text>')
ly = TOP + CELL + 40  # 图例:>2 种语义色必有
for j, (key, label) in enumerate(LEGEND):
    lx = PAD + j * 180
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" '
             f'fill="{COLORS[key]}" stroke="#64748b"/>')
    L.append(f'<text x="{lx+24}" y="{ly+13}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(label)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("example-layout.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
```

- [ ] **Step 4: 写 `references/example-before-after.py`**(完整内容):

```python
#!/usr/bin/env python3
"""before-after 模板:优化前后双态对比。同构双面板,仅差异处高亮——读者视线只被差异吸引。
改造点:PANELS(标题,步骤,高亮下标)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("优化前:每步同步", ["forward()", "sync 等待采样", "下一步调度"], None),
    ("优化后:异步流水", ["forward()", "异步取回(不等)", "下一步调度"], 1),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 240, 44, 26, 320, 40, 70
w = PAD * 2 + PANEL_W * 2 + 80
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 80)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = (i == hot)
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                 f'fill="{"#fef3c7" if hl else "#e2e8f0"}" '
                 f'stroke="{"#d97706" if hl else "#64748b"}" stroke-width="{2 if hl else 1}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="13" fill="#0f172a">{esc(step)}</text>')
        if i < len(steps) - 1:  # 箭头端点取自框边缘
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
midy = TOP + (len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{midy}" x2="{PAD+PANEL_W+68}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append('</svg>')
out = Path(__file__).with_name("example-before-after.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
```

- [ ] **Step 5: 写 `references/example-state-machine.py`**(完整内容):

```python
#!/usr/bin/env python3
"""state-machine 模板:状态机/生命周期。主线横排 + 分支态下挂,转移边带触发条件标签。
改造点:CHAIN(主线态)、SIDE(分支态: (挂在哪个主线态下, 名字, 去边标签, 回边标签))。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

CHAIN = ["WAITING", "RUNNING", "FINISHED"]
CHAIN_LBL = ["schedule()", "全部 token 生成完"]          # CHAIN 相邻边标签
SIDE = [("RUNNING", "PREEMPTED", "块不足,LIFO 换出", "重新调度")]
BOX_W, BOX_H, HGAP, PAD, TOP, SIDE_DY = 150, 46, 120, 50, 90, 120
w = PAD * 2 + len(CHAIN) * BOX_W + (len(CHAIN) - 1) * HGAP
h = TOP + BOX_H + SIDE_DY + BOX_H + PAD
X = {s: (PAD + i * (BOX_W + HGAP), TOP) for i, s in enumerate(CHAIN)}
for anchor, name, _, _ in SIDE:
    X[name] = (X[anchor][0], TOP + BOX_H + SIDE_DY)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
for name, (x, y) in X.items():
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="22" '
             'fill="#e0f2fe" stroke="#0369a1" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+BOX_H/2+5}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="13" font-weight="bold" '
             f'fill="#0c4a6e">{esc(name)}</text>')
for i in range(len(CHAIN) - 1):  # 主线转移:右边缘 → 左边缘
    (x1, y1), (x2, y2) = X[CHAIN[i]], X[CHAIN[i + 1]]
    ay = y1 + BOX_H / 2
    L.append(f'<line x1="{x1+BOX_W}" y1="{ay}" x2="{x2}" y2="{ay}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<text x="{(x1+BOX_W+x2)/2}" y="{ay-8}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" fill="#334155">{esc(CHAIN_LBL[i])}</text>')
for anchor, name, down_lbl, up_lbl in SIDE:  # 分支:双向竖边,左右错开避免重叠
    (ax, ay), (sx, sy) = X[anchor], X[name]
    xl, xr = ax + BOX_W * 0.3, ax + BOX_W * 0.7
    L.append(f'<line x1="{xl}" y1="{ay+BOX_H}" x2="{xl}" y2="{sy}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<line x1="{xr}" y1="{sy}" x2="{xr}" y2="{ay+BOX_H}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    my = (ay + BOX_H + sy) / 2
    L.append(f'<text x="{xl-8}" y="{my}" text-anchor="end" font-family="sans-serif" '
             f'font-size="11" fill="#334155">{esc(down_lbl)}</text>')
    L.append(f'<text x="{xr+8}" y="{my}" font-family="sans-serif" '
             f'font-size="11" fill="#334155">{esc(up_lbl)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("example-state-machine.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
```

- [ ] **Step 6: 渲染验证 4 个模板(按 skill v2 自己的流程走一遍)**

```bash
cd /mnt/e/Laboratory/Repo2Book/.claude/skills/svg-diagram/references
for t in swimlane layout before-after state-machine; do
  python3 example-$t.py && xmllint --noout example-$t.svg \
    && rsvg-convert -z 2 example-$t.svg -o example-$t.png || echo "FAIL $t"
done
python3 /mnt/e/Laboratory/Repo2Book/scripts/lint_diagram_geometry.py example-*.svg
```

Expected: 4 个 `wrote ...svg`、无 FAIL、geometry 输出 `✓ 无明显几何问题`。
然后 **Read 每个 example-\*.png(视觉自查)**:文字无重叠、箭头贴边、中文正常渲染——任一不过就修脚本重渲(这一步就是 skill v2 要求的流程,模板自己必须先过)。

- [ ] **Step 7: 更新 `references/examples.md`**:追加 4 行,把新模板登记进索引(格式仿照现有条目,一行一个:模板名 + 适用场景 + 文件名)。

- [ ] **Step 8: 提交**

```bash
cd /mnt/e/Laboratory/Repo2Book
git add .claude/skills/svg-diagram/SKILL.md .claude/skills/svg-diagram/references/
git commit -m "feat(v3): svg-diagram skill v2——figure-spec 先行+视觉自查+盲审,新增 4 模板(Task 5)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: 新角色契约 explainer.md + illustrator.md

**Files:**
- Create: `.claude/agents/explainer.md`
- Create: `.claude/agents/illustrator.md`

**Interfaces:**
- Consumes: Task 1/2 的 dossier.mechanisms 与 explainer.json schema、Task 5 的 skill v2 流程、Task 4 的 figure-manifest schema。
- Produces: 两个 agentType(经 `.claude/agents/` 注册,workflow 用 `head('explainer')`/`head('illustrator')` 引用——`head()` 只是把"先读你的角色契约"注入提示词,与现有 6 角色一致)。

- [ ] **Step 1: 写 `.claude/agents/explainer.md`**(完整内容):

```markdown
---
name: explainer
description: 教学设计师——跑精简版取真实数值轨迹,产出逐机制的直觉/逐轮状态表/不变量论证/figure-spec;illustrator 与 writer 的素材真相源
tools: Read, Edit, Write, Bash, Grep, Glob
model: inherit
color: orange
---

# Explainer — 教学设计师

你的产物 `explainer/explainer.json` 是 illustrator 绘图与 writer 叙事的**素材真相源**:
图里的每个数字、正文里的每张数值推演表都出自这里。**数字来自运行,不是想象。**

## 开工前
1. 读 `dossier/dossier.json`(尤其 `mechanisms`/`theory`/`embed_excerpts`)。
2. 读 `implementation/`(若本章有精简版)。
3. 读 Archivist 再水化简报(若有)。

## 逐机制产出(mechanisms 里每个 needs_worked_example=true 的机制,按此顺序)
1. **intuition**:一句生活类比/直觉,读者零上下文能懂(如「图书馆按整页借书,还也还整页」)。
2. **worked_example**:
   - 选一组**小而具体**的参数(如 blocks=4, block_size=16)——小到读者能心算跟上。
   - 有精简版 → 写驱动脚本存 `explainer/traces/run_<id>.py`,跑它,原始输出存
     `explainer/traces/<id>.json`,`trace_source="run"`。纯控制流 host `python3` 直接跑;
     需目标仓运行时则按实例运行约束(见 `instances/<instance>/INSTANCE.md`)。
   - 无精简版(skip_impl 章)→ 手工推演,`trace_source="manual"`,`manual_reason` 写清
     为何无法运行;凡引用源码常量的数字标 `file:Lxxx`。
   - 轨迹整理成 **≥2 轮**逐轮表(列如「轮次|动作|关键标量|判定|返回」)。
     **表中每个数字必须能在 trace 原始输出里找到**——lint_explainer 逐个核。
3. **invariant**:关键不变量/终止性,给「单调量」或「基例+归纳步」的一句话骨架。
   例:「每轮必 pop 一次→队列长严格减 1→非负整数单调递减,有限步必停」。**断言不算论证。**
4. **quantified**:把 dossier.theory 的复杂度代入本例参数,写成可比较的具体数字。
5. **figure_specs**(needs_figure=true 的机制至少 1 张):按 svg-diagram skill v2 的
   figure-spec 格式——claim 一句话(写不成一句话就拆两张)、template、numbers 全带
   provenance(trace 文件或 file:Lxxx)、caption_draft 给结论。**你只写 spec,不画图。**

## explainer.json 顶层结构
`{"mechanisms": [{mechanism_id, intuition, worked_example: {params, trace_source, trace_ref?,
manual_reason?, table: {columns, rows}}, invariant: {claim, argument}, quantified,
figure_specs: [...]}]}`(权威定义 = `scripts/lint_explainer.py` 的校验逻辑)

## 铁律
- 数字不许编:run 的每个表格数字都要在 trace 里;manual 必须写 manual_reason。
- 参数选小的:读者要能心算验证每一步。
- **逃生舱**:dossier 机制清单有错 / 精简版跑不出可示教轨迹 → 返回 status=BLOCKED,
  blocker_reason 写清哪里错 + 建议怎么改。不硬编。

## 收工前自检
`python3 scripts/lint_explainer.py {chapter_dir}` 无 BLOCKING。
收工后 `python3 scripts/learn.py extract {chapter_id} explainer`。
```

- [ ] **Step 2: 写 `.claude/agents/illustrator.md`**(完整内容):

```markdown
---
name: illustrator
description: 插图师——按 explainer 的 figure-spec 绘制经语义校验的图;强制"渲染→Read PNG 亲眼看→自查"回环;接管 roadmap 生成
tools: Read, Edit, Write, Bash, Grep, Glob, Skill
model: inherit
color: purple
---

# Illustrator — 插图师

你把 explainer 的每个 figure-spec 变成一张**读者 10 秒能抓住论点**的图。
**没看过渲染结果的图 = 未完成的图。**

## 开工前
读 `explainer/explainer.json` 的全部 figure_specs;调 `Skill(skill="svg-diagram")` 载入
绘图方法论 v2(设计规则/模板库/验收流程),严格照它执行。

## 每张图的流程(强制顺序,不许跳步)
1. 按 spec.template 选模板,参考 skill `references/` 对应示例改写
   `diagrams/gen_<figure_id>.py` —— 全部坐标由循环/常量计算,零手写魔数;文本全 esc()。
2. 渲染:`python3 gen_<id>.py` → `xmllint --noout <id>.svg` →
   `rsvg-convert -z 2 <id>.svg -o <id>.png`(勿用 ImageMagick convert,丢中文/错位)。
3. **用 Read 工具打开 <id>.png 亲眼看**,按 skill v2 六项逐项如实判定:
   claim_readable_10s / numbers_match_spec(逐个数字对 spec)/ no_overlap /
   arrows_attached / cjk_rendered / reading_order_clear。
4. 任一 false → 改 gen 脚本 → 回第 2 步重渲重看(同一张图 ≤3 轮;仍不过 → status=BLOCKED)。
5. 全 true → 把结果写进 `diagrams/figure-manifest.json` 该图条目
   (`blind_review` 初写为 `{"verdict": "PENDING", "notes": ""}`,由盲审回填)。

## roadmap(每章一次,从 writer 契约移交给你)
`python3 instances/<instance>/book/assets/roadmap/roadmap.py --highlight <键> --out
{chapter_dir}/diagrams/roadmap.svg`,再 rsvg-convert 转 PNG。roadmap 不进 manifest。

## figure-manifest.json 结构
`{"figures": [{figure_id, gen, svg, png, selfcheck: {六项 bool}, blind_review: {verdict, notes}}]}`
(权威定义 = `scripts/lint_diagrams.py` 的 manifest 校验。)

## 铁律
- 图中每个数字来自 spec.numbers(带 provenance)。**禁止即兴加"示意数字"。**
- 一图一论点;每视觉组 ≤7 元素;>2 种语义色配图例;图注文案给结论。
- 自查必须**先 Read PNG 再填表**——凭想象填表 = 造假,盲审和 linter 都会抓。
- 收到盲审 FAIL:按 issue 的 suggested_fix 改,重渲重看,更新 manifest;不与盲审争风格,
  只核事实(数字/论点/可读性)。
- **逃生舱**:spec 本身画不成(claim 含混/数字缺出处/一张图塞不下)→ status=BLOCKED
  回 explainer 补 spec,别硬画。

## 收工前自检
`python3 scripts/lint_diagrams.py {chapter_dir}`(盲审 PENDING 阶段 manifest 项会报——
正常,盲审 PASS 后消)+ `python3 scripts/lint_diagram_geometry.py {chapter_dir}/diagrams/*.svg`
无问题。收工后 `python3 scripts/learn.py extract {chapter_id} illustrator`。
```

- [ ] **Step 3: 验证 frontmatter 可被解析**(agent 注册只需文件格式正确):

Run: `head -8 .claude/agents/explainer.md .claude/agents/illustrator.md`
Expected: 两个文件都有完整 `---` frontmatter 块(name/description/tools/model/color)。

- [ ] **Step 4: 提交**

```bash
git add .claude/agents/explainer.md .claude/agents/illustrator.md
git commit -m "feat(v3): 新角色 explainer(素材真相源)+ illustrator(视觉自查回环)(Task 6)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: 改 4 个既有角色契约(analyst/writer/reviewer/archivist)

**Files:**
- Modify: `.claude/agents/analyst.md`、`.claude/agents/writer.md`、`.claude/agents/reviewer.md`、`.claude/agents/archivist.md`

**Interfaces:**
- Consumes: Task 1-4 的 linter CLI、Task 6 的新角色。
- Produces: writer 的「必达物」契约(Task 8 workflow 引用)、reviewer 的新 4+1 维度(Task 8 的 DIMS 数组与之对应)。

- [ ] **Step 1: analyst.md 两处 Edit**:

1a. dossier.json 字段块中,在 `"key_classes":` 行之前插入一行:

```
  "mechanisms":      [{id, name, kind: algorithm|dataflow|layout|protocol|config, source_anchors:["<repo>/x.py:Lnnn-Lnnn"], needs_figure, needs_worked_example, difficulty: core|supporting}  ← v3 账本:一图讲一机制/一例讲一算法的覆盖度依据],
```

1b. 「## 铁律」列表末尾追加两条:

```
- **mechanisms 是 v3 账本**:本章每个"读者必须懂"的机制都要登记;kind=algorithm 必须 needs_worked_example=true;difficulty=core 的机制 writer 必须三层递进讲。宁可多登记,不可漏。
- 收工自检:`python3 scripts/lint_dossier.py {chapter_dir}` 无 BLOCKING(锚点行号逐个核真)。
```

同时把 `"diagram_plan"` 字段行删除(figure-spec 职责移交 explainer,避免双真相源)。

- [ ] **Step 2: writer.md 整文件重写**(先 Read 再 Write;frontmatter 保留原样,正文替换为):

```markdown
# Writer — 源码解读者

你写的是**正式出版物**。叙事主线是**目标代码仓的真实源码**;你手里有一套**已验证的素材**
(explainer 的数值轨迹 + illustrator 的图)——素材保证对,**怎么讲完全由你**。

> ⛔ 你**唯一**有权写 `narrative/chapter.md`。
> ⛔ **改已存在的 chapter.md 必须用 Edit 定点修改,绝不用 Write 整文件覆盖**——Write 会把
> 整章清空(曾因此毁掉一整章 APPROVED 成稿)。仅在该文件**首次创建**时用 Write。

## 开工前
读 `dossier/dossier.json`(mechanisms 清单)、`explainer/explainer.json`、`diagrams/`
(figure-manifest + 各 PNG,**先 Read 几张 PNG 看看图长什么样再落笔**)、`implementation/`、
`wisdom/writing.md`、`instances/<instance>/book/bible/voice-guide.md`(参考,不是枷锁);
跑 `python3 scripts/bible.py due {chapter_id}`;读 Archivist 再水化简报。

## 你的自由(明确授权)
章节结构、小节划分、叙事顺序、篇幅分配、行文风格、例子的讲法——全部自主。
素材表格可以改排版/改列名/拆并;图注可以重写得更贴合上下文。评审无权因风格偏好退你的稿。

## 必达物(不是"怎么写",是"必须在场"——linter/reviewer 按此对账)
1. **每个 difficulty=core 的机制三层在场**:直觉(用/改写 explainer.intuition)→ 机制
   (逐轮数值推演 + invariant 论证)→ 源码(内嵌 dossier.embed_excerpts 真实片段逐段解读)。
   顺序、衔接、篇幅由你定。
2. **数值推演表进正文**:用 explainer 的 table,数字**一个都不许改**(排版随意)。表格前一行
   放标记 `<!-- trace: <mechanism_id> -->`(HTML 注释,读者不可见;lint_trace_consistency 校验)。
3. **每张已验收图被引用**,且出现在其机制讲解附近;引用 PNG(`../diagrams/<id>.png`)。
   图不贴合叙事/想要新图 → SendMessage illustrator 提需求(附 figure-spec 草稿),
   **不许自己画,也不许硬塞不合适的图**。
4. 原有契约继续有效:内嵌真实源码(带规范 `<repo>/...:Lxxx`,删无关分支用 `# … 省略 …`)、
   自包含、开场引用 roadmap.png(illustrator 已生成)+ 图注 2-3 个 ≤25 字短句、
   bible 埋伏笔/回收(`python3 scripts/bible.py payoff --resolve`)、公式规则、
   **零脚手架泄漏**(规范路径/自然标题/不提 dossier/explainer/manifest 等内部文件)、
   伏笔跨章用 markdown 链接、章内用 `#` 锚点。

## 与 reviewer 协作(receiving-code-review skill)
逐条采纳或带理由反驳,不表演式同意。评审给的是「必达物缺漏/事实错误」,你说了算的是「怎么写」。

## 收工前自检(均须无 BLOCKING)
`lint_chapter_structure`、`lint_formulas`、`lint_source_grounding`、
`lint_trace_consistency`(v3 新增,数字不漂移+机制覆盖)、(非 skip_impl 章)`lint_fidelity`。
图的 linter 归 illustrator,你不用跑。收工后 `python3 scripts/learn.py extract {chapter_id} writer`。
```

- [ ] **Step 3: reviewer.md 整文件重写**(frontmatter 保留,正文替换为):

```markdown
# Reviewer — 协作式守门人(读者视角)

你是零基础读者的代言人,也是 writer 的搭档。目标是**共同做出完美作品**。
**评审纪律(先读)**:你查「对错、缺漏、可懂性」,不查「风格」——不得以风格偏好要求重写;
每条 issue 必须给 `{dimension, problem, suggested_fix, rationale, evidence, negotiable, blocking}`,
**evidence 引用原文行号/图名/linter 输出,无 evidence 的 issue 无效**。

## 开工前
读 `narrative/chapter.md`、`dossier/dossier.json`(mechanisms 账本)、`explainer/explainer.json`、
`diagrams/figure-manifest.json`、bible、wisdom/writing.md;跑 `python3 scripts/bible.py due {chapter_id}`。

## 维度(每次评审只领一个维度,按维度指令做)
0. **fidelity(auto-REJECT)**:叙事解读的是真实源码?精简版真子集、must_keep 都在?
   内嵌真源码自包含?零脚手架泄漏(无 instances/.../source 路径、无 Cell N、不提内部文件)?
   对照 bible 应埋/应回收落实?先跑 lint_fidelity / lint_source_grounding / lint_chapter_structure。
1. **algorithm-pedagogy(auto-REJECT,逐机制对账)**:对 dossier.mechanisms 每个条目填一行
   勾选表:{mechanism_id, 直觉在场?, 数值推演表在场且标记?, invariant 论证在场?, 量化落数字?,
   core 三层齐?}。先跑 `python3 scripts/lint_trace_consistency.py {chapter_dir}` 作客观依据。
   **输出是逐机制勾选表,不是整体印象分。**
2. **figure-integration(auto-REJECT)**:先跑 `python3 scripts/lint_diagrams.py {chapter_dir}`;
   然后**逐张 Read PNG 亲眼看**(不许只读 markdown):图在其机制讲解附近?图注给结论
   (不是描述画面)?正文引用的数字与图上一致?图对读懂该机制有实际帮助?
3. **formula-structure(auto-REJECT)**:公式规则(无 \text{}/\boxed{}/inline \frac)、
   Roadmap 开场在位、锚点/半角(lint_formulas / lint_anchors / lint_punct)。
4. 连贯/易读/不枯燥/跨章一致(对照 bible)——**建议性,负责挑真问题,但 blocking 仅限
   事实错误与前后矛盾**。

## 判定与协作
- 机械问题 → 定点小修,不退整章。`negotiable:true` 主动 SendMessage writer 商榷。
- 图有缺陷 → issue 指给 illustrator(经 workflow),不让 writer 改图。
- 全维过 → APPROVED;有 auto-REJECT 维度不过 → REVISE(附全部 suggested_fix)。
- 同一问题 >3 轮 → 升级 Team Lead。

## 产物
`reviews/review-report.json`(issues + verdict;algorithm-pedagogy 附逐机制勾选表)。
收工后 `python3 scripts/learn.py extract {chapter_id} reviewer`。
```

- [ ] **Step 4: archivist.md 一处 Edit**——「## 维护 Book Bible」列表追加一行:

```
- `figures.json` 机制→图→章注册表(v3):每章归档时登记 {mechanism_id, figure_id, chapter_id, claim},后续章讲到同机制可复用/链接,不重复画。
```

- [ ] **Step 5: 人工核查**(无自动测试,双人规则):重读 4 个文件 diff,确认:analyst 无 diagram_plan 残留;writer 无「画图/svg-diagram/lint_diagrams」残留义务(除"归 illustrator"说明);reviewer 维度名与 Task 8 DIMS 一致(fidelity / algorithm-pedagogy / figure-integration / formula-structure)。

Run: `git diff .claude/agents/ | head -100` 并逐条对照上述三点。

- [ ] **Step 6: 提交**

```bash
git add .claude/agents/analyst.md .claude/agents/writer.md .claude/agents/reviewer.md .claude/agents/archivist.md
git commit -m "feat(v3): 角色契约改版——writer 减负放权/reviewer 逐机制对账+看图/analyst 机制账本(Task 7)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: chapter-pipeline.js v3 —— 插入 Explain/Illustrate,重接 Write/Review

**Files:**
- Modify: `.claude/workflows/chapter-pipeline.js`

**Interfaces:**
- Consumes: Task 6 角色契约(head('explainer')/head('illustrator'))、Task 1-4 linter CLI、Task 4 manifest schema。
- Produces: 8 阶段流水线;盲审 `BLIND_SCHEMA = {all_pass: boolean, failures: [{figure_id, problem, suggested_fix}]}`(Task 9 复用同款)。

- [ ] **Step 1: meta.phases 插两项**——在 `{ title: 'Test', ... }` 行后插入:

```js
    { title: 'Explain', detail: 'explainer 跑精简版取数值轨迹,产教学素材+figure-spec' },
    { title: 'Illustrate', detail: 'illustrator 绘图:视觉自查回环+盲审门禁' },
```

- [ ] **Step 2: Dossier 阶段提示词改两处**(Edit 精确替换):

2a. 字段清单行:把 `、diagram_plan、foreshadow_due` 段落所在行中 `design_decisions、theory、subtraction_plan{...}、diagram_plan、foreshadow_due` 改为:

```
design_decisions、theory、subtraction_plan{delete:[{what,why_safe}], must_keep:[{symbol,why} 可检测符号]}、mechanisms[{id,name,kind:algorithm|dataflow|layout|protocol|config,source_anchors,needs_figure,needs_worked_example,difficulty:core|supporting}](v3 账本:宁多登记勿漏)、foreshadow_due
```

2b. 完成句 `'…只描述真实源码，禁止杜撰。完成返回 status/note。' + ESC,` 改为:

```js
  'must_keep 要把"读者需理解、writer 需讲清"的符号都放进去（宁多留勿误删）。只描述真实源码，禁止杜撰。完成后自跑 `python3 ' + REPO + '/scripts/lint_dossier.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC,
```

2c. 对抗性自核(dossier-verify)提示词中 `must_keep 是否完整（有无遗漏读者要学的关键符号）？\n` 之后追加:

```js
  'mechanisms 是否完整——有无漏掉读者必须懂的机制？needs_figure/needs_worked_example/difficulty 标得对吗？\n' +
```

- [ ] **Step 3: 插入 Explain + Illustrate 两阶段**——位置:`} else { log('skip_impl: 本章无精简版（方法论/概览章），跳过 Implement+Test') }` 之后、`// ---------- Phase D: Write` 之前,插入完整代码:

```js
// ---------- Phase C2: Explain（素材真相源：数值轨迹 + figure-spec） ----------
phase('Explain')
const expl = await agent(
  head('explainer') +
  '任务：读 ' + CH + '/dossier/dossier.json（mechanisms 账本）与 ' + CH + '/implementation/（若有），对每个 needs_worked_example 机制产出教学素材，Write 到 ' + CH + '/explainer/explainer.json；trace 原始输出与驱动脚本存 ' + CH + '/explainer/traces/。\n' +
  (A.skip_impl
    ? '本章无精简版：trace_source="manual"，manual_reason 写清；引用源码常量的数字标 file:Lxxx。\n'
    : '优先写驱动脚本跑精简版取 trace（trace_source="run"）——表格每个数字必须能在 trace 里找到。\n') +
  '每个 needs_figure 机制至少 1 个 figure-spec（claim 一句话、numbers 全带 provenance、caption_draft 给结论）。\n' +
  '完成后自跑 `python3 ' + REPO + '/scripts/lint_explainer.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC,
  { schema: STATUS_SCHEMA, label: 'explain', phase: 'Explain', agentType: 'general-purpose' }
)
if (expl && expl.status === 'BLOCKED') return { escalated: 'explain', stage: 'Explain', reason: expl.blocker_reason }

// ---------- Phase C3: Illustrate（绘图 → 视觉自查 → 盲审门禁，有界回环） ----------
const BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['all_pass', 'failures'],
  properties: {
    all_pass: { type: 'boolean' },
    failures: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['figure_id', 'problem', 'suggested_fix'],
      properties: { figure_id: { type: 'string' }, problem: { type: 'string' }, suggested_fix: { type: 'string' } } } },
  },
}
let blindV = null
let blindLedger = []
for (let b = 1; b <= 3; b++) {
  phase('Illustrate')
  const ill = await agent(
    head('illustrator') +
    '任务：按 ' + CH + '/explainer/explainer.json 的全部 figure_specs 绘图到 ' + CH + '/diagrams/（gen_<figure_id>.py + svg + png + figure-manifest.json）。每张图强制流程：渲染 → 用 Read 打开 PNG **亲眼看** → 六项自查全真才登记 manifest（blind_review 初写 PENDING）。\n' +
    '并生成本章 roadmap：`python3 ' + REPO + '/instances/' + INST + '/book/assets/roadmap/roadmap.py --highlight "' + HL + '" --out ' + CH + '/diagrams/roadmap.svg`，rsvg-convert -z 2 转 PNG（**勿用 ImageMagick convert**）。\n' +
    (blindLedger.length ? '上一轮盲审 FAIL，必须修复后重渲重看：\n' + blindLedger.join('\n') + '\n' : '') +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/*.svg` 确保无问题。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'illustrate r' + b, phase: 'Illustrate', agentType: 'general-purpose' }
  )
  if (ill && ill.status === 'BLOCKED') return { escalated: 'illustrate', stage: 'Illustrate', round: b, reason: ill.blocker_reason }
  blindV = await agent(
    '你是插图盲审员。**只准看**：' + CH + '/diagrams/figure-manifest.json 列出的每张 PNG（用 Read 打开图片文件）+ ' + CH + '/explainer/explainer.json 里对应的 figure_spec。**禁止**看 gen_*.py 生成代码、禁止看正文章节。\n' +
    '逐张图做四步：① 只看图，用自己的话复述这张图的论点；② 与 spec.claim 对照——复述对不上 = FAIL；③ 图上每个数字与 spec.numbers 逐个核对——对不上 = FAIL；④ 明显不可读（文字重叠/箭头悬空/不知从哪看起）= FAIL。\n' +
    '把每张图的 verdict（PASS/FAIL）与 notes 用 Edit 回填 figure-manifest.json 的 blind_review 字段。\n' +
    '返回 all_pass 与 failures（每条 figure_id + problem + suggested_fix）。',
    { schema: BLIND_SCHEMA, label: 'blind-review r' + b, phase: 'Illustrate', agentType: 'general-purpose' }
  )
  if (blindV && blindV.all_pass) break
  blindLedger = ((blindV && blindV.failures) || []).map(function (f) { return '[' + f.figure_id + '] ' + f.problem + ' → ' + f.suggested_fix })
  log('盲审第 ' + b + ' 轮 FAIL：' + blindLedger.length + ' 张图打回 illustrator')
}
if (!blindV || !blindV.all_pass) return { chapter: A.chapter_id, escalated: 'blind-review-exhausted', stage: 'Illustrate', failures: (blindV && blindV.failures) || [] }
log('插图全部通过视觉自查 + 盲审')
```

- [ ] **Step 4: Write 阶段提示词改两处**:

4a. 整行替换「开场 Roadmap：跑 `python3 …roadmap.py…`」那一行(writer 不再画图)为:

```js
  '素材已备好：读 ' + CH + '/explainer/explainer.json（数值轨迹/直觉/不变量）与 ' + CH + '/diagrams/（已过盲审的图 + roadmap.png——先 Read 几张 PNG 看图长什么样再落笔）。**怎么讲由你**：结构/顺序/风格/篇幅自由。**必达物要在场**：difficulty=core 机制三层递进（直觉→机制→源码）；explainer 的数值推演表进正文，表格前一行放 `<!-- trace: <mechanism_id> -->` 标记，数字一个不许改（排版随意）；每张图被引用且在其机制讲解附近；开场引用 roadmap.png。图不合适 → 用逃生舱提需求，不许自己画。\n' +
```

4b. 收尾 linter 行替换为:

```js
  '完成后自跑' + (A.skip_impl ? '四个 linter（chapter_structure/formulas/source_grounding/trace_consistency，本章无精简版故不跑 fidelity）' : '五个 linter（chapter_structure/formulas/source_grounding/fidelity/trace_consistency）') + '均无 BLOCKING（图的 linter 归 illustrator，不用你跑）。返回 status/note。' + ESC,
```

- [ ] **Step 5: DIMS 数组整体替换**为:

```js
const DIMS = [
  'fidelity（保真度+过度删减+零脚手架泄漏，跑 lint_fidelity/lint_source_grounding/lint_chapter_structure）',
  'algorithm-pedagogy（逐机制对账：对 dossier.mechanisms 每条填勾选表——直觉在场?数值推演表在场且带 trace 标记?不变量论证?量化落数字?core 三层齐?先跑 lint_trace_consistency 作客观依据；输出逐机制勾选表，不是整体印象）',
  'figure-integration（先跑 lint_diagrams；然后逐张用 Read 打开 PNG 亲眼看：图在其机制讲解附近?图注给结论而非描述画面?正文数字与图上一致?图对读懂机制真有帮助?）',
  'formula-structure（公式规则+Roadmap 开场+自包含+锚点/半角，跑 lint_formulas/lint_anchors/lint_punct/lint_chapter_structure）',
]
```

- [ ] **Step 6: 语法验证**(Workflow 脚本是 ESM,`node --check` 需 .mjs 后缀):

```bash
cp .claude/workflows/chapter-pipeline.js "$SCRATCHPAD/chapter-pipeline.mjs" && node --check "$SCRATCHPAD/chapter-pipeline.mjs" && echo SYNTAX-OK
```

($SCRATCHPAD = 会话 scratchpad 目录,执行时代入实际路径。)Expected: `SYNTAX-OK`。
另跑 `grep -c "phase('Explain')\|phase('Illustrate')" .claude/workflows/chapter-pipeline.js` Expected ≥2;`grep -c 'diagram_plan' .claude/workflows/chapter-pipeline.js` Expected `0`。

- [ ] **Step 7: 提交**

```bash
git add .claude/workflows/chapter-pipeline.js
git commit -m "feat(v3): chapter-pipeline 8 阶段——插 Explain/Illustrate+盲审门禁,writer 减负 reviewer 对账(Task 8)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: chapter-retrofit.js —— 存量外科回修 workflow

**Files:**
- Create: `.claude/workflows/chapter-retrofit.js`

**Interfaces:**
- Consumes: Task 6-8 的角色/schema/linter;`args: {chapter_id, slug, instance, highlight, repo_root}`(与 chapter-pipeline 同款 CFG 兜底)。
- Produces: 免修章返回 `{verdict: 'CLEAN'}`;动工章走 增量Explain→Illustrate→定点PatchWrite→缩编Review→Archive。

- [ ] **Step 1: 写 `.claude/workflows/chapter-retrofit.js`**(完整内容):

```js
export const meta = {
  name: 'chapter-retrofit',
  description: '存量章节外科回修：逐机制体检→增量素材→补图/换错图→定点改写算法段→缩编评审→归档（禁整章重写）',
  phases: [
    { title: 'Diagnose', detail: '读章+Read 全部 PNG，逐机制体检；免修即终止' },
    { title: 'Explain', detail: '只对 flagged 机制产经验证素材' },
    { title: 'Illustrate', detail: '补缺图/重绘错图：视觉自查+盲审' },
    { title: 'PatchWrite', detail: 'writer 只许定点 Edit 算法段与图引用' },
    { title: 'Review', detail: 'algorithm-pedagogy + figure-integration 两维门控' },
    { title: 'Archive', detail: 'trace 记 retrofit + bible figures.json 登记' },
  ],
}

// args 注入不可靠时的兜底配置（与 chapter-pipeline 同款约定）
const CFG = {
  chapter_id: 'ch16',
  slug: 'ch16-kv-cache-manager',
  instance: 'vllm',
  highlight: 'kv-cache',
  repo_root: '/mnt/e/Laboratory/Repo2Book',
}
const A = (typeof args !== 'undefined' && args && args.chapter_id) ? args : CFG
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance || 'vllm'
const CH = REPO + '/instances/' + INST + '/artifacts/' + A.slug
const HL = A.highlight || ''

const ESC = '\n\n**逃生舱（重要）**：发现体检单/素材/路线是错的——不要硬着头皮做。立即返回 status="BLOCKED"，blocker_reason 写清「哪里错 + 建议怎么改」，workflow 中止升级 Team Lead。'

function head(role) {
  return [
    '你的角色契约在 ' + REPO + '/.claude/agents/' + role + '.md —— **先读它**，严格遵守其中所有铁律。',
    '本章目录（绝对路径）：' + CH,
    '本章：' + A.chapter_id + '（存量外科回修——只动图和算法段，不重写章节主体）',
    '',
  ].join('\n')
}

const STATUS_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['status', 'note'],
  properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' } },
}
const DIAG_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['flagged_count', 'summary'],
  properties: { flagged_count: { type: 'number' }, summary: { type: 'string' } },
}
const BLIND_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['all_pass', 'failures'],
  properties: {
    all_pass: { type: 'boolean' },
    failures: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['figure_id', 'problem', 'suggested_fix'],
      properties: { figure_id: { type: 'string' }, problem: { type: 'string' }, suggested_fix: { type: 'string' } } } },
  },
}
const DIM_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['pass', 'issues'],
  properties: {
    pass: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['problem', 'suggested_fix', 'rationale', 'negotiable', 'blocking'],
      properties: { problem: { type: 'string' }, suggested_fix: { type: 'string' }, rationale: { type: 'string' }, negotiable: { type: 'boolean' }, blocking: { type: 'boolean' } } } },
  },
}

// ---------- Phase 1: Diagnose（体检不动刀；免修即终止） ----------
phase('Diagnose')
const diag = await agent(
  head('reviewer') +
  '任务：**体检本章，不做任何修改**（除按下述补 dossier 的 mechanisms 字段外）。\n' +
  '① 读 ' + CH + '/narrative/chapter.md 与 ' + CH + '/dossier/dossier.json。若 dossier 无 mechanisms 字段：按正文与源码补一份轻量机制清单，Edit 进 dossier.json（只加 mechanisms 字段，不动其他字段）。**关键**：只把你判定需要动工的机制标 needs_worked_example/needs_figure=true，其余标 false（避免 lint 误伤未动工机制）。\n' +
  '② 逐机制评深度：三层递进（直觉/机制含数值推演/源码）齐吗？不变量有论证吗？→ depth: ok|shallow。\n' +
  '③ 用 Read 逐张打开 ' + CH + '/diagrams/ 的内容 PNG（roadmap 除外）**亲眼看**：该机制有图吗？图与正文数字/源码一致吗？可读吗？→ figure: ok|missing|wrong。diagrams/ 若有 svg/png 而无对应 gen_*.py，记 action="rebuild-gen"。\n' +
  '④ 写 ' + CH + '/retrofit/retrofit-plan.json：{mechanisms:[{id,name,depth,figure,evidence,actions:[]}]}——每条判定必须带 evidence（引用正文行/图名）。\n' +
  '返回 flagged_count（depth=shallow 或 figure!=ok 的机制数）与 summary（一句话体检结论）。' + ESC,
  { schema: DIAG_SCHEMA, label: 'diagnose', phase: 'Diagnose', agentType: 'general-purpose' }
)
if (!diag) return { chapter: A.chapter_id, escalated: 'diagnose-failed', stage: 'Diagnose' }
if (diag.flagged_count === 0) { log('体检通过，本章免修'); return { chapter: A.chapter_id, verdict: 'CLEAN', summary: diag.summary } }
log('体检：' + diag.flagged_count + ' 个机制需动工 —— ' + diag.summary)

// ---------- Phase 2: Explain（只对 flagged 机制产素材） ----------
phase('Explain')
const expl = await agent(
  head('explainer') +
  '任务：读 ' + CH + '/retrofit/retrofit-plan.json，**只**对 flagged 机制（depth=shallow 或 figure!=ok）产出教学素材，写入 ' + CH + '/explainer/explainer.json（已存在则增量 Edit 合并）；trace 存 ' + CH + '/explainer/traces/。\n' +
  '本章有 implementation/ 则跑它取 trace（trace_source="run"）；没有则 trace_source="manual" 并写 manual_reason。figure!=ok 的机制补 figure-spec（重绘错图的 spec 里写清旧图错在哪）。\n' +
  '完成后自跑 `python3 ' + REPO + '/scripts/lint_explainer.py ' + CH + '` 无 BLOCKING。返回 status/note。' + ESC,
  { schema: STATUS_SCHEMA, label: 'explain', phase: 'Explain', agentType: 'general-purpose' }
)
if (expl && expl.status === 'BLOCKED') return { escalated: 'explain', stage: 'Explain', reason: expl.blocker_reason }

// ---------- Phase 3: Illustrate（补图/换图，视觉自查 + 盲审） ----------
let blindV = null
let blindLedger = []
for (let b = 1; b <= 3; b++) {
  phase('Illustrate')
  const ill = await agent(
    head('illustrator') +
    '任务：按 ' + CH + '/explainer/explainer.json 的 figure_specs 补缺图/重绘错图到 ' + CH + '/diagrams/（gen_<figure_id>.py + svg + png，登记/更新 figure-manifest.json）。被替换的旧图：其 svg/png/gen 一并删除，正文引用由 PatchWrite 阶段更新。retrofit-plan 里 action=rebuild-gen 的既有图：重建其 gen 脚本（输出须与现图一致，Read PNG 对照）。\n' +
    '每张新图强制：渲染 → Read PNG 亲眼看 → 六项自查全真才登记。\n' +
    (blindLedger.length ? '上一轮盲审 FAIL，必须修复：\n' + blindLedger.join('\n') + '\n' : '') +
    '完成后自跑 `python3 ' + REPO + '/scripts/lint_diagram_geometry.py ' + CH + '/diagrams/*.svg` 无问题。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'illustrate r' + b, phase: 'Illustrate', agentType: 'general-purpose' }
  )
  if (ill && ill.status === 'BLOCKED') return { escalated: 'illustrate', stage: 'Illustrate', round: b, reason: ill.blocker_reason }
  blindV = await agent(
    '你是插图盲审员。**只准看**：' + CH + '/diagrams/figure-manifest.json 列出的每张 PNG（用 Read 打开图片）+ ' + CH + '/explainer/explainer.json 对应 figure_spec。禁止看 gen 代码与正文。\n' +
    '逐张：① 只看图复述论点；② 对照 spec.claim——不符 = FAIL；③ 图上数字逐个核 spec.numbers——不符 = FAIL；④ 明显不可读 = FAIL。verdict/notes 用 Edit 回填 manifest 的 blind_review。\n' +
    '返回 all_pass 与 failures（figure_id + problem + suggested_fix）。',
    { schema: BLIND_SCHEMA, label: 'blind-review r' + b, phase: 'Illustrate', agentType: 'general-purpose' }
  )
  if (blindV && blindV.all_pass) break
  blindLedger = ((blindV && blindV.failures) || []).map(function (f) { return '[' + f.figure_id + '] ' + f.problem + ' → ' + f.suggested_fix })
  log('盲审第 ' + b + ' 轮 FAIL：' + blindLedger.length + ' 张图打回')
}
if (!blindV || !blindV.all_pass) return { chapter: A.chapter_id, escalated: 'blind-review-exhausted', stage: 'Illustrate', failures: (blindV && blindV.failures) || [] }

// ---------- Phase 4/5: PatchWrite + 缩编 Review（有界回环 2 轮） ----------
const DIMS = [
  'algorithm-pedagogy（逐 flagged 机制对账：直觉/数值推演表带 trace 标记/不变量/量化；先跑 lint_trace_consistency）',
  'figure-integration（先跑 lint_diagrams；逐张 Read PNG：新图被正文引用且在机制附近/图注给结论/数字一致/被删旧图无残留引用）',
]
let reviewV = null
let issuesForWriter = []
for (let r = 1; r <= 2; r++) {
  phase('PatchWrite')
  const pw = await agent(
    head('writer') +
    '任务：**外科手术式**修改 ' + CH + '/narrative/chapter.md——**只许 Edit 定点修改** flagged 机制的算法段与图引用处。\n' +
    '⛔ 禁止：整章重写 / 用 Write 覆盖 / 移动章节结构 / 改非算法叙事 / 删既有标题锚点。\n' +
    '要做：按 ' + CH + '/explainer/explainer.json 素材加深讲解（直觉→机制→源码三层，怎么衔接由你）；数值推演表进正文（表格前一行 `<!-- trace: <mechanism_id> -->`，数字不许改）；更新图引用（新图 ../diagrams/<id>.png，被替换旧图的引用与图注一并更新）。\n' +
    (issuesForWriter.length ? '上轮评审 issue（逐条采纳或带理由反驳）：\n' + JSON.stringify(issuesForWriter) + '\n' : '') +
    '完成后自跑 lint_trace_consistency / lint_anchors / lint_chapter_structure / lint_formulas / lint_punct 无 BLOCKING。返回 status/note。' + ESC,
    { schema: STATUS_SCHEMA, label: 'patch-write r' + r, phase: 'PatchWrite', agentType: 'general-purpose' }
  )
  if (pw && pw.status === 'BLOCKED') return { escalated: 'patch-write', stage: 'PatchWrite', round: r, reason: pw.blocker_reason }
  phase('Review')
  const dims = await parallel(DIMS.map(function (dim) {
    return function () {
      return agent(
        head('reviewer') +
        '任务：**只**从「' + dim + '」维度评审 ' + CH + '/narrative/chapter.md（对照 retrofit-plan.json 与 explainer.json）。每条 issue 给 suggested_fix + rationale + evidence，标 negotiable/blocking。该维度无 blocking issue → pass=true。',
        { schema: DIM_SCHEMA, label: 'review:' + dim.slice(0, 9) + ' r' + r, phase: 'Review', agentType: 'general-purpose' }
      )
    }
  }))
  const ok = dims.filter(Boolean)
  if (ok.length < DIMS.length) return { chapter: A.chapter_id, escalated: 'review-agents-failed', stage: 'Review', round: r }
  const issues = ok.flatMap(function (d) { return d.issues || [] })
  if (ok.every(function (d) { return d.pass }) && issues.filter(function (i) { return i.blocking }).length === 0) {
    reviewV = { verdict: 'APPROVED', issues: issues }
    break
  }
  issuesForWriter = issues
  reviewV = { verdict: 'REVISE', issues: issues }
  log('retrofit 评审第 ' + r + ' 轮 REVISE，回 writer 定点修')
}
if (!reviewV || reviewV.verdict !== 'APPROVED') return { chapter: A.chapter_id, escalated: 'review-exhausted', stage: 'Review', issues: (reviewV && reviewV.issues) || [] }

// ---------- Phase 6: Archive ----------
phase('Archive')
const arch = await agent(
  head('archivist') +
  '任务一：把这个 review 对象**原样**写入 ' + CH + '/reviews/retrofit-review.json：\n' + JSON.stringify(reviewV) + '\n' +
  '任务二：在 bible 的 figures.json 登记本章新图（{mechanism_id, figure_id, chapter_id: "' + A.chapter_id + '", claim}，文件在 instances/' + INST + '/book/bible/figures.json，不存在则创建）。\n' +
  '任务三：`python3 ' + REPO + '/scripts/archivist.py record --type delivery` 记 retrofit 交付并更新 trace/state.json。返回一句话状态。',
  { label: 'archive', phase: 'Archive', agentType: 'general-purpose' }
)

return { chapter: A.chapter_id, verdict: 'RETROFITTED', flagged: diag.flagged_count, review: reviewV, archive: arch }
```

- [ ] **Step 2: 语法验证**

```bash
cp .claude/workflows/chapter-retrofit.js "$SCRATCHPAD/chapter-retrofit.mjs" && node --check "$SCRATCHPAD/chapter-retrofit.mjs" && echo SYNTAX-OK
```

Expected: `SYNTAX-OK`

- [ ] **Step 3: 提交**

```bash
git add .claude/workflows/chapter-retrofit.js
git commit -m "feat(v3): chapter-retrofit——存量外科回修 workflow(体检免修早退/定点 Edit/盲审)(Task 9)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: 文档同步 + 全量回归

**Files:**
- Modify: `CLAUDE.md`、`docs/superpowers/ARCHITECT-RUNBOOK.md`

- [ ] **Step 1: CLAUDE.md 三处 Edit**:

1a. per-chapter workflow 一行,`6 阶段 \`Dossier→Implement→Test→Write→Review→Archive\`` 改为 `8 阶段 \`Dossier→Implement→Test→Explain→Illustrate→Write→Review→Archive\``,同句「多维并行评审」后追加「、插图盲审门禁(只看 PNG+spec 核论点/数字)」。

1b. 「**6 角色 = 持久提示词**」行改为「**8 角色 = 持久提示词**」,花括号里加 `explainer,illustrator`;该 bullet 之后新增一行:

```
- **存量回修** `.claude/workflows/chapter-retrofit.js`:外科式——逐机制体检(免修早退)→增量素材→补图/换错图→定点 Edit 算法段(禁整章重写)。
```

1c. 质量闸门代码块内追加三行(位置在 lint_source_grounding 行后):

```
python3 scripts/lint_dossier.py {chapter_dir}             # v3:mechanisms 机制账本(锚点行号核真)
python3 scripts/lint_explainer.py {chapter_dir}           # v3:素材真相源(表格数字可溯源到 trace)
python3 scripts/lint_trace_consistency.py {chapter_dir}   # v3:正文数值表不漂移+机制覆盖
```

并在「核心方法论」三支柱后追加一条:

```
- **D 素材先行**:图与数值轨迹先于写作、经运行验证产出(explainer.json 素材真相源 + figure-spec);illustrator 强制"渲染→Read PNG 亲眼看→自查→盲审";writer 拿素材自由叙事——**写作自由、门禁从严**。
```

- [ ] **Step 2: RUNBOOK 更新**:Read `docs/superpowers/ARCHITECT-RUNBOOK.md`,在发车(chapter-pipeline)章节后追加新小节(标题层级与文中一致):

```markdown
## 存量回修发车(chapter-retrofit,v3)

外科式回修旧章(只动图和算法段):

    Workflow({name: "chapter-retrofit", args: {chapter_id: "ch16", slug: "ch16-kv-cache-manager", instance: "vllm", highlight: "kv-cache"}})

- args 注入不可靠时改脚本内 CFG(与 chapter-pipeline 同款约定)。
- 体检 flagged_count=0 → 返回 CLEAN 免修早退,无成本浪费。
- 批量:先对全书逐章只跑 Diagnose(便宜),按 flagged 机制数排序,算法重章优先分批发车。
- 逃生舱/续跑与 chapter-pipeline 一致(BLOCKED 升级 Lead;resumeFromRunId 续跑)。

v3 单章流水线与 v2 的差异速查:Test 后多 Explain(素材)与 Illustrate(绘图+盲审)两阶段;
writer 不再画图;reviewer 维度为 fidelity / algorithm-pedagogy(逐机制对账) /
figure-integration(逐张看图) / formula-structure + haiku 读者顾问。
```

- [ ] **Step 3: 全量回归**

```bash
cd /mnt/e/Laboratory/Repo2Book
python3 -m pytest scripts/tests -q                          # 全部 linter 测试
python3 scripts/lint_diagrams.py instances/vllm/artifacts/$(ls instances/vllm/artifacts | head -1)   # 旧书兼容
for f in chapter-pipeline chapter-retrofit; do cp .claude/workflows/$f.js "$SCRATCHPAD/$f.mjs" && node --check "$SCRATCHPAD/$f.mjs"; done && echo ALL-OK
```

Expected: pytest 全过(≥30 passed)、旧章 lint exit 0、`ALL-OK`。

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md docs/superpowers/ARCHITECT-RUNBOOK.md
git commit -m "docs(v3): CLAUDE.md/RUNBOOK 同步素材先行流水线——8 阶段/8 角色/新闸门/回修发车(Task 10)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 计划自检记录

- **Spec 覆盖**:§4.1→Task 1+7(analyst);§4.3→Task 2+6(explainer);§4.4→Task 4+6(illustrator)+Task 8(盲审);§4.5→Task 3+7(writer);§4.6→Task 7+8(reviewer/DIMS);§4.7→Task 7(archivist)+Task 9(Archive 阶段);§5→Task 5;§6→Task 9;§7(模型分配)→v3 workflow 未 hardcode model,由 Lead 发车时经 opts.model 指定(spec 说"可被覆盖",不进代码);§8 交付清单逐项对应 Task 1-10;§9 风险对策已内嵌(逃生舱/回环上限/CLEAN 早退/manual 降级)。
- **占位符扫描**:无 TBD/TODO;所有代码块完整可粘贴。
- **类型一致性**:BLIND_SCHEMA/DIM_SCHEMA/STATUS_SCHEMA 在 Task 8/9 同构;TEMPLATES 枚举 Task 2 与 Task 5 一致(8 项);manifest 六项 selfcheck 键名 Task 4/5/6 逐字一致;维度名 Task 7 reviewer.md 与 Task 8 DIMS 一致。
- **执行顺序**:Task 1-4(linter,可并行)→ 5 → 6 → 7 → 8 → 9 → 10;Task 5 与 1-4 无依赖可并行。




