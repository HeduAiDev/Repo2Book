# 论文原理篇 + primer kind 系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec `docs/superpowers/specs/2026-07-04-paper-primer-system-design.md`——primer 章 kind 工厂能力(论文根基门禁/账本字段/gap 审计 workflow)+ ascend 书 Part VIII 六章原理篇的全部前置工件。

**Architecture:** 豁免与替代门禁成对出现(primer 免 lint_fidelity ↔ 强制 lint_paper_grounding);kind 标记三处贯通(workflow args.kind → analyst 写 dossier 顶层 kind → linter/评审据此分流);章节生产走既有 chapter-pipeline(发车属运营阶段,见文末,非 SDD 任务)。

**Tech Stack:** Python 3 stdlib + pytest(scripts/ 既有约定)、Claude Code Workflow JS、arXiv 论文包(markdown)。

## Global Constraints

- 全流水线执行模型 opus/sonnet(spec 前作 §7);新增 agent 调用须显式 model。
- 新 linter 风格与现有一致:stdlib-only、`lint_x(chapter_dir)->dict`+`print_report(res,cd)->int`、阻断 exit 1、中文报告、tmp_path 测试(参照 scripts/tests/test_lint_diagrams.py)。
- **成对门禁原则(spec §2.4)**:任何对硬规则 2 的豁免仅限 kind=primer,且必须同时启用 lint_paper_grounding——两者出现在同一次 Edit 里,不允许只做一半。
- **旧章零影响**:所有新检查以 `dossier.json` 顶层 `"kind":"primer"` 或字段存在为开关;无标记的章行为不变(回归验证强制)。
- workflow 文件语法检查用 async-wrapper 法(RUNBOOK §3 处方),不用裸 `node --check`。
- git:每任务定点 `git add <文件>`(工作区有回修产出等无关脏文件,严禁 -A);**绝不 push**。commit 消息尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- spec §2.3 细化(与用户确认过的意图一致):`defined_in` 不改扁平 glossary.json 的 schema,改由新文件 `book/bible/concepts.json`(`{"<术语>": "chNN"}`)承载,archivist 负责回写。

## File Structure(全景)

```
scripts/lint_paper_grounding.py            # 新:primer 章论文根基门禁(# PAPER 覆盖/arXiv 引用/公式锚)
scripts/tests/test_lint_paper_grounding.py # 新
scripts/lint_dossier.py                    # 改:paper_origin/prereq 校验(additive)
scripts/tests/test_lint_dossier.py         # 改:补用例
.claude/workflows/chapter-pipeline.js      # 改:kind=primer 契约切换(PRIMER 常量+5 处提示词分支)
.claude/workflows/book-gap-audit.js        # 新:全书概念覆盖审计
.claude/agents/{analyst,implementer,tester,reviewer,archivist}.md  # 改:primer 分支段/concepts.json
instances/vllm-ascend/book/assets/roadmap/roadmap.py       # 改:ALIASES 增 ch31-ch36
instances/vllm-ascend/book/cartography/outline-final.json  # 改:Part VIII + 6 章条目
instances/vllm-ascend/book/cartography/papers-map.json     # 新:论文算法盘点(§2.1)
instances/vllm-ascend/book/papers/<chNN-slug>/{paper.md,meta.json}  # 新:6 个论文包
CLAUDE.md / docs/superpowers/ARCHITECT-RUNBOOK.md / instances/vllm-ascend/INSTANCE.md  # 文档同步
```

---

### Task 1: scripts/lint_paper_grounding.py(TDD)

**Files:**
- Create: `scripts/lint_paper_grounding.py`
- Test: `scripts/tests/test_lint_paper_grounding.py`

**Interfaces:**
- Consumes: `dossier/dossier.json` 顶层 `kind` 字段;`implementation/*.py`;`narrative/chapter.md`;论文包 `../../book/papers/<章目录名>/paper.md`(章目录的 `../../book` 即实例 book/)。
- Produces: `lint_paper_grounding(chapter_dir)->dict`(keys: `impl, citation, formula, paper_ref, warn`;`formula/paper_ref/warn` 非阻断)+ CLI。**非 primer 章(无 kind 标记)一切为空、exit 0。**

- [ ] **Step 1: 写失败测试** `scripts/tests/test_lint_paper_grounding.py`:

```python
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_paper_grounding import lint_paper_grounding

IMPL_OK = '''
# PAPER: §3.1 Eq.4
def rejection_step(p, q, u):
    return u < min(1.0, p / q)


class Sampler:
    # PAPER: §3.2
    def draw(self):
        return 1
'''

IMPL_BAD = '''
def mystery(x):
    return x + 1
'''

MD_OK = """# 第 33 章

投机采样的保分布性由拒绝采样定理保证(arXiv:2211.17192 §3.1)。

$$
P(x) = \\min(1, p(x)/q(x))
$$

上式即论文 Eq.4 的接受准则。
"""

MD_NO_ARXIV = MD_OK.replace("arXiv:2211.17192 §3.1", "论文里")

PAPER_MD = "## 3.1 Speculative Sampling\\nEq.4 acceptance...\\n"


def _mk(tmp, kind="primer", impl=IMPL_OK, md=MD_OK, paper=PAPER_MD, sections=None):
    ch = tmp / "inst" / "artifacts" / "ch33-primer-speculative-sampling"
    (ch / "dossier").mkdir(parents=True)
    (ch / "implementation").mkdir(parents=True)
    (ch / "narrative").mkdir(parents=True)
    doc = {"mechanisms": [{"id": "m1", "paper_origin": {
        "paper": "arXiv:2211.17192", "sections": sections or ["§3.1"]}}]}
    if kind:
        doc["kind"] = kind
    (ch / "dossier" / "dossier.json").write_text(json.dumps(doc), encoding="utf-8")
    (ch / "implementation" / "ref_impl.py").write_text(impl, encoding="utf-8")
    (ch / "narrative" / "chapter.md").write_text(md, encoding="utf-8")
    if paper is not None:
        pd = tmp / "inst" / "book" / "papers" / "ch33-primer-speculative-sampling"
        pd.mkdir(parents=True)
        (pd / "paper.md").write_text(paper, encoding="utf-8")
    return str(ch)


def test_non_primer_chapter_all_empty(tmp_path):
    r = lint_paper_grounding(_mk(tmp_path, kind=None, impl=IMPL_BAD, md=MD_NO_ARXIV))
    assert not any(r[k] for k in ("impl", "citation", "formula", "paper_ref"))


def test_good_primer_passes(tmp_path):
    r = lint_paper_grounding(_mk(tmp_path))
    assert not r["impl"] and not r["citation"] and not r["formula"]


def test_missing_paper_anchor_blocking(tmp_path):
    assert lint_paper_grounding(_mk(tmp_path, impl=IMPL_BAD))["impl"]


def test_no_arxiv_in_narrative_blocking(tmp_path):
    assert lint_paper_grounding(_mk(tmp_path, md=MD_NO_ARXIV))["citation"]


def test_formula_without_nearby_anchor_warns(tmp_path):
    md = MD_OK.replace("上式即论文 Eq.4 的接受准则。", "就是这样。").replace(
        "(arXiv:2211.17192 §3.1)", "")
    md += "\\n\\narXiv:2211.17192\\n" + "\\n" * 30 + "$$\\ny = x\\n$$\\n" + "\\n" * 15
    r = lint_paper_grounding(_mk(tmp_path, md=md))
    assert r["formula"] and not r["citation"]


def test_section_not_in_paper_pack_warns(tmp_path):
    r = lint_paper_grounding(_mk(tmp_path, sections=["§9.9"]))
    assert r["paper_ref"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/e/Laboratory/Repo2Book && python3 -m pytest scripts/tests/test_lint_paper_grounding.py -q`
Expected: `ModuleNotFoundError: No module named 'lint_paper_grounding'`

- [ ] **Step 3: 实现 `scripts/lint_paper_grounding.py`**(完整内容):

```python
#!/usr/bin/env python3
"""论文根基 linter — primer(原理章)的替代门禁,与 lint_fidelity 成对:
primer 章豁免 subtract-only,但参考实现与推导必须锚定论文。

启用条件:dossier/dossier.json 顶层 "kind":"primer"。非 primer 章一切为空、exit 0。

阻断项:implementation/*.py 有 def/class 缺 `# PAPER:` 锚(定义行上下 3 行内);
        narrative 无任何 arXiv id(推导无出处)。
警告项:某 `$$` 公式块 ±10 行内无引用锚(§/Eq/式/arXiv);
        dossier paper_origin.sections 的小节号在论文包 paper.md 里 grep 不到。
用法:python3 lint_paper_grounding.py <chapter_dir>   阻断项存在则 exit 1。
"""
import json
import re
import sys
from pathlib import Path

ARXIV = re.compile(r'arXiv[:\s/]*(\d{4}\.\d{4,5})', re.I)
ANCHOR = re.compile(r'§|Eq\.?|arXiv|式\s*\(|PAPER', re.I)
DEF = re.compile(r'^\s*(?:def|class)\s+(\w+)')


def lint_paper_grounding(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    res = {"impl": [], "citation": [], "formula": [], "paper_ref": [], "warn": []}
    df = d / "dossier" / "dossier.json"
    try:
        doc = json.loads(df.read_text(encoding="utf-8")) if df.exists() else {}
    except ValueError:
        doc = {}
    if doc.get("kind") != "primer":
        return res

    # 1) 参考实现每个 def/class 有 # PAPER: 锚(定义行上 3 行或下 3 行内)
    for py in sorted((d / "implementation").glob("*.py")) if (d / "implementation").exists() else []:
        lines = py.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, ln in enumerate(lines):
            m = DEF.match(ln)
            if not m:
                continue
            window = lines[max(0, i - 3): i + 4]
            if not any("# PAPER:" in w for w in window):
                res["impl"].append(f"  {py.name}:{i+1} {m.group(1)} 缺 `# PAPER: §x Eq.y` 锚")

    # 2) 正文:必须有 arXiv id;每个 $$ 块 ±10 行内应有引用锚
    nar = d / "narrative" / "chapter.md"
    if nar.exists():
        text = nar.read_text(encoding="utf-8")
        if not ARXIV.search(text):
            res["citation"].append("  正文无任何 arXiv id——推导必须给论文出处")
        lines = text.splitlines()
        starts = [i for i, ln in enumerate(lines) if ln.strip() == "$$"]
        for a, b in zip(starts[0::2], starts[1::2]):
            lo, hi = max(0, a - 10), min(len(lines), b + 11)
            ctx = "\n".join(lines[lo:a] + lines[b + 1:hi])
            if not ANCHOR.search(ctx):
                res["formula"].append(f"  L{a+1} 公式块 ±10 行内无引用锚(§/Eq/arXiv)")
    else:
        res["warn"].append("  narrative/chapter.md 尚不存在(写作前跑属正常)")

    # 3) dossier paper_origin.sections 可在论文包里找到(WARNING)
    inst_book = d.resolve().parent.parent / "book"
    pack = inst_book / "papers" / d.resolve().name / "paper.md"
    ptext = pack.read_text(encoding="utf-8", errors="replace") if pack.exists() else None
    if ptext is None:
        res["warn"].append(f"  论文包缺失:{pack}(发车前应先落盘)")
    else:
        for mech in doc.get("mechanisms", []):
            po = mech.get("paper_origin") or {}
            for s in po.get("sections") or []:
                key = s.replace("§", "").replace("Eq.", "").strip()
                if key and key not in ptext:
                    res["paper_ref"].append(f"  {mech.get('id')}: 小节 {s} 在论文包里找不到")
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Paper-Grounding Lint: {cd}\n{'=' * 60}")
    blocking = len(res["impl"]) + len(res["citation"])
    for k, issues in res.items():
        mark = "❌ " if k in ("impl", "citation") else "⚠️ "
        for i in issues:
            print(mark + f"{k}: {i}")
    if blocking == 0:
        print("✓ 论文根基检查通过(# PAPER 全覆盖 / 正文有出处)")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_paper_grounding.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_paper_grounding(sys.argv[1]), sys.argv[1]))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest scripts/tests/test_lint_paper_grounding.py -q`
Expected: `6 passed`

- [ ] **Step 5: 回归旧章不受影响**

Run: `python3 scripts/lint_paper_grounding.py instances/vllm-ascend/artifacts/ch20-mla-on-npu`
Expected: `✓ 论文根基检查通过…` exit 0(无 kind 标记 → 空跑)

- [ ] **Step 6: 提交**

```bash
git add scripts/lint_paper_grounding.py scripts/tests/test_lint_paper_grounding.py
git commit -m "feat(primer): lint_paper_grounding——豁免 subtract-only 的成对替代门禁(Task 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: lint_dossier.py 扩展——paper_origin / prereq(TDD,additive)

**Files:**
- Modify: `scripts/lint_dossier.py`
- Test: `scripts/tests/test_lint_dossier.py`(追加,原 7 用例不动)

**Interfaces:**
- Consumes/Produces: `mechanisms[]` 新可选字段 `paper_origin: {"paper": "arXiv:NNNN.NNNNN|http(s)://…", "sections": [str]}`、`prereq: "chNN"`。校验规则:paper_origin 存在时 paper 格式合法且 sections 为非空 list → 违反 **blocking**(归 `mechanism`);prereq 存在时同实例 artifacts 下有 `chNN-*` 目录 → 缺失 **warn**(primer 章可能尚未建目录);`kind=algorithm` 且无 paper_origin → **warn**(提示确认)。dossier 顶层 `kind` 字段不校验值(自由字段)。

- [ ] **Step 1: 在 `scripts/tests/test_lint_dossier.py` 末尾追加用例**:

```python
def test_paper_origin_valid_passes(tmp_path):
    m = dict(GOOD_MECH, paper_origin={"paper": "arXiv:2211.17192", "sections": ["§3.1"]})
    r = lint_dossier(_mk(tmp_path, [m]))
    assert not r["mechanism"]


def test_paper_origin_bad_id_blocking(tmp_path):
    m = dict(GOOD_MECH, paper_origin={"paper": "某论文", "sections": ["§3"]})
    assert lint_dossier(_mk(tmp_path, [m]))["mechanism"]


def test_paper_origin_empty_sections_blocking(tmp_path):
    m = dict(GOOD_MECH, paper_origin={"paper": "arXiv:2211.17192", "sections": []})
    assert lint_dossier(_mk(tmp_path, [m]))["mechanism"]


def test_prereq_missing_dir_warns_only(tmp_path):
    m = dict(GOOD_MECH, prereq="ch31")
    r = lint_dossier(_mk(tmp_path, [m]))
    assert r["warn"] and not r["mechanism"]


def test_algorithm_without_paper_origin_warns(tmp_path):
    r = lint_dossier(_mk(tmp_path, [GOOD_MECH]))   # GOOD_MECH 无 paper_origin
    assert any("paper_origin" in w for w in r["warn"])
```

注意:`test_paper_origin_valid_passes` 里 GOOD_MECH 有 paper_origin → 不应触发 algorithm 无出处的 warn;`test_valid_mechanisms_pass`(旧用例)只断言 mechanism/anchor/invalid 为空,不断言 warn——新增的 algorithm-warn 不会破坏它(核对后如它断言了 warn 为空,把该断言改为不含 warn)。

- [ ] **Step 2: 跑测试确认新用例失败**

Run: `python3 -m pytest scripts/tests/test_lint_dossier.py -q`
Expected: 旧 7 过,新 5 中至少 `test_paper_origin_bad_id_blocking`/`test_algorithm_without_paper_origin_warns` 失败

- [ ] **Step 3: 修改 `scripts/lint_dossier.py`**——常量区加:

```python
PAPER_ID = re.compile(r'^(arXiv:\d{4}\.\d{4,5}(v\d+)?|https?://\S+)$')
```

机制循环(`for a in m.get("source_anchors") or []:` 之前)插入:

```python
        po = m.get("paper_origin")
        if po is not None:
            if not PAPER_ID.match(str(po.get("paper", ""))):
                res["mechanism"].append(f"  {mid}: paper_origin.paper 格式非法(应为 arXiv:NNNN.NNNNN 或 URL)")
            if not isinstance(po.get("sections"), list) or not po.get("sections"):
                res["mechanism"].append(f"  {mid}: paper_origin.sections 须为非空列表(§/Eq 锚)")
        elif m.get("kind") == "algorithm":
            res["warn"].append(f"  {mid}: kind=algorithm 且无 paper_origin——确认该算法确无论文出处")
        pr = m.get("prereq")
        if pr:
            arts = d.resolve()
            arts = next((p for p in arts.parents if p.name == "artifacts"), None)
            if arts is None or not list(arts.glob(pr + "-*")):
                res["warn"].append(f"  {mid}: prereq={pr} 对应章目录尚不存在(原理章未建则属正常)")
```

docstring 阻断项段追加一行:`paper_origin 格式非法(arXiv id/URL、sections 非空)。警告:algorithm 无 paper_origin;prereq 章目录缺失。`

- [ ] **Step 4: 跑测试确认全部通过**

Run: `python3 -m pytest scripts/tests/test_lint_dossier.py -q`
Expected: `12 passed`

- [ ] **Step 5: 回归**:`python3 -m pytest scripts/tests -q` Expected: 全过(46+新 11 ≈ 57)。

- [ ] **Step 6: 提交**

```bash
git add scripts/lint_dossier.py scripts/tests/test_lint_dossier.py
git commit -m "feat(primer): dossier 账本增 paper_origin/prereq 校验——逐章防概念裸奔(Task 2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: chapter-pipeline.js 增 kind=primer 契约切换

**Files:**
- Modify: `.claude/workflows/chapter-pipeline.js`

**Interfaces:**
- Consumes: 发车 args 新增 `kind: 'primer'`(缺省即码章,行为不变);Task 1 的 lint_paper_grounding CLI。
- Produces: PRIMER 分支贯通 Dossier/Implement/Test/Write/Review 五处提示词。

先 Read 全文件。所有 Edit 保持全角标点与既有 JS 串接风格。

- [ ] **Step 1: A 常量区(`const HL = ...` 行后)插入**:

```js
const PRIMER = A.kind === 'primer'
const PAPERS = REPO + '/instances/' + INST + '/book/papers/' + A.slug
```

- [ ] **Step 2: head() 返回数组**(`'本章：' + ...` 行后)插入一行:

```js
    PRIMER ? '本章为 **primer 原理章**：论文包在 ' + PAPERS + '/paper.md（先读它）。硬规则 2 豁免仅限本章 kind——实现是**论文忠实的小型参考实现**（非 subtract-only），替代门禁为 lint_paper_grounding。' : '',
```

- [ ] **Step 3: Dossier 提示词**——`'must_keep 要把…'` 行前插入:

```js
  (PRIMER ? '本章是 primer 原理章：深读论文包 ' + PAPERS + '/paper.md 与落地代码（' + PATHS + '）。dossier.json 顶层写 "kind":"primer"；每个机制**必填** paper_origin{paper,sections}；embed_excerpts 可含论文公式（带 §/Eq 锚）与代码双源；subtraction_plan 留空对象（primer 不做减法）。\n' : '') +
```

- [ ] **Step 4: Implement 提示词整段条件化**——现 Implement agent 调用第一个字符串参数,在 `'任务：读 ' + CH + '/dossier/dossier.json…'` 之前拼接:

```js
    (PRIMER
      ? '任务：读 ' + CH + '/dossier/dossier.json 与 ' + PAPERS + '/paper.md，产出**论文忠实的小型参考实现**到 ' + CH + '/implementation/（NumPy/纯 CPU torch，小参数可跑），TDD 先写测试到 ' + CH + '/tests/。\n每个 def/class 标 `# PAPER: §x Eq.y`（对标码章的 # SOURCE）。**不发明论文没有的机制**；实现规模以「explainer 能跑出可示教轨迹」为度。\n完成后自跑 `python3 ' + REPO + '/scripts/lint_paper_grounding.py ' + CH + '` 确保无 BLOCKING。返回 status/note。' + ESC
      : /* 原码章提示词整串原样保留 */ )
```

(实施法:把原 Implement 提示词串包进三元的 else 分支,不改其内容;`ledger` 注入行两分支共用——保持在三元外或两分支各带,以 node 检查为准。)

- [ ] **Step 5: Test 提示词同法条件化**,PRIMER 分支为:

```js
      '任务：验证 ' + CH + '/implementation/ **忠实复现论文断言**（非复现仓库行为）：对 dossier 各机制的论文性质设计测试——分布保持类跑统计检验（固定随机种子、宽松阈值防 flaky）、恒等类做数值对照、优化类验证目标量改善。host `python3 -m pytest ' + CH + '/tests -q`。\n写 ' + CH + '/tests/test-report.json（含 verdict 与每个性质对应的论文锚 §/Eq）。全过且 lint_paper_grounding 无 BLOCKING → APPROVED；否则 REJECTED 且 failures 写清。'
```

- [ ] **Step 6: Write 提示词两处**:
6a. 素材行(Task 8 前作加的 `'素材已备好：…'`)后追加:

```js
  (PRIMER ? '本章四段式必达物：动机 → 数学推导（**每个关键公式给论文锚 §/Eq + arXiv id**）→ 小参数数值推演（explainer 素材）→ 落地（vllm_ascend 真实代码锚点 + 链接对应码章）。\n' : '') +
```

6b. 收尾 linter 行的三元改为三分支(保持原两分支文本不动,新增 PRIMER 优先):

```js
  '完成后自跑' + (PRIMER ? '五个 linter（chapter_structure/formulas/source_grounding/trace_consistency/paper_grounding，primer 章不跑 fidelity）' : (A.skip_impl ? '…原文…' : '…原文…')) + '均无 BLOCKING（图的 linter 归 illustrator，不用你跑）。返回 status/note。' + ESC,
```

- [ ] **Step 7: DIMS 维度 0 条件化**:

```js
const DIMS = [
  PRIMER
    ? 'paper-fidelity（对照 ' + PAPERS + '/paper.md 逐公式核对：推导忠实于论文？符号一致？引用锚完备？跑 lint_paper_grounding；evidence 必须引论文小节）'
    : 'fidelity（保真度+过度删减+零脚手架泄漏，跑 lint_fidelity/lint_source_grounding/lint_chapter_structure）',
  /* 其余 3 维原样 */
]
```

revise 提示词收尾 linter 三元同 Step 6b 加 PRIMER 分支。

- [ ] **Step 8: 语法验证**(async-wrapper 法,scratchpad 代入实际路径):

```bash
python3 - <<'EOF'
import pathlib
src = pathlib.Path('.claude/workflows/chapter-pipeline.js').read_text(encoding='utf-8')
i = src.index('// ⚠️ 本环境实测')
out = src[:i].replace('export const meta', 'const meta', 1) + 'async function __main__() {\n' + src[i:] + '\n}\n'
pathlib.Path('$SCRATCHPAD/cp-primer.mjs').write_text(out, encoding='utf-8')
EOF
node --check $SCRATCHPAD/cp-primer.mjs && echo SYNTAX-OK-WRAPPED
grep -c 'PRIMER' .claude/workflows/chapter-pipeline.js   # Expected ≥8
```

- [ ] **Step 9: 提交**

```bash
git add .claude/workflows/chapter-pipeline.js
git commit -m "feat(primer): chapter-pipeline 增 kind=primer——论文包/参考实现/论文性质验证/paper-fidelity 评审(Task 3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: 角色契约 primer 分支段(5 个文件)

**Files:**
- Modify: `.claude/agents/{analyst,implementer,tester,reviewer,archivist}.md`

各文件末尾(收工段之前)追加短节,全角标点:

- [ ] **Step 1: analyst.md 追加**:

```markdown
## primer 原理章分支（workflow 注明本章 kind=primer 时）
- 真相源=**论文包**（`book/papers/<slug>/paper.md`）+落地代码双源；dossier 顶层写 `"kind":"primer"`。
- mechanisms **必填** `paper_origin{paper: "arXiv:…", sections: ["§x","Eq.y"]}`；embed_excerpts 可含论文公式（带锚）。
- subtraction_plan 留空对象；自检仍跑 lint_dossier（会校验 paper_origin 格式）。
```

- [ ] **Step 2: implementer.md 追加**:

```markdown
## primer 原理章分支（唯一豁免「只做减法」的场合）
- 产出**论文忠实的小型参考实现**（NumPy/纯 CPU torch，小参数可跑）——不是仓库精简版。
- 每个 def/class 标 `# PAPER: §x Eq.y`（对标 # SOURCE）；**不发明论文没有的机制**。
- 自检换门禁：`python3 scripts/lint_paper_grounding.py {chapter_dir}` 无 BLOCKING（不跑 lint_fidelity）。
```

- [ ] **Step 3: tester.md 追加**:

```markdown
## primer 原理章分支
- 验证对象换为**论文断言**：分布保持→统计检验（固定种子、宽松阈值）；恒等变换→数值对照；优化目标→改善量。test-report.json 每个性质注明论文锚 §/Eq。
```

- [ ] **Step 4: reviewer.md**——维度 0 行后追加一行:

```markdown
   （primer 原理章：维度 0 换为 **paper-fidelity**——对照论文包逐公式核对推导忠实/符号一致/引用锚完备，跑 lint_paper_grounding；evidence 必须引论文小节。）
```

- [ ] **Step 5: archivist.md**——Book Bible 列表追加:

```markdown
- `concepts.json` 概念登记表（v4-gap 治理）：`{"<术语>": "chNN"}` 记录每个核心概念**在哪章建立**——gap 审计据此判定"前章已立"。归档时把本章新建立的概念写入。
```

- [ ] **Step 6: 核查与提交**:`grep -l 'primer' .claude/agents/*.md` Expected 4 个文件;`grep -c 'concepts.json' .claude/agents/archivist.md` Expected ≥1。

```bash
git add .claude/agents/analyst.md .claude/agents/implementer.md .claude/agents/tester.md .claude/agents/reviewer.md .claude/agents/archivist.md
git commit -m "feat(primer): 角色契约 primer 分支——豁免与替代门禁成对/论文性质验证/concepts 登记(Task 4)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: book-gap-audit.js——全书概念覆盖审计 workflow

**Files:**
- Create: `.claude/workflows/book-gap-audit.js`

**Interfaces:**
- Consumes: `args: {instance, chapters?: [slug], date, out?}`(chapters 缺省=全书;date 必传,脚本内禁 Date)。
- Produces: 审计报告落盘 `instances/<instance>/book/audits/gap-audit-<date>.json` + workflow 返回 top 悬崖摘要。

- [ ] **Step 1: 写完整文件**:

```js
export const meta = {
  name: 'book-gap-audit',
  description: '全书概念覆盖审计：每章术语/概念首现须「本章建立/前章已立(concepts.json)/有先修指路」三者居一，输出按严重度排序的 gap 清单',
  phases: [
    { title: 'Audit', detail: '每章一个审计 agent 并行扫' },
    { title: 'Merge', detail: '去重排序落盘报告' },
  ],
}

const CFG = { instance: 'vllm-ascend', chapters: null, date: 'undated', repo_root: '/mnt/e/Laboratory/Repo2Book' }
const A = (typeof args !== 'undefined' && args && args.instance) ? args : CFG
const REPO = A.repo_root || '/mnt/e/Laboratory/Repo2Book'
const INST = A.instance
const BOOK = REPO + '/instances/' + INST + '/book'
const ARTS = REPO + '/instances/' + INST + '/artifacts'
const OUT = A.out || (BOOK + '/audits/gap-audit-' + (A.date || 'undated') + '.json')

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false, required: ['issues'],
  properties: { issues: { type: 'array', items: { type: 'object', additionalProperties: false,
    required: ['concept', 'severity', 'evidence', 'suggested_fix'],
    properties: { concept: { type: 'string' }, severity: { type: 'string', enum: ['cliff', 'bump'] },
      evidence: { type: 'string' }, suggested_fix: { type: 'string' } } } } },
}

phase('Audit')
// 章清单：args 给定则用之；否则让首个 agent 列目录（脚本无 fs）
let slugs = A.chapters
if (!slugs || !slugs.length) {
  const LIST_SCHEMA = { type: 'object', additionalProperties: false, required: ['slugs'],
    properties: { slugs: { type: 'array', items: { type: 'string' } } } }
  const ls = await agent('列出目录 ' + ARTS + ' 下所有形如 chNN-* 的子目录名（用 Bash ls），按章号排序返回 slugs。',
    { schema: LIST_SCHEMA, label: 'list-chapters', phase: 'Audit', model: 'haiku', agentType: 'general-purpose' })
  slugs = (ls && ls.slugs) || []
}
log('审计 ' + slugs.length + ' 章')

const perCh = await parallel(slugs.map(function (slug) {
  return function () {
    return agent(
      '你是概念覆盖审计员。只读：' + ARTS + '/' + slug + '/narrative/chapter.md、' + BOOK + '/bible/glossary.json、' + BOOK + '/bible/concepts.json（可能不存在）、' + BOOK + '/cartography/papers-map.json（可能不存在）。\n' +
      '任务：找出本章**首现即使用**的术语/概念中，不满足三者居一的：① 本章自己建立（有定义/推导/直觉）；② 前章已立（concepts.json 里登记且章号更早）；③ 有先修指路（正文链接到某原理章/前章锚点）。\n' +
      '判严重度：cliff=不读论文/外部资料跟不上正文主线；bump=一句话补丁即可。常见词（tensor/GPU/KV cache 这类全书公设）不算。\n' +
      '每条 evidence 引正文行号与原句片段。无问题返回 issues=[]。',
      { schema: AUDIT_SCHEMA, label: 'audit:' + slug.slice(0, 12), phase: 'Audit', model: 'sonnet', agentType: 'general-purpose' }
    ).then(function (r) { return { slug: slug, issues: (r && r.issues) || [] } })
  }
}))

phase('Merge')
const all = perCh.filter(Boolean)
const flat = all.flatMap(function (c) { return c.issues.map(function (i) { return Object.assign({ chapter: c.slug }, i) }) })
const cliffs = flat.filter(function (i) { return i.severity === 'cliff' })
const report = { date: A.date, instance: INST, chapters_audited: slugs.length,
  totals: { cliffs: cliffs.length, bumps: flat.length - cliffs.length }, issues: flat }
await agent(
  '把下面 JSON **原样** Write 到 ' + OUT + '（目录不存在则先建）。不要改写内容。写完返回 "written"。\n' + JSON.stringify(report),
  { label: 'write-report', phase: 'Merge', model: 'haiku', agentType: 'general-purpose' }
)
log('gap 审计完成：cliff ' + cliffs.length + ' / bump ' + (flat.length - cliffs.length) + ' → ' + OUT)
return { report: OUT, totals: report.totals,
  top_cliffs: cliffs.slice(0, 12).map(function (i) { return i.chapter + ': ' + i.concept }) }
```

- [ ] **Step 2: 语法验证**(async-wrapper 法,同 Task 3 Step 8 的 python 包裹脚本,把 index 标记换成 `const CFG`):Expected `SYNTAX-OK-WRAPPED`。

- [ ] **Step 3: 提交**

```bash
git add .claude/workflows/book-gap-audit.js
git commit -m "feat(primer): book-gap-audit——全书概念覆盖审计 workflow(cliff/bump 分级)(Task 5)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: papers-map + roadmap 键 + outline Part VIII(配置三件套)

**Files:**
- Create: `instances/vllm-ascend/book/cartography/papers-map.json`
- Modify: `instances/vllm-ascend/book/assets/roadmap/roadmap.py`(ALIASES 增 6 键)
- Modify: `instances/vllm-ascend/book/cartography/outline-final.json`(parts 增 P8;chapters 增 6 条)

- [ ] **Step 1: 写 papers-map.json**(完整内容;arXiv id 除 2211.17192/2606.19348 外由 Task 7 取回论文时核对回填,先写占位字段 `"arxiv": "VERIFY"` 的**不允许**——用下列已核对值,Task 7 若发现不符则改此文件并在报告注明):

```json
{
  "created": "2026-07-04",
  "note": "论文算法盘点(spec §2.1)。primer_chapter=chNN 表示该算法有独立原理章;inline 表示码章内一段即可;none 表示不铺垫。",
  "papers": [
    {"algorithm": "MLA(低秩 KV 压缩/解耦 RoPE/权重吸收)", "paper": {"title": "DeepSeek-V2", "arxiv": "arXiv:2405.04434", "sections_core_math": ["§2.1"]}, "mechanisms_in_repo": ["vllm_ascend/attention/mla_v1.py"], "primer_chapter": "ch31", "rationale": "ch20 最大认知悬崖:解耦 RoPE 的为什么"},
    {"algorithm": "稀疏注意力谱系 NSA→DSA(Lightning Indexer)", "paper": {"title": "Native Sparse Attention / DeepSeek-V3.2 DSA", "arxiv": "arXiv:2502.11089", "sections_core_math": ["§3"]}, "mechanisms_in_repo": ["vllm_ascend/attention/sfa_v1.py", "vllm_ascend/attention/dsa_v1.py"], "primer_chapter": "ch32", "rationale": "ch21 机制推导好但论文谱系零引用"},
    {"algorithm": "投机采样:拒绝采样定理+MTP(+DSpark 前瞻)", "paper": {"title": "Fast Inference via Speculative Decoding + DeepSeek-V3 MTP", "arxiv": "arXiv:2211.17192", "sections_core_math": ["§3", "Thm.1"]}, "mechanisms_in_repo": ["vllm_ascend/models/deepseek_v4_mtp.py"], "primer_chapter": "ch33", "rationale": "ch29 零算法纯管线"},
    {"algorithm": "EPLB 专家负载均衡算法", "paper": {"title": "DeepSeek-V3 (EPLB 部分) + deepseek-ai/EPLB", "arxiv": "arXiv:2412.19437", "sections_core_math": ["§3.4"]}, "mechanisms_in_repo": ["vllm_ascend/eplb"], "primer_chapter": "ch34", "rationale": "ch09 只讲迁移机器,均衡规划器黑盒"},
    {"algorithm": "量化数学:scale/zero-point→GPTQ/AWQ/SmoothQuant", "paper": {"title": "GPTQ/AWQ/SmoothQuant 三篇", "arxiv": "arXiv:2210.17323", "sections_core_math": ["各论文核心节,见论文包 meta"]}, "mechanisms_in_repo": ["vllm_ascend/quantization"], "primer_chapter": "ch35", "rationale": "两书均无量化数学"},
    {"algorithm": "V4 CSA/HCA 两级压缩混合注意力", "paper": {"title": "DeepSeek-V4", "arxiv": "arXiv:2606.19348", "sections_core_math": ["架构章"]}, "mechanisms_in_repo": ["vllm_ascend/worker/kvcomp_utils.py", "vllm_ascend/models/deepseek_v4.py"], "primer_chapter": "ch36", "rationale": "pin 已含 V4 代码,现有章未覆盖;与 ch21 构成 V3.2→V4 演进线"}
  ]
}
```

- [ ] **Step 2: roadmap.py ALIASES 末尾(ch30 条目后)追加**:

```python
    # Part VIII 原理篇（primer）——挂到所属子系统 Part，callout 标「原理篇」
    "ch31": ("attention",   "原理篇：MLA 低秩压缩与解耦 RoPE"),
    "ch32": ("attention",   "原理篇：NSA→DSA 稀疏注意力"),
    "ch33": ("models",      "原理篇：投机采样与拒绝采样定理"),
    "ch34": ("parallel-kv", "原理篇：EPLB 均衡算法"),
    "ch35": ("models",      "原理篇：量化数学"),
    "ch36": ("attention",   "原理篇：V4 CSA/HCA 压缩注意力"),
```

验证:`python3 instances/vllm-ascend/book/assets/roadmap/roadmap.py --highlight ch31 --out $SCRATCHPAD/rm31.svg && rsvg-convert -z 2 $SCRATCHPAD/rm31.svg -o $SCRATCHPAD/rm31.png` 后 **Read PNG 亲眼看**(callout 文案在位、无重叠)。

- [ ] **Step 3: outline-final.json**——`parts` 数组追加:

```json
{"id": "P8", "title": "Part VIII — 算法原理篇：论文里的 DeepSeek", "intent": "为码章的论文级认知悬崖补地基：每章精读一篇算法论文——动机→数学推导（锚论文公式）→小参数数值推演（经运行验证）→落到 vllm_ascend 真实代码。与 ch09/20/21/27/29 双向指路。"}
```

`chapters` 数组追加 6 条(字段齐全,mode 用 "primer";est_size 用 "M"):

```json
{"chapter_id": "ch31", "slug": "ch31-primer-mla", "title": "MLA：低秩 KV 压缩、解耦 RoPE 与权重吸收", "focus": "DeepSeek-V2 MLA 论文精读：KV 联合低秩压缩为何可行；RoPE 位置旋转为何不可吸收进 W_UK（解耦 RoPE 的存在理由，ch20 最大悬崖）；权重吸收恒等推导与 q 侧低秩（q_lora）；数值推演小例；落地 vllm_ascend/attention/mla_v1.py 并回指 ch20。", "part": "P8", "subsystem": "attention", "key_source_paths": ["vllm_ascend/attention/mla_v1.py"], "pairs_with": ["vllm/model_executor (MLA 基座实现)"], "deps": ["ch20"], "est_size": "M", "mode": "primer"},
{"chapter_id": "ch32", "slug": "ch32-primer-sparse-attention", "title": "稀疏注意力谱系：从 NSA 到 DSA 与 Lightning Indexer", "focus": "NSA(arXiv:2502.11089)→V3.2 DSA 演进：indexer 打分函数为何能代理相关性；top-k 稀疏为何不掉点（训练协同适配）；O(L·d_idx+k·d) 成本模型推导与数值例；落地 sfa_v1.py/dsa_v1.py 并回指 ch21。依赖 ch31 的 MLA 记号。", "part": "P8", "subsystem": "attention", "key_source_paths": ["vllm_ascend/attention/sfa_v1.py", "vllm_ascend/attention/dsa_v1.py"], "pairs_with": [], "deps": ["ch21", "ch31"], "est_size": "M", "mode": "primer"},
{"chapter_id": "ch33", "slug": "ch33-primer-speculative-sampling", "title": "投机采样：拒绝采样定理、MTP 与 DSpark 前瞻", "focus": "Leviathan(arXiv:2211.17192) 拒绝采样保分布定理完整证明+期望接受长度推导；DeepSeek-V3 MTP 作草稿源；数值推演（接受率→加速比）；落地 deepseek_v4_mtp.py 回指 ch29；末节前瞻 DSpark（pin 无代码，注明 RFC #11126）。", "part": "P8", "subsystem": "models", "key_source_paths": ["vllm_ascend/models/deepseek_v4_mtp.py"], "pairs_with": ["vllm ch28 (拒绝采样 kernel)"], "deps": ["ch29"], "est_size": "M", "mode": "primer"},
{"chapter_id": "ch34", "slug": "ch34-primer-eplb", "title": "EPLB：专家负载均衡的算法本体", "focus": "DeepSeek-V3(§EPLB)+官方 repo：冗余专家/分层重排的均衡目标（负载方差/最热专家）；重排算法逐步推演（小例：2 rank × 8 expert）；均衡前后指标对比；落地 vllm_ascend eplb 模块回指 ch09。", "part": "P8", "subsystem": "parallel-kv", "key_source_paths": ["vllm_ascend/eplb"], "pairs_with": [], "deps": ["ch09"], "est_size": "M", "mode": "primer"},
{"chapter_id": "ch35", "slug": "ch35-primer-quantization", "title": "量化数学：从 scale/zero-point 到 GPTQ、AWQ、SmoothQuant", "focus": "均匀量化基础（scale/zero-point/per-channel）→GPTQ 二阶补偿、AWQ 激活感知缩放、SmoothQuant 迁移难度——各给核心公式推导+小矩阵数值例；W8A8 误差分析；落地 vllm_ascend/quantization 回指 ch27。", "part": "P8", "subsystem": "models", "key_source_paths": ["vllm_ascend/quantization"], "pairs_with": [], "deps": ["ch27"], "est_size": "M", "mode": "primer"},
{"chapter_id": "ch36", "slug": "ch36-primer-v4-csa-hca", "title": "DeepSeek-V4 的 CSA/HCA：两级压缩混合注意力", "focus": "V4 技术报告(arXiv:2606.19348)：CSA（m=4 softmax 压缩+DSA 式 top-k 稀疏）与 HCA（m'=128 重压缩+稠密）层间交错；1M 上下文 FLOPs 27%/KV 10% 的账怎么算（数值推演）；与 ch31 MLA/ch32 DSA 的演进关系；落地 kvcomp_utils.py/models/deepseek_v4.py 回指 ch21/ch22/ch30。", "part": "P8", "subsystem": "attention", "key_source_paths": ["vllm_ascend/worker/kvcomp_utils.py", "vllm_ascend/models/deepseek_v4.py"], "pairs_with": [], "deps": ["ch31", "ch32"], "est_size": "L", "mode": "primer"}
```

验证:`python3 -c "import json; d=json.load(open('instances/vllm-ascend/book/cartography/outline-final.json')); assert len(d['parts'])==8 and len([c for c in d['chapters'] if c['part']=='P8'])==6; print('OK')"`

- [ ] **Step 4: 提交**

```bash
git add instances/vllm-ascend/book/cartography/papers-map.json instances/vllm-ascend/book/cartography/outline-final.json instances/vllm-ascend/book/assets/roadmap/roadmap.py
git commit -m "feat(primer): papers-map 盘点 + roadmap 原理篇键 + outline Part VIII 六章(Task 6)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: 论文包取回(6 章,WebFetch)

**Files:**
- Create: `instances/vllm-ascend/book/papers/<slug>/{paper.md,meta.json}` × 6(ch35 为三篇合并:paper.md + paper-awq.md + paper-smoothquant.md)

执行者需带 WebFetch/WebSearch。对每章:
1. 按 papers-map 的 arXiv id 取 `https://arxiv.org/abs/<id>` 确认标题匹配 → 取 HTML 全文(`https://arxiv.org/html/<id>` 或 ar5iv `https://ar5iv.labs.arxiv.org/html/<id>`)转 markdown,**保留公式(LaTeX)与小节号**;HTML 不可得则取 abs 页+结论并在 meta 注明 `"fulltext": false`。
2. id 与算法不符(papers-map 可能有误)→ WebSearch 找正确 id,**改 papers-map.json 并在报告注明**。ch33 另附 DeepSeek-V3 MTP 节选(arXiv:2412.19437 §MTP);ch35 三篇:GPTQ arXiv:2210.17323 / AWQ arXiv:2306.00978 / SmoothQuant arXiv:2211.10438(逐一核对标题)。
3. `meta.json`: `{"title","arxiv","fetched":"2026-07-04","source_url","fulltext":true|false,"license_note":"内部参考,正文仅引公式+出处"}`。
4. **公式抽查**:每个 paper.md 用 grep 确认 ≥3 处 LaTeX 公式(`\\frac|\\sum|\\min` 等)存在——纯文本无公式的转换视为失败,换源重取。
5. 验收:`ls instances/vllm-ascend/book/papers/*/paper.md | wc -l` ≥6;每个 meta.json 合法 JSON。

提交:

```bash
git add instances/vllm-ascend/book/papers/
git commit -m "feat(primer): 六章论文包落盘(arXiv→markdown,公式抽查)(Task 7)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: 文档同步 + 全量回归

**Files:**
- Modify: `CLAUDE.md`、`docs/superpowers/ARCHITECT-RUNBOOK.md`、`instances/vllm-ascend/INSTANCE.md`

- [ ] **Step 1: CLAUDE.md 两处**:
1a. HARD RULES 第 2 条末尾追加:`(**唯一豁免**:kind=primer 原理章——论文忠实参考实现,成对门禁 lint_paper_grounding,见 RUNBOOK)`。
1b. 质量闸门代码块 lint_trace_consistency 行后加:

```
python3 scripts/lint_paper_grounding.py {chapter_dir}    # primer 原理章:# PAPER 全覆盖/正文有出处(码章空跑)
```

- [ ] **Step 2: RUNBOOK**:发车章节后追加小节(格式随文):

```markdown
## 原理章(primer)发车与 gap 审计(v4)

primer 章 = 论文精读章(动机→推导→数值→落地),豁免 subtract-only、成对启用 lint_paper_grounding:

    Workflow({name:"chapter-pipeline", args:{kind:"primer", chapter_id:"ch31", slug:"ch31-primer-mla", instance:"vllm-ascend", highlight:"ch31", paths:[…落地代码…], focus:"…"}})

- **发车前置**:论文包已在 `instances/<x>/book/papers/<slug>/paper.md`(Lead WebFetch 落盘,勿赌 workflow 内网络);papers-map.json 有该章条目。
- 评审维度 0 自动换 paper-fidelity;lint_fidelity 不跑。
- **gap 审计**(每 Part 收尾/全书体检):`Workflow({name:"book-gap-audit", args:{instance:"vllm-ascend", date:"YYYY-MM-DD"}})` → 报告在 book/audits/;cliff 级逐条决定 retrofit/立 primer 章/接受。
- 新书开局(§0)同步产出 papers-map.json——论文算法在 cartography 期就规划,不等成书后盘点。
```

- [ ] **Step 3: INSTANCE.md**(vllm-ascend):状态段追加:Part VIII 六章规划(ch31-36,mode=primer)、硬规则 2 豁免仅限 primer kind、论文包位置、2026-07-04 gap 盘点 6 悬崖清单及其对应原理章。

- [ ] **Step 4: 全量回归**:

```bash
python3 -m pytest scripts/tests -q                    # 全过(≈57+6)
python3 scripts/lint_diagrams.py instances/vllm/artifacts/ch01-config-and-wiring       # 旧章 exit 0
python3 scripts/lint_paper_grounding.py instances/vllm-ascend/artifacts/ch20-mla-on-npu # 码章空跑 exit 0
# 两个 workflow async-wrapper 语法检查(chapter-pipeline / book-gap-audit)→ SYNTAX-OK-WRAPPED
```

- [ ] **Step 5: 提交**

```bash
git add CLAUDE.md docs/superpowers/ARCHITECT-RUNBOOK.md instances/vllm-ascend/INSTANCE.md
git commit -m "docs(primer): 硬规则豁免注记/primer 发车与 gap 审计 RUNBOOK/INSTANCE 状态(Task 8)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 发车阶段(运营,非 SDD 任务——Lead 在 Task 1-8 完成后执行)

1. **串行线**:ch31 → ch32 → ch36(记号/概念递进,后章 dossier 读前章 bible 回写);**并行线**:ch33、ch34、ch35 互不依赖。驱动 workflow 仿 retrofit-all 写法:两 lane(串行线一个 lane、并行三章各自 workflow() 顺次或 parallel),每章 args 见 Task 6 outline 条目(kind:"primer"、highlight=chNN、paths=key_source_paths、focus=outline focus 全文)。
2. 每章 APPROVED 后抽查:Read 1-2 张图 + 核四段式在场。
3. **先修指路框**(6 章全部 APPROVED 后):一个 sonnet agent 对 ch20/21/29/09/27 各做一处定点 Edit——在该章引入对应概念的首现段后加一句「本章默认你已了解 X;其数学推导见[第 NN 章:标题](../chNN-…/narrative/chapter.md)」,自跑 lint_anchors/lint_punct;bible arc-map 登记伏笔/回收对。
4. **闭环验收**:`Workflow({name:"book-gap-audit", args:{instance:"vllm-ascend", date:"<当日>"}})` 全书重跑——2026-07-04 盘点的 6 处悬崖(解耦 RoPE/拒绝采样/EPLB 算法/量化数学/DSA 谱系/MTP)必须消解为「已建立/有指路」;报告存档,INSTANCE.md 记账。
5. 全程产出不提交,待用户验收后分批 commit(push 由用户前台执行)。

## 计划自检记录

- **Spec 覆盖**:§2.1→Task 6(papers-map)+Task 8(RUNBOOK §0 条款);§2.2→Task 2;§2.3→Task 5+Task 4(archivist concepts.json;defined_in 以 concepts.json 承载=Global Constraints 注明的细化);§2.4→Task 3+4;§2.5→Task 7+RUNBOOK;§2.6→Task 1;§3.1→Task 6(outline)+发车阶段 1;§3.2→Task 6(roadmap)+发车阶段 3;§3.3→发车阶段 4;§4 清单逐项对应;§5 风险:论文包质量→Task 7 公式抽查、V4 单源→ch36 focus 已注明依据、参考实现走样→Task 3 Step 5 tester+DIMS paper-fidelity 双闸、DSpark→ch33 focus 前瞻节、指路框→发车阶段 3 的 lint 纪律。
- **占位符**:无 TBD/TODO;Task 3 Step 4 的 `/* 原码章提示词整串原样保留 */` 是保留既有代码的指令而非占位(执行者持有文件原文)。
- **类型一致性**:`kind:"primer"`(dossier 顶层/args)、`# PAPER:`、`paper_origin{paper,sections}`、`concepts.json` 键名在 Task 1/2/3/4/5/6 逐字一致;lint_paper_grounding 的 res 键与测试一致;AUDIT_SCHEMA 的 cliff|bump 与 RUNBOOK 文案一致。
- **执行顺序**:Task 1→2 可并行;3 依赖 1;4 依赖 3 文案;5 独立;6 独立;7 依赖 6(papers-map);8 收尾。


