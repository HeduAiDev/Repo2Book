# 章节插入体系 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 spec `docs/superpowers/specs/2026-07-05-chapter-interleave-renumber-design.md`——通用重编号引擎、跨章引用三规 lint、补章 SOP,并对 ascend 书执行原理篇交错归位(36 章新序)。

**Architecture:** 引擎两阶段(临时名目录迁移→全量引用重写)+内置校验器,plan.json 显式全量映射;重写顺序:链接路径规范修正(全书 228 处 `../chNN` 系统性差一层,本次一并修为 `../../chNN`)→目录名整串替换→裸 chNN 占位符同步替换→「第 N 章」单趟正则同步替换。三规由 lint_anchors 增强承接,防复发。

**Tech Stack:** Python 3 stdlib、pytest(tmp fixture,引擎测试不依赖 git——git mv 失败时回退 rename)、writer/illustrator agent 做接缝。

## Global Constraints

- 新章序唯一真相 = spec §1 表;plan.json 落盘 `instances/vllm-ascend/book/cartography/renumber-2026-07-05.json`。
- 姊妹书(instances/vllm)本次**完全不动**;trace/deliveries 历史文件、docs/superpowers 历史文档、.superpowers 台账不改(考古凭映射记录)。
- 同步替换纪律:目录名(唯一整串)顺序替换即安全;裸 `chNN`(后面不跟 `-`)与「第 N 章」必须**同步替换**(占位符两趟 / 单趟 re.sub 函数),防级联误替(如 ch09→ch10 再被 ch10→ch11 二次替换)。
- 执行窗口内不发任何 ascend 章 workflow;5h 循环 tick 只做状态检查。
- git:定点 add;绝不 push;commit 尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 迁移执行(Task 4)前工作区的 ascend 章节产出**必须先提交**(否则 git mv 与未跟踪/修改文件混战)——Task 4 Step 0 专门处理,提交文案标注"迁移前快照"。

## File Structure

```
scripts/renumber_chapters.py                 # 新:通用重编号引擎(plan/insert/dry-run/validate)
scripts/tests/test_renumber_chapters.py      # 新
scripts/lint_anchors.py                      # 改:跨章三规(目标存在/链接文字章号=目标号/裸章号 warn)
scripts/tests/test_lint_anchors_cross.py     # 新
.claude/agents/writer.md                     # 改:跨章链接正确形式 ../../ + 三规
docs/superpowers/ARCHITECT-RUNBOOK.md        # 改:补章 SOP 节 + §0 位置开局规划条款
CLAUDE.md                                    # 改:质量闸门注记(lint_anchors 三规)
instances/vllm-ascend/book/cartography/renumber-2026-07-05.json   # 新:本次 plan
instances/vllm-ascend/**                     # Task 4-6:迁移执行+接缝+复验(运营)
```

---

### Task 1: 重编号引擎(TDD)

**Files:**
- Create: `scripts/renumber_chapters.py`
- Test: `scripts/tests/test_renumber_chapters.py`

**Interfaces:**
- Produces: CLI `python3 scripts/renumber_chapters.py --instance <name> --plan <plan.json> [--dry-run]`、`--insert <slug>@before:<目标dir>`(仅打印级联 plan)、`--validate`(仅跑校验器)。库函数 `load_plan(path)->list[Move]`、`apply(inst_dir, moves, dry_run)->Report`、`validate(inst_dir)->list[str]`。plan.json schema:`{"moves": [{"old_dir": "ch34-primer-eplb", "new_id": "ch09"}]}`。

- [ ] **Step 1: 写失败测试** `scripts/tests/test_renumber_chapters.py`(完整):

```python
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import renumber_chapters as rc

MD_A = """# 第 1 章 开篇
见[第 2 章：乙](../ch02-beta/narrative/chapter.md)与[第 3 章：丙](../../ch03-gamma/narrative/chapter.md)。
正文提到第 2 章与第 3 章的内容。
"""


def _mk(tmp):
    """迷你实例:ch01-alpha / ch02-beta / ch03-gamma,含链接/裸章号/JSON 引用/roadmap 键。"""
    inst = tmp / "instances" / "mini"
    for d, md in [("ch01-alpha", MD_A), ("ch02-beta", "# 乙\n"), ("ch03-gamma", "# 丙\n")]:
        (inst / "artifacts" / d / "narrative").mkdir(parents=True)
        (inst / "artifacts" / d / "narrative" / "chapter.md").write_text(md, encoding="utf-8")
        (inst / "artifacts" / d / "reviews").mkdir(parents=True)
        (inst / "artifacts" / d / "reviews" / "run-ledger.json").write_text(
            json.dumps({"chapter_id": d[:4]}), encoding="utf-8")
    (inst / "book" / "cartography").mkdir(parents=True)
    (inst / "book" / "assets" / "roadmap").mkdir(parents=True)
    (inst / "book" / "bible").mkdir(parents=True)
    (inst / "trace").mkdir(parents=True)
    (inst / "book" / "assets" / "roadmap" / "roadmap.py").write_text(
        '"ch02": ("s", "乙"),\n"ch03": ("s", "丙"),\n', encoding="utf-8")
    (inst / "book" / "bible" / "concepts.json").write_text(
        json.dumps({"乙概念": "ch02"}), encoding="utf-8")
    (inst / "trace" / "state.json").write_text(json.dumps({"ch02": {"s": 1}}), encoding="utf-8")
    (inst / "INSTANCE.md").write_text("现状:第 2 章已交付\n", encoding="utf-8")
    return inst


SWAP = [{"old_dir": "ch03-gamma", "new_id": "ch02"}, {"old_dir": "ch02-beta", "new_id": "ch03"}]


def test_moves_and_simultaneous_swap(tmp_path):
    inst = _mk(tmp_path)
    rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    assert (inst / "artifacts" / "ch02-gamma").is_dir()
    assert (inst / "artifacts" / "ch03-beta").is_dir()
    assert not (inst / "artifacts" / "ch02-beta").exists()
    md = (inst / "artifacts" / "ch01-alpha" / "narrative" / "chapter.md").read_text(encoding="utf-8")
    # 链接路径规范化为 ../../ 且目录名/文字章号同步交换
    assert "(../../ch03-beta/narrative/chapter.md)" in md
    assert "(../../ch02-gamma/narrative/chapter.md)" in md
    assert "[第 3 章：乙]" in md and "[第 2 章：丙]" in md
    assert "第 3 章与第 2 章的内容" in md            # 裸文字同步互换


def test_json_and_config_rewrites(tmp_path):
    inst = _mk(tmp_path)
    rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    assert json.loads((inst / "book" / "bible" / "concepts.json").read_text(encoding="utf-8")) == {"乙概念": "ch03"}
    assert "ch02" in json.loads((inst / "trace" / "state.json").read_text(encoding="utf-8")) or \
           "ch03" in json.loads((inst / "trace" / "state.json").read_text(encoding="utf-8"))
    assert json.loads((inst / "trace" / "state.json").read_text(encoding="utf-8")) == {"ch03": {"s": 1}}
    rp = (inst / "book" / "assets" / "roadmap" / "roadmap.py").read_text(encoding="utf-8")
    assert '"ch03": ("s", "乙")' in rp and '"ch02": ("s", "丙")' in rp
    rl = json.loads((inst / "artifacts" / "ch03-beta" / "reviews" / "run-ledger.json").read_text(encoding="utf-8"))
    assert rl["chapter_id"] == "ch03"
    assert "第 3 章已交付" in (inst / "INSTANCE.md").read_text(encoding="utf-8")


def test_dry_run_touches_nothing(tmp_path):
    inst = _mk(tmp_path)
    rep = rc.apply(inst, rc.parse_moves(SWAP), dry_run=True)
    assert (inst / "artifacts" / "ch02-beta").is_dir()
    assert rep.planned_moves == 2 and rep.files_changed >= 3


def test_idempotent_second_apply(tmp_path):
    inst = _mk(tmp_path)
    rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    rep2 = rc.apply(inst, rc.parse_moves(SWAP), dry_run=False)
    assert rep2.skipped_moves == 2 and rep2.files_changed == 0


def test_validator_catches_dangling(tmp_path):
    inst = _mk(tmp_path)
    bad = inst / "artifacts" / "ch01-alpha" / "narrative" / "chapter.md"
    bad.write_text(bad.read_text(encoding="utf-8") + "\n[坏](../../ch09-nope/narrative/chapter.md)\n", encoding="utf-8")
    probs = rc.validate(inst)
    assert any("ch09-nope" in p for p in probs)


def test_insert_generates_cascade_plan(tmp_path):
    inst = _mk(tmp_path)
    plan = rc.build_insert_plan(inst, new_slug="delta", before_dir="ch02-beta")
    m = {x["old_dir"]: x["new_id"] for x in plan["moves"]}
    assert m == {"ch02-beta": "ch03", "ch03-gamma": "ch04"}
    assert plan["new_chapter_dir"] == "ch02-delta"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /mnt/e/Laboratory/Repo2Book && python3 -m pytest scripts/tests/test_renumber_chapters.py -q`
Expected: `ModuleNotFoundError: No module named 'renumber_chapters'`

- [ ] **Step 3: 实现 `scripts/renumber_chapters.py`**(完整):

```python
#!/usr/bin/env python3
"""章节重编号引擎——补章/交错的通用迁移工具(spec 2026-07-05-chapter-interleave)。

plan.json: {"moves": [{"old_dir": "ch34-primer-eplb", "new_id": "ch09"}, ...]}(显式全量)
两阶段幂等:①目录经临时名迁移(git mv,非 git 环境回退 rename);②全量引用重写——
  链接路径规范化(../chNN → ../../chNN,修全书历史笔误)→ 目录名整串替换 →
  裸 chNN(不跟 -)占位符同步替换 → 「第 N 章」单趟同步替换。
每处替换写迁移日志;--dry-run 只报不改;validate() 扫悬空引用。

用法:
  python3 scripts/renumber_chapters.py --instance vllm-ascend --plan <plan.json> [--dry-run]
  python3 scripts/renumber_chapters.py --instance vllm-ascend --insert <slug>@before:<目标dir>
  python3 scripts/renumber_chapters.py --instance vllm-ascend --validate
"""
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIRPAT = re.compile(r'^ch(\d{2})-(.+)$')


@dataclass
class Move:
    old_dir: str
    new_id: str

    @property
    def old_id(self):
        return self.old_dir[:4]

    @property
    def new_dir(self):
        return self.new_id + self.old_dir[4:]


@dataclass
class Report:
    planned_moves: int = 0
    done_moves: int = 0
    skipped_moves: int = 0
    files_changed: int = 0
    log: list = field(default_factory=list)


def parse_moves(raw):
    return [Move(m["old_dir"], m["new_id"]) for m in raw]


def load_plan(path):
    return parse_moves(json.loads(Path(path).read_text(encoding="utf-8"))["moves"])


def _mv(src: Path, dst: Path):
    try:
        subprocess.run(["git", "mv", str(src), str(dst)], check=True, capture_output=True,
                       cwd=str(ROOT))
    except (subprocess.CalledProcessError, FileNotFoundError):
        src.rename(dst)


def _rewrite_targets(inst: Path):
    pats = ["artifacts/*/narrative/*.md", "artifacts/*/dossier/*.json", "artifacts/*/explainer/*.json",
            "artifacts/*/reviews/*.json", "artifacts/*/retrofit/*.json", "artifacts/*/diagrams/*.json",
            "book/cartography/*.json", "book/bible/*.json", "book/assets/roadmap/roadmap.py",
            "trace/state.json", "INSTANCE.md"]
    out = []
    for p in pats:
        out += sorted(inst.glob(p))
    return out


def _rewrite_text(text: str, moves, report, fname):
    orig = text
    # 0) 链接路径规范化:](../chNN- → ](../../chNN-(历史笔误,narrative/ 出发需两层)
    text = re.sub(r'\]\(\.\./(ch\d{2}-)', r'](../../\1', text)
    # 1) 目录名整串替换(slug 唯一,顺序安全)
    for m in moves:
        text = text.replace(m.old_dir, m.new_dir)
    # 2) 裸 chNN(后不跟 -):占位符两趟同步替换
    idmap = {m.old_id: m.new_id for m in moves}
    for old, _ in idmap.items():
        text = re.sub(r'\b' + old + r'\b(?!-)', '\x00' + old + '\x00', text)
    for old, new in idmap.items():
        text = text.replace('\x00' + old + '\x00', new)
    # 3) 「第 N 章」单趟同步替换(半角数字,N∈映射集)
    nummap = {str(int(m.old_id[2:])): str(int(m.new_id[2:])) for m in moves}

    def _num(mo):
        n = mo.group(1)
        return '第 ' + nummap.get(n, n) + ' 章' if n in nummap else mo.group(0)

    text = re.sub(r'第\s*(\d{1,3})\s*章', _num, text)
    if text != orig:
        report.files_changed += 1
        report.log.append(f"rewrote {fname}")
    return text


def apply(inst: Path, moves, dry_run: bool) -> Report:
    rep = Report()
    todo = [m for m in moves if (inst / "artifacts" / m.old_dir).exists()]
    rep.planned_moves = len(todo)
    rep.skipped_moves = len(moves) - len(todo)
    if dry_run:
        for f in _rewrite_targets(inst):
            t = f.read_text(encoding="utf-8", errors="replace")
            r = Report()
            _rewrite_text(t, moves, r, f.name)
            rep.files_changed += r.files_changed
            rep.log += r.log
        rep.log.insert(0, f"[dry-run] moves={rep.planned_moves} skipped={rep.skipped_moves}")
        return rep
    # 阶段一:目录经临时名迁移(artifacts + book/papers)
    for base in (inst / "artifacts", inst / "book" / "papers"):
        if not base.exists():
            continue
        for m in todo:
            src = base / m.old_dir
            if src.exists():
                _mv(src, base / ("__tmp__" + m.new_dir))
        for m in todo:
            tmp = base / ("__tmp__" + m.new_dir)
            if tmp.exists():
                _mv(tmp, base / m.new_dir)
                rep.done_moves += 1
    # 阶段二:引用重写
    for f in _rewrite_targets(inst):
        t = f.read_text(encoding="utf-8", errors="replace")
        nt = _rewrite_text(t, moves, rep, str(f.relative_to(inst)))
        if nt != t:
            f.write_text(nt, encoding="utf-8")
    return rep


def validate(inst: Path):
    probs = []
    dirs = {d.name for d in (inst / "artifacts").iterdir() if d.is_dir()}
    link = re.compile(r'\]\((?:\.\./)+(ch\d{2}-[\w\-]+)/')
    for f in sorted(inst.glob("artifacts/*/narrative/*.md")):
        for mo in link.finditer(f.read_text(encoding="utf-8", errors="replace")):
            if mo.group(1) not in dirs:
                probs.append(f"{f.relative_to(inst)}: 悬空跨章链接 → {mo.group(1)}")
    return probs


def build_insert_plan(inst: Path, new_slug: str, before_dir: str):
    dirs = sorted(d.name for d in (inst / "artifacts").iterdir()
                  if d.is_dir() and DIRPAT.match(d.name))
    if before_dir not in dirs:
        raise SystemExit(f"目标章不存在: {before_dir}")
    pos = int(before_dir[2:4])
    moves = [{"old_dir": d, "new_id": f"ch{int(d[2:4]) + 1:02d}"}
             for d in dirs if int(d[2:4]) >= pos]
    return {"new_chapter_dir": f"ch{pos:02d}-{new_slug}", "moves": moves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--plan")
    ap.add_argument("--insert")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    inst = ROOT / "instances" / a.instance
    if a.validate:
        probs = validate(inst)
        print("\n".join(probs) if probs else "✓ 无悬空跨章链接")
        sys.exit(1 if probs else 0)
    if a.insert:
        slug, _, before = a.insert.partition("@before:")
        print(json.dumps(build_insert_plan(inst, slug, before), ensure_ascii=False, indent=2))
        return
    rep = apply(inst, load_plan(a.plan), a.dry_run)
    print("\n".join(rep.log))
    print(f"moves done={rep.done_moves} skipped={rep.skipped_moves} files_changed={rep.files_changed}")
    probs = [] if a.dry_run else validate(inst)
    if probs:
        print("\n".join(probs))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest scripts/tests/test_renumber_chapters.py -q`
Expected: `6 passed`(全量 suite `python3 -m pytest scripts/tests -q` 同步全绿)

- [ ] **Step 5: 提交**

```bash
git add scripts/renumber_chapters.py scripts/tests/test_renumber_chapters.py
git commit -m "feat(insert): 章节重编号引擎——两阶段迁移/同步替换/校验器/insert 级联(Task 1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: lint_anchors 跨章三规增强(TDD)

**Files:**
- Modify: `scripts/lint_anchors.py`(新增 `check_cross(path)`,wire 进 check/main;原章内检查不动)
- Test: `scripts/tests/test_lint_anchors_cross.py`

**Interfaces:**
- Produces: `check_cross(path)->dict`,keys:`broken`(跨章链接目标目录不存在,BLOCKING)、`num_mismatch`(链接文字「第 N 章」与目标目录号不符,BLOCKING)、`bad_depth`(`](../chNN-` 单层旧写法,BLOCKING——迁移后全书应为 `../../`)、`bare`(裸文字章号无链接,WARN)。main 的退出码计入前三类。

- [ ] **Step 1: 写失败测试** `scripts/tests/test_lint_anchors_cross.py`(完整):

```python
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_anchors import check_cross


def _mk(tmp, md):
    arts = tmp / "artifacts"
    (arts / "ch02-beta" / "narrative").mkdir(parents=True)
    (arts / "ch01-alpha" / "narrative").mkdir(parents=True)
    f = arts / "ch01-alpha" / "narrative" / "chapter.md"
    f.write_text(md, encoding="utf-8")
    return str(f)


def test_good_cross_link_passes(tmp_path):
    r = check_cross(_mk(tmp_path, "见[第 2 章：乙](../../ch02-beta/narrative/chapter.md)。\n"))
    assert not r["broken"] and not r["num_mismatch"] and not r["bad_depth"] and not r["bare"]


def test_broken_target_blocking(tmp_path):
    r = check_cross(_mk(tmp_path, "[第 9 章](../../ch09-nope/narrative/chapter.md)\n"))
    assert r["broken"]


def test_number_mismatch_blocking(tmp_path):
    r = check_cross(_mk(tmp_path, "[第 3 章：乙](../../ch02-beta/narrative/chapter.md)\n"))
    assert r["num_mismatch"]


def test_single_depth_legacy_blocking(tmp_path):
    r = check_cross(_mk(tmp_path, "[第 2 章](../ch02-beta/narrative/chapter.md)\n"))
    assert r["bad_depth"]


def test_bare_chapter_number_warns(tmp_path):
    r = check_cross(_mk(tmp_path, "详见第 2 章的讨论。\n"))
    assert r["bare"] and not r["broken"]


def test_bare_inside_link_text_not_flagged(tmp_path):
    r = check_cross(_mk(tmp_path, "[第 2 章：乙](../../ch02-beta/narrative/chapter.md)\n"))
    assert not r["bare"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 -m pytest scripts/tests/test_lint_anchors_cross.py -q`
Expected: `ImportError: cannot import name 'check_cross'`

- [ ] **Step 3: 实现**——`scripts/lint_anchors.py` 追加(main/check 接线由实现者按现有结构接,`--all` 输出保持原格式,新增类别打印带前缀):

```python
CROSSLINK = re.compile(r'\[([^\]]*)\]\(((?:\.\./)+)(ch\d{2})-([\w\-]+)/[^)]*\)')
BARENUM = re.compile(r'第\s*(\d{1,3})\s*章')


def check_cross(path: str):
    import pathlib
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    arts = p.resolve().parents[2]          # narrative/ → 章目录 → artifacts/
    res = {"broken": [], "num_mismatch": [], "bad_depth": [], "bare": []}
    spans = []
    for m in CROSSLINK.finditer(text):
        spans.append(m.span())
        label, dots, cid, slug = m.group(1), m.group(2), m.group(3), m.group(4)
        if dots == "../":
            res["bad_depth"].append(f"  {p.name}: 单层相对路径 ]({dots}{cid}-…(narrative/ 出发须 ../../)")
        if not (arts / f"{cid}-{slug}").is_dir():
            res["broken"].append(f"  {p.name}: 悬空跨章链接 → {cid}-{slug}")
        nm = BARENUM.search(label)
        if nm and int(nm.group(1)) != int(cid[2:]):
            res["num_mismatch"].append(
                f"  {p.name}: 链接文字「第 {nm.group(1)} 章」≠ 目标目录 {cid}")
    for m in BARENUM.finditer(text):
        if not any(s <= m.start() < e for s, e in spans):
            res["bare"].append(f"  {p.name}: 裸文字章号「{m.group(0)}」无链接(warn)")
    return res
```

接线要求:`--all` 与单文件模式均调用 check_cross;`broken/num_mismatch/bad_depth` 计入退出码 1,`bare` 仅打印 ⚠️。**存量兼容**:Task 4 迁移会把全书修成 `../../` 并同步号——本 linter 在 Task 4 之后才对 ascend 书全绿,Task 2 收工自检只跑 pytest,不跑 --all(计划内预期)。

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 -m pytest scripts/tests/test_lint_anchors_cross.py -q` → `6 passed`;全量 suite 全绿。

- [ ] **Step 5: 提交**(`feat(insert): lint_anchors 跨章三规——目标存在/文字号一致/深度规范/裸章号 warn(Task 2)` + 尾注)

---

### Task 3: 规范与 SOP 文档 + 本次 plan.json 落盘

**Files:**
- Modify: `.claude/agents/writer.md`、`docs/superpowers/ARCHITECT-RUNBOOK.md`、`CLAUDE.md`
- Create: `instances/vllm-ascend/book/cartography/renumber-2026-07-05.json`

- [ ] **Step 1: writer.md**——契约第 6 条(零脚手架泄漏)附近的跨章链接示例整改 + 新增三规。找到「**跨章**引用/回收用 markdown 链接跳目标章(如 `[第 7 章:IPC 边界](../ch07-xxx/narrative/chapter.md)`)」类句子,改为:

```markdown
**跨章**引用一律 markdown 链接、且从 narrative/ 出发用两层相对路径（`[第 7 章：IPC 边界](../../ch07-xxx/narrative/chapter.md)`）；链接文字里的章号必须与目标目录号一致（lint_anchors 三规核验）；**禁止裸文字章号**（「详见第 21 章」无链接——插章重编号时它是最大的迁移债）；导语/图注衔接按内容措辞（「上一章的 MHA 后端」），章号只活在链接里。
```

- [ ] **Step 2: RUNBOOK**——新增「补章发车(SOP)」节(放在 primer 发车节后),内容:

```markdown
## 补章发车(SOP,任何补充章的标准流程)

1. **定位先于内容**:outline-final.json 把新章条目插到目标位置(deps/part 定好);primer 章同步 papers-map。
2. **生产**:新书/尾部追加→直接按最终章号发 chapter-pipeline,零迁移;存量书中段插入→先以临时尾号生产,APPROVED 后走第 3 步。
3. **插入迁移**:`python3 scripts/renumber_chapters.py --instance <x> --insert <slug>@before:<目标dir>` 生成级联 plan → 存盘 book/cartography/ → `--plan <file> --dry-run` 审阅日志 → 执行(自动跑悬空校验)。执行窗口内不发该实例任何章 workflow。
4. **接缝导语**:writer 定点重写插入点前后章的开场/收尾(按内容措辞衔接);受影响章 roadmap.png 重渲(roadmap.py 循环);bible 章号已由引擎重写,抽查 due。
5. **复验**:lint_anchors --all(三规)/lint_punct --all/逐章 structure/gap-audit 增量。

**首创期预留**(新书 §0 即执行):cartography 定稿时可预见的原理/扩展章直接占号进 outline——插章成本趋零;跨章引用三规(见 writer 契约)让日后任何重编号只剩"跑引擎+接缝导语"。
```

并在 §0(新书开局)加一句:「outline 定稿时把 papers-map 规划的 primer 章直接占号排进物理序(参见补章发车 SOP)」。

- [ ] **Step 3: CLAUDE.md**——质量闸门块 lint_anchors 行注释改为:`# 章内锚点+跨章三规(目标存在/文字号一致/../../ 深度)`。

- [ ] **Step 4: 写 `instances/vllm-ascend/book/cartography/renumber-2026-07-05.json`**(完整,28 moves;§1 表的显式全量):

```json
{"note": "原理篇交错归位 2026-07-05;spec: docs/superpowers/specs/2026-07-05-chapter-interleave-renumber-design.md",
 "moves": [
  {"old_dir": "ch34-primer-eplb", "new_id": "ch09"},
  {"old_dir": "ch09-eplb-expert-load-balancing", "new_id": "ch10"},
  {"old_dir": "ch10-pd-disaggregation-mooncake", "new_id": "ch11"},
  {"old_dir": "ch11-kv-pooling-ascend-store", "new_id": "ch12"},
  {"old_dir": "ch12-kv-offloading-host-cpu", "new_id": "ch13"},
  {"old_dir": "ch13-npuworker-execution-control", "new_id": "ch14"},
  {"old_dir": "ch14-npumodelrunner-cuda-monkeypatch", "new_id": "ch15"},
  {"old_dir": "ch15-single-step-forward-context-dp-sync", "new_id": "ch16"},
  {"old_dir": "ch16-kv-cache-allocation-reshape-bind", "new_id": "ch17"},
  {"old_dir": "ch17-310p-inference-chip-specialization", "new_id": "ch18"},
  {"old_dir": "ch18-attention-backend-selection", "new_id": "ch19"},
  {"old_dir": "ch19-ascend-attention-mha", "new_id": "ch20"},
  {"old_dir": "ch31-primer-mla", "new_id": "ch21"},
  {"old_dir": "ch20-mla-on-npu", "new_id": "ch22"},
  {"old_dir": "ch32-primer-sparse-attention", "new_id": "ch23"},
  {"old_dir": "ch21-sparse-attention-sfa-dsa", "new_id": "ch24"},
  {"old_dir": "ch22-kv-manager-and-schedulers", "new_id": "ch25"},
  {"old_dir": "ch36-primer-v4-csa-hca", "new_id": "ch26"},
  {"old_dir": "ch23-customop-oot-replacement", "new_id": "ch27"},
  {"old_dir": "ch24-torch-library-and-meta", "new_id": "ch28"},
  {"old_dir": "ch25-ascend-compiler-aclgraph", "new_id": "ch29"},
  {"old_dir": "ch26-fusedmoe-batch-invariant", "new_id": "ch30"},
  {"old_dir": "ch35-primer-quantization", "new_id": "ch31"},
  {"old_dir": "ch27-ascend-quantization-framework", "new_id": "ch32"},
  {"old_dir": "ch28-sampling-npu-adaptation", "new_id": "ch33"},
  {"old_dir": "ch33-primer-speculative-sampling", "new_id": "ch34"},
  {"old_dir": "ch29-speculative-decode-npu", "new_id": "ch35"},
  {"old_dir": "ch30-model-lora-netloader-registration", "new_id": "ch36"}
 ]}
```

- [ ] **Step 5: 验证**:python 读该 json 断言 28 moves、new_id 覆盖 ch09-ch36 无重复;`grep -c '../../ch' .claude/agents/writer.md` ≥1。
- [ ] **Step 6: 提交**(`docs(insert): 跨章引用三规进契约/补章 SOP 进 RUNBOOK/交错 plan 落盘(Task 3)` + 尾注)

---

## 运营阶段(Task 4-6,Lead 驱动,SDD 任务完成后执行)

### Task 4: ascend 书迁移执行

1. **Step 0 迁移前快照提交**:把工作区全部 ascend 章节产出(六章原理篇/回修/指路框/bible/trace)按内容分批提交(消息注明"迁移前快照");`git status` 对 instances/vllm-ascend 必须 clean 后才继续。
2. dry-run:`python3 scripts/renumber_chapters.py --instance vllm-ascend --plan instances/vllm-ascend/book/cartography/renumber-2026-07-05.json --dry-run` → Lead 审日志(重点:「第 N 章」替换的误伤抽查——日志含文件清单)。
3. 执行(去掉 --dry-run;引擎自动跑 validate,须 0 悬空)。
4. 结构核验:36 目录号连续无缺;`--validate` 通过;papers 目录同步;outline parts 检查(P8 已并入——引擎只改号,**parts 数组的 P8 条目删除与 P3/P5/P7 intent 更新由本步一次 Edit 手工完成**,因语义改写不宜正则)。
5. 提交(单 commit:`refactor(insert): ascend 书原理篇交错归位——36 章新序(引擎迁移)` + 尾注)。

### Task 5: 接缝导语 + 图重生成

1. roadmap 批量重渲(bash 循环 28 个重编号章:roadmap.py --highlight chNN → rsvg-convert;抽 3 张 Read PNG 核 callout)。
2. writer 一次性接缝任务(opus):15 处接缝定点 Edit——6 原理章开场/收尾接真实邻章、5 承接码章开场(含改写昨日指路框为"上一章"措辞)、新 ch27 开场核对、每处自跑 structure+anchors;纪律=只动导语/图注/首末段。
3. illustrator 任务:ch01 book-map 图按 36 章新序重生成(gen 脚本改+渲染+亲眼看);writer 随后核 ch01 正文 Part 描述。
4. 本任务产出不提交,与 Task 6 复验通过后一并提交(`feat(insert): 接缝导语+全书地图随新序重写`)。

### Task 6: 复验闭环

1. `python3 scripts/renumber_chapters.py --instance vllm-ascend --validate` → 0;
2. REPO2BOOK_INSTANCE=vllm-ascend:lint_anchors --all(三规首次全书生效,须全绿)/lint_punct --all/lint_diagram_geometry --all;逐章 lint_chapter_structure 循环;
3. gap-audit 全书重跑(date=当日)→ cliffs 必须仍为 0;
4. bible.py due 抽 3 章;台账记一条 exp(跨章链接深度笔误全书修正——lint 三规防复发);
5. 全绿后提交 Task 5 产出;INSTANCE.md 记新序生效。

## 计划自检记录

- **Spec 覆盖**:§1→Task 3 plan.json+Task 4;§2 引擎→Task 1(含 --insert/--validate/dry-run/幂等/临时名/链接规范化);§3 接缝→Task 5;§4 门禁→Task 6;§5 SOP→Task 3 RUNBOOK;§6 三规→Task 2(lint)+Task 3(writer 契约/RUNBOOK §0 条款);§7 风险:同步替换纪律(Global)/执行窗口冻结(Global+SOP)/trace 不改(Global)/「第 N 章」误伤靠 dry-run 日志抽查。
- **占位符**:无;所有代码/JSON/文档块完整。
- **类型一致性**:Move/Report/parse_moves/load_plan/apply/validate/build_insert_plan 名称在 Task 1 代码与测试一致;check_cross 的四个 res 键与测试一致;plan.json 的 moves 键名 Task 1/3/4 一致。
- **顺序**:1→2→3(可并行 2/3)→4→5→6 串行(4 依赖 3 的 plan.json 与 Step 0 快照;5/6 依赖 4)。

