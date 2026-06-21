# vLLM 源码解读书 · 系统重建 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成"档案为真相源 + 只做减法精简版 + 自包含源码内嵌"的新 per-chapter 流水线，并在 ch04（AsyncLLM 三段式）上端到端跑通。

**Architecture:** 混合编排——Workflow 跑并行/确定性阶段（档案扇出、实现+测试、叙事、多维评审），少量持久角色（Archivist 持全书记忆 + Book Bible）；implementer/writer/reviewer 为章节级命名角色，迭代期用 SendMessage 续接。质量靠确定性 linter（fidelity / chapter-structure / formula / source-grounding）前置闸门。

**Tech Stack:** Python 3.11 + pytest 9（linter & bible CLI，TDD）；JS（Workflow 脚本）；Markdown（agent 提示词、章节）；svg-diagram skill（Roadmap 母版 & 配图）。

## Global Constraints

- 源码 pin：`f3fef123`，根目录 `instances/vllm/source/`；所有 file:line 引用以此 commit 为准。
- 设计依据：`docs/superpowers/specs/2026-06-21-vllm-source-reading-book-system.md`（§3 方法论、§4 章节模板、§5 角色、§6 workflow、§7.5–7.7 收敛/连贯/协同）。
- 大纲：`instances/vllm/book/cartography/outline-final.json`（8 Part/33 章）；架构事实查 `cartography/map.json` 与 `ARCHITECTURE.md`。
- **HARD RULE（沿用 CLAUDE.md）**：主编排者不得直接写/编 `narrative/chapter.md`，仅 Writer 角色可写。
- 新书章节目录用 `ch`-前缀 slug（如 `ch04-async-llm`），与旧 `NN-*` artifacts 区分，置于 `instances/vllm/artifacts/` 下。
- linter 约定（对齐 `scripts/lint_formulas.py`）：取路径参数，`print_report` 输出，遇阻断项 `sys.exit(1)`。
- **vLLM 相关代码调试/运行一律在 vLLM Docker 容器内进行**（host WSL2 无 CUDA/vLLM）。约定镜像 `vllm/vllm-openai:latest`（容器内 `/usr/bin/python3` 3.12 / vllm 0.15.1 / CUDA 可用，entrypoint=bash），用助手 `scripts/vllm_docker.sh` 以 `docker run --rm --gpus all --entrypoint /usr/bin/python3 -v <repo>:/work -w /work` 跑。**分工**：精简版自身单元测试（不 `import vllm`）host `pytest` 即可；任何 `import vllm` / 触 CUDA / 对照真实 vLLM 行为的运行 → 进容器。`test-report.json` 须记录 docker 命令 + 镜像 tag + vllm 版本（见 `wisdom/testing.md`）。**注意**：容器装的是 vllm 0.15.1，与源码 pin `f3fef123` 行号可能略差——**file:line 引用以 `f3fef123` 源码为准，容器仅用于观察/验证行为**。
- 不删除旧产出/脚本（仅标记弃用）；删除是不可逆操作，待新体系跑通后另行处理。
- 所有 git 提交信息结尾加：`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`。提交仅在计划步骤明示处进行。

## File Structure

**新增 Python（可测）**
- `scripts/lint_fidelity.py` — 保真度闸门：精简版 `# SOURCE:` 全覆盖、无杜撰标记、叙事真源码引用占比。
- `scripts/lint_chapter_structure.py` — 章节结构闸门：Roadmap 段存在、内嵌真源码块数达标。
- `scripts/bible.py` — Book Bible CLI：术语/接口/伏笔回收登记与查询。
- `scripts/vllm_docker.sh` — 在 vLLM 容器内跑 vLLM 相关代码/测试的标准化助手。
- `scripts/tests/test_lint_fidelity.py`、`test_lint_chapter_structure.py`、`test_bible.py`

**Book Bible 数据（持久连贯性）**
- `instances/vllm/book/bible/glossary.json` — 术语/译名
- `instances/vllm/book/bible/interfaces.json` — 各章精简版类/方法签名
- `instances/vllm/book/bible/arc-map.json` — 伏笔/回收/承诺登记（开局设计）
- `instances/vllm/book/bible/voice-guide.md` — 叙述声线/风格指南

**Roadmap 资产**
- `instances/vllm/book/assets/roadmap/roadmap.py` — 生成全书母版图（svg-diagram）
- 产物 `roadmap.svg` / `roadmap.png`

**Agent 提示词**
- 新增 `.claude/agents/analyst.md`
- 重写 `.claude/agents/{implementer,tester,writer,reviewer,archivist}.md`

**编排**
- `.claude/workflows/chapter-pipeline.js` — per-chapter workflow（A 档案 / B 实现+测试 / C 叙事 / D 评审 / E 归档）

**ch04 试点产物**
- `instances/vllm/artifacts/ch04-async-llm/{dossier,implementation,tests,narrative,reviews,diagrams}/`

---

## Task 1: 保真度 linter（lint_fidelity.py）

**Files:**
- Create: `scripts/lint_fidelity.py`
- Test: `scripts/tests/test_lint_fidelity.py`

**Interfaces:**
- Produces: `lint_fidelity(chapter_dir: str) -> dict[str, list[str]]`（键：`missing_source`、`invention`、`narrative_grounding`、`no_subtraction`）；CLI `python3 scripts/lint_fidelity.py <chapter_dir>`，阻断项 exit 1。

- [ ] **Step 1: 写失败测试**

```python
# scripts/tests/test_lint_fidelity.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_fidelity import lint_fidelity

def _mk(tmp, impl_files: dict, narrative: str = ""):
    d = tmp / "ch"
    (d / "implementation").mkdir(parents=True)
    for name, body in impl_files.items():
        (d / "implementation" / name).write_text(body, encoding="utf-8")
    (d / "narrative").mkdir(parents=True)
    (d / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    return str(d)

def test_missing_source_comment_is_blocking(tmp_path):
    impl = {"scheduler.py": "def schedule():\n    return 1\n"}  # no # SOURCE:
    res = lint_fidelity(_mk(tmp_path, impl))
    assert res["missing_source"], "function without # SOURCE: must be flagged"

def test_source_comment_passes(tmp_path):
    impl = {"scheduler.py": "def schedule():\n    # SOURCE: vllm/v1/core/sched/scheduler.py:L352\n    # SUBTRACTED: no preemption — vllm L466\n    return 1\n"}
    res = lint_fidelity(_mk(tmp_path, impl))
    assert not res["missing_source"]
    assert not res["no_subtraction"]

def test_invention_marker_blocking(tmp_path):
    impl = {"x.py": "def f():\n    # SOURCE: vllm/a.py:L1\n    # TOY: fake loop\n    return 1\n"}
    res = lint_fidelity(_mk(tmp_path, impl))
    assert res["invention"]

def test_narrative_overexplains_companion(tmp_path):
    impl = {"x.py": "def f():\n    # SOURCE: vllm/a.py:L1\n    # SUBTRACTED: x\n    return 1\n"}
    nar = "see implementation/x.py and implementation/x.py and implementation/x.py\nvllm/a.py:L1\n"
    res = lint_fidelity(_mk(tmp_path, impl, nar))
    assert res["narrative_grounding"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest scripts/tests/test_lint_fidelity.py -v`
Expected: FAIL（`ModuleNotFoundError: lint_fidelity`）

- [ ] **Step 3: 实现 linter**

```python
# scripts/lint_fidelity.py
#!/usr/bin/env python3
"""Fidelity linter — enforces the subtract-only companion contract."""
import ast, re, sys
from pathlib import Path

MIN_VLLM_REFS = 5
INVENTION_MARKERS = ("# ADDED", "# TOY", "# FAKE", "# INVENTED")

def _spans_missing_source(pyfile: Path):
    src = pyfile.read_text(encoding="utf-8")
    lines = src.splitlines()
    out = []
    try:
        tree = ast.parse(src, filename=str(pyfile))
    except SyntaxError as e:
        return [f"  {pyfile.name}: syntax error {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            ctx = "\n".join(lines[max(0, start - 1):end])
            if "# SOURCE:" not in ctx:
                out.append(f"  {pyfile.name}:{node.lineno} `{node.name}` 无 # SOURCE: 引用")
    return out

def lint_fidelity(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    impl = d / "implementation"
    narrative = d / "narrative" / "chapter.md"
    res = {"missing_source": [], "invention": [], "narrative_grounding": [], "no_subtraction": []}
    pyfiles = [p for p in impl.glob("*.py") if p.name != "__init__.py"] if impl.exists() else []
    subtraction_seen = False
    for p in pyfiles:
        text = p.read_text(encoding="utf-8")
        res["missing_source"] += _spans_missing_source(p)
        for m in INVENTION_MARKERS:
            if m in text:
                res["invention"].append(f"  {p.name}: 禁止标记 {m}")
        if "# SUBTRACTED:" in text:
            subtraction_seen = True
    if pyfiles and not subtraction_seen:
        res["no_subtraction"].append("  无任何 # SUBTRACTED: 标记（只做减法应有删除注释）")
    if narrative.exists():
        nt = narrative.read_text(encoding="utf-8")
        vllm_refs = len(re.findall(r"vllm/[\w/]+\.py", nt))
        comp_refs = len(re.findall(r"implementation/[\w/]+\.py", nt))
        if vllm_refs < MIN_VLLM_REFS:
            res["narrative_grounding"].append(f"  真实 vllm/ 引用仅 {vllm_refs} 处（需 >= {MIN_VLLM_REFS}）")
        if comp_refs > vllm_refs:
            res["narrative_grounding"].append(f"  叙事引用精简版({comp_refs}) 多于真实 vllm/({vllm_refs}) — 喧宾夺主")
    return res

def print_report(res: dict, chapter_dir: str) -> int:
    total = sum(len(v) for v in res.values())
    print(f"Fidelity Lint: {chapter_dir}\n{'=' * 60}")
    if total == 0:
        print("✓ 保真度检查全部通过！")
        return 0
    for k, issues in res.items():
        if issues:
            print(f"\n❌ {k} ({len(issues)}):")
            for i in issues:
                print(i)
    blocking = len(res["missing_source"]) + len(res["invention"]) + len(res["narrative_grounding"])
    print(f"\n{'=' * 60}")
    print(f"🔴 {blocking} BLOCKING" if blocking else "🟢 仅警告（no_subtraction）")
    return 1 if blocking else 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_fidelity.py <chapter_dir>"); sys.exit(1)
    sys.exit(print_report(lint_fidelity(sys.argv[1]), sys.argv[1]))
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest scripts/tests/test_lint_fidelity.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add scripts/lint_fidelity.py scripts/tests/test_lint_fidelity.py
git commit -m "feat(lint): add fidelity linter for subtract-only companion contract"
```

---

## Task 2: 章节结构 linter（lint_chapter_structure.py）

**Files:**
- Create: `scripts/lint_chapter_structure.py`
- Test: `scripts/tests/test_lint_chapter_structure.py`

**Interfaces:**
- Produces: `lint_structure(md_path: str) -> dict`（键 `no_roadmap`、`no_embedded_source`）；CLI 阻断项 exit 1。

- [ ] **Step 1: 写失败测试**

```python
# scripts/tests/test_lint_chapter_structure.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_chapter_structure import lint_structure

def _w(tmp, text):
    p = tmp / "chapter.md"; p.write_text(text, encoding="utf-8"); return str(p)

def test_missing_roadmap_blocking(tmp_path):
    text = "# 第四章\n正文\n```python\n# vllm/v1/engine/async_llm.py:L280\nx=1\n```\n```python\n# vllm/v1/engine/async_llm.py:L637\ny=2\n```\n"
    assert lint_structure(_w(tmp_path, text))["no_roadmap"]

def test_missing_embedded_source_blocking(tmp_path):
    text = "## Roadmap 你在这里\n正文没有源码块\n"
    assert lint_structure(_w(tmp_path, text))["no_embedded_source"]

def test_good_chapter_passes(tmp_path):
    text = ("## Roadmap：你在这里\n地图\n正文\n"
            "```python\n# vllm/v1/engine/async_llm.py:L280\nasync def add_request(): ...\n```\n"
            "解读\n```python\n# vllm/v1/engine/async_llm.py:L637\nasync def _run_output_handler(): ...\n```\n")
    res = lint_structure(_w(tmp_path, text))
    assert not res["no_roadmap"] and not res["no_embedded_source"] and not res["scaffold_leak"]

def test_scaffold_leak_blocking(tmp_path):
    text = ("## Roadmap 你在这里\n"
            "```python\n# instances/vllm/source/vllm/v1/engine/async_llm.py:L280\nx=1\n```\n"
            "## Cell 3 源码走读\n详见 impl-notes.md\n"
            "```python\n# vllm/v1/engine/async_llm.py:L637\ny=2\n```\n")
    res = lint_structure(_w(tmp_path, text))
    assert res["scaffold_leak"]
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest scripts/tests/test_lint_chapter_structure.py -v`
Expected: FAIL（import error）

- [ ] **Step 3: 实现**

```python
# scripts/lint_chapter_structure.py
#!/usr/bin/env python3
"""Chapter-structure linter — Roadmap present + self-contained embedded source."""
import re, sys
from pathlib import Path

MIN_SOURCE_BLOCKS = 2

def lint_structure(md_path: str) -> dict:
    text = Path(md_path).read_text(encoding="utf-8")
    res = {"no_roadmap": [], "no_embedded_source": [], "scaffold_leak": []}
    head = "\n".join(text.splitlines()[:60])
    if not re.search(r"(roadmap|路线图|你在这里)", head, re.I):
        res["no_roadmap"].append("  开头 60 行内无 Roadmap/路线图/你在这里 段")
    blocks = re.findall(r"```python.*?```", text, re.S)
    embedded = [b for b in blocks if re.search(r"vllm/[\w/]+\.py", b)]
    if len(embedded) < MIN_SOURCE_BLOCKS:
        res["no_embedded_source"].append(f"  内嵌真源码块仅 {len(embedded)}（需 >= {MIN_SOURCE_BLOCKS}，块内含 vllm/ 路径标注）")
    # 零脚手架泄漏（读者视角）：正文不得含本仓库脚手架痕迹
    SCAFFOLD = [
        (r"instances/vllm/source", "出现脚手架路径 instances/vllm/source（应用规范 vllm/ 路径）"),
        (r"\bCell\s*\d+\b", "出现 'Cell N' 脚手架标题（应用自然标题）"),
        (r"impl-notes\.md|dossier", "引用内部脚手架文件（impl-notes.md/dossier）"),
        (r"详[见细]文档|完整文档见|这里只?截取", "提到出版物中不存在的外部文档/截取说明"),
    ]
    for pat, msg in SCAFFOLD:
        if re.search(pat, text):
            res["scaffold_leak"].append(f"  {msg}")
    return res

def print_report(res: dict, path: str) -> int:
    total = sum(len(v) for v in res.values())
    print(f"Chapter-Structure Lint: {path}\n{'=' * 60}")
    if total == 0:
        print("✓ 结构检查通过（Roadmap + 自包含源码）"); return 0
    for k, issues in res.items():
        for i in issues:
            print(f"❌ {k}: {i}")
    return 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_chapter_structure.py <chapter.md>"); sys.exit(1)
    sys.exit(print_report(lint_structure(sys.argv[1]), sys.argv[1]))
```

- [ ] **Step 4: 运行确认通过**

Run: `python3 -m pytest scripts/tests/test_lint_chapter_structure.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add scripts/lint_chapter_structure.py scripts/tests/test_lint_chapter_structure.py
git commit -m "feat(lint): add chapter-structure linter (roadmap + embedded source)"
```

---

## Task 2b: vLLM Docker 执行助手（scripts/vllm_docker.sh）

**Files:**
- Create: `scripts/vllm_docker.sh`

**Interfaces:**
- Produces: `scripts/vllm_docker.sh <args...>` → 在 vLLM 容器内以 `python3` 执行（repo 挂载在 `/work`、`--gpus all`）。tester/workflow 跑任何 vLLM 相关代码均经此。

- [ ] **Step 1: 写助手**

```bash
#!/usr/bin/env bash
# Run vLLM-related code/tests inside the vLLM container (host has no CUDA/vLLM).
# Usage: scripts/vllm_docker.sh -m pytest /work/instances/vllm/artifacts/ch04-async-llm/tests
#        scripts/vllm_docker.sh -c "import vllm; print(vllm.__version__)"
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
exec docker run --rm --gpus all --entrypoint /usr/bin/python3 \
  -v "$REPO":/work -w /work "$IMAGE" "$@"
```

- [ ] **Step 2: 赋可执行 + 验证容器内 vllm 可用**

Run:
```bash
chmod +x scripts/vllm_docker.sh
scripts/vllm_docker.sh -c "import sys,vllm;print(sys.version.split()[0], vllm.__version__)"
```
Expected: 打印 `3.12.x 0.15.1`

- [ ] **Step 3: 提交**

```bash
git add scripts/vllm_docker.sh
git commit -m "feat(infra): add vllm_docker.sh helper to run vLLM-related code in container"
```

---

## Task 3: Book Bible CLI（bible.py）+ 数据脚手架

**Files:**
- Create: `scripts/bible.py`, `scripts/tests/test_bible.py`
- Create: `instances/vllm/book/bible/{glossary.json,interfaces.json,arc-map.json}`（初始为空结构）

**Interfaces:**
- Produces: CLI `python3 scripts/bible.py due <chapter_id>` 打印本章「应埋伏笔 + 应回收项」；`bible.py foreshadow --add --plant chN --payoff chM --what "..."`；`bible.py term --add <zh> <en>`；`bible.py iface --add <chapter> <sig>`。数据根 `instances/vllm/book/bible/`。

- [ ] **Step 1: 写失败测试**

```python
# scripts/tests/test_bible.py
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bible

def test_due_lists_plant_and_payoff(tmp_path):
    arc = tmp_path / "arc-map.json"
    arc.write_text(json.dumps([
        {"id": "f1", "what": "per-request 队列", "plant": "ch04", "payoff": "ch08", "status": "open"},
        {"id": "f2", "what": "DP wave", "plant": "ch21", "payoff": "ch21", "status": "open"},
    ]), encoding="utf-8")
    due = bible.due("ch04", arc_path=str(arc))
    assert any(x["id"] == "f1" for x in due["plant"])
    assert not due["payoff"]
    due8 = bible.due("ch08", arc_path=str(arc))
    assert any(x["id"] == "f1" for x in due8["payoff"])
```

- [ ] **Step 2: 运行确认失败**

Run: `python3 -m pytest scripts/tests/test_bible.py -v`
Expected: FAIL（import error）

- [ ] **Step 3: 实现 bible.py（含 due/foreshadow/term/iface）**

```python
# scripts/bible.py
#!/usr/bin/env python3
"""Book Bible — cross-chapter continuity store (terms, interfaces, foreshadow/payoff)."""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "instances/vllm/book/bible"
ARC = ROOT / "arc-map.json"
GLOSS = ROOT / "glossary.json"
IFACE = ROOT / "interfaces.json"

def _load(p, default):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

def _save(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def due(chapter_id: str, arc_path=ARC) -> dict:
    arc = _load(arc_path, [])
    return {
        "plant": [a for a in arc if a.get("plant") == chapter_id],
        "payoff": [a for a in arc if a.get("payoff") == chapter_id and a.get("status") != "resolved"],
    }

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("due"); d.add_argument("chapter_id")
    f = sub.add_parser("foreshadow"); f.add_argument("--add", action="store_true")
    f.add_argument("--plant", required=True); f.add_argument("--payoff", required=True); f.add_argument("--what", required=True)
    t = sub.add_parser("term"); t.add_argument("--add", nargs=2, metavar=("ZH", "EN"))
    i = sub.add_parser("iface"); i.add_argument("--add", nargs=2, metavar=("CHAPTER", "SIG"))
    a = ap.parse_args()
    if a.cmd == "due":
        res = due(a.chapter_id)
        print(f"== {a.chapter_id} 应埋伏笔 ==")
        for x in res["plant"]:
            print(f"  [{x['id']}] {x['what']} → 回收于 {x['payoff']}")
        print(f"== {a.chapter_id} 应回收 ==")
        for x in res["payoff"]:
            print(f"  [{x['id']}] {x['what']} （埋于 {x['plant']}）")
    elif a.cmd == "foreshadow" and a.add:
        arc = _load(ARC, [])
        nid = f"f{len(arc) + 1}"
        arc.append({"id": nid, "what": a.what, "plant": a.plant, "payoff": a.payoff, "status": "open"})
        _save(ARC, arc); print(f"added {nid}")
    elif a.cmd == "term" and a.add:
        g = _load(GLOSS, {}); g[a.add[0]] = a.add[1]; _save(GLOSS, g); print("ok")
    elif a.cmd == "iface" and a.add:
        it = _load(IFACE, {}); it.setdefault(a.add[0], []).append(a.add[1]); _save(IFACE, it); print("ok")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过 + 建空数据文件**

Run: `python3 -m pytest scripts/tests/test_bible.py -v`
Expected: 1 passed

```bash
mkdir -p instances/vllm/book/bible
echo "[]" > instances/vllm/book/bible/arc-map.json
echo "{}" > instances/vllm/book/bible/glossary.json
echo "{}" > instances/vllm/book/bible/interfaces.json
```

- [ ] **Step 5: 提交**

```bash
git add scripts/bible.py scripts/tests/test_bible.py instances/vllm/book/bible/
git commit -m "feat(bible): add cross-chapter continuity store CLI + data scaffolding"
```

---

## Task 4: 弧线设计（arc-map / glossary / voice-guide 内容）

**Files:**
- Modify: `instances/vllm/book/bible/arc-map.json`
- Create: `instances/vllm/book/bible/voice-guide.md`
- Modify: `instances/vllm/book/bible/glossary.json`

**Interfaces:**
- Consumes: `cartography/ARCHITECTURE.md`（19 步主线 + 依赖图）、`outline-final.json`。
- Produces: 开局设计的伏笔/回收弧线（至少覆盖 Part I–II，含 ch04），供 `bible.py due` 查询。

- [ ] **Step 1: 通读架构地图，列出贯穿全书的概念线**（如 per-request `RequestOutputCollector` 队列、token 预算、分页 block、persistent batch、IPC 进程边界）。

- [ ] **Step 2: 写 arc-map.json**（每条 `{id,what,plant,payoff,status:"open"}`）。ch04 必含至少：

```json
[
  {"id":"f1","what":"per-request RequestOutputCollector 队列（异步多路复用的关键）","plant":"ch04","payoff":"ch08","status":"open"},
  {"id":"f2","what":"EngineCore 进程边界 / IPC 是三段式解耦的物理前提","plant":"ch04","payoff":"ch07","status":"open"},
  {"id":"f3","what":"output_handler 背景 asyncio 任务与 generate() 的生产者-消费者关系","plant":"ch04","payoff":"ch08","status":"open"}
]
```

- [ ] **Step 3: 写 voice-guide.md**（叙述者声线=“白板边的内行朋友”；大白话+技术深度；公式前给直觉、公式后给数值；Cell 2 轻松→Cell 4 严谨→小结轻松；禁书面语清单）。

- [ ] **Step 4: 验证**

Run: `python3 scripts/bible.py due ch04`
Expected: 列出 f1/f2/f3 作为「应埋伏笔」。

- [ ] **Step 5: 提交**

```bash
git add instances/vllm/book/bible/
git commit -m "feat(bible): seed cross-chapter arc map, glossary, voice guide"
```

---

## Task 5: Roadmap 母版（svg-diagram）

**Files:**
- Create: `instances/vllm/book/assets/roadmap/roadmap.py`
- Produce: `instances/vllm/book/assets/roadmap/roadmap.svg` + `roadmap.png`

**Interfaces:**
- Consumes: 19 步主线 + 分层（`cartography/ARCHITECTURE.md` / `map.json`）。
- Produces: 一张全书母版图（请求生命周期主线 + 子系统分层），支持「高亮某章位置」参数，供各章 Roadmap 段复用。

- [ ] **Step 1: 读 svg-diagram skill**

Run: `cat .claude/skills/svg-diagram/SKILL.md`（遵循其 Python→xmllint→PNG 流程）。

- [ ] **Step 2: 写 roadmap.py**：按 L0→L6 分层画子系统盒 + 19 步主线箭头；接受 `--highlight <subsystem|step>` 高亮当前章；输出 SVG。（用嵌套矩形 + 有向箭头，正是 skill 适用的稠密多元素场景。）

- [ ] **Step 3: 生成并校验**

Run:
```bash
python3 instances/vllm/book/assets/roadmap/roadmap.py --highlight async-engine --out instances/vllm/book/assets/roadmap/roadmap.svg
xmllint --noout instances/vllm/book/assets/roadmap/roadmap.svg && echo "SVG VALID"
```
Expected: `SVG VALID`

- [ ] **Step 4: 转 PNG**

Run: `magick instances/vllm/book/assets/roadmap/roadmap.svg instances/vllm/book/assets/roadmap/roadmap.png` （或 skill 指定命令）

- [ ] **Step 5: 提交**

```bash
git add instances/vllm/book/assets/roadmap/
git commit -m "feat(roadmap): add reusable book-map master diagram generator"
```

---

## Task 6: Analyst 提示词（新角色，产出档案）

**Files:**
- Create: `.claude/agents/analyst.md`

**Interfaces:**
- Produces: 一个 agent 提示词；其产物是 `dossier.json`，schema 含 `code_spine`(file:Lxxx 范围)、`embed_excerpts`(**要内嵌的真实源码片段逐字 + 省略标记建议**)、`design_decisions`、`data_flow`、`theory`(推导/复杂度)、`subtraction_plan`(删什么/为什么/必须保留)、`diagram_plan`、`foreshadow_due`(来自 bible)。

- [ ] **Step 1: 写 analyst.md**，必须含以下**逐字契约**：
  - 「你的产物是 implementer 与 writer 的**共同唯一真相源**；二者都不以对方产物为准。」
  - 「先 `python3 scripts/bible.py due {chapter_id}` 取应埋/应回收条目并纳入档案。」
  - 「`embed_excerpts` 必须摘录**真实 vLLM 源码逐字片段**（带 `vllm/...:Lxxx`），标出可省略的无关分支——读者不开源码也能懂。」
  - 「`subtraction_plan` 明确：删什么、为什么安全、**哪些必须保留**（保留 vLLM 的同名/同结构/同控制流）。」
  - 「只描述真实源码，禁止建议任何 vLLM 没有的新抽象。」
- [ ] **Step 2: 自检**：提示词含 dossier schema 全字段 + bible 集成 + 内嵌片段要求。
- [ ] **Step 3: 提交**

```bash
git add .claude/agents/analyst.md
git commit -m "feat(agents): add analyst role producing the shared source dossier"
```

---

## Task 7: 重写 Implementer 提示词（只做减法 + TDD）

**Files:**
- Modify: `.claude/agents/implementer.md`（整体重写，可批判参考旧版）

**Interfaces:**
- Consumes: `dossier.json`（尤其 `subtraction_plan`）。
- Produces: `implementation/*.py`（subtract-only 精简版）+ `impl-notes.md`（Source Map 表）。

- [ ] **Step 1: 重写 implementer.md**，逐字契约：
  - 「**只做减法不做加法**：与 vLLM 同名、同结构、同控制流，只删不增。每处删除 `# SUBTRACTED: <删了什么·为什么·原 file:Lxxx>`；每 def/class `# SOURCE: vllm/...:Lxxx`。」
  - 「**禁止**：杜撰 vLLM 没有的抽象/数据结构、改名、无注释简化、加玩具模拟（如自动生成 token 的伪 forward）。」
  - 「验收判据：把 vLLM 删掉所有 SUBTRACTED 分支应当≈得到你的精简版。」
  - 「**TDD（test-driven-development skill）**：先按 dossier 记录的 vLLM 行为写测试，再实现到通过。」
  - 「收工前自跑 `python3 scripts/lint_fidelity.py {chapter_dir}` 必须无 BLOCKING。」
- [ ] **Step 2: 自检**：契约齐全、引用 dossier、含 fidelity 自检命令。
- [ ] **Step 3: 提交**

```bash
git add .claude/agents/implementer.md
git commit -m "refactor(agents): rewrite implementer for subtract-only fidelity + TDD"
```

---

## Task 8: 重写 Tester 提示词（验证闸门）

**Files:**
- Modify: `.claude/agents/tester.md`

**Interfaces:**
- Consumes: `implementation/`、`dossier.json`（期望行为）。
- Produces: `tests/test_*.py` + `tests/test-report.json`（`verdict` 字段为闸门真值）。

- [ ] **Step 1: 重写 tester.md**，契约：「行为对齐 dossier 记录的 vLLM 行为，而非只测精简版自洽」；「verification-before-completion skill：先跑命令看到实际输出再下结论，禁止橡皮图章」；「任一测试失败 → REJECTED 回 implementer（带 revision ledger 条目）」；「跑 `lint_fidelity` 作为附加闸门」；「**容器约束**：精简版纯单元测试 host `python3 -m pytest` 跑；任何 `import vllm`/CUDA/对照真实行为的验证用 `scripts/vllm_docker.sh -m pytest /work/<chapter>/tests`；`test-report.json` 记录 docker 命令 + 镜像 tag + vllm 版本」。
- [ ] **Step 2: 提交**

```bash
git add .claude/agents/tester.md
git commit -m "refactor(agents): rewrite tester as verification-first backpressure gate"
```

---

## Task 9: 重写 Writer 提示词（解读真源码 + 内嵌 + Roadmap）

**Files:**
- Modify: `.claude/agents/writer.md`

**Interfaces:**
- Consumes: `dossier.json`、`implementation/`、`bible.py due`、`roadmap.py`。
- Produces: `narrative/chapter.md`（含 Roadmap 段 + 内嵌真源码 + 精简版交叉验证）+ `diagrams/`。

- [ ] **Step 1: 重写 writer.md**，逐字契约：
  - 「**叙事主线是真实 vLLM 源码**。精简版只作‘剥掉 X/Y 分支后可运行的这几行’的交叉验证物，**不是主角**。若你需要大篇幅讲精简版，说明你解读不够或档案有缺——回 analyst/implementer，别硬写。」
  - 「**自包含**：直接内嵌真实源码片段（```python + `vllm/...:Lxxx`），删无关分支用 `# … 省略：… ` 标记。读者不开源码也能懂。」
  - 「**每章首个段落是 Roadmap**：调用 `roadmap.py --highlight` 出‘你在这里’图 + 上一章立了什么/本章解决什么/下一章接什么。」
  - 「写前 `python3 scripts/bible.py due {chapter_id}`：埋下应埋的伏笔、回收应回收项；写后回写 bible（新术语/接口/已埋/已回收）。」
  - 「收到 reviewer 反馈用 receiving-code-review skill：逐条采纳或带理由反驳，不表演式同意。」
  - 「**零脚手架泄漏（读者视角，正式出版物）**：引用源码用规范 vLLM 路径（`vllm/v1/engine/async_llm.py:L280`），**绝不**写 `instances/vllm/source/...`；章节用**自然标题**，**绝不**出现 'Cell N'；**绝不**提及内部文件（impl-notes.md/dossier/“详见 xxx.md/这里截取”）。」
  - 「收工前自跑 `lint_chapter_structure`、`lint_formulas`、`lint_source_grounding`、`lint_fidelity` 均无 BLOCKING。」
- [ ] **Step 2: 自检**：五项契约（主线/自包含/Roadmap/bible/lint）齐全。
- [ ] **Step 3: 提交**

```bash
git add .claude/agents/writer.md
git commit -m "refactor(agents): rewrite writer for real-source interpretation, self-contained embeds, roadmap"
```

---

## Task 10: 重写 Reviewer 提示词（协作式 + 保真度维度）

**Files:**
- Modify: `.claude/agents/reviewer.md`

**Interfaces:**
- Produces: `reviews/review-report.json`，schema 每条 issue 含 `{dimension, problem, suggested_fix, rationale, negotiable}`；顶层 `verdict`。

- [ ] **Step 1: 重写 reviewer.md**，契约：
  - 新增**首要维度 vLLM 保真度**：叙事是否在解读真源码（而非精简版）；精简版是否真子集；内嵌源码是否到位（自包含）；Roadmap 是否存在；对照 `bible` 检查应埋/应回收是否落实。
  - 「**协作式（requesting-code-review 精神）**：每个问题必须给 `suggested_fix + rationale`，标 `negotiable` 表示可与 writer 商榷；目标是合作共赢、共同做出完美作品，**不死板卡住 writer**。」
  - 「检查**零脚手架泄漏**：正文无 `instances/vllm/source` 路径、无 'Cell N' 标题、无对 impl-notes.md/dossier 等内部文件的引用、无“详见 xxx.md/这里截取”——读者完全不懂这些，正式出版物里也不存在。」
  - 「跑确定性 linter 作为客观依据；机械问题让 writer 定点小修，不退整章。」
  - 「>3 轮同一问题 → 升级 Team Lead。」
- [ ] **Step 2: 自检**：保真度维度 + 结构化建议 schema + 协作语气 + 升级阈值。
- [ ] **Step 3: 提交**

```bash
git add .claude/agents/reviewer.md
git commit -m "refactor(agents): rewrite reviewer as cooperative fidelity-first gate"
```

---

## Task 11: 重写 Archivist 提示词（Book Bible 守护 + 再水化）

**Files:**
- Modify: `.claude/agents/archivist.md`

**Interfaces:**
- Produces: trace 记录、`state.json` 更新、再水化简报；维护 `instances/vllm/book/bible/`。

- [ ] **Step 1: 重写 archivist.md**，契约：「你是唯一全书持久角色，是连贯性的守护者」；「维护 Book Bible（术语/接口/伏笔回收）」；「每章开工前发再水化简报（含本章 `bible.py due` 结果 + 前序章节摘要 + 用户反馈）」；「每完成一个 Part 触发 continuity-audit（后续计划实现）」。
- [ ] **Step 2: 提交**

```bash
git add .claude/agents/archivist.md
git commit -m "refactor(agents): rewrite archivist as continuity keeper of the Book Bible"
```

---

## Task 12: chapter-pipeline workflow（编排）

**Files:**
- Create: `.claude/workflows/chapter-pipeline.js`

**Interfaces:**
- Consumes: `args = {chapter_id, slug, source_root, focus, paths[]}`；agentType 用 Task 6–11 的角色。
- Produces: 端到端跑完一章（dossier→impl→test→narrative→review→archive），返回各阶段结果 + 闸门状态。

- [ ] **Step 1: 写 chapter-pipeline.js**，结构（plain JS，沿用 cartography workflow 模式）：
  - `meta`（name/description/phases: 档案/实现测试/叙事/评审/归档）。
  - **Phase A 档案**：`parallel` 扇出 4 个 analyst 子分析（code_spine ∥ theory ∥ subtraction_plan ∥ diagram_plan），各带 schema，合并为 dossier（barrier）；写 `dossier/dossier.json`。
  - **Phase B 实现+测试**：`agent(implementer, dossier)` → Bash 跑 `lint_fidelity` + 测试（精简版纯测试 host `pytest`；如需 `import vllm`/CUDA 则 `scripts/vllm_docker.sh -m pytest /work/...`）；失败把报告写入 revision ledger 重试 implementer，最多 3 轮。
  - **Phase C 叙事**：`agent(writer, {dossier, impl})` 产出 chapter.md；Bash 跑 `lint_chapter_structure/formulas/source_grounding/fidelity`。
  - **Phase D 评审**：`parallel` 扇出多维（保真∥可读∥算法∥公式）→ 合并 verdict + 协作反馈；REVISE 注入 ledger 回 writer，最多 3 轮，超限返回 ESCALATE。
  - **Phase E 归档**：`agent(archivist)` 记录 + 回写 bible。
- [ ] **Step 2: 语法校验**

Run: `node --check .claude/workflows/chapter-pipeline.js && echo "JS OK"`
Expected: `JS OK`

- [ ] **Step 3: 提交**

```bash
git add .claude/workflows/chapter-pipeline.js
git commit -m "feat(workflow): add per-chapter pipeline orchestration (dossier→impl→test→write→review→archive)"
```

---

## Task 13: ch04 试点脚手架

**Files:**
- Create: `instances/vllm/artifacts/ch04-async-llm/{dossier,implementation,tests,narrative,reviews,diagrams}/`
- Create: `instances/vllm/artifacts/ch04-async-llm/context.json`

**Interfaces:**
- Produces: ch04 章节目录骨架 + workflow 入参（focus=AsyncLLM 三段式；paths=async_llm.py 等）。

- [ ] **Step 1: 建目录骨架 + context.json**（chapter_id=ch04, slug=ch04-async-llm, subsystem=async-engine, deps=ch03；focus/paths 取自 `map.json` 的 async-engine 条目与 §11）。
- [ ] **Step 2: 确认 bible due 就绪**

Run: `python3 scripts/bible.py due ch04`
Expected: 列出 f1/f2/f3 应埋伏笔。

- [ ] **Step 3: 提交**

```bash
git add instances/vllm/artifacts/ch04-async-llm/
git commit -m "chore(ch04): scaffold AsyncLLM 3-stage pilot chapter"
```

---

## Task 14: 跑 ch04 流水线（集成验证）

**Files:**
- Produce: ch04 的 dossier/implementation/tests/narrative/reviews 全部产物。

**Interfaces:**
- Consumes: Task 12 workflow + Task 6–11 角色 + Task 1–5 闸门。

- [ ] **Step 1: 启动 workflow**

通过 Workflow 工具：`{name: "chapter-pipeline", args: {chapter_id:"ch04", slug:"ch04-async-llm", source_root:"instances/vllm/source", focus:"AsyncLLM 三段式异步解耦", paths:["vllm/v1/engine/async_llm.py","vllm/v1/engine/__init__.py"]}}`

- [ ] **Step 2: 闸门全绿验证**

Run:
```bash
D=instances/vllm/artifacts/ch04-async-llm
python3 -m pytest $D/tests/ -q
python3 scripts/lint_fidelity.py $D
python3 scripts/lint_chapter_structure.py $D/narrative/chapter.md
python3 scripts/lint_formulas.py $D/narrative/chapter.md
python3 scripts/lint_source_grounding.py $D
jq -r '.verdict' $D/reviews/review-report.json
```
Expected: pytest 通过；四个 linter 无 BLOCKING；verdict=APPROVED。

- [ ] **Step 3: 人工抽检**：叙事是否以真实 `async_llm.py` 为主线、内嵌源码自包含、Roadmap 到位、精简版仅作交叉验证（不喧宾夺主）。
- [ ] **Step 4: 提交**

```bash
git add instances/vllm/artifacts/ch04-async-llm/ instances/vllm/book/bible/
git commit -m "feat(ch04): produce AsyncLLM 3-stage chapter via new pipeline (pilot)"
```

---

## Task 15: 试点复盘 + 提示词迭代

**Files:**
- Modify: 相关 `.claude/agents/*.md`（按复盘结论）
- Create: `instances/vllm/trace/context_summaries/ch04-pilot-retro.md`

- [ ] **Step 1: 复盘**：脱节是否根除（writer 讲的是 vLLM 还是精简版）？自包含/Roadmap/伏笔是否到位？重做轮次？记入 retro。
- [ ] **Step 2: 据复盘改提示词**（架构师持续迭代职责）；若 fidelity 阈值需调，改 linter 常量 + 测试。
- [ ] **Step 3: 提交**

```bash
git add .claude/agents/ scripts/ instances/vllm/trace/
git commit -m "refactor(agents): iterate prompts from ch04 pilot retro"
```

---

## Task 16: 接线与旧体系弃用（非破坏性）

**Files:**
- Modify: `CLAUDE.md`（新流水线/角色/方法论）、`repo2book.json`（pipeline stages 加 analyst、artifacts 命名约定）
- Modify: 旧编排脚本头部加弃用注释（`scripts/{monitor,signal_pipeline,spawn_pipeline,hook_idle}.py` 若存在）

- [ ] **Step 1: 更新 CLAUDE.md**：写入「档案为真相源 / 只做减法 / 自包含内嵌 / Roadmap / 混合编排 / 两个新 linter / Book Bible」，并指向 spec 与本 plan。
- [ ] **Step 2: 更新 repo2book.json**：pipeline.stages 增加 `analyst`（置于 implementer 之前）；记录 ch-slug 命名约定。
- [ ] **Step 3: 旧脆弱脚本加弃用注释**（不删除）。
- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md repo2book.json scripts/
git commit -m "docs(system): wire new pipeline into CLAUDE.md/config, deprecate fragile orchestration scripts"
```

---

## 后续计划（不在本 plan）
- continuity-audit workflow（每完成一个 Part 并行扫全 Part 的伏笔回收/术语/接口）。
- 批量编排（多章并行 + worktree 隔离）。
- 旧 artifacts 实际清理（待新体系稳定）。

## Self-Review

- **Spec 覆盖**：§3 方法论→T6/7/9；§4 模板(Roadmap/内嵌)→T2/9；§5 角色→T6–11；§6 workflow→T12；§7.5 重做收敛→T12(ledger/重试/升级)；§7.6 连贯→T3/4/11；§7.7 协同→T12(结构化反馈)；§9 linter→T1/2；ch04 试点(§11)→T13/14/15。✓
- **占位符**：linter/bible 为完整可跑代码 + 测试；提示词任务给逐字契约而非"TBD"。✓
- **类型一致**：`lint_fidelity(chapter_dir)`、`lint_structure(md_path)`、`bible.due(chapter_id)` 在定义与调用处一致。✓
