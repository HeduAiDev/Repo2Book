#!/usr/bin/env python3
"""Fidelity linter — enforces the subtract-only companion contract.

Checks (blocking unless noted):
  1. Every top-level def/class in implementation/*.py has a `# SOURCE: vllm/...` ref in its span.
  2. No invention markers (# ADDED / # TOY / # FAKE / # INVENTED).
  3. Narrative grounds in real vLLM: count of `vllm/...py` refs >= refs to `implementation/`,
     and >= MIN_VLLM_REFS.
  4. (warning) at least one `# SUBTRACTED:` marker present.

Usage: python3 lint_fidelity.py <chapter_dir>
Exit 1 if any blocking issue.
"""
import ast
import json
import re
import sys
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
    res = {"missing_source": [], "invention": [], "narrative_grounding": [],
           "over_subtraction": [], "no_subtraction": []}
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
            res["narrative_grounding"].append(
                f"  叙事引用精简版({comp_refs}) 多于真实 vllm/({vllm_refs}) — 喧宾夺主")

    # 过度删减/误删：dossier 声明的 must_keep 符号必须出现在精简版
    dossier = d / "dossier" / "dossier.json"
    if dossier.exists() and pyfiles:
        impl_text = "\n".join(p.read_text(encoding="utf-8") for p in pyfiles)
        try:
            doss = json.loads(dossier.read_text(encoding="utf-8"))
            must_keep = (doss.get("subtraction_plan") or {}).get("must_keep") or []
        except (ValueError, AttributeError):
            must_keep = []
        for entry in must_keep:
            sym = entry.get("symbol") if isinstance(entry, dict) else entry
            if not sym:
                continue
            leaf = str(sym).split(".")[-1]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", leaf) and leaf not in impl_text:
                res["over_subtraction"].append(
                    f"  must_keep 符号 `{sym}` 未出现在精简版（疑似过度删减/误删）")
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
    blocking = (len(res["missing_source"]) + len(res["invention"])
                + len(res["narrative_grounding"]) + len(res["over_subtraction"]))
    print(f"\n{'=' * 60}")
    print(f"🔴 {blocking} BLOCKING" if blocking else "🟢 仅警告（no_subtraction）")
    return 1 if blocking else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_fidelity.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_fidelity(sys.argv[1]), sys.argv[1]))
