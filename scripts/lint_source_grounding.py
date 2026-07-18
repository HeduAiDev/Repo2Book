#!/usr/bin/env python3
"""
Source Grounding Linter — verify chapter is anchored to vLLM source code.

Usage:
    python scripts/lint_source_grounding.py artifacts/02-kv-cache/

Checks:
    1. Chapter narrative has vLLM file:line references per section
    2. Implementation code has # REFERENCE: comments
    3. Source Mapping Table has 5+ rows
    4. impl-notes.md has vLLM file list
"""

import re, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import instance as _instance
    # 多根仓走 canonical_prefixes（Triton：python/ lib/ include/ third_party/ …）；单前缀仓退回 [canonical_prefix]。
    _SRC_PREFIXES = list(_instance.canonical_prefixes())
    # 姊妹篇：章节合法地同时引用本仓与对照基座源码，两者都算源码落点。
    try:
        import json as _json
        _root = _json.load(open(Path(__file__).resolve().parent.parent / "repo2book.json"))
        _dep = (_root.get("instances", {}).get(_instance.active_name(), {}) or {}).get("depends_on")
        if _dep:
            _bsrc = _json.load(open(
                Path(__file__).resolve().parent.parent / "instances" / _dep / "repo2book.json"
            )).get("source", {})
            for _bp in (_bsrc.get("canonical_prefixes") or [_bsrc.get("canonical_prefix") or _dep]):
                if _bp and _bp not in _SRC_PREFIXES:
                    _SRC_PREFIXES.append(_bp)
    except Exception:
        pass
except Exception:
    _SRC_PREFIXES = ["vllm"]


def lint_source_grounding(chapter_dir: str) -> dict:
    """Run all grounding checks."""
    base = Path(chapter_dir)
    results = {}

    narrative = base / "narrative" / "chapter.md"
    impl_dir = base / "implementation"
    impl_notes = impl_dir / "impl-notes.md"
    ctx_path = base / "context.json"

    # ── primer(原理章) 分支探测：dossier 顶层 "kind":"primer" ──
    # primer 章的 动机/推导 section 天然引用论文而非源码文件，该检查由
    # lint_paper_grounding 接管；这里只需放行、不再重复 BLOCK。
    dossier_path = base / "dossier" / "dossier.json"
    is_primer = False
    if dossier_path.exists():
        try:
            dossier_doc = json.loads(dossier_path.read_text(encoding="utf-8"))
            is_primer = dossier_doc.get("kind") == "primer"
        except ValueError:
            is_primer = False

    # ── Check 1: vLLM references in chapter narrative ──
    issues = []
    if narrative.exists():
        text = narrative.read_text(encoding="utf-8")
        sections = re.split(r'^## ', text, flags=re.MULTILINE)

        vllm_ref_pattern = re.compile(
            r'(?:vllm/)?[\w/]+\.(?:py|pyi|td|cpp|cc|cu|cuh|h|hpp)(?:[\s:]*L?\d+(?:-L?\d+)?)?', re.IGNORECASE
        )

        refs_per_section = {}
        for sec in sections:
            title = sec.split('\n')[0].strip() if sec.strip() else '(intro)'
            refs = vllm_ref_pattern.findall(sec)
            refs_per_section[title] = len(refs)

        # Meta sections that don't need source refs
        meta_patterns = [r'验证', r'总结', r'这章要做什么', r'^#\s*第\d+章']
        sections_without_refs = [
            t for t, n in refs_per_section.items()
            if n == 0 and not any(re.search(p, t) for p in meta_patterns)
        ]
        if sections_without_refs and not is_primer:
            issues.append(
                f"  Sections without vLLM source references: {sections_without_refs}"
            )
    results["narrative_vllm_refs"] = issues

    # ── Check 2: source-anchoring comments in implementation ──
    # 新体系 HARD RULE 用 `# SOURCE:`（lint_fidelity 校验）标注真实 vLLM 位置；
    # 兼容旧体系的 `# REFERENCE:`。两者任一即视为源码锚点。
    # primer 章的参考实现用 `# PAPER:` 锚论文（lint_paper_grounding 校验其位置），
    # 这里同样计入锚点数、不重复 BLOCK。
    issues = []
    ref_count = 0
    anchor_pattern = (
        r'#\s*(?:REFERENCE|SOURCE|PAPER):\s*(.+)' if is_primer
        else r'#\s*(?:REFERENCE|SOURCE):\s*(.+)'
    )
    if impl_dir.exists():
        # rglob（递归）而非 glob：多根仓 fork 的精简版按真实包树组织
        # （implementation/python/triton/…、implementation/third_party/ascend/…），
        # # SOURCE: 锚点落在嵌套 *.py 里；非递归 glob 只看顶层会全部漏掉。
        for py_file in impl_dir.rglob("*.py"):
            code = py_file.read_text(encoding="utf-8")
            refs = re.findall(anchor_pattern, code)
            ref_count += len(refs)
        if ref_count < 3:
            issues.append(
                f"  Only {ref_count} SOURCE/REFERENCE comments found (need >= 3)"
            )
    results["implementation_references"] = issues

    # ── Check 3: Source Mapping Table in impl-notes ──
    issues = []
    if impl_notes.exists():
        notes = impl_notes.read_text(encoding="utf-8")
        # Count rows in source mapping table
        table_rows = len(re.findall(r'^\|.*\|.*\|.*\|$', notes, re.MULTILINE))
        if table_rows < 5:
            issues.append(
                f"  Source Mapping Table has {table_rows} rows (need >= 5)"
            )
    results["source_mapping_table"] = issues

    # ── Check 4: vLLM source files referenced (发布正文为准，非内部 impl-notes.md) ──
    # 实例无关：用活动实例规范前缀 + 对照基座前缀（姊妹篇引用基座 vllm/ 也算）计源码文件
    alt = "|".join(re.escape(p) for p in _SRC_PREFIXES)
    # 含 MLIR/C++ 层后缀：Part V 起正文引用 .td/.cpp/.cc/.h，不再是清一色 .py。
    src_ref_re = rf'(?:{alt})/[\w/-]+\.(?:py|pyi|td|cpp|cc|cu|cuh|h|hpp)'
    issues = []
    if narrative.exists():
        text = narrative.read_text(encoding="utf-8")
        src_files = set(re.findall(src_ref_re, text))
        if len(src_files) < 3:
            issues.append(
                f"  Only {len(src_files)} source files referenced in narrative "
                f"({'/'.join(_SRC_PREFIXES)}; need >= 3)"
            )
    # chapter.md 尚不存在（写作前跑）：不判 FAIL，跳过——落到下面的 impl-notes 提示项自查即可。
    results["vllm_files_listed"] = issues

    # ── Check 4b: impl-notes.md 源文件登记完整性（非阻断提示，供 implementer 自查）──
    issues = []
    if impl_notes.exists():
        notes = impl_notes.read_text(encoding="utf-8")
        src_files = set(re.findall(src_ref_re, notes))
        if len(src_files) < 3:
            issues.append(
                f"  [info] impl-notes.md only lists {len(src_files)} source files "
                f"({'/'.join(_SRC_PREFIXES)}; need >= 3) — internal bookkeeping only, "
                f"does not affect narrative grounding"
            )
    results["impl_notes_incomplete"] = issues

    return results


def print_report(results: dict, chapter_dir: str):
    total = sum(len(v) for v in results.values())
    print(f"Source Grounding Lint: {chapter_dir}")
    print(f"{'=' * 60}")

    if total == 0:
        print("✓ All grounding checks passed!")
        return

    for check, issues in results.items():
        if issues:
            print(f"\n❌ {check} ({len(issues)} issue(s)):")
            for issue in issues:
                print(issue)

    blocking = len(results["narrative_vllm_refs"]) + len(results["implementation_references"])
    print(f"\n{'=' * 60}")
    if blocking > 0:
        print(f"🔴 {blocking} BLOCKING issue(s) — auto-REJECT")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lint_source_grounding.py <chapter_dir>")
        sys.exit(1)
    results = lint_source_grounding(sys.argv[1])
    print_report(results, sys.argv[1])
