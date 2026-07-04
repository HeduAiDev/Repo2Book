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


def lint_paper_grounding(chapter_dir: str, expect_primer: bool = False) -> dict:
    d = Path(chapter_dir)
    res = {"impl": [], "citation": [], "formula": [], "paper_ref": [], "warn": [], "expect": []}
    df = d / "dossier" / "dossier.json"
    try:
        doc = json.loads(df.read_text(encoding="utf-8")) if df.exists() else {}
    except ValueError:
        doc = {}
    if doc.get("kind") != "primer":
        if expect_primer:
            res["expect"].append(
                '  期望 primer 章但 dossier 顶层缺 "kind":"primer"(lint 分流开关)——analyst 须补写'
            )
        res["warn"].append("  非 primer 章(dossier 顶层无 kind:primer)——本检查跳过")
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
            po = mech.get("paper_origin")
            if not isinstance(po, dict):
                continue
            for s in po.get("sections") or []:
                key = s.replace("§", "").replace("Eq.", "").strip()
                if key and key not in ptext:
                    res["paper_ref"].append(f"  {mech.get('id')}: 小节 {s} 在论文包里找不到")
    return res


def print_report(res: dict, cd: str) -> int:
    print(f"Paper-Grounding Lint: {cd}\n{'=' * 60}")
    blocking = len(res["impl"]) + len(res["citation"]) + len(res.get("expect", []))
    for k, issues in res.items():
        mark = "❌ " if k in ("impl", "citation", "expect") else "⚠️ "
        for i in issues:
            print(mark + f"{k}: {i}")
    if blocking == 0:
        print("✓ 无 BLOCKING")
        return 0
    print(f"\n{'=' * 60}\n🔴 {blocking} BLOCKING")
    return 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    expect_primer = "--expect-primer" in argv
    argv = [a for a in argv if a != "--expect-primer"]
    if len(argv) < 1:
        print("Usage: python3 lint_paper_grounding.py <chapter_dir> [--expect-primer]")
        sys.exit(1)
    sys.exit(print_report(lint_paper_grounding(argv[0], expect_primer=expect_primer), argv[0]))
