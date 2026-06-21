#!/usr/bin/env python3
"""Chapter-structure linter — Roadmap present + self-contained embedded source + no scaffold leakage.

Usage: python3 lint_chapter_structure.py <chapter.md>
Exit 1 if any issue (all blocking).
"""
import re
import sys
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
        res["no_embedded_source"].append(
            f"  内嵌真源码块仅 {len(embedded)}（需 >= {MIN_SOURCE_BLOCKS}，块内含 vllm/ 路径标注）")

    # 零脚手架泄漏（读者视角）：正文不得含本仓库脚手架痕迹
    scaffold = [
        (r"instances/vllm/source", "出现脚手架路径 instances/vllm/source（应用规范 vllm/ 路径）"),
        (r"\bCell\s*\d+\b", "出现 'Cell N' 脚手架标题（应用自然标题）"),
        (r"impl-notes\.md|dossier", "引用内部脚手架文件（impl-notes.md/dossier）"),
        (r"详[见细]文档|完整文档见|这里只?截取", "提到出版物中不存在的外部文档/截取说明"),
    ]
    for pat, msg in scaffold:
        if re.search(pat, text):
            res["scaffold_leak"].append(f"  {msg}")
    return res


def print_report(res: dict, path: str) -> int:
    total = sum(len(v) for v in res.values())
    print(f"Chapter-Structure Lint: {path}\n{'=' * 60}")
    if total == 0:
        print("✓ 结构检查通过（Roadmap + 自包含源码 + 零脚手架泄漏）")
        return 0
    for k, issues in res.items():
        for i in issues:
            print(f"❌ {k}: {i}")
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_chapter_structure.py <chapter.md>")
        sys.exit(1)
    sys.exit(print_report(lint_structure(sys.argv[1]), sys.argv[1]))
