#!/usr/bin/env python3
"""Chapter-structure linter — Roadmap present + self-contained embedded source + no scaffold leakage.

Usage: python3 lint_chapter_structure.py <chapter.md>
Exit 1 if any issue (all blocking).
"""
import re
import sys
from pathlib import Path

MIN_SOURCE_BLOCKS = 2

# 内嵌「真源码」路径前缀：活动实例的规范前缀(如 vllm_ascend) + 对照基座(如 vllm) + C/C++ 源目录(csrc)。
# 与 lint_fidelity.py / lint_source_grounding.py 一致——off-spine 实例章节的真源码不只是基座 vllm/。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json as _json
_PREFIXES = ["vllm", "csrc"]
try:
    import instance as _instance
    _p = _instance.canonical_prefix()
    if _p and _p not in _PREFIXES:
        _PREFIXES.append(_p)
    try:
        _root = _json.load(open(Path(__file__).resolve().parent.parent / "repo2book.json"))
        _dep = (_root.get("instances", {}).get(_instance.active_name(), {}) or {}).get("depends_on")
        if _dep:
            _bp = _json.load(open(Path(__file__).resolve().parent.parent / "instances" / _dep / "repo2book.json")).get("source", {}).get("canonical_prefix") or _dep
            if _bp and _bp not in _PREFIXES:
                _PREFIXES.append(_bp)
    except Exception:
        pass
except Exception:
    pass
# 块内含真源码路径标注（规范前缀 + 代码后缀）即算「内嵌真源码」
_SRC_PATH_RE = re.compile(
    r"(?:" + "|".join(re.escape(p) for p in sorted(set(_PREFIXES), key=len, reverse=True)) +
    r")/[\w./-]+\.(?:py|cpp|cc|cxx|h|hpp|cu)")
_CODE_FENCE_RE = re.compile(r"```(?:python|py|cpp|c\+\+|cc|cxx|c|cuda)\b.*?```", re.S | re.I)


def lint_structure(md_path: str) -> dict:
    text = Path(md_path).read_text(encoding="utf-8")
    res = {"no_roadmap": [], "no_embedded_source": [], "scaffold_leak": [], "halfwidth_punct": []}

    head = "\n".join(text.splitlines()[:60])
    if not re.search(r"(roadmap|路线图|你在这里)", head, re.I):
        res["no_roadmap"].append("  开头 60 行内无 Roadmap/路线图/你在这里 段")

    blocks = _CODE_FENCE_RE.findall(text)
    embedded = [b for b in blocks if _SRC_PATH_RE.search(b)]
    if len(embedded) < MIN_SOURCE_BLOCKS:
        res["no_embedded_source"].append(
            f"  内嵌真源码块仅 {len(embedded)}（需 >= {MIN_SOURCE_BLOCKS}，块内含规范源码路径标注，如 "
            f"{'/'.join(sorted(set(_PREFIXES)))}/…）")

    # 零脚手架泄漏（读者视角）：正文不得含本仓库脚手架痕迹
    scaffold = [
        (r"instances/[\w.-]+/source", "出现脚手架路径 instances/<instance>/source（应用规范源码路径，如 vllm/…）"),
        (r"\bCell\s*\d+\b", "出现 'Cell N' 脚手架标题（应用自然标题）"),
        (r"impl-notes\.md|dossier", "引用内部脚手架文件（impl-notes.md/dossier）"),
        (r"must_keep|subtraction_plan|embed_excerpt", "引用内部 dossier 机制术语（must_keep/subtraction_plan/embed_excerpt——读者视角不该出现）"),
        (r"详[见细]文档|完整文档见|这里只?截取", "提到出版物中不存在的外部文档/截取说明"),
    ]
    for pat, msg in scaffold:
        if re.search(pat, text):
            res["scaffold_leak"].append(f"  {msg}")

    # 中文之间误用半角逗号（应全角 '，'）；排除代码块
    no_code = re.sub(r'```.*?```', '', text, flags=re.S)
    for mm in re.finditer(r'[一-鿿],', no_code):
        ctx = no_code[max(0, mm.start() - 6):mm.start() + 2].replace('\n', ' ')
        res["halfwidth_punct"].append(f"  中文后误用半角逗号（应 '，'）：…{ctx}…")
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
