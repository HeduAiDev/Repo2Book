#!/usr/bin/env python3
"""图面脚手架泄漏 linter — 扫渲染出的 SVG <text> 节点,禁止内部工件路径印上图。

HARD RULE 3(零脚手架泄漏)对正文由 lint_chapter_structure 把关,但**图注/副标题里
渲染的内部路径**(如 `traces/dump_ir.json`、`explainer/traces/*.md`、`dossier.json`、
`impl-notes`)linter 一直漏检——正文 linter 只扫 markdown、不扫图内渲染文字。本 linter
补这个洞:只看 SVG <text> 节点(=真正印在图上的字),命中内部工件路径即阻断。

允许:规范源码路径(.py/.cpp/.h/.td/.cc/.cu + 行号)、真实仓库文件(CMakeLists.txt、
README.md 等)、reader 可见的出处措辞(如「Triton v3.2.0 headless 实测」)。
禁止:`traces/…`、`explainer/…`、`dossier`、`impl-notes`、`instances/<x>/source/…` 前缀、
以及裸的中间数据文件名(`*.json`/`*.txt` 当且仅当带 traces/explainer 前缀)。

用法:python3 lint_diagram_scaffolding.py <chapter_dir>|--all   命中则 exit 1。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from instance import active_instance_dir  # type: ignore
except Exception:
    active_instance_dir = None

# 内部工件路径 token(渲染进图即泄漏)。故意不抓裸 *.json/*.txt(那可能是 reader 语境),
# 只抓带 traces//explainer/ 前缀的中间数据,以及 dossier/impl-notes/…/source 脚手架标识。
LEAK = re.compile(
    r'traces/[\w./-]+'
    r'|explainer/[\w./-]+'
    r'|\bimpl-notes\b'
    r'|\bdossier\.json\b'
    r'|instances/[\w-]+/source/'
)
TEXTNODE = re.compile(r'<text[^>]*>(.*?)</text>', re.S)
TAG = re.compile(r'<[^>]+>')


def scan_svg(svg: Path):
    try:
        s = svg.read_text(encoding='utf-8')
    except Exception:
        return []
    hits = []
    for m in TEXTNODE.finditer(s):
        body = TAG.sub('', m.group(1))
        for lm in LEAK.finditer(body):
            hits.append(lm.group(0))
    return sorted(set(hits))


def lint_dir(chapter_dir: Path):
    issues = []
    for svg in sorted(chapter_dir.glob('diagrams/*.svg')):
        h = scan_svg(svg)
        if h:
            issues.append((svg, h))
    return issues


def main():
    args = sys.argv[1:]
    targets = []
    if '--all' in args:
        base = active_instance_dir() if active_instance_dir else None
        root = (base / 'artifacts') if base else Path('instances/triton/artifacts')
        targets = [p for p in sorted(root.glob('ch*')) if p.is_dir()]
    else:
        targets = [Path(a) for a in args if not a.startswith('--')]
    total = 0
    print(f"Diagram-Scaffolding Lint: {'--all' if '--all' in args else ' '.join(map(str, targets))}")
    print('=' * 60)
    for t in targets:
        for svg, hits in lint_dir(t):
            total += len(hits)
            rel = str(svg).split('/artifacts/')[-1]
            print(f"❌ {rel}: 图面渲染内部工件路径 {hits}")
    if total == 0:
        print("✓ 图面无脚手架路径泄漏")
        return 0
    print(f"\n{'=' * 60}\n🔴 {total} BLOCKING(图面脚手架泄漏——改 gen 脚本渲染文字去掉内部路径、重渲染)")
    return 1


if __name__ == '__main__':
    sys.exit(main())
