#!/usr/bin/env python3
"""Diagram-quality linter — 机械检查一章的 diagrams/。

渲染约定：PNG 由 **rsvg-convert**（librsvg/Pango）生成——它对 sans-serif 做逐字 CJK 回退
（latin 用 latin 字体、中文自动回退到系统 CJK 字体），排版正确。**不要**用 ImageMagick
convert（不做 CJK 回退、排版差），也**不要**在 SVG 里强制 CJK 字体（会把 latin 也汉化、错位）。

阻断项：SVG 未过 xmllint；SVG 无对应 PNG / PNG 过小；图 PNG 未在 narrative 引用（孤儿图）；
        环境缺 rsvg-convert（无法正确出图）。
警告项：text 估算超出 viewBox。

用法：python3 lint_diagrams.py <chapter_dir>　阻断项存在则 exit 1。
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

CJK = re.compile(r'[㐀-䶿一-鿿]')


def lint_diagrams(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    dia = d / "diagrams"
    res = {"svg_invalid": [], "png_missing": [], "orphan": [], "no_renderer": [], "overflow": []}
    if shutil.which("rsvg-convert") is None:
        res["no_renderer"].append("  缺 rsvg-convert（apt install librsvg2-bin）——PNG 无法正确渲染中文")
    if not dia.exists():
        return res
    nar = d / "narrative" / "chapter.md"
    nar_text = nar.read_text(encoding="utf-8") if nar.exists() else ""

    for svg in sorted(dia.glob("*.svg")):
        if subprocess.run(["xmllint", "--noout", str(svg)], capture_output=True).returncode != 0:
            res["svg_invalid"].append(f"  {svg.name}: xmllint 失败")
        text = svg.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', text)
        vbw = float(m.group(1)) if m else None
        if vbw:
            for tm in re.finditer(r'<text\b([^>]*)>(.*?)</text>', text, re.S):
                attrs, content = tm.group(1), tm.group(2)
                xm = re.search(r'\bx="([\d.\-]+)"', attrs)
                fsm = re.search(r'font-size="([\d.]+)"', attrs)
                if not (xm and fsm):
                    continue
                x, fs = float(xm.group(1)), float(fsm.group(1))
                stripped = re.sub(r'\s', '', content)
                wd = sum(1.0 if CJK.match(c) else 0.55 for c in stripped) * fs
                if 'text-anchor="middle"' in attrs:
                    left, right = x - wd / 2, x + wd / 2
                elif 'text-anchor="end"' in attrs:
                    left, right = x - wd, x
                else:
                    left, right = x, x + wd
                if left < -2 or right > vbw + 2:
                    res["overflow"].append(
                        f"  {svg.name}: text «{content[:18]}» 估算 [{left:.0f},{right:.0f}] 超出 viewBox 0..{vbw:.0f}")
        if not (dia / (svg.stem + ".png")).exists():
            res["png_missing"].append(f"  {svg.stem}.png 缺失（有 SVG 无 PNG）")

    for png in sorted(dia.glob("*.png")):
        if png.stat().st_size < 2048:
            res["png_missing"].append(f"  {png.name}: PNG 过小/疑似空 ({png.stat().st_size}B)")
        if png.name not in nar_text:
            res["orphan"].append(f"  {png.name}: 未在正文引用（孤儿图）")
    return res


def print_report(res: dict, cd: str) -> int:
    total = sum(len(v) for v in res.values())
    print(f"Diagram Lint: {cd}\n{'=' * 60}")
    if total == 0:
        print("✓ 图示检查通过（SVG 有效 / PNG 在位且被引用 / rsvg 渲染器就绪）")
        return 0
    for k, issues in res.items():
        for i in issues:
            print(f"❌ {k}: {i}")
    blocking = (len(res["svg_invalid"]) + len(res["png_missing"])
                + len(res["orphan"]) + len(res["no_renderer"]))
    print(f"\n{'=' * 60}")
    print(f"🔴 {blocking} BLOCKING" if blocking else "🟢 仅警告（overflow 估算）")
    return 1 if blocking else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_diagrams.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_diagrams(sys.argv[1]), sys.argv[1]))
