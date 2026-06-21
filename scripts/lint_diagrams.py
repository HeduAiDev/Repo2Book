#!/usr/bin/env python3
"""Diagram-quality linter — 机械检查一章的 diagrams/。

阻断项：
  - SVG 未过 xmllint
  - SVG 无对应 PNG / PNG 过小（疑似空）
  - 图 PNG 未在 narrative/chapter.md 引用（孤儿图）
  - 含中文(CJK)的 text 的 font-family 非 "CJK 安全字体" —— PNG 渲染器(ImageMagick)
    对 sans-serif/monospace 不做逐字 CJK 回退，会整段丢中文字形。
    经隔离验证：含 CJK 的 text 必须**单独**用一个已知 CJK 字体（如 "Droid Sans Fallback"），
    且**不能**与 latin 族（sans-serif/monospace）逗号并列（一并列就又丢字）。bold 无用。
警告项：text 估算超出 viewBox。

用法：python3 lint_diagrams.py <chapter_dir>　阻断项存在则 exit 1。
"""
import re
import subprocess
import sys
from pathlib import Path

CJK = re.compile(r'[㐀-䶿一-鿿　-〿＀-￯]')
# 已知含 CJK 字形的字体（前缀匹配）；可按部署环境扩充
CJK_FONTS = ("Droid Sans Fallback", "Noto Sans CJK", "Noto Serif CJK",
             "Source Han", "WenQuanYi", "AR PL", "Microsoft YaHei", "SimHei", "SimSun")


def _cjk_font_safe(font_family: str) -> bool:
    fam = (font_family or "").strip()
    if not fam or "," in fam:  # 未设 或 与 latin 族并列 → 不安全
        return False
    return any(fam == f or fam.startswith(f) for f in CJK_FONTS)


def lint_diagrams(chapter_dir: str) -> dict:
    d = Path(chapter_dir)
    dia = d / "diagrams"
    res = {"svg_invalid": [], "png_missing": [], "orphan": [], "cjk_font": [], "overflow": []}
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
        for tm in re.finditer(r'<text\b([^>]*)>(.*?)</text>', text, re.S):
            attrs, content = tm.group(1), tm.group(2)
            if CJK.search(content):
                fm = re.search(r'font-family="([^"]*)"', attrs)
                fam = fm.group(1) if fm else ""
                if not _cjk_font_safe(fam):
                    res["cjk_font"].append(
                        f"  {svg.name}: 含中文 text «{content[:24]}» font-family='{fam or '(未设)'}' 非 CJK 安全"
                        f" — PNG 会丢字形（单独设 font-family=\"Droid Sans Fallback\"，勿与 latin 族逗号并列）")
            if vbw:
                xm = re.search(r'\bx="([\d.\-]+)"', attrs)
                fsm = re.search(r'font-size="([\d.]+)"', attrs)
                if xm and fsm:
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
        print("✓ 图示检查通过（SVG 有效 / PNG 在位且被引用 / 中文用 CJK 安全字体）")
        return 0
    for k, issues in res.items():
        for i in issues:
            print(f"❌ {k}: {i}")
    blocking = (len(res["svg_invalid"]) + len(res["png_missing"])
                + len(res["orphan"]) + len(res["cjk_font"]))
    print(f"\n{'=' * 60}")
    print(f"🔴 {blocking} BLOCKING" if blocking else "🟢 仅警告（overflow 估算）")
    return 1 if blocking else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lint_diagrams.py <chapter_dir>")
        sys.exit(1)
    sys.exit(print_report(lint_diagrams(sys.argv[1]), sys.argv[1]))
