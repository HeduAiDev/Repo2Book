#!/usr/bin/env python3
"""Chapter-Map Linter — 「本章地图」SVG(源码剖面图)的确定性门禁。

检查(有图必核):
    1. SVG 文本里的 `§N.M` 徽标(必须带 § 前缀——裸 `N.M` 数字如延迟/版本号不算徽标)
       必须 ⊆ 正文 `## N.M` 标题集合，且 N = 本章目录号(目录名 chNN-slug 的 NN)——
       防图文漂移(改标题/重编号后忘重绘)。若本章是自然标题(无 `## N.M` 编号标题，
       heading_set 为空)而图上仍出现 §N.M 徽标，报错会明确指引：本章应改用标题词
       作站牌，禁用 §N.M 徽标。
    2. SVG 文本里形似代码符号的 token(含 `_`、`(` 或内部 `.`，len >= 4，如 `forward_impl`
       或 `attention.forward`)必须能在 dossier.json 原文或 chapter.md 正文里找到原样
       子串——防插画师杜撰不存在的符号。token 会先剥离尾部句点(`decode.`→`decode`)，
       `.` 触发仅当剥离后仍有内部 `.` 且后随字母/下划线——`decode.`/`e.g.`/`etc.` 这类
       自然语言收尾标点或缩写不入核对。
       dossier 顶层 "kind":"primer" 时，改核 book/papers/<chapter_dir_name>/*.md 论文包。

`--require`(试点期默认不开，铺开后 pipeline 里启用)额外核:
    3. 存在性：diagrams/chapter-map.svg 必须存在。
    4. 位置：chapter.md 第一个**内容分节**标题(`## ` 标题中排除标题文本匹配
       `roadmap|路线图|你在这里`——不区分大小写，口径同 lint_chapter_structure.py 的
       Roadmap 检测正则——的开篇导航标题，如 `## 你在这里`/`## N.1 你在这里`)之前必须
       有一处 `chapter-map.png` 的引用(开篇 hook 段之后、进入分节之前贴图；开篇导航
       标题本身可以在图之前或之后，不参与该判定)。
    5. 选读指引：该图引用之后 5 个非空行内必须出现 `§` 或 `节`(读者要知道能跳去哪)。

无 `--require` 且 diagrams/chapter-map.svg 不存在 → exit 0(试点期豁免，图还没画)。

Usage:
    python3 scripts/lint_chapter_map.py <chapter_dir> [--require]
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_BADGE_RE = re.compile(r'§(\d{1,2})\.(\d{1,2})')
_HEADING_RE = re.compile(r'^##\s+(\d{1,2})\.(\d{1,2})', re.M)
_TOKEN_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_.]{2,}\(?\)?')
_DOT_TRIGGER_RE = re.compile(r'\.[A-Za-z_]')
_CHNUM_RE = re.compile(r'^ch(\d{1,3})')
_IMG_REF_RE = re.compile(r'chapter-map\.png')
_GUIDANCE_RE = re.compile(r'[§节]')
# 开篇导航标题(如 `## 你在这里`/`## 24.1 你在这里`，常带 roadmap 图)——不算「内容分节」，
# 位置检查要跳过它才不会与全书 86% 章的开篇导航标题互斥。口径与
# lint_chapter_structure.py 的 Roadmap 检测正则一致(不区分大小写)。
_ALL_H2_RE = re.compile(r'^##[ \t]+(.*)$', re.M)
_NAV_HEADING_TEXT_RE = re.compile(r'(roadmap|路线图|你在这里)', re.I)


def _svg_text_segments(svg_path: Path):
    """解析 SVG，逐个 <text> 元素用 itertext() 聚合其下 <tspan> 拼出完整文本，
    元素之间各自独立(不跨元素合并)——既支持 tspan 拆行的徽标/符号，
    又不会把相邻两个 <text> 的内容意外粘连成新 token。

    pretty-print 风格的 SVG 里 <tspan> 常各占一行、带缩进空白(如 illustrator
    生成器输出)。早期实现对每段 strip 后直接无分隔拼接("")，这对"一个标识符
    被换行拆成两截"的场景能重新拼回完整词，但对"同一 <text> 里并列放着两个
    独立符号(如 forward_impl 和 _get_fia_params，各占一个 tspan)"的场景会把
    它们错误粘成一个从未出现过的假 token(forward_impl_get_fia_params)，误报
    成杜撰符号。

    改为空格拼接("" -> " "，且丢弃 strip 后为空的段，即缩进/换行本身不贡献
    任何内容)：并列独立符号之间保留天然的词边界，不再被粘连误判；至于确实被
    错误截断的标识符(如 "forward_i" / "mpl")，靠后续子串匹配的宽松语义兜底——
    "forward_i" 仍是正文 "forward_impl" 的合法子串，不会被误判杜撰，过短的碎片
    (如 "mpl")则被 len>=4 的门槛滤掉，不参与核对。段内本身的自然语言空格(如
    "latency 20.5 ms")不受影响，因为空段已被丢弃、非空段的空格由 join 分隔符
    统一负责。"""
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    segments = []
    for el in root.iter():
        tag = el.tag.split('}')[-1] if isinstance(el.tag, str) else ''
        if tag == 'text':
            segments.append(" ".join(seg.strip() for seg in el.itertext() if seg.strip()))
    return segments


def _dir_number(chapter_dir: Path):
    m = _CHNUM_RE.match(chapter_dir.name)
    return int(m.group(1)) if m else None


def _load_dossier(chapter_dir: Path):
    """dossier.json 可能在章目录根(试点期/本 linter 的 fixture 约定)或既有
    dossier/dossier.json 布局下——两处都探一下，找不到就返回 (None, "")。"""
    for cand in (chapter_dir / "dossier.json", chapter_dir / "dossier" / "dossier.json"):
        if cand.exists():
            raw = cand.read_text(encoding="utf-8", errors="replace")
            try:
                return json.loads(raw), raw
            except ValueError:
                return {}, raw
    return None, ""


def lint_chapter_map(chapter_dir: str, require: bool = False) -> dict:
    cd = Path(chapter_dir)
    res = {
        "missing_map": [],
        "badge_not_in_headings": [],
        "fabricated_symbol": [],
        "missing_opening_ref": [],
        "missing_skip_guidance": [],
    }

    svg_path = cd / "diagrams" / "chapter-map.svg"
    md_path = cd / "narrative" / "chapter.md"
    md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    if not svg_path.exists():
        if require:
            res["missing_map"].append(
                f"  {svg_path} 不存在——本章缺「本章地图」SVG（--require 已启用，试点豁免期已过）"
            )
        return res  # 豁免期：无图不核标题/符号/位置

    segments = _svg_text_segments(svg_path)

    # ── ① §徽标 ⊆ 正文标题集，且 N = 目录号 ──────────────────────────────
    dir_num = _dir_number(cd)
    heading_set = {(int(a), int(b)) for a, b in _HEADING_RE.findall(md_text)}
    seen_badges = set()
    for seg in segments:
        for a, b in _BADGE_RE.findall(seg):
            key = (int(a), int(b))
            if key in seen_badges:
                continue
            seen_badges.add(key)
            label = f"{a}.{b}"
            if not heading_set:
                res["badge_not_in_headings"].append(
                    f"  徽标 §{label}——本章为自然标题(无编号)，chapter-map 应用标题词作站牌，禁用 §N.M 徽标"
                )
            elif dir_num is not None and int(a) != dir_num:
                res["badge_not_in_headings"].append(
                    f"  徽标 §{label} 的章号 {a} 与本章目录号 ch{dir_num:02d} 不符——图挂错章"
                )
            elif key not in heading_set:
                res["badge_not_in_headings"].append(
                    f"  徽标 §{label} 在正文找不到对应 `## {label}` 标题——图文不一致（标题已改或图未更新）"
                )

    # ── ② 代码符号防杜撰 ────────────────────────────────────────────────
    dossier_doc, dossier_raw = _load_dossier(cd)
    if dossier_doc is not None and dossier_doc.get("kind") == "primer":
        pack_dir = cd.resolve().parent.parent / "book" / "papers" / cd.resolve().name
        pack_files = sorted(pack_dir.glob("*.md")) if pack_dir.exists() else []
        ground_texts = [p.read_text(encoding="utf-8", errors="replace") for p in pack_files]
        ground_texts.append(md_text)
    else:
        ground_texts = [dossier_raw, md_text]

    seen_tokens = set()
    for seg in segments:
        for m in _TOKEN_RE.finditer(seg):
            tok = m.group(0).rstrip('.')
            if not tok:
                continue
            if tok in seen_tokens:
                continue
            has_dot_trigger = bool(_DOT_TRIGGER_RE.search(tok))
            if "_" not in tok and "(" not in tok and not has_dot_trigger:
                continue
            if len(tok) < 4:
                continue
            seen_tokens.add(tok)
            if not any(tok in g for g in ground_texts):
                res["fabricated_symbol"].append(
                    f"  疑似杜撰符号 `{tok}`——既不是 dossier.json 原文子串也不在正文里出现"
                )

    # ── --require 专属:存在性已在上面处理;此处核位置 + 选读指引 ──────────
    if require:
        first_content_heading_pos = len(md_text)
        for hm in _ALL_H2_RE.finditer(md_text):
            if _NAV_HEADING_TEXT_RE.search(hm.group(1)):
                continue  # 开篇导航标题(你在这里/Roadmap)不算内容分节，跳过继续找
            first_content_heading_pos = hm.start()
            break
        opening_ref = next(
            (m for m in _IMG_REF_RE.finditer(md_text) if m.start() < first_content_heading_pos), None
        )
        if opening_ref is None:
            res["missing_opening_ref"].append(
                "  第一个内容分节标题(排除开篇导航标题「你在这里/Roadmap」)之前找不到 "
                "chapter-map.png 引用——本章地图应贴在开篇 hook 段之后、进入分节之前"
            )
        else:
            after = md_text[opening_ref.end():]
            nonblank = [ln for ln in after.splitlines() if ln.strip()][:5]
            if not any(_GUIDANCE_RE.search(ln) for ln in nonblank):
                res["missing_skip_guidance"].append(
                    "  本章地图引用之后 5 个非空行内无「§/节」选读指引——读者不知道能跳去哪"
                )

    return res


def print_report(res: dict, chapter_dir: str) -> int:
    total = sum(len(v) for v in res.values())
    print(f"Chapter-Map Lint: {chapter_dir}")
    print(f"{'=' * 60}")

    if total == 0:
        print("✓ 本章地图检查通过")
        return 0

    for check_name, issues in res.items():
        if issues:
            label = check_name.replace("_", " ").title()
            print(f"\n❌ {label} ({len(issues)} issue(s)):")
            for issue in issues:
                print(issue)

    print(f"\n{'=' * 60}")
    print(f"Total: {total} issue(s) found")
    print()
    print(f"✗ {total} 项违规——见上")
    return 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    require = "--require" in argv
    argv = [a for a in argv if a != "--require"]
    if len(argv) < 1:
        print("Usage: python3 lint_chapter_map.py <chapter_dir> [--require]")
        sys.exit(1)

    chapter_dir = argv[0]
    results = lint_chapter_map(chapter_dir, require=require)
    sys.exit(print_report(results, chapter_dir))
