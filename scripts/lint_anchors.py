#!/usr/bin/env python3
"""章内锚点校验：检查每章 narrative/chapter.md 里的 `[文字](#锚点)` 章内链接
是否能解析到本章某个标题的 GitHub 风格 slug。跨章链接(../chNN/...)不查。

GitHub slug 规则(近似): 小写 → 去掉非「字母/数字/CJK/下划线/空格/连字符」的字符
(标点含 CJK 的 、，：（）→ 等被直接删除、不转连字符) → 空格转连字符。

用法: python3 scripts/lint_anchors.py <chapter.md> [更多...]
      python3 scripts/lint_anchors.py --all      # 扫当前活动实例的全部章节（见 scripts/instance.py）
退出码 1 = 有断锚。
"""
import re
import sys
import glob
import pathlib
import instance

HEADING = re.compile(r'^#{1,6}\s+(.*?)\s*$', re.M)
# 章内锚点链接: [text](#anchor) —— 排除 ](../  跨章 与 ](http
INCHAPTER = re.compile(r'\]\(#([^)]+)\)')

# 跨章链接: [text](../../chNN-slug/...) 或旧写法 [text](../chNN-slug/...)
CROSSLINK = re.compile(r'\[([^\]]*)\]\(((?:\.\./)+)(ch\d{2})-([\w\-]+)/[^)]*\)')
# 裸文字章号(无链接): "第 N 章"
BARENUM = re.compile(r'第\s*(\d{1,3})\s*章')
# 章目录名开头的两位章号，如 "ch20-foo" → 20
CHAPTER_DIR_NUM = re.compile(r'^ch(\d{2})')

# 行文节号「N.M」精度优先版(仅两种模式触发，见 check_stale_section_prefix docstring)：
# ① 「§N.M」：§ 前缀恒为节引用。
STALE_SECTION_MARK = re.compile(r'§(\d{1,2})\.(\d{1,2})(?![\d.])')
# ② 裸「N.M」候选(排除版本号 v0.21.0 / 小数连写 3.5.1 / 紧邻 §)——是否 warn 还要看上下文(见下)。
STALE_SECTION_BARE = re.compile(r'(?<![\dvV.§])(\d{1,2})\.(\d{1,2})(?![\d.])')
# 裸「N.M」的后缀语境：紧邻 0-1 个空格 + 「节」/「小节」。
_SECTION_SUFFIX_CTX = re.compile(r'^\s?(?:小节|节)(?![一-鿿])')  # 节 后接 CJK 即复合词(节拍/节点),不算节引用
# 裸「N.M」的前缀语境："见 "/"详见 "/"参见 "/"接 "(该词自带 1 个空格，允许再多带 1 个)。
_SECTION_PREFIX_CTX = re.compile(r'(?:见|详见|参见|接)\s{1,2}$')


def slugify(heading: str) -> str:
    s = heading.strip().lower()
    s = s.replace('`', '')                       # 去行内代码反引号
    # GitHub slug 保留下划线; 强调标记 * ~ 会被下面的非\w\s\- 删除规则一并去掉。
    # 保留 unicode 字母(含 CJK)/数字/下划线/空格/连字符，其余(标点 含 *~)删除
    s = re.sub(r'[^\w\s\-]', '', s, flags=re.UNICODE)
    s = s.replace(' ', '-')
    return s


def check(path: str):
    try:
        text = open(path, encoding='utf-8').read()
    except OSError as e:
        return [f'打不开: {e}']
    # 收集标题 slug（GitHub 对重复 slug 加 -1/-2，这里简化为集合）
    slugs = set()
    for m in HEADING.finditer(text):
        slugs.add(slugify(m.group(1)))
    broken = []
    for m in INCHAPTER.finditer(text):
        anchor = m.group(1)
        if anchor not in slugs:
            # 行号
            ln = text[:m.start()].count('\n') + 1
            broken.append(f'L{ln}: #{anchor}  ← 无匹配标题')
    return broken


def check_cross(path: str):
    """跨章链接三规 + 裸章号提示：
    - broken: 链接目标章目录不存在(BLOCKING)
    - num_mismatch: 链接文字「第 N 章」与目标目录号不符(BLOCKING)
    - bad_depth: 单层相对路径 ](../chNN- 旧写法，narrative/ 出发须 ../../(BLOCKING)
    - bare: 裸文字章号(不在任何链接内)，无链接可跳转(WARN)
    """
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    arts = p.resolve().parents[2]          # narrative/ → 章目录 → artifacts/
    res = {"broken": [], "num_mismatch": [], "bad_depth": [], "bare": []}
    spans = []
    for m in CROSSLINK.finditer(text):
        spans.append(m.span())
        label, dots, cid, slug = m.group(1), m.group(2), m.group(3), m.group(4)
        if dots == "../":
            res["bad_depth"].append(f"  {p.name}: 单层相对路径 ]({dots}{cid}-…(narrative/ 出发须 ../../)")
        if not (arts / f"{cid}-{slug}").is_dir():
            res["broken"].append(f"  {p.name}: 悬空跨章链接 → {cid}-{slug}")
        nm = BARENUM.search(label)
        if nm and int(nm.group(1)) != int(cid[2:]):
            res["num_mismatch"].append(
                f"  {p.name}: 链接文字「第 {nm.group(1)} 章」≠ 目标目录 {cid}")
    for m in BARENUM.finditer(text):
        if not any(s <= m.start() < e for s, e in spans):
            res["bare"].append(f"  {p.name}: 裸文字章号「{m.group(0)}」无链接(warn)")
    return res


def check_stale_section_prefix(chapter_dir: str, all_chapter_nums: set):
    """行文节号前缀异常(WARN，不计退出码，同既有裸章号 warn 通道)：精度优先版。

    仅以下两种模式触发候选(其余裸『N.M』——数值/版本/比率——一律不 warn)：
      ① 「§N.M」形式(§ 前缀恒为节引用，近零假阳)；
      ② 裸「N.M」仅当紧邻节引用语境：后接 0-1 空格+『节』/『小节』，
         或前有『见 』/『详见 』/『参见 』/『接 』(允许其后再带一个空格)。

    命中候选后仍须满足——
      - N 命中实例现存章号集合(all_chapter_nums)
      - N ≠ 本章目录号(不是"本章 N.M"这种自引用)
      - 该行没有指向 ch{N:02d}- 的跨章链接(不是正当的"见第 N 章…N.M 节"引用)
    —— 视为交错重编号(chapter interleave/renumber)后遗留的旧节号，提示人核。

    跳过标题行/代码围栏内/图片行，避免误伤代码字面量或图注。
    primer 原理章(目录名含 -primer-)整章豁免：其 §N.M 几乎全指论文自身章节
    (合法引用，由 lint_paper_grounding 管辖)，非交错残留。
    """
    p = pathlib.Path(chapter_dir)
    if "-primer-" in p.name:
        return []
    md = p / "narrative" / "chapter.md"
    if not md.is_file():
        return []
    m = CHAPTER_DIR_NUM.match(p.name)
    own_num = int(m.group(1)) if m else None
    text = md.read_text(encoding="utf-8", errors="replace")
    warns = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("#") or stripped.startswith("!["):
            continue

        def _emit(sm):
            n = int(sm.group(1))
            if n not in all_chapter_nums or n == own_num:
                return
            if f"ch{n:02d}-" in line:
                return
            warns.append(
                f"  {p.name}: L{lineno} 节号「{sm.group(0)}」疑似交错重编号残留"
                f"(ch{n:02d} 无本行链接，人核是否应更新/应为本章节号)(warn)")

        for sm in STALE_SECTION_MARK.finditer(line):
            _emit(sm)
        for sm in STALE_SECTION_BARE.finditer(line):
            start, end = sm.span()
            if _SECTION_PREFIX_CTX.search(line[:start]) or _SECTION_SUFFIX_CTX.match(line[end:]):
                _emit(sm)
    return warns


def main():
    args = sys.argv[1:]
    if args == ['--all'] or not args:
        files = sorted(glob.glob(instance.chapters_glob()))
    else:
        files = args
    total = 0
    cross_total = 0
    warn_total = 0
    stale_total = 0
    nums_cache = {}
    for f in files:
        broken = check(f)
        ch = f.split('/')[-3] if '/' in f else f
        if broken:
            total += len(broken)
            print(f'❌ {ch}: {len(broken)} 处断锚')
            for b in broken:
                print(f'    {b}')
        cross = check_cross(f)
        blocking = cross['broken'] + cross['num_mismatch'] + cross['bad_depth']
        if blocking:
            cross_total += len(blocking)
            print(f'❌ {ch}: {len(blocking)} 处跨章链接问题(悬空/文字号不符/单层旧路径)')
            for b in blocking:
                print(f'    {b}')
        if cross['bare']:
            warn_total += len(cross['bare'])
            print(f'⚠️  {ch}: {len(cross["bare"])} 处裸文字章号(无链接)')
            for b in cross['bare']:
                print(f'    {b}')
        chapter_dir = pathlib.Path(f).resolve().parent.parent
        arts = chapter_dir.parent
        if arts not in nums_cache:
            n = set()
            for d in arts.glob('ch??-*'):
                m = CHAPTER_DIR_NUM.match(d.name)
                if m:
                    n.add(int(m.group(1)))
            nums_cache[arts] = n
        stale = check_stale_section_prefix(str(chapter_dir), nums_cache[arts])
        if stale:
            stale_total += len(stale)
            print(f'⚠️  {ch}: {len(stale)} 处行文节号前缀异常(疑似交错重编号残留)')
            for s in stale:
                print(f'    {s}')
    if total == 0 and cross_total == 0:
        print('✓ 全部章内锚点都能解析')
    if total:
        print(f'\n共 {total} 处断锚')
    if cross_total:
        print(f'共 {cross_total} 处跨章链接问题(悬空/文字号不符/单层旧路径，BLOCKING)')
    if warn_total:
        print(f'⚠️  另有 {warn_total} 处裸文字章号(warn，不计入退出码)')
    if stale_total:
        print(f'⚠️  另有 {stale_total} 处行文节号前缀异常(warn，不计入退出码)')
    return 1 if (total or cross_total) else 0


if __name__ == '__main__':
    sys.exit(main())
