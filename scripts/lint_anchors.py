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


def main():
    args = sys.argv[1:]
    if args == ['--all'] or not args:
        files = sorted(glob.glob(instance.chapters_glob()))
    else:
        files = args
    total = 0
    cross_total = 0
    warn_total = 0
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
    if total == 0 and cross_total == 0:
        print('✓ 全部章内锚点都能解析')
    if total:
        print(f'\n共 {total} 处断锚')
    if cross_total:
        print(f'共 {cross_total} 处跨章链接问题(悬空/文字号不符/单层旧路径，BLOCKING)')
    if warn_total:
        print(f'⚠️  另有 {warn_total} 处裸文字章号(warn，不计入退出码)')
    return 1 if (total or cross_total) else 0


if __name__ == '__main__':
    sys.exit(main())
