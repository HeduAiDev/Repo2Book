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
import instance

HEADING = re.compile(r'^#{1,6}\s+(.*?)\s*$', re.M)
# 章内锚点链接: [text](#anchor) —— 排除 ](../  跨章 与 ](http
INCHAPTER = re.compile(r'\]\(#([^)]+)\)')


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


def main():
    args = sys.argv[1:]
    if args == ['--all'] or not args:
        files = sorted(glob.glob(instance.chapters_glob()))
    else:
        files = args
    total = 0
    for f in files:
        broken = check(f)
        ch = f.split('/')[-3] if '/' in f else f
        if broken:
            total += len(broken)
            print(f'❌ {ch}: {len(broken)} 处断锚')
            for b in broken:
                print(f'    {b}')
    if total == 0:
        print('✓ 全部章内锚点都能解析')
        return 0
    print(f'\n共 {total} 处断锚')
    return 1


if __name__ == '__main__':
    sys.exit(main())
