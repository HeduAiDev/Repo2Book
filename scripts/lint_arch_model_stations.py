#!/usr/bin/env python3
"""Arch-Model 图文对账 —— 正文里引用的「站号」是否超出本章真实走线。

为什么存在：
  ch31 试点把「第 5 站」「第 12–13、17 站」这类**硬编码站号**写进了正文。一旦上游走线变化、
  或插章重编号，站号就漂移，正文静默地说着错的坐标。lint_chapter_map 只对 chapter-map 的
  §徽标做图文对账，没人核「正文站号 ↔ 本章真实走线」——本 linter 补的就是这个洞。

判据（抓住真漂移、不误报图里刻意不标的站）：
  真相源 = book/cartography/arch-model.json 里本章的 code_spine（走线总站数）。
  · 图上只标了落在**本章展开组件**上的站；落在其他章已讲组件上的站，图只给「另有 N 站」的总数、
    不逐站标出 —— 所以**不能用「图上有没有」判漂移**（那些站在图上本就不出现）。
  · 正文引用的站号若**超过本章走线总站数**，才是铁证如山的漂移（不可能合法）。
  更细的对账（某一站具体落在哪个组件）超出确定性 linter 能可靠断言的范围，留给盲审。

用法：
  python3 scripts/lint_arch_model_stations.py <chapter_dir> [--strict]
"""
import json
import re
import sys
from pathlib import Path

NS = '{http://www.w3.org/2000/svg}'


def _claims(md_path):
    txt = Path(md_path).read_text(encoding='utf-8')
    out = set()
    for m in re.finditer(r'第\s*([0-9]+(?:\s*[–—,-]\s*[0-9]+)*(?:\s*[、,]\s*[0-9]+)*)\s*站', txt):
        out.update(int(x) for x in re.findall(r'\d+', m.group(1)))
    return out


def _model_total(cd):
    inst = None
    for anc in cd.parents:
        if (anc / 'book' / 'cartography' / 'arch-model.json').exists():
            inst = anc
            break
    if not inst:
        return 0
    m = re.match(r'(ch\d+)', cd.name)
    if not m:
        return 0
    try:
        mm = json.load(open(inst / 'book' / 'cartography' / 'arch-model.json', encoding='utf-8'))
        return len(mm['chapters'][m.group(1)]['spine'])
    except Exception:
        return 0


def lint(chapter_dir, strict=False):
    cd = Path(chapter_dir)
    md = cd / 'narrative' / 'chapter.md'
    svg = cd / 'diagrams' / 'arch-model.svg'
    issues = []
    if not svg.exists():
        if strict:
            issues.append(f'无 {svg}（--strict：要求每章都有架构模型图）')
        return issues, 0, 0
    if not md.exists():
        return [f'无 {md}'], 0, 0
    total = _model_total(cd)
    claims = _claims(md)
    over = sorted(n for n in claims if total and n > total)
    if over:
        issues.append(f'正文引用了不存在的站号 {over} —— 本章走线只有 {total} 站，已漂移，需重渲染并重写回指')
    if total == 0 and claims:
        issues.append('找不到本章 code_spine（arch-model.json 缺该章）——无法核对站号，请先重建模型')
    return issues, len(claims), total


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    strict = '--strict' in sys.argv
    if not args:
        print(__doc__)
        sys.exit(2)
    bad = False
    for cd in args:
        issues, n, total = lint(cd, strict=strict)
        if issues:
            bad = True
            print(f'✗ {cd}')
            for i in issues:
                print('   -', i)
        else:
            print(f'✓ {cd}  正文引用站号 {n} 个，均在本章 {total} 站走线内')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
