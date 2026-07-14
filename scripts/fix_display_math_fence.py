#!/usr/bin/env python3
r"""把块级数学 $$…$$ 统一改写成 GitHub 的 ```math 围栏。

为什么(2026-07-14,用户报 GitHub 上 \left 报错后经 markdown API 实测):
    GitHub 在数学扩展拿到内容**之前**先做 CommonMark 反斜杠转义,于是 $$ 块里所有
    「反斜杠+可转义标点」的 LaTeX 命令都被吃掉反斜杠:
        \,  → ,   (细间距变字面逗号)      \;  → ;    \!  → !
        \{  → {   (于是 \left\{ 变非法 \left{ → Missing delimiter 报错)
        \_  → _   \\  → \   (aligned/矩阵换行被砍)   \% → %(MathJax 注释,吞行)
    全书实测 242 个 $$ 块中 139 个受影响(46 章)。
    而 ```math 围栏是代码围栏语义——内容逐字节保留,实测(含引用块内 > ```math)全部免疫;
    行内数学的 $`…`$ 转义式同理(code span 语义),两者构成同一条规则:
        **数学内容永远放在「不做 markdown 转义」的容器里。**

用法:
    python3 scripts/fix_display_math_fence.py <file.md> [...]        # 就地改写
    python3 scripts/fix_display_math_fence.py --check <file.md> ...  # 只报告
幂等;跳过代码围栏内的 $$;支持引用块(> $$ … > $$)。
"""
import re
import sys

_BQ = re.compile(r'^(\s*(?:>\s?)*)')


def _split(line):
    m = _BQ.match(line)
    prefix = m.group(1) if m else ''
    return prefix, line[len(prefix):]


def convert_file(path, write=True):
    lines = open(path, encoding='utf-8').read().split('\n')
    out = []
    in_code = False
    n = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        prefix, rest = _split(line)
        if rest.startswith('```'):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if not in_code and rest.strip() == '$$':
            j = i + 1
            while j < len(lines):
                _, r2 = _split(lines[j])
                if r2.strip() == '$$':
                    break
                j += 1
            if j < len(lines):
                out.append(prefix + '```math')
                out.extend(lines[i + 1:j])
                out.append(prefix + '```')
                n += 1
                i = j + 1
                continue
        out.append(line)
        i += 1
    if write and n:
        open(path, 'w', encoding='utf-8').write('\n'.join(out))
    return n


if __name__ == '__main__':
    argv = [a for a in sys.argv[1:] if a != '--check']
    write = '--check' not in sys.argv
    grand = 0
    for p in argv:
        n = convert_file(p, write=write)
        grand += n
        if n:
            print(('改写 ' if write else '待改写 ') + str(n) + '\t' + p)
    print('==== 合计 ' + str(grand) + ' 个 $$ 块 → ```math 围栏')
