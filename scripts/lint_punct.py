#!/usr/bin/env python3
"""半角标点校验：找中文字符后面紧跟半角标点（应为全角）的地方，排除代码块/行内代码。
覆盖 , ? ! ; :（中文句子里这些应是 ，？！；：）。

用法: python3 scripts/lint_punct.py --all     # 扫全书
      python3 scripts/lint_punct.py <chapter.md> ...
退出码 1 = 有问题。
"""
import re
import sys
import glob
import instance

# 中文字符后紧跟半角标点（句末/句中标点；句号 . 不查，因 0.5/L37 等大量合法用法）
HALF = re.compile(r'([一-鿿])([,?!;:])')


def strip_code(text: str) -> str:
    """把代码块(```)与行内代码(`...`)替换成等长占位，避免误报；保留换行与偏移。"""
    out = []
    i = 0
    n = len(text)
    in_fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
            out.append('\n' * line.count('\n') + ' ' * (len(line) - line.count('\n')))
            continue
        if in_fence:
            out.append(re.sub(r'[^\n]', ' ', line))
            continue
        # 行内代码 `...` 抹平
        line = re.sub(r'`[^`\n]*`', lambda m: ' ' * len(m.group(0)), line)
        out.append(line)
    return ''.join(out)


def check(path: str):
    try:
        text = open(path, encoding='utf-8').read()
    except OSError as e:
        return [f'打不开: {e}']
    scan = strip_code(text)
    hits = []
    for m in HALF.finditer(scan):
        ln = scan[:m.start()].count('\n') + 1
        ctx = text.splitlines()[ln - 1].strip()[:50] if ln - 1 < len(text.splitlines()) else ''
        hits.append(f'L{ln}: 「{m.group(1)}{m.group(2)}」 应全角  | {ctx}')
    return hits


def main():
    args = sys.argv[1:]
    files = sorted(glob.glob(instance.chapters_glob())) \
        if (args == ['--all'] or not args) else args
    total = 0
    for f in files:
        hits = check(f)
        ch = f.split('/')[-3] if '/' in f else f
        if hits:
            total += len(hits)
            print(f'❌ {ch}: {len(hits)} 处半角标点')
            for h in hits[:8]:
                print(f'    {h}')
            if len(hits) > 8:
                print(f'    … 另 {len(hits) - 8} 处')
    if total == 0:
        print('✓ 无半角标点问题')
        return 0
    print(f'\n共 {total} 处')
    return 1


if __name__ == '__main__':
    sys.exit(main())
