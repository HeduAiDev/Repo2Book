#!/usr/bin/env python3
"""修 **粗体** 在 CJK 正文里被 flanking 规则卡死、GitHub 吐字面 ** 的问题。

CommonMark 里汉字既不是空白也不是标点,而全角标点(。：「」（）)是标点,于是:
    是**「编译」…**        开定界符「前接汉字 + 后接全角括号」→ 不 left-flanking → 开不了
    **…怎么读：**第一个     闭定界符「前接全角冒号 + 后接汉字」→ 不 right-flanking → 关不上
两种都让 ** 原样显示,**并且连累其中的行内数学一起不渲染**。

修法(全部经 GitHub markdown API 实测):
    开定界符卡住 → 在 ** 前补半角空格      …是 **「编译」…**
    闭定界符卡住 → 在 ** 后补半角空格      **…怎么读：** 第一个
    闭定界符前有空格 → 把该空格挪到 ** 外  **…分子 $`x`$** 。
即与行内数学同一条口诀:**空格永远在定界符的外侧**。

用法: python3 scripts/fix_emphasis_flanking.py <file.md> [...]
"""
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from lint_formulas import _can_close, _can_open, _mask_line  # noqa: E402


def fix_line(line: str) -> tuple:
    """返回 (新行, 修复数)。反复扫直到不动点(一次插入会改变后续下标)。"""
    fixed = 0
    for _ in range(12):
        masked = _mask_line(line)
        runs = [m for m in re.finditer(r'(?<!\*)\*\*(?!\*)', masked)]
        changed = False
        for opener, closer in zip(runs[0::2], runs[1::2]):
            o, c = opener.start(), closer.start()
            if not _can_open(line, o, 2):
                line = line[:o] + " " + line[o:]
                changed = True
                break
            if not _can_close(line, c, 2):
                # 情形 A:闭定界符前是空白 → 该空白应在 ** 外侧,挪出去
                if c > 0 and line[c - 1].isspace():
                    j = c
                    while j > 0 and line[j - 1].isspace():
                        j -= 1
                    line = line[:j] + line[c:c + 2] + line[j:c] + line[c + 2:]
                else:
                    # 情形 B:闭定界符前是标点、后是汉字 → 在 ** 后补半角空格
                    line = line[:c + 2] + " " + line[c + 2:]
                changed = True
                break
        if not changed:
            break
        fixed += 1
    return line, fixed


def fix_file(path: str) -> int:
    lines = open(path, encoding="utf-8").read().split("\n")
    in_fence = in_disp = False
    total = 0
    for i, line in enumerate(lines):
        st = line.strip()
        if st.startswith("```"):
            in_fence = not in_fence
            continue
        if st == "$$":
            in_disp = not in_disp
            continue
        if in_fence or in_disp or "**" not in line:
            continue
        new, n = fix_line(line)
        if n:
            lines[i] = new
            total += n
    if total:
        open(path, "w", encoding="utf-8").write("\n".join(lines))
    return total


if __name__ == "__main__":
    grand = 0
    for p in sys.argv[1:]:
        n = fix_file(p)
        grand += n
        if n:
            print("修复 " + str(n) + "\t" + p)
    print("==== 合计修复 " + str(grand) + " 处 ** flanking")
