#!/usr/bin/env python3
"""把正文里的行内数学 $…$ 统一改写成 GitHub 官方转义形式 $`…`$。

为什么(2026-07-13,经 GitHub markdown API 逐条实测):
    朴素的 `$…$` 在 GitHub(cmark-gfm)上有 **六** 种静默失效方式,任一踩中就整段吐裸源码:
      1. 开定界符紧贴 CJK / 全角标点          压到$d_c$维          ✗
      2. 定界符内侧带空格                      压到 $ d_c $ 维       ✗
      3. 开定界符前接半角标点(, ; : . - /)     省一半;$N$ 一路涨     ✗
      4. 数学里含「可开强调」的下划线          $\\mathbf{q}_{t,j}$   ✗(_ 被吃成 <em>)
      5. 整段被单星号斜体包住                  *图 1 同一 $L$ 齐涨*  ✗
      6. 被 ** 粗体 flanking 失败连累          是**「x」**——         ✗
    而 `$`…`$` 形式对以上 **全部** 免疫(正文/表格/粗体/斜体/列表/标题/紧贴 CJK/前接标点,
    实测均正常渲染),且 LaTeX 源码逐字不变。于是把六条易错规则收敛成一条:
        **行内数学一律写 $`…`$**
    块级 $$…$$ 不受影响(叶子块,不走行内解析),保持原样。

用法:
    python3 scripts/fix_inline_math_escape.py <file.md> [...]        # 就地改写
    python3 scripts/fix_inline_math_escape.py --check <file.md> ...  # 只报告不改
幂等:已是 $`…`$ 的不再包一层。
"""
import re
import sys

_GH_ESCAPED = re.compile(r'\$`[^`]*`\$')


def convert_line(line: str) -> tuple:
    """返回 (新行, 改写数量)。跳过已转义的、code span 内的。"""
    # 1) 先把已转义的数学与普通 code span 挖出来占位,避免误伤
    holes = []

    def _stash(m):
        holes.append(m.group(0))
        return "\x00%d\x00" % (len(holes) - 1)

    work = _GH_ESCAPED.sub(_stash, line)
    work = re.sub(r'`[^`]*`', _stash, work)

    # 2) 顺序配对剩下的裸 $,逐对包成 $`…`$
    pos = [m.start() for m in re.finditer(r'(?<!\\)(?<!\$)\$(?!\$)', work)]
    if len(pos) < 2:
        out = work
        n = 0
    else:
        pieces, last, n = [], 0, 0
        for a, b in zip(pos[0::2], pos[1::2]):
            content = work[a + 1:b]
            if not content.strip():
                continue
            pieces.append(work[last:a])
            pieces.append("$`" + content.strip() + "`$")
            last = b + 1
            n += 1
        pieces.append(work[last:])
        out = "".join(pieces)

    # 3) 还原占位
    for i, h in enumerate(holes):
        out = out.replace("\x00%d\x00" % i, h)
    return out, n


def convert_file(path: str, write: bool = True) -> int:
    lines = open(path, encoding="utf-8").read().split("\n")
    in_fence = False
    in_disp = False
    total = 0
    for i, line in enumerate(lines):
        st = line.strip()
        if st.startswith("```"):
            in_fence = not in_fence
            continue
        if st == "$$":
            in_disp = not in_disp
            continue
        if in_fence or in_disp:
            continue
        new, n = convert_line(line)
        if n:
            lines[i] = new
            total += n
    if write and total:
        open(path, "w", encoding="utf-8").write("\n".join(lines))
    return total


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--check"]
    write = "--check" not in sys.argv
    grand = 0
    for p in argv:
        n = convert_file(p, write=write)
        grand += n
        if n:
            print(("改写 " if write else "待改写 ") + str(n) + "\t" + p)
    print("==== 合计 " + str(grand) + " 处行内数学 → $`…`$")
