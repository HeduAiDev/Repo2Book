#!/usr/bin/env python3
"""fig-ch19-trait-perf: state-table 模板改造——四条关键 trait 的性能承诺表。
论点：认出 op 挂了哪些 trait，就能预测优化器对它的动作/约束。
长文本列做自动换行（按 CJK/空白切 token 贪心装箱），行高按各行最多行数动态算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)


def char_w(c):
    o = ord(c)
    if o == 0x20:
        return 0.30
    if 0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F:
        return 1.0
    if c.isascii() and c.isalnum():
        return 0.58
    return 0.5


def text_w(s, size):
    return size * sum(char_w(c) for c in s)


def tokenize(s):
    tokens, buf = [], ''
    for ch in s:
        o = ord(ch)
        is_cjk = 0x2E80 <= o <= 0x9FFF or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF
        if is_cjk:
            if buf:
                tokens.append(buf)
                buf = ''
            tokens.append(ch)
        elif ch == ' ':
            if buf:
                tokens.append(buf)
                buf = ''
            tokens.append(' ')
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    return tokens


def wrap(s, max_w, size):
    tokens = tokenize(s)
    lines, cur, curw = [], '', 0.0
    for tok in tokens:
        tw = text_w(tok, size)
        if curw + tw > max_w and cur.strip():
            lines.append(cur.strip())
            cur, curw = '', 0.0
        cur += tok
        curw += tw
    if cur.strip():
        lines.append(cur.strip())
    return lines or ['']


TITLE = "关键 trait 的性能承诺 —— 认出标签就能预测优化器的动作"
SUBTITLE = "前两条由算子按需声明；后两条由基类 TT_Op 用 !listconcat 自动挂给全体 tt 算子（TritonOps.td:L31-L34）"

COLS = ["trait", "承诺", "性能落点 / 读者据此预测", "怎么挂上的"]
FONT = 12
COL_TARGET_W = [190, 230, 340, 270]  # 内容排版目标宽度（换行依据）

ROWS = [
    ["Pure", "无副作用 + 结果只依赖操作数",
     "可被 CSE 合并、DCE 删除；tt 里 addptr/splat/make_range/dot/reduce/scan/trans 多标 Pure，"
     "load/store/atomic_* 不标",
     "各算子按需声明（如 make_range 的 [Pure]，L803）"],
    ["SameOperandsAndResultEncoding", "操作数与结果 encoding 必须一致",
     "布局沿数据流传播的通道；tt 层 encoding 空→恒放行，ttg 层填 layout 后才真正约束",
     "各算子声明（如 addptr，L199）；执法 verifySameEncoding，Traits.cpp:L30-L31"],
    ["TensorSizeTrait", "张量元素数 ≤ 2^20=1048576 且必须是 2 的幂",
     "防灾难性寄存器压力把编译器挂死（H100 有 256KB 寄存器）；也是 join/split『元素数总是 2 的幂』的执法处",
     "TT_Op 基类 !listconcat 自动挂全体算子，L33；执法 verifyTensorSize，Traits.cpp:L67-L82"],
    ["VerifyTensorLayoutsTrait", "统一的布局合法性闸门",
     "对带 encoding 的张量校验 layout 的 rank/threads-per-warp/warps-per-CTA 与 module 一致；"
     "tt 层 encoding 空则放行，主要在 ttg 层生效",
     "TT_Op 基类 !listconcat 自动挂全体算子，L33"],
]

PAD = 30
TOP = 96
HEADER_H = 34
LINE_H = 18
ROW_VPAD = 16
COL_GAP = 20
CELL_PAD = 12

# 每格换行
WRAPPED = [[wrap(row[j], COL_TARGET_W[j], FONT) for j in range(4)] for row in ROWS]
# 每格实际宽度（取该格所有行里最宽的一行，且不超过 target+少量余量）
col_w = []
for j in range(4):
    max_line_w = max(text_w(line, FONT) for row in WRAPPED for line in row[j])
    max_header = text_w(COLS[j], 12.5) + 4
    col_w.append(max(max_line_w, max_header) + CELL_PAD * 2)

col_x = [PAD]
for j in range(1, 4):
    col_x.append(col_x[j - 1] + col_w[j - 1] + COL_GAP)

row_lines = [max(len(WRAPPED[i][j]) for j in range(4)) for i in range(len(ROWS))]
row_h = [n * LINE_H + ROW_VPAD * 2 for n in row_lines]
row_y = [TOP + HEADER_H]
for rh in row_h[:-1]:
    row_y.append(row_y[-1] + rh)

w = col_x[-1] + col_w[-1] + PAD
h = row_y[-1] + row_h[-1] + 46

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#1e293b">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

for j, name in enumerate(COLS):
    cx = col_x[j]
    L.append(f'<rect x="{cx:.0f}" y="{TOP}" width="{col_w[j]:.0f}" height="{HEADER_H}" '
              'fill="#3b82f6"/>')
    L.append(f'<text x="{cx+CELL_PAD:.0f}" y="{TOP+HEADER_H/2+4.5:.0f}" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="white">{esc(name)}</text>')

table_left = col_x[0]
table_w = col_x[-1] + col_w[-1] - table_left
# 前两条 trait 按需声明 / 后两条基类自动挂 —— 用左侧色条区分
AUTO_TAG = [False, False, True, True]
for i, row in enumerate(ROWS):
    ry = row_y[i]
    rh = row_h[i]
    if i % 2 == 1:
        L.append(f'<rect x="{table_left:.0f}" y="{ry:.0f}" width="{table_w:.0f}" '
                  f'height="{rh}" fill="#f8fafc"/>')
    L.append(f'<line x1="{table_left:.0f}" y1="{ry:.0f}" x2="{table_left+table_w:.0f}" '
              f'y2="{ry:.0f}" stroke="#e2e8f0" stroke-width="1"/>')
    bar_color = "#7c3aed" if AUTO_TAG[i] else "#0ea5e9"
    L.append(f'<rect x="{table_left:.0f}" y="{ry:.0f}" width="4" height="{rh}" '
              f'fill="{bar_color}"/>')
    for j in range(4):
        lines = WRAPPED[i][j]
        n = len(lines)
        cy0 = ry + rh / 2 - (n - 1) * LINE_H / 2 + 4.5
        weight = 'font-weight="bold" ' if j == 0 else ''
        color = "#1e40af" if j == 0 else "#374151"
        family = "monospace" if j == 0 else "sans-serif"
        for k, line in enumerate(lines):
            L.append(f'<text x="{col_x[j]+CELL_PAD:.0f}" y="{cy0+k*LINE_H:.0f}" '
                      f'font-family="{family}" font-size="{FONT}" {weight}'
                      f'fill="{color}">{esc(line)}</text>')

bottom_y = row_y[-1] + row_h[-1]
L.append(f'<line x1="{table_left:.0f}" y1="{bottom_y:.0f}" x2="{table_left+table_w:.0f}" '
          f'y2="{bottom_y:.0f}" stroke="#94a3b8" stroke-width="1.5"/>')
legend_y = bottom_y + 26
L.append(f'<rect x="{PAD}" y="{legend_y-11:.0f}" width="16" height="10" fill="#0ea5e9"/>')
L.append(f'<text x="{PAD+22}" y="{legend_y+1:.0f}" font-family="sans-serif" font-size="11" '
          f'fill="#475569">各算子按需声明</text>')
lx2 = PAD + 22 + text_w("各算子按需声明", 11) + 30
L.append(f'<rect x="{lx2:.0f}" y="{legend_y-11:.0f}" width="16" height="10" fill="#7c3aed"/>')
L.append(f'<text x="{lx2+22:.0f}" y="{legend_y+1:.0f}" font-family="sans-serif" font-size="11" '
          f'fill="#475569">TT_Op 基类自动挂给全体算子</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch19-trait-perf.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
