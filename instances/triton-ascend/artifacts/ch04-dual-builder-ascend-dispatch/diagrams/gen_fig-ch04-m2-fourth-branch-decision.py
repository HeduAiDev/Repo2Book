#!/usr/bin/env python3
"""fig-ch04-m2-fourth-branch-decision — flow 模板（visit_Call 四岔决策树）。
主链纵向四个判定，命中即右出；第③岔（统一 builtin 入口门）之后，fork 加的
第④岔按 extension.is_builtin(fn) 在 ascend_builder / builder 间二选一——
这是 fork 在语言层的接缝。全部坐标由常量/循环计算，文本全 esc()。
源码字面量逐字核对 pin 源码（含 self. 前缀），不省略。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def text_w(s, fs):
    return fs * sum((0.98 if '一' <= c <= '鿿' else 0.58) for c in s)


TITLE = "visit_Call：四岔决策树"
SUBTITLE = "同一道 __triton_builtin__ 入口门之后，fork 加的第④岔按 __ascend_builtin__ 二选一 builder"

EXIT_W, EXIT_H = 340, 58
PAD = 40
TOP = 108
GAP_MX = 110  # 主链 -> 出口 水平间距

# (标题, 条件行(可多行 list), 行号, 是否 fork 新增, 出口标记, 出口文本)
CHAIN = [
    ("① 常量折叠",
     ["self.statically_implemented_functions.get(fn) is not None"],
     "L1170-L1172", False,
     "是", "return static_implementation(self, node)"),
    ("② JIT 组合子",
     ["isinstance(fn, JITFunction)"],
     "L1176", False,
     "是", "return self.call_JitFunction(fn, args, kws)"),
    ("③ 入口门（统一 builtin）",
     ["language.core.is_builtin(fn)  读 __triton_builtin__"],
     "L1179", False,
     "否", "兜底：return fn(*args, **kws)"),
    ("④ 选 builder（fork 加）",
     ["_builder = self.ascend_builder if extension.is_builtin(fn)",
      "else self.builder"],
     "L1183", True,
     None, None),
]

# 主链宽度按最长条件行估算
longest = max(text_w(line, 11) for _, lines, *_ in CHAIN for line in lines)
MAIN_W = int(longest) + 60
MAIN_H_1LINE = 68
MAIN_H_2LINE = 88

main_x = PAD
exit_x = main_x + MAIN_W + GAP_MX
w = exit_x + EXIT_W + PAD

row_y = []
y_cur = TOP
row_h = []
for _, lines, *_ in CHAIN:
    rh = MAIN_H_2LINE if len(lines) > 1 else MAIN_H_1LINE
    row_h.append(rh)
    row_y.append(y_cur)
    y_cur += rh + 60

# 第④岔之后：两个落点框（ascend_builder / builder）
FORK_W, FORK_H = 320, 78
fork_gap = 70
fork_y = row_y[3] + row_h[3] + 78
fork_total = FORK_W * 2 + fork_gap
fork_left_x = max(PAD, main_x + (MAIN_W - fork_total) / 2)
fork_right_x = fork_left_x + FORK_W + fork_gap
w = max(w, fork_right_x + FORK_W + PAD)

h = fork_y + FORK_H + 100

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 主链四个判定框
for i, (name, lines, loc, hot, mark, etext) in enumerate(CHAIN):
    y = row_y[i]
    rh = row_h[i]
    fill = "#fef3c7" if hot else "#e0f2fe"
    stroke = "#b45309" if hot else "#0369a1"
    sw = 2.6 if hot else 1.6
    L.append(f'<rect x="{main_x}" y="{y}" width="{MAIN_W}" height="{rh}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    L.append(f'<text x="{main_x+16}" y="{y+22}" font-family="sans-serif" font-size="13.5" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    for k, line in enumerate(lines):
        L.append(f'<text x="{main_x+16}" y="{y+40+k*17}" font-family="sans-serif" font-size="11" '
                  f'fill="#334155">{esc(line)}</text>')
    L.append(f'<text x="{main_x+MAIN_W-14}" y="{y+rh-9}" text-anchor="end" font-family="sans-serif" '
              f'font-size="10.5" font-weight="bold" fill="{stroke}">{esc(loc)}</text>')

# 主链竖向连接（否，继续）
for i in range(3):
    y1 = row_y[i] + row_h[i]
    y2 = row_y[i + 1]
    cx = main_x + MAIN_W / 2
    L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" '
              f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
    L.append(f'<text x="{cx+10}" y="{(y1+y2)/2+4}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc("否，继续")}</text>')

# 前三岔的「是/否」出口（横向指向右侧出口框）
for i in range(3):
    _, _, _, _, mark, etext = CHAIN[i]
    y = row_y[i]
    rh = row_h[i]
    cy = y + rh / 2
    L.append(f'<line x1="{main_x+MAIN_W}" y1="{cy}" x2="{exit_x}" y2="{cy}" '
              f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
    L.append(f'<text x="{(main_x+MAIN_W+exit_x)/2}" y="{cy-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
              f'fill="#b45309">{esc(mark)}</text>')
    ey = y + rh / 2 - EXIT_H / 2
    L.append(f'<rect x="{exit_x}" y="{ey}" width="{EXIT_W}" height="{EXIT_H}" rx="9" '
              f'fill="#f1f5f9" stroke="#64748b" stroke-width="1.4"/>')
    L.append(f'<text x="{exit_x+EXIT_W/2}" y="{cy+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#334155">{esc(etext)}</text>')

# 第④岔 -> 两落点框
node4_cx = main_x + MAIN_W / 2
node4_by = row_y[3] + row_h[3]
mid_y = (node4_by + fork_y) / 2
L.append(f'<line x1="{node4_cx}" y1="{node4_by}" x2="{node4_cx}" y2="{mid_y}" '
          f'stroke="#334155" stroke-width="1.8"/>')
left_cx = fork_left_x + FORK_W / 2
right_cx = fork_right_x + FORK_W / 2
L.append(f'<path d="M {node4_cx} {mid_y} L {left_cx} {mid_y} L {left_cx} {fork_y}" '
          f'fill="none" stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<path d="M {node4_cx} {mid_y} L {right_cx} {mid_y} L {right_cx} {fork_y}" '
          f'fill="none" stroke="#1d4ed8" stroke-width="2" marker-end="url(#b)"/>')
L.append(f'<text x="{left_cx}" y="{mid_y-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#15803d">'
          f'{esc("true（∈ ascend 内建）")}</text>')
L.append(f'<text x="{right_cx}" y="{mid_y-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#1d4ed8">{esc("false")}</text>')

L.append(f'<rect x="{fork_left_x}" y="{fork_y}" width="{FORK_W}" height="{FORK_H}" rx="10" '
          f'fill="#dcfce7" stroke="#15803d" stroke-width="2.2"/>')
L.append(f'<text x="{left_cx}" y="{fork_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#14532d">{esc("_builder = self.ascend_builder")}</text>')
L.append(f'<text x="{left_cx}" y="{fork_y+46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#166534">{esc("al.* 落昇腾方言（fork 加）")}</text>')
L.append(f'<text x="{left_cx}" y="{fork_y+64}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#15803d" font-weight="bold">{esc("code_generator.py:L1183")}</text>')

L.append(f'<rect x="{fork_right_x}" y="{fork_y}" width="{FORK_W}" height="{FORK_H}" rx="10" '
          f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.2"/>')
L.append(f'<text x="{right_cx}" y="{fork_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#1e3a8a">{esc("_builder = self.builder")}</text>')
L.append(f'<text x="{right_cx}" y="{fork_y+46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#1e40af">{esc("tl.* 落标准 Triton IR（基座既有）")}</text>')
L.append(f'<text x="{right_cx}" y="{fork_y+64}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#1d4ed8" font-weight="bold">{esc("code_generator.py:L1183")}</text>')

foot_y = h - 22
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("基座此处只有单一 self.builder；fork 把第③岔内部一分为二——第④岔不是新增顶层分支，是同一入口门下多一个 builder 选择。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch04-m2-fourth-branch-decision.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
