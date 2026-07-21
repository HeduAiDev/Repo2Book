#!/usr/bin/env python3
"""fig-ch07-01-register-gate — flow 模板（register_custom_op 的八条断言闸门）。
主链纵向走：入口 -> 闸门①(isclass) -> 兜底(name) -> 闸门②(唯一性) ->
闸门③~⑤(三必填字段 hasattr) -> 闸门⑥~⑧(三类型 isinstance) -> 抄写签名 -> 入表。
四个可拒绝节点各引一条旁路箭头到共享的 AssertionError 汇聚框——任一闸门失败，
表长不变。全部坐标由常量/循环计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def text_w(s, fs):
    return fs * sum((0.98 if '一' <= c <= '鿿' else 0.58) for c in s)


TITLE = "register_custom_op：八条断言的闸门"
SUBTITLE = "过不了闸门，注册表一个字都不写——共 8 条断言（1+1+3+3），任一失败即 AssertionError"

# (标题, 正文行(list), 是否可拒绝, 断言数(仅可拒绝节点标注))
CHAIN = [
    ("入口", ["@register_custom_op 收到一个对象 op"], False, None),
    ("闸门① · 必须是类", ["assert inspect.isclass(op)"], True, 1),
    ("兜底 · 名字", ["未设 name → setattr(op, 'name', op.__name__)"], False, None),
    ("闸门② · 唯一性", ["assert op.name not in _custom_op_registry"], True, 1),
    ("闸门③~⑤ · 三个必填字段",
     ["assert hasattr(op, 'core')", "assert hasattr(op, 'pipe')", "assert hasattr(op, 'mode')"],
     True, 3),
    ("闸门⑥~⑧ · 三个类型检查",
     ["assert isinstance(op.core, CORE)", "assert isinstance(op.pipe, PIPE)", "assert isinstance(op.mode, MODE)"],
     True, 3),
    ("抄写签名", ["op.signature = inspect.signature(op)", "例：_index_select → 抽到 9 个形参"], False, None),
    ("入表", ["_custom_op_registry[op.name] = op", "本例 7 轮后条目数 = 3"], False, None),
]

MAIN_PAD_X = 20
longest = max(text_w(line, 12) for _, lines, *_ in CHAIN for line in lines)
MAIN_W = int(longest) + MAIN_PAD_X * 2 + 30
ROW_H1 = 56
ROW_H2 = 70
ROW_H3 = 92
GAP = 46

PAD = 40
TOP = 100
main_x = PAD

row_h = []
for _, lines, *_ in CHAIN:
    n = len(lines)
    row_h.append(ROW_H1 if n == 1 else (ROW_H2 if n == 2 else ROW_H3))

row_y = []
y_cur = TOP
for rh in row_h:
    row_y.append(y_cur)
    y_cur += rh + GAP

REJECT_W = 300
gap_mx = 130
reject_x = main_x + MAIN_W + gap_mx
w = reject_x + REJECT_W + PAD

# reject 框纵向铺满四个可拒绝节点(①/②/③~⑤/⑥~⑧)的整个跨度——
# 每条旁路箭头都在框的左边缘、各自的 y 上直接落地，不留悬空线头。
reject_idx = [i for i, (_, _, rej, _) in enumerate(CHAIN) if rej]
reject_y = row_y[reject_idx[0]]
reject_bot = row_y[reject_idx[-1]] + row_h[reject_idx[-1]]
REJECT_H = reject_bot - reject_y
reject_cy = (reject_y + reject_bot) / 2

h = max(y_cur - GAP + PAD + 90, reject_bot + 110)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for i, (name, lines, rej, cnt) in enumerate(CHAIN):
    y = row_y[i]
    rh = row_h[i]
    if rej:
        fill, stroke, sw = "#fee2e2", "#b91c1c", 2.4
    elif i == len(CHAIN) - 1:
        fill, stroke, sw = "#dcfce7", "#15803d", 2.4
    else:
        fill, stroke, sw = "#e0f2fe", "#0369a1", 1.6
    L.append(f'<rect x="{main_x}" y="{y}" width="{MAIN_W}" height="{rh}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    L.append(f'<text x="{main_x+16}" y="{y+22}" font-family="sans-serif" font-size="13.5" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    for k, line in enumerate(lines):
        L.append(f'<text x="{main_x+16}" y="{y+42+k*18}" font-family="sans-serif" font-size="11.5" '
                  f'fill="#334155">{esc(line)}</text>')
    if cnt is not None:
        L.append(f'<text x="{main_x+MAIN_W-14}" y="{y+rh-10}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="11" font-weight="bold" '
                  f'fill="#b91c1c">{esc(f"{cnt} 条断言")}</text>')

# 主链竖向连接（继续）
for i in range(len(CHAIN) - 1):
    y1 = row_y[i] + row_h[i]
    y2 = row_y[i + 1]
    cx = main_x + MAIN_W / 2
    L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" '
              f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

# 四个可拒绝节点 -> 共享 AssertionError 汇聚框
for i in reject_idx:
    y = row_y[i]
    rh = row_h[i]
    cy = y + rh / 2
    L.append(f'<line x1="{main_x+MAIN_W}" y1="{cy}" x2="{reject_x}" y2="{cy}" '
              f'stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#r)"/>')

L.append(f'<rect x="{reject_x}" y="{reject_y}" width="{REJECT_W}" height="{REJECT_H}" rx="10" '
          f'fill="#fef2f2" stroke="#b91c1c" stroke-width="2.2"/>')
L.append(f'<text x="{reject_x+REJECT_W/2}" y="{reject_cy-28}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#7f1d1d">{esc("AssertionError")}</text>')
L.append(f'<text x="{reject_x+REJECT_W/2}" y="{reject_cy-2}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#991b1b">'
          f'{esc("任一闸门失败 → 表长不变（3 → 3）")}</text>')
L.append(f'<text x="{reject_x+REJECT_W/2}" y="{reject_cy+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#991b1b">'
          f'{esc("本例 7 轮中 4 轮被拒（轮 4~7）")}</text>')

foot_y = h - 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("third_party/ascend/language/cann/extension/custom_op.py:L324-345——8 条断言全过，才轮到最后一句写表；这是一道类装饰器，不是函数装饰器。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch07-01-register-gate.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
