#!/usr/bin/env python3
"""phi-merge-diamond: if/else 汇合处 phi 节点按前驱选值，恢复 SSA 单赋值。
菱形 CFG：entry(cond) 分叉 then/else，汇入 merge，merge 头部画 phi 伪操作。
右侧红线边注框标 phi 是记号非算法。全部坐标由常量/循环计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

BOX_W, BOX_H = 200, 64
PAD = 40
COL_GAP = 60          # then/else 之间的水平间隙
ROW_GAP = 70           # 纵向层间距
NOTE_W = 300

# 三层纵坐标
ENTRY_Y = PAD + 26
BRANCH_Y = ENTRY_Y + BOX_H + ROW_GAP
MERGE_Y = BRANCH_Y + BOX_H + ROW_GAP + 20
MERGE_H = 88

# 横坐标：then 在左、else 在右，entry/merge 居中于两者之间
THEN_X = PAD
ELSE_X = THEN_X + BOX_W + COL_GAP
CENTER_X = (THEN_X + ELSE_X) / 2  # 两块中点
ENTRY_X = CENTER_X - BOX_W / 2
MERGE_X = CENTER_X - BOX_W / 2

DIAGRAM_W = ELSE_X + BOX_W + PAD
W = DIAGRAM_W + NOTE_W + PAD
H = MERGE_Y + MERGE_H + PAD + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# 标题
L.append(f'<text x="{PAD}" y="24" font-family="sans-serif" font-size="15" '
          f'font-weight="bold" fill="#0f172a">{esc("x₃ = φ(x₁, x₂)：if/else 汇合处按前驱选值")}</text>')
L.append(f'<text x="{PAD}" y="42" font-family="sans-serif" font-size="12" '
          f'fill="#475569">{esc("前驱数 n=2 → φ 携 2 个实参；运行时走 1 条前驱 → 选出 1 个值")}</text>')

def box(x, y, w, h, fill, stroke, lines, bold_first=True):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>']
    n = len(lines)
    line_h = 18
    start_y = y + h / 2 - (n - 1) * line_h / 2 + 5
    for i, txt in enumerate(lines):
        fw = 'bold' if (i == 0 and bold_first) else 'normal'
        fs = 13 if i == 0 else 12
        out.append(f'<text x="{x+w/2}" y="{start_y+i*line_h}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" '
                    f'fill="#0f172a">{esc(txt)}</text>')
    return '\n'.join(out)

# entry
L.append(box(ENTRY_X, ENTRY_Y, BOX_W, BOX_H, '#e2e8f0', '#64748b',
             ['entry: if cond', '(cond=True → then, False → else)']))

# then (P1)
L.append(box(THEN_X, BRANCH_Y, BOX_W, BOX_H, '#dbeafe', '#2563eb',
             ['then 块 P1', 'x₁ = a = 5']))

# else (P2)
L.append(box(ELSE_X, BRANCH_Y, BOX_W, BOX_H, '#fee2e2', '#dc2626',
             ['else 块 P2', 'x₂ = b = 7']))

# merge
L.append(box(MERGE_X, MERGE_Y, BOX_W, MERGE_H, '#fef3c7', '#d97706',
             ['merge 块', 'φ: x₃ = φ(x₁, x₂)', '（按走哪条前驱选值）']))

# 箭头: entry -> then
entry_bl = (ENTRY_X + BOX_W * 0.25, ENTRY_Y + BOX_H)
then_top = (THEN_X + BOX_W / 2, BRANCH_Y)
L.append(f'<line x1="{entry_bl[0]}" y1="{entry_bl[1]}" x2="{then_top[0]}" y2="{then_top[1]}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<text x="{(entry_bl[0]+then_top[0])/2-14}" y="{(entry_bl[1]+then_top[1])/2}" '
          f'text-anchor="end" font-family="sans-serif" font-size="11" fill="#334155">{esc("cond=True")}</text>')

# entry -> else
entry_br = (ENTRY_X + BOX_W * 0.75, ENTRY_Y + BOX_H)
else_top = (ELSE_X + BOX_W / 2, BRANCH_Y)
L.append(f'<line x1="{entry_br[0]}" y1="{entry_br[1]}" x2="{else_top[0]}" y2="{else_top[1]}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<text x="{(entry_br[0]+else_top[0])/2+14}" y="{(entry_br[1]+else_top[1])/2}" '
          f'text-anchor="start" font-family="sans-serif" font-size="11" fill="#334155">{esc("cond=False")}</text>')

# then -> merge (P1, 第1实参 x1)
then_bot = (THEN_X + BOX_W / 2, BRANCH_Y + BOX_H)
merge_tl = (MERGE_X + BOX_W * 0.25, MERGE_Y)
L.append(f'<line x1="{then_bot[0]}" y1="{then_bot[1]}" x2="{merge_tl[0]}" y2="{merge_tl[1]}" '
          'stroke="#2563eb" stroke-width="1.8" marker-end="url(#a)"/>')
mx, my = (then_bot[0] + merge_tl[0]) / 2, (then_bot[1] + merge_tl[1]) / 2
L.append(f'<text x="{mx-8}" y="{my-6}" text-anchor="end" font-family="sans-serif" '
          f'font-size="11" fill="#1d4ed8">{esc("从 P1 取 x₁（第1实参）")}</text>')

# else -> merge (P2, 第2实参 x2)
else_bot = (ELSE_X + BOX_W / 2, BRANCH_Y + BOX_H)
merge_tr = (MERGE_X + BOX_W * 0.75, MERGE_Y)
L.append(f'<line x1="{else_bot[0]}" y1="{else_bot[1]}" x2="{merge_tr[0]}" y2="{merge_tr[1]}" '
          'stroke="#dc2626" stroke-width="1.8" marker-end="url(#a)"/>')
mx2, my2 = (else_bot[0] + merge_tr[0]) / 2, (else_bot[1] + merge_tr[1]) / 2
L.append(f'<text x="{mx2+8}" y="{my2-6}" text-anchor="start" font-family="sans-serif" '
          f'font-size="11" fill="#b91c1c">{esc("从 P2 取 x₂（第2实参）")}</text>')

# 结果标注：走 then 得 5，走 else 得 7
L.append(f'<text x="{MERGE_X+BOX_W/2}" y="{MERGE_Y+MERGE_H+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#0f172a">'
          f'{esc("走 then → x₃=5　　走 else → x₃=7")}</text>')

# 右侧红线边注框
note_x = DIAGRAM_W + 10
note_y = BRANCH_Y - 10
note_h = MERGE_Y + MERGE_H - note_y
L.append(f'<rect x="{note_x}" y="{note_y}" width="{NOTE_W-20}" height="{note_h}" rx="10" '
          'fill="#fef2f2" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="5,3"/>')
note_lines = [
    "红线：",
    "φ 是理解汇合语义的记号，",
    "帮你想清值从哪来——",
    "不是 Triton 要跑的算法。",
    "MLIR 用块参数实现它，",
    "Triton 用 AST scope 差集",
    "构造它，全程不算",
    "支配边界 / 最小 φ 插入。",
]
for i, t in enumerate(note_lines):
    fw = 'bold' if i == 0 else 'normal'
    L.append(f'<text x="{note_x+16}" y="{note_y+26+i*20}" font-family="sans-serif" '
              f'font-size="12" font-weight="{fw}" fill="#7f1d1d">{esc(t)}</text>')

L.append('</svg>')
out = Path(__file__).with_name('phi-merge-diamond.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
