#!/usr/bin/env python3
"""phi-vs-block-arg: before-after 顿悟图。左『φ：拉』(汇合块回头问前驱要值，虚线箭头
从 merge 指向前驱)；右『块参数：推』(前驱 terminator 主动把值推进块参数，实线箭头从
前驱指向 merge)。中间等号 + 一句话；底部 Appel 1998 引用。全部坐标由常量/循环计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

BOX_W, BOX_H = 150, 60
MERGE_H = 64
TOP_GAP = 24          # then/else 之间的水平间隙
ROW_GAP = 92           # top 行到 merge 行的纵向间距
PAD = 40
PANEL_W = BOX_W * 2 + TOP_GAP
MID_W = 170            # 中间等号区宽度
TOP_Y = 96

W = PAD * 2 + PANEL_W * 2 + MID_W
H = TOP_Y + BOX_H + ROW_GAP + MERGE_H + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
          '<marker id="solid" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="dash" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
          '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="15" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("φ「拉」与块参数「推」：同一 SSA 汇合语义的两种等价写法")}</text>')
L.append(f'<text x="{PAD}" y="46" font-family="sans-serif" font-size="12" '
          f'fill="#475569">{esc("then 传入值=5，else 传入值=7；前驱数 n=2 → 2 个 φ 实参 ↔ 2 条 terminator 各推 1 个实参")}</text>')


def box(x, y, w, h, fill, stroke, lines, first_bold=True):
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>']
    n = len(lines)
    lh = 17
    sy = y + h / 2 - (n - 1) * lh / 2 + 5
    for i, t in enumerate(lines):
        fw = 'bold' if (i == 0 and first_bold) else 'normal'
        fs = 12 if i == 0 else 11
        out.append(f'<text x="{x+w/2}" y="{sy+i*lh}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" '
                    f'fill="#0f172a">{esc(t)}</text>')
    return '\n'.join(out)


def panel(px, title, title_color, merge_label_lines, pull):
    """pull=True: 虚线箭头 merge->前驱(问值)；pull=False: 实线箭头 前驱->merge(推值)"""
    out = []
    out.append(f'<text x="{px+PANEL_W/2}" y="{TOP_Y-14}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="14" font-weight="bold" '
                f'fill="{title_color}">{esc(title)}</text>')
    then_x, else_x = px, px + BOX_W + TOP_GAP
    out.append(box(then_x, TOP_Y, BOX_W, BOX_H, '#dbeafe', '#2563eb',
                    ['then 块 P1', 'x₁ = 5']))
    out.append(box(else_x, TOP_Y, BOX_W, BOX_H, '#fee2e2', '#dc2626',
                    ['else 块 P2', 'x₂ = 7']))
    merge_y = TOP_Y + BOX_H + ROW_GAP
    merge_x = px + PANEL_W / 2 - BOX_W / 2 * 1.35
    merge_w = BOX_W * 1.35
    out.append(box(merge_x, merge_y, merge_w, MERGE_H, '#fef3c7', '#d97706', merge_label_lines))

    then_bot = (then_x + BOX_W / 2, TOP_Y + BOX_H)
    else_bot = (else_x + BOX_W / 2, TOP_Y + BOX_H)
    merge_tl = (merge_x + merge_w * 0.28, merge_y)
    merge_tr = (merge_x + merge_w * 0.72, merge_y)

    if pull:  # merge -> 前驱, 虚线, 紫色, "回头问"
        out.append(f'<line x1="{merge_tl[0]}" y1="{merge_tl[1]}" x2="{then_bot[0]}" y2="{then_bot[1]}" '
                    'stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#dash)"/>')
        out.append(f'<line x1="{merge_tr[0]}" y1="{merge_tr[1]}" x2="{else_bot[0]}" y2="{else_bot[1]}" '
                    'stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#dash)"/>')
        mx1, my1 = (merge_tl[0] + then_bot[0]) / 2, (merge_tl[1] + then_bot[1]) / 2
        mx2, my2 = (merge_tr[0] + else_bot[0]) / 2, (merge_tr[1] + else_bot[1]) / 2
        out.append(f'<text x="{mx1-8}" y="{my1}" text-anchor="end" font-family="sans-serif" '
                    f'font-size="11" fill="#6d28d9">{esc("问 P1 要 x₁")}</text>')
        out.append(f'<text x="{mx2+8}" y="{my2}" text-anchor="start" font-family="sans-serif" '
                    f'font-size="11" fill="#6d28d9">{esc("问 P2 要 x₂")}</text>')
    else:  # 前驱 -> merge, 实线, 深灰, "推"
        out.append(f'<line x1="{then_bot[0]}" y1="{then_bot[1]}" x2="{merge_tl[0]}" y2="{merge_tl[1]}" '
                    'stroke="#334155" stroke-width="1.8" marker-end="url(#solid)"/>')
        out.append(f'<line x1="{else_bot[0]}" y1="{else_bot[1]}" x2="{merge_tr[0]}" y2="{merge_tr[1]}" '
                    'stroke="#334155" stroke-width="1.8" marker-end="url(#solid)"/>')
        mx1, my1 = (then_bot[0] + merge_tl[0]) / 2, (then_bot[1] + merge_tl[1]) / 2
        mx2, my2 = (else_bot[0] + merge_tr[0]) / 2, (else_bot[1] + merge_tr[1]) / 2
        out.append(f'<text x="{mx1-8}" y="{my1}" text-anchor="end" font-family="sans-serif" '
                    f'font-size="11" fill="#334155">{esc("br ^merge(5)")}</text>')
        out.append(f'<text x="{mx2+8}" y="{my2}" text-anchor="start" font-family="sans-serif" '
                    f'font-size="11" fill="#334155">{esc("br ^merge(7)")}</text>')
    return '\n'.join(out), merge_y + MERGE_H


left_x = PAD
right_x = PAD + PANEL_W + MID_W
pl, bottom_y = panel(left_x, 'φ：拉（回头选值）', '#6d28d9',
                      ['merge 块头部', 'x₃ = φ(x₁, x₂)'], pull=True)
pr, _ = panel(right_x, '块参数：推（前驱主动传入）', '#334155',
              ['merge 块（块参数）', '^merge(%x3: T)'], pull=False)
L.append(pl)
L.append(pr)

# 中间等号
eq_x = PAD + PANEL_W + MID_W / 2
eq_y = TOP_Y + BOX_H + ROW_GAP / 2 + 8
L.append(f'<text x="{eq_x}" y="{eq_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="30" font-weight="bold" fill="#0f172a">=</text>')
L.append(f'<text x="{eq_x}" y="{eq_y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#475569">{esc("同一语义")}</text>')
L.append(f'<text x="{eq_x}" y="{eq_y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#475569">{esc("方向相反")}</text>')
L.append(f'<text x="{eq_x}" y="{eq_y+58}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#475569">{esc("都是 SSA")}</text>')

# 底部结果 + 引用
res_y = bottom_y + 30
L.append(f'<text x="{W/2}" y="{res_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#0f172a">'
          f'{esc("走 then 前驱 → 两写法均得 5　　走 else 前驱 → 两写法均得 7")}</text>')
cite_y = res_y + 30
L.append(f'<rect x="{PAD}" y="{cite_y-18}" width="{W-2*PAD}" height="30" rx="6" '
          'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
L.append(f'<text x="{W/2}" y="{cite_y+2}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#334155">'
          f'{esc("Appel 1998：基本块 = 函数，块参数 = 形参，φ 各路来源 = 各调用点实参，跳转 = 尾调用")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('phi-vs-block-arg.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
