#!/usr/bin/env python3
"""before-after 模板(定制,逐行映射版):Python 5 参数签名 -> 追踪期 4 参数 tt.func。
每行左(Python 参数)右(IR 落点)一一对应,BLOCK_SIZE 那行对应到虚线折叠框(不在签名里)。
改造点:ROWS(左文案,右文案,右框是否为『折叠/无 IR 位』样式,是否携带 divisibility)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def multiline(lines, cx, y0, size=12, weight=False, fill="#0f172a", lh=15, anchor="middle"):
    out = []
    wattr = 'font-weight="bold" ' if weight else ''
    for k, line in enumerate(lines):
        out.append(f'<text x="{cx}" y="{y0 + k * lh}" text-anchor="{anchor}" '
                    f'font-family="sans-serif" font-size="{size}" {wattr}'
                    f'fill="{fill}">{esc(line)}</text>')
    return out


LEFT_TITLE = "Python @triton.jit 签名(5 参数)"
RIGHT_TITLE = "追踪期 tt.func 签名(4 个 IR 参数)"

# (left_lines, right_lines, style) style in {"div","plain","folded"}
ROWS = [
    (["x_ptr"], ["%arg0", "{tt.divisibility = 16}"], "div"),
    (["y_ptr"], ["%arg1", "{tt.divisibility = 16}"], "div"),
    (["out_ptr"], ["%arg2", "{tt.divisibility = 16}"], "div"),
    (["n_elements"], ["%arg3", "(1000,无 divisibility)"], "plain"),
    (["BLOCK_SIZE: tl.constexpr = 1024"], ["折叠成 arith.constant 1024", "(不在签名里,无 idx)"], "folded"),
]

STYLE = {
    "div": ("#fef3c7", "#b45309"),
    "plain": ("#e0e7ff", "#4338ca"),
    "folded": ("#f1f5f9", "#94a3b8"),
}

BOX_W, BOX_H, VGAP, PANEL_GAP, PAD, TOP = 260, 62, 22, 160, 50, 96
n = len(ROWS)
w = PAD * 2 + BOX_W * 2 + PANEL_GAP
h = TOP + n * (BOX_H + VGAP) + PAD + 80

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
          '<marker id="f" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("constexpr 参数不占 IR 位;指针参数携带 tt.divisibility=16")}</text>')

left_cx = PAD + BOX_W / 2
right_cx = PAD + BOX_W + PANEL_GAP + BOX_W / 2

L.append(f'<text x="{left_cx}" y="{TOP-20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(LEFT_TITLE)}</text>')
L.append(f'<text x="{right_cx}" y="{TOP-20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(RIGHT_TITLE)}</text>')

for i, (llines, rlines, style) in enumerate(ROWS):
    y = TOP + i * (BOX_H + VGAP)
    fill, stroke = STYLE[style]
    dash = ' stroke-dasharray="6,4"' if style == "folded" else ''
    # left box(Python 参数,统一中性色)
    L.append(f'<rect x="{left_cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#e2e8f0" stroke="#334155" stroke-width="1.4"/>')
    L += multiline(llines, left_cx, y + BOX_H/2 + 5, size=12.5, weight=True)
    # arrow
    ax1, ax2 = left_cx + BOX_W/2 + 6, right_cx - BOX_W/2 - 6
    ay = y + BOX_H/2
    marker = "f" if style == "folded" else "a"
    acolor = "#94a3b8" if style == "folded" else "#64748b"
    dash_attr = ' stroke-dasharray="5,4"' if style == "folded" else ''
    L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{acolor}" '
              f'stroke-width="1.6" marker-end="url(#{marker})"{dash_attr}/>')
    # right box
    L.append(f'<rect x="{right_cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{1.8 if style != "folded" else 1.4}"{dash}/>')
    L += multiline(rlines, right_cx, y + BOX_H/2 + (0 if len(rlines) > 1 else 5) - (7 if len(rlines) > 1 else 0),
                    size=12, weight=(style != "folded"), fill=stroke)

foot_y = TOP + n * (BOX_H + VGAP) + 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("Python 参数数=5 -> IR 参数数=4(Triton v3.2.0 headless 实测);BLOCK_SIZE 折叠值=1024")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("divisibility 标注数=3,值=16(backends/compiler.py:L77)")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+40}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("黄=携带 tt.divisibility 属性;蓝=普通 IR 参数(无对齐提示);灰虚线=不占 IR 位的编译期常量")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("ch16-fig-constexpr-not-in-ir.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
