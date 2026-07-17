#!/usr/bin/env python3
"""fig-m3-blockptr-flattened: before-after 模板。
左:源码 tl.make_block_ptr(结构化块指针,shape/strides/order 显式)。
右:TTIR 里 add_rewrite_tensor_pointer 抹平后 —— 逐元素指针张量 + tt.dot/tt.reduce 成形。
数字/行号全部来自 _attn_fwd.ttir(spec.numbers)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "AST → TTIR:make_block_ptr 被 add_rewrite_tensor_pointer 抹平"

LEFT_TITLE = "源码(python/tutorials/06-fused-attention.py)"
LEFT_STEPS = [
    ("tl.make_block_ptr(Q, ...)", "shape=(Z,H,N,D), strides, order=(1,0)"),
    ("block_shape=(128,64)", "结构化块指针:形状/步长/order 随核\n生命周期不变,只有 offset 会挪"),
]
LEFT_LOC = "L121-L145(Q/K/V_block_ptr)"

RIGHT_TITLE = "TTIR(_attn_fwd.ttir)"
RIGHT_STEPS = [
    ("tt.addptr %Q/%K/%V", "块指针塌成裸指针(L63/L67/L69)"),
    ("tt.splat → tensor<128x64x!tt.ptr<f16>>", "逐元素指针张量登场(L78)"),
    ("tt.addptr + 偏移张量 → tt.load", "逐元素访存(L91)"),
    ("tt.dot → tensor<128x64xf32>", "QK^T 成形(L105);128x64xf16 * 64x64xf16"),
    ("tt.reduce×2 + math.exp2×2", "在线 softmax 成形(L106,119,120,126)"),
]

BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 58, 22, 340, 40, 172
n_max = max(len(LEFT_STEPS), len(RIGHT_STEPS))
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + n_max * (BOX_H + VGAP) + PAD + 56

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']


def panel(px, panel_title, steps, loc, box_fill, box_stroke, highlight_last=False):
    cx = px + PANEL_W / 2
    out = [f'<text x="{cx}" y="{TOP-62}" text-anchor="middle" font-family="sans-serif" '
           f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(panel_title)}</text>']
    if loc:
        out.append(f'<text x="{cx}" y="{TOP-42}" text-anchor="middle" font-family="sans-serif" '
                    f'font-size="11" fill="#64748b">{esc(loc)}</text>')
    for i, (head, sub) in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        is_last_hl = highlight_last and i == len(steps) - 1
        fill = "#fef3c7" if is_last_hl else box_fill
        stroke = "#d97706" if is_last_hl else box_stroke
        sw = "2" if is_last_hl else "1.5"
        text_fill = "#92400e" if is_last_hl else "#0f172a"
        sub_fill = "#92400e" if is_last_hl else "#334155"
        out.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        out.append(f'<text x="{cx}" y="{y+19}" text-anchor="middle" font-family="sans-serif" '
                    f'font-size="12.5" font-weight="bold" fill="{text_fill}">{esc(head)}</text>')
        sub_lines = sub.split("\n")
        for k, sl in enumerate(sub_lines):
            out.append(f'<text x="{cx}" y="{y+35+k*13}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="10.5" fill="{sub_fill}">{esc(sl)}</text>')
        if i < len(steps) - 1:
            out.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                        'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    return out


px_left = PAD
px_right = PAD + PANEL_W + 90
L += panel(px_left, LEFT_TITLE, LEFT_STEPS, LEFT_LOC, "#e0f2fe", "#0369a1")
L += panel(px_right, RIGHT_TITLE, RIGHT_STEPS, None, "#f1f5f9", "#475569", highlight_last=True)

# 中央大箭头:降级方向
midy = TOP + (n_max * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{px_left+PANEL_W+8}" y1="{midy}" x2="{px_right-8}" y2="{midy}" '
          'stroke="#d97706" stroke-width="3" marker-end="url(#b)"/>')
L.append(f'<text x="{(px_left+PANEL_W+px_right)/2}" y="{midy-12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#d97706">{esc("add_rewrite_tensor_pointer")}</text>')

foot_lines = [
    "结论:块指针消失,逐元素指针张量登场;tt.dot/tt.reduce/math.exp2 成形但还没有任何布局",
    "(琥珀框 = 布局分配前的最终态,下一跳 TTGIR 才有 #mma/#blocked)。",
]
foot_y0 = h - 22 - (len(foot_lines) - 1) * 18
for i, fl in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*18}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(fl)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m3-blockptr-flattened.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
