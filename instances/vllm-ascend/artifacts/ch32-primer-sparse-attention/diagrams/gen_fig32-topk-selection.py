#!/usr/bin/env python3
"""before-after 模板改造:DSA top-k 选择——k=L 时退化为稠密(数值完全一致),
k<L 时只对 top-k 选中的 KV 算主注意力。左稠密(全 8 个 token 全亮),
右稀疏(仅 {2,4,6} 三个高亮,其余灰暗淘汰)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "DSA Top-k 选择:k=L 退化为稠密;k<L 只算选中的 KV"
SUBTITLE = "L=8 个前驱 token;index score 排序后取 top-k"

TOKENS = list(range(8))
TOPK3 = {2, 4, 6}

PANEL_W, PAD, TOP, TOK_SIZE, TOK_GAP = 340, 40, 110, 34, 8
w = PAD * 2 + PANEL_W * 2 + 80
h = TOP + 200

def draw_panel(L, px, title, selected, note1, note2):
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    row_w = len(TOKENS) * (TOK_SIZE + TOK_GAP) - TOK_GAP
    start_x = cx - row_w / 2
    for i in TOKENS:
        x = start_x + i * (TOK_SIZE + TOK_GAP)
        y = TOP
        hit = i in selected
        fill = "#3b82f6" if hit else "#e2e8f0"
        stroke = "#1e3a5f" if hit else "#94a3b8"
        text_fill = "white" if hit else "#94a3b8"
        L.append(f'<rect x="{x}" y="{y}" width="{TOK_SIZE}" height="{TOK_SIZE}" rx="5" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+TOK_SIZE/2}" y="{y+TOK_SIZE/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'font-weight="bold">{i}</text>')
    L.append(f'<rect x="{cx-row_w/2-10}" y="{TOP+TOK_SIZE+18}" width="{row_w+20}" height="60" rx="6" '
              'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<text x="{cx}" y="{TOP+TOK_SIZE+38}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#1e40af">{esc(note1)}</text>')
    L.append(f'<text x="{cx}" y="{TOP+TOK_SIZE+58}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#475569">{esc(note2)}</text>')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

px0 = PAD
draw_panel(L, px0, "before: k=8=L(稠密/退化)", set(TOKENS),
           "选中全部 8 个 → 与稠密一致", "输出最大绝对差 = 0.0(数值验证退化正确)")
px1 = PAD + PANEL_W + 80
draw_panel(L, px1, "after: k=3(稀疏)", TOPK3,
           "top-3 选中索引 {2,4,6}", "每 query 点积数 8→3,降 2.67x")

midy = TOP + TOK_SIZE / 2
L.append(f'<line x1="{px0+PANEL_W+8}" y1="{midy}" x2="{px0+PANEL_W+68}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')

foot_y = h - 20
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#64748b">落地 L=131072, k=512 → 每 query 主注意力 q·k 降 256x(k=2048 训练值降 64x)</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig32-topk-selection.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
