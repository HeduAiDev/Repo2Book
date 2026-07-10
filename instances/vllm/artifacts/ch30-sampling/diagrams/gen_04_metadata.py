#!/usr/bin/env python3
"""SamplingMetadata：N 个逐请求 SamplingParams → 批量张量列 + 字典 + list + 容器。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

W, H = 1080, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def box(x, y, w, h, fill, stroke, lines, fs=13, tw="bold", lcol="#1e293b"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    n = len(lines)
    cy = y + h/2 - (n-1)*0.5*(fs+3)
    for i, t in enumerate(lines):
        fw = 'bold' if (tw == 'bold' and i == 0) else 'normal'
        L.append(f'<text x="{x+w/2}" y="{cy + i*(fs+3) + fs*0.35:.0f}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" fill="{lcol}">{esc(t)}</text>')

L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="bold" '
         f'fill="#1e293b">SamplingMetadata：逐请求参数 → 一次性批量张量化</text>')

# 上方：N 个请求各自的 SamplingParams
reqs = [
    ("req 0", ["temp=0.0", "seed=None", "min_tok=4"]),
    ("req 1", ["temp=0.7", "top_p=0.9", "seed=42"]),
    ("req 2", ["temp=1.0", "top_k=50", "freq_pen=0.5"]),
]
rw = 220
rh = 96
gap = 60
x0 = (W - (len(reqs)*rw + (len(reqs)-1)*gap))/2
y_top = 56
for i, (name, fields) in enumerate(reqs):
    x = x0 + i*(rw+gap)
    box(x, y_top, rw, rh, "#eff6ff", "#3b82f6", [name+"  SamplingParams"]+fields, fs=12.5)

# 中间 收敛箭头 → 大框
big_y = 300
big_x = 90
big_w = W - 180
big_h = 280
L.append(f'<rect x="{big_x}" y="{big_y}" width="{big_w}" height="{big_h}" rx="10" fill="#f8fafc" stroke="#0f172a" stroke-width="1.8"/>')
L.append(f'<text x="{big_x+16}" y="{big_y+26}" font-family="sans-serif" font-size="14" font-weight="bold" '
         f'fill="#0f172a">SamplingMetadata（整批一份）</text>')

for i in range(len(reqs)):
    x = x0 + i*(rw+gap) + rw/2
    L.append(f'<line x1="{x}" y1="{y_top+rh}" x2="{W/2}" y2="{big_y}" stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')

# 大框内三类载体
iy = big_y + 50
ix = big_x + 20
# 列1：[batch] 张量
box(ix, iy, 300, 100, "#dcfce7", "#16a34a",
    ["[batch] 张量列", "temperature [0.0, 0.7, 1.0]", "top_p / top_k / 三种 penalty", "all_greedy / all_random 批级标志"], fs=12, lcol="#14532d")
# 列2：字典
box(ix+330, iy, 300, 100, "#fef9c3", "#ca8a04",
    ["逐请求字典 / list", "generators: {1:gen42, ...}", "output_token_ids: [[..],[..],[..]]", "bad_words_token_ids: {idx:[...]}"], fs=12, lcol="#713f12")
# 列3：容器
box(ix+660, iy, 300, 100, "#ede9fe", "#7c3aed",
    ["已就绪的容器", "logitsprocs:", "  argmax_invariant=[min_p]", "  non_argmax_invariant=[min_tok, bias]"], fs=12, lcol="#4c1d95")

L.append(f'<text x="{big_x+16}" y="{big_y+big_h-16}" font-family="sans-serif" font-size="12.5" fill="#475569">'
         f'Sampler 全程吃这一个对象 · 逐请求差异编码进张量行 / 字典键，避免 Python 逐请求循环。</text>')

L.append('</svg>')
open("04-sampling-metadata.svg", "w", encoding="utf-8").write('\n'.join(L))
print("ok")
