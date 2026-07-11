#!/usr/bin/env python3
"""flow 模板:KV 注入管线——target 隐藏态一次融合投影出全层 K/V,逐层写进 draft 的 KV cache。
数字来自 explainer/traces/kv_injection.json(见 explainer.json mechanism kv-injection)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "target 隐藏态一次融合投影出全层 K/V,逐层写进 draft 每层的 KV cache"
SUBTITLE = "选定 target 层数 L=5(浅到深均匀采样)——不是只喂输入层,而是喂进 draft 每一层的注意力"

STEPS = [
    ("H^(l1)..H^(l5)", "target 5 个选定层\n的隐藏态"),
    ("拼接 + W_c 投影\n+ RMSNorm", "共享投影矩阵\nW_c: D x 5D"),
    ("H_t", "融合后的\ntarget 条件向量"),
    ("一次融合 GEMM\n出全层 K/V", "权重形状 [32, 8]\n数值差 0.0(与逐层等价)"),
    ("layer-major\n[2,L,ctx,nkv,hd]", "逐层切片\n天然 contiguous"),
    ("逐层写入\ndraft KV cache", "每层 attention\n直接复用,不重算"),
]

BOX_W, BOX_H, GAP_X, PAD, TOP = 168, 74, 46, 44, 130
w = PAD * 2 + len(STEPS) * BOX_W + (len(STEPS) - 1) * GAP_X
h = TOP + BOX_H + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

x = PAD
centers = []
for i, (main, sub) in enumerate(STEPS):
    highlight = (i == 3)
    fill = "#fef3c7" if highlight else "#dbeafe"
    stroke = "#d97706" if highlight else "#2563eb"
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    lines = main.split("\n")
    y0 = TOP + BOX_H/2 - (len(lines)-1)*8 - 2
    for j, ln in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+j*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="#0f172a">{esc(ln)}</text>')
    sub_lines = sub.split("\n")
    sy = TOP + BOX_H + 18
    for j, ln in enumerate(sub_lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{sy+j*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#64748b">{esc(ln)}</text>')
    centers.append(x + BOX_W/2)
    if i < len(STEPS) - 1:
        ax1 = x + BOX_W
        ax2 = ax1 + GAP_X
        ay = TOP + BOX_H/2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" '
                  'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
    x += BOX_W + GAP_X

step_num_y = TOP - 14
for i, cx in enumerate(centers):
    L.append(f'<text x="{cx}" y="{step_num_y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#94a3b8">{i+1}</text>')

foot_y = TOP + BOX_H + 78
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">融合与逐层数值差 0.0(max|K_融合 - K_逐层|)——融合 GEMM 只省 kernel 启动,不改数值。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">换掉 target 特征后 draft 层输出变化 0.2289——证明这份注入的条件确实在起作用,不是摆设。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-kv-injection-pipeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
