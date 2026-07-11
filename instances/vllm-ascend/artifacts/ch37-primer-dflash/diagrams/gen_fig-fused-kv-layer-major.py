#!/usr/bin/env python3
"""layout 模板:全 L 层 K/V 投影权重堆叠成一个大矩阵,一次 GEMM 出全层 K/V,
再 permute 成 layer-major 布局让逐层切片天然 contiguous。
数字来自 explainer/traces/kv_injection.json(mechanism fused-kv-projection-layer-major,L=2)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "L 层 K/V 权重堆成一个大矩阵,一次 GEMM 出全层 K/V,permute 成 layer-major"
SUBTITLE = "本例 L=2、num_ctx=3、kv_size=8——融合与逐层数值差 0.0,纯省 kernel 启动、不改语义"

PAD, TOP = 46, 130
w = 1180

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 520">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="520" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# Stage 1: stacked weight matrix [32, 8]
S1_X, S1_W, S1_H = PAD, 200, 220
L.append(f'<text x="{S1_X+S1_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">1. 堆叠融合权重</text>')
L.append(f'<rect x="{S1_X}" y="{TOP}" width="{S1_W}" height="{S1_H}" rx="6" '
          'fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/>')
# draw 4 stripes (2 layers x K/V) inside
for i in range(4):
    sy = TOP + i * (S1_H/4)
    if i > 0:
        L.append(f'<line x1="{S1_X}" y1="{sy}" x2="{S1_X+S1_W}" y2="{sy}" '
                  'stroke="#2563eb" stroke-width="1" stroke-dasharray="4,3"/>')
    lbl = ["layer0 W^K", "layer0 W^V", "layer1 W^K", "layer1 W^V"][i]
    L.append(f'<text x="{S1_X+S1_W/2}" y="{sy+S1_H/8+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#1e3a8a">{esc(lbl)}</text>')
L.append(f'<text x="{S1_X+S1_W/2}" y="{TOP+S1_H+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#1e40af">形状 [32, 8]</text>')
L.append(f'<text x="{S1_X+S1_W/2}" y="{TOP+S1_H+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">= L*2*kv_size(2*2*8) x hidden</text>')

# arrow 1 -> 2
A1X = S1_X + S1_W
A2X = A1X + 60
AY = TOP + S1_H/2
L.append(f'<line x1="{A1X}" y1="{AY}" x2="{A2X}" y2="{AY}" '
          'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{(A1X+A2X)/2}" y="{AY-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">F.linear</text>')

# Stage 2: GEMM output [num_ctx, L, 2, nkv, hd]
S2_X, S2_W, S2_H = A2X, 260, 220
L.append(f'<text x="{S2_X+S2_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">2. 一次 GEMM 输出</text>')
L.append(f'<rect x="{S2_X}" y="{TOP}" width="{S2_W}" height="{S2_H}" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.8"/>')
for i in range(3):
    sy = TOP + i * (S2_H/3)
    if i > 0:
        L.append(f'<line x1="{S2_X}" y1="{sy}" x2="{S2_X+S2_W}" y2="{sy}" '
                  'stroke="#d97706" stroke-width="1" stroke-dasharray="4,3"/>')
    L.append(f'<text x="{S2_X+S2_W/2}" y="{sy+S2_H/6+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#92400e">ctx 位 {i}</text>')
L.append(f'<text x="{S2_X+S2_W/2}" y="{TOP+S2_H+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#92400e">[ctx=3, L=2, 2, nkv=2, hd=4]</text>')
L.append(f'<text x="{S2_X+S2_W/2}" y="{TOP+S2_H+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">未按层聚合,切片非 contiguous</text>')

# arrow 2 -> 3
A3X = S2_X + S2_W
A4X = A3X + 100
L.append(f'<line x1="{A3X}" y1="{AY}" x2="{A4X}" y2="{AY}" '
          'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{(A3X+A4X)/2}" y="{AY-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="#334155">permute</text>')
L.append(f'<text x="{(A3X+A4X)/2}" y="{AY-2}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="#334155">(2,1,0,3,4)</text>')

# Stage 3: layer-major [2, L, ctx, nkv, hd]
S3_X, S3_W, S3_H = A4X, 260, 220
L.append(f'<text x="{S3_X+S3_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">3. layer-major 布局</text>')
L.append(f'<rect x="{S3_X}" y="{TOP}" width="{S3_W}" height="{S3_H}" rx="6" '
          'fill="#dcfce7" stroke="#16a34a" stroke-width="1.8"/>')
for i in range(2):
    sy = TOP + i * (S3_H/2)
    if i > 0:
        L.append(f'<line x1="{S3_X}" y1="{sy}" x2="{S3_X+S3_W}" y2="{sy}" '
                  'stroke="#16a34a" stroke-width="1" stroke-dasharray="4,3"/>')
    L.append(f'<text x="{S3_X+S3_W/2}" y="{sy+S3_H/4+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#166534">'
              f'{esc("all_k[" + str(i) + "] / all_v[" + str(i) + "]（layer " + str(i) + "，contiguous）")}</text>')
L.append(f'<text x="{S3_X+S3_W/2}" y="{TOP+S3_H+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#166534">[2, L=2, ctx=3, nkv=2, hd=4]</text>')
L.append(f'<text x="{S3_X+S3_W/2}" y="{TOP+S3_H+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">逐层切片天然 contiguous,直接喂 do_kv_cache_update</text>')

foot_y = TOP + S1_H + 90
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">融合 vs 逐层数值差 max|K_融合 - K_逐层| = 0.0——线性投影对行分块可分配,只省 kernel 启动次数,数值不变。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">真实模型 L≈5、context 覆盖整段 prompt——一次融合 GEMM 省下的是约 5 倍 kernel 启动开销。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-fused-kv-layer-major.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
