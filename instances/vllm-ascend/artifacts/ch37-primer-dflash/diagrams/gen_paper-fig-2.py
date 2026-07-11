#!/usr/bin/env python3
"""重绘自 arXiv:2602.06036 Figure 2(DFlash Inference Design)。
布局对齐原图(ref_x2.png,已下载核对):Target Model 抽取隐藏特征(浅蓝)+ Target Embedding
产出的 decode token(橙)/mask token(绿)序列,一起送进每个 Draft Layer 的 KV Cache;
Draft Layer 内部 Bidirectional Attention + MLP;多层 Draft Layer 串联,最终 Target LM Head
输出用于投机解码的 token。图注/文字译中,配色沿用原图三语义色。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "DFlash 推理设计(重绘自 arXiv:2602.06036 Fig.2)"
SUBTITLE = "target 隐藏特征融合后直接注入每个 draft 层的 KV cache——不是只喂输入层"

FEAT_BLUE = "#bfe3f0"
FEAT_BLUE_STROKE = "#3a8fb0"
DECODE_ORANGE = "#f5b942"
DECODE_ORANGE_STROKE = "#c98a1a"
MASK_GREEN = "#8fd98f"
MASK_GREEN_STROKE = "#2f8f2f"

PAD = 44
STRIP_CENTER_Y = 320  # 整条 token 带 / draft layer 行的垂直中心,留够顶部标题空间
w = 1400
h = 700

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#c98a1a"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

TM_H = 50
row_y = STRIP_CENTER_Y - TM_H/2

# Target Model box (left)
TM_X, TM_W = PAD, 130
L.append(f'<rect x="{TM_X}" y="{row_y}" width="{TM_W}" height="{TM_H}" rx="8" '
          'fill="#e2e8f0" stroke="#334155" stroke-width="1.6"/>')
L.append(f'<text x="{TM_X+TM_W/2}" y="{row_y+TM_H/2+5}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#0f172a">Target Model</text>')
L.append(f'<text x="{TM_X+TM_W/2}" y="{row_y-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">"Diffusion is good"</text>')

# Target Embedding box
TE_X = TM_X + TM_W + 46
TE_W, TE_H = 150, 50
L.append(f'<line x1="{TM_X+TM_W}" y1="{row_y+TM_H/2}" x2="{TE_X}" y2="{row_y+TM_H/2}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<rect x="{TE_X}" y="{row_y}" width="{TE_W}" height="{TE_H}" rx="8" '
          'fill="#e2e8f0" stroke="#334155" stroke-width="1.6"/>')
L.append(f'<text x="{TE_X+TE_W/2}" y="{row_y+TM_H/2+5}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#0f172a">Target Embedding</text>')
L.append(f'<text x="{TE_X+TE_W/2}" y="{row_y-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">for &lt;mask&gt;&lt;mask&gt;&lt;mask&gt;</text>')

# token strip after embedding: 4 blue(context feature) + 1 orange(decode) + 4 green(mask)
STRIP_X = TE_X + TE_W + 40
SQ = 26
GAP_SQ = 4
tokens = ["blue"]*4 + ["orange"] + ["green"]*4
strip_top = STRIP_CENTER_Y - (len(tokens)*(SQ+GAP_SQ))/2
colors = {"blue": (FEAT_BLUE, FEAT_BLUE_STROKE), "orange": (DECODE_ORANGE, DECODE_ORANGE_STROKE),
          "green": (MASK_GREEN, MASK_GREEN_STROKE)}
tok_centers_y = []
for i, t in enumerate(tokens):
    fy = strip_top + i*(SQ+GAP_SQ)
    fill, stroke = colors[t]
    L.append(f'<rect x="{STRIP_X}" y="{fy}" width="{SQ}" height="{SQ}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    tok_centers_y.append(fy+SQ/2)
L.append(f'<line x1="{TE_X+TE_W}" y1="{row_y+TM_H/2}" x2="{STRIP_X}" y2="{row_y+TM_H/2}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# KV cache dashed box with 3 blue squares (draft layer context, inside draft layer box)
DL_X = STRIP_X + SQ + 46
DL_W = 330
DL_TOP = strip_top - 20
DL_H = len(tokens)*(SQ+GAP_SQ) + 40
L.append(f'<rect x="{DL_X}" y="{DL_TOP}" width="{DL_W}" height="{DL_H}" rx="12" '
          'fill="#dcfce7" stroke="#16a34a" stroke-width="2" stroke-dasharray="7,4"/>')
L.append(f'<text x="{DL_X+DL_W/2}" y="{DL_TOP-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#166534">Draft Layer 1</text>')

KV_X = DL_X + 16
KV_W = 60
KV_H = 3*(SQ+GAP_SQ)
KV_TOP = DL_TOP + 20
L.append(f'<rect x="{KV_X}" y="{KV_TOP}" width="{KV_W}" height="{KV_H+8}" rx="6" '
          'fill="none" stroke="#3a8fb0" stroke-width="1.4" stroke-dasharray="4,3"/>')
L.append(f'<text x="{KV_X+KV_W/2}" y="{KV_TOP-6}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" fill="#1e5c73">KV Cache</text>')
for i in range(3):
    fy = KV_TOP + 4 + i*(SQ+GAP_SQ)
    L.append(f'<rect x="{KV_X+8}" y="{fy}" width="{SQ}" height="{SQ}" rx="4" '
              f'fill="{FEAT_BLUE}" stroke="{FEAT_BLUE_STROKE}" stroke-width="1.4"/>')

# arrows from strip tokens into draft layer (context features -> KV cache; decode/mask -> attention)
for i, t in enumerate(tokens):
    ty = tok_centers_y[i]
    if t == "blue":
        target_x = KV_X + 8
    else:
        target_x = DL_X + 120
    L.append(f'<line x1="{STRIP_X+SQ}" y1="{ty}" x2="{target_x}" y2="{ty}" '
              f'stroke="{colors[t][1]}" stroke-width="1.2" opacity="0.65" marker-end="url(#a)"/>')

# Bidirectional Attention + MLP boxes inside draft layer
ATT_X = DL_X + 120
ATT_W = 90
ATT_TOP = DL_TOP + 20
ATT_H = DL_H - 40
L.append(f'<rect x="{ATT_X}" y="{ATT_TOP}" width="{ATT_W}" height="{ATT_H}" rx="8" '
          'fill="#f1f5f9" stroke="#334155" stroke-width="1.4"/>')
L.append(f'<text x="{ATT_X+ATT_W/2}" y="{ATT_TOP+ATT_H/2-6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#0f172a">Bidirectional</text>')
L.append(f'<text x="{ATT_X+ATT_W/2}" y="{ATT_TOP+ATT_H/2+10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#0f172a">Attention</text>')

MLP_X = ATT_X + ATT_W + 20
MLP_W = 60
L.append(f'<rect x="{MLP_X}" y="{ATT_TOP}" width="{MLP_W}" height="{ATT_H}" rx="8" '
          'fill="#f1f5f9" stroke="#334155" stroke-width="1.4"/>')
L.append(f'<text x="{MLP_X+MLP_W/2}" y="{ATT_TOP+ATT_H/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#0f172a">MLP</text>')
L.append(f'<line x1="{ATT_X+ATT_W}" y1="{ATT_TOP+ATT_H/2}" x2="{MLP_X}" y2="{ATT_TOP+ATT_H/2}" '
          'stroke="#334155" stroke-width="1.4" marker-end="url(#a)"/>')

# output tokens after MLP (orange + 4 green) leading to Draft Layer 2 ... N
OUT_X = MLP_X + MLP_W + 30
out_tokens = ["orange"] + ["green"]*4
out_top = ATT_TOP + ATT_H/2 - (len(out_tokens)*(SQ+GAP_SQ))/2
for i, t in enumerate(out_tokens):
    fy = out_top + i*(SQ+GAP_SQ)
    fill, stroke = colors[t]
    L.append(f'<rect x="{OUT_X}" y="{fy}" width="{SQ}" height="{SQ}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
L.append(f'<line x1="{MLP_X+MLP_W}" y1="{ATT_TOP+ATT_H/2}" x2="{OUT_X}" y2="{ATT_TOP+ATT_H/2}" '
          'stroke="#334155" stroke-width="1.4" marker-end="url(#a)"/>')

# Draft Layer 2..N box (compressed) + ellipsis
DL2_X = DL_X + DL_W + 40
DL2_W = 90
L.append(f'<rect x="{DL2_X}" y="{DL_TOP}" width="{DL2_W}" height="{DL_H}" rx="10" '
          'fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
L.append(f'<text x="{DL2_X+DL2_W/2}" y="{DL_TOP+DL_H/2-4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#166534">Draft</text>')
L.append(f'<text x="{DL2_X+DL2_W/2}" y="{DL_TOP+DL_H/2+12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#166534">Layer 2..N</text>')
L.append(f'<line x1="{OUT_X+SQ}" y1="{ATT_TOP+ATT_H/2}" x2="{DL2_X}" y2="{ATT_TOP+ATT_H/2}" '
          'stroke="#334155" stroke-width="1.4" marker-end="url(#a)"/>')

# top loop-back line: feature strip feeds every draft layer's KV cache (drawn as line over top)
loop_y = DL_TOP - 34
L.append(f'<line x1="{STRIP_X+SQ/2}" y1="{strip_top}" x2="{STRIP_X+SQ/2}" y2="{loop_y}" '
          'stroke="#3a8fb0" stroke-width="1.3" stroke-dasharray="4,3"/>')
L.append(f'<line x1="{STRIP_X+SQ/2}" y1="{loop_y}" x2="{DL2_X+DL2_W/2}" y2="{loop_y}" '
          'stroke="#3a8fb0" stroke-width="1.3" stroke-dasharray="4,3"/>')
L.append(f'<line x1="{DL2_X+DL2_W/2}" y1="{loop_y}" x2="{DL2_X+DL2_W/2}" y2="{DL_TOP}" '
          'stroke="#3a8fb0" stroke-width="1.3" stroke-dasharray="4,3" marker-end="url(#a)"/>')
L.append(f'<text x="{(STRIP_X+DL2_X+DL2_W/2)/2}" y="{loop_y-6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="9.5" fill="#1e5c73">同一份融合特征逐层复用</text>')

# Target LM Head + output
LM_X = DL2_X + DL2_W + 46
LM_W = 110
L.append(f'<rect x="{LM_X}" y="{DL_TOP+DL_H/2-25}" width="{LM_W}" height="50" rx="8" '
          'fill="#e2e8f0" stroke="#334155" stroke-width="1.6"/>')
L.append(f'<text x="{LM_X+LM_W/2}" y="{DL_TOP+DL_H/2+5}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#0f172a">Target LM Head</text>')
L.append(f'<line x1="{DL2_X+DL2_W}" y1="{DL_TOP+DL_H/2}" x2="{LM_X}" y2="{DL_TOP+DL_H/2}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<text x="{LM_X+LM_W+70}" y="{DL_TOP+DL_H/2+5}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">for speculative</text>')
L.append(f'<text x="{LM_X+LM_W+70}" y="{DL_TOP+DL_H/2+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">decoding &lt;eos&gt;</text>')
L.append(f'<line x1="{LM_X+LM_W}" y1="{DL_TOP+DL_H/2}" x2="{LM_X+LM_W+30}" y2="{DL_TOP+DL_H/2}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# legend
LEG_Y = DL_TOP + DL_H + 60
LEG_X = PAD
items = [("Fused Target Context Feature", FEAT_BLUE, FEAT_BLUE_STROKE),
         ("Target Decode Token", DECODE_ORANGE, DECODE_ORANGE_STROKE),
         ("Mask Token", MASK_GREEN, MASK_GREEN_STROKE)]
L.append(f'<rect x="{LEG_X-10}" y="{LEG_Y-24}" width="{w-2*PAD+20}" height="46" rx="8" '
          'fill="none" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>')
lx = LEG_X + 10
for label, fill, stroke in items:
    L.append(f'<rect x="{lx}" y="{LEG_Y-10}" width="18" height="18" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{lx+26}" y="{LEG_Y+4}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 26 + 12*len(label) + 40

foot_y = LEG_Y + 50
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">与 EAGLE 的关键区别:融合特征(浅蓝)不是只拼在输入层,而是逐层写进每个 draft 层各自的 KV cache,供该层注意力直接读取。</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-2.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
