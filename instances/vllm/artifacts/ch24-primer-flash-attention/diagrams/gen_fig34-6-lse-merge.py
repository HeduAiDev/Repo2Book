#!/usr/bin/env python3
"""flow 模板:两段部分注意力 (O,lse) 经 max-稳定化加权合并,输出与一次性注意力在浮点舍入内恒等。
以 token 0 为例给出具体数值。box 高度按内容行数计算,零魔数、零溢出。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "LSE 合并(merge_attn_states 数学原型)— token 0"
SUBTITLE = "两段各交出 (部分输出 O, logsumexp lse);以 max(lse) 稳定化后加权合并,结果与拼接 KV 一次性注意力在浮点舍入内恒等"

PAD, TOP = 40, 96
BOX_W = 280
TITLE_TOP, LINE_H, BOX_PAD_BOTTOM = 22, 19, 16

def box_height(has_title, n_lines):
    start = TITLE_TOP + (20 if has_title else 0)
    return start + max(n_lines - 1, 0) * LINE_H + BOX_PAD_BOTTOM

PREFIX_LINES = ["O_pre=[0.5, 0.5], lse_pre=1.4003"]
SUFFIX_LINES = ["O_suf=[2.0, 0.0], lse_suf=0.3536"]
MERGE_LINES = ["M=max(lse_pre,lse_suf)=1.4003", "w_pre≈0.7405, w_suf≈0.2595",
               "合并 O=[0.8898, 0.3701]", "合并 lse=1.7012"]
REF_LINES = ["拼接 KV 一次性注意力", "O=[0.8898, 0.3701]", "误差≈2e-16(舍入)"]

PREFIX_H = box_height(True, len(PREFIX_LINES))
SUFFIX_H = box_height(True, len(SUFFIX_LINES))
MERGE_H = box_height(True, len(MERGE_LINES))
REF_H = box_height(True, len(REF_LINES))

# 三列:前缀、后缀 -> 合并 -> 一次性参照
prefix_x = PAD
suffix_x = PAD
merge_x = PAD + BOX_W + 190
ref_x = merge_x + BOX_W + 190

prefix_y = TOP
suffix_y = TOP + PREFIX_H + 60
mid_between = (prefix_y + PREFIX_H / 2 + suffix_y + SUFFIX_H / 2) / 2
merge_y = mid_between - MERGE_H / 2
ref_y = mid_between - REF_H / 2

w = ref_x + BOX_W + PAD
h = suffix_y + SUFFIX_H + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#047857"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+10}" font-family="sans-serif" font-size="12" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

def box(x, y, box_h, fill, stroke, lines, title, title_fill):
    out = [f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{box_h}" rx="8" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>']
    yy = y + TITLE_TOP
    out.append(f'<text x="{x+BOX_W/2}" y="{yy}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                f'fill="{title_fill}">{esc(title)}</text>')
    yy += 20
    for line in lines:
        out.append(f'<text x="{x+BOX_W/2}" y="{yy}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="12" fill="{title_fill}">{esc(line)}</text>')
        yy += LINE_H
    return out

L += box(prefix_x, prefix_y, PREFIX_H, "#dbeafe", "#1d4ed8",
         PREFIX_LINES, "前缀段(causal=False)", "#1e3a8a")
L += box(suffix_x, suffix_y, SUFFIX_H, "#fef3c7", "#b45309",
         SUFFIX_LINES, "后缀段(causal=True)", "#78350f")
L += box(merge_x, merge_y, MERGE_H, "#ecfdf5", "#047857",
         MERGE_LINES, "LSE 加权合并", "#047857")
L += box(ref_x, ref_y, REF_H, "#eff6ff", "#1d4ed8",
         REF_LINES, "一次性参照", "#1e3a8a")

# 箭头:前缀/后缀 -> 合并(端点落在合并框左边缘的上/下四分位)
L.append(f'<line x1="{prefix_x+BOX_W}" y1="{prefix_y+PREFIX_H/2}" '
          f'x2="{merge_x}" y2="{merge_y+MERGE_H*0.3}" stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{suffix_x+BOX_W}" y1="{suffix_y+SUFFIX_H/2}" '
          f'x2="{merge_x}" y2="{merge_y+MERGE_H*0.7}" stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')
# 箭头:合并 -> 参照(对照,绿色表示相等)
L.append(f'<line x1="{merge_x+BOX_W}" y1="{merge_y+MERGE_H/2}" '
          f'x2="{ref_x}" y2="{ref_y+REF_H/2}" stroke="#047857" stroke-width="2.2" marker-end="url(#g)"/>')
L.append(f'<text x="{(merge_x+BOX_W+ref_x)/2}" y="{(merge_y+MERGE_H/2+ref_y+REF_H/2)/2-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#047857">{esc("舍入内恒等")}</text>')

foot_y = h - 20
FOOT = "以 max_lse 稳定化后按 e^(lse-max) 求两段权重,加权合并 O——结果与不拆分的精确注意力只差 float64 舍入(~2e-16),这是 cascade attention 拆前缀/后缀而不损精度的正确性保证。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc(FOOT)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig34-6-lse-merge.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
