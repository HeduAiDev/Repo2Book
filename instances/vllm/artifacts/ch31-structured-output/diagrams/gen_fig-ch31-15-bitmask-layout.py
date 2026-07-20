#!/usr/bin/env python3
"""fig-ch31-15: 位掩码内存布局——每 token 1 位打包进 int32，一行宽 ceil(|V|/32)，比同一行 logits 小约 32 倍。
template: layout"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1220, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("位掩码按每 token 1 位打包进 int32，一行只有同一行 logits 的 1/32")}</text>')

PAD = 60
TOP = 70

# 张量形状说明
L.append(f'<text x="{PAD}" y="{TOP}" font-family="sans-serif" font-size="13.5" font-weight="bold" '
          f'fill="#1e293b">{esc("张量形状：(max_num_seqs, (vocab_size + 31) // 32)，dtype=int32")}</text>')

# 一行的可视化：8 个代表性 int32 字（示意，真实是 4688 个字）
ROW_Y = TOP + 40
CELL_W, CELL_H, GAP = 92, 54, 8
N_SHOW = 8
row_w = N_SHOW * (CELL_W + GAP) - GAP
L.append(f'<text x="{PAD}" y="{ROW_Y-14}" font-family="sans-serif" font-size="13" fill="#334155">'
          f'{esc("一行（batch_index 这一条序列）= 4688 个 int32 字（|V|=150000 时），示意前 8 个：")}</text>')
for i in range(N_SHOW):
    x = PAD + i * (CELL_W + GAP)
    L.append(f'<rect x="{x}" y="{ROW_Y}" width="{CELL_W}" height="{CELL_H}" rx="6" '
              f'fill="#fee2e2" stroke="#dc2626" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CELL_W/2}" y="{ROW_Y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#7f1d1d">{esc(f"字 {i}")}</text>')
    L.append(f'<text x="{x+CELL_W/2}" y="{ROW_Y+40}" text-anchor="middle" font-family="monospace" '
              f'font-size="11" fill="#991b1b">{esc("32 bit")}</text>')
L.append(f'<text x="{PAD+row_w+14}" y="{ROW_Y+CELL_H/2+5}" font-family="sans-serif" font-size="13" '
          f'fill="#64748b">{esc("… 共 4688 个字")}</text>')

# 单字放大：32 个 bit，标出全 1（初值 -1 = 全部允许）
ZOOM_Y = ROW_Y + CELL_H + 60
L.append(f'<text x="{PAD}" y="{ZOOM_Y-14}" font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#1e293b">{esc("单字放大：初值 -1（补码 32 个 1）= 这 32 个 token 全部允许")}</text>')
BIT_W, BIT_H = 34, 34
for i in range(32):
    x = PAD + i * BIT_W
    fill = "#bbf7d0" if True else "#fecaca"
    L.append(f'<rect x="{x}" y="{ZOOM_Y}" width="{BIT_W-2}" height="{BIT_H}" '
              f'fill="{fill}" stroke="#16a34a" stroke-width="1"/>')
    L.append(f'<text x="{x+(BIT_W-2)/2}" y="{ZOOM_Y+BIT_H/2+5}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" fill="#14532d">1</text>')
L.append(f'<text x="{PAD}" y="{ZOOM_Y+BIT_H+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("bit 31 ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← → bit 0（32 个连续 token 的允许位）")}</text>')

# 数字对比条：18.3KB vs 585.9KB
BAR_Y = ZOOM_Y + BIT_H + 70
L.append(f'<text x="{PAD}" y="{BAR_Y-14}" font-family="sans-serif" font-size="13.5" font-weight="bold" '
          f'fill="#1e293b">{esc("对照：同一条序列，位掩码一行 vs logits 一行（约小 32 倍）")}</text>')
bar_max = 700
mask_bytes, logits_bytes = 18752, 600000
mask_w = bar_max * mask_bytes / logits_bytes
logits_w = bar_max
L.append(f'<rect x="{PAD}" y="{BAR_Y}" width="{mask_w:.1f}" height="26" rx="5" fill="#dc2626"/>')
L.append(f'<text x="{PAD+mask_w+10:.1f}" y="{BAR_Y+18}" font-family="sans-serif" font-size="13" '
          f'fill="#1e293b">{esc("位掩码：4688 个 int32 = 18752 B（18.3 KB）")}</text>')
L.append(f'<rect x="{PAD}" y="{BAR_Y+38}" width="{logits_w:.1f}" height="26" rx="5" fill="#2563eb"/>')
L.append(f'<text x="{PAD+logits_w+10:.1f}" y="{BAR_Y+56}" font-family="sans-serif" font-size="13" '
          f'fill="#1e293b">{esc("logits：150000 个 float32 = 600000 B（585.9 KB）")}</text>')
L.append(f'<text x="{PAD}" y="{BAR_Y+100}" font-family="sans-serif" font-size="13.5" font-weight="bold" '
          f'fill="#b45309">{esc("比值：31.997 ≈ 32 倍——正因为便宜到这个程度，掩码才能在每步解码热路径上重填一遍")}</text>')

# fill_bitmask 写入标注
FB_Y = BAR_Y + 130
L.append(f'<rect x="{PAD}" y="{FB_Y}" width="{W-2*PAD}" height="44" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-dasharray="5 3"/>')
L.append(f'<text x="{W/2}" y="{FB_Y+27}" text-anchor="middle" font-family="sans-serif" font-size="13" '
          f'fill="#3730a3">{esc("fill_bitmask(bitmask, batch_index) 只写 batch_index 这一行，行宽公式：ceil(vocab_size / 32)")}</text>')

L.append('</svg>')
out = Path("fig-ch31-15-bitmask-layout.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
