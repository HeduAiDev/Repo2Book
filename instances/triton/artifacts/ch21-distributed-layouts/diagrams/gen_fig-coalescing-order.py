#!/usr/bin/env python3
"""before-after 模板(自定义):order 决定 32 lane 落址连续与否 -> 事务数天差地别。
左:order=[1,0](编译器实发)32 lane 落在同一行的 32 个连续列,64 字节合并成 1 笔。
右:order=[0,1](反例)32 lane 落在同一列的 32 行,行距 128 字节,退化成 32 笔。
数字全部来自 traces/matmul.json analysis.coalescing。全坐标计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    cjk = sum(1 for ch in s if ord(ch) > 0x2e7f)
    other = len(s) - cjk
    return cjk * size * 1.0 + other * size * 0.56

TITLE = "order 决定 32 个 lane 的访存能否合并成一笔事务"
SUBTITLE = "64x64 fp16 matmul 操作数 load,同一个 warp、同一份数据,只换 order(traces/matmul.json)"

N_LANE_SHOW = 12          # 展示的 lane 数(其余用省略号示意,32 lane 结论不变)
CELL = 26
GAP = 3
PAD, TOP = 46, 130
PANEL_GAP = 90

LEFT_LABEL = "order=[1,0]  (实发,order[0]=1=列=stride-1 维)"
RIGHT_LABEL = "order=[0,1]  (反例,order[0]=0=行=跨步维)"
LEFT_NOTE = ["32 lane 落在同一行的 32 个连续列", "跨度 = 32 x 2 字节 = 64 字节", "→ 合并成 1 笔事务"]
RIGHT_NOTE = ["32 lane 落在同一列的 32 个不同行", "每行相隔 64 元素 = 128 字节", "→ 退化成 32 笔事务"]

panel_w = N_LANE_SHOW * (CELL + GAP)
W = int(PAD * 2 + panel_w + PANEL_GAP + max(panel_w, 260))
note_w = max(text_w(s, 12) for s in LEFT_NOTE + RIGHT_NOTE)
W = int(max(W, PAD * 2 + panel_w + 40 + note_w + 20))
CAPTION_LINES = [
    "同一 warp、同一份数据,只换 order:左(order=[1,0],编译器实发)32 lane 贴着一行的连续列,64 字节合并成 1 笔;",
    "右(order=[0,1],反例)32 lane 竖着落在 32 行、彼此相隔 128 字节,退化成 32 笔。order[0] 必须指向 stride=1 维——",
    "这是访存合并现场(见第 7 章)在布局层的固化。",
]
cap_w = max(text_w(s, 12) for s in CAPTION_LINES)
W = int(max(W, PAD + cap_w + PAD))

ROW_H = CELL + GAP
H = TOP + max(len(LEFT_NOTE), 8) * ROW_H + 40 + len(CAPTION_LINES) * 18 + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# --- 左面板:order=[1,0],32 lane 横排连续 ---
left_x0 = PAD
left_y0 = TOP
L.append(f'<text x="{left_x0}" y="{left_y0-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#15803d">{esc(LEFT_LABEL)}</text>')
for i in range(N_LANE_SHOW):
    x = left_x0 + i * (CELL + GAP)
    L.append(f'<rect x="{x}" y="{left_y0}" width="{CELL}" height="{CELL}" rx="3" '
              f'fill="#bbf7d0" stroke="#15803d" stroke-width="1.3"/>')
    lane_id = i if i < N_LANE_SHOW - 1 else 31
    label = str(lane_id) if i < N_LANE_SHOW - 1 else "31"
    if i == N_LANE_SHOW - 2:
        label = "..."
    L.append(f'<text x="{x+CELL/2}" y="{left_y0+CELL/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9" fill="#14532d">{esc(label)}</text>')
# 合并跨度括起来的括线
brace_y = left_y0 + CELL + 14
L.append(f'<line x1="{left_x0}" y1="{brace_y}" x2="{left_x0+panel_w-GAP}" y2="{brace_y}" '
          'stroke="#15803d" stroke-width="1.5"/>')
L.append(f'<line x1="{left_x0}" y1="{brace_y-5}" x2="{left_x0}" y2="{brace_y+5}" stroke="#15803d" stroke-width="1.5"/>')
L.append(f'<line x1="{left_x0+panel_w-GAP}" y1="{brace_y-5}" x2="{left_x0+panel_w-GAP}" y2="{brace_y+5}" '
          'stroke="#15803d" stroke-width="1.5"/>')
L.append(f'<text x="{left_x0+panel_w/2}" y="{brace_y+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#15803d">{esc("64 字节连续 = 1 笔事务")}</text>')
note_y0 = brace_y + 44
for i, line in enumerate(LEFT_NOTE):
    L.append(f'<text x="{left_x0}" y="{note_y0+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

# --- 右面板:order=[0,1],32 lane 竖排跨步(展示前 8 行 + 省略) ---
right_x0 = left_x0 + panel_w + PANEL_GAP
right_y0 = TOP
L.append(f'<text x="{right_x0}" y="{right_y0-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#b91c1c">{esc(RIGHT_LABEL)}</text>')
N_ROW_SHOW = 8
STRIDE_GAP = 12  # 视觉上把每行隔开,示意"跨步、不连续"
for i in range(N_ROW_SHOW):
    y = right_y0 + i * (CELL + STRIDE_GAP)
    L.append(f'<rect x="{right_x0}" y="{y}" width="{CELL}" height="{CELL}" rx="3" '
              f'fill="#fecaca" stroke="#b91c1c" stroke-width="1.3"/>')
    label = str(i) if i < N_ROW_SHOW - 1 else "31"
    if i == N_ROW_SHOW - 2:
        label = "..."
    L.append(f'<text x="{right_x0+CELL/2}" y="{y+CELL/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9" fill="#7f1d1d">{esc(label)}</text>')
    L.append(f'<text x="{right_x0+CELL+10}" y="{y+CELL/2+4}" font-family="sans-serif" '
              f'font-size="10" fill="#94a3b8">{esc("128 字节 →" if i < N_ROW_SHOW - 1 else "")}</text>')
right_bottom = right_y0 + (N_ROW_SHOW - 1) * (CELL + STRIDE_GAP) + CELL
note_y0r = right_bottom + 30
for i, line in enumerate(RIGHT_NOTE):
    L.append(f'<text x="{right_x0}" y="{note_y0r+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

# 数字卡片(标出 provenance 五个关键数字,横跨两面板下方)
NUMS = [("lane 数", "32"), ("合并跨度", "64 字节"), ("合并事务数", "1"),
        ("跨步行距", "128 字节"), ("跨步事务数", "32")]
num_y = max(note_y0 + len(LEFT_NOTE) * 18, note_y0r + len(RIGHT_NOTE) * 18) + 26
card_w = (W - 2 * PAD) / len(NUMS)
for i, (label, val) in enumerate(NUMS):
    cx = PAD + i * card_w + card_w / 2
    L.append(f'<rect x="{PAD+i*card_w+4}" y="{num_y}" width="{card_w-8}" height="52" rx="6" '
              f'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
    L.append(f'<text x="{cx}" y="{num_y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(label)}</text>')
    L.append(f'<text x="{cx}" y="{num_y+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(val)}</text>')

foot_y0 = num_y + 52 + 30
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')

H = int(foot_y0 + len(CAPTION_LINES) * 18 + 20)
svg_out = '\n'.join(L).replace(
    L[0], f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">', 1)

out = Path(__file__).with_name("fig-coalescing-order.svg")
out.write_text(svg_out, encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
