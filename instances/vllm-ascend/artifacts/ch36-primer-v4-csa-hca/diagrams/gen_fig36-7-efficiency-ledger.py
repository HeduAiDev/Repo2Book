#!/usr/bin/env python3
"""state-table 模板改造:CSA/HCA/dense 三种层的 KV 与 FLOPs 逐笔账,
按层比例平均后 hybrid 远低于 dense 基线;下方再叠混合精度与论文口径对照。
数字来自 traces/efficiency.json。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "效率账本:逐层 KV/FLOPs 拆开算,按比例平均后 hybrid 远低于 dense"
SUBTITLE = "L=1,000,000, head_dim=128, k=2048, n_win=1024;示意 compress_ratios=[4,4,4,128]x9(非论文逐层配置复现)"

COLS = ["KV 存量(单 token)", "单 token FLOPs 代理", "相对基线"]
ROW_LABELS = ["CSA (m=4)", "HCA (m'=128)", "dense 基线", "hybrid 平均"]
CELLS = {
    "CSA (m=4)":    ["250,000.0", "64,393,216.0", "—"],
    "HCA (m'=128)": ["7,812.5", "1,131,072.0", "—"],
    "dense 基线":    ["1,000,000.0", "128,000,000.0", "基线(1.0x)"],
    "hybrid 平均":   ["189,453.1", "48,577,680.0", "FLOPs 0.3795 / KV 0.1895"],
}
STATUS = {"hybrid 平均": ["win", "win", "win"], "dense 基线": ["base", "base", "base"]}
COLOR = {"win": ("#ecfdf5", "#047857"), "base": ("#f1f5f9", "#475569")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 220, 46, 36, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 190 + PAD
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        fs = "11.5" if j < 2 else "12"
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

# 混合精度 callout
callout_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 24
box_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{box_w}" height="50" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+20}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#92400e">再叠混合精度存储:每条目 192.0 字节(RoPE维BF16+其余FP8) vs 纯 BF16 256.0 字节 = 0.75</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+38}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">压序列长(0.1895) x 低精度(0.75) —— 两笔账相乘,KV 又近乎减半</text>')

# 论文口径对照(非复现声明)
box2_y = callout_y + 62
L.append(f'<rect x="{PAD}" y="{box2_y}" width="{box_w}" height="66" rx="6" '
          'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{box2_y+22}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#b91c1c">论文口径:27% FLOPs / 10% KV(§1);与本例 0.3795 / 0.1895 方向一致但不是同一数字</text>')
L.append(f'<text x="{box2_y+16 if False else PAD+16}" y="{box2_y+42}" font-family="sans-serif" '
          f'font-size="11.5" fill="#b91c1c">本例用示意 compress_ratios 与训练态 k/n_win 验证账本模型自洽、hybrid 确实双优于两条基线</text>')
L.append(f'<text x="{PAD+16}" y="{box2_y+60}" font-family="sans-serif" font-size="11.5" '
          f'fill="#b91c1c">论文的具体百分比需 DeepSeek 未公开的完整逐层配置 + 低精度细节才能精确复现</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig36-7-efficiency-ledger.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
