#!/usr/bin/env python3
"""state-table 模板:coalescing 成不成看相邻 lane 地址是否连续。三行对比连续/
中间跨步/最坏跨步,列=lane 地址跨度/事务数/有效带宽,每行配一条 32-lane 地址
条形示意图。连续行(stride 1)同色块=同落一条 128B 事务行(32 格全同色);中间行
(stride 8/32B 步)每 4 个连号共享一色、共 8 色,配色确实代表事务分组(4 lane
共享一条 128B 事务行);最坏行(stride 32/128B 步)每格独占一笔事务,配色仅为
区分相邻格、不代表分组——三行语义不同,行内小注各自标出。
数字取自正文 §6 定稿数值表(narrative/chapter.md 表:连续/跨步 stride 8/跨步
stride 32 三行,及不变量段 r(i)=⌊32i/128⌋=⌊i/4⌋ 的逐步推导)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "coalescing：相邻 lane 地址是否连续决定合并成几笔事务"
SUBTITLE = "warp=32 lane，elem_size=4B(fp32)，txn_size=128B（32 个 4B 元素恰好装满 1 笔事务）"

MODES = [
    # name, addr_expr, span, txns, bw, color, kind(uniform=32全同色/grouped=真分组/decor=仅区分相邻不代表分组)
    ("连续 stride 1(4B 步)", "base + 4·i", "128 B", 1, "100%", "#16a34a", "uniform"),
    ("跨步 stride 8(32B 步)", "base + 32·i", "1024 B", 8, "12.5%", "#d97706", "grouped"),
    ("跨步 stride 32(128B 步)", "base + 128·i", "4096 B", 32, "3.1%", "#dc2626", "decor"),
]

STEP_BYTES = {1: 4, 8: 32, 32: 128}
GROUP_PALETTE = ["#93c5fd", "#86efac", "#fde047", "#fca5a5",
                 "#c4b5fd", "#67e8f9", "#fdba74", "#a5b4fc"]  # 8 支互异色,代表真实分组
DECOR_PALETTE = ["#93c5fd", "#86efac", "#fde047", "#fca5a5", "#c4b5fd", "#67e8f9"]  # 6 色循环,仅装饰性区分相邻格
LEGEND = {
    "uniform": "32 个 lane（同色=同一 128B 事务行，32 格全落一笔）",
    "grouped": "32 个 lane（每 4 连号同色=同一 128B 事务行，共 8 色=8 笔；配色代表真实分组）",
    "decor": "32 个 lane（每格独占一笔事务；配色仅区分相邻格、不表示分组）",
}

PAD, TOP = 40, 118
LABEL_W = 210
BAR_W = 620
BAR_H = 30
ROW_GAP = 88
STAT_W = 260

w = PAD * 2 + LABEL_W + BAR_W + STAT_W
h = TOP + len(MODES) * ROW_GAP + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="12.5" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

N_LANE = 32
TXN = 128  # bytes
ELEM = 4

for r, (name, addr_expr, span, txns, bw, color, kind) in enumerate(MODES):
    ry = TOP + r * ROW_GAP
    L.append(f'<text x="{PAD}" y="{ry+14}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{PAD}" y="{ry+32}" font-family="monospace" font-size="11.5" '
              f'fill="#475569">{esc(addr_expr)}</text>')

    bar_x = PAD + LABEL_W
    bar_y = ry
    # 32 个 lane 映射到事务行:step 由 txns 数决定 lane 分组,txn_row = (i*step)//128
    step_bytes = STEP_BYTES[txns]
    lane_w = BAR_W / N_LANE
    for i in range(N_LANE):
        addr = i * step_bytes
        txn_row = addr // TXN
        if kind == "uniform":
            cell_color = "#86efac"
        elif kind == "grouped":
            cell_color = GROUP_PALETTE[txn_row % len(GROUP_PALETTE)]
        else:  # decor:仅装饰性区分相邻格,不代表真实分组
            cell_color = DECOR_PALETTE[txn_row % len(DECOR_PALETTE)]
        x = bar_x + i * lane_w
        L.append(f'<rect x="{x}" y="{bar_y}" width="{lane_w-0.6}" height="{BAR_H}" '
                  f'fill="{cell_color}" stroke="#475569" stroke-width="0.4"/>')
    L.append(f'<rect x="{bar_x}" y="{bar_y}" width="{BAR_W}" height="{BAR_H}" '
              'fill="none" stroke="#0f172a" stroke-width="1.3"/>')
    legend = LEGEND[kind]
    L.append(f'<text x="{bar_x}" y="{bar_y+BAR_H+16}" font-family="sans-serif" font-size="10.5" '
              f'fill="#64748b">{esc(legend)}</text>')

    stat_x = bar_x + BAR_W + 24
    L.append(f'<rect x="{stat_x}" y="{bar_y-4}" width="{STAT_W-24}" height="{BAR_H+8}" rx="6" '
              f'fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="1.6"/>')
    L.append(f'<text x="{stat_x+10}" y="{bar_y+14}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="{color}">{esc(f"{txns} 笔事务")}</text>')
    L.append(f'<text x="{stat_x+10}" y="{bar_y+30}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="{color}">{esc(f"带宽 {bw}")}</text>')

foot_y0 = TOP + len(MODES) * ROW_GAP + 24
FOOT = [
    "结论:32 lane 触及的对齐事务行数随地址步长单调不减——1 笔(100%)→8 笔(12.5%)→32 笔(3.1%),",
    "步长每涨一档、事务数跟着涨,带宽跟着掉;步长达 128B 时每 lane 独占一行、32 笔封顶。",
    "事务量差 32×,这是本章性能收益的落点:block pointer 携带 stride/order,比 legacy 更易让编译器",
    "生成连续地址(合并)。",
]
for i, line in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*19}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch07-coalescing.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
