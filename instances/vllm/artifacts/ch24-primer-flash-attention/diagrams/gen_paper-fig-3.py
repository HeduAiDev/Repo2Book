#!/usr/bin/env python3
"""论文精髓图重绘:arXiv:2307.08691 Fig.3 —
FA-2 前向传播里不同 warp 之间具体工作划分的示意图:(a) FlashAttention 的 split-K
(K、V 按 warp 切分,Q 全体 warp 共享访问,算完还要跨 warp 通过 shared memory 相加部分结果);
(b) FlashAttention-2 的 split-Q(改成 Q 按 warp 切分,K、V 全体 warp 共享访问,每个 warp
独立算出自己的输出切片,warp 间无需通信)。
忠实重绘自 ar5iv 抓到的原图(assets/figs/flash_partitioning.png、flash2_partitioning.png),
信息结构对齐,配色套本书视觉语言,文字译中。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

FIG_ID = "paper-fig-3"
TITLE = "重绘自 arXiv:2307.08691 Fig.3"
SUBTITLE = "warp 间工作划分:split-K(FlashAttention) vs split-Q(FlashAttention-2)"

SHARED_FILL, SHARED_STROKE = "#dbeafe", "#2563eb"
SPLIT_FILL, SPLIT_STROKE = "#fed7aa", "#c2410c"

PAD, TOP = 40, 118
PANEL_W = 430
PANEL_GAP = 60
N_WARP = 4
BLK_W, BLK_H, GAP_S = 76, 46, 8

w = PAD * 2 + PANEL_W * 2 + PANEL_GAP

L = []
L.append(f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">{esc(SUBTITLE)}</text>')


def warp_row(x, y, n, split, label_prefix):
    """split=True: n 个各自独立小块(橙色);split=False: 一个跨度整块(蓝色,标 Warp 1-4)。"""
    out = []
    if split:
        for i in range(n):
            bx = x + i * (BLK_W + GAP_S)
            out.append(f'<rect x="{bx}" y="{y}" width="{BLK_W}" height="{BLK_H}" rx="5" '
                        f'fill="{SPLIT_FILL}" stroke="{SPLIT_STROKE}" stroke-width="1.8" '
                        f'stroke-dasharray="5,3"/>')
            out.append(f'<text x="{bx+BLK_W/2}" y="{y+BLK_H/2+4}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="11" fill="#7c2d12">'
                        f'{esc(f"{label_prefix} {i+1}")}</text>')
    else:
        full_w = n * BLK_W + (n - 1) * GAP_S
        out.append(f'<rect x="{x}" y="{y}" width="{full_w}" height="{BLK_H}" rx="5" '
                    f'fill="{SHARED_FILL}" stroke="{SHARED_STROKE}" stroke-width="1.8" '
                    f'stroke-dasharray="2,3"/>')
        out.append(f'<text x="{x+full_w/2}" y="{y+BLK_H/2+4}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="12" font-weight="bold" '
                    f'fill="#1e3a8a">{esc(f"{label_prefix} 1-4（全体共享）")}</text>')
    return out


def warp_col(x, y, n, split, label_prefix):
    out = []
    if split:
        for i in range(n):
            by = y + i * (BLK_H + GAP_S)
            out.append(f'<rect x="{x}" y="{by}" width="{BLK_W}" height="{BLK_H}" rx="5" '
                        f'fill="{SPLIT_FILL}" stroke="{SPLIT_STROKE}" stroke-width="1.8" '
                        f'stroke-dasharray="5,3"/>')
            out.append(f'<text x="{x+BLK_W/2}" y="{by+BLK_H/2+4}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="11" fill="#7c2d12">'
                        f'{esc(f"{label_prefix} {i+1}")}</text>')
    else:
        full_h = n * BLK_H + (n - 1) * GAP_S
        out.append(f'<rect x="{x}" y="{y}" width="{BLK_W}" height="{full_h}" rx="5" '
                    f'fill="{SHARED_FILL}" stroke="{SHARED_STROKE}" stroke-width="1.8" '
                    f'stroke-dasharray="2,3"/>')
        out.append(f'<text x="{x+BLK_W/2}" y="{y+full_h/2}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#1e3a8a" '
                    f'transform="rotate(-90 {x+BLK_W/2} {y+full_h/2})">'
                    f'{esc(f"{label_prefix} 1-4（全体共享）")}</text>')
    return out


def panel(px, title, q_split, kv_split, verdict_lines, verdict_color):
    lines = []
    lines.append(f'<text x="{px+PANEL_W/2}" y="{TOP-6}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="14.5" font-weight="bold" fill="#0f172a">{esc(title)}</text>')

    kt_row_y = TOP + 34
    kt_x = px + 30
    lines.append(f'<text x="{kt_x}" y="{kt_row_y-10}" font-family="sans-serif" font-size="12.5" '
                 f'font-weight="bold" fill="#0f172a">{esc("Kᵀ")}</text>')
    lines += warp_row(kt_x, kt_row_y, N_WARP, kv_split, "Warp")

    v_col_y = kt_row_y + BLK_H + 40
    v_col_x = px + PANEL_W - BLK_W - 30
    lines.append(f'<text x="{v_col_x+BLK_W/2}" y="{v_col_y-10}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                 f'fill="#0f172a">{esc("V")}</text>')
    lines += warp_col(v_col_x, v_col_y, N_WARP, kv_split, "Warp")

    q_x = px + 30
    q_y = v_col_y
    lines.append(f'<text x="{q_x+BLK_W/2}" y="{q_y-10}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc("Q")}</text>')
    lines += warp_col(q_x, q_y, N_WARP, q_split, "Warp")

    box_y = v_col_y + N_WARP * (BLK_H + GAP_S) - GAP_S + 34
    box_h = 22 + 18 * len(verdict_lines)
    lines.append(f'<rect x="{px}" y="{box_y}" width="{PANEL_W}" height="{box_h}" rx="6" '
                 f'fill="{verdict_color[0]}" stroke="{verdict_color[1]}" stroke-width="1.6"/>')
    for i, vl in enumerate(verdict_lines):
        lines.append(f'<text x="{px+PANEL_W/2}" y="{box_y+22+i*18}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                     f'fill="{verdict_color[1]}">{esc(vl)}</text>')
    return lines, box_y + box_h


px_a = PAD
px_b = PAD + PANEL_W + PANEL_GAP

lines_a, bottom_a = panel(px_a, "(a) FlashAttention — split-K",
           q_split=False, kv_split=True,
           verdict_lines=["每个 warp 各算 QKᵀ 的一部分,须写 shared memory",
                          "再跨 warp 同步相加 —— 通信开销拖慢前向"],
           verdict_color=("#fee2e2", "#b91c1c"))
lines_b, bottom_b = panel(px_b, "(b) FlashAttention-2 — split-Q",
           q_split=True, kv_split=False,
           verdict_lines=["每个 warp 独立算出自己的输出切片",
                          "warp 间无需通信 —— 省掉 shared-memory 读写"],
           verdict_color=("#dcfce7", "#15803d"))
L += lines_a
L += lines_b
panel_bottom = max(bottom_a, bottom_b)

# 中间箭头(split-K -> split-Q)
mid_x = (px_a + PANEL_W + px_b) / 2
mid_y = TOP + 140
L.append(f'<line x1="{px_a+PANEL_W-6}" y1="{mid_y}" x2="{px_b+6}" y2="{mid_y}" '
         'stroke="#64748b" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#gray)"/>')
L.append(f'<text x="{mid_x}" y="{mid_y-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" font-weight="bold" fill="#334155">{esc("切分对象换一边")}</text>')

# 图例(放在两侧 verdict 框下方,留足间距,不与其重叠)
legend_y = panel_bottom + 34
L.append(f'<rect x="{PAD}" y="{legend_y}" width="26" height="18" rx="3" '
         f'fill="{SHARED_FILL}" stroke="{SHARED_STROKE}" stroke-width="1.6" stroke-dasharray="2,3"/>')
L.append(f'<text x="{PAD+34}" y="{legend_y+14}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("全体 warp 共享访问(Accessed by all warps)")}</text>')
L.append(f'<rect x="{PAD+430}" y="{legend_y}" width="26" height="18" rx="3" '
         f'fill="{SPLIT_FILL}" stroke="{SPLIT_STROKE}" stroke-width="1.6" stroke-dasharray="5,3"/>')
L.append(f'<text x="{PAD+464}" y="{legend_y+14}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("按 warp 切分(Split across different warps)")}</text>')

foot_y = legend_y + 44
FOOT = "两图内容(warp 数=4、切分对象、通信开销结论)取自论文原图与正文 §3.3 描述,非杜撰。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="10.5" '
         f'fill="#64748b">{esc(FOOT)}</text>')

h = foot_y + 20
header = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
          '<defs><marker id="gray" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
          f'<rect width="{w}" height="{h}" fill="white"/>']
L = header + L + ['</svg>']
out = Path(__file__).with_name(f"{FIG_ID}.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, canvas {w}x{h}")
