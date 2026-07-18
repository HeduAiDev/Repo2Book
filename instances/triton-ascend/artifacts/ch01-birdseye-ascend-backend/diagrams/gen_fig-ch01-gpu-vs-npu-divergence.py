#!/usr/bin/env python3
"""fig-ch01-gpu-vs-npu-divergence — before-after 模板变体（两侧步数不同：5 vs 3）。
共同祖先 ttir 之后，GPU 走 5 段保留 SIMT 指针张量；NPU 走 3 段早早换成结构化
memref。分叉点精确落在 add_stages。坐标全部由循环/常量计算。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

COMMON = "ttir\n（make_ttir，两边共用）"
GPU_STEPS = ["ttir", "ttgir\n（叠 SIMT layout/warp）", "llir", "ptx", "cubin"]
NPU_STEPS = ["ttir", "ttadapter\n（结构化 Linalg memref）", "npubin"]

BOX_W, BOX_H, VGAP, PAD = 250, 56, 20, 40
COL_GAP = 120
TOP_COMMON = 70
TOP_SPLIT = TOP_COMMON + BOX_H + 60

n_max = max(len(GPU_STEPS), len(NPU_STEPS))
col_h = n_max * (BOX_H + VGAP) - VGAP
w = PAD * 2 + BOX_W * 2 + COL_GAP
h = TOP_SPLIT + col_h + 24 + 18 + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc("共同祖先 ttir 之后：GPU 5 段 vs NPU 3 段——分叉点在 add_stages")}</text>']

# 共同祖先节点（居中）
common_cx = w / 2
lines = COMMON.split("\n")
L.append(f'<rect x="{common_cx-BOX_W/2}" y="{TOP_COMMON}" width="{BOX_W}" height="{BOX_H}" rx="10" '
         f'fill="#e0e7ff" stroke="#4338ca" stroke-width="2.4"/>')
y0 = TOP_COMMON + BOX_H/2 - (len(lines)-1)*8 + 4
for k, ln in enumerate(lines):
    L.append(f'<text x="{common_cx}" y="{y0+k*15}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="#3730a3">{esc(ln)}</text>')

# 两列起点
gpu_cx = PAD + BOX_W/2
npu_cx = w - PAD - BOX_W/2

# 从共同节点分叉的两条线
split_y = TOP_COMMON + BOX_H
L.append(f'<path d="M {common_cx} {split_y} L {common_cx} {split_y+24} L {gpu_cx} {split_y+24} '
         f'L {gpu_cx} {TOP_SPLIT-4}" fill="none" stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<path d="M {common_cx} {split_y} L {common_cx} {split_y+24} L {npu_cx} {split_y+24} '
         f'L {npu_cx} {TOP_SPLIT-4}" fill="none" stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')

L.append(f'<text x="{gpu_cx}" y="{TOP_SPLIT-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#0369a1">{esc("GPU 路（基座 CUDABackend）")}</text>')
L.append(f'<text x="{npu_cx}" y="{TOP_SPLIT-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#b45309">{esc("NPU 路（AscendBackend）")}</text>')

def draw_col(cx, steps, color_fill, color_stroke, color_text, hot_idx):
    for i, step in enumerate(steps):
        y = TOP_SPLIT + i * (BOX_H + VGAP)
        hl = (i == hot_idx)
        lines = step.split("\n")
        fill = "#fef3c7" if hl else color_fill
        stroke = "#d97706" if hl else color_stroke
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{2.2 if hl else 1.4}"/>')
        n = len(lines)
        y0 = y + BOX_H/2 - (n-1)*8 + 4
        for k, ln in enumerate(lines):
            L.append(f'<text x="{cx}" y="{y0+k*15}" text-anchor="middle" font-family="sans-serif" '
                     f'font-size="12" fill="{color_text}">{esc(ln)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#94a3b8" stroke-width="1.6" marker-end="url(#a)"/>')

draw_col(gpu_cx, GPU_STEPS, "#e0f2fe", "#0369a1", "#0c4a6e", hot_idx=1)
draw_col(npu_cx, NPU_STEPS, "#fef3c7", "#b45309", "#78350f", hot_idx=1)

# 计数标注（拆两行，居中于各自列宽内，避免溢出画布）
gpu_count_y = TOP_SPLIT + len(GPU_STEPS) * (BOX_H + VGAP) - VGAP + 24
npu_count_y = TOP_SPLIT + len(NPU_STEPS) * (BOX_H + VGAP) - VGAP + 24
L.append(f'<text x="{gpu_cx}" y="{gpu_count_y}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="#0369a1">'
         f'{esc("5 段：ttir/ttgir/llir/ptx/cubin")}</text>')
L.append(f'<text x="{gpu_cx}" y="{gpu_count_y+18}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11.5" fill="#0369a1">'
         f'{esc("（nvidia compiler.py:L385-L389）")}</text>')
L.append(f'<text x="{npu_cx}" y="{npu_count_y}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="#b45309">'
         f'{esc("3 段：ttir/ttadapter/npubin")}</text>')
L.append(f'<text x="{npu_cx}" y="{npu_count_y+18}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11.5" fill="#b45309">'
         f'{esc("（compiler.py:L941/L949/L959）")}</text>')

foot_y = h - 42
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc("根因：NPU 不是 SIMT 架构，昇腾在 ttadapter 就把 tensor-of-pointers 逆向还原成结构化 memref，")}</text>')
L.append(f'<text x="{w/2}" y="{foot_y+18}" text-anchor="middle" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc("不需要 TTGIR 那层 warp/layout。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch01-gpu-vs-npu-divergence.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
