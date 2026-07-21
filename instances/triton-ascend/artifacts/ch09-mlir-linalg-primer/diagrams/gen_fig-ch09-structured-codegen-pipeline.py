#!/usr/bin/env python3
"""fig-ch09-structured-codegen-pipeline — m11 结构化 codegen 全景流水。
重绘自 arXiv:2202.03293 Fig.1。自上而下 5 个层级带,Tiled structured 带标注"融合发生在这一层",
目标层三个出口;左侧一条 optionality 虚线旁路箭头。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def esc_bold(s):
    # rsvg-convert/Droid Sans Fallback 在 font-weight="bold" 下把「量」(U+91CF)
    # 误渲染成实心方块(逐次重渲复现,与字号/字体族无关)；用 tspan 把该字降回
    # normal 权重规避——此字体的中文本就不随 bold 变粗,视觉零回归。
    return esc(s).replace('量', '<tspan font-weight="normal">量</tspan>')

TITLE = "结构化 codegen:从张量代数一路降到目标层,每一步都物化在 IR 里"
SUBTITLE = "重绘自 arXiv:2202.03293 Fig.1——tiled structured 这一层同时做融合与多级 tiling"

LEVELS = [
    ("Structured IR", "稠密/稀疏张量代数算子,函数式程序(张量不可变,def-use SSA)"),
    ("Tiled structured", "scf.for + extract_slice/insert_slice;循环体仍是同一算子的小号版本"),
    ("向量抽象", "每操作数一次 transfer_read,向量化计算,transfer_write 写回"),
    ("Buffer 层", "bufferization:张量落地为 memref,尽量少分配少拷贝"),
    ("目标层", "翻到 llvm 方言 / GPU kernel offload / 异步块+任务并行运行时"),
]

PAD = 48
BAND_W = 760
BAND_H = 74
BAND_GAP = 30
TOP = 116
LEFT = PAD + 130   # 给左侧 optionality 旁路留空间

w = LEFT + BAND_W + 480
h = TOP + len(LEVELS) * (BAND_H + BAND_GAP) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
         '<marker id="a2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
         '<marker id="a3" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc_bold(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12" '
         f'fill="#475569">{esc(SUBTITLE)}</text>')

band_y = []
for i, (name, desc) in enumerate(LEVELS):
    by = TOP + i * (BAND_H + BAND_GAP)
    band_y.append(by)
    is_tiled = (name == "Tiled structured")
    fill = "#fef3c7" if is_tiled else "#eff6ff"
    stroke = "#b45309" if is_tiled else "#1d4ed8"
    L.append(f'<rect x="{LEFT}" y="{by}" width="{BAND_W}" height="{BAND_H}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{LEFT+20}" y="{by+28}" font-family="sans-serif" font-size="14.5" '
              f'font-weight="bold" fill="#0f172a">{esc_bold(name)}</text>')
    L.append(f'<text x="{LEFT+20}" y="{by+50}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(desc)}</text>')
    if is_tiled:
        L.append(f'<text x="{LEFT+BAND_W-14}" y="{by+28}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#92400e">{esc("融合发生在这一层")}</text>')
        # 多级 tiling 回环箭头
        loop_r = 16
        loop_cx = LEFT + BAND_W - 36
        loop_cy = by + 58
        L.append(f'<path d="M {loop_cx-loop_r} {loop_cy} '
                  f'a {loop_r} {loop_r} 0 1 1 {2*loop_r} 0" '
                  f'fill="none" stroke="#92400e" stroke-width="1.6" marker-end="url(#a2)"/>')
        L.append(f'<text x="{loop_cx}" y="{loop_cy+30}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="#92400e">'
                  f'{esc("可多级、渐进 tiling")}</text>')
    if i < len(LEVELS) - 1:
        y2 = by + BAND_H
        L.append(f'<line x1="{LEFT+BAND_W/2}" y1="{y2}" x2="{LEFT+BAND_W/2}" y2="{y2+BAND_GAP-4}" '
                  f'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# ---- optionality 旁路(左侧虚线箭头,从 Structured IR 直接绕到目标层)----
op_x = PAD + 40
top_y = band_y[0] + BAND_H / 2
bot_y = band_y[-1] + BAND_H / 2
L.append(f'<path d="M {LEFT-14} {top_y} C {op_x} {top_y}, {op_x} {bot_y}, {LEFT-14} {bot_y}" '
         f'fill="none" stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="6,4" '
         f'marker-end="url(#a3)"/>')
note_y0 = (top_y + bot_y) / 2 - 26
L.append(f'<text x="{PAD}" y="{note_y0}" font-family="sans-serif" font-size="10.5" '
         f'font-weight="bold" fill="#7c3aed">{esc("optionality:")}</text>')
L.append(f'<text x="{PAD}" y="{note_y0+16}" font-family="sans-serif" font-size="10" '
         f'fill="#7c3aed">{esc("论文口径:对某些算子,")}</text>')
L.append(f'<text x="{PAD}" y="{note_y0+32}" font-family="sans-serif" font-size="10" '
         f'fill="#7c3aed">{esc("跳过某些层级、甚至走")}</text>')
L.append(f'<text x="{PAD}" y="{note_y0+48}" font-family="sans-serif" font-size="10" '
         f'fill="#7c3aed">{esc("完全不同的路都可行")}</text>')

# ---- 目标层三出口 ----
last_by = band_y[-1]
exits = ["llvm 方言(CPU)", "GPU kernel offload", "异步块 + 任务并行运行时"]
exit_y = last_by + BAND_H + 60
exit_w = 210
gap = (BAND_W - 3*exit_w) / 2
L.append(f'<line x1="{LEFT+BAND_W/2}" y1="{last_by+BAND_H}" x2="{LEFT+BAND_W/2}" y2="{exit_y-24}" '
         f'stroke="#334155" stroke-width="1.6"/>')
for i, ex in enumerate(exits):
    ex_x = LEFT + i * (exit_w + gap)
    L.append(f'<line x1="{LEFT+BAND_W/2}" y1="{exit_y-24}" x2="{ex_x+exit_w/2}" y2="{exit_y}" '
              f'stroke="#334155" stroke-width="1.4" marker-end="url(#a)"/>')
    L.append(f'<rect x="{ex_x}" y="{exit_y}" width="{exit_w}" height="46" rx="7" '
              f'fill="#dcfce7" stroke="#15803d" stroke-width="1.6"/>')
    L.append(f'<text x="{ex_x+exit_w/2}" y="{exit_y+28}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
              f'fill="#14532d">{esc(ex)}</text>')

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="10.5" '
         f'fill="#64748b">{esc("依据:arXiv:2202.03293 §2.1;tiling 粒度用途见论文原型例(按 cache 层级切矩阵乘)")}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-ch09-structured-codegen-pipeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
