#!/usr/bin/env python3
"""fig-mxfp4-before-after: before-after 模板。
把 indexer QK 路径 FP8→MXFP4 的局部量化优化,换算成 top-k 选择器加速/召回与整机 FLOPs/KV cache
收益。同构双面板(FP8 / MXFP4),仅字节布局与精度标签处高亮,底部三个统计卡片给整机收益。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text(x, y, s, size=13, anchor="middle", weight="normal", fill="#0f172a"):
    fw = f' font-weight="{weight}"' if weight != "normal" else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" '
            f'font-size="{size}"{fw} fill="{fill}">{esc(s)}</text>')

def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

W = 1180
PAD = 40
TOP = 92

RED_FILL, RED_STROKE = "#fee2e2", "#dc2626"
GREEN_FILL, GREEN_STROKE = "#dcfce7", "#16a34a"
GRAY_FILL, GRAY_STROKE = "#e2e8f0", "#64748b"
AMBER_FILL, AMBER_STROKE = "#fef3c7", "#d97706"

PANEL_W = (W - 2 * PAD - 60) / 2
PANEL_X = [PAD, PAD + PANEL_W + 60]
y_panel = TOP
h_panel = 220
y_stats = y_panel + h_panel + 56
h_stats = 118
H = y_stats + h_stats + 50

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(text(W / 2, 32, "MXFP4 indexer 缓存:局部量化换整机收益", size=16.5, weight="bold"))
L.append(text(W / 2, 54,
              "重绘自 arXiv:2606.19348 Figure 1(右)与 §5.2.1:QK 路径量化 + QAT 换来 top-k 提速/高召回/整机降本",
              size=11.5, fill="#475569"))

# ---- 面板 0: FP8(优化前) ----
p0x = PANEL_X[0]
L.append(box(p0x, y_panel, PANEL_W, h_panel, RED_FILL, RED_STROKE))
L.append(text(p0x + PANEL_W / 2, y_panel + 28, "优化前:FP8 indexer QK 路径", size=13.5, weight="bold", fill="#991b1b"))

bar_x = p0x + 30
bar_w = PANEL_W - 60
bar_y = y_panel + 50
L.append(box(bar_x, bar_y, bar_w * 0.85, 40, "#fecaca", RED_STROKE, rx=4, sw=1.2))
L.append(text(bar_x + bar_w * 0.85 / 2, bar_y + 25, "head_dim 字节值", size=11, fill="#7f1d1d"))
L.append(box(bar_x + bar_w * 0.85, bar_y, bar_w * 0.15, 40, "#fde68a", "#d97706", rx=4, sw=1.2))
L.append(text(bar_x + bar_w * 0.85 + bar_w * 0.15 / 2, bar_y + 25, "4B", size=9.5, fill="#92400e"))
L.append(text(p0x + PANEL_W / 2, bar_y + 60, "FP8 值 + 4 字节 fp32 scale", size=10.8, fill="#334155"))

L.append(box(p0x + 30, bar_y + 92, PANEL_W - 60, 40, GRAY_FILL, GRAY_STROKE, rx=6, sw=1.2))
L.append(text(p0x + PANEL_W / 2, bar_y + 116, "index 分数精度:FP32", size=12, weight="bold", fill="#334155"))

L.append(text(p0x + PANEL_W / 2, y_panel + h_panel - 14, "top-k 选择器基准速度、基准召回", size=10.3, fill="#7f1d1d"))

# ---- 面板 1: MXFP4(优化后) ----
p1x = PANEL_X[1]
L.append(box(p1x, y_panel, PANEL_W, h_panel, GREEN_FILL, GREEN_STROKE))
L.append(text(p1x + PANEL_W / 2, y_panel + 28, "优化后:MXFP4 indexer QK 路径", size=13.5, weight="bold", fill="#166534"))

bar_x1 = p1x + 30
L.append(box(bar_x1, bar_y, bar_w * 0.6, 40, "#bbf7d0", GREEN_STROKE, rx=4, sw=1.2))
L.append(text(bar_x1 + bar_w * 0.6 / 2, bar_y + 25, "head_dim/2 字节(2 值/字节)", size=10, fill="#14532d"))
L.append(box(bar_x1 + bar_w * 0.6, bar_y, bar_w * 0.4, 40, "#fde68a", "#d97706", rx=4, sw=1.2))
L.append(text(bar_x1 + bar_w * 0.6 + bar_w * 0.4 / 2, bar_y + 25, "head_dim/32 字节 ue8m0", size=9, fill="#92400e"))
L.append(text(p1x + PANEL_W / 2, bar_y + 60, "MXFP4 值(2 值/字节) + 每 32 值一个 ue8m0 scale", size=10.5, fill="#334155"))

L.append(box(p1x + 30, bar_y + 92, PANEL_W - 60, 40, "#bbf7d0", GREEN_STROKE, rx=6, sw=1.2))
L.append(text(p1x + PANEL_W / 2, bar_y + 116, "index 分数精度:FP32 → BF16", size=12, weight="bold", fill="#166534"))

L.append(text(p1x + PANEL_W / 2, y_panel + h_panel - 14, "配合 QAT(量化感知训练)保精度", size=10.3, fill="#166534"))

# 中间横向箭头
mid_y = y_panel + h_panel / 2
L.append(f'<line x1="{p0x+PANEL_W+8}" y1="{mid_y}" x2="{p1x-8}" y2="{mid_y}" '
         f'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')

# ---- 统计卡片(整机收益) ----
stat_w = (W - 2 * PAD - 2 * 30) / 3
stats = [
    ("top-k 选择器加速", "2×", "arXiv:2606.19348 §5.2.1"),
    ("KV 条目召回率", "99.7%", "arXiv:2606.19348 §5.2.1"),
    ("V4-Pro 百万 token(对 V3.2)", "27% FLOPs / 10% KV cache", "arXiv:2606.19348 Figure 1(右)"),
]
for i, (label, val, src) in enumerate(stats):
    sx = PAD + i * (stat_w + 30)
    L.append(box(sx, y_stats, stat_w, h_stats, AMBER_FILL, AMBER_STROKE))
    L.append(text(sx + stat_w / 2, y_stats + 28, label, size=12, weight="bold", fill="#92400e"))
    L.append(text(sx + stat_w / 2, y_stats + 66, val, size=19, weight="bold", fill="#7c2d12"))
    L.append(text(sx + stat_w / 2, y_stats + 96, src, size=9.5, fill="#a16207"))

L.append(text(W / 2, y_stats + h_stats + 24,
              "开关:attention_config.use_fp4_indexer_cache;builder 断言仅 Blackwell 数据中心 GPU(sm_10x)可用",
              size=10.5, fill="#64748b"))

L.append('</svg>')
out = Path(__file__).with_name("fig-mxfp4-before-after.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
