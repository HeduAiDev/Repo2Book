#!/usr/bin/env python3
"""layout 变体：三层继承关系——上游 4 个 MLIR 接口 -> HFusionStructuredBase_Op ->
5 个显式结构化 op + 9 个 yaml named op。展示"HFusion 是 Linalg 超集"在代码层
是接口继承，而非口号。全坐标计算，零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "HFusionStructuredBase_Op 继承上游接口——Linalg 超集的代码证据"
SUBTITLE = "块终结符复用 linalg.yield；reifyResultShapes 直接 cast 成 linalg::LinalgOp（HFusionStructuredOps.td:L38-L57）"

UPSTREAM_IFACES = ["LinalgStructuredInterface", "DestinationStyleOpInterface",
                   "ReifyRankedShapedTypeOpInterface", "MemoryEffectsOpInterface"]
EXPLICIT_OPS = ["reduce_with_index", "arange", "gather", "gather_mask", "conv1d"]
YAML_OPS = ["load", "store", "elemwise_unary", "elemwise_binary", "compare",
            "select", "cast", "bitcast", "group_matmul"]

PAD, TOP = 40, 96
IF_W, IF_H, IF_GAP = 260, 44, 16
CENTER_W, CENTER_H = 460, 78
CHIP_W, CHIP_H, CHIP_GAP = 118, 34, 10
ARROW_GAP = 46

n_if = len(UPSTREAM_IFACES)
row_w_if = n_if * IF_W + (n_if - 1) * IF_GAP
cluster_gap = 60
explicit_w = 5 * CHIP_W + 4 * CHIP_GAP
yaml_w = 5 * CHIP_W + 4 * CHIP_GAP
total_cluster_w = explicit_w + cluster_gap + yaml_w + 24  # +24 = 簇虚线框左右各 12 内边距
content_w = max(row_w_if, CENTER_W, total_cluster_w)
w = content_w + PAD * 2

if_y = TOP
center_y = if_y + IF_H + ARROW_GAP
cluster_y = center_y + CENTER_H + ARROW_GAP
# 显式 op：一行 5 个；yaml op：两行 5+4
YAML_ROW1, YAML_ROW2 = YAML_OPS[:5], YAML_OPS[5:]
cluster_label_h = 22
explicit_h = cluster_label_h + CHIP_H + 14
yaml_h = cluster_label_h + CHIP_H * 2 + CHIP_GAP + 14
h = cluster_y + max(explicit_h, yaml_h) + PAD + 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 第一层：上游 4 个 MLIR 接口（灰蓝）——标题放row上方，避免侧标签溢出画布
if_x0 = (w - row_w_if) / 2
if_label_y = if_y - 10
L.append(f'<text x="{if_x0}" y="{if_label_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">upstream MLIR 接口（4 个，被下方基类实现）</text>')
if_centers = []
for i, name in enumerate(UPSTREAM_IFACES):
    x = if_x0 + i * (IF_W + IF_GAP)
    if_centers.append(x + IF_W / 2)
    L.append(f'<rect x="{x}" y="{if_y}" width="{IF_W}" height="{IF_H}" rx="6" '
              'fill="#e2e8f0" stroke="#475569" stroke-width="1.5"/>')
    L.append(f'<text x="{x+IF_W/2}" y="{if_y+IF_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
              f'fill="#1e293b">{esc(name)}</text>')

# 中间层：HFusionStructuredBase_Op（中心框）
center_x = (w - CENTER_W) / 2
L.append(f'<rect x="{center_x}" y="{center_y}" width="{CENTER_W}" height="{CENTER_H}" rx="8" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.5"/>')
L.append(f'<text x="{w/2}" y="{center_y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#1e3a8a">HFusionStructuredBase_Op</text>')
L.append(f'<text x="{w/2}" y="{center_y+46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#1e40af">终结符复用 mlir::linalg::YieldOp</text>')
L.append(f'<text x="{w/2}" y="{center_y+62}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#1e40af">reifyResultShapes 直接 cast 成 linalg::LinalgOp</text>')

# 箭头：4 接口 -> 中心（每个接口正下方汇入中心框顶部对应位置）
for cx in if_centers:
    x_end = center_x + (cx - center_x) if center_x <= cx <= center_x + CENTER_W else w / 2
    x_end = min(max(cx, center_x + 10), center_x + CENTER_W - 10)
    L.append(f'<line x1="{cx}" y1="{if_y+IF_H}" x2="{x_end}" y2="{center_y}" '
              'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)" opacity="0.75"/>')
L.append(f'<text x="{w/2}" y="{if_y+IF_H+ARROW_GAP/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#64748b">实现（implements）</text>')

# 下层两个簇：左=5 个显式结构化 op，右=9 个 yaml named op（宽度已在 w 计算时统一定义）
lx0 = (w - (explicit_w + cluster_gap + yaml_w)) / 2
rx0 = lx0 + explicit_w + cluster_gap

# 簇边框
def cluster_box(x0, width, height, label, n, fill, stroke):
    L.append(f'<rect x="{x0-12}" y="{cluster_y-6}" width="{width+24}" height="{height+16}" '
              f'rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5" stroke-dasharray="5,3"/>')
    L.append(f'<text x="{x0}" y="{cluster_y+14}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="{stroke}">{esc(label)}</text>')

cluster_box(lx0, explicit_w, explicit_h, f"{len(EXPLICIT_OPS)} 个显式结构化 op（.td 直接 def）",
            len(EXPLICIT_OPS), "#fef9c3", "#a16207")
cluster_box(rx0, yaml_w, yaml_h, f"{len(YAML_OPS)} 个 yaml named op（HFusionNamedStructuredOps.yaml）",
            len(YAML_OPS), "#fce7f3", "#be185d")

chip_top = cluster_y + cluster_label_h
for i, name in enumerate(EXPLICIT_OPS):
    x = lx0 + i * (CHIP_W + CHIP_GAP)
    L.append(f'<rect x="{x}" y="{chip_top}" width="{CHIP_W}" height="{CHIP_H}" rx="6" '
              'fill="#fffbeb" stroke="#b45309" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CHIP_W/2}" y="{chip_top+CHIP_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#78350f">{esc(name)}</text>')

for i, name in enumerate(YAML_ROW1):
    x = rx0 + i * (CHIP_W + CHIP_GAP)
    L.append(f'<rect x="{x}" y="{chip_top}" width="{CHIP_W}" height="{CHIP_H}" rx="6" '
              'fill="#fdf2f8" stroke="#be185d" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CHIP_W/2}" y="{chip_top+CHIP_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#831843">{esc(name)}</text>')
row2_top = chip_top + CHIP_H + CHIP_GAP
for i, name in enumerate(YAML_ROW2):
    x = rx0 + i * (CHIP_W + CHIP_GAP)
    L.append(f'<rect x="{x}" y="{row2_top}" width="{CHIP_W}" height="{CHIP_H}" rx="6" '
              'fill="#fdf2f8" stroke="#be185d" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CHIP_W/2}" y="{row2_top+CHIP_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#831843">{esc(name)}</text>')

# 箭头：中心 -> 两簇
for tx in [lx0 + explicit_w / 2, rx0 + yaml_w / 2]:
    L.append(f'<line x1="{w/2}" y1="{center_y+CENTER_H}" x2="{tx}" y2="{cluster_y-6}" '
              'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)" opacity="0.75"/>')
L.append(f'<text x="{w/2}" y="{center_y+CENTER_H+ARROW_GAP/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#64748b">继承（inherits）</text>')

foot_y = h - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">口径注：计数仅含 HFusion 自身 .td/.yaml 实定义的 op，不含继承自 Linalg 的算子（架构文档不给此项计数）</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch21-m3-structured-superset.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
