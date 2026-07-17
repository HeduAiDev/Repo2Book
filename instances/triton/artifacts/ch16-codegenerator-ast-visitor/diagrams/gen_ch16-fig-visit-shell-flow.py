#!/usr/bin/env python3
"""flow 模板(定制,仿 ch14 driver-loop 主干+分支写法):visit 外壳如何把每个 AST 节点
先钉 MLIR loc、再按类型名分派、异常统一包成 CompilationError。
主干:node -> visit 外壳(set_loc) -> super().visit 分派 -> 分两支(已实现/未实现);
已实现分支再分两支(正常退出 vs 抛异常)。
改造点:MAIN(主干节点)、SPLIT1(已实现/未实现两支)、SPLIT2(正常/异常两支)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def multiline(lines, cx, y0, size=12, weight=False, fill="#0f172a", lh=15, anchor="middle"):
    out = []
    wattr = 'font-weight="bold" ' if weight else ''
    for k, line in enumerate(lines):
        out.append(f'<text x="{cx}" y="{y0 + k * lh}" text-anchor="{anchor}" '
                    f'font-family="sans-serif" font-size="{size}" {wattr}'
                    f'fill="{fill}">{esc(line)}</text>')
    return out


BOX_W, BOX_H, VGAP = 480, 62, 40
PAD_L, TOP = 60, 80

MAIN = [
    ("①", ["AST 节点 node"], None, "#e2e8f0", "#334155"),
    ("②", ["visit 外壳:set_loc(...)"],
     "set_loc(file_name, begin_line + node.lineno, col_offset)  — code_generator.py:L1197", "#dbeafe", "#1d4ed8"),
    ("③", ["super().visit(node)"],
     "按 type(node).__name__ 分派到 visit_<Type>", "#dbeafe", "#1d4ed8"),
]

# 分支 1:已实现 / 未实现
BR1_W, BR1_H = 300, 76
BR1 = [
    ("已实现", ["visit_FunctionDef / visit_Call /", "visit_Assign / visit_BinOp …"], "#dcfce7", "#15803d"),
    ("未实现", ["generic_visit(node)"], "#fee2e2", "#b91c1c"),
]

# 分支 1a 之下再分:正常退出 / 抛异常(仅"已实现"支下挂)
LEAF_W, LEAF_H = 300, 66
LEAF_NORMAL = ("恢复上一个 loc", ["visit_<Type> 正常返回", "退出前把 loc 还原成上一层"], "#dcfce7", "#15803d")
LEAF_EXC = ("CompilationError", ["非 CompilationError 异常", "raise CompilationError(jit_fn.src, node) from None"], "#fee2e2", "#b91c1c")
LEAF_UNSUP = ("UnsupportedLanguageConstruct", ["generic_visit 兜底", "raise UnsupportedLanguageConstruct(未实现节点)"], "#fee2e2", "#b91c1c")

n_main = len(MAIN)
main_block_h = n_main * (BOX_H + VGAP)
br1_top = TOP + main_block_h
br1_gap_x = 90
br1_total_w = 2 * BR1_W + br1_gap_x
lane_cx = br1_total_w / 2 + PAD_L

leaf_top = br1_top + BR1_H + 70
leaf_gap_x = 60

w = PAD_L * 2 + max(lane_cx + BR1_W + br1_gap_x / 2, 3 * LEAF_W + 2 * leaf_gap_x + 40)
h = leaf_top + LEAF_H + 140

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
          '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
          '</defs>')
L.append(f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>')
L.append(f'<text x="{PAD_L}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("visit 外壳:先钉 MLIR loc,再按类型分派,异常统一包成 CompilationError")}</text>')
L.append(f'<text x="{PAD_L}" y="56" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("白名单式:没有对应 visit_<Type> 的语法直接报错,而非静默产错")}</text>')

# 主干三节点
y = TOP
main_centers = []
for badge, title_lines, detail, fill, stroke in MAIN:
    cx = lane_cx
    main_centers.append(y)
    L.append(f'<rect x="{cx - BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<circle cx="{cx - BOX_W/2 + 22}" cy="{y + BOX_H/2}" r="15" fill="{stroke}"/>')
    L.append(f'<text x="{cx - BOX_W/2 + 22}" y="{y + BOX_H/2 + 5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" fill="white">{esc(badge)}</text>')
    title_y = y + (BOX_H/2 + 5 if not detail else BOX_H/2 - 6)
    L += multiline(title_lines, cx + 14, title_y, size=13.5, weight=True)
    if detail:
        L.append(f'<text x="{cx}" y="{y + BOX_H - 9}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10" fill="#334155">{esc(detail)}</text>')
    y += BOX_H + VGAP

# 主干箭头(相邻节点相连)
for i in range(n_main - 1):
    y1 = main_centers[i] + BOX_H
    y2 = main_centers[i + 1]
    L.append(f'<line x1="{lane_cx}" y1="{y1}" x2="{lane_cx}" y2="{y2 - 4}" '
              'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# 分支 1:已实现(左) / 未实现(右) —— 共用一段主干竖线,再各自水平分岔
br1_y = main_centers[-1] + BOX_H + VGAP
br1_x = [lane_cx - br1_total_w/2, lane_cx - br1_total_w/2 + BR1_W + br1_gap_x]
labels1 = ["分派命中已实现方法", "分派落到 generic_visit"]
fork_y = main_centers[-1] + BOX_H + VGAP / 2
L.append(f'<line x1="{lane_cx}" y1="{main_centers[-1] + BOX_H}" x2="{lane_cx}" y2="{fork_y}" '
          'stroke="#334155" stroke-width="1.6"/>')
for i, ((name, lines, fill, stroke), bx) in enumerate(zip(BR1, br1_x)):
    cx = bx + BR1_W / 2
    L.append(f'<path d="M {lane_cx},{fork_y} L {cx},{fork_y} '
              f'L {cx},{br1_y - 4}" fill="none" stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
    mid_x = (lane_cx + cx) / 2
    L.append(f'<text x="{mid_x}" y="{fork_y - 8}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" font-weight="bold" fill="#334155">{esc(labels1[i])}</text>')
    L.append(f'<rect x="{bx}" y="{br1_y}" width="{BR1_W}" height="{BR1_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{cx}" y="{br1_y + 20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="{stroke}">{esc(name)}</text>')
    L += multiline(lines, cx, br1_y + 38, size=10.5, fill="#334155")

# 分支 1a(已实现)之下再分:正常(左) / 异常(中)
impl_cx = br1_x[0] + BR1_W / 2
unsup_cx = br1_x[1] + BR1_W / 2
leaf_x = [PAD_L + LEAF_W/2, PAD_L + LEAF_W/2 + LEAF_W + leaf_gap_x, PAD_L + LEAF_W/2 + 2*(LEAF_W + leaf_gap_x)]

leaf_defs = [
    (leaf_x[0], LEAF_NORMAL, impl_cx, "g", "正常返回"),
    (leaf_x[1], LEAF_EXC, impl_cx, "r", "抛非 CompilationError 异常"),
    (leaf_x[2], LEAF_UNSUP, unsup_cx, "r", "generic_visit 兜底"),
]

for lx, (name, lines, fill, stroke), src_cx, marker, edge_label in leaf_defs:
    L.append(f'<path d="M {src_cx},{br1_y + BR1_H} L {src_cx},{leaf_top - 24} L {lx},{leaf_top - 24} '
              f'L {lx},{leaf_top - 4}" fill="none" stroke="{stroke}" stroke-width="1.6" '
              f'marker-end="url(#{marker})"/>')
    L.append(f'<text x="{lx}" y="{leaf_top - 30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="{stroke}">{esc(edge_label)}</text>')
    L.append(f'<rect x="{lx - LEAF_W/2}" y="{leaf_top}" width="{LEAF_W}" height="{LEAF_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{lx}" y="{leaf_top + 22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{stroke}">{esc(name)}</text>')
    L += multiline(lines, lx, leaf_top + 40, size=10, fill="#334155")

foot_y = leaf_top + LEAF_H + 34
L.append(f'<text x="{PAD_L}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("begin_line = 源码起始行 - 1(node.lineno 从 1 起,L200);每个节点先钉位置,报错才能精确指到 kernel 源码那一行")}</text>')
L.append(f'<text x="{PAD_L}" y="{foot_y + 20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("绿=正常路径;红=报错路径(异常包装 / 未实现语法);蓝=主干工序")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("ch16-fig-visit-shell-flow.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w:.0f}x{h:.0f}")
