#!/usr/bin/env python3
"""flow 模板改造:AffinityDAG 把 IR 建成二部数据流图——方块=OpNode,圆=ValueNode。
matmul+bias epilogue kernel G:16 节点(7 OpNode+9 ValueNode)。正向数据边(灰实线)+
absorb 反向回吸方向标注(橙色虚线,从 dot 指回 %a/load_a)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
EDGE = "#94a3b8"
ABSORB = "#ea580c"
OPFILL = "#e2e8f0"
OPSTROKE = "#334155"
VALFILL = "#eff6ff"
VALSTROKE = "#3b82f6"

W, H = 1200, 620
PAD = 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{EDGE}"/></marker>'
     '<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{ABSORB}"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="18" font-weight="bold" '
     f'fill="{INK}">{esc("AffinityDAG:matmul+bias kernel 的二部数据流图")}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="13" fill="{GRAY}">'
     f'{esc("方块=OpNode(7 个)、圆=ValueNode(9 个),共 16 节点;absorb 沿 ValueNode.outputs 反向回吸(橙色)")}</text>']


def op_box(cx, cy, label, w=96, h=44):
    L.append(f'<rect x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" rx="8" '
              f'fill="{OPFILL}" stroke="{OPSTROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="{INK}">{esc(label)}</text>')
    return (cx, cy, w, h)


def val_node(cx, cy, label, r=26):
    L.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{VALFILL}" stroke="{VALSTROKE}" '
              'stroke-width="1.5"/>')
    L.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="{VALSTROKE}">{esc(label)}</text>')
    return (cx, cy, r)


def edge(x1, y1, x2, y2, marker="a", color=EDGE, dash=None, curve=None):
    dasharray = f' stroke-dasharray="{dash}"' if dash else ''
    if curve:
        L.append(f'<path d="M{x1},{y1} Q{curve[0]},{curve[1]} {x2},{y2}" fill="none" '
                  f'stroke="{color}" stroke-width="1.8"{dasharray} marker-end="url(#{marker})"/>')
    else:
        L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                  f'stroke-width="1.8"{dasharray} marker-end="url(#{marker})"/>')


TOP = 130
# --- matmul operand chains (top: A-path, bottom: B-path) ---
row_a_y = TOP
row_b_y = TOP + 160
x_pa, x_load, x_val, x_dot = 130, 300, 460, 620

val_node(x_pa, row_a_y, "%pa")
op_box(x_load, row_a_y, "load_a")
val_node(x_val, row_a_y, "%a")
val_node(x_pa, row_b_y, "%pb")
op_box(x_load, row_b_y, "load_b")
val_node(x_val, row_b_y, "%b")
dot_y = (row_a_y + row_b_y) / 2
op_box(x_dot, dot_y, "dot", w=90, h=48)

edge(x_pa + 26, row_a_y, x_load - 48, row_a_y)
edge(x_load + 48, row_a_y, x_val - 26, row_a_y)
edge(x_pa + 26, row_b_y, x_load - 48, row_b_y)
edge(x_load + 48, row_b_y, x_val - 26, row_b_y)
edge(x_val + 26, row_a_y, x_dot - 45, dot_y + 6, curve=((x_val + x_dot) / 2, row_a_y))
edge(x_val + 26, row_b_y, x_dot - 45, dot_y - 6, curve=((x_val + x_dot) / 2, row_b_y))

# --- epilogue: dot -> %c -> addf <- %bias -> %d -> store <- %po ---
x_c, x_addf, x_bias_y = 760, 900, TOP - 60
val_node(x_c, dot_y, "%c")
op_box(x_addf, dot_y, "addf")
val_node(x_addf, x_bias_y, "%bias")

edge(x_dot + 45, dot_y, x_c - 26, dot_y)
edge(x_c + 26, dot_y, x_addf - 48, dot_y)
edge(x_addf, x_bias_y + 26, x_addf, dot_y - 22)

x_d2 = 1030
val_node(x_d2, dot_y, "%d")
edge(x_addf + 48, dot_y, x_d2 - 26, dot_y)
x_store2 = 1150
x_po_y2 = dot_y + 130
op_box(x_store2, dot_y, "store", w=90, h=48)
edge(x_d2 + 26, dot_y, x_store2 - 45, dot_y)
val_node(x_store2, x_po_y2, "%po")
edge(x_store2, x_po_y2 - 26, x_store2, dot_y + 22)

# --- isolated component: const -> %z ; return (disconnected, same residual group) ---
iso_y = row_b_y + 150
x_const, x_z, x_return = 130, 300, 460
op_box(x_const, iso_y, "const")
val_node(x_z, iso_y, "%z")
op_box(x_return, iso_y, "return")
edge(x_const + 48, iso_y, x_z - 26, iso_y)
L.append(f'<text x="{(x_z+x_return)/2}" y="{iso_y-18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="{GRAY}">{esc("无边(不消费 %z)")}</text>')
L.append(f'<text x="{x_const}" y="{iso_y+50}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="{GRAY}">{esc("孤立分量:const/%z/return")}</text>')

# --- absorb reverse-direction annotation: dot -> %a -> load_a (orange dashed) ---
ay = row_a_y - 34
edge(x_dot - 20, dot_y - 30, x_val + 8, row_a_y + 34, marker="ab", color=ABSORB, dash="7,4",
     curve=((x_dot + x_val) / 2, dot_y - 60))
edge(x_val - 20, row_a_y - 34, x_load + 8, row_a_y - 34, marker="ab", color=ABSORB, dash="7,4")
L.append(f'<text x="{(x_val+x_load)/2}" y="{row_a_y-44}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="{ABSORB}">'
          f'{esc("absorb 回吸方向(沿 outputs 反向)")}</text>')

foot_y = H - 24
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("OpNode 构造边(operand/result)见 DAG.cpp:L184-281;absorb 求核沿 ValueNode.outputs(消费者)反向回吸,约束从下游 dot 流回上游 load")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch16-node-graph.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
