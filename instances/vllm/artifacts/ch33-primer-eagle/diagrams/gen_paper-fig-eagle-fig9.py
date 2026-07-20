#!/usr/bin/env python3
"""paper-fig-eagle-fig9: 论文精髓图重绘。
重绘自 arXiv:2401.15077 Fig.9（Appendix A.1 Tree Structure）——EAGLE 真实使用的草稿树
结构（左，27 个节点、深度 5、逐层分支数不均）与去掉树注意力后对应的链式结构（右，6 节点直链）
并排对比。原图是 graphviz 风格的斜向级联布局；本图改用标准自顶向下 tidy-tree 布局重绘
（信息结构——每层真实分支数、总节点数、深度——与原图对齐，不做像素级复制）。
树的分支结构来自逐像素读原图统计(illustrator 报告);全部坐标由递归布局函数计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# 树结构:嵌套 dict,{} 表示叶子。分支数与深度取自原图 x9.png 逐像素统计。
TREE = {
    "N1": {
        "N1a": {
            "N1a1": {
                "N1a1a": {"L1": {}, "L2": {}},
                "L3": {}, "L4": {},
            },
            "L5": {}, "L6": {},
        },
        "N1b": {"L7": {}, "L8": {}},
        "N1c": {"L9": {}, "L10": {}},
    },
    "N2": {
        "A": {"L11": {}},
        "B": {},
    },
    "N3": {"L12": {}, "L13": {}},
    "N4": {
        "N4a": {"L14": {}},
    },
}

SLOT = 56
ROW_H = 92
R = 15  # 节点半径

def layout(node, depth, next_x, out_nodes, out_edges, parent_id, my_id):
    if not node:
        x = next_x[0]
        next_x[0] += SLOT
        out_nodes[my_id] = (x, depth)
        if parent_id is not None:
            out_edges.append((parent_id, my_id))
        return x
    xs_children = []
    for name, child in node.items():
        cid = f"{my_id}.{name}"
        cx = layout(child, depth + 1, next_x, out_nodes, out_edges, my_id, cid)
        xs_children.append(cx)
    x = sum(xs_children) / len(xs_children)
    out_nodes[my_id] = (x, depth)
    if parent_id is not None:
        out_edges.append((parent_id, my_id))
    return x

nodes, edges = {}, []
next_x = [0]
layout(TREE, 0, next_x, nodes, edges, None, "root")
tree_width = next_x[0] - SLOT
tree_depth = max(d for _, d in nodes.values())
n_total = len(nodes)

# 链:query + 5 节点直链(与树同深度,depth 0..5)
CHAIN_LEN = 6

PAD, TOP = 40, 150
TREE_W = tree_width + 2 * R + 20
GAP = 90
CHAIN_W = 140
W = PAD * 2 + TREE_W + GAP + CHAIN_W
H = TOP + (tree_depth + 1) * ROW_H + 90

def nx_(nid):
    x, d = nodes[nid]
    return PAD + x + R + 10
def ny_(nid):
    x, d = nodes[nid]
    return TOP + d * ROW_H

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="40" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc("EAGLE 真实草稿树 vs 去掉树注意力的链式结构")}</text>',
     f'<text x="{W/2}" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("重绘自 arXiv:2401.15077 Fig.9：真实草稿树 " + str(n_total) + " 个节点、深度 " + str(tree_depth) + "，分支数逐层不均；同深度链只有 " + str(CHAIN_LEN) + " 个节点")}</text>']

# 左：树
tree_cx = PAD + TREE_W / 2
L.append(f'<text x="{tree_cx}" y="{TOP-50}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#1e40af">{esc("左：论文 Fig.9 真实草稿树（树注意力）")}</text>')
for pid, cid in edges:
    x1, y1 = nx_(pid), ny_(pid)
    x2, y2 = nx_(cid), ny_(cid)
    L.append(f'<line x1="{x1:.1f}" y1="{y1+R:.1f}" x2="{x2:.1f}" y2="{y2-R:.1f}" '
              'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')
for nid, (x, d) in nodes.items():
    cx, cy = nx_(nid), ny_(nid)
    is_root = nid == "root"
    fill = "#3b82f6" if is_root else "#eef2ff"
    stroke = "#1e3a5f" if is_root else "#6366f1"
    rr = R + 6 if is_root else R
    L.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{rr}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    if is_root:
        L.append(f'<text x="{cx:.1f}" y="{cy+3:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="8.5" font-weight="bold" fill="white">{esc("query")}</text>')

# 右：链
chain_x0 = PAD + TREE_W + GAP
chain_cx = chain_x0 + CHAIN_W / 2
L.append(f'<text x="{chain_cx}" y="{TOP-50}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#92400e">{esc("右：无树注意力的链式结构")}</text>')
chain_ys = [TOP + i * ROW_H for i in range(CHAIN_LEN)]
for i in range(CHAIN_LEN - 1):
    y1, y2 = chain_ys[i] + R, chain_ys[i + 1] - R
    L.append(f'<line x1="{chain_cx}" y1="{y1}" x2="{chain_cx}" y2="{y2}" '
              'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')
for i, cy in enumerate(chain_ys):
    is_root = i == 0
    fill = "#f59e0b" if is_root else "#fef3c7"
    stroke = "#92400e" if is_root else "#d97706"
    rr = R + 6 if is_root else R
    L.append(f'<circle cx="{chain_cx}" cy="{cy}" r="{rr}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    if is_root:
        L.append(f'<text x="{chain_cx}" y="{cy+3}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="8.5" font-weight="bold" fill="white">{esc("query")}</text>')

foot_y = TOP + (tree_depth + 1) * ROW_H + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc(f"同一份算力预算：树式一次前向验证 {n_total} 个候选（深度不均，越深分支越少）；链式一次前向只验证 {CHAIN_LEN} 个候选（每层恰 1 个）。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("paper-fig-eagle-fig9.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}  nodes={n_total} depth={tree_depth} width={tree_width}")
