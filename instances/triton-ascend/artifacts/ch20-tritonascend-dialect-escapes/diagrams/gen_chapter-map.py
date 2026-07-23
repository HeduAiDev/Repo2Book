#!/usr/bin/env python3
"""chapter-map for ch20《TritonAscend 方言与三条逃生舱》——kind=deep 源码剖面图。

主线：ascend 方言（共享容器,11 个 op）→ 管线挂载点（compiler.py:L148-157,序
hivm→hfusion→llvm）→ 三条逃生舱各自转换（TritonToHIVM/HFusion/LLVM.cpp,并行,
无先后依赖，仅"同挂在管线里"）→ 汇合到驱动器之别+小结。

三舱节点按【管线真实挂载序】从左到右排列（hivm, hfusion, llvm）——不是按小节
讲解顺序（正文里是 §四 HFusion→§五 HIVM→§六 LLVM,从简到繁），这个物理顺序与
讲解顺序的差异，正是正文第三节明说的一点（"下面三节的讨论顺序，不等于三舱在
管线里的挂载顺序"）。为避免"画反"管线序，此图物理布局忠实挂载序,lane 标签
里也直接写明；§ 徽标仍标各节真实编号，供读者对照正文小节，不代表物理阅读顺序。

■ 不可变（同 example-chapter-map.py 约定）：§徽标胶囊样式/入口绿/出口橙/主线
  蓝/路线条实线蓝-虚线灰/cjk_text_width() 宽度估算。
■ 可变：LANES/NODES/EDGES/ROUTES/FOOTNOTE。

六项自查记录见 figure-manifest.json。

用法：python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变) ----------------
LANES = [
    "方言定义层（TritonAscendOps.td）",
    "管线挂载与驱动层（compiler.py）",
    "转换舱实装层（按管线挂载序排列：hivm → hfusion → llvm）",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("entry",    0, 0, 0, "TritonAscendOps.td",     "共享容器：11 个 NPU 专用 op",      "§一–二"),
    ("position", 1, 1, 0, "compiler.py:L148-157",   "挂载序 hivm→hfusion→llvm",         "§三"),
    ("hivm",     2, 2, 0, "TritonToHIVM.cpp",       "ascend.custom→双核同步翻转",       "§五"),
    ("hfusion",  2, 3, 0, "TritonToHFusion.cpp",    "3 检查员贪婪放行（3 pattern）",    "§四"),
    ("llvm",     2, 4, 0, "TritonToLLVM.cpp",       "内联汇编直落，按 32 位装箱",        "§六"),
    ("exit",     1, 5, 0, "驱动器之别 + 小结",       "贪婪 vs partial conversion",        "§七–八"),
]
EDGES = [  # (src_id, dst_id) —— 挂载/数据流向边，统一主线蓝
    ("entry", "position"),
    ("position", "hivm"), ("position", "hfusion"), ("position", "llvm"),
    ("hivm", "exit"), ("hfusion", "exit"), ("llvm", "exit"),
]
# (路线名, [(列, §编号), ...], 是否高亮)
ROUTES = [
    ("全通道（精读 8 节）",
     [(0, "§一–二"), (1, "§三"), (2, "§五"), (3, "§四"), (4, "§六"), (5, "§七–八")], True),
    ("速览（跳过方言定义）",
     [(1, "§三"), (2, "§五"), (3, "§四"), (4, "§六"), (5, "§七–八")], False),
]
LEGEND = [("#22c55e", "入口：主链上游 IR（已发射方言 op）"),
          ("#3b82f6", "章内结构 / 挂载关系"),
          ("#f97316", "出口：交主链 TritonToLinalg 收官")]
TITLE = "第 20 章 · TritonAscend 方言与三条逃生舱（源码剖面：挂载序 hivm→hfusion→llvm）"
FOOTNOTE = [
    "三舱合计 5 个 pattern（HFusion 3 + HIVM 1 + LLVM 1），相较主链大量结构化 pattern 是极小子集；",
    "ascend 方言共 11 个 op，三舱只消费其中 2 个（ascend.mod → HFusion 舱，ascend.custom → HIVM 舱），其余 9 个走主链其它 pass 或另有归属。",
    "图中三舱汇入「驱动器之别 + 小结」为叙事汇总、无因果·仅示意——三舱互不调用，各自独立处理不同 op 集合，无先后依赖。",
]

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 172, 78
COL_GAP, ROW_GAP = 34, 20
EDGE_MARGIN, STUB_W, STUB_H = 14, 66, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 28
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
FOOT_LINE_H, FOOT_GAP = 16, 10
BADGE_W, BADGE_H = 54, 20

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
routes_h = ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H
foot_h = FOOT_GAP + len(FOOTNOTE) * FOOT_LINE_H
h = routes_top + routes_h + foot_h + BOTTOM_PAD


def badge(cx, cy, text):
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10.5" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 26

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("主链上游")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("交主链")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用/挂载边(主线蓝),多条边汇入同一节点时终点 y 错开
_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 10, y, sec)

# 底部阅读路线
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, sec in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, sec)

# 底部数字小结(footnote,数字均见正文/dossier:11 op/2 消费/5 pattern=3+1+1)
foot_top = routes_top + routes_h + FOOT_GAP
for i, line in enumerate(FOOTNOTE):
    L.append(f'<text x="16" y="{foot_top + i * FOOT_LINE_H:.1f}" font-family="sans-serif" '
              f'font-size="10.5" fill="#0f172a">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size={w}x{h}, ratio={w / h:.2f}")
