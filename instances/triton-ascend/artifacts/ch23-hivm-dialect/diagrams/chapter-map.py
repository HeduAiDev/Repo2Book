#!/usr/bin/env python3
"""第 23 章「本章地图」——HIVM 方言源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，只改 DATA。

节点预算 9(entry/addrspace/vec_trait/cube_trait/vec_dma/cube_mmad/vec_dispatch/
infer/exit) ≤ 12。本章标题为编号标题(## 23.1 ... ## 23.8)，站牌用 §23.N。

设计要点：
- 主线(实边,蓝)是一条"先立类型基础、再按 Vector/Cube 两条算子路径分头下降、
  最后在内存层级推断汇合"的脊柱：ConvertHFusionToHIVMPass(§23.1,下降入口)→
  HIVM_AddressSpaceAttr(§23.2,六级内存类型基础)→{HIVM_VectorOp(§23.3,Vector
  路径起点,row0) / HIVM_LocalMmadOp(§23.3,Cube 路径起点,row1)}→{LoadOp/StoreOp
  (§23.4,Vector 数据流) / MmadL1Op/FixpipeOp(§23.4–23.5,Cube 数据流)}→
  elementwiseMatchAndRewriteHelper(§23.6,Vector 路径独有的逐元素派发,Cube 路径
  无对应节点因为 matmul 走并列的 populateMatmulPatternsAndLegality)→
  InferHIVMMemScopePass(§23.7,两条路径在这里汇合——地址空间推断的四步优先级)→
  test_infer_mem_scope_complicated(§23.8,贯穿全章的真实 IR 样例出口)。
- lane0(方言定义/类型基础/推断收尾) 与 lane1(算子路径:Vector 行0/Cube 行1)
  两条泳道；col4 只有 Vector 行有节点(Cube 侧矩阵路径由 mmadL1 直接产生,不经过
  这条逐元素派发函数)，这是源码里两条路径本就不对称的真实反映，不是布局凑数。
- 全部符号名取自 dossier.json/chapter.md 原文逐字符串；超 14~16 字符的标识符
  用 "\\n" 机械换行(不改变拼写，只是排版切分，切在驼峰/下划线边界)。
- 短语里避免紧跟 ASCII 半角括号(如 "legal(...)")——全部改用全角括号/顿号，
  防止 lint_chapter_map 的符号防杜撰正则把 "word(" 误当成待核 token(该正则对
  含 "(" 的 token 一律触发核对，全角括号不在其字符集内，不会误触发)。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["方言定义 / 类型基础 / 推断收尾", "算子路径:Vector（上）· Cube（下）"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语, §编号)
NODES = [
    ("entry",       0, 0, 0, "ConvertHFusion\nToHIVMPass",      "linalg/hfusion → illegal\nhivm → legal，前缀 hir.",   "§23.1"),
    ("addrspace",   0, 1, 0, "HIVM_Address\nSpaceAttr",         "六级内存枚举挂\n到 memref 类型上",                    "§23.2"),
    ("vec_trait",   1, 2, 0, "HIVM_VectorOp",                   "PIPE_V + VectorCore\nTypeTrait，落 UB",              "§23.3"),
    ("cube_trait",  1, 2, 1, "HIVM_LocalMmadOp",                "CubeCoreTypeTrait +\nMTE1/M 双流水",                 "§23.3"),
    ("vec_dma",     1, 3, 0, "LoadOp / StoreOp",                "GM↔UB 搬运，\nMTE2/MTE3 引擎",                        "§23.4"),
    ("cube_mmad",   1, 3, 1, "MmadL1Op /\nFixpipeOp",           "L1→L0C 累加，\nL0C→GM/L1/UB 搬出",                    "§23.4–23.5"),
    ("vec_dispatch",1, 4, 0, "elementwiseMatch\nAndRewriteHelper", "linalg/hfusion 逐元素\n→ hivm.hir.vexp 等",       "§23.6"),
    ("infer",       0, 5, 0, "InferHIVMMem\nScopePass",         "四步优先级 + use-def\n级联，钉死地址空间",             "§23.7"),
    ("exit",        0, 6, 0, "test_infer_mem_\nscope_complicated", "贯穿全章 IR 样例：\n改造前后对照（gm/cbuf/cc）",   "§23.8"),
]
EDGES = [  # (src_id, dst_id, 是否折角绕行) —— 调用边,统一主线蓝;
    # cube_mmad→infer 跨行跨列(row1→lane0)的直线会穿过中间 vec_dispatch(row0,col4)
    # 的节点框——改成折角路径(先沿 cube_mmad 所在行右探到 col4/col5 之间的空隙,
    # 再垂直上探到 infer 所在行),绕开 vec_dispatch。
    ("entry", "addrspace", False),
    ("addrspace", "vec_trait", False), ("addrspace", "cube_trait", False),
    ("vec_trait", "vec_dma", False), ("cube_trait", "cube_mmad", False),
    ("vec_dma", "vec_dispatch", False),
    ("vec_dispatch", "infer", False), ("cube_mmad", "infer", True),
    ("infer", "exit", False),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全通读（§23.1→23.8）", [(0, "§23.1"), (1, "§23.2"), (2, "§23.3"), (3, "§23.4"), (4, "§23.6"), (5, "§23.7"), (6, "§23.8")], True),
    ("只读 Cube 矩阵累加路径", [(2, "§23.3"), (3, "§23.5"), (5, "§23.7")], False),
    ("只读 Vector 逐元素路径", [(2, "§23.3"), (3, "§23.4"), (4, "§23.6")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 23 章 · HIVM 方言源码剖面（六级内存 + Cube/Vector 双核分工 + § 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# 7 列 × 2 泳道，符号名普遍偏长——不加宽节点，改机械换行；NODE_W 只需装下"半个符号名"。
NODE_W, NODE_H = 155, 90
COL_GAP, ROW_GAP = 28, 22
EDGE_MARGIN, STUB_W, STUB_H = 12, 55, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 46, 20

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
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断。本章 cube_mmad 站牌是
    聚合站"§23.4–23.5"(10 字符,比常规 "§23.N" 长得多)，固定 BADGE_W=46 会把
    文字挤出胶囊两侧(渲染验证过：首版即是如此)，故按 cjk_text_width 动态算。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy) —— 节点用它贴右上角,路线legend用它居中挂线上。"""
    bw = badge_w(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 34

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝,画在节点下面这条先画后画都行,这里先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px,如 2 条即 ±8px),
# 否则重合的终点在视觉上看不出"汇合"、像一条线断头。
_dst_total = {}
for _, dst, _e in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst, elbow in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    if not elbow:
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    else:
        # 折角路径:先沿 src 所在行右探到"dst 列前一格的空隙"(该空隙必空——
        # 列间隙从不放节点),再垂直上探/下探到 dst 所在行,最后横向进入 dst。
        # 这样绕开夹在 src/dst 列之间、位于 src/dst 行之外的其它节点(本例即
        # col4/row0 的 vec_dispatch),不与其相交。
        dst_col = NODE_BY_ID[dst][2]
        turn_x = COLX[dst_col - 1] + NODE_W + COL_GAP / 2
        path = (f'M {p1[0]:.1f},{p1[1]:.1f} L {turn_x:.1f},{p1[1]:.1f} '
                f'L {turn_x:.1f},{p2[1]:.1f} L {p2[0]:.1f},{p2[1]:.1f}')
        L.append(f'<path d="{path}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
                  f'marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行,始终锚在节点下半区) + 右上角 § 徽标)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 24, 40
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 71, 66, 80
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines = symbol.split("\n")
    sym_ys = [y + SYMBOL_1LINE_Y] if len(sym_lines) == 1 else [y + SYMBOL_2LINE_Y1, y + SYMBOL_2LINE_Y2]
    for line, ly in zip(sym_lines, sym_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = phrase.split("\n")
    phrase_ys = [y + PHRASE_1LINE_Y] if len(phrase_lines) == 1 else [y + PHRASE_2LINE_Y1, y + PHRASE_2LINE_Y2]
    for line, ly in zip(phrase_lines, phrase_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge(x + NODE_W - badge_w(sec) / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, sec in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，首轮见 diagrams/figure-manifest.json 该条目)
