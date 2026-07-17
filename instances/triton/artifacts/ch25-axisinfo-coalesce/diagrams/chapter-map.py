#!/usr/bin/env python3
"""ch25《AxisInfo 静态分析 + Coalesce 改写》本章地图——源码剖面图。

两条泳道 = 两个源文件/两个半场：分析半场(lib/Analysis/AxisInfo.cpp) 在上，
改写半场(lib/Dialect/TritonGPU/Transforms/Coalesce.cpp) 在下。全章 8 节严格
线性递进(§1→§8)，为塞进画布宽度预算，折成两行 4 列——§4 join() 算完格值后，
一条"折行"边(elbow，非直线对角)绕到下一行左端交给 §5 setCoalescedEncoding()，
避免长对角线穿过中间无关节点。

■ 不可变(全书统一视觉语言，换章节数据时不要动这些，只改下面的 DATA)：
  与 example-chapter-map.py 完全一致——§徽标胶囊 / 入口绿#22c55e-出口橙#f97316-
  主线蓝#3b82f6 / 高亮实线蓝-次要虚线灰 / cjk_text_width() 宽度估算。

■ 可变：LANES / NODES / EDGES(行内直线) / WRAP_EDGE(跨行折行边，本章专属，
  因两行共用同一组列坐标) / ROUTES / LEGEND / TITLE。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录)：
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  (§ 徽标 §1-§8 逐一核对正文实际 `## §N` 标题；8 个代码符号 AxisInfo /
  getPessimisticValueState / visitOperation / join / setCoalescedEncoding /
  getNumElementsPerThread / coalesceOp / runOnOperation 均在 dossier.json
  code_spine/mechanisms 与正文代码块中原样出现。)

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变：本章数据) ----------------
LANES = ["分析半场 · lib/Analysis/AxisInfo.cpp (只读推断)",
         "改写半场 · lib/Dialect/TritonGPU/Transforms/Coalesce.cpp (据推断落地)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("axisinfo",    0, 0, 0, "AxisInfo",                  "三元组:contiguity/divisibility/constancy", "§1"),
    ("pessimistic", 0, 1, 0, "getPessimisticValueState",  "悲观初值:从 tt.divisibility 起手",          "§2"),
    ("visit",       0, 2, 0, "visitOperation",            "per-op visitor 前向传播",                   "§3"),
    ("join",        0, 3, 0, "join",                      "逐轴 gcd,提示会被冲淡",                     "§4"),
    ("setcoal",     1, 0, 0, "setCoalescedEncoding",      "argSort(contiguity) 定 order",              "§5"),
    ("perthread",   1, 1, 0, "getNumElementsPerThread",   "三道 min 定每线程向量宽",                   "§6"),
    ("coalesceop",  1, 2, 0, "coalesceOp",                "convert+新op+convert回+替换",                "§7"),
    ("runop",       1, 3, 0, "runOnOperation",            "闭环骨架:先分析后改写",                     "§8"),
]
EDGES = [  # 行内直线(同泳道相邻列)，跨泳道的 join→setcoal 折行边单列 WRAP_EDGE
    ("axisinfo", "pessimistic"), ("pessimistic", "visit"), ("visit", "join"),
    ("setcoal", "perthread"), ("perthread", "coalesceop"), ("coalesceop", "runop"),
]
WRAP_EDGE = ("join", "setcoal")  # 折行边：§4 分析半场收尾 → §5 改写半场开局
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("分析半场", [(0, "§1"), (1, "§2"), (2, "§3"), (3, "§4")], True),
    ("改写半场", [(0, "§5"), (1, "§6"), (2, "§7"), (3, "§8")], True),
    ("只抓提示为什么不生效", [(1, "§2"), (3, "§4")], False),
]
LEGEND = [("#22c55e", "入口:上游提示进 seed(ch09/ch16 打的标记)"),
          ("#3b82f6", "章内主线:分析→改写调用/数据依赖边"),
          ("#f97316", "出口:交给下一章同范式变体 pass")]
TITLE = "第 25 章 · AxisInfo 静态分析 → Coalesce 改写剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替，仅装饰，非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W, NODE_H = 210, 60
COL_GAP, ROW_GAP = 34, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 78, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 30
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_W, BADGE_H = 40, 20
WRAP_GAP = 22  # 折行边:绕出节点右侧的横向余量

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


def badge(cx, cy, text):
    """§ 徽标胶囊，居中挂在 (cx,cy)。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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

# 入口/出口接口桩
ex, ey = NODE_XY["axisinfo"]; ey += NODE_H / 2
xx, xy = NODE_XY["runop"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("上游提示")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 行内直线调用边(主线蓝)
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

# 折行边(join → setcoal)：右侧绕出 → 下降到"泳道1标签+留白"这条空白带(节点顶部
# 之上、分界线之下，全程无节点/无文字) → 沿这条空白带一路向左 → 短距下降落入
# 下一行首节点顶部。全程不穿过任何节点(尤其不穿过 setcoal/perthread/coalesceop
# 所在的整片行 1 节点区域)，箭头在节点正上方的留白里清晰可见再落进节点。
wsrc, wdst = WRAP_EDGE
wx1, wy1 = NODE_XY[wsrc]; wx2, wy2 = NODE_XY[wdst]
p_start = (wx1 + NODE_W, wy1 + NODE_H / 2)
turn_x = wx1 + NODE_W + WRAP_GAP
drop_y = wy2 - 8  # 紧贴 wdst 节点顶部之上的空白带,不落入节点内部
p_mid1 = (turn_x, wy1 + NODE_H / 2)
p_mid2 = (turn_x, drop_y)
p_mid3 = (wx2 + NODE_W / 2, drop_y)
p_end = (wx2 + NODE_W / 2, wy2)
L.append(f'<polyline points="{p_start[0]:.1f},{p_start[1]:.1f} {p_mid1[0]:.1f},{p_mid1[1]:.1f} '
         f'{p_mid2[0]:.1f},{p_mid2[1]:.1f} {p_mid3[0]:.1f},{p_mid3[1]:.1f} {p_end[0]:.1f},{p_end[1]:.1f}" '
         f'fill="none" stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio {w / h:.2f})")
