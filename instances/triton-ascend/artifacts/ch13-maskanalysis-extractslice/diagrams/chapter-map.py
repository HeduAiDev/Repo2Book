#!/usr/bin/env python3
"""ch13 本章地图:MaskAnalysis 源码剖面——从掩码表达式解析到 extract_slice/subview 落地。

两行泳道各 4 站,左→右即正文阅读顺序:
  上行 §13.2→§13.5:MaskState 状态 → parse 递归下降 → parseCmp 熔合 → clamp 夹负值
  下行 §13.6→§13.9:parseAnd 矩形交 → 三种不直白变体 → 两个发射器 → 消费端落地
底部路线条给出全书通读两段 + 三条选读跳转(核心动作/AND为何无OR/接回tt.load),
口径取自正文"选读指引"段原话。

坐标/尺寸全部由循环与常量计算,不手写魔数;文本全部 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):全角(ord>0x2E80)按 1.0×size,
    半角按 0.58×size 求和。中文字符是方块字,按半角系数估算会算短。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["解析:掩码表达式 → 矩形边界", "熔合变体 → 切片落地"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("maskstate", 0, 0, 0, "MaskState",
     "五字段/三形态:标量·range·矩形", "§13.2"),
    ("parse", 0, 1, 0, "MaskState::parse",
     "递归下降:11 分支 TypeSwitch 分派", "§13.3"),
    ("parsecmp", 0, 2, 0, "parseCmp",
     "cmpi 熔成(offset,dim)——本章心脏", "§13.4"),
    ("clamp", 0, 3, 0, "clampToNonNegativeIndex",
     "负维度夹 0(仅对常量)", "§13.5"),
    ("parseand", 1, 0, 0, "parseAnd / minStates",
     "andi=矩形交,无 parseOr", "§13.6"),
    ("variants", 1, 1, 0, "parseAdd/parseSplat/parseSel",
     "平移·布尔splat·select 障眼法", "§13.7"),
    ("emit", 1, 2, 0, "getExtractSlice / getSubview",
     "配全 1 strides,发射切片算子", "§13.8"),
    ("consumer", 1, 3, 0, "LoadStoreConverter",
     "parse→getSubview→copy 有效片", "§13.9"),
]
EDGES = [  # (src_id, dst_id) —— 同泳道内的直行数据流边,统一主线蓝(几何上保证不穿节点)
    ("maskstate", "parse"),
    ("parse", "parsecmp"),
    ("parsecmp", "clamp"),
    ("parseand", "variants"),
    ("variants", "emit"),
    ("emit", "consumer"),
]
WRAP_EDGE = ("clamp", "parseand")  # 唯一的跨泳道续接边,走右侧走廊单独绘制(避免穿过节点)
# (路线名, [(列, §编号), ...], 是否高亮:True=实线蓝/False=虚线灰)
# 前两条拼起来 = 正文"想跟全程,按序读";后三条 = 正文"选读指引"三个单点跳转原话。
ROUTES = [
    ("解析出矩形:按序读(上行)", [(0, "§13.2"), (1, "§13.3"), (2, "§13.4"), (3, "§13.5")], True),
    ("熔合落地:按序读(下行)", [(0, "§13.6"), (1, "§13.7"), (2, "§13.8"), (3, "§13.9")], True),
    ("只抓核心动作:比较→切片", [(2, "§13.4")], False),
    ("为什么没有 parseOr", [(0, "§13.6")], False),
    ("接回上一章 tt.load 落地", [(3, "§13.9")], False),
]
LEGEND = [
    ("#22c55e", "入口:算子带 mask,建空 MaskState"),
    ("#3b82f6", "章内数据流:表达式 → 矩形 → 切片"),
    ("#f97316", "出口:切片交给 memref.copy 落地"),
]
TITLE = "第 13 章 · MaskAnalysis 剖面:mask 表达式如何变成 extract_slice/subview"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 245, 60
COL_GAP, ROW_GAP = 34, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
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
FIRST_ID, LAST_ID = NODES[0][0], NODES[-1][0]

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
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

# 入口/出口接口桩:入口挂在第一个节点(MaskState)左侧,出口挂在最后一个节点(LoadStoreConverter)右侧
ex, ey = NODE_XY[FIRST_ID]; ey += NODE_H / 2
xx, xy = NODE_XY[LAST_ID]; xy += NODE_H / 2
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

# 同泳道直行边(主线蓝)——每条边 src 右中→dst 左中,src/dst 恒同泳道同行,
# 几何上不会穿过其他节点(几个节点严格按列递增排列)。
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 跨泳道续接边(唯一一条,§13.5→§13.6):走右侧走廊绕行,不穿过任何节点——
# 从 clamp 右侧出发→沿节点区右外侧的窄走廊下行→从 parseAnd 顶部落下,
# 全程只经过"没有节点"的空白区(节点区右边界与出口接口桩之间)。
_wsrc, _wdst = WRAP_EDGE
wx1, wy1 = NODE_XY[_wsrc]; wx2, wy2 = NODE_XY[_wdst]
_corridor_x = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + 10  # 节点区右边界外 10px
_wp = [
    (wx1 + NODE_W, wy1 + NODE_H / 2),
    (_corridor_x, wy1 + NODE_H / 2),
    (_corridor_x, band_top[1] - 6),
    (wx2 + NODE_W / 2, band_top[1] - 6),
    (wx2 + NODE_W / 2, wy2),
]
_wpts = " ".join(f"{x:.1f},{y:.1f}" for x, y in _wp)
L.append(f'<polyline points="{_wpts}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
         f'marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=通读 / 虚线灰=选读跳转)")}</text>')
_label_w = max(cjk_text_width(name, 12) for name, _, _ in ROUTES)
_route_x0 = 16 + _label_w + 24
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    if len(stops) == 1:
        col, sec = stops[0]
        cx = COLX[col] + NODE_W / 2
        L += badge(cx, ry, sec)
        L.append(f'<line x1="{_route_x0:.1f}" y1="{ry:.1f}" x2="{cx - BADGE_W / 2 - 6:.1f}" y2="{ry:.1f}" '
                  f'stroke="{C_ROUTE_DIM}" stroke-width="1.5" stroke-dasharray="6,4"/>')
    else:
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
print(f"wrote {out}  canvas={w:.0f}x{h:.0f}  ratio={w / h:.2f}")
