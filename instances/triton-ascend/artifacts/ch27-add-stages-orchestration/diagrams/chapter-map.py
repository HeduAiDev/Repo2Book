#!/usr/bin/env python3
"""ch27「本章地图」——add_stages 三段编排剖面：装配层(add_stages 登记) → 常规路径
实现体(make_ttir/ttir_to_linalg/npubin 二选一) 与 快路径实现体(ttir_to_npubin)
两条真实分叉，末尾并排一个「昇腾 vs 基座」对照站(§27.5)。全部符号取自
third_party/ascend/backend/compiler.py 同一文件，故不再逐节点重复路径行
(法法与 ch15 的多文件跨泳道场景不同，这里省掉 path 行以省画布)。

本章是**数字编号章**(`## 27.1`…`## 27.6`)，站牌用 §27.N 徽标。

节点预算(7 个，≤12)：
  entry(add_stages,§27.1) / make_ttir(§27.2) / ttir_to_linalg(§27.3，ttadapter
  主脊,11 个 add_* 编排——细节已有专图 fig-ch27-pass-pipeline，本图只画收口
  站，不逐个铺 11 个 pass 节点，避免超预算) / knobs(§27.3 内部旋钮，聚合
  27.3.1-27.3.3 三个小节，聚合站——按契约"超长章聚合"处理，不逐个开 3 个
  子节点) / npubin_regular(§27.1 末段二选一) / ttir_to_npubin(§27.4 快路径) /
  compare(§27.5 三段 vs 五段对照)。

主线(实线蓝)=add_stages 的四处登记(EDGES_MAIN)；旁支(虚线灰)=非因果的
"内部细节/事后对照"关系(EDGES_SIDE)：ttir_to_linalg→knobs(展开 27.3 内部
三个旋钮，不是调用顺序)、npubin_regular→compare(读完主链后的回望对照，
不是数据流)。

跨章标注(exp-2026-07-18-04 硬规则：目标章号 > 本章号用「预告」，< 用「回指」)：
  入口桩(绿)："回指 ch26" —— add_stages 与 stages 字典已在上一章挂好。
  出口桩(橙)："预告 ch28" —— npubin 产物(常规/快路径两个来源都收敛于此)
                交下一章 bishengir-compile，命令行细节归 ch28。

底部阅读路线：常规路径(默认,3 段) vs 快路径(force_simt_only,2 段)，两条
路线共享前两站(add_stages/make_ttir)，从第三站起分叉——正是正文 27.1 的
"分叉从第二段才开始"论点的视觉版本。

不可变视觉语言(全书统一，来自 example-chapter-map.py 模板 + ch15 的动态换行/
自适应符号字号改法)：§徽标胶囊 fill #eef2ff / stroke #6366f1、入口绿 #22c55e /
出口橙 #f97316 / 主线蓝 #3b82f6 / 旁支虚线灰 #94a3b8、cjk_text_width() 逐字符
宽度估算。

六项自查(渲染→Read PNG 亲眼看后如实记录)：见同目录 figure-manifest.json 该图 selfcheck。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


_BREAK_AFTER = set("，；：、/ ,;)")


def wrap_claim(text, max_w, size):
    """一句论点太长时换行——只在标点/斜杠/空格之后断行，不劈开一个标识符或中文词。
    贪心找"prefix 仍不超宽的最靠后一个合法断点"；找不到合法断点才整句照旧单行放行。"""
    breaks = [i for i, ch in enumerate(text) if ch in _BREAK_AFTER]
    best = None
    for i in breaks:
        if cjk_text_width(text[:i + 1], size) <= max_w:
            best = i
    if best is None:
        return [text]
    line1, line2 = text[:best + 1].rstrip(), text[best + 1:].lstrip()
    if cjk_text_width(line2, size) <= max_w:
        return [line1, line2]
    return [line1] + wrap_claim(line2, max_w, size)


# ---------------- DATA(可变：本章数据) ----------------
LANES = [
    "third_party/ascend/backend/compiler.py · 装配层：add_stages 登记总入口",
    "third_party/ascend/backend/compiler.py · 常规路径实现体(默认，3 段)",
    "third_party/ascend/backend/compiler.py · 快路径实现体(force_simt_only，2 段)",
    "对照：昇腾 AscendBackend vs 基座 CUDABackend",
]

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 一句论点, §编号)
NODES = [
    ("entry", 0, 0, 0,
     ["add_stages"],
     "无条件登记 stages['ttir']；两处 if 分叉决定接下来登记哪几段",
     "27.1"),
    ("make_ttir", 1, 1, 0,
     ["make_ttir"],
     "8 个 TTIR 前端优化 pass，与所有后端共享，一行没改",
     "27.2"),
    ("ttir_to_linalg", 1, 2, 0,
     ["ttir_to_linalg"],
     "按拓扑序挂 11 个 ascend.passes.ttir.add_*，收口成 Linalg named op",
     "27.3"),
    ("knobs", 1, 2, 1,
     ["add_auto_scheduling", "auto_blockify_size"],
     "旋钮全从 metadata 取；自动调度默认关；未开并行块映射则强制归 1",
     "27.3"),
    ("npubin_regular", 1, 3, 0,
     ["compile_910_95", "compile_A2_A3"],
     "compile_on_910_95 二选一，两代硬件实现职责相同",
     "27.1"),
    ("ttir_to_npubin", 2, 2, 0,
     ["ttir_to_npubin"],
     "跳过 ttadapter 段(11 个 add_* 归 0)，TTIR 直编 npubin",
     "27.4"),
    ("compare", 3, 3, 0,
     ["AscendBackend", "CUDABackend"],
     "3 段(无 TTGIR)vs 基座 5 段；因无真实 warp，warp_size=0",
     "27.5"),
]
NODE_BY_ID = {n[0]: n for n in NODES}
ENTRY_NODE = "entry"
EXIT_NODE = "npubin_regular"        # 出口桩以此节点的行高对齐(常规路径末段)

# 主线(实线蓝)——add_stages 的四处登记(注册关系，真实调用)。entry 与
# ttir_to_linalg/npubin_regular 同排(row0)但隔着 make_ttir 列，entry 与
# ttir_to_npubin 更隔着一整个泳道——直线连接会穿框，故一律走"下沉主干→
# 行间空隙横移→垂直探入目标"的折线，探入点严格落在 row0/row1 之间、
# lane1/lane2 之间两条天然空隙(由 NODE_H/ROW_GAP 反推，非手写魔数)，
# 详见下方绘制代码 EDGES_MAIN_ELBOW。
EDGES_SIDE = [  # 虚线灰——非因果的"内部细节/事后对照"关系，不是调用序
    ("ttir_to_linalg", "knobs"),
    ("npubin_regular", "compare"),
]

ROUTES = [  # (路线名, [(列, §编号), ...]按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
    ("常规路径(默认，3 段)", [(0, "27.1"), (1, "27.2"), (2, "27.3"), (3, "27.1")], True),
    ("快路径(force_simt_only，2 段)", [(0, "27.1"), (1, "27.2"), (2, "27.4")], False),
]
LEGEND = [
    ("#22c55e", "入口(回指 ch26)：add_stages 与 stages 字典已在上一章挂好"),
    ("#3b82f6", "主线：add_stages 登记 → 各段实现体(常规 3 段 / 快路径 2 段)"),
    ("#f97316", "出口(预告 ch28)：npubin 产物(两条路径都汇入)交下一章 bishengir-compile"),
]
TITLE = "第 27 章 · add_stages 三段编排剖面(源码走线 + § 讲解站牌)"
SUBNOTE = "全部符号同属 third_party/ascend/backend/compiler.py；ttir_to_linalg 内部 11 个 add_* 的逐个对应见正文 27.3 表格与专图，本图只画收口站"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 190
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 78, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 26
LANE_LABEL_H, BAND_PAD = 22, 12
TOP_PAD, TITLE_H, SUBNOTE_H, LEGEND_H, BOTTOM_PAD = 12, 26, 22, 3 * 14.5 + 10, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_W, BADGE_H = 46, 20
SYM_FONT, SYM_LINE_H = 12.0, 15
CLAIM_FONT = 9.6

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

# 每个节点的论点先按 NODE_W 预算换行一遍，符号行数取全章最大值——统一定 NODE_H。
CLAIM_MAXW = NODE_W - 14
_CLAIM_LINES = {n[0]: wrap_claim(n[5], CLAIM_MAXW, CLAIM_FONT) for n in NODES}
_max_claim_lines = max(len(v) for v in _CLAIM_LINES.values())
_max_sym_lines = max(len(n[4]) for n in NODES)
SYM_TOP = 24
CLAIM_TOP = SYM_TOP + (_max_sym_lines - 1) * SYM_LINE_H + 18
NODE_H = CLAIM_TOP + (_max_claim_lines - 1) * 12 + 16

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + SUBNOTE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD
assert w <= 1500 and w / h <= 2.6, f"画布预算超标：{w}x{h}, {w / h:.2f}:1"


def badge(cx, cy, text):
    """§ 徽标胶囊，居中挂在 (cx,cy)。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc("§" + text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Side", C_ROUTE_DIM))
) + '</defs>')
L.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="white"/>')

# 标题 + 副注
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 17}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
_subnote_lines = wrap_claim(SUBNOTE, w - 2 * PAD_L, 9.2)
for si, sline in enumerate(_subnote_lines[:2]):
    L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + TITLE_H + 8 + si * 11:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="9.2" fill="{C_NODE_SUB}">{esc(sline)}</text>')

# 图例(3 种语义色必须画图例；纵向列表，逐条文字较长，横向排会挤)
for li, (color, label) in enumerate(LEGEND):
    _row_y = TOP_PAD + TITLE_H + SUBNOTE_H + 12 + li * 14.5
    L.append(f'<rect x="{PAD_L}" y="{_row_y - 10}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 17}" y="{_row_y}" font-family="sans-serif" font-size="9.8" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="14" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="10.8" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
                 f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w:.1f}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(跨章标注：目标章号 > 本章号用「预告」，< 本章号用「回指」)
ex, ey = NODE_XY[ENTRY_NODE]; ey += NODE_H / 2
xx, xy = NODE_XY[EXIT_NODE]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("回指 ch26")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("预告 ch28")}</text>')
# 出口边:常规路径末段(npubin_regular)——直线，右侧留白区(PAD_R)天然无框，不需要绕
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')


def draw_elbow(points, color, width, marker, dash=False):
    """按折线依次画线段，箭头只挂在最后一段——用于绕开中间列的节点框。"""
    dasharray = ' stroke-dasharray="6,4"' if dash else ''
    for i in range(len(points) - 1):
        (x1, y1), (x2, y2) = points[i], points[i + 1]
        is_last = i == len(points) - 2
        marker_attr = f' marker-end="url(#{marker})"' if is_last else ''
        L.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{color}" stroke-width="{width}"{dasharray}{marker_attr}/>')


# 主线(实线蓝)——add_stages 的四处登记。entry 与 ttir_to_linalg/npubin_regular
# 同行(row0)但隔着 make_ttir 列，entry 与 ttir_to_npubin 更隔着一整个泳道——
# 直连会穿框，改走"入口所在行先横移到目标列正上方 → 再垂直探入目标边缘"折线，
# 横移全程都在 lane0(装配层)这一空荡荡的泳道内部，不经过任何其它节点/文字。
# 三个 row0 目标的列心(431/651/871)都落在 lane1 泳道标签文字右侧，垂直下探
# 不会切过标签；zero手写魔数——安全 x 阈值即 lane1 标签实测宽度(cjk_text_width
# 逐字符估算，14 是标签左内边距)。下探线只从 entry_mid_y(已低于 lane0 自己
# 的标签行)出发一路向下，故只需核对会途经的 lane1 标签，无需管 lane0/lane2。
LANE_LABEL_SAFE_X = 14 + cjk_text_width(LANES[1], 10.8)

row0_top = NODE_XY["make_ttir"][1]
knobs_bottom = NODE_XY["knobs"][1] + NODE_H
fast_top = NODE_XY["ttir_to_npubin"][1]
BUS_Y2 = (knobs_bottom + fast_top) / 2  # knobs 底边与 ttir_to_npubin 顶边之间的夹层
# 快路径垂直下探要绕开 ttir_to_linalg/knobs 所在整列，改走 col1/col2 之间的
# COL_GAP 走廊(该走廊天生无框，且早已越过 LANE_LABEL_SAFE_X)。
BYPASS_X = COLX[1] + NODE_W + COL_GAP / 2
assert BYPASS_X > LANE_LABEL_SAFE_X, "旁路走廊仍落在泳道标签文字范围内，需收窄标签或加宽走廊"

# entry 的右缘(不是 ex——ex 是 entry 左缘，专供入口桩箭头用)才是主线扇出的起点
entry_right_x = NODE_XY[ENTRY_NODE][0] + NODE_W
entry_mid_y = NODE_XY[ENTRY_NODE][1] + NODE_H / 2
_entry_exit_pt = (entry_right_x, entry_mid_y)
for target in ("make_ttir", "ttir_to_linalg", "npubin_regular"):
    tx, ty = NODE_XY[target]
    cx = tx + NODE_W / 2
    assert cx > LANE_LABEL_SAFE_X, f"{target} 列心落在泳道标签文字范围内"
    draw_elbow([_entry_exit_pt, (cx, entry_mid_y), (cx, row0_top)], C_MAIN, 2, "mMain")
tx, ty = NODE_XY["ttir_to_npubin"]
cx2 = tx + NODE_W / 2
draw_elbow([_entry_exit_pt, (BYPASS_X, entry_mid_y), (BYPASS_X, BUS_Y2), (cx2, BUS_Y2), (cx2, fast_top)],
           C_MAIN, 2, "mMain")

# 旁支(虚线灰)——非因果的"内部细节/事后对照"关系，不是调用序
for src, dst in EDGES_SIDE:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1 = (xs_ + NODE_W / 2, ys_ + NODE_H)
    p2 = (xd + NODE_W / 2, yd) if NODE_BY_ID[dst][1] != NODE_BY_ID[src][1] and NODE_BY_ID[dst][2] == NODE_BY_ID[src][2] else (xd, yd + NODE_H / 2)
    if NODE_BY_ID[dst][2] == NODE_BY_ID[src][2]:
        # 同列，纵向连(如 ttir_to_linalg → knobs)
        L.append(f'<line x1="{xs_ + NODE_W / 2:.1f}" y1="{ys_ + NODE_H:.1f}" '
                 f'x2="{xd + NODE_W / 2:.1f}" y2="{yd:.1f}" '
                 f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4" '
                 f'marker-end="url(#mSide)"/>')
    else:
        # 跨列，横向连(如 npubin_regular → compare)
        L.append(f'<line x1="{xs_ + NODE_W:.1f}" y1="{ys_ + NODE_H / 2:.1f}" '
                 f'x2="{xd:.1f}" y2="{yd + NODE_H / 2:.1f}" '
                 f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4" '
                 f'marker-end="url(#mSide)"/>')

# 节点(圆角框 + 符号(自适应字号) + 论点(自适应换行) + 右上角 § 徽标)
for nid, lane, col, row, syms, claim, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="11" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_w_budget = NODE_W - 16
    sym_size = SYM_FONT
    while max(cjk_text_width(s, sym_size) for s in syms) > sym_w_budget and sym_size > 7.0:
        sym_size -= 0.2
    for si, s in enumerate(syms):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + SYM_TOP + si * SYM_LINE_H:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{sym_size:.1f}" '
                 f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(s)}</text>')
    for ci, cline in enumerate(_CLAIM_LINES[nid]):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + CLAIM_TOP + ci * 12:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{CLAIM_FONT}" '
                 f'fill="{C_NODE_SUB}">{esc(cline)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 6, y, sec)

# 底部阅读路线：复用列坐标 COLX，§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="14" y="{routes_top + 14:.1f}" font-family="sans-serif" font-size="11" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌；实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (rname, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="14" y="{ry + 3.5:.1f}" font-family="sans-serif" font-size="10.2" '
             f'fill="{C_NODE_TITLE}">{esc(rname)}</text>')
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, aspect {w / h:.2f}:1)")
