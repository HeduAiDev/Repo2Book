#!/usr/bin/env python3
"""第 33 章「本章地图」——全书最后一章，能力边界证据链的源码剖面图。

本章是自然标题章(节标题是"什么叫「通过」：…"/"支持面：…"这类自然语言标题，
无 `## N.M` 编号)——按契约禁用 §N.M 徽标，站牌一律用正文实际标题词(冒号前的
部分)本身，如"支持面""边界卡在哪一层""三面俱到"。

结构：本章不是分支/分流的调用图，而是一条**线性证据链**——四条泳道自上而下
对应正文八节的真实阅读顺序(判据→支持面→未支持面反面清单→分层归因→
skip/xfail/skipif 语义→边界粒度→flaky 半支持→三面俱到收官)：
  L0 判据层(1 节点)→ L1 支持面证据(1 节点)→
  L2 未支持面证据·反面清单与分层(同列纵向堆叠 2 节点，对应两节连续深挖)→
  L3 语义/粒度/半支持与结论(同列纵向堆叠 3 节点 + 收官节点，对应收尾四节)。
全图 8 个代码节点 ≤ 12，5 列×NODE_W=200 使宽度远低于 1500 上限。

■ 不可变(全书统一视觉语言，来自 skill 模板，换章节时不要动)：入口绿#22c55e-
  出口橙#f97316-主线蓝#3b82f6/图例规则/cjk_text_width()/节点圆角框样式。
■ 本章沿用 ch32 已验证的自然标题章扩展(非任意发挥)：
  1) badge() 胶囊宽度按文字实测(cjk_text_width + 左右 8px)动态撑开，下限 46px。
  2) 同列(NODE_COL 相同) = 纵向连接边(上下居中)；异列 = 横向连接边(右中→
     左中)——用于"同一大节内由浅入深"的下钻关系(如反面清单→分层归因同列
     纵向堆叠；三种请假条→边界的粒度→flaky 半支持同列纵向堆叠)。
  3) 本章末节"三面俱到"既是收官结论也是全书收官——出口桩标"收官"而非
     "下一章"(本章无下一章)，入口桩标"读者"(读者带着"到底能用到什么程度"
     的疑问切入判据一节)。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录)：见 figure-manifest.json。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用，非精确排版)：全角(ord>0x2E80)按 1.0×size，
    半角按 0.58×size，求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变：本章数据) ----------------
LANES = [
    "判据层 · 全套通用判据",
    "支持面证据",
    "未支持面证据 · 反面清单与分层",
    "语义、粒度、半支持与结论",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 两行短语(用|分行), 站牌——自然标题词，禁用 §N.M)
NODES = [
    ("entry",      0, 0, 0, "test_common.validate_cmp",
     "fp16/bf16 容差 1e-3|fp32 1e-4 · 整型逐位相等", "什么叫「通过」"),
    ("support",    1, 1, 0, "extension.compile_hint",
     "317 个测试:tutorials 01-18|逐算子 + 昇腾专属扩展各有专测", "支持面"),
    ("unsup_list", 2, 2, 0, "test_device_print_int8",
     "pytest_ut 生效标记 40 处|按 reason 字符串归堆", "未支持面"),
    ("layers",     2, 2, 1, "test_pow_vv",
     "等 TA (13) / 等 bishengir (9)|NPUIR 回退 (5) / UB (3) / attn_cp (3)", "边界卡在哪一层"),
    ("semantics",  3, 3, 0, "test_dot_2_allow_tf32",
     "skip 止血,恒定 s|xfail 真跑,XPASS 提醒撤标", "三种请假条"),
    ("grain",      3, 3, 1, "test_matrix_multiplication",
     "skip 沉到 pytest.param 级|精确到单 shape 或激活", "边界的粒度"),
    ("flaky",      3, 3, 2, "test_max_vector",
     "randomly failed 共 4 处|不算「不能」,是「还不够稳」", "半支持"),
    ("exit",       3, 4, 2, "conftest.assign_npu",
     "autouse 绑真机 NPU|host 无卡,只能静态核 marker", "三面俱到"),
]
EDGES = [  # (src_id, dst_id) —— 同列=纵向下钻边，异列=横向前进边(见文件头说明)
    ("entry", "support"),
    ("support", "unsup_list"),
    ("unsup_list", "layers"),
    ("layers", "semantics"),
    ("semantics", "grain"), ("grain", "flaky"),
    ("flaky", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 与正文开篇导读的两条路线一一对应："想跟着证据一条条核，从对拍判据读起"/
# "只想要结论，跳到本章末尾的「三面俱到」小节"。
ROUTES = [
    ("逐条核证据(顺序通读)", [(0, "什么叫「通过」"), (1, "支持面"), (2, "未支持面"),
                        (3, "三种请假条"), (4, "三面俱到")], True),
    ("只看结论(跳读)", [(0, "什么叫「通过」"), (4, "三面俱到")], False),
]
LEGEND = [("#22c55e", "入口:读者带着疑问切入判据"), ("#3b82f6", "章内主线走线"), ("#f97316", "出口:全书收官,无下一章")]
TITLE = "第 33 章 · 能力边界的证据链:从对拍判据到三面俱到（源码剖面图）"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替，仅装饰，非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W, NODE_H = 200, 78
COL_GAP, ROW_GAP = 24, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 70, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留：接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H = 20
BADGE_FONT_SIZE = 10

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
NODE_COL = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
    NODE_COL[nid] = col
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_width(text):
    """站牌胶囊按文字实测宽度撑开(下限 46px，与模板短数字 badge 视觉一致)。"""
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + 16)


def badge(cx, cy, text):
    """站牌胶囊(本章为自然标题，文字是标题词而非 §N.M)，居中挂在 (cx,cy)。
    宽度按文字实测动态撑开(下限 46px)；其余样式(圆角/配色/字号)照模板不变。
    """
    bw = badge_width(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT_SIZE}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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

# 入口/出口接口桩(给入口/出口箭头一个可附着的框，兼表达"读者带疑问切入判据/
# 章末即全书收官")
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("读者")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("收官")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)：列相同 → 纵向下钻边(同一大节内由浅入深)；列不同 → 横向
# 前进边(右中→左中)。
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    if NODE_COL[src] == NODE_COL[dst]:
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 两行短语 + 右上角站牌)。短语用 "|" 分两行，避免
# 长中英混排解释在单行里撑破节点宽度、压到相邻节点。
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.24:.1f}" text-anchor="middle" '
              f'font-family="monospace" font-size="11" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    phrase_lines = phrase.split("|")
    for pi, pl in enumerate(phrase_lines):
        py = y + NODE_H * (0.52 + pi * 0.24)
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{py:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(pl)}</text>')
    bw = badge_width(sec)
    L += badge(x + NODE_W + 8 - bw / 2, y, sec)  # 右边缘固定在 x+NODE_W+8，向左铺开 bw

# 底部阅读路线：复用列坐标 COLX，站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}  size={w:.0f}x{h:.0f}  ratio={w / h:.2f}")
