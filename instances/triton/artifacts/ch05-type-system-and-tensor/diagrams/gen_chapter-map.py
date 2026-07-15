#!/usr/bin/env python3
"""chapter-map（本章地图）：第 5 章「值的三层类型 → 下降 → 承载 → 转换」源码剖面。
横向泳道 = 源码文件（core.py / _utils.py / semantic.py），圆角节点 = 真实符号 + 一行短语，
节点右上角挂 §N 讲解站牌，画布左右边缘各有一个「调用方 / 返回上层」接口桩，
底部用同一批 § 站牌拼出两条阅读路线（顺读全程 / 性能账速通道）。

本章七节被聚合成 6 站（§6 cast 与 §7 bitcast 是 semantic.py 里对照的一对 dispatch，
合成一站「§6·§7 cast/bitcast」），主线按 data_flow 推进：
  构造三层类型（§1）→ 分账 fp8 三元组（§2）→ to_ir 下降查后端（§3）→
  block 形状把关（§4）→ tensor 承载值（§5）→ cast/bitcast 转换（§6·§7）。

■ 不可变（全书统一视觉语言，只改下面的 DATA / 几何常量）：
  §徽标胶囊 fill #eef2ff / stroke #6366f1；入口绿 #22c55e / 出口橙 #f97316 / 主线蓝 #3b82f6；
  路线条：高亮=实线蓝、次要=虚线灰；文本宽度一律 cjk_text_width() 估算（中英混排必须）。

■ 六项自查（渲染→Read PNG 亲眼看后如实记录）：
  claim_readable_10s / numbers_match_spec / no_overlap / arrows_attached /
  cjk_rendered / reading_order_clear —— 见 figure-manifest.json。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size，求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(s, max_w, base, floor=9.5):
    """自动收字号：让文本 s 在 max_w 内排下，最大不超过 base、最小 floor。"""
    unit = cjk_text_width(s, 1.0)
    if unit <= 0:
        return base
    return max(floor, min(base, max_w / unit))


# ---------------- DATA（可变：本章数据） ----------------
LANES = ["core.py · 类型体系与 tensor", "_utils.py · 形状校验", "semantic.py · 类型转换"]  # 上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, § 站牌)
NODES = [
    ("s1",  0, 0, 0, "dtype / pointer_type / block_type", "三层套娃：标量 → 指针 → 块",      "§1"),
    ("s2",  0, 1, 0, "fp_mantissa_width · exponent_bias", "8 bit 三元组：精度↔量程账",        "§2"),
    ("s3",  0, 2, 0, "to_ir → builder.get_*_ty",          "下降 IR：先查 fp8 后端支持",       "§3"),
    ("s4",  1, 3, 0, "validate_block_shape",              "每维 2 的幂 · numel ≤ 1048576",   "§4"),
    ("s5",  0, 4, 0, "tensor = (handle, type)",           "一张提货单，非数据本身",          "§5"),
    ("s67", 2, 5, 0, "cast / bitcast",                    "大 dispatch：每支发一个 IR op",   "§6·§7"),
]
# (src_id, dst_id) —— 主线蓝，按 data_flow 推进（列号严格递增，故箭头一律向右）
EDGES = [
    ("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5"), ("s5", "s67"),
]
ENTRY_ID, EXIT_ID = "s1", "s67"  # 入口桩接 §1 左侧，出口桩接 §6·§7 右侧
# (路线名, [(列, § 站牌), ...] 按阅读顺序, 是否高亮)
ROUTES = [
    ("顺读全程（§1 → §7）", [(0, "§1"), (1, "§2"), (2, "§3"), (3, "§4"), (4, "§5"), (5, "§6·§7")], True),
    ("性能账速通道（dtype 选型）", [(1, "§2"), (3, "§4"), (5, "§6·§7")], False),
]
LEGEND = [("#22c55e", "入口：从上一章进入"), ("#3b82f6", "章内主线（data flow）"), ("#f97316", "出口：接下一章")]
TITLE = "第 5 章 · 值的三层类型 → 下降 → 承载 → 转换（源码剖面 + § 站牌）"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量（全计算，零魔数） ----------------
NODE_W, NODE_H = 190, 72
COL_GAP, ROW_GAP = 26, 20
EDGE_MARGIN, STUB_W, STUB_H = 14, 56, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 30
LANE_LABEL_H, BAND_PAD = 24, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 18
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
BADGE_H = 20

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

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊，居中挂在 (cx,cy)。宽度按文字自适应（≥46）。"""
    bw = max(46, cjk_text_width(text, 11) + 16)
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

# 图例（>2 种语义色必须画图例）
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

# 入口 / 出口接口桩
ex, ey = NODE_XY[ENTRY_ID]; ey += NODE_H / 2
xx, xy = NODE_XY[EXIT_ID]; xy += NODE_H / 2
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

# 调用边（主线蓝）：列号严格递增，src 右缘 → dst 左缘
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点（圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标；符号/短语自动收字号防越界）
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    fs_sym = fit_size(symbol, NODE_W - 18, 13)
    fs_sub = fit_size(phrase, NODE_W - 18, 11)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{fs_sym:.1f}" font-weight="bold" '
             f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.70:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{fs_sub:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - 30, y, sec)

# 底部阅读路线：复用列坐标 COLX，§ 徽标与图上节点竖向对齐
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线（标号=图上 § 站牌；实线蓝=推荐顺读 / 虚线灰=按需速通）")}</text>')
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
print(f"wrote {out}  viewBox=0 0 {w} {h}  ratio={w / h:.2f}:1  cols={n_cols}")
