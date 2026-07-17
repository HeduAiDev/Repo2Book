#!/usr/bin/env python3
"""ch32 本章地图:五级台阶登记 → 第一跳内部机制(TypeConverter/ConversionTarget/
TritonDotPattern) → TTGIR 三件事。改自 .claude/skills/svg-diagram/references/
example-chapter-map.py 模板(§徽标胶囊/入口绿-出口橙-主线蓝/路线条高亮实线蓝-
次要虚线灰/cjk_text_width() 不可变,只改下面的 DATA)。

本章标题用 §1..§6(单层,无 N.M)+ 自然标题「小结」,不是 §N.M 格式——
lint_chapter_map 的 §N.M 正则(带小数点)不会匹配这些站牌,与本章 headings
天然一致(headings 本身就写的是「## §1 ...」这种单层编号)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_font(s, box_w, max_size, min_size):
    """按文本实际宽度反解一个不溢出 box_w 的字号(封顶 max_size,不低于
    min_size)——真实符号名(如 TritonGPUConversionTarget)比短标签长得多,
    与其手改每个节点宽度,不如让字号跟着文本长度自适应,零手写魔数。"""
    unit_w = cjk_text_width(s, 1.0)
    if unit_w <= 0:
        return max_size
    return max(min_size, min(max_size, box_w / unit_w))


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["五级台阶登记", "第一跳内部机制", "结果：TTGIR 三件事"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, [一行短语 ×2], §编号/自然标题)
NODES = [
    ("add_stages",  0, 0, 0, "add_stages",
     ["注册五段回调", "ttir→ttgir→llir→ptx→cubin"], "§1"),
    ("make_ttir",   0, 1, 0, "make_ttir",
     ["TTIR 级清理", "降解 block pointer"], "§2"),
    ("make_ttgir",  0, 2, 0, "make_ttgir",
     ["触发第一跳", "add_convert_to_ttgpuir"], "§3"),
    ("run_on_op",   1, 2, 0, "runOnOperation",
     ["搭三大组件", "驱动 applyPartialConversion"], "§3"),
    ("type_conv",   1, 3, 0, "TritonGPUTypeConverter",
     ["无布局张量", "→ 默认贴 Blocked 编码"], "§4"),
    ("conv_target", 1, 3, 1, "TritonGPUConversionTarget",
     ["tt.dot 合法性", "两操作数皆需 DotOperand"], "§5"),
    ("dot_pattern", 1, 4, 0, "TritonDotPattern",
     ["焊 convert_layout 胶水", "A/B→dot_op，C→结果布局"], "§6"),
    ("outcome",     2, 5, 0, "TTGIR 三件事",
     ["#blocked ＋ dot_op", "＋ convert_layout 胶水"], "小结"),
]
EDGES = [  # (src_id, dst_id) —— 调用/推导边，统一主线蓝
    ("add_stages", "make_ttir"),
    ("make_ttir", "make_ttgir"),
    ("make_ttgir", "run_on_op"),
    ("run_on_op", "type_conv"),
    ("run_on_op", "conv_target"),
    ("type_conv", "dot_pattern"),
    ("conv_target", "dot_pattern"),
    ("dot_pattern", "outcome"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("从头顺读（推荐）",        [(0, "§1"), (1, "§2"), (2, "§3"), (3, "§4"), (4, "§6"), (5, "小结")], True),
    ("只看 dot 胶水机制（跳读）", [(3, "§5"), (4, "§6"), (5, "小结")], False),
]
LEGEND = [("#22c55e", "入口：进入五级台阶第一段"), ("#3b82f6", "章内主线调用/推导边"), ("#f97316", "出口：TTGIR 交给下一跳")]
TITLE = "第 32 章 · 五级台阶与第一跳 TTIR→TTGIR 剖面（源码走线 + § 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 175, 74
COL_GAP, ROW_GAP = 28, 24
SYMBOL_MAX, SYMBOL_MIN = 13, 9.5
PHRASE_MAX, PHRASE_MIN = 9.8, 7.8
TEXT_PAD = 14  # 节点内左右各留的安全边距(box_w = NODE_W - TEXT_PAD)
EDGE_MARGIN, STUB_W, STUB_H = 14, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32
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

L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')

_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 34

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

ex, ey = NODE_XY["add_stages"]; ey += NODE_H / 2
xx, xy = NODE_XY["outcome"]; xy += NODE_H / 2
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

_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    src_col, dst_col = NODE_BY_ID[src][2], NODE_BY_ID[dst][2]
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    if src_col == dst_col:
        # 同列跨泳道(从上一泳道垂直落到下一泳道):走底边中点→顶边中点,
        # 不走左右边——同列意味着右边→左边要"倒着走"一整个节点宽,箭头
        # 反而扎进目标框内部被节点矩形盖住(arrows_attached 会挂)。
        x_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2 + x_offset, y2)
    else:
        y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

box_w = NODE_W - TEXT_PAD
for nid, lane, col, row, symbol, phrase_lines, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_size = fit_font(symbol, box_w, SYMBOL_MAX, SYMBOL_MIN)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.32:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sym_size:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    for li, line in enumerate(phrase_lines):
        ln_size = fit_font(line, box_w, PHRASE_MAX, PHRASE_MIN)
        ly = y + NODE_H * (0.58 + li * 0.21)
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{ln_size:.1f}" fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线（标号=图上 § 站牌；实线蓝=推荐 / 虚线灰=次要）")}</text>')
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio={w/h:.2f})")
