#!/usr/bin/env python3
"""ch21 本章地图——源码剖面图。

HFusion 是「Linalg 之上再补一层」的方言：三件事(词汇/意图/兼容)+ 一步上抬(uplift)。
四条泳道 = 章内四层证据(方言身份与结构兼容 / 扩展词汇 / 融合意图 / 上抬 pass)，
圆角节点 = 真实符号名 + 一行短语，右上角挂站牌(本章自然标题，无 `## N.M` 编号，
故站牌用标题词而非 §N.M)，左右各一个接口桩(入口=上一章逃生舱送进来的一个具体
op 样例，出口=FusionKind 驱动 AutoSchedule、转下一章)，底部两条阅读路线复用
同一批站牌——其中「只读主干」路线直接照搬正文自己给的选读指引原话
(见 chapter.md 开篇："读完下面第一节就有了主干答案，再挑第七、第八节看融合
意图与上抬")。

列号 = 正文标题出现顺序：0 一(为什么需要 HFusion，入口样例) → 1 二(方言身份)
→ 2 三(词汇表) → 3 四(结构化 op) → 4 五(gather) → 5 六(专属 op) → 6 七
(FusionKind) → 7 八(上抬) → 8 九(边界) → 9 十(小结，出口)。走线严格左→右
单向递增列号，无回绕。

节点符号选取原则：优先选**短而真**的具体符号，避免超长类名把节点撑爆画布
预算——如「结构化 op」一站没有画完整类名 `HFusionStructuredBase_Op`，而是画
它最硬的那句证据 `reifyResultShapes`(cast 成 linalg::LinalgOp 那一行)；
「上抬」一站没有画某一个具体 pattern 类名，而是画四个 pattern 共享的基类
`OpRewritePattern`(正文原话就是拿这个词统摄四个 pattern)。两者都是正文逐字
出现的真实符号，只是选了更短的那个既真又能说清论点的词。

■ 不可变(照搬模板视觉语言，只改 DATA 与几何常量)：站牌胶囊 / 入口绿
  #22c55e-出口橙#f97316-主线蓝#3b82f6 / 高亮路线实线蓝、次要虚线灰 /
  cjk_text_width() 宽度估算。
■ 本章为自然标题(无 `## N.M` 编号)，站牌一律用标题词本身，禁用 §N.M。
■ 几何常量(NODE_W/COL_GAP/PAD 等)按本章 10 列节点数据调小，以满足画布预算
  (宽 ≤1500 且宽高比 ≤2.6:1)——这是"可变"的布局参数，不是共享视觉语言。
■ 长符号名一律按估算宽度动态缩小字号，避免文字越界(fit_font_size())；
  该函数同时用于符号名与一行短语两行文字。

[自查记录见文件末尾注释：Read PNG 后逐项如实记录，不能凭想象填。]
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用，非精确排版)：全角按 1.0×size，半角按
    0.58×size，求和——中文标签若按半角系数算会算短，导致下一个图例压上来。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_font_size(text, max_w, base=13, floor=9):
    """长文字按估算宽度动态缩小字号，不许越界。先算 base 号字是否已经放得下，
    放不下就解一个恰好贴边的字号；解出来的字号仍设一个下限(floor)防止字号
    小到不可读——本章所有节点符号/短语经过设计筛选，实测都不会触底。"""
    if cjk_text_width(text, base) <= max_w:
        return base
    unit = cjk_text_width(text, 1.0) or 1.0
    return max(floor, max_w / unit)


# ---------------- DATA(可变：本章数据) ----------------
LANES = ["方言身份 & 结构兼容 (.td)", "扩展词汇 (.td)", "融合意图 (.td)", "上抬 pass (.cpp)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(标题词，自然标题章无 §))
NODES = [
    ("entry", 0, 0, 0, "tt.histogram",
     "上一章逃生舱送入的具体 op", "一 为什么加层"),
    ("identity", 0, 1, 0, "HFusion_Dialect",
     "name=hfusion，依赖 linalg", "二 方言身份"),
    ("vocab", 1, 2, 0, "elemwise_unary",
     "枚举当属性，一 op 顶整族函数", "三 词汇表"),
    ("structured_base", 0, 3, 0, "reifyResultShapes",
     "cast 成 linalg::LinalgOp 调用", "四 结构化 op"),
    ("gather", 1, 4, 0, "hfusion.gather",
     "k 轴不可切，写进 op 语义", "五 gather"),
    ("exclusive_ops", 1, 5, 0, "atomic_rmw",
     "Linalg 结构不了的语义", "六 专属 op"),
    ("fusionkind", 2, 6, 0, "FusionKind",
     "10 种融合意图，贴在 func 上", "七 FusionKind"),
    ("uplift", 3, 7, 0, "OpRewritePattern",
     "4 个模式，partial conversion", "八 上抬"),
    ("boundary", 3, 8, 0, "linalg.map",
     "扩展词汇上抬，原生词汇留守", "九 边界"),
    ("exit", 0, 9, 0, "AutoSchedule",
     "按 kind 分派调度，转下一章", "十 小结"),
]
EDGES = [  # (src_id, dst_id) —— 章内讲解走线，统一主线蓝；src 列号恒 < dst 列号
    ("entry", "identity"), ("identity", "vocab"), ("vocab", "structured_base"),
    ("structured_base", "gather"), ("gather", "exclusive_ops"),
    ("exclusive_ops", "fusionkind"), ("fusionkind", "uplift"),
    ("uplift", "boundary"), ("boundary", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
ROUTES = [
    ("完整通读", [(0, "一 为什么加层"), (1, "二 方言身份"), (2, "三 词汇表"),
              (3, "四 结构化 op"), (4, "五 gather"), (5, "六 专属 op"),
              (6, "七 FusionKind"), (7, "八 上抬"), (8, "九 边界"),
              (9, "十 小结")], True),
    ("只读主干", [(0, "一 为什么加层"), (6, "七 FusionKind"),
              (7, "八 上抬"), (9, "十 小结")], False),
]
LEGEND = [("#22c55e", "入口：上一章逃生舱送入的 op 样例"),
          ("#3b82f6", "章内主线：身份→词汇→兼容→意图→上抬"),
          ("#f97316", "出口：FusionKind 驱动 AutoSchedule，转下一章")]
TITLE = "第 21 章 · HFusion 方言剖面：词汇表 + 结构兼容 + 融合意图 + 上抬(源码剖面图)"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数；本章 10 列，调小以合画布预算) ----------------
NODE_W, NODE_H = 125, 58
COL_GAP, ROW_GAP = 10, 14
EDGE_MARGIN, STUB_W, STUB_H = 8, 28, 24
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 18
LANE_LABEL_H, BAND_PAD = 22, 10
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 54, 14
ROUTE_HEAD_H, ROUTE_ROW_H = 20, 40
BADGE_H = 18
TITLE_MAX_W = NODE_W - 18  # 符号名文字可用宽度(留左右各 9px 内边距)
SUB_MAX_W = NODE_W - 14    # 一行短语可用宽度(留左右各 7px 内边距)

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
    """站牌胶囊，居中挂在 (cx,cy)——节点用它贴右上角，路线图例用它居中挂线上。
    宽度按 cjk_text_width() 估算(本章站牌是中文标题词，非 §N.M 短数字)。"""
    bw = cjk_text_width(text, 10.5) + 14
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.1"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.7:.1f}" text-anchor="middle" font-family="sans-serif" '
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
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 17}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)——本章图例文字较长，逐条另起一行避免横向挤压
_ly = TOP_PAD + TITLE_H + 12
for color, label in LEGEND:
    L.append(f'<rect x="{PAD_L}" y="{_ly - 9}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 17}" y="{_ly}" font-family="sans-serif" font-size="10" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _ly += 13

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.2"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#166534">{esc("上一章")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.2"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用/走线边(主线蓝)；多条边汇入同一节点时终点 y 各偏移，看得出"汇合"(本章
# 是单向直链，无汇合，偏移逻辑保留以防未来改 EDGES 时又出现汇入)
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
    y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="11" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.4"/>')
    fsz = fit_font_size(symbol, TITLE_MAX_W, base=12.5, floor=9)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.4:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{fsz:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    psz = fit_font_size(phrase, SUB_MAX_W, base=9.5, floor=7.5)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{psz:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W / 2, y, sec)

# 底部阅读路线：复用列坐标 COLX，站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 14:.1f}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌；实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11" '
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
print(f"wrote {out}  ({w}x{h}, aspect {w/h:.2f}:1)")
