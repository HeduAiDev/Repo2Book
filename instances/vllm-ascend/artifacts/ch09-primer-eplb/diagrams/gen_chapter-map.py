#!/usr/bin/env python3
"""ch09-primer-eplb 本章地图:EPLB 负载均衡算法本体的源码/论文剖面图。

[2026-07-14 重绘] writer 把全章从 8 个子站(动机/目标/复制/装箱/折叠/映射/
编排/策略)重构收敛成 6 个自然标题分节(一~六),第五节把旧版的"折叠/映射/
编排"三站合并成一节"输出一张放置表",第六节把旧版的"策略"从"七"降编号为
"六"。本轮按新分节 1:1 重画,不再保留旧版三站同挂"五"的做法。

本章仍是自然标题章(无 `## N.M` 编号,只有"一、二、三…"中文数字标题),按契约
禁用 §N.M 徽标,站牌改用标题词本身。kind=primer,主线既有纯理论站(动机/下界,
用真实公式 `max_g L_g` 与真实观测组件 `EplbWorker` 挂实——`_compute_imbalance`
这个函数名旧版正文提过、新版正文已不点名,故本轮换成新版正文里仍点名的
`EplbWorker`,避免 lint_chapter_map 的杜撰符号检查落空)也有真实落地代码站
(`route_expert_redundancy` 摊薄递推、`rebalance_experts` 编排、`DefaultEplb`/
`PolicyFactory` 策略工厂)。

两行泳道(折成两行避免画布超宽):
  第一行"论证与算法核心"= 一动机→二下界→三削峰→四铺平
  第二行"落地编排"      = 五放置表→六策略
两行之间用一条"续行"折线连接(pack → output),不用直线穿过整个画布。

■ 不可变(同全书其余 chapter-map,不要动):
  1. §徽标胶囊配色(此章徽标文字非 §N.M,而是中文标题词,颜色/形状不变)。
  2. 入口=绿 #22c55e / 出口=橙 #f97316 接口桩。
  3. 节点间调用边(主线)= 蓝 #3b82f6。
  4. 路线条:高亮=实线蓝(粗) / 次要=虚线灰(细)。
  5. >2 种语义色画图例。
  6. cjk_text_width() 逐字符宽度估算,不用半角系数硬乘 len(s)。
■ 可变:本章把 BADGE_W 从模板默认 46 放宽到 60(中文站牌比 §N.M 数字宽);
  NODE_W/NODE_H 放宽到 200x70;真实符号名过长(如 route_expert_redundancy,
  23 字符)时符号行降字号到 10.5,避免撑破节点框(见 SYMBOL_FONT_SIZE)。

六项自查(渲染→Read PNG 亲眼看后记录,见文末字符串)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据,按 writer 定稿后的六节自然标题 1:1 重画) ----------------
LANES = ["论证与算法核心(一、动机 → 二、下界 → 三、削峰 → 四、铺平)",
         "落地编排(五、放置表 → 六、策略)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌)
NODES = [
    ("motivate", 0, 0, 0, "EplbWorker",
     "par=最热卡负载/平均，量化拖慢倍数", "一、动机"),
    ("bound", 0, 1, 0, "max_g L_g",
     "下界=ΣW/G；热度守恒，专家整块不可拆", "二、下界"),
    ("replicate", 0, 2, 0, "route_expert_redundancy",
     "摊薄递推(k+1)/(k+2)：锯最长木板", "三、削峰"),
    ("pack", 0, 3, 0, "LPT",
     "降序装箱填最轻卡，副本不共卡", "四、铺平"),
    ("output", 1, 0, 0, "rebalance_experts",
     "折叠热度→就地映射→0.95变更闸门", "五、放置表"),
    ("policy", 1, 1, 0, "DefaultEplb",
     "默认全局贪心，对照分层策略", "六、策略"),
]
# 行内直连边(同一泳道内,统一主线蓝)——跨行的 pack→output 走单独的折线,不放在这里
EDGES = [
    ("motivate", "bound"), ("bound", "replicate"), ("replicate", "pack"),
    ("output", "policy"),
]
WRAP_EDGE = ("pack", "output")  # 续行折线:第一行末→第二行首

# 符号文本超长时降字号,避免撑破节点框(route_expert_redundancy 23 字符)
SYMBOL_FONT_SIZE = {"replicate": 10.5}
DEFAULT_SYMBOL_FONT_SIZE = 13

# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 与正文开篇选读指引原句对齐:"只想抓…读一到四节就够;五节是…收尾,六节是…对照,可按需跳读"
ROUTES = [
    ("通读一~四：论证到核心", [(0, "一、动机"), (1, "二、下界"), (2, "三、削峰"), (3, "四、铺平")], True),
    ("五~六按需跳读:收尾/策略", [(0, "五、放置表"), (1, "六、策略")], False),
]
LEGEND = [("#22c55e", "入口:上一章→本章"), ("#3b82f6", "论证/落地主线"), ("#f97316", "出口:本章→下一章")]
TITLE = "EPLB 负载均衡算法本体：论证骨架→算法核心→落地编排"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(本章放宽 NODE_W/H、BADGE_W 以容纳中文站牌与较长符号别名) ----------------
NODE_W, NODE_H = 200, 70
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 60, 20

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

# 入口/出口接口桩:入口挂在 motivate(全图最左的第一站),出口挂在 policy(全图最末站)
ex, ey = NODE_XY["motivate"]; ey += NODE_H / 2
xx, xy = NODE_XY["policy"]; xy += NODE_H / 2
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

# 行内直连边(主线蓝)
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 续行折线(pack 第一行末 → output 第二行首):纯几何转弯,不穿过中间节点
wsrc, wdst = WRAP_EDGE
wx1, wy1 = NODE_XY[wsrc]; wx2, wy2 = NODE_XY[wdst]
w_bottom = wy1 + NODE_H
w_top = wy2
# 折线的水平段贴着目的地泳道的节点顶部走(而非两泳道中点),避开中点处的
# 泳道标题文字行——泳道标题贴在泳道带顶部,中点正好压在标题文字上。
gutter_y = w_top - 6
w_from_x = wx1 + NODE_W / 2
w_to_x = wx2 + NODE_W / 2
L.append(f'<path d="M {w_from_x:.1f},{w_bottom:.1f} L {w_from_x:.1f},{gutter_y:.1f} '
          f'L {w_to_x:.1f},{gutter_y:.1f} L {w_to_x:.1f},{w_top:.1f}" fill="none" '
          f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    fsize = SYMBOL_FONT_SIZE.get(nid, DEFAULT_SYMBOL_FONT_SIZE)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{fsize}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.70:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio={w/h:.2f})")

# ---------------- 六项自查(渲染 + Read PNG 后如实记录,见 illustrator 返回值) ----------------
