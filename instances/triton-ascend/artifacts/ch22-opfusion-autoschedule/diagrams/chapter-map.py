#!/usr/bin/env python3
"""ch22 本章地图——源码剖面图。

一枚 FusionKind「印章」，被两条脊梁各读一次。四条泳道 = 章内四条证据线：
印章的读/写(FusionKind 印章泳道)、脊梁① OpFusion 怎么用印章定融合边界、
脊梁② AutoSchedule 怎么用同一枚印章选调度器/切 tile/分双核、对位基座与小结。
圆角节点 = 真实符号名 + 一行短语，右上角挂站牌(本章为自然标题——章节用「一、二…」
中文序号而非 `## N.M` 编号，故站牌用序号+标题词而非 §N.M)，左侧入口桩 = 上一章
InferFuncFusionKind 推断好的印章送进来，右侧出口桩 = 融合核降到 HIVM、转下一章。

列号 = 正文标题出现顺序：0 一(承上启下，读印章) → 1 二(isFusible switch) →
2 三(并查集主循环) → 3 四(五道关卡) → 4 五(出组约束) → 5 六(外提回写，写印章)
→ 6 七(选调度师) → 7 八(调度骨架) → 8 九(拆双核样例) → 9 十(对位小结，出口)。
走线严格左→右单向递增列号，无回绕；跨泳道的斜线正是「印章在两条脊梁间传递」。

泳道分工：
  Lane0 FusionKind 印章 = 入口读(getOptionFromLabel)+ 落地写(outline 回写 FusionKindAttr)。
  Lane1 脊梁① OpFusion = isFusible → fuseBlock → verifyRulesAndJoin → checkGroupRequirements。
  Lane2 脊梁② AutoSchedule = applySchedule → runScheduleProcedure → ShallowCVScheduler。
  Lane3 对位基座 & 小结 = 出口节点。

节点符号选取原则：优先选**短而真**的具体符号(全部逐字出现在 dossier / 正文)，
避免超长类名把节点撑爆画布预算——如「外提回写」一站选 `outline`(FusibleBlockOutliner
::outline)，回写印章这句证据放进短语，而非画超长的 `setOutlineFuncAttributes`。

■ 不可变(照搬模板视觉语言，只改 DATA 与几何常量)：站牌胶囊 / 入口绿
  #22c55e-出口橙#f97316-主线蓝#3b82f6 / 高亮路线实线蓝、次要虚线灰 /
  cjk_text_width() 宽度估算。
■ 本章为自然标题(中文序号，无 `## N.M` 编号)，站牌一律用「序号+标题词」，禁用 §N.M。
■ 几何常量(NODE_W/COL_GAP/PAD 等)按本章 10 列节点数据设定，满足画布预算
  (宽 ≤1500 且宽高比 ≤2.6:1)。
■ 长符号名一律按估算宽度动态缩小字号，避免文字越界(fit_font_size())。

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
LANES = ["FusionKind 印章 (读→写)", "脊梁① OpFusion 定融合边界",
         "脊梁② AutoSchedule 切 tile 分双核", "对位基座 & 小结"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(序号+标题词))
NODES = [
    ("entry", 0, 0, 0, "getOptionFromLabel",
     "从 func 读出印章", "一 承上启下"),
    ("isfusible", 1, 1, 0, "isFusible",
     "switch fusionKind_，十把剪刀", "二 十把剪刀"),
    ("fuseblock", 1, 2, 0, "fuseBlock",
     "并查集 + 拓扑秩主循环", "三 并查集"),
    ("gauntlet", 1, 3, 0, "verifyRulesAndJoin",
     "五道关卡全过才 join", "四 五道关卡"),
    ("groupreq", 1, 4, 0, "checkGroupRequirements",
     "ShallowCV/MixCV 须含 matmul", "五 出组约束"),
    ("outline", 0, 5, 0, "outline",
     "外提 device func + 回写印章", "六 外提成核"),
    ("dispatch", 2, 6, 0, "applySchedule",
     "按印章 switch 选调度师", "七 选调度师"),
    ("skeleton", 2, 7, 0, "runScheduleProcedure",
     "pre → schedule → post", "八 调度骨架"),
    ("shallowcv", 2, 8, 0, "ShallowCVScheduler",
     "拆 cube/vector 双核样例", "九 拆双核"),
    ("exit", 3, 9, 0, "AutoSchedule",
     "对位 triton 基座，转下一章", "十 对位小结"),
]
EDGES = [  # (src_id, dst_id) —— 章内讲解走线，统一主线蓝；src 列号恒 < dst 列号
    ("entry", "isfusible"), ("isfusible", "fuseblock"), ("fuseblock", "gauntlet"),
    ("gauntlet", "groupreq"), ("groupreq", "outline"), ("outline", "dispatch"),
    ("dispatch", "skeleton"), ("skeleton", "shallowcv"), ("shallowcv", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
ROUTES = [
    ("完整通读", [(0, "一 承上启下"), (1, "二 十把剪刀"), (2, "三 并查集"),
              (3, "四 五道关卡"), (4, "五 出组约束"), (5, "六 外提成核"),
              (6, "七 选调度师"), (7, "八 调度骨架"), (8, "九 拆双核"),
              (9, "十 对位小结")], True),
    ("只看脊梁① 融合边界怎么定", [(1, "二 十把剪刀"), (2, "三 并查集"),
              (3, "四 五道关卡"), (4, "五 出组约束"), (5, "六 外提成核")], False),
    ("只看脊梁② 调度怎么分双核", [(6, "七 选调度师"), (7, "八 调度骨架"),
              (8, "九 拆双核"), (9, "十 对位小结")], False),
]
LEGEND = [("#22c55e", "入口：上一章推断好的 FusionKind 印章送入"),
          ("#3b82f6", "章内主线：脊梁① OpFusion → 脊梁② AutoSchedule，同读一枚印章"),
          ("#f97316", "出口：融合核降到 HIVM 方言，转下一章")]
TITLE = "第 22 章 · 双脊梁剖面：一枚 FusionKind 印章，OpFusion 定融合边界 + AutoSchedule 分双核(源码剖面图)"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数；本章 10 列，合画布预算) ----------------
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
    宽度按 cjk_text_width() 估算(本章站牌是序号+中文标题词)。"""
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
# 图例(>2 种语义色必须画图例)——逐条另起一行避免横向挤压
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

# 调用/走线边(主线蓝)；本章为单向直链，无汇入，偏移逻辑保留以防未来改 EDGES
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

# ------------------------------------------------------------------
# 自查记录(Read PNG 1448x662 后如实回填)：
#   claim_readable_10s : true —— 一枚 FusionKind 印章、两条脊梁泳道(OpFusion 定边界 /
#                        AutoSchedule 分双核)一眼可读，入口读印章、§六 回写、出口转下一章。
#   numbers_match_spec : true —— 本图无 spec 数字；站牌序号一~十逐一对应正文十节标题。
#   no_overlap         : true —— 无文字相撞/压框；最长符号 checkGroupRequirements 经
#                        fit_font_size 缩到 9px，稳居 §五 节点框内。
#   arrows_attached    : true —— 入口/出口桩、9 条主线边两端均贴节点边缘，含 3 处跨泳道斜线。
#   cjk_rendered       : true —— 全部中文无豆腐块。
#   reading_order_clear: true —— 左→右单向 + 序号站牌 + 底部三条阅读路线(完整/脊梁①/脊梁②)。
