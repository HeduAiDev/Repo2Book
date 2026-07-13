#!/usr/bin/env python3
"""第 27 章「本章地图」——Lightning Indexer 论文主线五段论(概念剖面图)。

writer 把本章重构成纯「原理篇·论文精读」:正文不再内嵌/引用任何 vLLM 源码符号
(self.indexer/wq_b/fp8_fp4_mqa_logits/...全部删除),只讲论文的数学与证明。上一版
chapter-map 画的是"推理调用链"(源码剖面图),现在与正文脱节——本版改画论文主线
的五段论证链,站牌全部换成正文实际标题词/小节标题/严谨callout短语。

本章是自然标题章(节标题为中文数字「一、二、三、四」,无 `## N.M` 编号)——按契约
禁用 §N.M 徽标,站牌一律用正文里实际出现的标题词/callout 短语本身。

结构:上泳道是主线论证链(自左向右——打分函数 Eq.(1) → top-k 闸门 Eq.(2) → 复杂度
诚实账 O(L²)→O(Lk) → KL 对齐 Eq.(3)(4) → 独立缓存与量化);下泳道是撑住每一站的
严谨性证明/不变量(各自垂直挂在它所支撑的上泳道节点正下方,对应正文里的"不变量"
小标题或"严谨(...)"callout)。

■ 不可变(全书统一视觉语言,来自 skill 模板,换章节时不要动):入口绿#22c55e-出口
  橙#f97316-主线蓝#3b82f6/图例规则/cjk_text_width()/节点圆角框样式。
■ 本章新增(相对模板的必要扩展,非任意发挥):
  1) 同列=纵向支撑边、异列=横向主线边的分流(继承自上一版,模板原生只处理同泳道
     左右调用边)。
  2) badge() 胶囊宽度从模板固定的 46px(为 §13.2 这类短数字设计)改成按文字实测
     宽度(cjk_text_width + 左右各 8px 内边距)动态撑开,下限仍是 46px——本章站牌
     是"独立缓存与量化""不变量:选择正确性"这类 6~9 字的自然语言短语,固定 46px
     会被文字撑破胶囊边框(压框),这是自然标题章必然遇到的问题,不是任意发挥。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录,详见 figure-manifest.json):
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):全角(ord>0x2E80)按 1.0×size,
    半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["主线论证(打分 → 选块 → 复杂度账 → 可信 → 续命)", "严谨支撑(不变量 / 证明 callout)"]

# (节点id, 泳道下标, 列, 泳道内行号, 数学记号, 两行短语(用|分行), 站牌——自然标题词/callout短语,禁用 §N.M)
NODES = [
    ("score",  0, 0, 0, "Eq.(1)",            "三刀砍出打分算子|ReLU 截负,砍 value/输出",  "打分函数"),
    ("topk",   0, 1, 0, "Eq.(2)",            "分数不出闸,只出名单|空槽填 -1",             "top-k 闸门"),
    ("cplx",   0, 2, 0, "O(L^2) -> O(Lk)",   "indexer 打分仍 O(L^2)|主注意力降到 O(Lk)",  "复杂度诚实账"),
    ("kl",     0, 3, 0, "Eq.(3)(4)",         "拿主注意力分布|当 indexer 的标准答案",      "KL 对齐"),
    ("cache",  0, 4, 0, "IndexCache/MXFP4",  "K^IComp 与 C^Comp|并行产出,各写各缓存",     "独立缓存与量化"),

    ("relu_b", 1, 0, 0, "max(x,0) >= 0",     "总分有下界 0|负相关只截零,不倒扣",          "严谨:单调性"),
    ("sel_c",  1, 1, 0, "argsort",           "分数全序|并列时索引小者优先",               "严谨:选择正确性"),
    ("gain_c", 1, 2, 0, "L/(2k)",            "稀疏<=稠密恒成立|收益随 L 线性放大",         "不变量:收益守恒"),
    ("kl_leg", 1, 3, 0, "KL 散度",           "两侧非负和为 1|先稠密热身,后稀疏收窄",       "严谨:KL 为何用得起来"),
    ("indep",  1, 4, 0, "K^IComp/C^Comp",    "两块缓存各自分配|互不引用",                 "独立性:物理不是逻辑"),
]
EDGES = [  # (src_id, dst_id) —— 同列 = 纵向支撑边,异列 = 横向主线边(见文件头说明)
    ("score", "topk"), ("topk", "cplx"), ("cplx", "kl"), ("kl", "cache"),
    ("score", "relu_b"), ("topk", "sel_c"), ("cplx", "gain_c"), ("kl", "kl_leg"), ("cache", "indep"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("打分+复杂度账(一/二节)", [(0, "打分函数"), (1, "top-k 闸门"), (2, "复杂度诚实账")], True),
    ("只关心可信(三节)",       [(3, "KL 对齐")], False),
    ("只关心续命(四节)",       [(4, "独立缓存与量化")], False),
]
LEGEND = [("#22c55e", "入口:论文主线命题"), ("#3b82f6", "主线论证 / 严谨支撑"), ("#f97316", "出口:落地到模型架构章")]
TITLE = "第 27 章 · Lightning Indexer 论文主线:打分 → 选块 → 复杂度账 → 可信 → 续命"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 178, 84
COL_GAP, ROW_GAP = 22, 20
EDGE_MARGIN, STUB_W, STUB_H = 14, 68, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
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
    """站牌胶囊按文字实测宽度撑开(下限 46px,与模板短数字 badge 视觉一致)——见
    badge() 文档。"""
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + 16)


def badge(cx, cy, text):
    """站牌胶囊(本章为自然标题,文字是标题词/callout 短语而非 §N.M),居中挂在 (cx,cy)。

    宽度按文字实测宽度动态撑开(下限 46px,与模板短数字 badge 视觉一致)——模板
    固定 46px 是为 "§13.2" 这类 5 字符数字设计,本章站牌是"独立缓存与量化"这类
    6~9 字自然语言短语,固定宽度会压框,故加此扩展(其余样式——胶囊圆角/配色/
    字号——原样继承模板,不变)。
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

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"论证从论文命题出发/
# 最终落地到落地章")
ex, ey = NODE_XY["score"]; ey += NODE_H / 2
xx, xy = NODE_XY["cache"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("主线命题")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("模型架构章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 主线边/支撑边(统一主线蓝):列相同 → 纵向支撑边(上泳道节点正下方挂一块严谨
# 支撑);列不同 → 横向主线边(右中→左中)。
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

# 节点(圆角框 + 数学记号 + 两行短语 + 右上角站牌)。短语用 "|" 分两行,避免长中文
# 解释在单行里撑破节点宽度、压到相邻节点(自然标题章的站牌/短语天然比 §N.M 数字长)。
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.28:.1f}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    phrase_lines = phrase.split("|")
    for pi, pl in enumerate(phrase_lines):
        py = y + NODE_H * (0.55 + pi * 0.23)
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{py:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">{esc(pl)}</text>')
    bw = badge_width(sec)
    L += badge(x + NODE_W + 8 - bw / 2, y, sec)  # 右边缘固定在 x+NODE_W+8,向左铺开 bw

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
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
