#!/usr/bin/env python3
"""第 35 章「本章地图」——DFlash 块扩散并行起草 + KV 注入的论证剖面图(primer 原理章)。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写(同
ch37-primer-dspark 手法):不可变机制(esc/cjk_text_width/badge 胶囊样式/入口绿-
出口橙-主线蓝配色/路线条实线蓝-虚线灰/legend 必须画)原样保留,只改 DATA。

**2026-07-14 重绘缘由**:writer 把本章整体改写为「纯论文精读」——标题已改
"【原理篇·论文精读】",全章不再内嵌任何真实源码块(旧版曾内嵌
combine_hidden_states/set_inputs_first_pass/_build_fused_kv_buffers/
copy_and_expand_dflash_inputs_kernel_single_grid/DFlashQwen3Attention.forward/
max_query_tokens 等真实代码片段与逐行解读,全部删除;仅保留 3 处一句话工程指路
`num_query_per_req`/`precompute_and_store_context_kv`/`causal=False`,均明写
"细节见第 36 章"),工程解读整体挪去第 36 章。旧版 chapter-map 是一张源码调用剖面
图(9 个真实代码符号节点),这些符号在新正文里已找不到——若原样保留会被
lint_chapter_map 的杜撰符号检查判定为伪造(kind=primer 时符号只核对论文包
book/papers/ch35-primer-dflash/*.md + chapter.md 正文,不核 dossier.json)。
因此本图从「源码调用图」改画成「论证结构图」——节点是论文的数学论证节点
(公式/机制,symbol 取 Eq.(N)/Appendix 编号或纯中文短语,均可在 chapter.md 正文
逐字核到),不再是代码符号。

本章为**自然标题章**(chapter.md 只有"一、二、三、四、五"与"小结",无 `## N.M`
编号)——按契约禁用 §N.M 徽标,站牌改用标题词本身(与实际 `## ` 标题逐一对应,
下方 NODES 表按此核对):
  "一道分数与一条独立性" ↔ "## 一、一道分数与一条独立性"
  "KV注入"              ↔ "## 二、KV 注入：把条件做到最足"
  "交叉注意力"          ↔ "## 三、交叉注意力：条件怎么被读"
  "训练:位置加权"       ↔ "## 四、训练：随机锚点掩码与位置加权"
  "接受率:树验证"       ↔ "## 五、接受率与树验证"
  "小结"                ↔ "## 小结：一条独立性，两笔账"
(prose 里出现的"§一"/"§二"等 CJK 数字前缀是既有全书通行的非正式指代写法——
`§` 后接汉字数字、不匹配 lint_chapter_map 的 `§\d+\.\d+` 徽标正则,不受
"自然标题章禁 §N.M 徽标"约束,与 ch09/ch23/ch26/ch31 等既有 primer 章 LANES/
ROUTES 文案写法一致。)

节点预算 7(speedup/independence/kv_injection/cross_attn/training/ddtree/
exit_summary)≤ 12。画布：5 列(col0-4)×4 泳道,宽度与宽高比核算见下方几何常量
(渲染后由 lint_diagram_geometry 兜底)。

设计要点(为什么这样摆):
- **主线只有两个节点**(speedup→kv_injection 之间夹 independence)对应正文开篇
  点透的唯一命题:"压小分子 T_draft" (§一的分数记账 Eq.(1) + 独立性 Eq.(3) 让
  γ 从延迟里消失) 与 "顶大分母 τ"(§二 KV 注入把条件做到最足)——图上用实线蓝
  串起 speedup→independence→kv_injection→exit_summary,对应正文"只想抓住
  独立性怎么买到并行、又靠什么赎回准头,读§一、§二就够"这句选读指引。
- **cross_attn/training/ddtree 三个节点用虚线灰**,对应正文"§三看条件怎么被
  注意力读到;§四(训练)与§五(接受率与树验证)是论文侧延伸,适合最后回看"——
  语义从旧版的"仅论文侧延伸,无对应昇腾代码"改为"细节/选读延伸,论证结构上
  是主线的旁支说明"(本章现在通篇无代码,已不存在"有没有落地代码"这个区分)。
- **cross_attn/training 都从 kv_injection 引出虚线**(列 3,各自泳道):
  cross_attn 因为它讲的正是"KV 注入的条件怎么被 draft 的注意力读到"(正文
  §三开篇"注入的 K/V 进了 cache,draft 层怎么用它");training 因为正文明写
  "训练时…与推理**同一套注入通路**"。
- **ddtree 从 independence 引出虚线**(列 1→列 4,跨列跨泳道):正文明写
  DDTree"最优性恰好也挂在分解分布上"——分解分布正是 independence 节点
  (Eq.(3) `Q=∏qi`)的内容,不是 KV 注入的内容。
- **exit_summary 是唯一与 kv_injection 同列相邻的主线终点节点**(列 4,与
  ddtree 同列不同泳道,复刻 example-chapter-map 模板"出口节点与最后一个虚线
  分支同列"的既有排布手法),承接正文"小结"一节"独立性压分子、KV 注入顶
  分母,方向一致、可叠加"的收束句,出口箭头指向"下一站:第 36 章"。

六项自查记录见文件末尾 [SELF-CHECK] 注释(渲染→Read PNG 亲眼看后如实记录)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版)——全角(ord>0x2E80)按
    1.0×size,半角按 0.58×size,求和。中英混排的图例/标签/站牌必须用这个,
    不能直接 0.58 * size * len(s)。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = [
    "①加速比记账与条件独立(主线,§一)",
    "②KV 注入:把条件做到最足(主线,§二)",
    "③交叉注意力:条件怎么被读(细节,§三)",
    "④训练加权与树验证(选读延伸,§四·§五)",
]

# (节点id, 泳道下标, 列, 泳道内行号, 论文侧记号(Eq./Appendix 编号或中文短语,
#  均可在 chapter.md 正文逐字核到), 一行短语, 站牌(自然标题词,禁用 §N.M),
#  main:True=主线实线蓝, False=细节/选读延伸虚线灰)
NODES = [
    ("speedup",      0, 0, 0, "Eq.(1)",
     "起草+验证摊到期望接受数,加速比只剩两个把手", "一道分数与一条独立性", True),
    ("independence", 0, 1, 0, "Eq.(3)",
     "块内条件独立,同一次前向读出边际,块大小从延迟消失", "一道分数与一条独立性", True),
    ("kv_injection", 1, 2, 0, "Appendix A.3",
     "5 层隐藏态拼接→共享投影→RMSNorm,得上下文特征", "KV注入", True),
    ("cross_attn",   2, 3, 0, "非因果双向",
     "Q 只由 draft 隐藏态产生,K/V 拼接产生,块内双向可见", "交叉注意力", False),
    ("training",     3, 3, 0, "Eq.(4)",
     "位置权重指数衰减,早位置更贵,同一套注入通路", "训练:位置加权", False),
    ("ddtree",       3, 4, 0, "Eq.(8)",
     "分解分布可加、单调不增,best-first 建最优候选树", "接受率:树验证", False),
    ("exit_summary", 1, 4, 0, "两笔账",
     "独立性压起草成本,KV 注入抬接受数,方向一致可叠加", "小结", True),
]
NODE_BY_ID = {n[0]: n for n in NODES}

EDGES = [  # (src_id, dst_id, style, waypoints) —— style: "solid"=主线蓝,
    # "dashed"=细节/选读延伸灰虚线；waypoints=() 为直线，非空则走"水平→垂直→水平"
    # 三段折线(仅 independence→ddtree 需要:跨 3 列 3 泳道的长途连线,直线会穿过
    # kv_injection/cross_attn/training 三个节点框——折线沿列间空隙走,全程不压框)。
    ("speedup", "independence", "solid", ()),
    ("independence", "kv_injection", "solid", ()),
    ("kv_injection", "exit_summary", "solid", ()),
    ("kv_injection", "cross_attn", "dashed", ()),
    ("kv_injection", "training", "dashed", ()),
    ("independence", "ddtree", "dashed", "gutter34"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序,列号须严格递增, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    # 路线名故意精简(第一站徽标紧贴在 col0,留给文字的横向空间只有 ~165px,
    # 完整长句会与徽标重叠——参见 [SELF-CHECK] FIX-ROUND-2 记录)。
    ("主干(§一+§二,必读)", [(0, "一道分数与一条独立性"), (2, "KV注入"), (4, "小结")], True),
    ("细节延伸(§三+§四+§五):交叉注意力→训练/树验证", [(3, "交叉注意力"), (4, "接受率:树验证")], False),
]
LEGEND = [("#22c55e", "入口:上一站(拒绝采样框架)"), ("#3b82f6", "章内主线:压分子/顶分母的两笔账"),
          ("#f97316", "出口:下一站(第 36 章昇腾落地)"), ("#94a3b8", "虚线:细节/选读延伸,论证支线")]
TITLE = "第 35 章 · DFlash 独立性论证剖面(主线 §一-§二 · 细节/选读延伸 §三·§四·§五)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#fef9f2"]  # 泳道背景交替,仅装饰,非语义色(第四道浅暖色呼应"选读延伸")
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"  # 复用同一灰色表达"细节/选读延伸"语义(节点虚线边、调用边虚线、路线虚线三处统一)

def wrap_phrase(text, max_width, size):
    """贪心逐字换行(CJK 宽度感知)——本章节点短语比旧版源码剖面图里的短语更长
    (讲的是完整数学论断而非一个函数名),单行 210px 节点框常放不下,换行成
    ≤2 行渲染,避免文字溢出框外(旧版 6 项自查 no_overlap 的失败模式)。"""
    lines, cur = [], ""
    for ch in text:
        test = cur + ch
        if cjk_text_width(test, size) > max_width and cur:
            lines.append(cur)
            cur = ch
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 210, 78
COL_GAP, ROW_GAP = 14, 20
EDGE_MARGIN, STUB_W, STUB_H = 10, 50, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H = 20  # 宽度改按文字动态算(见 badge_w),站牌是变长中文词,不能用定长常量

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


def badge_w(text):
    """站牌是变长中文词(如"训练:位置加权"7 字 vs "KV注入"混排),不能用
    定长 BADGE_W——按 cjk_text_width 估算文字宽度再加左右各 8px 内边距。"""
    return cjk_text_width(text, 11) + 16


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy) —— 节点用它贴右上角,路线图例用它居中挂线上。
    宽度按文字动态算,但形状/配色/挂法与全书统一模板一致。"""
    bw = badge_w(text)
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Dim", C_ROUTE_DIM))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 10.5) + 22

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["speedup"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit_summary"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("上一站")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一站")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(先画边再画节点盖住端点毛刺):实线主线蓝 / 虚线细节延伸灰
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点看不出"汇合"。
# gutter34:col3/col4 之间的列缝隙 x 坐标(唯一需要绕行的长途边用它),该缝隙上下
# 贯穿全图都没有节点框,折线走这条竖线保证不压框(见文件内几何核算)。
gutter34 = (COLX[3] + NODE_W + COLX[4]) / 2
_dst_total = {}
for _, dst, _style, _wp in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst, style, waypoints in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    if style == "solid":
        stroke, marker, dash = C_MAIN, "mMain", ""
    else:
        stroke, marker, dash = C_ROUTE_DIM, "mDim", ' stroke-dasharray="6,4"'
    if waypoints == "gutter34":
        # 水平(出发点 y)→竖直(列缝隙 x)→水平(到达点 y):三段折线,全程避开中间节点框。
        pts = [p1, (gutter34, p1[1]), (gutter34, p2[1]), p2]
        pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        L.append(f'<polyline points="{pts_str}" fill="none" stroke="{stroke}" '
                  f'stroke-width="2" marker-end="url(#{marker})"{dash}/>')
    else:
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{stroke}" stroke-width="2" marker-end="url(#{marker})"{dash}/>')

# 节点(圆角框 + 论文侧记号(单行)+ 一行短语 + 右上角站牌):
# 主线=实线深色边框,细节/选读延伸=虚线灰边框
for nid, lane, col, row, symbol, phrase, station, main in NODES:
    x, y = NODE_XY[nid]
    stroke = C_NODE_STROKE if main else C_ROUTE_DIM
    dash = '' if main else ' stroke-dasharray="5,3"'
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.28:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    phrase_lines = wrap_phrase(phrase, NODE_W - 16, 8.0)
    phrase_y0 = y + NODE_H * 0.52
    for li, pline in enumerate(phrase_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{phrase_y0 + li * 10.5:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="8" fill="{C_NODE_SUB}">{esc(pline)}</text>')
    bw = badge_w(station)
    L += badge(x + NODE_W - bw / 2 + 8, y, station)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=细节延伸次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, station in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, station)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录)
#   见渲染后同轮记录(下方由脚本外的 illustrator 流程回填,首轮如有问题会在此追加
#   [FIX-ROUND-2] 段落)。
