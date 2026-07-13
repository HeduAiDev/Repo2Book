#!/usr/bin/env python3
"""第 37 章「本章地图」——DSpark 半自回归投机解码剖面图(前瞻 primer 章)。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge 胶囊样式/入口绿-出口橙-主线蓝配色/路线条
实线蓝-虚线灰/legend 必须画)原样保留，只改 DATA 与(为适配自然标题站牌文字
长度不一)badge() 的宽度算法(同 ch26 手法：按 cjk_text_width() 逐条计算宽度，
不用定长 BADGE_W 常量)。

本章为**前瞻 primer 章**+**自然标题章**(chapter.md 只有"一、二、三…七"与
"### 低秩/### 块内采样循环"这类叙事内小标题，无 `## N.M` 编号)——按契约禁用
§N.M 徽标，站牌改用标题词/文中黑体短语本身(如"锚点即首预测位"对应正文
"**其一，锚点即首个预测位**"、"对位昇腾"对应第六节收尾原话)。

本章额外的诚实约束(前瞻 primer 的核心论点)：并行骨干 + 序列 Markov 头是
**本 PR #46995 已落地**的代码；置信度头与硬件感知调度器 Algorithm 1 是
**仅论文/checkpoint 侧、本 PR 快照未接入推理路径**的机制。为了让"落地到哪"
这条落差在图上一眼可辨，本图在模板既有的三色语义(绿入口/蓝主线/橙出口)之外，
新增一种**复用既有 C_ROUTE_DIM 灰色**的第四语义：仅论文侧节点用虚线灰边框
(node 元组第 8 个字段 landed=False)，边也用同一灰色虚线(EDGES 第 3 个字段
style="dashed")——不引入任何模板未定义的新颜色，只是复用路线图例里本就有的
"次要虚线灰"语义，扩展到节点与调用边上，并在 LEGEND 里补一条说明。

节点预算 11(entry/cfg/backbone/spec_init/compute_logits/swa_kernel/
markov_head/sample_loop/confidence/algo1/exit) ≤ 12。

设计要点：
- 三条泳道对应章的三层现实：L0"装配期(构造一次，已落地)"=第六节讲的权重加载
  /配置分支 + 第二节骨干与 speculator 的类定义；L1"运行期已落地(骨干前向+
  序列修正)"=第二、三节的真正 forward 路径；L2"仅论文侧(未接入推理路径)"=
  第四、五节的置信度头与调度器——三层自上而下摆开，"已落地 vs 仅论文侧"的
  落差在纵轴上一眼可辨，不需要读完全文才明白代码到哪了。
- 所有调用边只跨相邻一列(entry/cfg→backbone/spec_init→compute_logits/
  swa_kernel/markov_head→sample_loop/confidence→exit/algo1)，避免长对角线
  穿过中间列的节点框(同 ch26 [FIX-ROUND-2] 踩过的坑)。
- markov_head 由 backbone→markov_head 一条边表达"骨干构造函数里挂载
  markov_head"(正文原话"整个类体只多了一个 markov_head")，不是虚构的调用
  关系而是真实的属性挂载。
- confidence 收到 compute_logits 与 markov_head 两条虚线边——对应论文公式
  c_k=sigmoid(w^T[h_k; W1[x_{k-1}]])真正需要的两路输入(h_k 来自骨干、
  W1[x_{k-1}]来自 markov 头)，用虚线明确标"论文侧設計如此，本 PR 未接线"，
  不是暗示这两路数据真的流向了一个不存在的模块。

六项自查记录(渲染→Read PNG 亲眼看后如实记录，见文件末尾 [SELF-CHECK] 注释)。
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
LANES = ["装配期(构造一次,已落地)", "运行期已落地(骨干前向 + 序列修正)", "仅论文侧(未接入推理路径)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(自然标题词/正文原话短语,禁用 §N.M), landed:True=已落地实线/False=仅论文侧虚线)
NODES = [
    ("entry",          0, 0, 0, "embed_tokens/lm_head",
     "草稿加载时别名共享,省显存保证词表一致", "落地对位", True),
    ("cfg",            0, 0, 1, 'method="dspark"',
     "parallel_drafting=True,强制并行骨干", "落地对位", True),
    ("backbone",       0, 1, 0, "Qwen3DSparkModel",
     "继承 DFlash 骨干,挂载 markov_head", "并行骨干", True),
    ("spec_init",      0, 1, 1, "DSparkSpeculator",
     "N 个锚点查询,dflash_causal=False", "锚点即首预测位", True),
    ("compute_logits", 1, 2, 0, "compute_logits",
     "骨干前向→基础 logits U_k(仅算一次)", "并行骨干", True),
    ("swa_kernel",     1, 2, 1, "noncausal_index_width",
     "块内非因果:看得见未来 query 位置", "非因果滑窗", True),
    ("markov_head",    1, 2, 2, "DSparkMarkovHead",
     "W1(V×r)+W2(V×r) 低秩转移偏置", "低秩转移偏置", True),
    ("sample_loop",    1, 3, 0, "_sample_sequential",
     "偏置修正→采样→prev 逐位递推", "采样循环", True),
    ("confidence",     2, 3, 0, "confidence_head",
     "load_weights 显式跳过,权重未接入", "置信度头(未接入)", False),
    ("exit",           1, 4, 0, "self.draft_tokens",
     "逐位写回,返回验证器/未来对位昇腾", "对位昇腾", True),
    ("algo1",          2, 4, 0, "Algorithm 1",
     "累计存活概率贪心早停,本 PR 无调度器", "调度(仅论文侧)", False),
]
NODE_BY_ID_TMP = {n[0]: n for n in NODES}

EDGES = [  # (src_id, dst_id, style) —— style: "solid"=已落地主线蓝, "dashed"=仅论文侧灰虚线
    ("entry", "backbone", "solid"),
    ("cfg", "backbone", "solid"),
    ("entry", "spec_init", "solid"),
    ("backbone", "compute_logits", "solid"),
    ("spec_init", "swa_kernel", "solid"),
    ("backbone", "markov_head", "solid"),
    ("compute_logits", "sample_loop", "solid"),
    ("markov_head", "sample_loop", "solid"),
    ("compute_logits", "confidence", "dashed"),
    ("markov_head", "confidence", "dashed"),
    ("confidence", "algo1", "dashed"),
    ("sample_loop", "exit", "solid"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序,列号须严格递增, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("已落地主线",
     [(0, "落地对位"), (1, "并行骨干"), (2, "低秩转移偏置"), (3, "采样循环"), (4, "对位昇腾")], True),
    ("仅论文侧(四/五节)",
     [(3, "置信度头(未接入)"), (4, "调度(仅论文侧)")], False),
]
LEGEND = [("#22c55e", "入口:草稿步触发/建模期装配进入"), ("#3b82f6", "章内主线:已落地调用/数据流"),
          ("#f97316", "出口:返回验证器/未来对位昇腾"), ("#94a3b8", "虚线:仅论文侧,本 PR 未接入推理路径")]
TITLE = "第 37 章 · DSpark 半自回归投机解码剖面(并行骨干+Markov头已落地 / 置信度头+调度器仅论文侧·前瞻)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#fef9f2"]  # 泳道背景交替,仅装饰,非语义色(第三道浅暖色呼应"仅论文侧")
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"  # 复用同一灰色表达"仅论文侧/次要"语义(节点虚线边、调用边虚线、路线虚线三处统一)

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 58
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
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
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌是变长中文词(如"置信度头(未接入)"7 字 vs "HCA"3 字母),不能用
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
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 26

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

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
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

# 调用边(先画边再画节点盖住端点毛刺):实线主线蓝 / 虚线仅论文侧灰
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点看不出"汇合"。
_dst_total = {}
for _, dst, _style in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst, style in EDGES:
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
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{stroke}" stroke-width="2" marker-end="url(#{marker})"{dash}/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌):已落地=实线深色边框,仅论文侧=虚线灰边框
for nid, lane, col, row, symbol, phrase, station, landed in NODES:
    x, y = NODE_XY[nid]
    stroke = C_NODE_STROKE if landed else C_ROUTE_DIM
    dash = '' if landed else ' stroke-dasharray="5,3"'
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_w(station)
    L += badge(x + NODE_W - bw / 2 + 8, y, station)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=仅论文侧次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
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

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录,共 2 轮渲染)
#   第 1 轮:claim_readable_10s=True numbers_match_spec=True(本图无数据数字,
#     N/A 记 True) no_overlap=False arrows_attached=True cjk_rendered=True
#     reading_order_clear=True —— lint_diagram_geometry.py 抓到:路线名
#     "已落地主线(装配→骨干前向→序列修正)"过长,与该路线第一个站牌"落地对位"
#     文字相撞(text-text 相撞 + tag-on-title 框内相压)。
#   第 2 轮修复:路线名精简为"已落地主线"/"仅论文侧(四/五节)"。重渲后 Read
#     PNG 复核:六项全 True——两条路线名不再与站牌重叠;col0→col1 的四条边
#     (entry/cfg→backbone/spec_init)在列间隙里交叉成 X 但箭头清晰各自贴住
#     目标框(已放大裁剪核对,无穿框/无文字压线);markov_head/compute_logits
#     →confidence 的两条虚线灰边在列间隙内陡峭下行但严格落在 col2→col3
#     的 x 范围内(不穿过 swa_kernel/markov_head 本身的节点框),箭头清晰
#     贴住 confidence_head 虚线框左侧;confidence→algo1 虚线边同样清晰;
#     仅论文侧两个节点(confidence_head/Algorithm 1)统一用灰色虚线圆角框
#     与已落地节点的实线深色框形成鲜明对比,"论文全貌 vs 代码到哪"的落差
#     一眼可辨;中文/公式记号(W1(V×r)+W2(V×r)、method="dspark"、
#     self.draft_tokens)渲染正常;阅读顺序(装配期→运行期已落地→仅论文侧,
#     自上而下;每层自左而右)清楚。lint_chapter_map.py 与
#     lint_diagram_geometry.py 均无问题(exit 0)。
#   §徽标/符号自查:本章为自然标题章(chapter.md 无 `## N.M`),图上无任何
#     §N.M 徽标,11 个站牌全部取自正文自然标题词或黑体短语原话("落地对位"
#     "并行骨干"对应"六、落地"/"二、并行骨干";"锚点即首预测位"对应正文
#     "**其一，锚点即首个预测位**";"非因果滑窗"对应"其二，块内非因果注意力"；
#     "低秩转移偏置"对应"### 低秩：省参数，不是近似 softmax"；"采样循环"
#     对应"### 块内采样循环"；"置信度头(未接入)"/"调度(仅论文侧)"对应
#     四、五两节标题的括注；"对位昇腾"取自第六节收尾原话)。11 个代码符号
#     (load_dspark_model/method="dspark"/Qwen3DSparkModel/DSparkSpeculator/
#     compute_logits/noncausal_index_width/DSparkMarkovHead/
#     _sample_sequential/confidence_head/Algorithm 1/self.draft_tokens)
#     逐一核对均为 chapter.md 正文原样子串,lint_chapter_map.py 的杜撰符号
#     检查同步确认无报告。
#
#   第 3 轮(writer 重构骨架后复绘,2026-07-14):writer 定稿把第六节"装配"段
#     改写为泛化叙述,不再逐字点名 load_dspark_model 这个函数名(dossier.json
#     里仍有,但本章 kind=primer——lint_chapter_map 的杜撰符号核对口径是
#     book/papers/ch37-primer-dspark/*.md + chapter.md,不含 dossier);
#     lint_chapter_map.py --require 报 fabricated_symbol:load_dspark_model。
#     entry 节点符号改为"embed_tokens/lm_head"(对应正文§六"共享 embed/lm_head
#     （别名，省显存…）"及 skip_substrs 代码块里的 embed_tokens/lm_head 两个
#     字面子串,均在 chapter.md 正文原样出现),phrase 同步改"草稿加载时别名
#     共享,省显存保证词表一致"。重渲后 Read PNG 复核:六项全 True——entry
#     节点新文字"embed_tokens/lm_head"未与右上角"落地对位"站牌重叠、与
#     backbone/spec_init 的两条交叉边仍清晰贴住目标框(裁剪核对无穿框);
#     其余 10 个节点/站牌未受影响,布局与第 2 轮一致。lint_chapter_map.py
#     --require 与 lint_diagram_geometry.py 均 exit 0。blind_review 因内容
#     变更已重置为 PENDING,待下一轮盲审回填。
