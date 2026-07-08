#!/usr/bin/env python3
"""第 26 章「本章地图」——DeepSeek-V4 混合注意力(CSA/HCA)装配剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge 胶囊样式/入口绿-出口橙-主线蓝配色/路线条
实线蓝-虚线灰/legend 必须画)原样保留，只改 DATA 与(为适配自然标题站牌文字
长度不一)badge() 的宽度算法——模板的 BADGE_W 是给 "§20.3" 这种定长短标签用的
常量，本章站牌是"落地装配""运行期近似"这类变长中文词，改成按
cjk_text_width() 逐条计算宽度，宽度变化只影响徽标向左侧的延展(向右侧突出
量固定为 8px，见 badge_cx 的算法)，不影响相邻列节点的碰撞安全边界。

本章为自然标题章(chapter.md 无 `## N.M` 编号标题，只有"一、二、三、四"与
"2.1/2.2..."这类叙事内编号)——按契约禁用 §N.M 徽标，站牌改用标题词本身
(如"CSA压缩"对应"### 2.1　CSA 压缩：..."一节)。

节点预算 9(entry/dispatch/csa_compress/hca_compress/csa_select/kvcomp_select/
hca_pass/grouped_output/exit) ≤ 12。

设计要点：
- 这是一张"装配剖面图"而非纯运行期调用栈——entry(DeepseekV4Attention 构造)
  与 dispatch(get_dsv4_compress_ratio 读开关表)发生在建模期(每层构造一次)，
  grouped_output/exit 则是运行期真正执行的 forward 路径；这正是第四节"落地：
  这套数学在 vllm_ascend 里怎么装配"本身讲的内容——静态开关表决定了运行期
  走哪条路径，图跟着章的叙事顺序走，不强行伪装成一次性的纯运行期调用链。
- csa_compress 与 hca_compress 是同一个真实类 Compressor 的两个不同构造实例
  (compress_ratio=4 时 overlap=True、compress_ratio=128 时 overlap=False)——
  两个节点共用同一符号名 "Compressor"，用短语文字区分参数，忠实于正文 2.3
  节"同一个 Compressor，靠 coff 分叉"的原话，不杜撰两个不存在的类名。
- csa_select(Indexer，论文算法定义的 lightning indexer top-k)、kvcomp_select
  (HashEncoder，运行期工程近似)、hca_pass(HCA 稠密:不挑块)三者并列同一
  "选块"泳道——对应正文 4.3 节"KVComp 落地:LSH hash + Hamming top-k 近似
  indexer 选块"这条支线关系，以及 2.3 节"HCA 对全部压缩块做稠密 MQA、不做
  top-k"。[FIX-ROUND-2] 最初把 hca_compress 直接一条边跨两泳道连到
  grouped_output(省掉 hca_pass)，渲染后 Read PNG 发现:这条长对角线在
  几何上必然穿过"选块"泳道那一整列(indexer/hashencoder 两行摆满了该列
  的全部高度，任何跨两泳道的直线都会在列宽范围内切过其中一行的框)——
  实测穿过了 HashEncoder 的节点框，制造了一条视觉上不存在的"HCA→
  HashEncoder"关系。改法:给 HCA 补一个同泳道第三行的真实节点(HCA 不挑、
  直接稠密 MQA，是正文明确描述的行为，不是杜撰)，让每条边只跨相邻一个
  泳道，彻底消除穿框。
- grouped_output 的真实符号取 "wo_a / wo_b"(4.1 节 L774-789 嵌入源码的分组
  输出投影线性层)而非虚构一个"核注意力()"函数——wo_a/wo_b 是本章唯一在
  嵌入代码块里出现、且明确代表"MQA 之后"这一步的真实符号，比编造一个查
  无实据的 forward 符号更可核。

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
LANES = ["建模期装配", "序列压缩(Compressor)", "选块:算法 vs 运行期近似", "输出与残差包裹"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(自然标题词,禁用 §N.M))
NODES = [
    ("entry",          0, 0, 0, "DeepseekV4Attention",
     "按层号读开关表分流", "落地装配"),
    ("dispatch",       0, 1, 0, "get_dsv4_compress_ratio",
     "4/128/0 三态开关", "交错互补"),
    ("csa_compress",   1, 2, 0, "Compressor",
     "重叠压缩(m=4)", "CSA压缩"),
    ("hca_compress",   1, 2, 1, "Compressor",
     "不重叠压缩(m'=128)", "HCA"),
    ("csa_select",     2, 3, 0, "Indexer",
     "ReLU打分+top-k选块", "CSA稀疏"),
    ("kvcomp_select",  2, 3, 1, "HashEncoder",
     "hash指纹+Hamming选块", "运行期近似"),
    ("hca_pass",       2, 3, 2, "稠密 MQA (全部块)",
     "不挑,不做top-k", "HCA稠密"),
    ("grouped_output", 3, 4, 0, "wo_a / wo_b",
     "MQA+滑窗+sink 后分组输出投影", "分组输出投影"),
    ("exit",           3, 5, 0, "DeepseekV2DecoderLayer",
     "mHC包裹,forward输出", "mHC包裹"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;每条边只跨相邻一个泳道,
           # 避免长对角线穿过中间泳道整列节点(见上方 [FIX-ROUND-2] 说明)
    ("entry", "dispatch"),
    ("dispatch", "csa_compress"), ("dispatch", "hca_compress"),
    ("csa_compress", "csa_select"), ("csa_compress", "kvcomp_select"),
    ("hca_compress", "hca_pass"),
    ("csa_select", "grouped_output"), ("kvcomp_select", "grouped_output"),
    ("hca_pass", "grouped_output"),
    ("grouped_output", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("CSA 全链路",
     [(0, "落地装配"), (1, "交错互补"), (2, "CSA压缩"), (3, "CSA稀疏"), (4, "分组输出投影"), (5, "mHC包裹")], True),
    ("HCA 全链路",
     [(0, "落地装配"), (1, "交错互补"), (2, "HCA"), (3, "HCA稠密"), (4, "分组输出投影"), (5, "mHC包裹")], False),
    ("运行期近似(KVComp)",
     [(2, "CSA压缩"), (3, "运行期近似"), (4, "分组输出投影")], False),
]
LEGEND = [("#22c55e", "入口:上层模型逐层装配/调用进入"), ("#3b82f6", "章内主线:装配→压缩→选块→分组输出"),
          ("#f97316", "出口:mHC 包裹后返回上层")]
TITLE = "第 26 章 · DeepSeek-V4 混合注意力装配剖面(CSA/HCA 分流 + KVComp 运行期近似)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 58
COL_GAP, ROW_GAP = 20, 20
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
    """站牌是变长中文词(如"运行期近似"5 字 vs "HCA"3 字母),不能用定长
    BADGE_W——按 cjk_text_width 估算文字宽度再加左右各 8px 内边距。"""
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

# 调用边(主线蓝,先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点看不出"汇合"。
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
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)
for nid, lane, col, row, symbol, phrase, station in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_w(station)
    L += badge(x + NODE_W - bw / 2 + 8, y, station)

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
    for col, station in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, station)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录,共 3 轮渲染)
#   第 1 轮:claim_readable_10s=True numbers_match_spec=True(本图无数据数字,
#     N/A 记 True) no_overlap=False arrows_attached=True cjk_rendered=True
#     reading_order_clear=True —— Read PNG 发现两处问题:①底部路线名过长
#     (如"CSA 全链路(压缩→算法选块→输出)")与该路线第一个站牌"落地装配"
#     视觉重叠;②hca_compress 跨两条泳道直连 grouped_output 的对角线,几何上
#     穿过了"选块"泳道整列(indexer/hashencoder 摆满该列全部高度),实测
#     穿过了 HashEncoder 节点框,制造了一条视觉上不存在的"HCA→HashEncoder"
#     关系。lint_chapter_map.py 同轮还抓到符号 "MQA(" 杜撰(短语里
#     "MQA(全部块)" 无空格导致连读成一个不存在的 token)。
#   第 2 轮修复:①路线名精简为"CSA 全链路"/"HCA 全链路"/"运行期近似
#     (KVComp)";②新增 hca_pass 节点("稠密 MQA (全部块)",第三行同泳道),
#     把 hca_compress→grouped_output 拆成 hca_compress→hca_pass→
#     grouped_output 两段相邻泳道边,不再跨两泳道;③"MQA(全部块)"改
#     "MQA (全部块)"补空格消歧。重渲后 Read PNG 复核:六项全 True——路线名
#     与站牌不再重叠,三条汇入 grouped_output 的箭头清晰错开(±16px),
#     HCA 支线不再穿过 Indexer/HashEncoder 框,中文/公式记号(m=4、m'=128)
#     渲染正常,阅读顺序(建模期装配→序列压缩→选块→输出与残差包裹,自上而下
#     自左而右)清楚。lint_chapter_map.py 与 lint_diagram_geometry.py 均
#     无问题(exit 0)。
#   [独立盲审](只看 PNG + chapter.md 标题列表,不看本文件其余注释)复述:
#     从绿色"调用方"桩进入 DeepseekV4Attention(落地装配)→ 内部先调
#     get_dsv4_compress_ratio(交错互补)按层号读开关表 → 4=CSA 分支走
#     Compressor 重叠压缩 m=4(CSA压缩)、128=HCA 分支走另一个 Compressor
#     不重叠压缩 m'=128(HCA)→ CSA 分支的压缩块并行喂给 Indexer(算法
#     top-k,CSA稀疏)与 HashEncoder(运行期 hash 近似,运行期近似)两条选块
#     路径,HCA 分支块少不挑、直接稠密 MQA(HCA稠密)→ 三条路径汇入
#     wo_a/wo_b 分组输出投影 → DeepseekV2DecoderLayer mHC 包裹(mHC包裹)→
#     橙色"返回上层"桩离开。八个站牌逐一对得上正文"2.1 CSA 压缩/2.2 CSA
#     稀疏/2.3 HCA…+稠密 MQA/2.4 为什么交错/2.5 分组输出投影/4.1 装配器/
#     4.2 mHC 包裹/4.3 运行期选块的工程近似"这些自然标题,底部三条阅读
#     路线("CSA 全链路"/"HCA 全链路"/"运行期近似(KVComp)")与图上站牌
#     一一对应可直接跳读。verdict=PASS,一轮通过,未启用第 2 轮盲审配额。
