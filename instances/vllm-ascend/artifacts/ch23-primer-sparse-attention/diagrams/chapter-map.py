#!/usr/bin/env python3
"""第 23 章(稀疏注意力谱系：从 NSA 到 DSA Lightning Indexer)——本章地图:
论文推导 + NPU 源码剖面图。

本章是 primer(原理篇·论文精读)章,正文用自然标题(一/二/三/四/五/六/七 +
七内的三级 `### ` 子标题,不是 `## N.M` 二级编号标题)——按契约禁用 §N.M 徽标,
站牌改用标题词本身;符号真实性核对改对 book/papers/ch23-primer-sparse-attention/
*.md 论文包 + 正文(lint_chapter_map.py 对 kind=primer 章的口径)。

三段折行(画布预算:宽 ≤1500 且宽高比 ≤2.6:1):
  段0"动机与打分推导"——AscendDSAImpl.forward(动机:O(L²) 税单入口)→ NSA 框架
    (Eq.5 三支路 + N_t≪t)→ Lightning Indexer 打分函数(Eq.1)→ 细粒度 top-k
    选择(Eq.2),四步是论文正文的推导主线,从左到右;
  段1"为什么不掉点 + 成本账"——训练协同适配(Eq.3/4 两阶段 KL 对齐)→ 成本模型
    (论文 2.3 Inference Costs 节,k=512 时主注意力降 256x、端到端约 8.69x),
    两站单独一段,呼应正文「推导」与「数值推演」两个阶段;
    (节点符号用不带 § 的"2.3 Inference Costs"——本章自然标题,禁用 §N.M 徽标,
    §N.M 的正则连出现在节点主符号里也会被 lint 当成违规徽标拦下。)
  段2"落地代码链"——造 indexer key(indexer_select_pre_process)→ 打分+top-k
    一体算子(npu_quant_lightning_indexer)→ top-k 索引→稀疏注意力(cmp_sparse_indices)
    → 层间复用 top-k(skip_topk),对应正文第七节四个 `### ` 子标题,一路都是真实
    源码符号。
两条跨段边:细粒度top-k选择→训练协同适配(对应正文原话"它凭什么不掉点？答案
不在这一节——在下一节的训练协同适配");成本模型→造indexer key(对应正文"推导
讲完，现在看这套数学在昇腾代码里怎么成真")。

用法: python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角(ASCII/拉丁等)按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["动机与打分推导", "为什么不掉点 + 成本账", "落地代码链"]  # 折成三段,上→下

# (节点id, 段下标, 段内列, 段内行号, 真实符号名, 一行短语, 站牌文字)
NODES = [
    ("motivation", 0, 0, 0, "AscendDSAImpl.forward",
     "prefill/decode 主链,对全部前驱打分", "动机"),
    ("nsa_framework", 0, 1, 0, "Eq.(5)/Eq.(6)",
     "cmp+slc+win 门控求和,N_t ≪ t", "NSA 框架"),
    ("dsa_indexer", 0, 2, 0, "Eq.(1)",
     "每头点积→ReLU→加权求和,权重取自 weights_proj", "Lightning Indexer 打分函数"),
    ("topk_cut", 0, 3, 0, "Eq.(2)",
     "只算 top-k 个 KV,O(L²)→O(L·k)", "细粒度 top-k 选择"),
    ("coadapt", 1, 0, 0, "Eq.(3)/Eq.(4)",
     "两阶段续训:KL 对齐真注意力", "训练协同适配"),
    ("cost_model", 1, 1, 0, "2.3 Inference Costs",
     "k=512:主注意力降 256x,端到端 8.69x", "成本模型"),
    ("indexer_key", 2, 0, 0, "indexer_select_pre_process",
     "投影→k_norm→RoPE,造 k^I", "造 indexer key"),
    ("fused_score_topk", 2, 1, 0, "npu_quant_lightning_indexer",
     "点积→ReLU→加权求和→top-k,一算子全包", "打分 + top-k 一体算子"),
    ("sparse_attn", 2, 2, 0, "cmp_sparse_indices",
     "top-k 索引喂入 attn_op,只算选中 KV", "top-k 索引 → 稀疏注意力"),
    ("layer_reuse", 2, 3, 0, "skip_topk",
     "IndexCache 层间复用,省一次打分", "层间复用 top-k"),
]
EDGES = [  # (src_id, dst_id) —— 调用边;同段=段内左→右主线蓝,跨段=桥接带竖向蓝
    ("motivation", "nsa_framework"),
    ("nsa_framework", "dsa_indexer"),
    ("dsa_indexer", "topk_cut"),
    ("topk_cut", "coadapt"),          # 跨段(0→1):它凭什么不掉点?答案在训练协同适配
    ("coadapt", "cost_model"),
    ("cost_model", "indexer_key"),    # 跨段(1→2):推导讲完,看这套数学在昇腾代码里怎么成真
    ("indexer_key", "fused_score_topk"),
    ("fused_score_topk", "sparse_attn"),
    ("sparse_attn", "layer_reuse"),
]
BRIDGE_CAPTIONS = {
    ("topk_cut", "coadapt"): "它凭什么不掉点？答案在训练协同适配",
    ("cost_model", "indexer_key"): "推导讲完，看这套数学在昇腾代码里怎么成真",
}
# 阅读顺序上的 10 个站牌(与正文 一/二/三/四/五/六/七的四个子标题 一一对应),
# 用于底部阅读路线的独立时间轴——不复用图上节点的段内列号(折段后同一列号被
# 多段各用一次,若路线条也用列号,不同段的站牌会在同一 x 位置叠在一起)。
# 这里用比节点角标更短的别名(节点角标本身用完整站名,见 NODES 最后一列)——
# 10 站长站名累加起来太宽,底部时间轴容不下,缩写只用于这条时间轴,不影响
# 图上节点角标与正文标题的对应关系。
READING_ORDER = ["动机", "NSA框架", "Indexer打分", "top-k选择",
                 "训练适配", "成本模型", "造key", "打分+topk",
                 "top-k→attn", "层间复用"]
# (路线名, [站牌文字,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程精读(推导→数值账→落地)", READING_ORDER, True),
    ("只看落地代码(跳论文推导)",
     ["动机", "Indexer打分", "造key", "打分+topk", "top-k→attn", "层间复用"], False),
]
LEGEND = [
    ("#22c55e", "入口：从 O(L²) 注意力税问题切入"),
    ("#3b82f6", "章内主线：论文推导 → NPU 源码落地"),
    ("#f97316", "出口：层间复用后返回上层调用"),
]
TITLE = "第 23 章 · 稀疏注意力谱系：NSA/DSA 论文推导 → Lightning Indexer NPU 源码剖面图"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_BRIDGE_CAPTION = "#475569"

# ---------------- 几何常量(全计算,零魔数) ----------------
BADGE_FONT_SIZE = 11
BADGE_PAD_X = 14
BADGE_H = 20


def badge_width(text):
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + BADGE_PAD_X * 2)


NODE_H = 70
COL_GAP, ROW_GAP = 26, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
# 段间空白:专放跨段箭头 + 简短说明文字。本章两条跨段边都只是单条直箭头(不像
# 双向汇入那样需要更宽的交汇空间),取值够放一行说明文字 + 呼吸感即可。
INTER_LANE_GAP = 96

# 节点宽度:同一批节点统一宽度(保列对齐),按本章最长的符号名/短语算
_SYMBOL_FONT, _PHRASE_FONT = 12.5, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16  # 左右各留:接口桩 + 一段箭头

n_cols = max(n[2] for n in NODES) + 1  # 段内最多列数(各段各自独立复用这批列号)
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_band = [0] * len(LANES)
for _id, band, col, row, *_ in NODES:
    rows_per_band[band] = max(rows_per_band[band], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_band]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += INTER_LANE_GAP  # 段与段之间插入桥接带(不给背景色,留白给跨段箭头)
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, band, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[band] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
node_w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD

# 底部阅读路线的站牌宽度参差不齐(短则"动机"46px,长则"Indexer打分"~95px)——
# 均匀等分位置会让长站牌在窄槽位里互相压住(改动前的教训:10 站等分进节点区
# 决定的画布宽度,长标签必重叠)。改成按累计徽标半宽 + 固定间隙摆放,保证任意
# 两个相邻徽标之间至少留 ROUTE_GAP 的空隙,画布宽度按需反推,不再假设"节点区
# 宽度天然够用"。
ROUTE_GAP = 16
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_first_stop_half_w = badge_width(READING_ORDER[0]) / 2
_route_left = 16 + _route_label_w + 24 + _first_stop_half_w
_route_x = {}
_cx, _prev_half = _route_left, 0.0
for i, name in enumerate(READING_ORDER):
    bw = badge_width(name)
    if i > 0:
        _cx = _cx + _prev_half + ROUTE_GAP + bw / 2
    _route_x[name] = _cx
    _prev_half = bw / 2
_route_right_edge = _route_x[READING_ORDER[-1]] + badge_width(READING_ORDER[-1]) / 2

w = max(node_w, _route_right_edge + PAD_R)


def badge(cx, cy, text):
    """§/站牌徽标胶囊,居中挂在 (cx,cy)——宽度按文字自适应(见 badge_width),
    颜色/圆角/描边视觉语言与模板一致,不变的是"胶囊+靛蓝描边+深靛蓝粗体文字"。"""
    bw = badge_width(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT_SIZE}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="white"/>')

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

# 泳道背景 + 标签(桥接带本身不上色,留白给跨段箭头,视觉上与相邻段区分开)
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
    L.append(f'<line x1="0" y1="{band_top[i] + band_h[i]:.1f}" x2="{w:.1f}" y2="{band_top[i] + band_h[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["motivation"]; ey += NODE_H / 2
xx, xy = NODE_XY["layer_reuse"]; xy += NODE_H / 2
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

# 调用边:同段(band 相同)= 段内左→右,右中→左中;跨段(band 不同)= 桥接带上下沿,
# 上中/下中 attach(不经过任何节点框内部,因为桥接带本身是留白区)。
bridge_captions = []  # (x, y, text) —— 桥接带箭头旁的简短说明,渲后统一追加避免被箭头压住
for src, dst in EDGES:
    src_band = NODE_BY_ID[src][1]
    dst_band = NODE_BY_ID[dst][1]
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    if src_band == dst_band:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
    elif dst_band > src_band:  # 上段→下段:讲完这一步,浮向下一段
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:  # 下段→上段(本章未用到,保留通用分支)
        p1 = (x1 + NODE_W / 2, y1)
        p2 = (x2 + NODE_W / 2, y2 + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    cap = BRIDGE_CAPTIONS.get((src, dst))
    if cap:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        bridge_captions.append((mx + 16, my, cap))

for cx, cy, cap in bridge_captions:
    L.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="sans-serif" font-size="12.5" '
              f'font-style="italic" fill="{C_BRIDGE_CAPTION}">{esc(cap)}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)
for nid, band, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W:.1f}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_SYMBOL_FONT}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_PHRASE_FONT}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_width(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:10 个站牌按 READING_ORDER 排开,x 坐标已在几何常量段按累计
# 徽标宽度算好(_route_x),这里只管画。
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first, x_last = _route_x[stops[0]], _route_x[stops[-1]]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for sec in stops:
        L += badge(_route_x[sec], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f})")
