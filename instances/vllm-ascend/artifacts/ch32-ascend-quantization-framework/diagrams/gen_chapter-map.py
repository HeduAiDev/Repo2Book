#!/usr/bin/env python3
"""第 32 章「本章地图」——昇腾量化框架源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则/entry-exit stub)原样保留，只改 DATA
与两处"同列纵向"连接的绘制方式(见下)。

节点预算：10 个(reg_config/reg_scheme/dispatch/resolve/wrap_linear/wrap_kv/wrap_moe/
scheme_impl/npu_ops/granularity) ≤ 12。本章标题为编号标题(## 32.1 ... ## 32.6)，
站牌用 §32.N。

设计要点：
- 四条泳道 = 本章四个自然分层：注册层(§32.1 Config 注册 / §32.2 scheme 注册表)→
  分发层(§32.3 按层分发 + 逐层查表)→适配层(§32.4 三个 wrapper)→执行层(§32.5
  W8A8_DYNAMIC 全链 + §32.6 粒度谱延伸)。
- 入口(绿)有两个并行触发点——reg_config(Config 装饰器注册)与 reg_scheme(scheme
  注册表落表)，两者都由"vLLM-Ascend 模块 import"触发，参照 ch03 的双入口画法
  (主干线到第一个节点的 y，再拉一条纵向连接到第二个节点的 y)。
- dispatch→resolve、scheme_impl→granularity 是"同列不同行"的顺承关系(get_quant_method
  内部调用 create_scheme_for_layer；scheme 的 per-channel scale 引出粒度谱概念)。
  若沿用模板默认的"src 右边→dst 左边"通用画法，会因为两者同列而画成一条向左倒退
  的斜线(视觉上像箭头往回指)。改成专门的纵向连接：上节点底边中点→下节点顶边
  中点，方向和语义都对得上。
- wrap_kv / wrap_moe 只画到适配层为止、不再向执行层延伸——本章"走通全链"只选了
  linear 路径的 W8A8_DYNAMIC(§32.5 原文明说"选它最常用也最好讲")，attention/moe
  两个 wrapper 结构同构但本章没有逐行拆到底，故不虚构它们的下游调用边。
  [FIX-ROUND-2] wrap_linear 特意排在适配层三行的最后一行(row2，紧邻执行层)而非
  第一行——若排第一行，它到 scheme_impl 的延续边要纵穿 wrap_kv/wrap_moe 所在的
  行区，在窄列间距下会擦到那两个节点的 §32.4 徽标(渲染后 Read PNG 发现线段与
  AscendKVCacheMethod 徽标几乎贴住)。把 wrap_linear 挪到最后一行，让它与 scheme_impl
  只隔一次泳道过渡(纵向位移最短)，延续边自然贴着列间隙走，不再靠近另外两个
  节点——不改变 EDGES/连接关系，只调整同一泳道内三个并列节点的行序。
- granularity(QuantTypeMapping，§32.6)是 scheme_impl 的一条延伸支线，不接回出口——
  它是"per-channel scale 之上的概念扩展"，不是主线返回路径的一部分。

六项自查记录(渲染→Read PNG 亲眼看后如实记录)：见 figure-manifest.json 对应条目。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["注册层", "分发层", "适配层", "执行层"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语(可含 "\n"), §编号)
NODES = [
    ("reg_config",  0, 0, 0, "register_\nquantization_config",
     "三入口注册 Config 类", "§32.1"),
    ("reg_scheme",  0, 0, 1, "_SCHEME_REGISTRY",
     "register_scheme 装饰器\nimport 即注册落表", "§32.2"),
    ("dispatch",    1, 1, 0, "get_quant_method",
     "按 isinstance(layer) 四岔分发", "§32.3"),
    ("resolve",     1, 1, 1, "create_scheme_for_layer",
     "查注册表，选 scheme 类", "§32.3"),
    ("wrap_kv",     2, 2, 0, "AscendKVCacheMethod",
     "attn 层 wrapper，转交 scheme", "§32.4"),
    ("wrap_moe",    2, 2, 1, "AscendFusedMoEMethod",
     "moe 层 wrapper，转交 scheme", "§32.4"),
    ("wrap_linear", 2, 2, 2, "AscendLinearMethod",
     "linear 层 wrapper，转交 scheme", "§32.4"),
    ("scheme_impl", 3, 3, 0, "AscendW8A8Dynamic\nLinearMethod",
     "int8 权重 + per-channel scale", "§32.5"),
    ("npu_ops",     3, 4, 0, "npu_dynamic_quant\nnpu_quant_matmul",
     "动态量化激活 → 量化 GEMM", "§32.5"),
    ("granularity", 3, 3, 1, "QuantTypeMapping",
     "per-tensor/channel/group\n微缩放粒度谱一览", "§32.6"),
]
# 横向调用边(主线蓝) —— 不含两条"同列纵向"关系(dispatch→resolve / scheme_impl→granularity，
# 那两条单独画,见下方 VERTICAL_EDGES)
EDGES = [
    ("reg_config", "dispatch"),
    ("reg_scheme", "resolve"),
    ("resolve", "wrap_linear"), ("resolve", "wrap_kv"), ("resolve", "wrap_moe"),
    ("wrap_linear", "scheme_impl"),
    ("scheme_impl", "npu_ops"),
]
# 同列不同行的顺承关系(纵向箭头,同为主线蓝,不用横向公式画——见文件头说明)
VERTICAL_EDGES = [("dispatch", "resolve"), ("scheme_impl", "granularity")]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("通读主线", [(0, "§32.1"), (1, "§32.3"), (2, "§32.4"), (4, "§32.5")], True),
    ("跳读:两张注册表怎么填", [(0, "§32.2"), (1, "§32.3")], False),
    ("跳读:量化粒度谱单独看", [(2, "§32.4"), (3, "§32.6")], False),
]
LEGEND = [("#22c55e", "入口:vLLM-Ascend 模块 import / 建层调用触发"),
          ("#3b82f6", "章内主线调用/查表边"),
          ("#f97316", "出口:量化结果返回 vLLM 层")]
TITLE = "第 32 章 · 昇腾量化框架剖面(注册表 + 三 wrapper 适配 + W8A8 全链)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# 符号名/短语最长两行(如 register_quantization_config、npu_dynamic_quant+npu_quant_matmul)
# 靠 NODES 里手工换行("\n")拆行，NODE_H 留够 2 行符号 + 2 行短语的空间。
NODE_W, NODE_H = 200, 90
COL_GAP, ROW_GAP = 34, 22
EDGE_MARGIN, STUB_W, STUB_H = 12, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
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
    """§ 徽标胶囊,居中挂在 (cx,cy) —— 节点用它贴右上角,路线legend用它居中挂线上。"""
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

# 入口接口桩:两个并行触发点(reg_config / reg_scheme,同为"模块 import"触发)——
# 参照 ch03 两段式 monkey-patch 图的双入口画法:主干线接到第一个节点的 y,
# 再拉一条纵向连接到第二个节点的 y。
ex, ey = NODE_XY["reg_config"]; ey += NODE_H / 2
ex2, ey2 = NODE_XY["reg_scheme"]; ey2 += NODE_H / 2
xx, xy = NODE_XY["npu_ops"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{EDGE_MARGIN + STUB_W}" y2="{ey2:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2"/>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey2:.1f}" x2="{ex2:.1f}" y2="{ey2:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 横向调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移,避免看起来像断头线。
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

# 纵向顺承边(同列不同行:dispatch→resolve、scheme_impl→granularity)。
# 若套用上面"src 右边→dst 左边"的通用公式,会因同列而画成一条向左倒退的斜线——
# 改画"上节点底边中点→下节点顶边中点"的纵向箭头,方向和语义都对得上。
for src, dst in VERTICAL_EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    L.append(f'<line x1="{x1 + NODE_W / 2:.1f}" y1="{y1 + NODE_H:.1f}" '
              f'x2="{x2 + NODE_W / 2:.1f}" y2="{y2:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名[1~2 行] + 一行短语[1~2 行,始终锚在节点下半区] + 右上角 § 徽标)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 24, 40
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 71, 66, 80
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines = symbol.split("\n")
    sym_ys = [y + SYMBOL_1LINE_Y] if len(sym_lines) == 1 else [y + SYMBOL_2LINE_Y1, y + SYMBOL_2LINE_Y2]
    for line, ly in zip(sym_lines, sym_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = phrase.split("\n")
    phrase_ys = [y + PHRASE_1LINE_Y] if len(phrase_lines) == 1 else [y + PHRASE_2LINE_Y1, y + PHRASE_2LINE_Y2]
    for line, ly in zip(phrase_lines, phrase_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}: {w:.0f}x{h:.0f}")
