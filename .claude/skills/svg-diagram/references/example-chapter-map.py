#!/usr/bin/env python3
"""chapter-map 模板:每章开篇的「本章地图」——源码剖面图。
横向泳道=代码分层(如 调度层/执行层/算子层),圆角节点=真实符号名+一行短语,
节点右上角挂 §N.M 讲解站牌,画布左右边缘各有一个"调用方/返回上层"接口桩,
底部用同一批 § 站牌拼出本章的多条阅读路线(如 快通道/全通道)。

示例(假想章数据):forward() 先按 5 态判定分流——DECODE 态走快通道(轻量单步),
其余 4 态走全通道(通用变长)——两条路径各过一次执行层/算子层后在出口汇合。

■ 不可变(全书 72 章统一视觉语言,换章节数据时不要动这些,只改下面的 DATA):
  1. §徽标胶囊:圆角矩形(pill),fill #eef2ff / stroke #6366f1 / 文字 indigo 深色,
     贴在节点框的右上角、跨在上边框上(见 badge() 的 corner 用法);
     底部路线条复用同一个 badge() 画法,只是居中挂在路线上而非贴节点角。
  2. 入口/出口:画布左右边缘各放一个"调用方 / 返回上层"接口桩(stub),
     入口箭头 = 绿 #22c55e,出口箭头 = 橙 #f97316。
  3. 节点间调用边(主线)统一 = 蓝 #3b82f6。
  4. 底部路线条:高亮/推荐路线 = 实线蓝 #3b82f6(粗);其余路线 = 虚线中性灰
     #94a3b8(细)。"高亮=蓝实线,其余=灰虚线"这条语义不能变,但"哪条路线该
     高亮"由本章内容决定(ROUTES 第 3 个字段)。
  5. >2 种语义色(绿/橙/蓝)必须画图例——顶部 LEGEND 行不要删。
  6. 按字符估算文本宽度的布局(如图例间距)一律用 cjk_text_width(),不要直接
     用半角系数(0.58×size)乘 len(s)——中文全角字符实际显示宽度接近整个字号,
     半角系数会把中文标签算短,导致下一个图例/文字压上来(见 [FIX-ROUND-2])。

■ 可变(换章节时改这些):LANES(泳道数与名字)、NODES(节点:所在泳道/列/行/
  真实符号名/一行短语/§编号)、EDGES(节点间调用边)、ROUTES(底部阅读路线,
  每条给出"沿哪些列、每列对应哪个 §"——列号必须复用 NODES 里已出现的列)。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录;每次改动 DATA/布局代码后必须
  重新核对一遍,不能照抄上一轮结果):
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  [FIX-ROUND-2](本轮修复后重新渲染+Read PNG 复核,替换第一轮记录,如实记):
    - 第一轮 no_overlap 曾误记为 True——图例宽度估算(11.5*0.58*len(label))对
      中文全角字符按半角系数算,把标签宽度算短了,实际渲染中第一条图例"入口:
      从上层调用进入"末字被第二条图例的蓝色色块压住。本轮改用 cjk_text_width()
      逐字符估算(全角 ord>0x2E80 按 1.0×size / 半角按 0.58×size)重渲染并
      Read PNG 复核图例区,确认零重叠,现在的 True 是重新核对后的结果。
    - 同轮顺带修了两处不影响自查判定但影响观感/健壮性的问题:①"调用方"/
      "返回上层"/阅读路线表头三处固定 UI 文案补了 esc() 转义;②merge_and_output()
      两条汇入边此前终点 y 完全重合(看不出"汇合"),现按 ±8px 错开。

用法:python3 example-chapter-map.py → 同目录 example-chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(CJK 及其他东亚文字,ord>0x2E80:含中日韩表意文字/假名/谚文/CJK 标点等)按 1.0×size,
    半角(ASCII/拉丁/半角标点等)按 0.58×size,求和。中文字符是方块字,实际显示宽度
    接近整个字号,若仍按半角系数估算会算少——中英混排的图例/标签必须用这个,
    不能直接 0.58 * size * len(s)。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["调度层", "执行层", "算子层"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("entry",       0, 0, 0, "forward()",               "统一入口,接收隐藏状态",   "§20.1"),
    ("dispatch",    0, 1, 0, "_dispatch_by_state()",     "按 5 态判定选快/全通道",  "§20.2"),
    ("fast_attn",   1, 2, 0, "decode_attention()",       "解码态:单步轻量路径",     "§20.3"),
    ("full_attn",   1, 2, 1, "prefill_attention()",      "预填充态:变长通用路径",   "§20.5"),
    ("fast_kernel", 2, 3, 0, "fused_decode_kernel()",    "单核完成注意力全部计算",  "§20.4"),
    ("full_kernel", 2, 3, 1, "chunked_prefill_kernel()", "分块核,兼容变长 KV",      "§20.6"),
    ("exit",        0, 4, 0, "merge_and_output()",       "合并两路径,写回输出",     "§20.7"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("entry", "dispatch"),
    ("dispatch", "fast_attn"), ("dispatch", "full_attn"),
    ("fast_attn", "fast_kernel"), ("full_attn", "full_kernel"),
    ("fast_kernel", "exit"), ("full_kernel", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("快通道(DECODE)",   [(0, "§20.1"), (1, "§20.2"), (2, "§20.3"), (3, "§20.4"), (4, "§20.7")], True),
    ("全通道(其余 4 态)", [(0, "§20.1"), (1, "§20.2"), (2, "§20.5"), (3, "§20.6"), (4, "§20.7")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 20 章 · forward() 分流剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 58
COL_GAP, ROW_GAP = 42, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
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

# 调用边(主线蓝,画在节点下面这条先画后画都行,这里先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px,如 2 条即 ±8px),
# 否则重合的终点在视觉上看不出"汇合"、像一条线断头。
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

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
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
out = Path(__file__).with_name("example-chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
