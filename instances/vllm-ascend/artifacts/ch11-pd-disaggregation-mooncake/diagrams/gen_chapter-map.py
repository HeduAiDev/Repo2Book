#!/usr/bin/env python3
"""第 11 章(PD 分离：连接器分发、mooncake P2P 传输与 KV 亲和调度)——本章地图:源码剖面图。

本章是自然标题章(chapter.md 无 `## N.M` 编号,只有"第一层/挑 layerwise 讲透/第二层/
第三层/高潮"这类自然标题)——按契约禁用 §N.M 徽标,站牌改用标题词本身(取自实际
`## `/`### ` 标题的关键词,短标签,呼应 vllm 实例 ch24 primer 章的先例:同样自然标题,
同样用 badge_width() 按文字宽度自适应,而非模板里给 §N.M 设计的固定 46px 胶囊)。

四条泳道(上→下):proxy 层(第三层,请求入口) / 连接器分发层(第一层) / mooncake P2P 层
(第二层) / KV 亲和(高潮)。拓扑依据 dossier.json 的 data_flow 与 design_decisions:
  - entry = assign_instances——chapter.md 原文"一条请求的完整编排"一节的真实入口函数,
    proxy 收到请求后先挑最闲 prefiller/decoder。
  - AscendMultiConnector 是本章唯一的"分叉点":它是 fan-out 包装器,同时把请求路由给
    MooncakeLayerwiseConnector(主线,layerwise 直传)和外部 KV 池连接器(高潮,亲和路由)——
    data_flow 原文称后者"★ AFFINITY (orthogonal pool path, the climax)",两条路正交,
    共享同一个上游分叉点,分头各走各的,最终都要向 get_finished 汇报完成——它是
    "至此第一层闭合"一段明确点出的收尾动作,选作出口。
  - register_connector 是模块加载期的一次性工厂覆写("代码的入口在
    vllm_ascend/distributed/kv_transfer/__init__.py"——chapter.md 原文原话),画成连接器
    分发层的第一个节点,喂给 AscendMultiConnector,解释"为什么后面跑的是 Ascend 的类"。

节点预算 9 ≤ 12。画布:5 列,自适应 NODE_W(按本章最长符号/短语算,不用模板定死的 190)。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角(ASCII/拉丁等)按 0.58×size,求和。中文字符是
    方块字,实际显示宽度接近整个字号,若仍按半角系数估算会算少。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["proxy 层（第三层，请求入口）", "连接器分发层（第一层）", "mooncake P2P 层（第二层）", "KV 亲和（高潮）"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文字(自然标题关键词))
NODES = [
    ("entry",       0, 0, 0, "assign_instances",           "proxy 接请求，备选 P/D 实例",   "完整编排"),
    ("pick_srv",    0, 1, 0, "_pick_server",                "打分＋懒删除堆，选最闲",         "两个打分，两个堆"),
    ("register",    1, 0, 0, "register_connector",          "覆写注册表，换成 Ascend 类",     "工厂覆写"),
    ("divergence",  1, 1, 0, "AscendMultiConnector",        "永远给 layerwise 真 blocks",     "关键分歧"),
    ("facade_push", 2, 2, 0, "MooncakeLayerwiseConnector",  "两半身：定方向＋逐层推",         "两半身"),
    ("mooncake_p2p",2, 3, 0, "GlobalTE",                    "单例引擎，地址合批直传",         "一个引擎"),
    ("lookup",      3, 2, 0, "LookupKeyClient",             "zmq 问外部池命中前缀",           "命中查询"),
    ("gap",         3, 3, 0, "KVPoolScheduler",             "只搬缺口＝命中－已算",           "算缺口"),
    ("exit",        1, 4, 0, "get_finished",                "汇报完成，释放块",               "Worker 半边"),
]
EDGES = [  # (src_id, dst_id) —— 调用/分发边,统一主线蓝
    ("entry", "pick_srv"),
    ("pick_srv", "divergence"),        # proxy 挑完，请求进到接收节点的连接器层
    ("register", "divergence"),        # 工厂覆写是前提：没有它就轮不到 Ascend 的类
    ("divergence", "facade_push"),     # 分叉①：fan-out 到 layerwise 直传主线
    ("divergence", "lookup"),          # 分叉②：fan-out 到外部 KV 池亲和路由（正交，高潮）
    ("facade_push", "mooncake_p2p"),
    ("lookup", "gap"),
    ("mooncake_p2p", "exit"),
    ("gap", "exit"),
]
# (路线名, [(列, 站牌文字), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("三层主线",
     [(0, "完整编排"), (1, "关键分歧"), (2, "两半身"), (3, "一个引擎"), (4, "Worker 半边")], True),
    ("高潮：KV 亲和",
     [(0, "完整编排"), (1, "关键分歧"), (2, "命中查询"), (3, "算缺口"), (4, "Worker 半边")], False),
]
LEGEND = [("#22c55e", "入口：请求从客户端进入 proxy"), ("#3b82f6", "章内主线：调用／分发／合批"),
          ("#f97316", "出口：worker 汇报完成，返回调度循环")]
TITLE = "第 11 章 · PD 分离源码剖面：proxy 分发 → 连接器分发 → mooncake P2P／KV 亲和"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数;NODE_W/badge 宽度按本章文字自适应) --------
BADGE_FONT_SIZE, BADGE_PAD_X, BADGE_H = 11, 12, 20


def badge_width(text):
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + BADGE_PAD_X * 2)


_SYMBOL_FONT, _PHRASE_FONT, _NODE_TEXT_PAD = 12.5, 10.5, 22
NODE_H = 58
COL_GAP, ROW_GAP = 34, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44

NODE_W = max(
    170.0,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24  # 左右各留:接口桩 + 一段箭头

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
    """站牌胶囊,居中挂在 (cx,cy)——宽度按文字自适应(badge_width),视觉语言与模板一致
    (圆角矩形 + 靛蓝描边 + 深靛蓝粗体文字);本章自然标题,文字是标题关键词而非 §N.M。"""
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

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w:.1f}" y2="{lanes_bottom:.1f}" '
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

# 调用边(主线蓝)。多条边汇入同一节点时终点 y 各偏移,否则汇合处看不出"汇合"。
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
for nid, lane, col, row, symbol, phrase, sec in NODES:
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f})")
