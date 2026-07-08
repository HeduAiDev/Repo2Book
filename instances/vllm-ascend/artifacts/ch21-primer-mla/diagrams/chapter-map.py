#!/usr/bin/env python3
"""第 21 章(【原理篇·论文精读】MLA)——本章地图:论文/源码剖面图。

本章是 primer(原理篇)章,正文用自然标题(一、二、三、四 + 二内的 2.1-2.4 均是
`### ` 三级标题,不是 `## N.M` 二级编号标题)——按契约禁用 §N.M 徽标,站牌改用
标题词本身;符号真实性核对改对 book/papers/ch21-primer-mla/paper.md 论文包 +
正文(lint_chapter_map.py 对 kind=primer 章的口径)。

两段折行(画布预算:宽 ≤1500 且宽高比 ≤2.6:1):
  上段"数学地基"——动机 → 低秩 KV 压缩 → 权重吸收恒等式,三步建立起"压缩+吸收"
    这套能省显存的机制,行内按推导顺序从左到右;
  下段"位置编码与落地"——解耦 RoPE·核心(全章最硬的一问,承接上段"权重吸收为什么
    在这里失效") → q 侧低秩(支线,目的与 KV 侧不同) → 账单(汇总四种机制对比) →
    落地(decode 吸收 vs prefill 物化,两路汇入 o_proj)。
两段之间只有一条跨段边(absorb → rope),对应正文原话"打分就变成潜空间里的一次
内积——省算力,而结果一丝不差"到"这是全章最硬、也是第 22 章读者最容易卡住的一问"
的转折点,配一句桥接说明文字。

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
LANES = ["数学地基：低秩压缩与权重吸收", "解耦 RoPE 与落地"]  # 泳道→折成上下两段,上→下

# (节点id, 段下标(0=上段/1=下段), 段内列, 段内行号, 真实符号名, 一行短语, 站牌文字)
NODES = [
    ("motivation", 0, 0, 0, "get_kv_cache_shape",
     "MLA 缓存无 2·n_h·d_h", "动机"),
    ("compress",   0, 1, 0, "exec_kv_decode",
     "只落盘潜向量 c_kv", "低秩 KV 压缩"),
    ("absorb",     0, 2, 0, "_q_proj_and_k_up_proj",
     "W_UK 折进 query 潜空间", "权重吸收恒等式"),
    ("rope",       1, 0, 0, "rope_single",
     "q_pe/k_pe 单开一路", "解耦 RoPE·核心"),
    ("qside",      1, 1, 0, "fused_qkv_a_proj",
     "只降训练激活,不动缓存", "q 侧低秩"),
    ("ledger",     1, 2, 0, "kv_lora_rank",
     "576=d_c+d_h^R,不含头数", "账单"),
    ("landing",    1, 3, 0, "o_proj",
     "decode 吸收 / prefill 物化", "落地"),
]
EDGES = [  # (src_id, dst_id) —— 调用边;同段=段内左→右主线蓝,跨段=桥接带竖向蓝
    ("motivation", "compress"),
    ("compress", "absorb"),
    ("absorb", "rope"),          # 跨段(上→下):权重吸收讲完,直奔"位置旋转能不能也吸收"
    ("rope", "qside"),
    ("qside", "ledger"),
    ("ledger", "landing"),
]
# 阅读顺序上的 7 个站牌(与正文 一/2.1/2.2/2.3/2.4/三/四 一一对应),用于底部
# 阅读路线的独立时间轴——不复用图上节点的段内列号(折行后同一列号被两段各用
# 一次,若路线条也用列号,不同段的站牌会在同一 x 位置叠在一起)。
READING_ORDER = ["动机", "低秩 KV 压缩", "权重吸收恒等式", "解耦 RoPE·核心",
                 "q 侧低秩", "账单", "落地"]
# (路线名, [站牌文字,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程精读(动机→压缩→吸收→RoPE→落地)", READING_ORDER, True),
    ("只看核心悬崖(RoPE 为什么不可吸收)",
     ["动机", "权重吸收恒等式", "解耦 RoPE·核心", "落地"], False),
]
LEGEND = [
    ("#22c55e", "入口：从 KV cache 瓶颈问题切入"),
    ("#3b82f6", "章内主线：数学推导→源码落地"),
    ("#f97316", "出口：两条路径汇入 o_proj"),
]
TITLE = "第 21 章 · MLA 数学推导→NPU 源码剖面图（DeepSeek-V2, arXiv:2405.04434）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
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
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
# 桥接带:两段之间的空白间隔,专放跨段箭头 + 简短说明文字。本章只有一条跨段边
# (不像 FA 章那样有"潜入/浮出"两条),间隔按容纳一行说明文字 + 视觉呼吸感取值,
# 不需要 FA 章 450 那么宽(那是给双向箭头留交汇空间)。
INTER_LANE_GAP = 170

# 节点宽度:同一批节点统一宽度(保列对齐),按本章最长的符号名/短语算
_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16  # 左右各留:接口桩 + 一段箭头

n_cols = max(n[2] for n in NODES) + 1  # 段内最多列数(两段各自独立复用这批列号)
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
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


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

# 泳道背景 + 标签(桥接带本身不上色,留白给跨段箭头,视觉上与两段区分开)
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
xx, xy = NODE_XY["landing"]; xy += NODE_H / 2
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
    elif dst_band > src_band:  # 上段→下段:讲完权重吸收,浮向本章最硬的一问
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:  # 下段→上段(本章未用到,保留通用分支)
        p1 = (x1 + NODE_W / 2, y1)
        p2 = (x2 + NODE_W / 2, y2 + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if src_band != dst_band:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        cap = "权重吸收讲完，位置旋转能不能也吸收？" if dst_band > src_band else "浮回落地"
        bridge_captions.append((mx + 20, my, cap))

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

# 底部阅读路线:7 个站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上
# 节点的段内列号——折行后同一列号被两段各用一次)。_route_left 是第一个站牌
# 徽标的圆心 x,而 24 这段留白量的是"文字末端→圆心"的距离,若不扣掉徽标半宽,
# 圆心往左挪半个徽标宽后,徽标左边缘会反过来吃进文字尾部(圆心-半宽 < 文字末端)
# ——量出第一个站牌(READING_ORDER[0])的实际徽标宽度,把它的半宽也算进留白。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_first_stop_half_w = badge_width(READING_ORDER[0]) / 2
_route_left = 16 + _route_label_w + 24 + _first_stop_half_w
_n_stops = len(READING_ORDER)
_route_x = {name: _route_left + i * (w - PAD_R - _route_left) / (_n_stops - 1)
            for i, name in enumerate(READING_ORDER)}

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
