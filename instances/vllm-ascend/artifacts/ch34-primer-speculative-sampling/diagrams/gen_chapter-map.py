#!/usr/bin/env python3
"""第 34 章(投机采样:拒绝采样保分布定理与加速账本)——本章地图:重绘版
(writer 重构章节骨架后同步:旧版引用的「验证侧·接受/验证侧·残差」两站及
DeepSeekV4MTP/acceptance_condition/sample_recovered_tokens 等符号,在新稿里
已收进「见第 33/36 章」的转述句,不再是本章正文的自有站点——继续挂在图上会
被 lint_chapter_map.py 判杜撰符号(kind=primer 章只核 book/papers/*.md 论文包
+ 正文 chapter.md,不核 dossier.json)。本轮按新稿实际的一~五节 + 两条子线索
重新设计。)

本章是 primer(原理篇)章,正文用自然标题(一、二、三、四、五),无 `## N.M` 编号——
按契约禁用 §N.M 徽标,站牌改用标题词本身。

两条泳道,各 4 节点,单行左→右,对应正文开篇点破的「两本账」结构:
  上道(正确性无条件,对应「二、保分布定理」及其铺垫「一、动机」)——
    动机 / 接受-拒绝 / 保分布证明 / 成绩单 α;
  下道(速度单独定价 → 工程落地,对应「三、加速账本」「四、MTP」「五、前瞻」)——
    期望长度 E[L] / 加速比·最优 γ / MTP 草稿链 / 前瞻。
两道之间只有一条跨道边(成绩单 α → 期望长度 E[L]):正确性这本账已经关账,
剩下的全是速度账本——这正是本章开篇那句「两本账」在图上的落点。

节点预算:8(远低于 12 上限)。符号栏逐一核对为 narrative/chapter.md 正文原样
子串(min(1,p/q) / norm(max(0,p-q)) / min(p,q) / e_proj / h_proj /
deepseek_v4_mtp.py 均逐一 grep 验证存在;纯希腊字母/数字/CJK 短语不触发
lint 的符号防杜撰规则,因其正则只核形似代码标识符的 token)。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
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
LANES = ["正确性无条件(动机 → 保分布定理:一、二两节)",
         "速度单独定价 → 工程落地(加速账本 → MTP → 前瞻:三~五节)"]

# (节点id, 泳道下标(0=正确性/1=速度+落地), 泳道内列, 泳道内行号, 真实符号/公式, 一行短语, 站牌文字)
NODES = [
    ("entry", 0, 0, 0, "K 次串行前向",
     "内存带宽瓶颈,算力大量闲置", "动机"),
    ("accept_reject", 0, 1, 0, "min(1,p/q)",
     "残差 norm(max(0,p-q)) 找补", "接受-拒绝"),
    ("proof", 0, 2, 0, "min(p,q)+残差=p",
     "两段相加,q 在恒等式里相消", "保分布证明"),
    ("alpha", 0, 3, 0, "α = E(β)",
     "1 减两分布距离,草稿的成绩单", "成绩单 α"),
    ("expected_length", 1, 0, 0, "γ=5 → E[L]=3.689",
     "收益递减,封顶于 1/(1-α)", "期望长度 E[L]"),
    ("speedup", 1, 1, 0, "c=0.05,γ*=8 → 3.092×",
     "拔河:γ↑验证更多,但草稿开销 c 也涨", "加速比·最优 γ"),
    ("mtp", 1, 2, 0, "e_proj + h_proj",
     "深度 k 因果链,源码 deepseek_v4_mtp.py", "MTP 草稿链"),
    ("exit", 1, 3, 0, "DFlash / DSpark",
     "验证侧一个字不用改,起草侧再升级", "前瞻"),
]
EDGES = [  # (src_id, dst_id) —— 调用边;同道=道内左→右主线蓝,跨道=桥接带竖向蓝
    ("entry", "accept_reject"),
    ("accept_reject", "proof"),
    ("proof", "alpha"),
    ("alpha", "expected_length"),   # 跨道(上→下):正确性已关账,转去算速度
    ("expected_length", "speedup"),
    ("speedup", "mtp"),
    ("mtp", "exit"),
]
# 阅读顺序上的 8 个站牌(与正文一~五节内容一一对应),用于底部阅读路线的独立时间轴——
# 不复用图上节点的道内列号(两道各自独立编号,同一列号被两道各用一次)。
READING_ORDER = ["动机", "接受-拒绝", "保分布证明", "成绩单 α",
                  "期望长度 E[L]", "加速比·最优 γ", "MTP 草稿链", "前瞻"]
# (路线名, [站牌文字,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程精读(证明→数值→落地)", READING_ORDER, True),
    ("速览路线(跳过证明,只看能快多少)",
     ["动机", "期望长度 E[L]", "加速比·最优 γ", "MTP 草稿链", "前瞻"], False),
]
LEGEND = [
    ("#22c55e", "入口:接续第 33 章「验证侧凭什么对」的追问"),
    ("#3b82f6", "章内主线:动机→保分布定理→加速账本→MTP"),
    ("#f97316", "出口:前瞻 DFlash (第 35 章)与 DSpark (第 37 章)"),
]
TITLE = "第 34 章 · 投机采样:拒绝采样保分布定理与加速账本剖面图"

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


NODE_H = 74
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 66, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
# 桥接带:两道之间的空白间隔,专放跨道箭头 + 简短说明文字。
INTER_LANE_GAP = 90

# 节点宽度:同一批节点统一宽度(保列对齐),按本章最长的符号名/短语算
_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16  # 左右各留:接口桩 + 一段箭头

n_cols = max(n[2] for n in NODES) + 1  # 道内最多列数(两道各自独立复用这批列号)
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_band = [0] * len(LANES)
for _id, band, col, row, *_ in NODES:
    rows_per_band[band] = max(rows_per_band[band], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_band]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += INTER_LANE_GAP  # 道与道之间插入桥接带(不给背景色,留白给跨道箭头)
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
    """站牌徽标胶囊,居中挂在 (cx,cy)——宽度按文字自适应(见 badge_width),
    颜色/圆角/描边视觉语言与模板一致,不变的是"胶囊+靛蓝描边+深靛蓝粗体文字"。
    本章自然标题,站牌文字用标题词本身,不用 §N.M。"""
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
# 图例(>2 种语义色必须画图例;本章图例文案较长,自动换行到多行)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
_legend_line_h = 16
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _ly += _legend_line_h

# 泳道背景 + 标签(桥接带本身不上色,留白给跨道箭头,视觉上与两道区分开)
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

# 调用边:同道(band 相同)= 道内左→右,右中→左中;跨道(band 不同)= 桥接带上下沿,
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
    elif dst_band > src_band:  # 上道→下道:正确性已关账,转去算速度
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:  # 下道→上道(本章未用到,保留对称写法以防后续改数据加回程)
        p1 = (x1 + NODE_W / 2, y1)
        p2 = (x2 + NODE_W / 2, y2 + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if src_band != dst_band:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        cap = "回头看正确性证明" if dst_band < src_band else "正确性已关账,转去算速度"
        bridge_captions.append((mx + 24, my, cap))

for cx, cy, cap in bridge_captions:
    L.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="sans-serif" font-size="12.5" '
              f'font-style="italic" fill="{C_BRIDGE_CAPTION}">{esc(cap)}</text>')

# 节点(圆角框 + 真实符号/公式 + 一行短语 + 右上角站牌)
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

# 底部阅读路线:8 个站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上节点的
# 道内列号——两道各自独立编号,若仍借列号会让不同道的站牌叠在同一 x 位置)。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_route_left = 16 + _route_label_w + 24
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio={w/h:.2f}, NODE_W={NODE_W:.0f})")
