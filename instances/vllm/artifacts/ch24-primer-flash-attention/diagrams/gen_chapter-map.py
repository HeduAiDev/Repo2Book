#!/usr/bin/env python3
"""第 24 章(【原理篇·论文精读】FlashAttention)——本章地图:源码/论文剖面图。

本章是 primer(原理篇)章,正文用自然标题(一、二、三…),无 `## N.M` 编号——按契约
禁用 §N.M 徽标,站牌改用标题词本身;符号真实性核对改对 book/papers/ch24-primer-
flash-attention/*.md 论文包 + 正文(lint_chapter_map.py 对 kind=primer 章的口径)。

[2026-07-13 重绘] writer 重构骨架后正文改为八节(一~八),原「七、落地:调用面」整节
已删除(该内容降级为一句指路,详见[第 25 章]),新增「八、chunked prefill」大节(⊕
的反面注脚:拆 query 轴、连合并都不需要)——本图同步:下段(vLLM 源码落地)第 3 列从
旧的「调用面」换成「cascade attention」(原第 4 列出口内容整体前移一列),新增第 4
列「chunked prefill」作为新的画布出口(exit)。上段(论文推导)四节点不变。

两条泳道,折成上下两行、各自成段(画布预算:宽 ≤1500 且宽高比 ≤2.6:1,横向 8 列单
行版本 6.9:1 超预算,故改此布局):
  上段(论文推导,FlashAttention 论文族)——本行 4 节点:online-softmax / FlashAttention
    tiling / IO 复杂度账 / FA-2,行内从左到右按推导顺序排列;
  下段(vLLM 源码落地)——本行 4 节点:入口(黑盒调用)、merge_attn_states 合并 kernel、
    cascade attention(共享前缀合并)、chunked prefill(拆 query 轴、⊕ 的反面注脚)
    出口,行内从左到右按正文一~八节的落地顺序排列。
两段之间留一条"桥接带"(空白间隔,画两条跨段箭头 + 简短说明文字):入口在下段最左
潜入上段最左(online-softmax)开始推导,推导完(上段最右 FA-2)浮回下段(merge_kernel)
继续走完落地链——呼应正文"打开黑盒→推导→落地"的形状,只是不再靠横向无限延展,
改靠纵向的"潜入/浮出"桥接表达。

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
LANES = ["论文推导(FlashAttention 论文族)", "vLLM 源码落地"]  # 泳道→折成上下两段,上→下

# (节点id, 段下标(0=上段论文/1=下段源码), 段内列, 段内行号, 真实符号名, 一行短语, 站牌文字)
# 两段各自独立编号列 0..3(不再共享跨段列号)——这就是"折行"的关键:原先 8 个节点
# 依次占列 0..7(单行超宽),现在段内各自从列 0 起数,画布宽度只由"段内最多 4 列"决定。
# [2026-07-13] 下段第 3/4 列随正文改版更新:旧「调用面」(§七,已删除)→
# 「cascade attention」(现在的 §七);新增第 4 列「chunked prefill」(§八,新画布出口)。
NODES = [
    ("entry",        1, 0, 0, "flash_attn_varlen_func",
     "全书黑盒调用点：内部算法未知", "动机"),
    ("online_sm",    0, 0, 0, "Algorithm 3 · online-softmax",
     "running (m,d) 单遍递推、融三遍为一遍", "online-softmax"),
    ("tiling",       0, 1, 0, "Algorithm 1 · FlashAttention tiling",
     "分块入 SRAM 递推、N×N 从不落 HBM", "tiling"),
    ("io_acct",      0, 2, 0, "Theorem 2 · IO 复杂度账",
     "HBM 访问降至 Θ(N²d²/M)、越长越赚", "IO 复杂度账"),
    ("fa2",          0, 3, 0, "FA-2 Algorithm 1",
     "循环序对调、只存 L(一节带过)", "FA-2"),
    ("merge_kernel", 1, 1, 0, "merge_attn_states",
     "Triton kernel：按 LSE 权重精确合并两段输出", "LSE 合并"),
    ("cascade",      1, 2, 0, "cascade attention",
     "共享前缀只算一遍,两段各带 lse 合并", "cascade attention"),
    ("exit",         1, 3, 0, "flash_attn.py 零特判",
     "拆 query 轴、逐行独立，连 ⊕ 都不需要", "chunked prefill"),
]
EDGES = [  # (src_id, dst_id) —— 调用边;同段=段内左→右主线蓝,跨段=桥接带竖向/斜向蓝
    ("entry", "online_sm"),        # 跨段(下→上):潜入推导
    ("online_sm", "tiling"),
    ("tiling", "io_acct"),
    ("io_acct", "fa2"),
    ("fa2", "merge_kernel"),       # 跨段(上→下):浮回落地
    ("merge_kernel", "cascade"),
    ("cascade", "exit"),
]
# 阅读顺序上的 8 个站牌(与正文一~八节一一对应),用于底部阅读路线的独立时间轴——
# 不再复用图上节点的段内列号(折行后同一列号会被两段各用一次,若路线条也用列号,
# 不同段的站牌会在同一 x 位置叠在一起)。
READING_ORDER = ["动机", "online-softmax", "tiling", "IO 复杂度账", "FA-2",
                 "LSE 合并", "cascade attention", "chunked prefill"]
# (路线名, [站牌文字,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
# 速览路线取自正文开篇导语原句:"直接读『六、LSE 合并』到『八、chunked prefill』这三节"
# (外加开篇"动机"给上下文),与旧版"调用面"已随该节删除一并替换。
ROUTES = [
    ("全程精读(论文推导→代码落地)", READING_ORDER, True),
    ("速览路线(只看落地、跳过推导细节)",
     ["动机", "LSE 合并", "cascade attention", "chunked prefill"], False),
]
LEGEND = [
    ("#22c55e", "入口：黑盒调用被打开"),
    ("#3b82f6", "章内主线：论文推导→代码落地"),
    ("#f97316", "出口：回到真实调用现场"),
]
TITLE = "第 24 章 · FlashAttention：论文推导→vLLM 源码落地剖面图"

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
# 桥接带:两段之间的空白间隔,专放跨段箭头 + 简短说明文字(比普通行距宽得多,
# 这条间隔本身就是"折行"的可视化重点——两条跨段箭头在此交汇,而不是横向拉满画布)。
INTER_LANE_GAP = 450

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

# 调用边:同段(band 相同)= 段内左→右,右中→左中;跨段(band 不同)= 桥接带上下沿,
# 上中/下中 attach(避免"从右边出、绕回左边"的倒退斜线——同列跨段时这样能画成
# 干净的竖线,不同列时画成过桥接带的斜线,均不经过任何节点框内部,因为桥接带
# 本身是留白区,没有节点占用)。
bridge_captions = []  # (x, y, text) —— 桥接带箭头旁的简短说明,渲后统一追加避免被箭头压住
for src, dst in EDGES:
    src_band = NODE_BY_ID[src][1]
    dst_band = NODE_BY_ID[dst][1]
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    if src_band == dst_band:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
    elif dst_band > src_band:  # 上段→下段:浮回落地,src 下沿中点→dst 上沿中点
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:  # 下段→上段:潜入推导,src 上沿中点→dst 下沿中点
        p1 = (x1 + NODE_W / 2, y1)
        p2 = (x2 + NODE_W / 2, y2 + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if src_band != dst_band:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        cap = "打开黑盒，潜入论文推导" if dst_band < src_band else "推导完毕，浮回代码落地"
        bridge_captions.append((mx + 24, my, cap))

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

# 底部阅读路线:8 个站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上节点的
# 段内列号——折行后同一列号被两段各用一次,若仍借列号会让"动机"与"online-softmax"
# 两个不同站牌叠在同一 x 位置)。速览路线的 4 个站牌取 READING_ORDER 中对应下标的
# 同一 x,保持与全程路线纵向对齐。时间轴左端起点让给路线名文字(按最长路线名的
# 实际宽度算,不留固定魔数空档——避免长路线名与第一个站牌徽标压在一起)。
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f})")
