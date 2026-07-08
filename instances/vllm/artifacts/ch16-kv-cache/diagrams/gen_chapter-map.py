#!/usr/bin/env python3
"""第 16 章(KV 块分配与多注意力协调)——本章地图:源码剖面图。

本章是编号标题章(`## 16.N`),§徽标用 §16.N,徽标章号须与目录号 ch16 一致。

两条泳道,各自折成两行(画布预算:宽 ≤1500 且宽高比 ≤2.6:1;单行 7+ 列会横向超宽,
参照 ch14/ch24 chapter-map 的折行修复,两段各自复用同一批列号 0..3):
  上段(kv_cache_manager.py + single_type_kv_cache_manager.py)——allocate_slots
    三阶段主线(§16.1-§16.4),8 节点分两行:第一行 entry/seqgate/remove/budget 按
    调用顺序左→右;第二行接着 budget 往下续 alloc_computed/alloc_new/exit,外加
    一个从 remove 垂下的辅助节点 skip_calc(被 remove 与 budget 内部共用的判跳过
    公式,图上只画 remove 这条真实调用边,budget 那条在文字里点出不重复画线,
    避免同一站牌两条几乎重合的入边)。
  下段(kv_cache_coordinator.py)——协调层:构造期三态工厂(§16.5)+ 运行期 Hybrid
    不动点(§16.6)+ 准入上限单一真相源注入(§16.3 的另一半),3 节点分两行,factory
    独占一行(对齐上段 exit 所在列,一条竖直桥接边"exit → factory"表达"本函数
    全程用的 self.coordinator 是这里构造期选定的实例"这一构造期↔运行期关系
    ——不是真实调用边,故给这条桥接边配了说明文字,同一模式见 ch14/ch24)。

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
LANES = [
    "kv_cache_manager.py + single_type_kv_cache_manager.py：allocate_slots 三阶段主线",
    "kv_cache_coordinator.py：协调层——构造期三态工厂 + Hybrid 不动点",
]

# (节点id, 段下标(0=上段主线/1=下段协调层), 段内列, 段内行号, 真实符号名, 一行短语, §徽标)
# 两段各自独立编号列 0..3(折行的关键:不再让 7+ 个节点依次占列 0..6 单行超宽)。
NODES = [
    ("entry",           0, 0, 0, "allocate_slots",
     "统一入口，五段 token 布局分流三阶段", "§16.1"),
    ("seqgate",         0, 1, 0, "full_sequence_must_fit",
     "整条序列放不下就整体拒绝（默认关）", "§16.3"),
    ("remove",          0, 2, 0, "remove_skipped_blocks",
     "先释放窗外块，换 null 不删除", "§16.2"),
    ("budget",          0, 3, 0, "get_num_blocks_to_allocate",
     "净申请块数：skipped 折抵 + 可驱逐块计入", "§16.2"),
    ("skip_calc",       0, 0, 1, "get_num_skipped_tokens",
     "全注意力恒 0，SWA/chunked 各一条公式", "§16.2"),
    ("alloc_computed",  0, 1, 1, "allocate_new_computed_blocks",
     "touch 命中块、null 填、external 真分配", "§16.4"),
    ("alloc_new",       0, 2, 1, "allocate_new_blocks",
     "新建 new + lookahead 槽位", "§16.4"),
    ("exit",            0, 3, 1, "cache_blocks",
     "num_tokens_to_cache 封顶后写哈希、返回", "§16.4"),
    ("factory",         1, 3, 0, "get_kv_cache_coordinator",
     "构造期三态：关缓存 / 单组 / 多组", "§16.5"),
    ("admission_cap",   1, 2, 1, "get_manager_for_kv_cache_spec",
     "SWA/chunked-local 准入上限：单一真相源", "§16.3"),
    ("hybrid",          1, 3, 1, "find_longest_cache_hit",
     "Hybrid 多注意力命中：不动点迭代，收敛即停", "§16.6"),
]
EDGES = [  # (src_id, dst_id) —— 调用边;同段同行=左→右主线蓝,同段跨行/跨段=竖直桥接蓝
    ("entry", "seqgate"),
    ("seqgate", "remove"),
    ("remove", "budget"),
    ("remove", "skip_calc"),          # 辅助:remove 内部第一步就查这张跳过表
    ("budget", "alloc_computed"),      # 段内换行:接着预算检查往下走阶段二
    ("alloc_computed", "alloc_new"),
    ("alloc_new", "exit"),
    ("exit", "factory"),               # 唯一跨段桥接:构造期↔运行期,配说明文字
    ("factory", "admission_cap"),
    ("factory", "hybrid"),
]
# 阅读顺序上的 6 个 § 站牌(§16.7 是纯小结,不引入新符号,不设站)。
READING_ORDER = ["§16.1", "§16.2", "§16.3", "§16.4", "§16.5", "§16.6"]
# (路线名, [§徽标,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("完整读：三阶段 + 协调层", READING_ORDER, True),
    ("只看多注意力怎么协调收敛", ["§16.1", "§16.5", "§16.6"], False),
]
LEGEND = [
    ("#22c55e", "入口：调度器为请求安排显存时调用"),
    ("#3b82f6", "章内主线调用边（末尾一条构造期↔运行期桥接）"),
    ("#f97316", "出口：返回 KVCacheBlocks 或 None（预算不够）"),
]
TITLE = "第 16 章 · KV 块分配与多注意力协调剖面（源码走线 + § 讲解站牌）"

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


NODE_H = 64
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
# 段与段之间的桥接带:留白,专放跨段箭头 + 一条说明文字(构造期↔运行期不是真实
# 调用,单靠一条箭头不够,必须有文字澄清关系,故这条带子比普通行距宽得多)。
INTER_LANE_GAP = 110

# 节点宽度:同一批节点统一宽度(保列对齐),按本章最长的符号名/短语算
_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16  # 左右各留:接口桩 + 一段箭头

n_cols = max(n[2] for n in NODES) + 1  # 两段各自独立复用这批列号(折行的关键)
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_band = [0] * len(LANES)
for _id, band, col, row, *_ in NODES:
    rows_per_band[band] = max(rows_per_band[band], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_band]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += INTER_LANE_GAP
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
    """§ 徽标胶囊,居中挂在 (cx,cy)——宽度按文字自适应(见 badge_width),
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

# 调用边:同段同行(band 且 row 都相同)= 左→右(方向按实际 x 序,自适应,不假设
# EDGES 书写顺序总是从左到右的节点)。其余情形(同段跨行 / 跨段)一律按"谁在上
# 谁在下"取竖直方向 attach(上者下沿中点→下者上沿中点),不看列号差——这类边
# 全部落在行与行(或段与段)之间的空白间隔里,不会穿过任何节点框(行/段内节点
# 只占各自的 y 区间,间隔区间没有节点),故不论列号相差多远都不会压框。
bridge_captions = []  # (x, y, text) —— 跨段桥接箭头旁的说明,渲后统一追加避免被箭头压住
for src, dst in EDGES:
    sband, scol, srow = NODE_BY_ID[src][1], NODE_BY_ID[src][2], NODE_BY_ID[src][3]
    dband, dcol, drow = NODE_BY_ID[dst][1], NODE_BY_ID[dst][2], NODE_BY_ID[dst][3]
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    if sband == dband and srow == drow:
        if x1 <= x2:
            p1, p2 = (x1 + NODE_W, y1 + NODE_H / 2), (x2, y2 + NODE_H / 2)
        else:
            p1, p2 = (x1, y1 + NODE_H / 2), (x2 + NODE_W, y2 + NODE_H / 2)
    else:
        if y1 <= y2:
            p1, p2 = (x1 + NODE_W / 2, y1 + NODE_H), (x2 + NODE_W / 2, y2)
        else:
            p1, p2 = (x1 + NODE_W / 2, y1), (x2 + NODE_W / 2, y2 + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if sband != dband:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        cap = "构造期选定的 self.coordinator；同期为 SWA/chunked-local 注入 §16.3 准入上限"
        bridge_captions.append((mx - cjk_text_width(cap, 12) / 2, my, cap))

for cx, cy, cap in bridge_captions:
    L.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="sans-serif" font-size="12" '
              f'font-style="italic" fill="{C_BRIDGE_CAPTION}">{esc(cap)}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
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

# 底部阅读路线:6 个 § 站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上
# 节点的段内列号——两段各自复用列 0..3,若路线条仍借列号,§16.1(上段col0)与
# §16.5(下段col3)这类不同段同列/不同列的站牌会互相错位或叠住)。时间轴左端
# 起点让给路线名文字(按最长路线名的实际宽度算,不留固定魔数空档)。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_first_badge_half_w = badge_width(READING_ORDER[0]) / 2
_route_left = 16 + _route_label_w + 24 + _first_badge_half_w
_n_stops = len(READING_ORDER)
_route_x = {name: _route_left + i * (w - PAD_R - _route_left) / (_n_stops - 1)
            for i, name in enumerate(READING_ORDER)}

L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
