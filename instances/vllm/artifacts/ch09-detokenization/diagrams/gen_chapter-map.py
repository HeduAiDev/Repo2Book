#!/usr/bin/env python3
"""第 9 章「本章地图」——增量去 token 化与 stop string 源码剖面图。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py。本章 7 个内容
分节(9.1-9.7)对应 9 个真实符号节点，若按单行铺开需要 7 列——按模板画布预算
(宽 ≤1500 且宽高比 ≤2.6:1)会超；改用 ch24-primer-flash-attention 那版
gen_chapter-map.py 引入的「折成上下两段」布局(带专属跨段桥接边)，但本章不是
primer 章、两段之间是真实的调用/派发关系(不是论文↔代码的黑盒潜入)，故：

  上段(构造期，§9.1-§9.2，3 列)：from_new_request 三路分派 → Fast/Slow 两个
    具体子类(哪个被构造，决定了下面 decode_next 走哪条) → update 每步主循环。
  下段(运行期解码与收口，§9.3-§9.7，4 列)：update 内部按对象类型二选一分派到
    decode_next 的 Fast/Slow 实现 → check_stop_strings 窗口化查 stop →
    get_next_output_text 的 holdback → process_outputs 收口(改写 finish_reason /
    回灌 reqs_to_abort)。
  两段之间的桥接边(update → 两个 decode_next)是真实的虚方法分派(update 调用
  self.decode_next，运行期落到 Fast 或 Slow 的实现)，样式仍是章内主线蓝实线，
  不用 ch24 那种"非调用换段"灰虚线——因为这里确实是一次函数调用，只是画布折行
  把调用双方分到了两段。

■ 不可变(全书统一视觉语言，抄自模板，未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要，未改动上面的不可变部分):
  - Fast/Slow 两条后端路径在构造期(§9.1)和解码期(§9.5/§9.6)各自都有专属节点——
    没有合并成一个节点身兼两个 badge，避免 ch39 复盘记录过的教训("一个节点挂两个
    不相关小节的 badge，盲审顺着后一个 badge 跳过去发现符号对不上")。
  - 底部阅读路线用与图上节点无关的独立"位置索引"(POSITIONS，非节点列号)做 x 轴
    坐标：因为折行后上段列号 0..2 与下段列号 0..3 是各自独立复用的同一批数字，
    若路线条直接借列号，"构造期 §9.1"(上段列0)和"解码后端 §9.6"(下段列0)会被
    画在同一 x 位置、彼此重叠。改成 6 个逻辑位置(构造→主循环→解码后端→查stop→
    产出→收口)，Fast/Slow 两条路线在"解码后端"这一位置各自显示自己的 §(9.6/9.5)，
    其余 5 个位置两条路线共享同一 x、同一 §，视觉上两条线在此并列对齐、只在
    解码后端一站分叉成不同 badge。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用，非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size，半角(ASCII/拉丁等)按 0.58×size，求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["构造期 · 三路分派 → 主循环入口(§9.1-§9.2)", "运行期 · 解码后端 → 查stop → 产出 → 收口(§9.3-§9.7)"]

# (节点id, 段下标(0=上段构造期/1=下段运行期), 段内列, 段内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("entry",       0, 0, 0, "from_new_request",
     "按 tokenizer 类型/版本三路分派", "§9.1"),
    ("fast_ctor",   0, 1, 0, "FastIncrementalDetokenizer",
     "Fast tokenizer 且 tokenizers≥0.22.0", "§9.1"),
    ("slow_ctor",   0, 1, 1, "SlowIncrementalDetokenizer",
     "其余情况兜底，纯 Python 慢路径", "§9.1"),
    ("update",      0, 2, 0, "update",
     "逐 token decode，min_tokens 双闸推进", "§9.2"),
    ("fast_decode", 1, 0, 0, "decode_next",
     "Fast：_protected_step 驱动 DecodeStream", "§9.6"),
    ("slow_decode", 1, 0, 1, "decode_next",
     "Slow：双 offset 窗口对抗空格清理", "§9.5"),
    ("check_stop",  1, 1, 0, "check_stop_strings",
     "窗口内 find，算截断 offset", "§9.4"),
    ("holdback",    1, 2, 0, "get_next_output_text",
     "尾部扣 stop_buffer_length 字符", "§9.3"),
    ("exit",        1, 3, 0, "process_outputs",
     "stop_string→finish_reason，回灌 reqs_to_abort", "§9.7"),
]
EDGES = [  # (src_id, dst_id) —— 调用边，统一主线蓝；update→两个 decode_next 是跨段的真实虚方法分派
    ("entry", "fast_ctor"), ("entry", "slow_ctor"),
    ("fast_ctor", "update"), ("slow_ctor", "update"),
    ("update", "fast_decode"), ("update", "slow_decode"),
    ("fast_decode", "check_stop"), ("slow_decode", "check_stop"),
    ("check_stop", "holdback"),
    ("holdback", "exit"),
]
# 阅读路线用的逻辑位置(与图上节点列号无关，见文件头说明)：
# 0=构造期 1=主循环 2=解码后端(Fast/Slow 分叉) 3=查stop 4=产出holdback 5=收口
POSITIONS = [0, 1, 2, 3, 4, 5]
# (路线名, [(位置, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("Fast 路径(多数请求)", [(0, "§9.1"), (1, "§9.2"), (2, "§9.6"), (3, "§9.4"), (4, "§9.3"), (5, "§9.7")], True),
    ("Slow 路径(兜底)",     [(0, "§9.1"), (1, "§9.2"), (2, "§9.5"), (3, "§9.4"), (4, "§9.3"), (5, "§9.7")], False),
]
LEGEND = [
    ("#22c55e", "入口：从上层构造/调用进入"),
    ("#3b82f6", "章内主线调用边"),
    ("#f97316", "出口：返回上层（回灌 EngineCore）"),
]
TITLE = "第 9 章 · 增量去 token 化与 stop string 剖面（源码走线 + § 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 段背景交替，仅装饰，非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_BRIDGE_CAPTION = "#475569"

# ---------------- 几何常量(全计算，零魔数) ----------------
_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 24
NODE_H = 60
COL_GAP, ROW_GAP = 34, 18
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 22, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 24, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_W, BADGE_H = 46, 20
INTER_BAND_GAP = 130  # 两段之间的桥接带（放跨段箭头 + 简短说明文字）

NODE_W = max(
    170,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24  # 左右各留：接口桩 + 一段箭头

n_cols = max(n[2] for n in NODES) + 1  # 段内最多列数（两段各自独立复用这批列号）
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_band = [0, 0]
for _id, band, col, row, *_ in NODES:
    rows_per_band[band] = max(rows_per_band[band], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_band]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += INTER_BAND_GAP
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
    """§ 徽标胶囊，居中挂在 (cx,cy)——节点用它贴右上角，路线条复用它居中挂线上。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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

# 段背景 + 标签(桥接带本身不上色，留白给跨段箭头)
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
    L.append(f'<line x1="0" y1="{band_top[i] + band_h[i]:.1f}" x2="{w:.1f}" y2="{band_top[i] + band_h[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
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

# 调用边:同段(band 相同)=段内左→右，右中→左中；跨段(band 不同)=桥接带上下沿，
# 上中/下中 attach(不经过任何节点框内部，桥接带本身是留白区)。多条边汇入同一
# 节点时终点 y 各偏移，避免"汇合处看不出汇合、像断头线"。
bridge_captions = []
_dst_total = {}
for _e in EDGES:
    _dst_total[_e[1]] = _dst_total.get(_e[1], 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    src_band = NODE_BY_ID[src][1]
    dst_band = NODE_BY_ID[dst][1]
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    if src_band == dst_band:
        y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2 + y_offset)
    else:  # 跨段:上段→下段(本章桥接边方向恒定，update 在上段、两个 decode_next 在下段)
        x_offset = (i - (n - 1) / 2) * 20 if n > 1 else 0
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2 + x_offset, y2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if src_band != dst_band:
        bridge_captions.append(True)

if bridge_captions:
    # 两条跨段边(update→Fast/Slow decode_next)共用同一句说明，只画一次；固定放在
    # 桥接带左侧、左对齐——两条线都从 update 底边中点(x≈912)向左下方的
    # decode_next 扇出，在 y∈[254,384] 这段桥接带范围内线的 x 坐标恒 >340，
    # 贴左边缘(x=16)写字绝不会被线压住(见文件头 6 的画法说明)。
    _cap_y = (band_top[0] + band_h[0] + band_top[1]) / 2
    L.append(f'<text x="16" y="{_cap_y:.1f}" font-family="sans-serif" '
              f'font-size="11.5" font-style="italic" fill="{C_BRIDGE_CAPTION}">'
              f'{esc("update 每步调 self.decode_next，按对象类型落到 Fast/Slow 实现")}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, band, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W:.1f}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_SYMBOL_FONT}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.74:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_PHRASE_FONT}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:用独立的逻辑位置(POSITIONS)算 x，不复用图上节点的段内列号——
# 折行后上段列号与下段列号是各自独立复用的同一批数字，若路线条直接借列号，
# "构造期"(上段列0)和"解码后端"(下段列0)会被画在同一 x 位置、彼此重叠。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_route_left = 16 + _route_label_w + 24
_n_pos = len(POSITIONS)
_pos_x = {p: _route_left + p * (w - PAD_R - _route_left) / (_n_pos - 1) for p in POSITIONS}

L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first, x_last = _pos_x[stops[0][0]], _pos_x[stops[-1][0]]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for pos, sec in stops:
        L += badge(_pos_x[pos], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f})")
