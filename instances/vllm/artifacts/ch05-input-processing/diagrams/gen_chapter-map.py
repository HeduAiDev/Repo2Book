#!/usr/bin/env python3
"""第 5 章(Stage 1 输入处理:从 prompt 到 EngineCoreRequest)——本章地图:源码剖面图。

三条泳道,折成上中下三段(画布预算:宽 ≤1500 且宽高比 ≤2.6:1;9 个节点单行摆开要
9 列,横向严重超预算,故按内容天然分的三层——校验/归一化补料/组装派发——折成三行,
段内各自独立复用列号 0..2,画布宽度只由"段内最多 3 列"决定):
  上段(校验层)——从左到右:process_inputs() 入口做三道前置校验 → 分流(透传已渲染
    EngineInput / 兜底现场 preprocess())→ _validate_model_input 三类模型输入校验;
  中段(归一化补料层)——上段最右列(校验)竖直落到本段同列,再从右到左走完:取出
    prompt_token_ids/prompt_embeds → SamplingParams.clone() 补全 → argsort_mm_positions
    多模态展平(这样接回下段时又落在列 0,不用斜线);
  下段(组装派发层)——中段最左列竖直落到本段同列,再从左到右:组装 EngineCoreRequest
    → assign_request_id 注入随机后缀 → ParentRequest 处理 n>1 fan-out。
真实分流点在 assign_request_id 之后:n==1/池化直接返回,n>1 才继续走到 ParentRequest。
两个出口共用同一个"返回上层"接口桩——ParentRequest 走正常右向箭头;assign_request_id
另开一条经画布底部空白带的旁路折线(往下→横移→往上顶入桩底),避免穿过 ParentRequest
节点框,呼应正文"两条路都从 assign_request_id 之后分岔"的真实控制流。

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
LANES = ["校验层", "归一化补料层", "组装派发层"]  # 泳道→折成上中下三段,上→下

# (节点id, 段下标(0=校验/1=归一化补料/2=组装派发), 段内列, 段内行号, 真实符号名, 一行短语, §编号)
# 中段刻意按"列 2→1→0"倒序排列(段内从右到左),这样它与上段的衔接点(列2)、
# 与下段的衔接点(列0)都能画成直上直下的竖线,不需要斜线穿过别的节点。
NODES = [
    ("entry",      0, 0, 0, "process_inputs()", "入口:三道前置校验", "§5.2"),
    ("branch",     0, 1, 0, "InputPreprocessor", "透传EngineInput/兜底tokenize", "§5.3"),
    ("modelcheck", 0, 2, 0, "_validate_model_input", "长度/mm超限/token越界三种校验", "§5.4"),
    ("extract",    1, 2, 0, "prompt_token_ids", "三种载体(token/embeds)统一访问", "§5.5"),
    ("clone",      1, 1, 0, "SamplingParams.clone()", "补max_tokens/eos·bad_words", "§5.6"),
    ("mmflatten",  1, 0, 0, "argsort_mm_positions", "按offset排序展平", "§5.7"),
    ("assemble",   2, 0, 0, "EngineCoreRequest", "组装msgspec.Struct,备IPC", "§5.8"),
    ("assignid",   2, 1, 0, "assign_request_id", "注入8字符随机后缀,原id留存", "§5.9"),
    ("fanout",     2, 2, 0, "ParentRequest", "n>1时裂出n子,聚合子输出", "§5.10"),
]
EDGES = [  # (src_id, dst_id) —— 调用边;同段=段内主线蓝,跨段=竖直桥接蓝
    ("entry", "branch"), ("branch", "modelcheck"),
    ("modelcheck", "extract"),          # 跨段(校验→归一化补料),同列2,竖直下落
    ("extract", "clone"), ("clone", "mmflatten"),  # 段内从右到左
    ("mmflatten", "assemble"),          # 跨段(归一化补料→组装派发),同列0,竖直下落
    ("assemble", "assignid"), ("assignid", "fanout"),
]
# assignid → fanout 这条边旁加一句小字,点破"往下走是因为 n>1"
EDGE_NOTES = {("assignid", "fanout"): "n>1"}
# 阅读顺序上的 9 个 § 站牌,用于底部阅读路线的独立时间轴——折行后同一列号被三段
# 各用一次,若路线条仍复用列号,不同段的站牌会叠在同一 x 位置。
READING_ORDER = ["§5.2", "§5.3", "§5.4", "§5.5", "§5.6", "§5.7", "§5.8", "§5.9", "§5.10"]
# (路线名, [§编号,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("单请求主线(n=1/池化)", READING_ORDER[:-1], True),
    ("并行采样 fan-out (n>1)", READING_ORDER, False),
]
LEGEND = [
    ("#22c55e", "入口:从 add_request 调用进入"),
    ("#3b82f6", "章内主线调用边"),
    ("#f97316", "出口:n=1 直接返回 / n>1 聚合后返回"),
]
TITLE = "第5章 · process_inputs() 校验-归一化-组装-派发剖面(源码走线 + § 讲解站牌)"

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


NODE_H = 62
COL_GAP, ROW_GAP = 32, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
INTER_LANE_GAP = 30  # 段与段之间的空白(校验/归一化补料/组装派发衔接点都同列,竖线足矣,不需要大间隔)
BYPASS_GAP = 46      # 组装派发段下方专留给 assign_request_id 旁路折线的空白带

_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16

n_cols = max(n[2] for n in NODES) + 1  # 段内最多列数(三段各自独立复用这批列号)
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

routes_top = lanes_bottom + BYPASS_GAP + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy)——宽度按文字自适应(见 badge_width),
    颜色/圆角/描边视觉语言与模板一致。"""
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

# 泳道背景 + 标签
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
    L.append(f'<line x1="0" y1="{band_top[i] + band_h[i]:.1f}" x2="{w:.1f}" y2="{band_top[i] + band_h[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口接口桩(左侧,贴在 entry 节点)
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩(右侧,贴在最终节点 fanout 的行高;assignid 的旁路折线也扎进同一个桩)
xx, xy = NODE_XY["fanout"]; xy += NODE_H / 2
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
# 正常出口:fanout(n>1 聚合完毕) 右中 → 桩左中
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边:同段(band 相同)按左右相对位置接右中/左中;跨段(band 不同)接上下沿中点,
# 竖直下落(本章三段的衔接列刻意对齐,不会画成斜线穿过别的节点)。
for src, dst in EDGES:
    src_band = NODE_BY_ID[src][1]
    dst_band = NODE_BY_ID[dst][1]
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    if src_band == dst_band:
        if x1 <= x2:
            p1 = (x1 + NODE_W, y1 + NODE_H / 2)
            p2 = (x2, y2 + NODE_H / 2)
        else:
            p1 = (x1, y1 + NODE_H / 2)
            p2 = (x2 + NODE_W, y2 + NODE_H / 2)
    elif dst_band > src_band:
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:
        p1 = (x1 + NODE_W / 2, y1)
        p2 = (x2 + NODE_W / 2, y2 + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    note = EDGE_NOTES.get((src, dst))
    if note:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        L.append(f'<text x="{mx:.1f}" y="{my - 6:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" font-style="italic" fill="{C_BRIDGE_CAPTION}">{esc(note)}</text>')

# assignid 的旁路折线:n==1/池化不经过 ParentRequest,直接从 assign_request_id
# 下沿绕画布底部空白带,顶入出口桩底部——不与 fanout 节点框重叠。
ax, ay = NODE_XY["assignid"]
bypass_x = ax + NODE_W / 2
bypass_start = (bypass_x, ay + NODE_H)
elbow_y = lanes_bottom + BYPASS_GAP * 0.5
stub_cx = sx + STUB_W / 2
stub_bottom = xy + STUB_H / 2
bypass_pts = [bypass_start, (bypass_x, elbow_y), (stub_cx, elbow_y), (stub_cx, stub_bottom)]
pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in bypass_pts)
L.append(f'<polyline points="{pts_str}" fill="none" stroke="{C_EXIT}" stroke-width="2" '
          f'marker-end="url(#mExit)"/>')
L.append(f'<text x="{(bypass_x + stub_cx) / 2:.1f}" y="{elbow_y - 8:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-style="italic" '
          f'fill="{C_BRIDGE_CAPTION}">{esc("n=1/池化:提前返回,不入 ParentRequest")}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, band, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W:.1f}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.4:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_SYMBOL_FONT}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_PHRASE_FONT}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_width(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:9 个 § 站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上
# 节点的段内列号——折行后同一列号被三段各用一次,若仍借列号会让不同段的站牌叠在
# 同一 x 位置)。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_route_left = 16 + _route_label_w + 24
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
