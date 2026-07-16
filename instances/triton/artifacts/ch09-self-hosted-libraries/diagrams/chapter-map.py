#!/usr/bin/env python3
"""本章地图 — ch09《在 Triton 里写 Triton：自举标准库、数学/extern 与随机数》源码剖面图。

本章证明 tl.* 的上半部不是黑盒——四条并列的自举脉络，按 chapter.md 实际的
「一/二/三/四」四段叙事，各占一条泳道:
  一 · 自举的标准库(standard.py):cdiv 最小自举例 → 内联膨胀的洞见 → softmax
      数值稳定 → sort 的 bitonic 定长展开 → cumsum 借用 scan。
  二 · 无状态的 Philox(random.py):counter-based RNG → 刻意环绕算术。
  三 · 数学的两条路(math.py/core.py):内置建 IR → extern 链 libdevice → extra/
      的 pkgutil 后端插座。
  四 · 编译期诊断与优化提示(core.py):static_assert/print 追踪期诊断 →
      multiple_of/max_contiguous 给编译器贴标签(性能杠杆②)。

■ 不可变(全书统一视觉语言):
  1. 站牌胶囊:圆角矩形(pill),fill #eef2ff / stroke #6366f1;贴节点右上角。
  2. 入口/出口接口桩:绿 #22c55e(入口) / 橙 #f97316(出口)。
  3. 节点间主线边 = 蓝 #3b82f6。
  4. 底部路线条:高亮=实线蓝(粗)/次要=虚线灰 #94a3b8(细)。
  5. >2 种语义色须画图例。
  6. 文本宽度估算一律用 cjk_text_width(),不用半角系数硬乘 len(s)。

■ 本章特有(自然标题章,无 §N.M 编号——按 illustrator 契约:禁用 §N.M 徽标,
  站牌改用标题词本身,逐字取自 chapter.md 真实 `## ...`/`### ...` 标题的子串):
  - 12 个节点的站牌文本均为对应小节标题的逐字子串(如 "内联，不是调用" 取自
    `## 内联，不是调用：@jit 库函数在你的 IR 里铺开`);4 条泳道名取自对应的
    `# 一/二/三/四、...` 顶层分段标题子串。
  - 本章 4 段叙事彼此独立(没有真实调用关系——cdiv/softmax/sort/cumsum 是
    并列的标准库函数,不互相调用),泳道内的蓝色主线边表示"章内阅读顺序/同段
    展开关系",不是调用图;图例文案已按此措辞,避免误导读者以为它们互相调用。
  - 节点符号名尽量给出真实可检索的函数名(如 `sort / _bitonic_merge`,不带空括号
    ——lint_chapter_map 的杜撰符号检测按字面子串核对,`sort()` 这种空括号写法在
    正文/dossier 里从不出现,只有 `sort(x, ...)` 这类带参数的调用,故一律去掉
    符号名里的 `()`,与已有的 ch08 chapter-map 先例一致);
    "内联" 一站没有单一函数可代表这个洞见,故用章节实测使用的真实 IR 属性名
    `tt.call 恒为 0` 作为符号(可在正文 IR 表与 dossier m2 anchor 直接核对)。
  - 底部两条路线直接复用章节开篇给出的选读指引:"只想拿走两条结论,直接看
    「内联，不是调用」和「给编译器贴张标签」两节"(杠杆速览,高亮实线),
    以及"想看 tl.* 怎么被拼出来的,按序读"(按序通读,虚线,取一/二/三/四
    各一个代表站作为进度提示,列号严格单调)。
  - badge()/BADGE_W 改为按文本动态算宽(cjk_text_width + 内边距),因为本章
    站牌是完整词组(最长 8 字)而非 "§20.1" 这类定长短码,固定 46px 会溢出。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
  claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
  arrows_attached=True     cjk_rendered=True         reading_order_clear=True

用法: python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(text, max_w, base, min_size):
    """按 max_w 反解一个不超出的字号(单行,不换行)。"""
    unit = cjk_text_width(text, 1.0)
    if unit <= 0:
        return base
    return max(min_size, min(base, max_w / unit))


# ==================== DATA(可变:本章数据) ====================
LANES = [
    "一 · 自举的标准库",
    "二 · 无状态的 Philox",
    "三 · 数学的两条路",
    "四 · 编译期诊断与优化提示",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本[取自真实标题子串])
NODES = [
    ("cdiv",       0, 0, 0, "cdiv",
     "纯算术三步:(x+div-1)//div",              "最小的自举原子"),
    ("inline",     0, 1, 0, "tt.call 恒为 0",
     "@jit 函数体整段内联进 kernel IR",          "内联，不是调用"),
    ("softmax",    0, 2, 0, "softmax",
     "z = x - max(x,0) 防 exp 溢出",             "数值稳定"),
    ("sort",       0, 3, 0, "sort / _bitonic_merge",
     "constexpr 铺 log2(n) 个阶段",              "铺平的排序网络"),
    ("cumsum",     0, 4, 0, "cumsum",
     "借 associative_scan(回指第 8 章)",         "把 scan 借过来"),

    ("philox",     1, 0, 0, "philox_impl / randint4x",
     "counter→随机数,10 轮 umulhi+异或",         "计数器就是状态"),
    ("sanitize",   1, 1, 0, "sanitize_overflow=False",
     "刻意 mod 2^32 环绕(对照第 7 章)",          "故意让它溢出"),

    ("builtin",    2, 0, 0, "exp / umulhi",
     "create_exp 直接建原生 IR 节点",            "第一条路"),
    ("extern",     2, 1, 0, "extern_elementwise / dispatch",
     "按 dtype 元组查表→链 libdevice",           "第二条路"),
    ("extra",      2, 2, 0, "extra/__init__.py",
     "pkgutil 动态发现 cuda/hip 后端子包",       "后端的插座"),

    ("static",     3, 3, 0, "static_assert / static_print",
     "追踪期诊断,函数体是 pass",                 "追踪期就说话"),
    ("multipleof", 3, 4, 0, "multiple_of / max_contiguous",
     "打 divisibility/contiguity 标记",          "给编译器贴张标签"),
]

EDGES = [  # (src_id, dst_id) —— 章内阅读顺序/同段展开关系,非调用图,统一主线蓝
    ("cdiv", "inline"), ("inline", "softmax"), ("softmax", "sort"), ("sort", "cumsum"),
    ("philox", "sanitize"),
    ("builtin", "extern"), ("extern", "extra"),
    ("static", "multipleof"),
]

# (路线名, [(列, 站牌文本), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("杠杆速览:内联膨胀 → 编译提示",
     [(1, "内联，不是调用"), (4, "给编译器贴张标签")], True),
    ("按序通读:一 → 二 → 三 → 四",
     [(0, "最小的自举原子"), (1, "故意让它溢出"), (2, "后端的插座"), (3, "追踪期就说话")], False),
]
LEGEND = [
    ("#22c55e", "入口:从上一章 tl.dot/combine_fn 编成 IR 而来"),
    ("#3b82f6", "章内阅读顺序(同段自举/展开关系,非调用图)"),
    ("#f97316", "出口:下一章进宿主运行时(JITFunction/缓存)"),
]
TITLE = "第 9 章 · 自举标准库/Philox/两条数学路 剖面（源码走线 + 讲解站牌）"

# ==================== 不可变:配色 ====================
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ==================== 几何常量(全计算,零魔数) ====================
NODE_W, NODE_H = 200, 64
COL_GAP, ROW_GAP = 34, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 66, 30
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 26
LANE_LABEL_H, BAND_PAD = 22, 10
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 24, 14
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_PAD_X = 20, 10
BADGE_FONT = 11

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
    """站牌胶囊,居中挂在 (cx,cy)——宽度按文本动态算(本章站牌是完整词组,非定长短码)。"""
    bw = cjk_text_width(text, BADGE_FONT) + BADGE_PAD_X * 2
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.1"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')

# 图例(3 种语义色 → 必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 13
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 30

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["cdiv"]; ey += NODE_H / 2
xx, xy = NODE_XY["multipleof"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.2"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.2"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 章内主线(阅读顺序/展开关系),多条边汇入同一节点时终点 y 错开(本章无汇入 >1 的情形,仍保留通用逻辑)
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
    y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌),字号按文本长度自适应收缩避免溢出
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_size = fit_size(symbol, NODE_W - 18, 13, 8.5)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sym_size:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    ph_size = fit_size(phrase, NODE_W - 16, 10.5, 7.8)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{ph_size:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = cjk_text_width(sec, BADGE_FONT) + BADGE_PAD_X * 2
    L += badge(x + NODE_W - bw / 2 + 10, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上讲解站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio {w / h:.2f}:1)")
