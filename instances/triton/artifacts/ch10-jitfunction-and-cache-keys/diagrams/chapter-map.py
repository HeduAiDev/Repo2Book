#!/usr/bin/env python3
"""ch10「本章地图」——@triton.jit 到缓存键的源码剖面图(runtime/jit.py)。

本章是自然标题章(chapter.md 无 `## N.M` 编号,只有自然标题,如"两种写法，同一个
归宿"/"`__init__`：报到登记，零编译")——按契约禁用 §N.M 徽标,站牌一律改用标题
词本身的短摘要(如"报到登记"取自标题「`__init__`：报到登记，零编译」)。

剖面(左→右两段,对应正文「第一段：装饰期」/「第二段：发射期」):
  第 0 泳道(装饰期,只登记不编译):
    @triton.jit(两种写法收敛) → JITFunction.__init__(报到登记) → KernelParam(身份卡读注解)
  第 1 泳道(发射期,从实参到缓存键,一条主脊柱 + 两处并行原料 + 尾部双重校验):
    __getitem__(两步语法糖,只记 grid) → create_function_from_signature(exec 生成 binder)
    → 并行两支:compute_spec_key(特化位 D/1/N) 与 mangle_type(签名项 dtype 邮戳)
    → 汇合进 run() 拼缓存键(性能落点) → cache_key 源码哈希(另一把键) → used_global_vals 核对(最后一道关卡)
  装饰期 KernelParam 到发射期 __getitem__ 是唯一一条跨泳道边(不同列且不同泳道,
  走对角附着);col5 内三个节点(拼键/源码哈希/全局量核对)同列不同行,按正文
  三节顺序竖直连成一条尾链——这条尾链不再重复画进底部阅读路线(同列多点在
  水平路线上会重叠成一点,底部路线只挑跨列的路径)。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py 的不可变视觉语言
(徽标胶囊/入口绿-出口橙-主线蓝/高亮实线蓝-次要虚线灰/cjk_text_width)照搬,只改 DATA;
沿用 ch06 chapter-map.py 的两个通用文本适配工具(本章不少真实符号名较长,如
create_function_from_signature)：
  fit_size(text, max_w, sizes)   —— 从大到小试字号,选第一个能塞进 max_w 的
  wrap_symbol(text, max_w, sizes) —— 单行仍塞不下时,在 '_'/'(' 边界二分成两行
边路由改为按「两端实际 (x,y) 坐标」通用判定(不再依赖 lane 差):同列(x 相同)按
y 前后画竖直附着;不同列画对角附着——因为本章 col5 是「同泳道、不同行」的
纵向尾链(row0→row1→row2),ch06 原先按 lane 差分支的写法覆盖不到这种同泳道
多行场景。

六项自查(渲染→Read PNG 亲眼看后如实记录):见 figure-manifest.json 该图 selfcheck。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(text, max_w, sizes):
    """从大到小试字号,返回第一个能让 text 单行塞进 max_w 的字号;都不行则返回最小字号。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return size
    return sizes[-1]


def wrap_symbol(text, max_w, sizes):
    """符号名较长时的通用换行:先试单行从大到小的字号;仍塞不下,在 '_'/'(' 边界
    二分成两行(挑一个让两行里"更长的那行"最短的切点),用最小字号。返回 (lines, size)。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return [text], size
    size = sizes[-1]
    candidates = [i + 1 for i, c in enumerate(text) if c == '_'] + [i for i, c in enumerate(text) if c == '(']
    if not candidates:
        candidates = [len(text) // 2]
    best = None
    for idx in candidates:
        if idx <= 0 or idx >= len(text):
            continue
        a, b = text[:idx], text[idx:]
        w = max(cjk_text_width(a, size), cjk_text_width(b, size))
        if best is None or w < best[0]:
            best = (w, a, b)
    if best is None:
        return [text], size
    return [best[1], best[2]], size


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["装饰期:@triton.jit → JITFunction(零编译)", "发射期:从实参到缓存键"]

FONT_SIZES = (12.5, 11.5, 10.5, 9.5, 8.5)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌——自然标题摘要,禁用 §N.M)
NODES = [
    ("jit_deco",  0, 0, 0, "@triton.jit",                     "两种写法都收敛成同一个 JITFunction",     "两种写法"),
    ("init",      0, 1, 0, "__init__",                        "解析签名,切装饰器留 src",               "报到登记"),
    ("kparam",    0, 2, 0, "KernelParam",                     "按注解分流:constexpr 等三类",           "身份卡"),

    ("getitem",   1, 3, 0, "__getitem__",                     "fn[grid] 只记 grid,不发射",             "两步语法糖"),
    ("binder",    1, 4, 0, "create_function_from_signature",  "exec 生成 binder,摊薄发射开销",         "launch 快路径"),
    ("speckey",   1, 5, 0, "compute_spec_key",                "16 对齐压成 D/1/N 三桶",                "特化位"),
    ("mangle",    1, 5, 1, "mangle_type",                     "给实参盖类型邮戳(dtype 签名)",           "签名项"),
    ("cachekey",  1, 6, 0, "run()",                           "签名+特化位+常量值(性能落点)",           "缓存键"),
    ("srchash",   1, 6, 1, "cache_key",                       "源码哈希:含被调 kernel",                "源码哈希"),
    ("globalchk", 1, 6, 2, "used_global_vals",                "值变了就抛 RuntimeError",                "全局量核对"),
]
EDGES = [  # (src_id, dst_id) —— 调用/数据流边,统一主线蓝
    ("jit_deco", "init"), ("init", "kparam"),
    ("kparam", "getitem"),  # 唯一跨泳道边:装饰期收尾 → 发射期开局
    ("getitem", "binder"),
    ("binder", "speckey"), ("binder", "mangle"),
    ("speckey", "cachekey"), ("mangle", "cachekey"),
    ("cachekey", "srchash"), ("srchash", "globalchk"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 同列多点(col5/col6)不重复进同一条路线,否则水平路线上的站牌会挤在同一 x 上重叠。
ROUTES = [
    ("核心链:缓存键怎么拼",
     [(0, "两种写法"), (2, "身份卡"), (4, "launch 快路径"), (5, "特化位"), (6, "缓存键")], True),
    ("签名项那一支:mangle_type 怎么接进来",
     [(4, "launch 快路径"), (5, "签名项"), (6, "缓存键")], False),
    ("装饰期速览",
     [(0, "两种写法"), (1, "报到登记"), (2, "身份卡")], False),
    ("双保险:改代码/改全局量都会被拦下",
     [(4, "launch 快路径"), (6, "源码哈希")], False),
]
LEGEND = [
    ("#22c55e", "入口:用户写下 @triton.jit / 调用 fn[grid](args)"),
    ("#3b82f6", "章内主线:装饰期登记 → 发射期算缓存键"),
    ("#f97316", "出口:缓存键判定收尾,发射流程交给下一章"),
]
TITLE = "第 10 章 · JITFunction 装饰与缓存键剖面 —— python/triton/runtime/jit.py"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 168, 74
COL_GAP, ROW_GAP = 24, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 54, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_W, BADGE_H = 68, 20

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


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断(避免中文站牌被裁)。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——自然标题摘要,非 §N.M。"""
    bw = badge_w(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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
    _lx += 20 + cjk_text_width(label, 11.5) + 30

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

# 入口/出口接口桩:入口挂 jit_deco(最左),出口挂 globalchk(最右,发射尾链末端)
ex, ey = NODE_XY["jit_deco"]; ey += NODE_H / 2
xx, xy = NODE_XY["globalchk"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("用户代码")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝):按两端实际坐标通用判定——同列(x 相同)按 y 前后竖直附着;
# 不同列走对角附着(右中→左中)。不依赖 lane 差,能覆盖"同泳道、不同行"的
# 纵向尾链(col5 的 speckey→mangle 不存在此边,col6 的 cachekey→srchash→globalchk 属此类)。
for src, dst in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if xs_ != xd:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
    elif yd > ys_:
        p1 = (xs_ + NODE_W / 2, ys_ + NODE_H)
        p2 = (xd + NODE_W / 2, yd)
    else:
        p1 = (xs_ + NODE_W / 2, ys_)
        p2 = (xd + NODE_W / 2, yd + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(必要时自动换行/缩字号) + 一行短语 + 右上角站牌)
SYM_MAXW = NODE_W - 14
PHR_MAXW = NODE_W - 12
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines, sym_size = wrap_symbol(symbol, SYM_MAXW, FONT_SIZES)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.38:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.30:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.30 + sym_size + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    phr_size = fit_size(phrase, PHR_MAXW, (10.5, 9.5, 8.5, 8))
    L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.86:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{phr_size}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_w(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点(自然标题摘要,非 §N.M)
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌文字,对应正文对应小节;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
