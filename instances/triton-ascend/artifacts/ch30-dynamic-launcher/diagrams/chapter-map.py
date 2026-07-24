#!/usr/bin/env python3
"""ch30 本章地图:动态发射器剖面——driver.py 首次用时现拼 C++ wrapper 源码、即时编译成
.so、dlopen 拿到可调用体,一次 kernel 发射从解包 args 到真正上设备派发的定序流水。

本章 13 节为自然标题(无 §N.M 编号),故站牌徽标改用节序 一~十三(禁用 §N.M 徽标——
lint_chapter_map 校验)。13 节较多,走线折成三行蛇形泳道(上排 L→R、中排 R→L、下排
L→R),按节序 一→十三 连成一条发射流水;宽度压在页宽预算内。

模板来源:.claude/skills/svg-diagram/references/example-chapter-map.py 与上一章
ch29/diagrams/chapter-map.py(§徽标胶囊/入口绿-出口橙-主线蓝/cjk_text_width() 不可变;
几何常量按 13 节 / 三行蛇形调过)。

六项自查记录(渲染→Read PNG 亲眼看后如实记录):
  claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
  arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  (无 spec.numbers——本图是源码剖面/站牌图,不含数值型 claim,numbers_match_spec
  按"图上无数字断言"判定为 True;560/三样 等在正文,图上不复述具体数值。)

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
# 三行蛇形泳道:每行一段发射流水阶段(标签描述本行经过的站牌)
BANDS = [
    "① 编译一次:现拼发射器 → 报关格式单 · workspace 现分配 · 归零",
    "② 装箱 args → 真正上设备 → 异步派发 → 定序流水 → 剖析",
    "③ 双后端策略表 · __call__ 两条旁路 · 收官",
]
NCOL = 5  # 每行最多列数(蛇形宽度)

# (节点id, 行, 行内位置pos, 真实符号名, 一行短语, 节序徽标)
# 行 0/2 从左到右:col = pos;行 1 从右到左:col = NCOL-1-pos。
NODES = [
    # —— 上排 一~五(L→R) ——
    ("n1",  0, 0, "NPULauncher",              "发射器一生:只编译一次",   "一"),
    ("n2",  0, 1, "generate_npu_wrapper_src", "现拼 C++ wrapper→dlopen", "二"),
    ("n3",  0, 2, "PyArg_ParseTuple",         "format 报关格式单解包",   "三"),
    ("n4",  0, 3, "workspace_size",           "按 grid 现分配 workspace","四"),
    ("n5",  0, 4, "syncBlockLock",            "发令枪前先归零",           "五"),
    # —— 中排 六~十(R→L,六接在五的正下方) ——
    ("n6",  1, 0, "rtArgsEx_t",               "packed struct 装箱 args", "六"),
    ("n7",  1, 1, "rtKernelLaunch",           "真正上设备发射",           "七"),
    ("n8",  1, 2, "taskqueue",                "同步 / 异步派发",          "八"),
    ("n9",  1, 3, "_launch",                  "launch C 入口定序流水",    "九"),
    ("n10", 1, 4, "msprof",                   "内嵌剖析钩子",             "十"),
    # —— 下排 十一~十三(L→R,十一接在十的正下方) ——
    ("n11", 2, 0, "torch_npu",                "双后端:一套模板两框架",   "十一"),
    ("n12", 2, 1, "__call__",                 "两条旁路",                 "十二"),
    ("n13", 2, 2, "小结",                     "厚出三样:分配/异步/剖析", "十三"),
]
# 顺次连成一条发射流水(节序 一→十三);类型 R=向右 / L=向左 / D=向下
EDGES = [
    ("n1", "n2", "R"), ("n2", "n3", "R"), ("n3", "n4", "R"), ("n4", "n5", "R"),
    ("n5", "n6", "D"),
    ("n6", "n7", "L"), ("n7", "n8", "L"), ("n8", "n9", "L"), ("n9", "n10", "L"),
    ("n10", "n11", "D"),
    ("n11", "n12", "R"), ("n12", "n13", "R"),
]
# 底部阅读路线:(名称, [徽标...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程(推荐):一次发射从拼到派发", ["一", "二", "六", "七", "八", "九"], True),
    ("跳读:只看真正上设备与派发",     ["一", "七", "八"], False),
    ("旁支:剖析钩子与后端旁路",       ["十", "十一", "十二"], False),
]
LEGEND = [("#22c55e", "入口:上一章拿到 func 句柄"),
          ("#3b82f6", "章内发射流水调用边"),
          ("#f97316", "出口:编译→运行时主线收官")]
TITLE = "第 30 章 · 动态发射器剖面(driver.py 现拼 C++ wrapper → rtKernelLaunch 上设备)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 176, 58
COL_GAP, ROW_GAP = 24, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 62, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 28  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 22, 10
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 12, 30, 24, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 20, 34
BADGE_W, BADGE_H = 44, 19
SYMBOL_FS = 11  # 最长符号 generate_npu_wrapper_src 在 NODE_W 内可容

COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(NCOL)]
band_h = LANE_LABEL_H + BAND_PAD * 2 + NODE_H
band_top = [TOP_PAD + TITLE_H + LEGEND_H + r * band_h for r in range(len(BANDS))]
lanes_bottom = band_top[-1] + band_h


def node_col(row, pos):
    return pos if row != 1 else NCOL - 1 - pos


NODE_XY = {}
for nid, row, pos, *_ in NODES:
    x = COLX[node_col(row, pos)]
    y = band_top[row] + LANE_LABEL_H + BAND_PAD
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + NCOL * NODE_W + (NCOL - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """节序徽标胶囊,居中挂在 (cx,cy)。节点用它贴右上角,路线legend用它居中挂线上。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10.5" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 13
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 10}" width="13" height="13" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 18}" y="{_ly}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 18 + cjk_text_width(label, 10.5) + 26

# 泳道背景 + 行标签 + 分隔线
for i, name in enumerate(BANDS):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="14" y="{band_top[i] + LANE_LABEL_H - 5:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口接口桩(far-left,挂在首节点 n1 那一行)
ex, ey = NODE_XY["n1"]; ey += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 3.8:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("上一章")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩(far-right,挂在末节点 n13 那一行;n13 右侧本行为空,横向引出不压框)
xx, xy = NODE_XY["n13"]; xy += NODE_H / 2
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 3.8:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("收官")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 发射流水调用边(主线蓝),按边类型算附着点
for src, dst, kind in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    if kind == "R":      # 向右:src.right → dst.left
        p1 = (x1 + NODE_W, y1 + NODE_H / 2); p2 = (x2, y2 + NODE_H / 2)
    elif kind == "L":    # 向左:src.left → dst.right(marker orient=auto 自动指左)
        p1 = (x1, y1 + NODE_H / 2); p2 = (x2 + NODE_W, y2 + NODE_H / 2)
    else:                # 向下:src.bottom → dst.top(同列,直下)
        p1 = (x1 + NODE_W / 2, y1 + NODE_H); p2 = (x2 + NODE_W / 2, y2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角节序徽标)
for nid, row, pos, symbol, phrase, badge_txt in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{SYMBOL_FS}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9.6" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 7, y, badge_txt)

# 底部阅读路线:徽标沿路线线段等距铺开(蛇形无法竖向对列,故按顺序均匀排)
L.append(f'<text x="14" y="{routes_top + 14:.1f}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上节序站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
route_x0 = 260   # 路线线段起点(给左侧路线名留位)
route_x1 = w - 40
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="14" y="{ry + 3.8:.1f}" font-family="sans-serif" font-size="11" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    n = len(stops)
    xs_pts = [route_x0 + (route_x1 - route_x0) * (k / max(1, n - 1)) for k in range(n)]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{xs_pts[0]:.1f}" y1="{ry:.1f}" x2="{xs_pts[-1]:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for k, sec in enumerate(stops):
        L += badge(xs_pts[k], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  canvas={w:.0f}x{h:.0f}  ratio={w/h:.2f}")
