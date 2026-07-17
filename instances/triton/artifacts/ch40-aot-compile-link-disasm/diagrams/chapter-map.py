#!/usr/bin/env python3
"""ch40「本章地图」——AOT compile/link 与 SASS 反汇编的源码剖面图
(python/triton/tools/compile.py + python/triton/tools/link.py
 + python/triton/tools/disasm.py + 相关 compiler.py/code_generator.py 挂载点)。

本章是自然标题章(chapter.md 无 `## N.M` 编号,标题用 §1..§10 单层记号,如
"§4 函数名当暗号：kernel_suffix ↔ _match_suffix")——按契约禁用 §N.M 徽标(带
小数点的两级编号),站牌沿用章节自带的单层记号"§N",与 ch14 同一条"自然标题"
规则下的呈现方式(§N 字面就出现在标题文本里,不是虚构的子章节)。

剖面(三条泳道,主脊在中间——布局仿 ch14:主分支在中间 lane,两条下游分支各占
上/下 lane,靠"只跨一条相邻泳道边界"的竖向边接入,不做跨两条泳道的斜线,
避免走线穿过中间节点):
  ① 中间(主脊,compile.py):constexpr() 三分签名(§1)→ from_hints 物化特化(§2)
     → hexlify 内嵌 cubin 进 C(§3)——这是全章唯一的入口,从这里发生两条分支。
  ② 上方(link.py):从 §3 向上一条真实数据流边(函数名+cubin 元数据)喂给
     kernel_suffix/_match_suffix 编解码对(§4)→ HeaderParser 捞漂流瓶(§5)
     → 运行期整除性分派链(§6)→ algo_id 函数指针二级分派(§7)——这条链的终点
     是「脱离 Python」的成品:一份不依赖 Python 的 .so。
  ③ 下方(disasm.py):从 §3 向下一条真实数据流边(cubin 字节)喂给
     cuobjdump 两行一指令解析(§8)→ 64 位控制字解码(§9)→ BRA→LBB 重标(§10)
     ——这条链的终点是「读懂产物」:看懂 profiler 里那串调度控制码。

配色(全书统一):绿 #22c55e=入口(命令行调用触发 AOT 流程);蓝 #3b82f6=主线
调用边 + compile.py 产物喂给两条下游分支的真实数据流边;橙 #f97316=出口——
本章有两个出口(link.py 链尾→已部署 / disasm.py 链尾→已读懂),分列右侧上下,
不合并成一个虚构的"终点节点"(两条分支本就互相独立,不存在真实的汇合函数)。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py 的不可变视觉
语言(徽标胶囊/入口绿-出口橙-主线蓝/cjk_text_width)+ ch11/ch12/ch14
chapter-map.py 的 badge_w/wrap_symbol(长符号名自适应换行/缩字号,零手写魔数)。

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


def wrap_symbol(text, max_w, sizes):
    """符号名/短语较长时的通用换行:先试单行从大到小的字号;仍塞不下,在
    ' '/'_'/'('/',' 边界二分成两行(挑一个让两行里"更长的那行"最短的切点),
    用最小字号。返回 (lines, size)。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return [text], size
    size = sizes[-1]
    candidates = ([i + 1 for i, c in enumerate(text) if c == ' ']
                  + [i + 1 for i, c in enumerate(text) if c == '_']
                  + [i for i, c in enumerate(text) if c == '(']
                  + [i + 1 for i, c in enumerate(text) if c == ','])
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
LANES = [
    "link.py：暗号与分派——脱离 Python 下半场",
    "compile.py：报关与烙铸——脱离 Python 上半场(全章唯一入口)",
    "disasm.py：读懂 SASS",
]

FONT_SIZES = (12.5, 11.5, 10.5, 9.5, 8.5)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    # ① 中间主脊:compile.py
    ("s1", 1, 0, 0, "constexpr(s)",
     "报关单:一行签名切成 hints/constants/signature", "§1"),
    ("s2", 1, 1, 0, "AttrsDescriptor.from_hints",
     "hints 进货口:命令行数字物化成特化属性", "§2"),
    ("s3", 1, 2, 0, "binascii.hexlify",
     "cubin 十六进制内嵌进 compile.c/h 模板", "§3"),
    # ② 上方:link.py(从 s3 向上接入)
    ("s4", 0, 2, 0, "kernel_suffix / _match_suffix",
     "函数名当暗号:特化信息的编码↔解码", "§4"),
    ("s5", 0, 3, 0, "HeaderParser",
     "捞回头文件里的 tt-linker 漂流瓶", "§5"),
    ("s6", 0, 4, 0, "make_kernel_hints_dispatcher",
     "运行期整除性分派链(按 num_specs 降序)", "§6"),
    ("s7", 0, 5, 0, "make_func_pointers",
     "algo_id 函数指针表:第二级分派", "§7"),
    # ③ 下方:disasm.py(从 s3 向下接入)
    ("s8", 2, 2, 0, "extract",
     "cuobjdump 两行一指令解析+按 Function 分段", "§8"),
    ("s9", 2, 3, 0, "parseCtrl",
     "64 位控制字五字段解码(stall/yield/barrier)", "§9"),
    ("s10", 2, 4, 0, "processSassLines",
     "BRA 目标地址→LBB 标签两趟重映射", "§10"),
]
NODE_BY_ID = {n[0]: n for n in NODES}

# 主脊/各分支内部的顺序调用边(同一泳道相邻列,水平蓝实线)
EDGES = [
    ("s1", "s2"), ("s2", "s3"),
    ("s4", "s5"), ("s5", "s6"), ("s6", "s7"),
    ("s8", "s9"), ("s9", "s10"),
]
# 主脊→两条分支的跨泳道边(只跨一条相邻泳道边界,真实数据流,蓝实线):
# (源, 目标, 边上标注, "up"=目标在源上方一条泳道 / "down"=目标在源下方一条泳道)
BRANCH_EDGES = [
    ("s3", "s4", "函数名 / full_signature —— tt-linker 元数据", "up"),
    ("s3", "s8", "cubin 字节(kernel.asm['sass'] 的输入)", "down"),
]

# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("脱离 Python 部署:读 §1→§7", [
        (0, "§1"), (1, "§2"), (2, "§3"), (2, "§4"), (3, "§5"), (4, "§6"), (5, "§7"),
    ], True),
    ("只学读 SASS:跳过 §1–§7,直达 §8→§10", [
        (2, "§8"), (3, "§9"), (4, "§10"),
    ], False),
]
LEGEND = [
    ("#22c55e", "入口:开发者在命令行调用 compile.py"),
    ("#3b82f6", "章内主线调用边 / compile.py 产物喂给两条分支的真实数据流"),
    ("#f97316", "出口(两个,互相独立):脱离 Python 部署好 / 读懂 SASS"),
]
TITLE = "第 40 章 · AOT compile/link 与 SASS 反汇编——脱离 Python 部署,读懂它编成了什么"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 185, 92
COL_GAP, ROW_GAP = 26, 20
EDGE_MARGIN, STUB_W, STUB_H = 10, 54, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24
LANE_LABEL_H, BAND_PAD = 22, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 42
BADGE_W, BADGE_H = 40, 20

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

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——本章自带的 §N 单层记号(非虚构 §N.M)。"""
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
# 图例(3 色,>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 26

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 4:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口接口桩:命令行调用,挂在主脊起点 s1(compile.py 第一步)
ex, ey = NODE_XY["s1"]; ey += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("CLI 调用")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩:两个,互相独立——link.py 链尾 s7(已部署) / disasm.py 链尾 s10(已读懂)
sx = w - EDGE_MARGIN - STUB_W
for exit_id, exit_label in (("s7", "已部署"), ("s10", "已读懂")):
    xx, xy = NODE_XY[exit_id]; xy += NODE_H / 2
    L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
             f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
    L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10" font-weight="bold" '
             f'fill="#9a3412">{esc(exit_label)}</text>')
    L.append(f'<line x1="{NODE_XY[exit_id][0] + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 主脊/各分支内部的顺序调用边(同泳道相邻列,水平蓝实线)
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 主脊→两条分支的跨泳道边(只跨一条相邻泳道边界,真实数据流,蓝实线竖向)
for src, dst, label, direction in BRANCH_EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if direction == "up":
        p1 = (xs_ + NODE_W / 2, ys_)             # s3 顶边中点:compile 在中间 lane 向上出
        p2 = (xd + NODE_W / 2, yd + NODE_H)       # s4 底边中点:link 在上方 lane 从下方接入
        label_y = p1[1] - 6
    else:
        p1 = (xs_ + NODE_W / 2, ys_ + NODE_H)     # s3 底边中点
        p2 = (xd + NODE_W / 2, yd)                # s8 顶边中点:disasm 在下方 lane 从上方接入
        label_y = p2[1] - 6
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2.2" marker-end="url(#mMain)"/>')
    L.append(f'<text x="{p1[0] + 8:.1f}" y="{label_y:.1f}" text-anchor="start" '
              f'font-family="sans-serif" font-size="9.5" fill="{C_MAIN}">{esc(label)}</text>')

# 节点(圆角框 + 真实符号名(必要时自动换行/缩字号) + 一行短语 + 右上角 §N 站牌)
SYM_MAXW = NODE_W - 16
PHR_MAXW = NODE_W - 14
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.6"/>')
    sym_lines, sym_size = wrap_symbol(symbol, SYM_MAXW, FONT_SIZES)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.30:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.22:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.22 + sym_size + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    phr_lines, phr_size = wrap_symbol(phrase, PHR_MAXW, (9.5, 9, 8.5, 8))
    py0 = y + NODE_H * 0.56
    for k, pl in enumerate(phr_lines):
        L.append(f'<text x="{cx:.1f}" y="{py0 + k * (phr_size + 3):.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{phr_size}" fill="{C_NODE_SUB}">{esc(pl)}</text>')
    bw = badge_w(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§N 站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌,对应正文同号小节;实线蓝=推荐 / 虚线灰=次要跳读)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    row_top = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H
    name_y = row_top + 13
    ry = row_top + ROUTE_ROW_H - 16
    L.append(f'<text x="16" y="{name_y:.1f}" font-family="sans-serif" font-size="11" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    if x_first != x_last:
        L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
                  f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    # 同一路线里若两站共用一列(如 §3/§4 都挂在 compile→link 分支点那一列),
    # 胶囊会画在完全相同的坐标上、后画的整只盖住先画的——水平错开量按同列
    # 出现次数均分,避免这种"看不见的站牌"[FIX-ROUND-2]
    _col_n = {}
    for col, _ in stops:
        _col_n[col] = _col_n.get(col, 0) + 1
    _col_seen = {}
    for col, sec in stops:
        n = _col_n[col]
        i = _col_seen.get(col, 0)
        _col_seen[col] = i + 1
        off = (i - (n - 1) / 2) * 46 if n > 1 else 0
        L += badge(COLX[col] + NODE_W / 2 + off, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
