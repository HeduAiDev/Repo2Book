#!/usr/bin/env python3
"""ch04「本章地图」——tl.* 表面结构 + constexpr 分水岭 的源码剖面图。

本章是自然标题章(`## §1`…`## §6`,无 `## N.M` 编号)——站牌用正文实际使用的
§1…§6 标记(无小数点,不触发 lint_chapter_map 的 §N.M 徽标核对——与 ch01/ch03
chapter-map 同一处理口径)。

剖面(单脊柱,六节递进,每节点标『节标题 + 规范源码路径 + 一句论点』):
  §1 tl.* 表面怎么铺出来(language/__init__.py:四段 re-export 汇成 tl.*)
  → §2 一个函数两处落地(language/standard.py:tl.cdiv 走 JITFunction)
  → §3 @builtin 的调用契约(language/core.py:builtin/is_builtin 一位布尔分派)
  → §4 一次定义双调用(language/core.py:_tensor_member_fn 返回 fn 本身)
  → §5 constexpr 讲透(language/core.py:全部 dunder 转发,__index__/__bool__ 出壳)
  → §6 两个性能旋钮(compiler/code_generator.py:visit_For 按迭代器类型分道)

三条泳道即三条读法(写作契约要求):
  Lane0(§1–§4)= 表面结构与两套调用契约;Lane1(§5)= constexpr 讲透;
  Lane2(§6)= 两个性能旋钮。底部阅读路线额外给出"从头顺读"和两条直达跳转。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py;不可变视觉语言
(§徽标胶囊 / 入口绿-出口橙-主线蓝 / 高亮实线蓝-次要虚线灰 / cjk_text_width)照搬,
只改 DATA + 节点内部新增"路径行"(单块 monospace 小字,fit 在节点宽度内)。
边路由复用 ch01/ch03 的列感知路由(同列跨泳道→竖直附着,不同列→右到左附着)。

六项自查(渲染→Read PNG 亲眼看后如实记录):见 figure-manifest.json 该图 selfcheck。

用法:python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def mono_text_width(s, size):
    """monospace 路径行宽度估算:等宽字体每字符约 0.6×size(半角);CJK 极少出现
    在源码路径里,但仍留同一套 0.58/1.0 判定以防万一。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.6) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["§1–§4 表面结构与两套调用契约", "§5 constexpr 讲透", "§6 两个性能旋钮"]  # 上→下 = 三条读法

# (节点id, 泳道下标, 列, 泳道内行号, 节标题, 规范源码路径, 一句论点, § 站牌)
NODES = [
    ("surface", 0, 0, 0,
     "tl.* 表面怎么铺出来",
     "python/triton/language/__init__.py",
     "18+81+17+10 符号汇入 tl.*，__all__ 131 项门面",
     "§1"),
    ("twotier", 0, 1, 0,
     "一个函数两处落地",
     "python/triton/language/standard.py",
     "tl.cdiv 走 JITFunction；host 侧 triton.cdiv 当场返回 int",
     "§2"),
    ("builtin", 0, 2, 0,
     "@builtin 的调用契约",
     "python/triton/language/core.py",
     "无 _builder 调用点即 raise，is_builtin 一位布尔分派",
     "§3"),
    ("memberfn", 0, 3, 0,
     "一次定义，双调用形式",
     "python/triton/language/core.py",
     "_tensor_member_fn 返回 fn 本身，自由函数/方法共享实现",
     "§4"),
    ("constexpr", 1, 4, 0,
     "constexpr 讲透",
     "python/triton/language/core.py",
     "全部 dunder 转发内层值，__index__/__bool__ 是仅有出壳口",
     "§5"),
    ("ranges", 2, 5, 0,
     "两个性能旋钮",
     "python/triton/compiler/code_generator.py",
     "static_range 0 scf.for/8 addi 全展开；range 1 scf.for/2 addi+两提示",
     "§6"),
]
EDGES = [  # (src_id, dst_id) —— 章内递进主线,统一主线蓝
    ("surface", "twotier"), ("twotier", "builtin"), ("builtin", "memberfn"),
    ("memberfn", "constexpr"), ("constexpr", "ranges"),
]
# (路线名, [(列, § 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("从头顺读（全览）",
     [(0, "§1"), (1, "§2"), (2, "§3"), (3, "§4"), (4, "§5"), (5, "§6")], True),
    ("只看 constexpr 讲透", [(4, "§5")], False),
    ("只看两个性能旋钮", [(5, "§6")], False),
]
LEGEND = [
    ("#22c55e", "入口：读者从 §1 开始（tl.* 这张表面）"),
    ("#3b82f6", "章内主线：表面→两套契约→constexpr→旋钮"),
    ("#f97316", "出口：下一章深挖 tl.load / tl.dot 怎么追踪成 IR"),
]
TITLE = "第 4 章 · tl.* 表面结构与 constexpr 分水岭的源码剖面（§1–§6 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 214, 84
COL_GAP, ROW_GAP = 14, 20
EDGE_MARGIN, STUB_W, STUB_H = 8, 40, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 20
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 62, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
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
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy)。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例);三条说明偏长,纵向各占一行堆叠,避免横排挤出画布
for li, (color, label) in enumerate(LEGEND):
    _row_y = TOP_PAD + TITLE_H + 14 + li * 14
    L.append(f'<rect x="{PAD_L}" y="{_row_y - 11}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 18}" y="{_row_y}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')

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

# 入口/出口接口桩:入口挂 surface(最左),出口挂 ranges(最右)
ex, ey = NODE_XY["surface"]; ey += NODE_H / 2
xx, xy = NODE_XY["ranges"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("读者")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝):列感知路由——不同列走右→左附着(对角仍落源框右侧);
# 同列跨泳道走竖直 底心↔顶心 附着。本章六节递进,列号严格递增,故全部走右→左。
for src, dst in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    col_s, col_d = NODE_BY_ID[src][2], NODE_BY_ID[dst][2]
    lane_s, lane_d = NODE_BY_ID[src][1], NODE_BY_ID[dst][1]
    if col_s != col_d:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
    elif lane_d > lane_s:
        p1 = (xs_ + NODE_W / 2, ys_ + NODE_H)
        p2 = (xd + NODE_W / 2, yd)
    else:
        p1 = (xs_ + NODE_W / 2, ys_)
        p2 = (xd + NODE_W / 2, yd + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')


_BREAK_AFTER = set("，；：、/ ,;")


def wrap_claim(text, max_w, size):
    """一句论点太长时换行——只在标点/斜杠/空格之后断行,不允许劈开一个标识符
    (如 __index__、triton.cdiv)或一个中文词(如"函数")。贪心找"prefix 仍不超宽
    的最靠后一个合法断点";找不到合法断点(单段本身就超宽)才整句照旧单行放行,
    宁可稍微超宽,也不产出断在标识符中间的乱码断行。"""
    breaks = [i for i, ch in enumerate(text) if ch in _BREAK_AFTER]
    best = None
    for i in breaks:
        if cjk_text_width(text[:i + 1], size) <= max_w:
            best = i
    if best is None:
        return [text]
    line1, line2 = text[:best + 1].rstrip(), text[best + 1:].lstrip()
    if cjk_text_width(line2, size) <= max_w:
        return [line1, line2]
    return [line1, line2]  # 第二行万一仍略超宽:只轻微溢出,好过第二次强行断词


# 节点(圆角框 + 节标题 + 规范源码路径 + 一句论点 + 右上角 § 徽标)
CLAIM_MAXW = NODE_W - 16
for nid, lane, col, row, title, path, claim, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + 20:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(title)}</text>')
    path_size = 8.3
    while mono_text_width(path, path_size) > NODE_W - 16 and path_size > 6.5:
        path_size -= 0.3
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + 36:.1f}" text-anchor="middle" '
              f'font-family="monospace" font-size="{path_size:.1f}" '
              f'fill="{C_NODE_PATH}">{esc(path)}</text>')
    claim_lines = wrap_claim(claim, CLAIM_MAXW, 9.3)
    base_claim_y = y + 51 if len(claim_lines) == 1 else y + 48
    for ci, cline in enumerate(claim_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{base_claim_y + ci * 12:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.3" fill="{C_NODE_SUB}">{esc(cline)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 6, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
