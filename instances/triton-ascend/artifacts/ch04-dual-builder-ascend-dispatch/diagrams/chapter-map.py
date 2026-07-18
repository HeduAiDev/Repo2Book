#!/usr/bin/env python3
"""ch04「本章地图」——双 builder 构造 + visit_Call 第四岔 + 双标记 + 转接分机 +
with 分发 的源码剖面图。

本章是**自然标题章**(正文全是 `## 一支笔变两支笔：...` 这类自然标题，无
`## N.M` 编号)——按契约禁用 `§N.M` 徽标，站牌改用正文标题词本身(取每个标题
"："之前的那一半，逐字是正文里出现过的真实子串，供自查逐一核对)：
  “一支笔变两支笔”“同一道门，第四个岔口”“一枚图章盖两个印”“转接分机”
  “一张总机表”——五个站牌对应正文五个自然标题小节(“接力交棒”一节的插入点
  接力细节，作为跨两个节点都有的性质，直接写进 fourth_branch / unified_wire
  两个节点的 claim 里，不单独占一个节点——它本就不是独立机制，是同一动作在
  两处各出现一次)。“小结”一节不建独立节点，改用出口接口桩的说明文字收尾。

剖面(单脊柱，五节递进，每节点标『真实符号 + 规范源码路径 + 一句论点』)：
  ①CodeGenerator.__init__ 并挂 self.builder + self.ascend_builder(同一 context)
  → ②visit_Call 统一 builtin 入口门之后按 extension.is_builtin 选笔(第四岔)
  → ③al.builtin 同时打 __triton_builtin__/__ascend_builtin__ 两个标记(A⊆B)
  → ④setup_unified_builder 把 ascend 方法挂到主 builder、wrapper 同步插入点
  → ⑤visit_With 查 WITH_DISPATCH 表，把 with al.scope(...) 交 handle_scope_with

三条泳道即三条读法：Lane0(①④)=构造与接线；Lane1(②③)=函数调用路径 visit_Call
的选笔逻辑与支撑它的双标记；Lane2(⑤)=语句路径 visit_With 的查表分发。

模板：.claude/skills/svg-diagram/references/example-chapter-map.py；不可变视觉语言
(§/站牌徽标胶囊 / 入口绿-出口橙-主线蓝 / 高亮实线蓝-次要虚线灰 / cjk_text_width)
照搬，只改 DATA + 两处必要的尺寸自适应(自然标题比 §N.M 长得多，badge 宽度按
cjk_text_width 动态算，不再用模板里的定宽 BADGE_W；node 内追加一行等宽字体的
源码路径，宽度用 mono_text_width 估算并按需缩字号)——这两处沿用基座 triton
ch04 chapter-map 已验证过的同一手法，非本图新发明。

六项自查(渲染→Read PNG 亲眼看后如实记录)：见 figure-manifest.json 该图 selfcheck。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def mono_text_width(s, size):
    """monospace 路径行宽度估算：等宽字体每字符约 0.6×size(半角)。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.6) for ch in s)


# ---------------- DATA(可变：本章数据) ----------------
LANES = ["构造与接线", "函数调用路径 · visit_Call", "语句路径 · visit_With"]  # 上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号, 规范源码路径, 一句论点, 站牌=正文自然标题词)
NODES = [
    ("dual_builder", 0, 0, 0,
     "CodeGenerator.__init__",
     "python/triton/compiler/code_generator.py",
     "并挂 self.builder（标准 Triton IR）+ self.ascend_builder（hivm 方言），同一 context",
     "一支笔变两支笔"),
    ("fourth_branch", 1, 1, 0,
     "visit_Call",
     "python/triton/compiler/code_generator.py",
     "统一入口门 language.core.is_builtin 之后，第四岔按 extension.is_builtin 选笔；"
     "调用前后同步插入点/loc",
     "同一道门，第四个岔口"),
    ("dual_marker", 1, 2, 0,
     "al.builtin",
     "third_party/ascend/language/cann/extension/core.py",
     "同时打 __triton_builtin__ + __ascend_builtin__，令 ascend 内建集合 A ⊆ 全体内建 B",
     "一枚图章盖两个印"),
    ("unified_wire", 0, 3, 0,
     "setup_unified_builder",
     "third_party/ascend/language/cann/extension/builder.py",
     "把 ascend_builder 的 emit 方法挂到主 builder；wrapper 内把插入点搬进右笔、"
     "emit 完再搬回来",
     "转接分机"),
    ("with_dispatch", 2, 4, 0,
     "visit_With",
     "python/triton/compiler/code_generator.py",
     "查 WITH_DISPATCH 表，把 with al.scope(...) 交 handle_scope_with 落地",
     "一张总机表"),
]
EDGES = [  # (src_id, dst_id) —— 章内递进主线，统一主线蓝
    ("dual_builder", "fourth_branch"), ("fourth_branch", "dual_marker"),
    ("dual_marker", "unified_wire"), ("unified_wire", "with_dispatch"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
ROUTES = [
    ("从头顺读（全览）",
     [(0, "一支笔变两支笔"), (1, "同一道门，第四个岔口"), (2, "一枚图章盖两个印"),
      (3, "转接分机"), (4, "一张总机表")], True),
    ("只看第四岔怎么选笔", [(1, "同一道门，第四个岔口"), (2, "一枚图章盖两个印")], False),
    ("只看语句级分发", [(4, "一张总机表")], False),
]
LEGEND = [
    ("#22c55e", "入口：上一章 add_kernel 已把 tl.load/tl.store 过了一遍这道门"),
    ("#3b82f6", "章内主线：构造双笔 → 选笔第四岔 → 双标记支点 → 转接分机 → with 查表"),
    ("#f97316", "出口：下一章深入 al.copy / al.fixpipe 等 ascend_builder 算子本身"),
]
TITLE = "第 4 章 · 前端接缝：双 builder 与 Ascend 内建分发路由的源码剖面"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 224
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 10, 46, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 22
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 62, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H = 20
BADGE_FONT = 11
BADGE_PAD_X = 14  # 徽标左右各留的内边距(动态宽度=文本宽+2×BADGE_PAD_X)
CLAIM_FONT = 9.2

_BREAK_AFTER = set("，；：、/ ,;")


def wrap_claim(text, max_w, size):
    """一句论点太长时换行——只在标点/斜杠/空格之后断行，不允许劈开一个标识符
    (如 self.ascend_builder)或一个中文词。贪心找"prefix 仍不超宽的最靠后一个
    合法断点"；找不到合法断点才整句照旧单行放行。"""
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
    # 第二行仍超宽:递归再断一次(最多两次，三行封顶，够用且不至于无限递归)
    more = wrap_claim(line2, max_w, size)
    return [line1] + more


# 每个节点的论点先按 NODE_W 预算换行一遍，取全章最多的行数统一定 NODE_H——
# 同一行号跨泳道对齐用的是同一个 NODE_H，节点内容多寡不能各自决定框高，
# 否则矮框节点与旁边高框节点在同一行会错位、背景条也会被撑破。
CLAIM_MAXW = NODE_W - 16
_CLAIM_LINES = {n[0]: wrap_claim(n[6], CLAIM_MAXW, CLAIM_FONT) for n in NODES}
_max_claim_lines = max(len(v) for v in _CLAIM_LINES.values())
NODE_H = 51 + max(0, _max_claim_lines - 1) * 12 + 10  # 51=符号+路径占用的顶部;10=底部留白

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
    """站牌徽标胶囊，居中挂在 (cx,cy)。宽度按 cjk_text_width 动态算(自然标题
    站牌比 §N.M 长得多，模板里的定宽 BADGE_W 会把长站牌文字挤出胶囊，故此处
    改为文本自适应宽度——胶囊样式/配色/圆角高度仍是模板的不可变视觉语言)。"""
    bw = cjk_text_width(text, BADGE_FONT) + 2 * BADGE_PAD_X
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ], bw


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
# 图例(>2 种语义色必须画图例)；三条说明偏长，纵向各占一行堆叠，避免横排挤出画布
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

# 入口/出口接口桩：入口挂 dual_builder(最左)，出口挂 with_dispatch(最右)
ex, ey = NODE_XY["dual_builder"]; ey += NODE_H / 2
xx, xy = NODE_XY["with_dispatch"]; xy += NODE_H / 2
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

# 调用边(主线蓝)：列不同一律走右→左附着(源框右侧中点 → 目标框左侧中点)。
# 本章五节严格递进、列号严格递增，无需列感知路由的竖直附着分支。
for src, dst in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
    p2 = (xd, yd + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')


# 节点(圆角框 + 真实符号 + 规范源码路径 + 一句论点 + 右上角站牌徽标)
for nid, lane, col, row, symbol, path, claim, station in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + 20:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    path_size = 8.3
    while mono_text_width(path, path_size) > NODE_W - 16 and path_size > 6.3:
        path_size -= 0.3
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + 36:.1f}" text-anchor="middle" '
              f'font-family="monospace" font-size="{path_size:.1f}" '
              f'fill="{C_NODE_PATH}">{esc(path)}</text>')
    base_claim_y = y + 51
    for ci, cline in enumerate(_CLAIM_LINES[nid]):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{base_claim_y + ci * 12:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{CLAIM_FONT}" fill="{C_NODE_SUB}">{esc(cline)}</text>')
    badge_svg, _bw = badge(x + NODE_W - 8, y, station)
    L += badge_svg

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
