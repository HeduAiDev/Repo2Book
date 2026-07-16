#!/usr/bin/env python3
"""ch13《TRITON_INTERPRET》本章地图——源码剖面图。

自然标题章(chapter.md 无 `## N.M` 编号，只有自然标题)：禁用 §N.M 徽标，
站牌改用标题词本身的真实子串(见下方各节 badge 文本，均可在正文对应
`## ...` 标题里逐字找到)。

结构：本章是一条单向流水线(不像有的章那样在中段分叉两条业务路径)——
入口分叉(decorator/InterpretedFunction) → 为何重写 AST(visit_Assign/
to_tensor) → 改写流水线(rewrite_ast) → 串行遍历 grid(__call__/
_implicit_cvt) → 在 InterpreterBuilder 侧展开三个同构例子(program_id/
binary_op/masked_load，一次 fan-out) → 汇合到边界结论(RESERVED_KWS)。
底部两条阅读路线对应正文原有的导读句："全程通读"与"只要结论直接跳边界"。

■ 不可变(全书统一视觉语言，换章节数据时不要动这些，只改下面的 DATA)：
  1. 站牌胶囊：圆角矩形(pill)，fill #eef2ff / stroke #6366f1 / 文字 indigo 深色，
     贴在节点框右上角、跨在上边框上；底部路线条复用同一画法。
  2. 入口/出口：画布左右边缘各放一个"调用方 / 返回上层"接口桩，
     入口箭头 = 绿 #22c55e，出口箭头 = 橙 #f97316。
  3. 节点间调用边(主线) = 蓝 #3b82f6。
  4. 底部路线条：高亮/推荐路线 = 实线蓝(粗)；其余路线 = 虚线中性灰(细)。
  5. >2 种语义色必须画图例。
  6. 文本宽度估算一律用 cjk_text_width()。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录)：
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变：本章数据) ----------------
LANES = ["入口 / 改写层（jit.py, interpreter.py 改写）", "CPU 串行执行层（GridExecutor）", "InterpreterBuilder 出数层"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本)
NODES = [
    ("entry_fork", 0, 0, 0, "decorator", "TRITON_INTERPRET=1 时派替身", "入口分叉"),
    ("interp_fn", 0, 1, 0, "InterpretedFunction", "持有原函数,惰性触发 rewrite()", "入口分叉"),
    ("ast_rewrite", 0, 2, 0, "visit_Assign", "把赋值右值包成 to_tensor", "重写 AST"),
    ("to_tensor_n", 0, 3, 0, "to_tensor", "裸标量按大小提升为张量", "重写 AST"),
    ("rewriter_pipe", 0, 4, 0, "rewrite_ast", "源码→AST 变换→行号对齐", "小流水线"),
    ("grid_exec", 1, 5, 0, "__call__", "三重 for 串行遍历 grid", "串行遍历"),
    ("host_copy", 1, 5, 1, "_implicit_cvt", "指针换 uint64 地址", "串行遍历"),
    ("patch_lang", 1, 5, 2, "_patch_lang", "重绑 tl 到替身 builder", "串行遍历"),
    ("create_pid", 2, 6, 0, "create_get_program_id", "直接读 grid_idx,不建 IR", "同名接口"),
    ("binary_op", 2, 6, 1, "binary_op", "np.add / np.multiply", "同名接口"),
    ("masked_load", 2, 6, 2, "create_masked_load", "按 uint64 地址读写内存", "同名接口"),
    ("boundary", 2, 7, 1, "RESERVED_KWS", "并行旋钮被剔除,查对错不查快慢", "串行 ≠ 并行"),
]
EDGES = [  # (src_id, dst_id) —— 调用边，统一主线蓝
    ("entry_fork", "interp_fn"),
    ("interp_fn", "ast_rewrite"),
    ("ast_rewrite", "to_tensor_n"),
    ("to_tensor_n", "rewriter_pipe"),
    ("rewriter_pipe", "grid_exec"),
    ("grid_exec", "host_copy"),
    ("host_copy", "patch_lang"),
    ("patch_lang", "create_pid"),
    ("patch_lang", "binary_op"),
    ("patch_lang", "masked_load"),
    ("create_pid", "boundary"),
    ("binary_op", "boundary"),
    ("masked_load", "boundary"),
]
# (路线名, [(列, 站牌文本), ...] 按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程通读", [
        (0, "入口分叉"), (1, "入口分叉"), (2, "重写 AST"), (3, "重写 AST"),
        (4, "小流水线"), (5, "串行遍历"), (6, "同名接口"), (7, "串行 ≠ 并行"),
    ], True),
    ("只要结论", [(0, "入口分叉"), (7, "串行 ≠ 并行")], False),
]
LEGEND = [("#22c55e", "调用方发起:@triton.jit 装饰的核"), ("#3b82f6", "章内替身路径主线"), ("#f97316", "返回:数值写回原设备张量")]
TITLE = "TRITON_INTERPRET 替身执行剖面（本章地图）"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替，仅装饰，非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W, NODE_H = 155, 58
COL_GAP, ROW_GAP = 16, 16
EDGE_MARGIN, STUB_W, STUB_H = 10, 48, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 8  # 左右各留：接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 30, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 38
BADGE_H, BADGE_PAD_X = 20, 10

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
    return cjk_text_width(text, 11) + BADGE_PAD_X * 2


def badge(cx, cy, text):
    """站牌胶囊，居中挂在 (cx,cy)——节点用它贴右上角，路线图例用它居中挂线上。
    宽度按文本动态算(cjk_text_width)，右边缘固定 = 挂点右侧一个常量偏移，
    不会因文本变长而侵入下一列节点(见节点绘制处的 corner 用法)。"""
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
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
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
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["entry_fork"]; ey += NODE_H / 2
xx, xy = NODE_XY["boundary"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)——多条边汇入/发出同一节点时，端点 y 各错开，避免视觉粘连成一条断头线
_dst_total, _src_total = {}, {}
for s, d in EDGES:
    _dst_total[d] = _dst_total.get(d, 0) + 1
    _src_total[s] = _src_total.get(s, 0) + 1
_dst_seen, _src_seen = {}, {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    ns = _src_total[src]; i_s = _src_seen.get(src, 0); _src_seen[src] = i_s + 1
    y_off_s = (i_s - (ns - 1) / 2) * 14 if ns > 1 else 0
    nd = _dst_total[dst]; i_d = _dst_seen.get(dst, 0); _dst_seen[dst] = i_d + 1
    y_off_d = (i_d - (nd - 1) / 2) * 14 if nd > 1 else 0
    p1 = (x1 + NODE_W, y1 + NODE_H / 2 + y_off_s)
    p2 = (x2, y2 + NODE_H / 2 + y_off_d)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
              f'font-family="monospace" font-size="11" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.68:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9.3" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W + 8 - badge_w(sec) / 2, y, sec)

# 底部阅读路线：复用列坐标 COLX，站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标牌=图上节点站牌;实线蓝=推荐通读 / 虚线灰=只取结论)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio={w/h:.2f})")
