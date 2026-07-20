#!/usr/bin/env python3
"""第 32 章「本章地图」——源码剖面图。

四条泳道(上→下):调度层(scheduler.py/core.py) / 装配层(structured_output/__init__.py)
/ worker 落地层 / 执行层(库函数·Triton kernel)。§32.7 处一分为二:上排是默认部署
实际走的路(VLLM_USE_V2_MODEL_RUNNER 默认 False→GPUModelRunnerV1→xgrammar 库函数)，
下排是 opt-in 的 V2 路径(vLLM 自写 Triton kernel)。底部两条阅读路线复刻同一个分岔，
高亮实线蓝=默认部署(主线)，虚线灰=V2 opt-in(演进方向)——呼应本章开篇那句"必须先讲清"
的路线前提，避免读者把「kernel 位运算」误当默认行为。

坐标全部由循环/常量计算,不手写魔数;换章节时只改下面的 DATA。
用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["调度层", "装配层", "worker 落地层", "执行层"]

# (节点id, 泳道下标, 列, 泳道内行号, 符号行(元组=多行), 一行短语, §编号)
NODES = [
    ("gate", 0, 0, 0,
     ("get_grammar_bitmask",), "筛请求,收集 req_id 列表", "§32.1"),
    ("order", 0, 1, 0,
     ("GrammarOutput",), "行序不变式:随掩码同传", "§32.2"),
    ("spec_pref", 0, 2, 0,
     ("validate_tokens",), "过滤草稿+-1 补齐", "§32.3"),
    ("assemble", 1, 3, 0,
     ("grammar_bitmask",), "并行/串行二选一填掩码", "§32.4–32.5"),
    ("reason", 1, 4, 0,
     ("should_fill_bitmask", "should_advance"), "推理段两道独立的门", "§32.6"),
    ("fork", 2, 5, 0,
     ("VLLM_USE_V2", "MODEL_RUNNER"), "默认 False→选 V1 runner", "§32.7"),
    ("legacy_apply", 2, 6, 0,
     ("apply_grammar_bitmask",), "utils.py:重排 sorted_bitmask", "§32.8"),
    ("v2_apply", 2, 6, 1,
     ("apply_grammar_bitmask",), "copy_stream 搬运+映射", "§32.9–32.10"),
    ("legacy_kernel", 3, 7, 0,
     ("xgr.apply_token", "bitmask_inplace"), "xgrammar 库函数(默认执行者)", "§32.8"),
    ("v2_kernel", 3, 7, 1,
     ("_apply_grammar", "bitmask_kernel"), "Triton kernel:解包写 -inf", "§32.11"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("gate", "order"), ("order", "spec_pref"), ("spec_pref", "assemble"),
    ("assemble", "reason"), ("reason", "fork"),
    ("fork", "legacy_apply"), ("fork", "v2_apply"),
    ("legacy_apply", "legacy_kernel"), ("v2_apply", "v2_kernel"),
]
# 出口:两个终点节点都流向同一个下游(采样器),用一个跨两行的高出口桩承接
EXIT_SRC_IDS = ["legacy_kernel", "v2_kernel"]

# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("默认部署(主线)",
     [(0, "§32.1"), (1, "§32.2"), (2, "§32.3"), (3, "§32.4–32.5"), (4, "§32.6"),
      (5, "§32.7"), (6, "§32.8"), (7, "§32.8")], True),
    ("V2 opt-in(演进方向)",
     [(0, "§32.1"), (1, "§32.2"), (2, "§32.3"), (3, "§32.4–32.5"), (4, "§32.6"),
      (5, "§32.7"), (6, "§32.9–32.10"), (7, "§32.11")], False),
]
LEGEND = [("#22c55e", "入口:引擎/调度器调用进入"), ("#3b82f6", "章内主线调用边"),
          ("#f97316", "出口:交给采样器(ch30)")]
TITLE = "第 32 章 · 掩码装配到落地剖面（默认 xgrammar 路径 vs V2 opt-in Triton kernel）"
# 补充注记:§32.12 的结论挂在 kernel 终点旁,不占独立节点位
ANNOTATION = ("-inf 与温度/top-k/top-p 正交", "§32.12")

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_ANNOTATION = "#7c3aed"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 140, 74
COL_GAP, ROW_GAP = 20, 16
EDGE_MARGIN, STUB_W, STUB_H = 14, 64, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 30
LANE_LABEL_H, BAND_PAD = 22, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 24, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 20, 40
BADGE_W, BADGE_H = 50, 18

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
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.1"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.5:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 17}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 13
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 10}" width="13" height="13" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 18}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 18 + cjk_text_width(label, 11) + 26

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="14" y="{band_top[i] + LANE_LABEL_H - 5:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口接口桩(单行,喂给 gate)
ex, ey = NODE_XY["gate"]; ey += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.2"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("调度器")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩(高桩,跨 legacy_kernel/v2_kernel 两行,两条箭头分别汇入)
_exit_ys = []
for nid in EXIT_SRC_IDS:
    nx, ny = NODE_XY[nid]
    _exit_ys.append(ny + NODE_H / 2)
stub_top = min(_exit_ys) - STUB_H / 2
stub_bot = max(_exit_ys) + STUB_H / 2
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{stub_top:.1f}" width="{STUB_W}" height="{stub_bot - stub_top:.1f}" '
         f'rx="13" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.2"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{(stub_top + stub_bot) / 2 + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("采样器")}</text>')
for nid in EXIT_SRC_IDS:
    nx, ny = NODE_XY[nid]
    y_mid = ny + NODE_H / 2
    L.append(f'<line x1="{nx + NODE_W:.1f}" y1="{y_mid:.1f}" x2="{sx:.1f}" y2="{y_mid:.1f}" '
              f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝),多条边汇入/分出同一节点时按行数错开终点 y。
# ELBOW_EDGES:显式折线路由的边——(src,dst) 若直连对角线会擦过中间某节点的 §
# 徽标(如 legacy_apply→legacy_kernel 的直线在 y≈434 处会蹭到 v2_apply 的
# §32.9–32.10 徽标右边缘),改走"先在源节点行高上平移到目标列、再垂直下降"的
# 直角路径,途中 x 坐标提前跳过所有中间节点/徽标的横向范围,彻底避免擦碰。
ELBOW_EDGES = {("legacy_apply", "legacy_kernel")}
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
    if (src, dst) in ELBOW_EDGES:
        p_elbow = (p2[0], p1[1])  # 先平移到目标列 x,仍在源节点的行高 y 上
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p_elbow[0]:.1f}" y2="{p_elbow[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2"/>')
        L.append(f'<line x1="{p_elbow[0]:.1f}" y1="{p_elbow[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    else:
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 1~2 行符号 + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol_lines, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="10" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.4"/>')
    n_sym = len(symbol_lines)
    if n_sym == 1:
        sym_ys = [y + NODE_H * 0.36]
    else:
        sym_ys = [y + NODE_H * 0.26, y + NODE_H * 0.44]
    for sline, sy in zip(symbol_lines, sym_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{sy:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sline)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.78:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 6, y, sec)

# §32.12 注记:挂在 v2_kernel 旁,不占独立节点位(呼应 -inf 正交性的收尾论点)
_ax, _ay = NODE_XY["v2_kernel"]
_anno_x = _ax + NODE_W / 2
_anno_y = _ay + NODE_H + 14
L.append(f'<line x1="{_anno_x:.1f}" y1="{_ay + NODE_H:.1f}" x2="{_anno_x:.1f}" y2="{_anno_y - 6:.1f}" '
          f'stroke="{C_ANNOTATION}" stroke-width="1.2" stroke-dasharray="3,3"/>')
L.append(f'<text x="{_anno_x:.1f}" y="{_anno_y + 8:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9" font-style="italic" fill="{C_ANNOTATION}">{esc(ANNOTATION[0])}</text>')
L += badge(_anno_x, _anno_y + 22, ANNOTATION[1])

# 底部阅读路线
L.append(f'<text x="14" y="{routes_top + 14:.1f}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=默认部署 / 虚线灰=V2 opt-in)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="14" y="{ry + 4:.1f}" font-family="sans-serif" font-size="10.5" '
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, aspect {w / h:.2f}:1)")
