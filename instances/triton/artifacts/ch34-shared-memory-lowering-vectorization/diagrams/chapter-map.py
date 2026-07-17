#!/usr/bin/env python3
"""ch34 本章地图:共享内存降级 + 全局访存向量化剖面 —— 分配落地(offset 钉成属性,
全 kernel 共用一个 global_smem) → swizzle 相位公式编译成物理偏移(第 22 章兑现) →
storeDistributedToShared 写进 / ldmatrix 读回共享内存 → getVectorSize 三把夹子定
向量宽(min(128/位宽, contiguity, maskAlign)) → redundantDataMask 只留唯一写者 →
拼出 ld/st.global 内联 PTX → 屏障落地。

改自 .claude/skills/svg-diagram/references/example-chapter-map.py 模板(与 ch33/ch30
chapter-map.py 同构:站牌胶囊 / 入口绿#22c55e-出口橙#f97316-主线蓝#3b82f6 /
高亮实线蓝-次要虚线灰 / cjk_text_width() 宽度估算 —— 均不可变,只改下面的 DATA)。

本章顶层标题写作 `## §1 ...`(§ 前缀 + 单数字,无小数点),lint_chapter_map 的
`§N.M` 徽标正则(要求带小数点)不匹配它们 → 按自然标题处理 → **站牌一律用标题词逐字
子串,禁用 §N.M 小数点徽标**(与 ch33 natural-title 章同策略)。
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


# ---------------- DATA(可变:本章数据) ----------------
LANES = [
    "前半 · 共享内存：分配落地 → swizzle 相位物化 → 写进／读回",
    "后半 · 全局访存：三把夹子定向量宽 → 唯一写者 → 拼 PTX",
    "收尾 · 屏障落地：读写之间按需插一条 barrier",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本[标题词逐字子串,禁用 §N.M])
NODES = [
    ("t_alloc", 0, 0, 0, "AllocateSharedMemory",
     "offset 钉成 i32 属性，gep 进唯一 global_smem", "分配落地"),
    ("t_phase", 0, 1, 0, "LinearLayoutConversions",
     "swizzle 相位 φ(r) 编译成物理偏移（ch22 兑现）", "相位公式落进物理偏移"),
    ("t_store", 0, 2, 0, "storeDistributedToShared",
     "拆 laneId／warpId，xor 换相写进共享内存", "写进共享内存"),
    ("t_load", 0, 3, 0, "ldmatrix",
     "读回喂 dot：通用引擎 或 ldmatrix 加载器", "读回喂 dot"),

    ("t_vec", 1, 0, 0, "getVectorSize",
     "vec 取 128／位宽、contiguity、maskAlign 三者最小", "三把夹子定向量宽"),
    ("t_mask", 1, 1, 0, "redundantDataMask",
     "shape ＜ CTA tile 时只留唯一写者，流量砍半", "防重复写"),
    ("t_ptx", 1, 2, 0, "ld/st.global",
     "把 vec 拼成一条向量化访存内联 PTX", "拼出 ld/st.global 内联 PTX"),

    ("t_barrier", 2, 0, 0, "createBarrier",
     "写与读共享内存之间按需插一条 barrier", "屏障落地"),
]
EDGES = [  # 行内直线(同泳道:相邻列)
    ("t_alloc", "t_phase"), ("t_phase", "t_store"), ("t_store", "t_load"),
    ("t_vec", "t_mask"), ("t_mask", "t_ptx"),
]
# 跨泳道折行边:(src_id, dst_id, dst 落点 x 偏移)
WRAP_EDGES = [
    ("t_load", "t_vec", 0),
    ("t_ptx", "t_barrier", 0),
]
# (路线名, [(列, 站牌文本), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("① 完整链（通读）",
     [(0, "分配落地"), (2, "写进共享内存"), (3, "读回喂 dot")], True),
    ("② perf 命门（跳读）",
     [(0, "三把夹子定向量宽"), (1, "防重复写")], False),
]
LEGEND = [("#22c55e", "入口：TTGIR（张量已贴布局，路已选好）"),
          ("#3b82f6", "主线：分配 → 相位 → 读写 → 向量宽 → 去重"),
          ("#f97316", "出口：LLVM／PTX（下一章把 dot 降到 Tensor Core）")]
TITLE = "第 34 章 · 共享内存降级 + 全局访存向量化剖面：分配落地 → 相位物化 → 读写引擎 → 三把夹子定向量宽 → 唯一写者（源码走线 + 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 214, 60
COL_GAP, ROW_GAP = 44, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 96, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 30
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_PAD_X, BADGE_FONT = 20, 10, 11
WRAP_GAP = 22  # 折行边:绕出节点右侧的横向余量

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
w_grid = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
# 画布宽度须同时容得下图例一整行——取网格宽度与图例总宽度的较大者,否则长图例被裁掉。
w_legend = PAD_L + sum(20 + cjk_text_width(lbl, 11.5) + 34 for _, lbl in LEGEND) + PAD_R
w = max(w_grid, w_legend)
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——宽度按文本动态算(自然语言站牌,非定长短码)。"""
    bw = cjk_text_width(text, BADGE_FONT) + BADGE_PAD_X * 2
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
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
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 34

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

# 入口/出口接口桩
ex, ey = NODE_XY["t_alloc"]; ey += NODE_H / 2
xx, xy = NODE_XY["t_barrier"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("TTGIR 入")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("→ LLVM／PTX")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 行内直线调用边(主线蓝)——多条边汇入同一节点时终点 y 各偏移
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
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 跨泳道折行边(elbow):右侧绕出 → 下降到"下一泳道标签+留白"空白带 → 沿空白带向左 →
# 短距下降落入下一泳道节点顶部(含 dst_dx 偏移)。
for wsrc, wdst, dst_dx in WRAP_EDGES:
    wx1, wy1 = NODE_XY[wsrc]; wx2, wy2 = NODE_XY[wdst]
    p_start = (wx1 + NODE_W, wy1 + NODE_H / 2)
    turn_x = wx1 + NODE_W + WRAP_GAP
    drop_y = wy2 - 8
    p_mid1 = (turn_x, wy1 + NODE_H / 2)
    p_mid2 = (turn_x, drop_y)
    p_mid3 = (wx2 + NODE_W / 2 + dst_dx, drop_y)
    p_end = (wx2 + NODE_W / 2 + dst_dx, wy2)
    L.append(f'<polyline points="{p_start[0]:.1f},{p_start[1]:.1f} {p_mid1[0]:.1f},{p_mid1[1]:.1f} '
             f'{p_mid2[0]:.1f},{p_mid2[1]:.1f} {p_mid3[0]:.1f},{p_mid3[1]:.1f} {p_end[0]:.1f},{p_end[1]:.1f}" '
             f'fill="none" stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌),字号按文本长度自适应收缩避免溢出
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_size = fit_size(symbol, NODE_W - 18, 13, 9)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sym_size:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    ph_size = fit_size(phrase, NODE_W - 16, 10.5, 8)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{ph_size:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = cjk_text_width(sec, BADGE_FONT) + BADGE_PAD_X * 2
    L += badge(x + NODE_W - bw / 2 + 10, y, sec)

# 底部阅读路线
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线（标号=图上讲解站牌；实线蓝=推荐 / 虚线灰=次要）")}</text>')
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio={w/h:.2f})")
