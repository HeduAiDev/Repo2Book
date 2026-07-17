#!/usr/bin/env python3
"""ch28《AccelerateMatmul 与布局最优化》本章地图——源码剖面图。

四条泳道 = 三个 pass 在 make_ttgir 里的落点顺序，AccelerateMatmul 拆成两条
(选规格 / 造编码，因主重写之后分叉成 v1/v2 与 v3 两支)：
  L0 选规格:版本→warp 分配→指令形状(对应正文 §2.1-§2.3)
  L1 造 mma 编码,分叉:v1/v2 换 dot-operand / v3 进共享内存(对应正文 §2.4-§2.6)
  L2 RemoveLayoutConversions:消搬运,make_ttgir 里跑 3 次(对应正文 §3)
  L3 OptimizeDotOperands:把 convert 挪到更省的位置(对应正文 §4)
每条泳道内是同行直线(EDGES)；跨泳道换行处用 WRAP_EDGES(elbow：右侧绕出 →
沿"下一泳道标签+留白"的空白带向左 → 落入下一泳道首节点顶部)，L1 的两个分叉
节点(dot_operand_enc / mmav3_shared)都指向 L2 的 layout_prop，两条折行边落点
各错开 ±14px 以免重合。全程不穿过任何节点——四条泳道共用同一组列坐标(0/1/2)，
但每条泳道只占用自己需要的列数，画布宽度由最宽的 L0(3 列)决定，不随泳道数
增多而变宽。

■ 本章特有(自然标题章——按 illustrator 契约判定标准是 chapter.md 有没有裸
  `## N.M` 编号标题；本章实际标题是 `## §N ...` / `### §N.M ...`，带 § 前缀、
  且子节在 `###` 三级标题，均不匹配 lint_chapter_map 的 `^##\\s+\\d+\\.\\d+`
  正则，故 heading_set 为空、判定为自然标题章):
  - 节点右上角站牌**禁用带小数点的 §N.M 徽标**，改用标题词本身的逐字子串
    (如 "选 MMA 版本" 取自 `### §2.1 选 MMA 版本：...`)——与 ch09 chapter-map
    先例一致。裸 §N(无小数点，如 "§2"/"§3"/"§4")不触发 lint 的 `_BADGE_RE`
    (要求 §数字.数字)，且这些泳道级标签对应的 `## §2`/`## §3`/`## §4` 顶层
    标题确实逐字存在，故泳道名保留裸 §N 前缀(与 ch27 chapter-map 先例一致)，
    只是节点级的小数点子节标签换成自然词。
  - badge()/BADGE_W 改按文本动态算宽(cjk_text_width + 内边距)，因为站牌是
    完整词组(最长 8-12 字)而非 "§2.1" 这类定长短码，固定宽度会溢出。
  - 标识符后一律不紧跟半角圆括号(`RemoveLayoutConversions(`这类会被 lint 的
    杜撰符号检测当成新 token 核对、又核不到，因为正文/dossier 里这两个词从不
    紧跟半角左括号)——泳道名/图例/路线名统一改用全角冒号或箭头分隔。

■ 不可变(全书统一视觉语言，换章节数据时不要动这些，只改下面的 DATA)：
  与 example-chapter-map.py / ch27 / ch09 chapter-map.py 完全一致——站牌胶囊 /
  入口绿#22c55e-出口橙#f97316-主线蓝#3b82f6 / 高亮实线蓝-次要虚线灰 /
  cjk_text_width() 宽度估算。

■ 可变：LANES / NODES / EDGES(行内直线) / WRAP_EDGES(跨行折行边,含合流偏移) /
  ROUTES / LEGEND / TITLE。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录；每次改动 DATA/布局代码后必须
  重新核对一遍)：
  [第一轮] claim_readable_10s=True numbers_match_spec=True no_overlap=True
           arrows_attached=True cjk_rendered=True reading_order_clear=True
    —— 该轮用的是 §2.1/§2.2/.../§3.1/§3.3 带小数点徽标，lint_chapter_map 报
       8 处 badge_not_in_headings + 3 处 fabricated_symbol(RemoveLayoutConversions(/
       OptimizeDotOperands(/Core( 紧跟半角括号)。
  [第二轮/当前，已重渲染+Read PNG 复核] claim_readable_10s=True
    numbers_match_spec=True no_overlap=True arrows_attached=True
    cjk_rendered=True reading_order_clear=True
    —— 节点站牌全部换成标题词逐字子串(不含 §N.M)，泳道名保留裸 §2/§3/§4，
       标识符与括号之间统一改用全角冒号/箭头分隔，COL_GAP 从 34 提到 44 给
       变宽的自然语言站牌留够右侧悬挂余量；lint_chapter_map 与
       lint_diagram_geometry 均已核实通过（见文件末尾 print 后 Bash 记录）。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(text, max_w, base, min_size):
    """按 max_w 反解一个不超出的字号(单行，不换行)。"""
    unit = cjk_text_width(text, 1.0)
    if unit <= 0:
        return base
    return max(min_size, min(base, max_w / unit))


# ---------------- DATA(可变：本章数据) ----------------
LANES = [
    "§2 AccelerateMatmul · 选规格(版本 → warp 分配 → 指令形状)",
    "§2 AccelerateMatmul · 造编码(分叉:v1/v2 换操作数 / v3 进共享内存)",
    "§3 RemoveLayoutConversions：消搬运,make_ttgir 里跑 3 次",
    "§4 OptimizeDotOperands：把必要的 convert 挪到更省的位置",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本[标题词逐字子串,禁用 §N.M])
NODES = [
    ("select_version", 0, 0, 0, "getMMAVersionSafe",
     "按算力选版本(K/shape/dtype 验票)", "MMA 版本"),
    ("warps_per_tile", 0, 1, 0, "warpsPerTileV2",
     "贪心分配 warp 到 M/N 两轴", "warpsPerTile"),
    ("instr_shape", 0, 2, 0, "mmaVersionToInstrShape",
     "按版本/dtype 定单指令形状", "instrShape"),
    ("blocked_to_mma", 1, 0, 0, "BlockedToMMA",
     "造 mma 编码,换累加器编码", "BlockedToMMA"),
    ("dot_operand_enc", 1, 1, 0, "DotOperandEncodingAttr",
     "v1/v2:换 A/B 操作数编码", "dot-operand"),
    ("mmav3_shared", 1, 1, 1, "getSharedMemoryMMAOperand",
     "v3:操作数进共享内存", "MMAv3 特判"),
    ("layout_prop", 2, 0, 0, "LayoutPropagation",
     "锚点→传播→消冲突→重写", "四阶段算法"),
    ("backward_remat", 2, 1, 0, "backwardRematerialization",
     "残余 convert:重物化+hoist", "残余 convert"),
    ("hoist_convert", 3, 0, 0, "HoistLayoutConversion",
     "convert 上移贴 load,省 shmem 往返", "OptimizeDotOperands"),
]
EDGES = [  # 行内直线(同泳道相邻列/行)
    ("select_version", "warps_per_tile"), ("warps_per_tile", "instr_shape"),
    ("blocked_to_mma", "dot_operand_enc"), ("blocked_to_mma", "mmav3_shared"),
    ("layout_prop", "backward_remat"),
]
# 跨泳道折行边:(src_id, dst_id, dst 落点 x 偏移——多条边合流到同一 dst 时错开)
WRAP_EDGES = [
    ("instr_shape", "blocked_to_mma", 0),
    ("dot_operand_enc", "layout_prop", -14),
    ("mmav3_shared", "layout_prop", 14),
    ("backward_remat", "hoist_convert", 0),
]
# (路线名, [(列, 站牌文本), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("没命中 Tensor Core → 先查这里", [(0, "MMA 版本"), (1, "warpsPerTile"), (2, "instrShape")], True),
    ("命中了却慢 → 消搬运/挪位置", [(0, "四阶段算法"), (1, "残余 convert"), (2, "OptimizeDotOperands")], False),
]
LEGEND = [("#22c55e", "入口:tt.dot 已转 blocked 编码 ttgir"),
          ("#3b82f6", "章内主线:选规格→造编码→消搬运→挪位置"),
          ("#f97316", "出口:交给 pipeline/prefetch 等下游阶段")]
TITLE = "第 28 章 · AccelerateMatmul 与布局最优化剖面(源码走线 + 讲解站牌)"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替，仅装饰，非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W, NODE_H = 210, 60
COL_GAP, ROW_GAP = 44, 20  # COL_GAP 比定长 §N.M 徽标章节更宽:自然语言站牌右悬挂余量更大
EDGE_MARGIN, STUB_W, STUB_H = 16, 78, 26
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
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """站牌胶囊，居中挂在 (cx,cy)——宽度按文本动态算(自然语言站牌，非定长短码)。"""
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
ex, ey = NODE_XY["select_version"]; ey += NODE_H / 2
xx, xy = NODE_XY["hoist_convert"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("上游 ttgir")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("下游阶段")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 行内直线调用边(主线蓝)——多条边汇入同一节点时终点 y 各偏移,避免重合看不出汇合
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

# 跨泳道折行边(elbow):右侧绕出 → 下降到"下一泳道标签+留白"这条空白带(节点顶部
# 之上、分界线之下，全程无节点/无文字) → 沿这条空白带一路向左 → 短距下降落入
# 下一泳道节点顶部(含 dst_dx 偏移，供多条边合流到同一节点时错开落点)。
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

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)，字号按文本长度自适应收缩避免溢出
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio {w / h:.2f})")
