#!/usr/bin/env python3
"""ch35 本章地图:dot 矩阵乘指令选择 + 逐元素/归约/扫描降级 + PTX 出口剖面 ——
四条独立降级通路各走各的:①dot 按结果 NvidiaMma 布局版本分派 mma.884/1688/16816/
wgmma,或 FMA 兜底,Ampere 用 ValueTableV2 凑 4A+2B+nC 拼一条 mma.sync;②逐元素走
CRTP「拆-算-拼」模板,fp8 转换查 srcMap 表拿 Fp8ConversionDesc;③reduce 用
shuffleXor 蝶形树(跳过共享内存往返)、scan 用 Kogge-Stone(shuffleUp+mask);
四条通路的 PTX 串最终都汇入 PTXBuilder::launch 拼成 LLVM::InlineAsmOp,NVGPU
dialect(wgmma/mbarrier 等)另走一条配对脊柱 NVGPUToLLVMPass,两条脊柱都收尾于
processPhiStruct 拆 struct phi —— 五级降级阶梯到此走完。

改自 .claude/skills/svg-diagram/references/example-chapter-map.py 模板(与
ch33/ch34 chapter-map.py 同构:站牌胶囊 / 入口绿#22c55e-出口橙#f97316-主线蓝
#3b82f6 / 高亮实线蓝-次要虚线灰 / cjk_text_width() 宽度估算 —— 均不可变,只改
下面的 DATA)。

本章顶层标题写作 `## §1 ...`(§ 前缀 + 单数字,无小数点),lint_chapter_map 的
`§N.M` 徽标正则(要求带小数点)不匹配它们 → 按自然标题处理 → **站牌一律用标题词
逐字子串,禁用 §N.M 小数点徽标**(与 ch33/ch34 natural-title 章同策略)。
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
    "dot 矩阵乘",
    "逐元素降级",
    "归约／扫描降级",
    "PTX 出口 ＋ LLVM IR 收尾",
]
# 上面泳道名刻意留短(< 110px)——本图左侧 16~148px 是"调用方"入口三支折线穿行的
# 空白带,泳道名一旦太长会被入口折线的竖直段压穿(见 [FIX-ROUND-2]);完整主题描述
# 已经在 TITLE 与各节点的符号/短语里给全了,泳道名只需给一个类别标签。

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本[标题词逐字子串,禁用 §N.M])
NODES = [
    ("t_dot_dispatch", 0, 0, 0, "DotOpConversion",
     "按结果布局版本分派 mma.884/1688/16816/wgmma，或 FMA 兜底", "dot 降级派单"),
    ("t_mma_operand", 0, 1, 0, "callMmaAmpere",
     "ValueTableV2 凑 4 个 A / 2 个 B / nC，拼一条 mma.sync", "拼一条 mma.sync"),

    ("t_elem_template", 1, 0, 0, "ElementwiseOpConversionBase",
     "拆 struct→unpackI32→createDestOps→重打包", "逐元素降级"),
    ("t_fp8_table", 1, 1, 0, "FpToFpOpConversion",
     "查 srcMap 拿 Fp8ConversionDesc，原生 cvt 或位操作回退", "fp8 转换"),

    ("t_reduce_bfly", 2, 0, 0, "warpReduce",
     "shuffleXor 蝶形树，跳过共享内存往返", "归约降级"),
    ("t_scan_kogge", 2, 1, 0, "warpScan",
     "Kogge-Stone：shuffleUp ＋ mask(lane≥i) ＋ select", "扫描降级"),

    ("t_ptx_launch", 3, 2, 0, "PTXBuilder::launch",
     "拼 asm_string ＋ 约束串 → LLVM::InlineAsmOp", "PTX 出口"),
    ("t_nvgpu_pass", 3, 2, 1, "NVGPUToLLVMPass",
     "wgmma/mbarrier 等批量降 asm，第三方挂载接缝", "PTX 出口"),
    ("t_phi_struct", 3, 3, 0, "processPhiStruct",
     "拆 struct phi 收尾，make_llir 出口", "LLVM IR 收尾"),
]
EDGES = [  # 同泳道内直线(相邻列)
    ("t_dot_dispatch", "t_mma_operand"),
    ("t_elem_template", "t_fp8_table"),
    ("t_reduce_bfly", "t_scan_kogge"),
    ("t_ptx_launch", "t_phi_struct"),
    ("t_nvgpu_pass", "t_phi_struct"),
]
# 跨泳道折行边(elbow):(src_id, dst_id, dst 落点 x 偏移, 转折点额外右移, 水平段额外上移)
# —— 三条通路(mma/fp8/scan)末端都汇入同一个 t_ptx_launch,turn_extra/drop_extra
# 逐条错开,避免转折的竖直段/水平段彼此重合成一条看不出三条独立通路的粗线。
WRAP_EDGES = [
    ("t_mma_operand", "t_ptx_launch", -34, 0, 0),
    ("t_fp8_table", "t_ptx_launch", 0, 18, 14),
    ("t_scan_kogge", "t_ptx_launch", 34, 36, 28),
    ("t_dot_dispatch", "t_nvgpu_pass", 0, 0, 0),
]
# (路线名, [(列, 站牌文本), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("① Tensor Core 判据",
     [(0, "dot 降级派单"), (1, "拼一条 mma.sync")], True),
    ("② fp8 转换开销",
     [(0, "逐元素降级"), (1, "fp8 转换")], False),
    ("③ 归约／扫描 shuffle",
     [(0, "归约降级"), (1, "扫描降级")], False),
    ("④ PTX 拼装＋IR 收尾",
     [(2, "PTX 出口"), (3, "LLVM IR 收尾")], False),
]
LEGEND = [("#22c55e", "入口：TTGIR 的 tt.dot／逐元素／reduce／scan 到达降级"),
          ("#3b82f6", "主线：各 op 各走一条通路，末端汇入 PTX 出口"),
          ("#f97316", "出口：LLVM IR 收尾后交给 ptxas 编 PTX／cubin")]
TITLE = "第 35 章 · dot／逐元素／归约／扫描降级 ＋ PTX 出口剖面（源码走线 + 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 230, 62
COL_GAP, ROW_GAP = 46, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 100, 26
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

# 入口接口桩:三条独立通路(dot/逐元素/归约)各有一支绿色箭头,从同一个"调用方"桩发出。
# 每支都走"水平出桩 → 竖直转到目标行高 → 水平进入节点左边框"的直角折线,
# 保证落到节点边框的最后一段严格水平——避免斜线箭头连着箭头头戳进节点内的文字行。
entry_targets = ["t_dot_dispatch", "t_elem_template", "t_reduce_bfly"]
entry_ys = [NODE_XY[t][1] + NODE_H / 2 for t in entry_targets]
stub_mid_y = sum(entry_ys) / len(entry_ys)
L.append(f'<rect x="{EDGE_MARGIN}" y="{stub_mid_y - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{stub_mid_y + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("TTGIR op 入")}</text>')
_entry_turn_x0 = EDGE_MARGIN + STUB_W + 14
for i, (t, ey) in enumerate(zip(entry_targets, entry_ys)):
    ex, _ = NODE_XY[t]
    turn_x = _entry_turn_x0 + i * 12
    p0 = (EDGE_MARGIN + STUB_W, stub_mid_y)
    p1 = (turn_x, stub_mid_y)
    p2 = (turn_x, ey)
    p3 = (ex, ey)
    L.append(f'<polyline points="{p0[0]:.1f},{p0[1]:.1f} {p1[0]:.1f},{p1[1]:.1f} '
             f'{p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}" '
             f'fill="none" stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩:全部通路最终收尾于 processPhiStruct,一支橙色箭头返回上层
xx, xy = NODE_XY["t_phi_struct"]; xy += NODE_H / 2
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("→ ptxas／cubin")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 同泳道内直线调用边(主线蓝)——多条边汇入同一节点时终点 y 各偏移
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

# 跨泳道折行边(elbow):右侧绕出 → 下降到目标泳道上方的空白带 → 沿空白带向左 →
# 短距下降落入目标节点顶部(含 dst_dx 偏移)。turn_extra/drop_extra 逐条错开,
# 避免多条边汇入同一目标时竖直段/水平段彼此重合成一条看不出多条独立通路的粗线。
for wsrc, wdst, dst_dx, turn_extra, drop_extra in WRAP_EDGES:
    wx1, wy1 = NODE_XY[wsrc]; wx2, wy2 = NODE_XY[wdst]
    p_start = (wx1 + NODE_W, wy1 + NODE_H / 2)
    turn_x = wx1 + NODE_W + WRAP_GAP + turn_extra
    drop_y = wy2 - 8 - drop_extra
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
    sym_size = fit_size(symbol, NODE_W - 18, 13, 8.5)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.4:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sym_size:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    ph_size = fit_size(phrase, NODE_W - 16, 10.5, 8)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.7:.1f}" text-anchor="middle" '
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
