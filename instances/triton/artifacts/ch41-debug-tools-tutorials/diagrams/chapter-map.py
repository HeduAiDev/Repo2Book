#!/usr/bin/env python3
"""本章地图 — ch41《调试与学习：triton-opt 家族、tensor-layout 探针与 tutorials
阶梯》源码剖面图。

三条并列泳道,对应正文实际的三大段(彼此无调用关系,是三件独立工具/学习资源):
  1. triton-opt 家族(bin/):四个薄壳工具共用同一个 registerTritonDialects
     注册表,再各自交给不同的 MLIR 官方驱动。
  2. tensor-layout 探针(bin/→lib/):-l/-t 拼出带 encoding 的 tensor 类型,
     一路分派到 getDistributedLayoutStr 的四重循环求值。
  3. tutorials 学习阶梯(python/tutorials):01-vector-add 起步,06-fused-attention
     是阶梯顶端(FlashAttention v2 预告)。

■ 不可变(全书统一视觉语言,换章节数据时不要动这些,只改下面的 DATA):
  1. 站牌胶囊:圆角矩形(pill),fill #eef2ff / stroke #6366f1;贴节点右上角。
  2. 入口/出口接口桩:绿 #22c55e(入口) / 橙 #f97316(出口)。
  3. 节点间主线边 = 蓝 #3b82f6。
  4. 底部路线条:高亮=实线蓝(粗)/次要=虚线灰 #94a3b8(细)。
  5. >2 种语义色须画图例。
  6. 文本宽度估算一律用 cjk_text_width(),不用半角系数硬乘 len(s)。

■ 本章特有(自然标题章,无 §N.M 编号——按 illustrator 契约:禁用 §N.M 徽标,
  站牌改用标题词本身,逐字取自 chapter.md 真实 `## ...`/`### ...` 标题的子串):
  - 8 个节点的站牌均为对应小节标题的逐字子串(如 "从命令行到分派" 取自
    `### 机制：从命令行到分派`;"一道认知阶梯" 取自 `## tutorials 01→09：
    一道认知阶梯`)。3 条泳道名各取自本章三个顶层 `## ...` 标题的关键词。
  - 三条泳道彼此独立(triton-opt 家族 / tensor-layout 探针 / tutorials 阶梯
    互不调用),泳道内的蓝色主线边表示"章内调用/阅读顺序",不是跨泳道调用图;
    图例文案已按此措辞,避免误导读者以为三段互相调用。
  - tutorials 泳道两个节点(vector_add/fused_attn)刻意错列到 col2/col3(与
    另两条泳道共用的列坐标对齐),这样底部"按调试工作流通读"路线才能用单调
    递增的列号一次串起三大段,不必给 tutorials 单独让出新列(与 ch09 chapter-map
    「lane3 起步列刻意后移对齐路线」的先例做法一致)。
  - 节点符号名一律不带空括号(`getLayoutStr` 不写 `getLayoutStr()`)——
    lint_chapter_map 的杜撰符号检测按字面子串核对,dossier/正文里从不出现
    空括号写法,只有带参数的调用形式,与已有先例(ch09/ch13)一致。
  - 站牌/图例/短语一律用全角中文括号「（）」而非 ASCII "()" 包住汉字说明,
    避免半角括号与紧邻的代码 token 意外拼出杜撰子串(如 "BLOCK_SIZE)" 这种
    带残留右括号的假 token)。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
  claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
  arrows_attached=True     cjk_rendered=True         reading_order_clear=True

用法:python3 chapter-map.py → 同目录 chapter-map.svg
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


# ==================== DATA(可变:本章数据) ====================
LANES = [
    "triton-opt 家族（bin/）",
    "tensor-layout 探针（bin/→lib/）",
    "tutorials 学习阶梯（python/tutorials）",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本[取自真实标题子串])
NODES = [
    ("opt_main", 0, 0, 0, "triton-opt",
     "建 registry，交给 MlirOptMain 跑",
     "triton-opt 家族"),
    ("siblings", 0, 0, 1, "triton-reduce / triton-lsp",
     "同一 registry，换 MLIR 驱动跑",
     "薄到什么程度"),
    ("register", 0, 1, 0, "registerTritonDialects",
     "13 个 dialect ＋ 全部 pass，四工具共用",
     "registerTritonDialects 填了什么"),
    ("mlir_opt_main", 0, 2, 0, "MlirOptMain",
     "解析 --pass-name，只跑单个 pass",
     "单跑一个 pass"),

    ("tl_main", 1, 0, 0, "triton-tensor-layout",
     "解析 -l/-t，拼出带 encoding 的 tensor",
     "从命令行到分派"),
    ("layout_print", 1, 1, 0, "layoutPrint",
     "按 dialect 是否 triton_gpu 分派",
     "从命令行到分派"),
    ("get_layout_str", 1, 2, 0, "getLayoutStr",
     "Shared/Distributed 两路分派",
     "从命令行到分派"),
    ("get_distributed", 1, 3, 0, "getDistributedLayoutStr",
     "四重循环求 element 归属",
     "四重循环求值"),

    ("vector_add", 2, 2, 0, "add_kernel",
     "01：programming model 起点",
     "一道认知阶梯"),
    ("fused_attn", 2, 3, 0, "_attn_fwd_inner",
     "06：FlashAttention v2 在线 softmax",
     "一道认知阶梯"),
]

EDGES = [  # (src_id, dst_id) —— 泳道内调用/阅读顺序,不跨泳道,统一主线蓝
    ("opt_main", "register"), ("siblings", "register"),
    ("opt_main", "mlir_opt_main"),
    ("tl_main", "layout_print"), ("layout_print", "get_layout_str"),
    ("get_layout_str", "get_distributed"),
    ("vector_add", "fused_attn"),
]

# (路线名, [(列, 站牌文本), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("按调试工作流通读", [
        (0, "triton-opt 家族"), (1, "从命令行到分派"),
        (2, "四重循环求值"), (3, "一道认知阶梯"),
    ], True),
    ("只学布局怎么读", [(0, "从命令行到分派"), (3, "两种转置读法")], False),
]
LEGEND = [
    ("#22c55e", "入口：命令行敲 triton-opt / triton-tensor-layout，或翻开 tutorials 开始学"),
    ("#3b82f6", "泳道内调用边／tutorials 阶梯阅读顺序（三条泳道彼此不互相调用）"),
    ("#f97316", "出口：抓手带回自己的 kernel"),
]
TITLE = "调试与学习：triton-opt 家族、tensor-layout 探针与 tutorials 阶梯（本章地图）"

# ==================== 不可变:配色 ====================
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ==================== 几何常量(全计算,零魔数) ====================
NODE_W, NODE_H = 210, 62
COL_GAP, ROW_GAP = 30, 16
EDGE_MARGIN, STUB_W, STUB_H = 14, 62, 28
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24
LANE_LABEL_H, BAND_PAD = 22, 10
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 26, 14
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 38
BADGE_H, BADGE_PAD_X = 20, 10
BADGE_FONT = 11

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
grid_w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
# 图例一整行文字也可能比节点网格更宽(本章图例句子较长)——画布宽须取两者较大值,
# 否则最后一条图例会被 viewBox 裁掉(用图例循环里同样的 20+文字宽+28 步进公式预演一遍)。
LEGEND_FONT = 10.3
_legend_x = PAD_L
for _color, _label in LEGEND:
    _legend_x += 20 + cjk_text_width(_label, LEGEND_FONT) + 28
legend_w = _legend_x + PAD_R - 28  # 减掉最后一项多算的行间距,留 PAD_R 做右边距
w = max(grid_w, legend_w)
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——宽度按文本动态算(本章站牌是完整词组,非定长短码)。"""
    bw = cjk_text_width(text, BADGE_FONT) + BADGE_PAD_X * 2
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.1"/>',
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
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')

# 图例(3 种语义色 → 必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 13
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="{LEGEND_FONT}" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, LEGEND_FONT) + 28

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
ex, ey = NODE_XY["opt_main"]; ey += NODE_H / 2
xx, xy = NODE_XY["fused_attn"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.2"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.2"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 泳道内调用边/阅读顺序(不跨泳道),多条边汇入同一节点时终点 y 错开
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
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌),字号按文本长度自适应收缩避免溢出
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_size = fit_size(symbol, NODE_W - 16, 12.5, 8.5)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
              f'font-family="monospace" font-size="{sym_size:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    phrase_size = fit_size(phrase, NODE_W - 14, 9.6, 7.5)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{phrase_size:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W + 8 - (cjk_text_width(sec, BADGE_FONT) + BADGE_PAD_X * 2) / 2, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线（站牌＝图上节点站牌；实线蓝＝推荐通读／虚线灰＝只取结论）")}</text>')
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
