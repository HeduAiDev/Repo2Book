#!/usr/bin/env python3
"""第 33 章「本章地图」——两个采样器(AscendSampler / AscendRejectionSampler)的薄壳覆写剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，只改 DATA + 一处结构性
泛化——入口/出口从「单点」推广为「单个接口桩、多条箭头」，因为本章从 ch16 的
_sample 派发进来就分两条独立路线(无 spec → AscendSampler；有 spec →
AscendRejectionSampler)，两条路线各自独立入、独立出，不在图内合并。

节点预算 10 ≤ 12。本章标题为编号标题(## 33.1 ... ## 33.6)，站牌用 §33.N。

设计要点：
- 两条泳道 = 两个采样器各自的路径，不是调用栈深度分层：
  AscendSampler 路径(普通解码,§33.1/33.2/33.3/33.4/33.5) 与
  AscendRejectionSampler 路径(投机解码,§33.5/33.6)，两条路线共享同一个
  「HAS_TRITON 优雅回退」母题(体现为 col0 的两个 apply_penalties 节点)、
  同一套 Gumbel-max 数学(体现为 col3 的 random_sample 与 col2 的
  sample_recovered_tokens 都做 div_(q).argmax)。
- AscendSampler 路径内部再分叉：apply_penalties 之后，贪心分支(greedy_sample)
  直接到出口，随机分支(forward_native→apply_top_k_top_p→random_sample)绕一圈
  再到出口——对应基类 sample() 里 greedy_sampled / random_sampled 两路 torch.where
  合并的真实控制流(dossier data_flow #2)。
- AscendRejectionSampler 路径画的是随机接受判据的完整链路：rejection_sample→
  sample_recovered_tokens(接受判据未过时从残差分布重采)→output_token_ids。
  全贪心提前返回(`if sampling_metadata.all_greedy: return output_token_ids`)
  是真实存在的分支，但若在图上从 rejection_sample 另画一条直达 exit 的边，
  会和同一行的 sample_recovered_tokens 节点共线、边线穿过其box——[FIX-ROUND-2]
  改为只画随机路径这条完整链路，贪心早退分支留给正文讲，不在图上硬塞。
- 入口/出口桩：本章两条路线都来自 ch16 的同一个上层派发点、也都回到同一个
  上层调用方，所以左右各保留「一个」接口桩(呼应模板的"各放一个"不可变约定)，
  桩内用两条箭头分别接两条路线的第一/最后一个节点——不是模板代码的字面
  单箭头实现，是同一视觉语言在"一入口两路线"场景下的泛化。

符号可核性说明：长的 "ClassName.method_name" 符号用 "\\n" 机械换行成两行
分别渲染为两个独立 <text> 元素(与 ch20 chapter-map.py 同一模式)——lint 按
<text> 元素逐个核对，换行后 "ClassName" 因无 "_"/"."/"(" 触发不被核对，
".method_name" 的前导点不满足 token 起始字符会被丢弃，实际核对的是裸
"method_name"，均已在正文/dossier 原文出现，见各节点旁注。

六项自查记录见文件末尾 [SELF-CHECK] 注释(渲染→Read PNG 亲眼看后如实记录)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["AscendSampler：普通解码路径", "AscendRejectionSampler：投机解码路径"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语, §编号)
NODES = [
    ("pen_norm",  0, 0, 0, "AscendSampler\n.apply_penalties",        "HAS_TRITON 优雅回退基类原版",             "§33.5"),
    ("greedy",    0, 1, 0, "greedy_sample",                          "覆写:单卡同基类,加 TP 旁路",              "§33.1"),
    ("fwdnative", 0, 1, 1, "AscendTopKTopPSampler\n.forward_native", "覆写:top-k/top-p 后接 Gumbel 采样",       "§33.3"),
    ("topkp",     0, 2, 1, "apply_top_k_top_p",                      "A2/A3 走 AscendC,其余纯 torch",          "§33.4"),
    ("randsamp",  0, 3, 1, "random_sample",                          "Gumbel 继承上游,stream 异步是新增",       "§33.2"),
    ("exit_norm", 0, 4, 0, "torch.where",                            "按温度合并贪心/随机两路(继承)",           "§33.1"),
    ("pen_spec",  1, 0, 0, "AscendRejectionSampler\n.apply_penalties", "同 HAS_TRITON 回退,按 repeat_indices 广播", "§33.6"),
    ("rejsamp",   1, 1, 0, "rejection_sample",                       "HAS_TRITON 走 Triton,否则 *_pytorch 回退", "§33.6"),
    ("recover",   1, 2, 0, "sample_recovered_tokens",                "残差重采,复用同一 Gumbel 数学",           "§33.6"),
    ("exit_spec", 1, 4, 0, "output_token_ids",                       "接受/拒绝/bonus 写定,返回上层",           "§33.6"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("pen_norm", "greedy"), ("pen_norm", "fwdnative"),
    ("fwdnative", "topkp"), ("topkp", "randsamp"),
    ("greedy", "exit_norm"), ("randsamp", "exit_norm"),
    ("pen_spec", "rejsamp"),
    ("rejsamp", "recover"),
    ("recover", "exit_spec"),
]
ENTRY_IDS = ["pen_norm", "pen_spec"]  # 单个左侧接口桩,两条箭头分别接两条路线的第一个节点
EXIT_IDS = ["exit_norm", "exit_spec"]  # 单个右侧接口桩,两条箭头分别接两条路线的最后一个节点

# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("默认解码:随机采样主线", [(0, "§33.5"), (1, "§33.3"), (2, "§33.4"), (3, "§33.2"), (4, "§33.1")], True),
    ("投机解码:拒绝采样校验", [(0, "§33.6"), (1, "§33.6"), (2, "§33.6"), (4, "§33.6")], False),
    ("跳读:Gumbel-max 数学(两处复用同一等价式)", [(2, "§33.6"), (3, "§33.2")], False),
]
LEGEND = [("#22c55e", "入口:两个采样器均由上层派发进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 33 章 · 两个采样器的薄壳覆写剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 210, 90
COL_GAP, ROW_GAP = 32, 22
EDGE_MARGIN, STUB_W, STUB_H = 12, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 46, 20

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
    """§ 徽标胶囊,居中挂在 (cx,cy) —— 节点用它贴右上角,路线legend用它居中挂线上。"""
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
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 34

# 泳道背景 + 标签 + 分隔线。本章入口/出口桩纵跨两条泳道(见下文),
# 标签 x 起点须让在桩右侧(EDGE_MARGIN+STUB_W+16)，否则泳道标签文字会被
# 又高又不透明的桩矩形压在下面——[FIX-ROUND-2] 由 x=16 移到桩外。
LANE_LABEL_X = EDGE_MARGIN + STUB_W + 16
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="{LANE_LABEL_X}" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩:单个桩、多条箭头(两条路线各自独立进出,不在图内合并)。
# 桩的高度从最小 y 撑到最大 y、覆盖所有目标箭头的落点,标签居中放一次。
entry_ys = [NODE_XY[i][1] + NODE_H / 2 for i in ENTRY_IDS]
e_top, e_bot = min(entry_ys) - STUB_H / 2, max(entry_ys) + STUB_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{e_top:.1f}" width="{STUB_W}" height="{e_bot - e_top:.1f}" '
         f'rx="16" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{(e_top + e_bot) / 2 + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
for eid in ENTRY_IDS:
    ex, ey = NODE_XY[eid]; ey += NODE_H / 2
    L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
             f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

exit_ys = [NODE_XY[i][1] + NODE_H / 2 for i in EXIT_IDS]
x_top, x_bot = min(exit_ys) - STUB_H / 2, max(exit_ys) + STUB_H / 2
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{x_top:.1f}" width="{STUB_W}" height="{x_bot - x_top:.1f}" '
         f'rx="16" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{(x_top + x_bot) / 2 + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
for xid in EXIT_IDS:
    xx, xy = NODE_XY[xid]; xy += NODE_H / 2
    L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移(间距 16px),
# 否则重合的终点在视觉上看不出"汇合"、像一条线断头。
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

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(始终锚在节点下半区) + 右上角 § 徽标)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 24, 40
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 71, 66, 80
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines = symbol.split("\n")
    sym_ys = [y + SYMBOL_1LINE_Y] if len(sym_lines) == 1 else [y + SYMBOL_2LINE_Y1, y + SYMBOL_2LINE_Y2]
    for line, ly in zip(sym_lines, sym_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = phrase.split("\n")
    phrase_ys = [y + PHRASE_1LINE_Y] if len(phrase_lines) == 1 else [y + PHRASE_2LINE_Y1, y + PHRASE_2LINE_Y2]
    for line, ly in zip(phrase_lines, phrase_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
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
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见下方注释)
