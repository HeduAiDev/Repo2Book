#!/usr/bin/env python3
"""ch15《SSA 与结构化控制流：φ 节点、块参数与 loop-carried 变量》—— 本章地图（源码剖面图）。

本章是纯线性 primer（论文理论台阶，无分支控制流可画）：§1→§5 严格按讲解顺序
推进，且原文小结本身已把全章总结成「三层，同一件事」——记号（Cytron）→IR
（MLIR）→前端（Triton）。本图直接借用这三层做泳道分组（不是外加的结构，是
正文小结表格本来的分层），每层 1~2 站，比按单一大泳道摊平更贴合"本章讲的是
同一件事的三种落法"这个核心论点。

■ 不可变（同 example-chapter-map.py）：§徽标胶囊 / 入口绿#22c55e·出口橙#f97316·
  主线蓝#3b82f6 / 路线高亮实线蓝-次要虚线灰 / cjk_text_width() 宽度估算。

■ 本章专属改动（可变部分，因内容形状而非模板缺陷）：
  1. 正文标题是「## §1 …」「## §2 …」…「## §5 …」（自然 §N 单级编号，非
     「## N.M」两级),站牌沿用正文原生的 §1..§5(与 ch02 同款先例),不杜撰
     §N.M。「小结」无编号,不单独入图(其"三层"结构改用为泳道名)。
  2. 全章只有一条线性主线，没有分叉，故只有 1 条"阅读路线"(§1→§5 通读)，
     不强行编第二条路线充数(同 ch02 先例)。
  3. 三条泳道节点数不等(2/2/1)，第二、三层从画布左侧重新起笔(蛇形折行，
     跨泳道边走折角虚线并标"续下一行")——同 ch02 的折行手法,压宽高比。
  4. 节点 symbol 字段取本章两类可核对的"论文概念/代码符号"(primer 章节点=
     论文概念,但本章命门句与红线都逐字落在 dossier.mechanisms 的
     source_anchors/notation 上,故符号选取偏向"读者在正文里认得出的原样
     记号"):
       §1 set_value —— dossier.mechanisms[ssa-single-assignment].source_anchors
       §2 x₃=φ(x₁,x₂)  —— dossier.mechanisms[phi-merge-semantics].source_anchors 对应正文 §2 记号
       §3 block argument —— dossier.mechanisms[block-args-eq-phi] 命门引文的核心术语
       §4 scf.for / scf.if —— dossier.mechanisms[scf-for-arg-layout]/[scf-if-result]
       §5 local_defs ∩ liveins —— dossier.mechanisms[loop-carried-scope-diff].notation 原样

■ 六项自查（渲染→Read PNG 亲眼看后如实记录）：
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  （详见文件同目录 figure-manifest.json 的记录；本图不含 spec.numbers 意义上的
  "数字"——本图是论点/记号的源码剖面图，非数值轨迹图，numbers_match_spec 项
  核对的是"图上出现的 §编号/符号是否与正文/dossier 一一对应"。）

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["记号层（Cytron 1991）", "IR 层（MLIR，本章命门）", "前端层（Triton 落地）"]

# (节点id, 泳道下标, 泳道内局部列, 真实符号/概念名, 一行短语, §编号, 全局阅读序号)
NODES = [
    ("ssa",         0, 0, "set_value",
     "每次赋值重绑新版本；SSA 不变量：每个版本只被赋值一次", "§1", 0),
    ("phi",         0, 1, "x₃ = φ(x₁, x₂)",
     "汇合块按前驱选值恢复单赋值；红线：只是记号，非 Triton 算法", "§2", 1),
    ("blockarg",    1, 0, "block argument",
     "命门：terminator 把值推进后继块参数——φ「拉」↔块参数「推」等价", "§3", 2),
    ("scffor",      1, 1, "scf.for / scf.if",
     "arg(0)=归纳变量，arg(i+1)=iter_arg；scf.yield 把值推给下一轮/结果", "§4", 3),
    ("loopcarried", 2, 0, "local_defs ∩ liveins",
     "dry-run 探名字→交集认 loop-carried；红线闭环：零支配边界计算", "§5", 4),
]
# (src_id, dst_id, is_wrap) —— 主线蓝；is_wrap=True 的边走折角连接线（跨泳道换行）
EDGES = [
    ("ssa", "phi", False),
    ("phi", "blockarg", True),
    ("blockarg", "scffor", False),
    ("scffor", "loopcarried", True),
]
# 唯一一条阅读路线：本章严格线性，不编第二条路线充数
ROUTE_NAME = "唯一路线（三层递进：记号→IR→前端，红线贯穿三处）"
ROUTE_STOPS = [n[5] for n in NODES]  # 按 NODES 声明顺序取 § 徽标（已按阅读序排列）
LEGEND = [("#22c55e", "入口：读者进入本章"), ("#3b82f6", "本章讲解顺序/依赖链"), ("#f97316", "出口：带三层地基回到全书主线")]
TITLE = "第 15 章 · SSA 与结构化控制流剖面：φ 记号 → 块参数 → loop-carried 落地"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 210, 60
COL_GAP, ROW_GAP = 40, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 96, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 40, 20
WRAP_GAP = 30  # 折行处（跨泳道连接线）额外留白

# 每条泳道各自的局部列数(不跨泳道共享列号 —— 见文件头说明)
cols_per_lane = [0] * len(LANES)
for _id, lane, col, *_ in NODES:
    cols_per_lane[lane] = max(cols_per_lane[lane], col + 1)
n_cols_max = max(cols_per_lane)

# 画布宽度按最宽的一条泳道定
w = PAD_L + n_cols_max * NODE_W + (n_cols_max - 1) * COL_GAP + PAD_R

rows_per_lane = [1] * len(LANES)  # 每条泳道本章都只用 1 行
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += WRAP_GAP
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, symbol, phrase, sec, order in NODES:
    x = PAD_L + col * (NODE_W + COL_GAP)
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
h = routes_top + ROUTE_HEAD_H + ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
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
# 图例
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 34

# 泳道背景 + 标签
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')

# 入口/出口接口桩
ex, ey = NODE_XY["ssa"]; ey += NODE_H / 2
xx, xy = NODE_XY["loopcarried"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("读者入口")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("接下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用/讲解顺序边（主线蓝）；跨泳道折行的一条画成折角路径
for src, dst, is_wrap in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    if not is_wrap:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    else:
        # 折角连接线：从上一泳道节点右侧出发，先向右探出一小段，再折向下、折向左，
        # 落到下一泳道第一个节点的顶边中点——明确标出"续接下一行"
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        mid_x = x1 + NODE_W + 20
        p2 = (x2 + NODE_W / 2, y2)
        path = f'M {p1[0]:.1f},{p1[1]:.1f} L {mid_x:.1f},{p1[1]:.1f} L {mid_x:.1f},{p2[1] - 14:.1f} L {p2[0]:.1f},{p2[1] - 14:.1f} L {p2[0]:.1f},{p2[1]:.1f}'
        L.append(f'<path d="{path}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
                  f'stroke-dasharray="5,3" marker-end="url(#mMain)"/>')
        L.append(f'<text x="{mid_x + 6:.1f}" y="{(p1[1] + p2[1] - 14) / 2:.1f}" font-family="sans-serif" '
                  f'font-size="9.5" fill="{C_ROUTE_DIM}">{esc("续下一行")}</text>')

# 节点
for nid, lane, col, symbol, phrase, sec, order in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.34:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    # 短语可能偏长，按 cjk 宽度粗估是否需要拆两行（节点宽减去左右各 8px 内边距）
    max_w = NODE_W - 16
    if cjk_text_width(phrase, 9.5) <= max_w:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.62:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    else:
        # 按字符切两行(粗切:累计宽度超一半即换行)，但不许在一段连续的西文/符号
        # 词内部切断——向左回退到最近的 CJK 字符或空白边界(cjk_or_space 判定
        # 任一侧成立即可安全切)，避免出现断词。
        def _cjk_or_space(ch):
            return ch.isspace() or ord(ch) > 0x2E80

        acc, cut = 0.0, len(phrase)
        for i, ch in enumerate(phrase):
            acc += 9.5 * (1.0 if ord(ch) > 0x2E80 else 0.58)
            if acc > max_w:
                cut = i
                break
        while cut > 1 and not (_cjk_or_space(phrase[cut - 1]) or _cjk_or_space(phrase[cut])):
            cut -= 1
        line1, line2 = phrase[:cut], phrase[cut:]
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.56:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(line1)}</text>')
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.80:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(line2)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 6, y, sec)

# 底部阅读路线：本章唯一路线，用独立等距时间轴(不复用节点局部列坐标)
# 时间轴起点让在路线名文字之后(留够半个徽标宽的安全距离)，否则首个徽标会和
# 名称文字的尾字重叠——终点仍对齐画布右边界，与上方内容区宽度一致。
n_stops = len(ROUTE_STOPS)
ry = routes_top + ROUTE_HEAD_H + ROUTE_ROW_H / 2
name_w = cjk_text_width(ROUTE_NAME, 12) + 28
line_x0 = 16 + name_w + BADGE_W / 2
route_x1 = w - PAD_R
ROUTE_X = [line_x0 + i * (route_x1 - line_x0) / (n_stops - 1) for i in range(n_stops)]

L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;本章严格线性,只有一条路线)")}</text>')
L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
          f'fill="{C_NODE_TITLE}">{esc(ROUTE_NAME)}</text>')
L.append(f'<line x1="{ROUTE_X[0]:.1f}" y1="{ry:.1f}" x2="{ROUTE_X[-1]:.1f}" y2="{ry:.1f}" '
          f'stroke="{C_MAIN}" stroke-width="3"/>')
for i, sec in enumerate(ROUTE_STOPS):
    L += badge(ROUTE_X[i], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio={w / h:.2f}:1)")
