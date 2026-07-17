#!/usr/bin/env python3
"""ch20《布局即函数：GPU 张量凭什么和普通张量不同》—— 本章地图（源码剖面图）。

本章是自然标题 primer 章（chapter.md 无 `## N.M` 编号，只有自然标题）——按
illustrator 契约「自然标题章：禁用 §N.M 徽标，站牌改用标题词本身」，本图**不出现
任何 §N.M 徽标**，站牌一律用正文实际标题里的词（如「正式定义」「两大类分野」）。

本章叙事形状：两件套缺口 → 核心定义 L（顶悟图）→ 两大类分野（distributed 继续
往下展开、shared 在本节内就讲完）→ distributed 怎么算座位表（四级层次 → Blocked
三元组 → broadcast/wrap-around）→ 模块契约锁定线程总数 n → 前瞻 GF(2)。八个节点、
三段语义转折，沿用 ch02 chapter-map 验证过的「三泳道蛇形折行」几何（避免单行 8
节点把画布拉过宽）。

■ 不可变（同 example-chapter-map.py / ch02 chapter-map.py）：入口绿#22c55e·
  出口橙#f97316·主线蓝#3b82f6 / 路线高亮实线蓝-次要虚线灰 / cjk_text_width()
  宽度估算 / >2 种语义色画图例。

■ 本章专属改动（因内容形状 + 自然标题规则，非模板缺陷）：
  1. 无 §N.M 徽标——corner 站牌一律是标题词本身的自然语言短语（如「两大类分野」
     「broadcast 与 wrap-around」），不是 `§20.x` 编码。
  2. 站牌宽度按 cjk_text_width() 动态算（不同标题词长短差异大，如「前瞻」2 字 vs
     「broadcast 与 wrap-around」十几字），且站牌整体收在节点右上角**内部**
     （不像 example 模板那样跨出节点右边界一截）——避免长站牌把相邻列的节点
     文字压穿（见下方 ROUND-1 修复记录）。
  3. 三泳道各自局部列号（不共用 COLX），泳道间以折角虚线「续下一行」相接，
     与 ch02 处理手法一致：本章同样是一条主叙事线，没有真正的并行分叉需要
     用"同列=分支对齐"来表达。
  4. 底部只有一条阅读路线（本章严格线性：两大类分野讲完 shared 就地收尾，
     distributed 独自继续到模块契约），不强行编第二条路线充数。

■ 六项自查（渲染→Read PNG 亲眼看后如实记录）：
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  [ROUND-1]：
    - 首版站牌沿用 example 模板"跨出节点右边界"的挂法，「broadcast 与
      wrap-around」（约 17 个全角字符宽）跨出后压住了 broadcast/wrap-around
      节点右侧、bw→contract 折角线的文字标注。改为站牌整体收在节点框内部
      （右边距固定 4px，宽度按 cjk_text_width 动态算但不超出节点宽度），
      重渲后确认站牌不再压线/压字。
    - 「模块契约锁定 n」「L 是 GF(2) 线性映射」等节点副标题初版单行超宽，
      复用 ch02 的「按字符累计宽度过半即换行、且回退到最近 CJK/空白边界避免
      断词」逻辑拆两行，重渲后确认无断词、无溢出。
  [ROUND-2]（渲染后 Read PNG 发现，同轮修复重渲）：
    - 节点副标题里的长英文短语（"sizePerThread/threadsPerWarp/warpsPerCTA"、
      "bases/RREF"）用旧版 wrap 逻辑（仅认空白/CJK 为安全断点）会一路回退到
      词首，切出 "s"/"izePerThread" 这种断词；`_is_break_char` 补充
      `/ - , ; : ( ) =` 等软分隔符后重渲，确认不再断词、每行都在真实语义边界
      换行。
    - 底部阅读路线原用"首尾站牌定两端、中间等距切分"，但 8 个站牌宽度差异悬殊
      （"前瞻"2 字 vs "broadcast 与 wrap-around"十几字），等距切分让宽站牌
      彼此压住（"Blocked 三元组"与"broadcast 与 wrap-around"两个胶囊重叠、
      后者文字被截断）；改为按各自实际宽度**累加**布局（同图例 LEGEND 的累加
      摆法）+ 画布宽度按"节点区宽度 vs 路线所需宽度"取较大值，重渲后 8 个
      站牌间距均匀、零重叠。
  以上均为重渲染 + 重新 Read PNG 复核后的结果，非凭记忆照抄。

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
LANES = ["两件套缺口 → 核心定义 → 两大类分野", "distributed 怎么算座位表", "线程总数与前瞻"]

# (节点id, 泳道下标, 泳道内局部列, 真实符号/概念名, 一行短语, 站牌(自然标题词), 全局阅读序号)
NODES = [
    ("motiv",    0, 0, "两件套不够",
     "GPU 数据被多线程同时持有，切分方式须固化进类型",
     "两件套", 0),
    ("defn",     0, 1, "L：索引 → 线程集合",
     "L(0,0)={0,4} 顿悟例；回应上一章 encoding 恒空的悬念",
     "正式定义", 1),
    ("split",    0, 2, "distributed / shared",
     "distributed=小集合（继续展开）；shared=全员可见（本节收尾）",
     "两大类分野", 2),
    ("hier",     1, 0, "四级层次生成 L",
     "CTA→Warp→Thread→Value 自顶向下；上两级按 shape/order 连续填号",
     "四级计算层次", 3),
    ("blocked",  1, 1, "Blocked 三元组",
     "sizePerThread 等三元组决定每级占多少元素，16×16 例逐格核对",
     "Blocked 三元组", 4),
    ("bw",       1, 2, "broadcast / wrap-around",
     "一格多号=broadcast；一号多格=wrap-around，同一条取模公式两分支",
     "broadcast 与 wrap-around", 5),
    ("contract", 2, 0, "模块契约锁定 n",
     "num-warps 强制、其余缺省 → 锁定线程总数 n",
     "模块契约", 6),
    ("forward",  2, 1, "L 是 GF(2) 线性映射",
     "深化（bases/异或律/RREF）留给下一章，本章只带走一句",
     "前瞻", 7),
]
# (src_id, dst_id, is_wrap) —— 主线蓝；is_wrap=True 的两条走折角连接线（跨泳道换行）
EDGES = [
    ("motiv", "defn", False), ("defn", "split", False),
    ("split", "hier", True),
    ("hier", "blocked", False), ("blocked", "bw", False),
    ("bw", "contract", True),
    ("contract", "forward", False),
]
# 唯一一条阅读路线：本章严格线性（两大类分野讲完 shared 就地收尾），不编第二条充数
# （ROUTE_NAME / ROUTE_STOPS 已在上方"画布宽度"计算处定义，此处不重复定义）
LEGEND = [("#22c55e", "入口：读者进入本章"), ("#3b82f6", "章内讲解顺序/依赖链"), ("#f97316", "出口：带座位表读法进入下一章")]
TITLE = "第 20 章 · 布局即函数剖面：从两件套缺口到座位表"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 200, 62
COL_GAP, ROW_GAP = 40, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 108, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32
LANE_LABEL_H, BAND_PAD = 24, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H = 20
BADGE_PAD_X = 10  # 站牌内部左右各留的文字内边距
WRAP_GAP = 30  # 折行处（跨泳道连接线）额外留白

# 每条泳道各自的局部列数(不跨泳道共享列号 —— 见文件头说明)
cols_per_lane = [0] * len(LANES)
for _id, lane, col, *_ in NODES:
    cols_per_lane[lane] = max(cols_per_lane[lane], col + 1)
n_cols_max = max(cols_per_lane)

# 画布宽度取两者较大值：① 最宽的一条泳道（节点区）；② 底部阅读路线所需宽度。
# 路线站牌是自然标题词、长短差异极大（"前瞻"2 字 vs "broadcast 与 wrap-around"
# 十几字），若像 §N.M 数字徽标那样等距摆放，宽站牌会互相压住/文字被吃掉——
# 必须按各自实际宽度累加布局（同图例 LEGEND 的累加摆法），再据此反推路线总宽度。
w_lanes = PAD_L + n_cols_max * NODE_W + (n_cols_max - 1) * COL_GAP + PAD_R

ROUTE_FONT = 10.5
ROUTE_GAP = 16  # 相邻站牌之间的最小间距
ROUTE_STOPS = [n[5] for n in NODES]  # 站牌文字，按 NODES 声明顺序（已按阅读序排列）
ROUTE_NAME = "唯一路线（严格线性通读）"
_route_bw = [cjk_text_width(s, ROUTE_FONT) + 10 * 2 for s in ROUTE_STOPS]  # 10=BADGE_PAD_X，下方另有同名常量
_route_name_w = cjk_text_width(ROUTE_NAME, 12) + 28
_route_start = 16 + _route_name_w
w_route_required = _route_start + sum(_route_bw) + (len(ROUTE_STOPS) - 1) * ROUTE_GAP + 16

w = max(w_lanes, w_route_required)

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
for nid, lane, col, symbol, phrase, badge_text, order in NODES:
    x = PAD_L + col * (NODE_W + COL_GAP)
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
h = routes_top + ROUTE_HEAD_H + ROUTE_ROW_H + BOTTOM_PAD


def badge_pill(cx_right, cy, text, font_size=10.5):
    """站牌胶囊——本章为自然标题章，文字是标题词本身（非 §N.M）。整个胶囊收在
    节点右上角内部（右边界与节点内边距对齐，不跨出节点右边界），宽度按
    cjk_text_width() 动态算，避免长站牌（如「broadcast 与 wrap-around」）
    跨出节点边界压住相邻列。"""
    bw = cjk_text_width(text, font_size) + BADGE_PAD_X * 2
    bx = cx_right - bw
    by = cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{bx + bw / 2:.1f}" y="{cy + 3.6:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{font_size}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ], bw


def route_badge(cx, cy, text, font_size=10.5):
    """底部阅读路线上的站牌——居中挂在路线上（复用 badge_pill 的胶囊视觉，
    但按几何中心而非右边界定位）。"""
    bw = cjk_text_width(text, font_size) + BADGE_PAD_X * 2
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.6:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{font_size}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
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
ex, ey = NODE_XY["motiv"]; ey += NODE_H / 2
xx, xy = NODE_XY["forward"]; xy += NODE_H / 2
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
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一章：布局家族")}</text>')
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


_SOFT_BREAK_PUNCT = set("/-,;:()=，、；：（）")


def _is_break_char(ch):
    """可在其后安全换行的字符：空白、CJK（含全角标点，逐字断行天然安全）、
    以及英文/符号短语里常见的软分隔符（/、-、,、;、:、(、)、=）——
    没有这些，长的无空格英文短语（如 "sizePerThread/threadsPerWarp"、
    "bases/RREF"）回退查找会一路退到词首，切出形如 "s"/"izePerThread"
    的断词。"""
    return ch.isspace() or ord(ch) > 0x2E80 or ch in _SOFT_BREAK_PUNCT


def wrap_lines(phrase, font_size, max_w, max_lines=2):
    """贪心按宽度换行，最多 max_lines 行；每行末尾回退到最近的"安全断点"
    （_is_break_char 为真的位置之后），避免把一个连续的英文单词/标识符从
    中间切断。若找不到安全断点（整行都是无分隔的长英文词）才硬切，避免
    死循环。超出 max_lines 容量的剩余文本并入最后一行（宁可略拥挤，也不
    静默丢字）。"""
    lines = []
    remaining = phrase
    while remaining:
        if len(lines) == max_lines - 1 or cjk_text_width(remaining, font_size) <= max_w:
            lines.append(remaining)
            remaining = ""
            break
        acc, cut = 0.0, len(remaining)
        for i, ch in enumerate(remaining):
            acc += font_size * (1.0 if ord(ch) > 0x2E80 else 0.58)
            if acc > max_w:
                cut = i
                break
        safe_cut = cut
        while safe_cut > 1 and not _is_break_char(remaining[safe_cut - 1]):
            safe_cut -= 1
        if safe_cut <= 1:
            safe_cut = max(cut, 1)  # 找不到安全断点：硬切，避免死循环/空行
        lines.append(remaining[:safe_cut].rstrip())
        remaining = remaining[safe_cut:].lstrip()
    return lines


# 节点
for nid, lane, col, symbol, phrase, badge_text, order in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.32:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    max_w = NODE_W - 16
    phrase_lines = wrap_lines(phrase, 9.5, max_w, max_lines=2)
    if len(phrase_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.60:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(phrase_lines[0])}</text>')
    else:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.58:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(phrase_lines[0])}</text>')
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.82:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(phrase_lines[1])}</text>')
    pill, _bw = badge_pill(x + NODE_W - 4, y, badge_text)
    L += pill

# 底部阅读路线：本章唯一路线。站牌宽度差异极大（"前瞻"2 字 vs "broadcast 与
# wrap-around"十几字），改按各自实际宽度**累加**布局（同图例摆法），而非等距
# 切分——等距切分在站牌宽度悬殊时会让宽站牌互相压住（ROUND-2 修复，见下）。
n_stops = len(ROUTE_STOPS)
ry = routes_top + ROUTE_HEAD_H + ROUTE_ROW_H / 2
ROUTE_X = []
_rx = _route_start
for bw_i in _route_bw:
    ROUTE_X.append(_rx + bw_i / 2)
    _rx += bw_i + ROUTE_GAP
route_x1 = ROUTE_X[-1]

L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;本章严格线性,只有一条路线)")}</text>')
L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
          f'fill="{C_NODE_TITLE}">{esc(ROUTE_NAME)}</text>')
L.append(f'<line x1="{ROUTE_X[0]:.1f}" y1="{ry:.1f}" x2="{route_x1:.1f}" y2="{ry:.1f}" '
          f'stroke="{C_MAIN}" stroke-width="3"/>')
for i, stop in enumerate(ROUTE_STOPS):
    pill, _bw = route_badge(ROUTE_X[i], ry, stop)
    L += pill

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio={w / h:.2f}:1)")
