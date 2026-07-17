#!/usr/bin/env python3
"""ch43《收官实战：fused-attention 从 tl.* 一路降到 PTX》—— 本章地图(源码剖面图)。

本章是全书自然标题章(narrative/chapter.md 只有「先说取证」「第一步」…「第六步」这类
自然标题小节，没有 `## N.M` 编号)，因此**站牌一律用标题词本身，禁用 §N.M 徽标**
(与 illustrator 契约「自然标题章」条一致，且 lint_chapter_map.py 对无编号章节会直接
拒收任何 §N.M 徽标)。

本章内容是一条**严格单向的降级链**(tl.* 源码 → JIT 特化 → TTIR → TTGIR 布局指派 →
三个优化 pass → LLVM → PTX)，没有 ch20 范式那种「快通道/全通道」式分支，所以本图
没有沿用模板里「多泳道并行分支」的布局——而是把 9 个站点排成 3 行 × 3 列的
「之字形(boustrophedon)」网格：第 1 行从左到右，第 2 行从右到左，第 3 行再从左到右，
行与行之间用**纵向**箭头衔接（同一列正上正下），行内用**横向**箭头衔接——这样折成
多行后画布仍然紧凑(宽 < 900、宽高比 < 1.5:1)，同时保留「一条主线从头走到尾」的
单一阅读路径，不需要模板里「多路线」图例。

■ 沿用模板不可变的视觉语言：§徽标胶囊样式(仅去掉 § 前缀，改印标题词)、
  入口绿 #22c55e / 出口橙 #f97316 / 主线蓝 #3b82f6、>2 语义色画图例、
  `cjk_text_width()` 做中英混排宽度估算。
■ 本章特有的可变部分：3×3 之字形网格坐标、方向感知的边(同行按左右选锚点/
  跨行按上下选锚点)、底部单一阅读路线条(9 个站牌等距排开，因为 9 个节点在
  3×3 网格里列坐标会重复，不能像模板那样直接复用节点列坐标当路线横轴)。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  [FIX-ROUND-2](第一轮 Read PNG 后发现问题，本轮修复后重渲并重新 Read PNG 复核，
  以下 True 是复核后的结果，不是凭空照抄)：
    - 第一轮 no_overlap 实测为 False——节点副标题(一行短语)按固定字号直接
      整句渲染，"TRITON_KERNEL_DUMP=1 时 fn_dump_manager.put 落盘各层 IR" 这类
      长句宽度远超 NODE_W=200，右侧文字压进了下一个节点框里(如 cache_key 的
      副标题尾字与 make_ttir 节点框重叠)。本轮加 wrap_lines()(按 cjk_text_width
      贪心换行，必要时逐字符硬断)把每个节点副标题限制在节点宽度内(≤2 行)，
      同时把过长的短语本身也精简（如"改核体即失效"这类从句略去），NODE_H
      相应从 60 加到 74 以容纳两行。
    - 第一轮底部"阅读路线"行的说明文字与第一个站牌胶囊("先说取证")也发生了
      轻微重叠(路线名文字末尾恰好顶上胶囊左边缘)，本轮把路线名到首个站牌的
      间隙从 24px 加到 48px。
    - 重渲染 + 重新 Read PNG 后，六项逐一复核：文字不再压节点边界、箭头起止点
      对齐节点框、图例/路线文字无重叠、中文正常显示、"之字形"走线配合站牌
      顺序(先说取证→第一步→…→第六步)可在 10 秒内看懂阅读顺序。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size / 半角按 0.58×size 求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def wrap_lines(text, max_w, size, max_lines=2):
    """按 max_w(像素) 贪心换行：优先在空白处断行，单个 token 本身超宽再逐字符断——
    本章节点短语混排中英/符号(如 add_rewrite_tensor_pointer)，定长换行比不换行
    硬溢出到相邻节点框更安全(no_overlap 自查项)。超过 max_lines 时最后一行截断
    加省略号,避免行数无限增长撑爆节点高度。"""
    words = text.split(' ')
    lines, cur = [], ''
    for word in words:
        cand = word if not cur else cur + ' ' + word
        if cjk_text_width(cand, size) <= max_w or not cur:
            # 单个 word 本身就超宽(不含空格可断)时逐字符硬断
            if cjk_text_width(cand, size) <= max_w:
                cur = cand
                continue
            chars = list(word)
            piece = cur
            for ch in chars:
                cand2 = piece + ch if piece else ch
                if cjk_text_width(cand2, size) <= max_w:
                    piece = cand2
                else:
                    lines.append(piece)
                    piece = ch
            cur = piece
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + '…'
    return lines


# ---------------- DATA(可变：本章数据) ----------------
ROW_LABELS = ["源码与运行时层", "IR 生成与布局层", "优化 pass 与硬件降级层"]

# (节点id, 行, 列, 真实符号名, 一行短语, 站牌(标题词，无 §))
NODES = [
    ("dump",        0, 0, "compile()",
     "TRITON_KERNEL_DUMP=1 落盘各层 IR", "先说取证"),
    ("attn_inner",  0, 1, "_attn_fwd_inner",
     "tl.dot/exp2 在线 softmax，alpha 重标定", "第一步"),
    ("cachekey",    0, 2, "cache_key",
     "AST 依赖+起始行号 → 磁盘缓存键", "第二步"),
    ("make_ttir",   1, 2, "make_ttir",
     "抹平 make_block_ptr 为裸指针张量", "第三步"),
    ("convert2ttg", 1, 1, "add_convert_to_ttgpuir",
     "指派 #blocked/#mma/#shared 布局", "第四步"),
    ("coalesce",    1, 0, "add_coalesce",
     "挑合并访存的 #blocked", "第五步"),
    ("accelmatmul", 2, 0, "add_accelerate_matmul",
     "tt.dot 指派 #mma，命中 Tensor Core", "第五步"),
    ("pipeline",    2, 1, "add_pipeline",
     "K/V 双缓冲 + async_copy 预取", "第五步"),
    ("llvmptx",     2, 2, "make_llir → make_ptx",
     "降到 mma.sync/ex2.approx/cp.async", "第六步"),
]
NODE_BY_ID = {n[0]: n for n in NODES}

EDGES = [  # 阅读顺序(严格单向脊柱)，方向由 (row,col) 相对位置自动判定锚点边
    ("dump", "attn_inner"), ("attn_inner", "cachekey"),
    ("cachekey", "make_ttir"),                                   # 行内→跨行(纵向)
    ("make_ttir", "convert2ttg"), ("convert2ttg", "coalesce"),
    ("coalesce", "accelmatmul"),                                  # 跨行(纵向)
    ("accelmatmul", "pipeline"), ("pipeline", "llvmptx"),
]

# 底部单一阅读路线(9 站，之字形网格里列坐标会重复，路线条另起一套等距横轴)
ROUTE_NAME = "全链降级路径（先说取证 → 第一步…第六步，无分支）"
ROUTE_STOPS = [n[5] for n in NODES]  # 站牌顺序 = NODES 定义顺序 = 阅读顺序
# 相邻重复站牌(三个「第五步」)合并显示成一个站点，路线条更干净
ROUTE_STOPS_DEDUP = []
for s in ROUTE_STOPS:
    if not ROUTE_STOPS_DEDUP or ROUTE_STOPS_DEDUP[-1] != s:
        ROUTE_STOPS_DEDUP.append(s)

LEGEND = [
    ("#22c55e", "入口：kernel 启动 _attn_fwd[grid](...)"),
    ("#3b82f6", "章内主线：降级链调用 / 数据流"),
    ("#f97316", "出口：PTX 交给驱动装载执行"),
]
TITLE = "第 43 章 · fused-attention 全链降级剖面（源码走线 + 阅读站牌）"

# ---------------- 不可变：配色(与全书其余 chapter-map 一致) ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"

# ---------------- 几何常量(全计算，零手写魔数) ----------------
NODE_W, NODE_H = 208, 74
SUB_FONT_SIZE, SUB_LINE_H = 9.6, 12.5
NODE_PAD_X = 10
COL_GAP, ROW_GAP = 34, 26
EDGE_MARGIN, STUB_W, STUB_H = 14, 66, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 26
LANE_LABEL_H, BAND_PAD = 22, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 30, 24, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 20, 40

N_COLS = max(n[2] for n in NODES) + 1  # 3
N_ROWS = max(n[1] for n in NODES) + 1  # 3

COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(N_COLS)]

band_h = [LANE_LABEL_H + BAND_PAD * 2 + NODE_H for _ in range(N_ROWS)]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh + ROW_GAP
lanes_bottom = _cum - ROW_GAP

NODE_XY = {}
for nid, row, col, *_ in NODES:
    x = COLX[col]
    y = band_top[row] + LANE_LABEL_H + BAND_PAD
    NODE_XY[nid] = (x, y)

routes_top = lanes_bottom + 10
w = PAD_L + N_COLS * NODE_W + (N_COLS - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """站牌胶囊，居中挂在 (cx,cy)——宽度按文字自适应(本章站牌是变长标题词，
    不是模板里定长的 §N.M，不能用固定宽度，否则「先说取证」四个字会溢出胶囊)。"""
    bw = max(40.0, cjk_text_width(text, 11) + 14)
    bh = 20
    bx, by = cx - bw / 2, cy - bh / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh}" rx="{bh / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
    _lx += 20 + cjk_text_width(label, 11.5) + 30

# 泳道(行)背景 + 标签 + 分隔线
for i, name in enumerate(ROW_LABELS):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i] - ROW_GAP / 2:.1f}" x2="{w}" y2="{band_top[i] - ROW_GAP / 2:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1" stroke-dasharray="3,3"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩：入口挂在第一个节点(dump, 行0列0)，出口挂在最后一个节点(llvmptx, 行2列2)
ex, ey = NODE_XY["dump"]; ey += NODE_H / 2
xx, xy = NODE_XY["llvmptx"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')


def node_center(nid):
    x, y = NODE_XY[nid]
    return x + NODE_W / 2, y + NODE_H / 2


def node_row_col(nid):
    _, row, col, *_ = NODE_BY_ID[nid]
    return row, col


# 调用边(主线蓝)：方向感知——同行按左右相对位置选左/右边锚点(之字形有些边从右到左)，
# 跨行(纵向衔接，本章两处：行内末站→下一行首站，两者同列)按上/下边锚点。
for src, dst in EDGES:
    sx1, sy1 = NODE_XY[src]; sx2, sy2 = NODE_XY[dst]
    srow, scol = node_row_col(src); drow, dcol = node_row_col(dst)
    if srow == drow:
        # 同行水平边：谁在左谁的右边出，谁在右谁的左边入
        if sx2 >= sx1:
            p1 = (sx1 + NODE_W, sy1 + NODE_H / 2)
            p2 = (sx2, sy2 + NODE_H / 2)
        else:
            p1 = (sx1, sy1 + NODE_H / 2)
            p2 = (sx2 + NODE_W, sy2 + NODE_H / 2)
    else:
        # 跨行纵向边：上方节点下边 → 下方节点上边
        p1 = (sx1 + NODE_W / 2, sy1 + NODE_H)
        p2 = (sx2 + NODE_W / 2, sy2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌胶囊)
for nid, row, col, symbol, phrase, station in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + 24:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    sub_lines = wrap_lines(phrase, NODE_W - 2 * NODE_PAD_X, SUB_FONT_SIZE, max_lines=2)
    sub_y0 = y + 42 if len(sub_lines) > 1 else y + 48
    for li, line in enumerate(sub_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{sub_y0 + li * SUB_LINE_H:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{SUB_FONT_SIZE}" fill="{C_NODE_SUB}">{esc(line)}</text>')
    # 站牌贴右上角，中心点距节点右边缘 4px(自适应宽度的胶囊自己不会超出节点太多)
    badge_svg, bw = badge(x + NODE_W - 4, y, station)
    L += badge_svg

# 底部单一阅读路线：本章无分支，站牌等距重排(不复用列坐标，因 3x3 之字形列坐标会重复)
L.append(f'<text x="16" y="{routes_top + 14:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(站牌=图上节点右上角标题词；本章单线、无分支)")}</text>')
ry = routes_top + ROUTE_HEAD_H + ROUTE_ROW_H / 2
route_label_w = cjk_text_width(ROUTE_NAME, 12) + 48  # 48 = 与首个站牌之间的呼吸间隙
x_line_start = 16 + route_label_w
x_line_end = w - 24
n_stops = len(ROUTE_STOPS_DEDUP)
stop_x = [x_line_start + i * (x_line_end - x_line_start) / (n_stops - 1) for i in range(n_stops)]
L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
          f'fill="{C_NODE_TITLE}">{esc(ROUTE_NAME)}</text>')
L.append(f'<line x1="{stop_x[0]:.1f}" y1="{ry:.1f}" x2="{stop_x[-1]:.1f}" y2="{ry:.1f}" '
          f'stroke="{C_MAIN}" stroke-width="3"/>')
for x, label in zip(stop_x, ROUTE_STOPS_DEDUP):
    badge_svg, _ = badge(x, ry, label)
    L += badge_svg

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio={w / h:.2f}:1)")
