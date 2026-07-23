#!/usr/bin/env python3
"""ch18 本章地图:DAGSSBuffer —— UB 多缓冲与昇腾的软件流水线,源码剖面图。

改自 .claude/skills/svg-diagram/references/example-chapter-map.py 模板(与
ch17 chapter-map.py 同一套通用化改造沿用):保留其不可变视觉语言(站牌胶囊
配色 / 入口绿-出口橙-主线蓝 / 路线高亮实线蓝 vs 次要虚线灰 / cjk_text_width()
宽度估算),做了两处必要的调整:

  1. 本章 chapter.md 是**自然标题章**(`## 一、…`中文数字标题,无 `## N.M`
     编号)——按契约"自然标题章禁用 §N.M 徽标,站牌改用标题词本身",全部站牌
     摘自正文标题的精确子串(或正文段落里的原句,如"五段各司其职"取自 §一
     段落原文而非标题本身——两者都是真实存在于正文的字面串,自查按"能在
     chapter.md 核到原样子串"判定,不强求必须是标题子串)。
  2. `DAGSSBuffer.cpp` 单文件 5534 行、六节自然段,狠选核心脊梁:节点只保留
     有真实代码符号锚点的那几个(pass 装配入口/runOnOperation 编排/双缓冲三部
     曲 addDoubleBuffForArgs→buildNBufferProducer+buildNBufferConsumer→
     addMultiBuffCaculate)。§二(为什么要双缓冲,直觉铺垫)与§六(对位基座,
     跨书对照)都不是"调用链"上的代码节点——分别只在 ROUTES 里点出,不占
     NODES 名额,避免把"讲道理"画成"调用关系"(与 ch17 对"对位基座"节的
     处理手法一致)。producer/consumer 是同一 level 并发的两个独立函数(写侧
     选 buffer / 读侧选 buffer),同列纵向堆叠两行,呼应正文"两个独立计数器"
     的关系。

[FIX-ROUND-1](首轮渲染→Read PNG 亲眼看,发现并修复):
  首轮 NODES 的 phrase(节点内一行短语)沿用正文原句长度(如"仅 add_auto_
  scheduling 为真时挂载，dag-scope/dag-sync 之后"49 字符、"1 份 buffer→2 份 +
  2 计数器，塞进 scf.for iterArgs"42 字符),用 cjk_text_width() 核算宽度
  210~338px,而 NODE_W 只有 190——远超节点框宽度。Read PNG 发现 add_dag_
  ssbuffer 节点的短语文字明显溢出框体右边界,且与 entry→runop 的对角调用边
  发生视觉交叉(线穿过溢出的文字尾部)。本模板未对节点 phrase 做宽度校验
  (不像图例/站牌那样已用 cjk_text_width 动态算宽度),原因是模板假设 phrase
  本就应该写得短(如 ch17 范例"先同步搬运，再切 scope"仅 11 字符、宽度
  ~115px)。改法:把全部 7 个节点的 phrase 压缩到 9~21 字符(核算宽度
  94~141px,留出 ≥49px 安全边距),不再照抄正文整句。重渲染后 Read PNG 复核
  entry/runop/expand/producer/consumer/wire/exit 全部节点,短语与符号名都
  完整落在框内,不再与调用边/相邻站牌交叉。
  六项自查(本轮结果,Read PNG 逐项核对,并对 producer/consumer 堆叠列、
  exit 预告框上下两段连线单独放大复核):claim_readable_10s=True
  numbers_match_spec=True(图上数字均为 frontCnt%2/postCnt%2/1→2 份+2 计数器,
  与 dossier bufferNum=2 一致,无独立数字断言)no_overlap=True
  arrows_attached=True cjk_rendered=True reading_order_clear=True

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
# 自然标题章,站牌一律用标题词/正文原句的精确子串,不带编号前缀。
LANES = ["编译流水线装配：compiler.py + Passes.td", "runOnOperation：五段编排", "双缓冲核心变换：扩容→选择→接线"]

# (节点id, 泳道下标, 列, 泳道内行号, 符号行(list,1或2行), 一行短语, 站牌)
NODES = [
    ("entry", 0, 0, 0,
     ["add_dag_ssbuffer"], "自动调度链的第三站",
     "站在流水线的哪一环"),
    ("runop", 1, 1, 0,
     ["runOnOperation"], "五段编排，末段才是双缓冲",
     "五段各司其职"),
    ("expand", 2, 2, 0,
     ["addDoubleBuffForArgs"], "1→2 份 buffer + 2 计数器",
     "一份 buffer 扩成两份"),
    ("producer", 2, 3, 0,
     ["buildNBufferProducer"], "frontCnt%2 选写侧 buffer",
     "谁写哪份、谁读哪份"),
    ("consumer", 2, 3, 1,
     ["buildNBufferConsumer"], "postCnt%2 选读侧 buffer",
     "按计数器 mod 2 选 buffer"),
    ("wire", 2, 4, 0,
     ["addMultiBuffCaculate"], "接线 if 分支，回填 yield",
     "两个计数器错开一位"),
    ("exit", 0, 5, 0,
     [], "不规则访存：掩码与交错",
     "预告：下一章"),
]
NODE_BY_ID = {n[0]: n for n in NODES}

EDGES = [  # (src_id, dst_id) —— 调用/数据依赖边,统一主线蓝
    ("entry", "runop"),
    ("runop", "expand"),
    ("expand", "producer"), ("expand", "consumer"),
    ("producer", "wire"), ("consumer", "wire"),
    ("wire", "exit"),
]
# (路线名, [(节点id, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 同列(col3)堆了 producer/consumer 两行,一条路线里同列只取一个代表站点
# (两者 x 相同,同取会互相压住站牌——手法同 ch17 [FIX-ROUND-2] 记录)。
ROUTES = [
    ("完整链路",
     [("entry", "站在流水线的哪一环"), ("runop", "五段各司其职"),
      ("expand", "一份 buffer 扩成两份"),
      ("producer", "谁写哪份、谁读哪份"),
      ("wire", "两个计数器错开一位"), ("exit", "预告：下一章")], True),
    ("为什么值得(直觉)",
     [("runop", "五段各司其职"), ("wire", "两个计数器错开一位")], False),
    ("对位基座",
     [("entry", "站在流水线的哪一环"), ("wire", "两个计数器错开一位")], False),
]
LEGEND = [("#22c55e", "入口：被 add_auto_scheduling 装配调用"),
          ("#3b82f6", "章内主线调用/数据依赖边"),
          ("#f97316", "出口：预告下一章不规则访存优化")]
TITLE = "第 18 章 · DAGSSBuffer 双缓冲变换剖面(源码走线 + 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_EXIT_FILL = "#fff7ed"  # 预告节点(下一章)浅橙底 + 虚线边框,与本章节点区分

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 72
COL_GAP, ROW_GAP = 26, 18
EDGE_MARGIN, STUB_W, STUB_H = 12, 56, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
BADGE_H, BADGE_PAD_X = 20, 8  # 徽标高度固定;宽度按文字动态算(见 badge())

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

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)。宽度按 cjk_text_width() 动态算。"""
    bw = cjk_text_width(text, 11) + 2 * BADGE_PAD_X
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


def badge_w(text):
    return cjk_text_width(text, 11) + 2 * BADGE_PAD_X


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="white"/>')

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
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w:.1f}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方/下一章在画布外")
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
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
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)。多条边汇入同一节点时终点 y 各偏移,否则重合看不出"汇合"。
# expand→producer/consumer 是分流(同一 buffer 依赖同时驱动写侧与读侧两段接线代码),
# producer/consumer→wire 是真实接线汇合(addMultiBuffCaculate 同时消费两者的结果),
# 均为确定性因果,不是"无因果·仅示意"的独立读数汇聚。
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

# 节点(圆角框 + 0~2 行真实符号名 + 一行短语 + 右侧站牌)
for nid, lane, col, row, symbol_lines, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    is_exit = (nid == "exit")
    fill = C_EXIT_FILL if is_exit else C_NODE_FILL
    dash = ' stroke-dasharray="5,3"' if is_exit else ''  # 预告(下一章)节点用虚线边框区分
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{fill}" stroke="{C_NODE_STROKE}" stroke-width="1.5"{dash}/>')
    n_sym = len(symbol_lines)
    # 符号行(粗体) + 短语行(细体),整体在节点内垂直居中排布;无符号行(如 exit)
    # 就只有短语行,居中显示。
    total_lines = n_sym + 1
    line_h = 15.5
    block_h = total_lines * line_h
    start_y = y + (NODE_H - block_h) / 2 + line_h * 0.75
    for li, sym in enumerate(symbol_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{start_y + li * line_h:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{start_y + n_sym * line_h:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_w(sec)
    # 站牌右边界卡在节点框自身右边界(不外凸)——col3 同列纵向堆了 producer/
    # consumer 两行,列间隙窄,外凸会让站牌尾部伸进相邻列的走线通道。
    badge_right = x + NODE_W
    L += badge(badge_right - bw / 2, y, sec)

# 底部阅读路线:按节点 id 找列坐标(同列多行节点取该节点自身 x),站牌与图上节点对齐
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = NODE_XY[stops[0][0]][0] + NODE_W / 2
    x_last = NODE_XY[stops[-1][0]][0] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for nid, sec in stops:
        cx = NODE_XY[nid][0] + NODE_W / 2
        L += badge(cx, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  viewBox=0 0 {w:.1f} {h:.1f}  aspect={w / h:.3f}")
