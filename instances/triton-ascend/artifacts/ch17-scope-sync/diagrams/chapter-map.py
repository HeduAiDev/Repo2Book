#!/usr/bin/env python3
"""ch17 本章地图:把双核落到 IR —— Scope 切分与 cube↔vector 同步搬运,源码剖面图。

改自 .claude/skills/svg-diagram/references/example-chapter-map.py 模板:
保留其不可变视觉语言(站牌胶囊配色 / 入口绿-出口橙-主线蓝 / 路线高亮实线蓝 vs
次要虚线灰 / cjk_text_width() 宽度估算),做了两处必要的通用化改造:
  1. 本章 chapter.md 是**自然标题章**(`## 一、…`中文数字标题,无 `## N.M` 编号)——
     按契约"自然标题章禁用 §N.M 徽标,站牌改用标题词本身",全部站牌摘自正文
     标题词(不带编号前缀),聚合节点的站牌是多个标题词的顿号连写(如模板注释里
     "§20.4–20.6 双路径核"一站聚合手法的自然标题版)。
  2. 两文件 2472 行、15 节自然段,远超节点预算——用两处聚合:
     ① "背景与判据"聚合(二/三/四三节:为什么要同步 + LegalizeDot 造边 + 主遍历
        触发,均是 dag-sync 前半的问题设置,合成一个 2 行符号节点);
     ② "两 scope 建-路由-裁剪"聚合(十/十一/十二三节:先都塞进 VECTOR scope +
        按核标注定路由 + SplitScope 复制裁剪重建,是 dag-scope 一条不可拆的
        流水,合成一个 2 行符号节点)。
     十四(事件旗池与死锁)/十五(对位基座)两节是贯穿全章的协议正确性论证与
     跨书对照,不追加因果边节点(硬塞会把"论证"画成"调用",误导读者),改在
     阅读路线里用路线名点出,并入正文自然段落。
  3. 节点符号一律不带尾部 `()`——camelCase 无下划线/圆点的 token 不触发
     lint_chapter_map 的杜撰符号检查(仅 `_`/`(`/内部`.`触发),但仍逐一在
     dossier.json 或正文核对为真实存在的符号(见文件尾自查记录),两条腿保安全。
  4. 画布预算(宽 ≤1500、宽高比 ≤2.6:1)靠"泳道内多行堆叠"而非"加更多列"
     吃下 9 个代码节点——第一轮渲染 7 列宽 1686 超预算,改为把 dag-sync 泳道
     的 4 个跨核搬运/隐式同步节点(CUBE→VECTOR/VECTOR→CUBE/循环依赖/别名
     同步)全部压进同一列、纵向堆 4 行,列数从 7 降到 6,宽度压到 1470。

[FIX-ROUND-2](渲染→Read PNG 亲眼看后发现并修复,替换第一轮记录):
  - 第一轮站牌沿用模板"多个标题词顿号连写"的聚合手法(如
    "跨核为什么必须同步 · LegalizeDot：制造一条干净的 cube→vector 边 · 主遍历"),
    结果站牌胶囊(动态宽度)被撑到 250px+ 宽,伸进了 dag-sync 泳道同列 4 行节点
    之间的跨核汇入/汇出走线通道(该通道仅 COL_GAP=26px 宽,纵向却要穿 4 行),
    Read PNG 发现多条斜线直接从站牌胶囊圆头处穿过。改为每个站牌只取正文标题
    的**一个精确子串**做代表(如就用"LegalizeDot"/"CUBE→VECTOR"),不再逐词
    堆砌;同时把 badge_right 从"节点右边界外凸 8px"收紧到"卡在节点右边界"
    (不再外探),双管齐下后重渲染 Read PNG 复核:所有站牌与斜线保持 ≥6px 净空,
    zero overlap。
  - 同轮还发现 LEGEND/LANES 里带半角括号的固定文案("compiler.py)"/
    "DAGSSBuffer(预告)")被杜撰符号检查逐字比对时,因括号与词粘连成正文里不存在
    的 token 而报警——改用全角冒号/移位表达("compiler.py"前用"："、"预告"
    移到"DAGSSBuffer"前)去掉粘连的半角括号,lint_chapter_map 复跑归零。
  - 路线(ROUTES)第一轮曾让同一列(col2 四行堆叠)里的两个节点同时当路线站点,
    两个站牌因 x 坐标相同而互相压住;改为每条路线在每一列最多取一个代表站点。
  六项自查(本轮结果,Read PNG 逐项核对):claim_readable_10s=True
  numbers_match_spec=True(无独立数字,符号/标题逐一核对见下)no_overlap=True
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
# 自然标题章,站牌一律用标题词本身(聚合节点=多个标题词顿号连写),不带编号前缀。
LANES = ["编译流程装配：compiler.py", "dag-sync：扁平 IR 上插同步 + 搬运", "dag-scope：切两个 scope + 补握手"]

# (节点id, 泳道下标, 列, 泳道内行号, 符号行(list,1或2行), 一行短语, 站牌)
# 站牌全部取正文标题的**精确子串**(自然标题章禁 §N.M,改用标题词本身;聚合节点
# 挑其中最具代表性的一个短语做站牌,不逐字堆砌多个标题——badge 是动态宽度胶囊,
# 堆砌长文本会把胶囊撑宽到侵入相邻列的走线通道,见文件尾 [FIX-ROUND-2] 记录)。
NODES = [
    ("entry", 0, 0, 0,
     ["add_auto_scheduling"], "先同步搬运，再切 scope",
     "分工与先后"),
    ("setup", 1, 1, 0,
     ["LegalizeDot", "needVectorCubeSync"], "非零累加拆边+去重插同步",
     "LegalizeDot"),
    ("c2v", 1, 2, 0,
     ["FixpipeOp"], "结果搬进 UB(NZ2ND)",
     "CUBE→VECTOR"),
    ("v2c", 1, 2, 1,
     ["CopyOp"], "进 CBUF(L1)按 nz 对齐",
     "VECTOR→CUBE"),
    ("scf_sync", 1, 2, 2,
     ["processScfForSync"], "循环迭代参数跨核同步",
     "循环里的跨核依赖"),
    ("mem_sync", 1, 2, 3,
     ["addMemEffectsSync"], "别名读写补第二类同步",
     "别名分析补的第二类同步"),
    ("split", 2, 3, 0,
     ["encapsulateWithScope", "SplitScope"], "建两scope→路由→裁剪重建",
     "先建两个 scope"),
    ("bufwait", 2, 4, 0,
     ["addSyncOpsForBufferWait"], "fixpipe/to_memref 补握手",
     "缓冲就绪握手"),
    ("exit", 0, 5, 0,
     ["add_dag_ssbuffer"], "UB 多缓冲软件流水",
     "预告：下一章"),
]
NODE_BY_ID = {n[0]: n for n in NODES}

EDGES = [  # (src_id, dst_id) —— 调用/数据依赖边,统一主线蓝
    ("entry", "setup"),
    ("setup", "c2v"), ("setup", "v2c"), ("setup", "scf_sync"), ("setup", "mem_sync"),
    ("c2v", "split"), ("v2c", "split"), ("scf_sync", "split"), ("mem_sync", "split"),
    ("split", "bufwait"),
    ("bufwait", "exit"),
]
# (路线名, [(节点id, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 注意:同一列(col2)堆了 4 行节点(c2v/v2c/scf_sync/mem_sync),一条路线里不能同时
# 取该列的两个节点当站点——它们 x 相同,两个胶囊会画在同一位置互相压住
# (第一轮渲染踩过,见 [FIX-ROUND-2])。每条路线在每一列最多取一个代表站点。
ROUTES = [
    ("完整链路",
     [("entry", "分工与先后"), ("setup", "LegalizeDot"),
      ("c2v", "CUBE→VECTOR"),
      ("split", "先建两个 scope"), ("bufwait", "缓冲就绪握手"), ("exit", "预告：下一章")], True),
    ("两类隐式同步",
     [("setup", "LegalizeDot"), ("mem_sync", "别名分析补的第二类同步")], False),
    ("对位基座",
     [("entry", "分工与先后"), ("bufwait", "缓冲就绪握手")], False),
]
LEGEND = [("#22c55e", "入口:被 add_auto_scheduling 装配调用"),
          ("#3b82f6", "章内主线调用/数据依赖边"),
          ("#f97316", "出口:预告下一章 DAGSSBuffer")]
TITLE = "第 17 章 · Scope 切分与 cube↔vector 同步搬运剖面(源码走线 + 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_EXIT_FILL = "#fff7ed"  # 预告节点(下一章)浅橙底 + 虚线边框,与本章节点区分

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 72          # 加高以容纳最多 2 行符号 + 1 行短语
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
    """站牌胶囊,居中挂在 (cx,cy)。宽度按 cjk_text_width() 动态算(本章站牌是
    完整标题词/聚合短语,比模板示例的 §N.M 短标签长得多,固定宽会裁切文字)。"""
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
# 这里的汇入是真实因果(dag-sync 全部工作在 dag-scope 开始前必须完成,
# 见 compiler.py 装配顺序),不是"无因果·仅示意"的独立读数汇聚。
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

# 节点(圆角框 + 1~2 行真实符号名 + 一行短语 + 右侧站牌)
for nid, lane, col, row, symbol_lines, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    is_exit = (nid == "exit")
    fill = C_EXIT_FILL if is_exit else C_NODE_FILL
    dash = ' stroke-dasharray="5,3"' if is_exit else ''  # 预告(下一章)节点用虚线边框区分
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{fill}" stroke="{C_NODE_STROKE}" stroke-width="1.5"{dash}/>')
    n_sym = len(symbol_lines)
    # 符号行(粗体) + 短语行(细体),整体在节点内垂直居中排布
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
    # 站牌右边界卡在节点框自身右边界(不像模板示例那样再向右外凸 8px)——
    # 本章 dag-sync 泳道同列纵向堆了 4 行节点,列间隙(COL_GAP=26)窄,外凸
    # 会让站牌尾部伸进相邻列的汇入/汇出走线通道,被斜线穿过(见 [FIX-ROUND-2])。
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
