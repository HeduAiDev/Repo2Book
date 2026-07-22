#!/usr/bin/env python3
"""ch11「本章地图」——PtrAnalysis 逆向剖面：visitOperand 递归分派器如何把散落在
make_range/splat/broadcast/mul/sub/add 里的地址算术，逐维归并成一份
(stride,shape,dimIndex) + offset + source 的 PtrState（kind=deep，skip_impl=true；
本章为纯 C++ MLIR pass，无精简版，图上只呈现真实源码符号）。

本章是**编号标题章**（`## 11.1`…`## 11.10` + `## 小结`），按契约用 §N.M 徽标，
本图只挑与「递归分派 + 归并代数」这条核心主线直接相关的 8 个节：
  §11.2  递归分派器：visitOperand
  §11.3  叶子：make_range 的精确步幅公式
  §11.4  三个纯形状算子：splat / broadcast / expand_dims
  §11.5  张量 × 标量：mulState 与 subState
  §11.6  核心归并：addState 的完整代数
  §11.7  根节点：visitOperandAddptr
  §11.9  保守失败：认不出就体面退场（createNewPtr 的调用入口与失败落点都在这节）
  §11.10 规范化与进阶：normalizeState 与 rem/div
（§11.1 PtrState/StateInfo 已有专属 fig-m1-ptrstate-anatomy 图，§11.8 完整链走
读路线呼应、不建独立节点，rem/div 进阶细节点到为止折进 normalizeState 节的说明。）

剖面组织 = 真实源码目录（3 条泳道，上→下，均在 third_party/ascend）：
  Lane0 lib/TritonToStructured/MemOpConverter.cpp   —— 调用边界：createNewPtr(入口/出口两处)
  Lane1 include+lib/TritonToStructured/PtrAnalysis.{h,cpp} —— 递归分派 visitOperand
        与四类改写规则(make_range 叶子 / 三个形状算子 / mulState-subState / 根节点 addptr)
        同列不同行并列，都是 visitOperand 按 defining-op 分派出去的分支
  Lane2 lib/TritonToStructured/PtrAnalysis.cpp       —— 归并代数 addState 与收尾 normalizeState

主线（实线蓝，读者默认顺读路径）：
  entry(createNewPtr) → dispatch(visitOperand) → {四类改写规则} → addptr_root → addState
  → normalizeState → exit
四类改写规则都是 dispatch 按 defining-op 分派出去的分支（fan-out，同一个 visitOperand
调用点）。图上只有 addptr_root→addState 画了一条继续向下的边——源码里
visitOperandAddptr 确实直接调用 state.addState(...)，是唯一的单跳真调用；
make_range/形状算子/mulState-subState 三支的子状态要再经上一层递归里的另一次
arith.addi 才被归并，画一条"直连 addState"的边会暗示不存在的单跳调用，且四条同列边
收敛到同一目标时线会贴着中间几个节点的右边框穿过（首轮自查 no_overlap 命中过，
已改画法：只留 addptr_root 这一条真实调用边，其余三支在claim文字里点明"子状态经
上层递归归并"，不画容易误导又会压框的连线）。

跨章标注（exp-2026-07-18-04 硬规则：目标章号 > 本章号用「预告」，< 本章号用「回指」）：
  入口桩（绿）："回指 ch10" —— ch10 < ch11，上一章《分水岭总览》已点名 PtrAnalysis 的存在。
  出口桩（橙）："预告 ch12" —— ch12 > ch11，正文小结明确预告下一章把终态铸成
                memref.reinterpret_cast。

底部阅读路线复刻正文 hook 段的原话（"想先建立地基，按序读 §11.1→§11.2；只关心
「一条链怎么走完」，可直接跳 §11.8 看完整推演，再回头补代数细节"）+ 三条按单条改写
规则切入归并代数的跳读路线。

不可变视觉语言（全书统一，来自 example-chapter-map.py 模板 + ch10 的动态胶囊/自适应
换行/编号圆圈/monospace 路径行改法）：§徽标胶囊（圆角矩形 fill #eef2ff /
stroke #6366f1）、入口绿 #22c55e / 出口橙 #f97316 / 主线蓝 #3b82f6、cjk_text_width()
逐字符宽度估算。本章无旁支（EDGES_SIDE 为空），故图例只 3 行（无灰虚线一项）。

六项自查（渲染→Read PNG 亲眼看后如实记录）：见同目录 figure-manifest.json 该图 selfcheck。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def mono_text_width(s, size):
    """monospace 路径行宽度估算：等宽字体每字符约 0.6×size(半角)。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.6) for ch in s)


_BREAK_AFTER = set("，；：、/ ,;")


def wrap_claim(text, max_w, size):
    """一句论点太长时换行——只在标点/斜杠/空格之后断行，不劈开一个标识符或中文词。
    贪心找"prefix 仍不超宽的最靠后一个合法断点"；找不到合法断点才整句照旧单行放行。"""
    breaks = [i for i, ch in enumerate(text) if ch in _BREAK_AFTER]
    best = None
    for i in breaks:
        if cjk_text_width(text[:i + 1], size) <= max_w:
            best = i
    if best is None:
        return [text]
    line1, line2 = text[:best + 1].rstrip(), text[best + 1:].lstrip()
    if cjk_text_width(line2, size) <= max_w:
        return [line1, line2]
    return [line1] + wrap_claim(line2, max_w, size)


# ---------------- DATA(可变：本章数据) ----------------
LANES = [
    "third_party/ascend/lib/TritonToStructured · 调用边界 · MemOpConverter.cpp",
    "third_party/ascend/{include,lib}/TritonToStructured · 递归分派与改写规则 · PtrAnalysis.cpp",
    "third_party/ascend/lib/TritonToStructured · 归并代数与规范化 · PtrAnalysis.cpp",
]

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 省略前缀后的路径, 一句论点, §站牌)
NODES = [
    ("entry", 0, 0, 0,
     ["createNewPtr"],
     "…/MemOpConverter.cpp",
     "每个 load/store 的指针操作数，起一次 PtrAnalysis 实例来解析",
     "§11.9"),
    ("dispatch", 1, 1, 0,
     ["visitOperand"],
     "…/PtrAnalysis.cpp",
     "3 个前置判定(缓存/标量/指针)之后，按 defining-op 类型分派 14 个分支",
     "§11.2"),
    ("leaf_range", 1, 2, 0,
     ["visitOperandMakeRange"],
     "…/PtrAnalysis.cpp",
     "叶子：stride=(end-start+n-1)/n，结果不等于 1 即保守失败；子状态留给上层归并",
     "§11.3"),
    ("shape_ops", 1, 2, 1,
     ["splat / broadcast", "/ expand_dims"],
     "…/PtrAnalysis.cpp",
     "只改 stateInfo 的形状与维数，不碰 offset/source；子状态留给上层归并",
     "§11.4"),
    ("mul_sub", 1, 2, 2,
     ["mulState / subState"],
     "…/PtrAnalysis.cpp",
     "张量×标量；mul 靠 swap 归到 rhs，sub 只许 rhs 标量；子状态留给上层归并",
     "§11.5"),
    ("addptr_root", 1, 2, 3,
     ["visitOperandAddptr"],
     "…/PtrAnalysis.cpp",
     "拆 ptr/offset 两支子状态，校验 source 后交给 addState 合并",
     "§11.7"),
    ("add_state", 2, 3, 0,
     ["addState"],
     "…/PtrAnalysis.cpp",
     "按 dimIndex 双指针归并，shape 不为倍数则 failure，是倍数则拆维",
     "§11.6"),
    ("normalize", 2, 4, 0,
     ["normalizeState"],
     "…/PtrAnalysis.cpp",
     "合并相邻的零 stride 维，剔除多余的单元维",
     "§11.10"),
    ("exit", 0, 5, 0,
     ["shouldLinearize", "/ oldPtr"],
     "…/MemOpConverter.cpp",
     "失败→shouldLinearize=false，原样退回；成功→终态交下一章落 memref",
     "§11.9"),
]
NODE_ORDER = [n[0] for n in NODES]  # 阅读序①…⑨ = 本列表出现顺序
NODE_BY_ID = {n[0]: n for n in NODES}
ENTRY_NODE, EXIT_NODE = "entry", "exit"  # 主线的真实起止

EDGES_MAIN = [  # 主线，实线蓝——确定性前向数据流(章内自身语言，非字面"调用边")
    ("entry", "dispatch"),
    ("dispatch", "leaf_range"), ("dispatch", "shape_ops"),
    ("dispatch", "mul_sub"), ("dispatch", "addptr_root"),
    # 只画 addptr_root→add_state 这一条(源码里 visitOperandAddptr 确实直接调用
    # state.addState(...)，是唯一的单跳真调用)；leaf_range/shape_ops/mul_sub 不会画
    # 直接连去 add_state 的边——它们各自的子状态要经过更上层递归里的另一次 addi/
    # addState 才归并，画一条"直连"会在视觉上暗示单跳调用而失真，且四条同列边收敛
    # 到同一目标时线会贴着中间几个节点的右边框穿过(自查 no_overlap 命中过一次，
    # 已改用这条更准确也更干净的画法)。
    ("addptr_root", "add_state"),
    ("add_state", "normalize"),
    ("normalize", "exit"),
]
EDGES_SIDE = []  # 本章无旁支

# 路由用的可路由站牌(不含 entry/exit 的重复 §11.9——避免同一 § 出现两个物理位置时
# 路线索引二义)。顺序 = 底部路线条的左→右物理槽位序，按"四类改写规则先并列走完，
# 再汇入归并"的图面叙事排(§11.7 根节点排在 §11.6 归并之前，因为 addptr 调用 addState，
# 是数据真正流向 addState 的最后一支)——不是正文的顺读序号序，各路线的 stops 子集
# 均须在此序列里保持下标单调递增，否则路线胶囊会画出连线范围之外(见本文件模板注释)。
STATION_ORDER = ["§11.2", "§11.3", "§11.4", "§11.5", "§11.7", "§11.6", "§11.10"]
ROUTES = [  # (路线名, [§站牌…]按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
    ("全览：分派→四类改写规则→归并→收尾", STATION_ORDER, True),
    ("跳读：直奔递归的根(呼应 §11.8 完整推演)", ["§11.2", "§11.7", "§11.6"], False),
    ("只看 make_range 怎么并入归并", ["§11.2", "§11.3", "§11.6"], False),
    ("只看标量代数的不对称", ["§11.2", "§11.5", "§11.6"], False),
]
LEGEND = [
    ("#22c55e", "入口(回指 ch10)：分水岭总览已点名 PtrAnalysis，本章逐行拆开它"),
    ("#3b82f6", "主线：visitOperand 沿 SSA 定义链的确定性前向数据流"),
    ("#f97316", "出口(预告 ch12)：终态的 strides/sizes/offset 交下一章铸成 memref.reinterpret_cast"),
]
TITLE = "第 11 章 · PtrAnalysis 逆向剖面：visitOperand 递归分派 + addState 归并代数"
SUBNOTE = "节点路径省略公共前缀 third_party/ascend/(以 … 代替)；完整路径见正文行内夹注"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 205
COL_GAP, ROW_GAP = 20, 22
EDGE_MARGIN, STUB_W, STUB_H = 10, 50, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 14
LANE_LABEL_H, BAND_PAD = 22, 13
TOP_PAD, TITLE_H, SUBNOTE_H, LEGEND_H, BOTTOM_PAD = 12, 26, 16, 3 * 14.5 + 12, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_FONT, BADGE_PAD_X = 20, 11, 8
ROUTE_BADGE_FONT, ROUTE_BADGE_PAD_X = 10.5, 9
CLAIM_FONT = 9.0
SYM_FONT, SYM_LINE_H = 10.6, 13
ORD_R = 9

# 每个节点的论点先按 NODE_W 预算换行一遍，取全章最多的行数统一定 NODE_H；
# 符号行数同理取全章最大值——同一行号跨泳道对齐用的是同一个 NODE_H。
CLAIM_MAXW = NODE_W - 14
_CLAIM_LINES = {n[0]: wrap_claim(n[6], CLAIM_MAXW, CLAIM_FONT) for n in NODES}
_max_claim_lines = max(len(v) for v in _CLAIM_LINES.values())
_max_sym_lines = max(len(n[4]) for n in NODES)
SYM_TOP = 34
PATH_Y = SYM_TOP + _max_sym_lines * SYM_LINE_H  # 路径行基线
CLAIM_TOP = PATH_Y + 14                         # 首行论点基线
NODE_H = CLAIM_TOP + (_max_claim_lines - 1) * 11.5 + 10

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + SUBNOTE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
NODE_COL = {n[0]: n[2] for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD
assert w <= 1500 and w / h <= 2.6, f"画布预算超标：{w}x{h}, {w / h:.2f}:1"


def badge(cx, cy, text, font=BADGE_FONT, pad_x=BADGE_PAD_X):
    """§ 徽标胶囊，居中挂在 (cx,cy)，宽度按 cjk_text_width 动态算。"""
    bw = cjk_text_width(text, font) + 2 * pad_x
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{font}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ], bw


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Side", C_ROUTE_DIM))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题 + 省略前缀的说明
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 17}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + TITLE_H + 11}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(SUBNOTE)}</text>')

# 图例(3 种语义色必须画图例)
for li, (color, label) in enumerate(LEGEND):
    _row_y = TOP_PAD + TITLE_H + SUBNOTE_H + 13 + li * 14.5
    L.append(f'<rect x="{PAD_L}" y="{_row_y - 10.5}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 18}" y="{_row_y}" font-family="sans-serif" font-size="10" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                 f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(跨章标注：目标章号 > 本章号用「预告」，< 本章号用「回指」)
ex, ey = NODE_XY[ENTRY_NODE]; ey += NODE_H / 2
xx, xy = NODE_XY[EXIT_NODE]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#166534">{esc("回指 ch10")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#9a3412">{esc("预告 ch12")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 主线(实线蓝)——多条边收敛到同一目标(add_state 收 4 支)时，终点 y 按偏移错开，
# 否则重合的终点在视觉上看不出"汇合"；这不是"无因果·仅示意"的并列，是 addState
# 真实合并这些子状态的数据流，故不加"无因果"注记。
_dst_total = {}
for _, dst in EDGES_MAIN:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES_MAIN:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_off = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (xd, yd + NODE_H / 2 + y_off)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
# 旁支(虚线灰，非因果顺序)——本章为空，占位保留结构一致性
for src, dst in EDGES_SIDE:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1, p2 = (xs_ + NODE_W / 2, ys_ + NODE_H), (xd + NODE_W / 2, yd)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4" '
             f'marker-end="url(#mSide)"/>')

# 节点(圆角框 + 序号圆圈 + 符号 + 路径 + 论点 + 右上角 § 徽标)
for oi, nid in enumerate(NODE_ORDER):
    nid_, lane, col, row, syms, path, claim, sec = NODE_BY_ID[nid]
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="11" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<circle cx="{x + ORD_R + 4:.1f}" cy="{y + ORD_R + 4:.1f}" r="{ORD_R}" fill="{C_MAIN}"/>')
    L.append(f'<text x="{x + ORD_R + 4:.1f}" y="{y + ORD_R + 7.5:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#ffffff">{oi + 1}</text>')
    sym_w_budget = NODE_W - 20
    sym_size = SYM_FONT
    while max(cjk_text_width(s, sym_size) for s in syms) > sym_w_budget and sym_size > 7.5:
        sym_size -= 0.2
    for si, s in enumerate(syms):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + SYM_TOP + si * SYM_LINE_H:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{sym_size:.1f}" '
                 f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(s)}</text>')
    path_size = 8.0
    while mono_text_width(path, path_size) > NODE_W - 14 and path_size > 6.0:
        path_size -= 0.2
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + PATH_Y:.1f}" text-anchor="middle" '
             f'font-family="monospace" font-size="{path_size:.1f}" '
             f'fill="{C_NODE_PATH}">{esc(path)}</text>')
    for ci, cline in enumerate(_CLAIM_LINES[nid]):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + CLAIM_TOP + ci * 11.5:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{CLAIM_FONT}" '
                 f'fill="{C_NODE_SUB}">{esc(cline)}</text>')
    _bw_station = cjk_text_width(sec, BADGE_FONT) + 2 * BADGE_PAD_X
    badge_svg, _bw = badge(x + NODE_W - 6 - _bw_station / 2, y, sec)
    L += badge_svg

# 底部阅读路线
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(胶囊=图上 § 站牌；实线蓝=推荐 / 虚线灰=次要)")}</text>')
_name_w = max(cjk_text_width(r[0], 11.5) for r in ROUTES)
SLOT_L = 16 + _name_w + 14
SLOT_R = w - EDGE_MARGIN - 6
SLOT_W = (SLOT_R - SLOT_L) / len(STATION_ORDER)
_max_badge_w = max(cjk_text_width(s, ROUTE_BADGE_FONT) + 2 * ROUTE_BADGE_PAD_X for s in STATION_ORDER)
assert _max_badge_w <= SLOT_W, f"站牌胶囊 {_max_badge_w:.0f}px 放不进槽位 {SLOT_W:.0f}px"
SLOT_CX = [SLOT_L + i * SLOT_W + SLOT_W / 2 for i in range(len(STATION_ORDER))]

for ri, (rname, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(rname)}</text>')
    idxs = [STATION_ORDER.index(s) for s in stops]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    if len(idxs) > 1:
        L.append(f'<line x1="{SLOT_CX[idxs[0]]:.1f}" y1="{ry:.1f}" x2="{SLOT_CX[idxs[-1]]:.1f}" '
                 f'y2="{ry:.1f}" stroke="{C_MAIN if hi else C_ROUTE_DIM}" '
                 f'stroke-width="{3 if hi else 1.5}"{dash}/>')
    for i, s in zip(idxs, stops):
        L += badge(SLOT_CX[i], ry, s, ROUTE_BADGE_FONT, ROUTE_BADGE_PAD_X)[0]

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
