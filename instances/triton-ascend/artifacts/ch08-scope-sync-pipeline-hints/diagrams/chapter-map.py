#!/usr/bin/env python3
"""ch08「本章地图」——作用域的编译期特判 + 核间同步两代与契约 + 收窄链与编译提示
的源码剖面图。

本章是**自然标题章**(正文全是 `## 两趟 visit：先试跑数变量…` 这类自然标题，无
`## N.M` 编号)——按契约禁用 `§N.M` 徽标，站牌改用正文标题词本身(逐字是正文标题里
出现过的真实子串，供自查与 linter 逐一核对)：
  “上下文管理器”“两趟 visit”“关键字变属性”“外提成一个函数”“两条下降路径”
  “四条校验”“两侧 pipe”“两处登记”“全体同步”“收窄链”“贴条，不是改写”
  “还有一个入口”——十二个站牌对应正文十二个自然标题小节。
  “小结：语言层到这里交卷”一节不建独立节点，改由出口接口桩的说明文字收尾
  (同 ch04/ch05/ch06 惯例)。

剖面(蛇形三带，十二站递进，每节点标『真实符号 + 规范源码路径 + 一句论点』)：
  Lane0 左→右 ①with 被特判(查表键是 scope 类本身) → ②两趟 visit 的 SSA 穿线
        → ③关键字揭成 MLIR 属性 → ④scope.scope 被外提成 func.func
  Lane1 右→左 ⑤同步的两代下降路径 → ⑥四条参数校验 → ⑦两侧 pipe 的缺省配对
        → ⑧GetCore 按 op 名翻转落核
  Lane2 左→右 ⑨全体同步四模式 → ⑩PIPE / TCoreType 的收窄链 → ⑪compile_hint 贴条
        → ⑫循环上的第三个编排入口

蛇形(boustrophedon)排布的理由：十二站单排会把画布拉到 3000px 以上、远超
lint_chapter_map 的「宽 ≤1500 且宽高比 ≤2.6:1」预算。折成三带后跨带的走线是一段
短竖线(贴在最右/最左列中心)。阅读顺序由每个节点左上角的序号圆圈 ①…⑫ 显式给出。

模板：.claude/skills/svg-diagram/references/example-chapter-map.py；不可变视觉语言
(站牌徽标胶囊 / 入口绿-出口橙-主线蓝 / 高亮实线蓝-次要虚线灰 / cjk_text_width)
照搬同书 ch06 版，只改 DATA 与两处结构：①底部路线胶囊单独用一档更小的字号/内边距
(本章十二站，槽位比 ch06 的十站更窄，节点上那档字号放不进槽位)；②泳道标签的流向
箭头进入「跨带走线不压标签」断言(ch06 版把箭头写死成 `←`，蛇形三带里第三带是 `→`)。

凡图上给 IR 必标阶段(ttir / ttadapter)；IR 算子名带方言前缀(ascend. / hivm. /
scope. / annotation.)，不是 tt.。

六项自查(渲染→Read PNG 亲眼看后如实记录)：见 figure-manifest.json 该图 selfcheck。

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


# ---------------- DATA(可变：本章数据) ----------------
# 泳道文字要短：跨带竖线落在首/末列中心，标签太长会被竖线压到(下方有断言兜底)。
LANES = ["作用域 · 编译期特判",
         "核间同步 · 两代与契约",
         "全体同步 · 收窄与提示"]  # 上→下
LANE_DIR = [+1, -1, +1]  # 蛇形：+1 左→右，-1 右→左

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 规范源码路径, 一句论点, 站牌=正文自然标题词)
NODES = [
    ("withdisp", 0, 0, 0,
     ["visit_With → WITH_DISPATCH", "ASCEND_WITH_DISPATCH"],
     "python/triton/compiler/code_generator.py",
     "查表键是 scope 类对象本身、不是字符串；命中就把整条 with 的 AST 交给 handler，__enter__ / __exit__ 全程不跑",
     "上下文管理器"),
    ("twopass", 0, 1, 0,
     ["handle_scope_with", "scope_return"],
     "third_party/ascend/language/cann/extension/code_generator.py",
     "第一趟在 dummy 块里只数变量、随块作废；第二趟才真发 IR，跨界的值由 scope.return 交出再回填外层符号表",
     "两趟 visit"),
    ("attrs", 0, 2, 0,
     ["_extract_scope_attributes", "_build_mlir_attrs_from_scope_attrs"],
     "third_party/ascend/language/cann/extension/code_generator.py",
     "只揭 ast.Constant 关键字：noinline 默认开、core_mode 走两项白名单、disable_auto_sync 加前缀，其余透传；写错不报错",
     "关键字变属性"),
    ("outline", 0, 3, 0,
     ["create_scope_op", "scope.scope → func.func"],
     "third_party/ascend/ascend_ir.cc",
     "ttadapter 段的 outline-scope pass 把 region 外提成函数，tcore_type 从算子属性搬到函数属性；noinline 默认开正为它",
     "外提成一个函数"),
    ("twogen", 1, 3, 0,
     # IR 算子名 = 方言 let name + ODS 助记符，两处都回 .td 查，不从 C++ 类名推：
     # TritonAscendOps.td:L388 `TT_Ascend_Op<"custom", …>` + TritonAscendDialect.td:L15
     # `let name = "ascend"` ⇒ ascend.custom（triton::ascend::CustomOp 只是 C++ 类名）；
     # HIVMSynchronizationOps.td:L129 `HIVM_SynchronizationOp<"sync_block_set", …>` +
     # HIVMBase.td:L37 `let name = "hivm"` ⇒ hivm.sync_block_set。全图统一用 IR 名。
     ["custom_op → ascend.custom", "create_sync_block → hivm.sync_block_set"],
     "third_party/ascend/language/cann/extension/core.py",
     "旧代带 DeprecationWarning、经十二行手写分发落通用算子，receiver 与流水线在语言层就丢了；新代补两个 pipe，ttir 段即落 HIVM 专用算子",
     "两条下降路径"),
    ("contract", 1, 2, 0,
     ["sender / receiver 白名单", "0 <= event_id < 16"],
     "third_party/ascend/language/cann/extension/core.py",
     "四道检查排在唯一出口之前：核名只认 cube / vector、sender 等于 receiver 抛 ValueError、事件号 0～15、两个 pipe 必须是枚举实例",
     "四条校验"),
    ("pipepair", 1, 1, 0,
     ["sender_pipe / receiver_pipe", "PIPE_FIX / PIPE_MTE3 / PIPE_MTE2"],
     "third_party/ascend/language/cann/extension/core.py",
     "触发条件是两侧都为 None 才补缺省：cube 发用 FIX、vector 发用 MTE3、收方一律 MTE2；只给一边直接 TypeError",
     "两侧 pipe"),
    ("getcore", 1, 0, 0,
     # 同图内命名法必须一致：⑤ 已用 IR 名，这格讲的也是「建出什么算子」，故同用 IR 名
     ["GetCore / buildSyncBlockOp", "hivm.sync_block_set"],
     "third_party/ascend/ascend_ir.cc",
     "sender 说的是谁发，落核却由 op 名翻转：set 挂发方核、wait 挂收方核，两端恒互补；事件号统一提升到 64 位",
     "两处登记"),
    ("syncall", 2, 0, 0,
     ["sync_block_all", "GetSyncBlockModeAndPipes"],
     "third_party/ascend/ascend_ir.cc",
     "四种模式 all_cube / all_vector / all / all_sub_vector：模式名点到哪一侧，哪一侧就拿 PIPE_ALL，另一侧留空属性",
     "全体同步"),
    ("narrow", 2, 1, 0,
     ["PIPE：15 档 → 8 档 → 8 档", "TCoreType：4 → 4 → 2"],
     "third_party/ascend/…/Dialect/HIVM/IR/HIVMAttrs.td",
     "掉档位置不同：PIPE 掉在 pybind 导出这一级，TCoreType 掉在语言层白名单；数字要记在正确的那一级上",
     "收窄链"),
    ("hint", 2, 2, 0,
     ["compile_hint_impl 五路分派", "annotation.mark"],
     "third_party/ascend/language/cann/extension/aux_ops.py",
     "bool 必须排在假值判断之前，整数 0 会掉进假值分支变 unit 属性；其余类型是终止分支抛 ValueError。属性旁挂，原算子不动",
     "贴条，不是改写"),
    ("subblock", 2, 3, 0,
     ["class parallel(range)", "bind_sub_block"],
     "third_party/ascend/language/cann/extension/aux_ops.py",
     "第三个编排入口挂在循环上：继承基座 tl.range，只多一个关键字，告诉编译器这个循环由多个 vector 核一起跑",
     "还有一个入口"),
]
EDGES = [  # (src_id, dst_id) —— 章内递进主线，统一主线蓝
    ("withdisp", "twopass"), ("twopass", "attrs"), ("attrs", "outline"),
    ("outline", "twogen"),                   # 跨带：右列竖直下行
    ("twogen", "contract"), ("contract", "pipepair"), ("pipepair", "getcore"),
    ("getcore", "syncall"),                  # 跨带：左列竖直下行
    ("syncall", "narrow"), ("narrow", "hint"), ("hint", "subblock"),
]
# 站序槽位 = NODES 的顺序(①…⑫)；路线里的站牌按该槽位对齐
STATION_ORDER = [n[7] for n in NODES]
# (路线名, [站牌…] 按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
ROUTES = [
    ("从头顺读（全览）", STATION_ORDER, True),
    ("只想会用：写 scope", ["上下文管理器", "关键字变属性", "外提成一个函数"], False),
    ("只想会用：核间同步", ["两条下降路径", "四条校验", "两侧 pipe", "全体同步"], False),
    ("只关心编译期机制", ["两趟 visit", "两处登记", "收窄链", "贴条，不是改写"], False),
]
LEGEND = [
    ("#22c55e", "入口：上一章讲完怎么往语言里注册一条新算子；本章讲怎么指挥这些算子——哪段代码归哪种核、两种核怎么对表、怎么给编译器递条子"),
    ("#3b82f6", "章内主线：with 被编译器特判 → 两趟 visit 建 region → 关键字变属性 → 外提成函数 ‖ 同步两代与四条校验 → 缺省 pipe → 落核翻转 ‖ 收窄链与贴条"),
    ("#f97316", "出口：小结把语言层定性为「交卷」——它造出的都是半成品 IR；下一部分转向 MLIR 与 Linalg，沿下降链一站站接住"),
]
TITLE = "第 8 章 · 作用域、核间同步与流水线提示：scope / sync_block / compile_hint 的源码剖面"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 250
COL_GAP, ROW_GAP = 32, 20
EDGE_MARGIN, STUB_W, STUB_H = 10, 52, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 22
LANE_LABEL_H, BAND_PAD = 24, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 62, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 42
BADGE_H = 20
BADGE_FONT = 11
BADGE_PAD_X = 13  # 徽标左右各留的内边距(动态宽度=文本宽+2×BADGE_PAD_X)
# 本章十二站，底部槽位比 ch06 的十站更窄——路线条上的胶囊单独用一档更小的字号/内边距，
# 否则最长站牌放不进槽位(下方 assert 兜底)。节点上那档字号不变，仍是模板的 BADGE_FONT。
ROUTE_BADGE_FONT, ROUTE_BADGE_PAD_X = 10, 9
CLAIM_FONT = 9.2
SYM_FONT, SYM_LINE_H = 12.0, 14
ORD_R = 9  # 序号圆圈半径

_BREAK_AFTER = set("，；：、/ ,;")


def wrap_claim(text, max_w, size):
    """一句论点太长时换行——只在标点/斜杠/空格之后断行，不允许劈开一个标识符
    (如 create_sync_block)或一个中文词。贪心找“prefix 仍不超宽的最靠后一个
    合法断点”；找不到合法断点才整句照旧单行放行。"""
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
    more = wrap_claim(line2, max_w, size)
    return [line1] + more


# 每个节点的论点先按 NODE_W 预算换行一遍，取全章最多的行数统一定 NODE_H；
# 符号行数同理取全章最大值——同一行号跨泳道对齐用的是同一个 NODE_H。
CLAIM_MAXW = NODE_W - 16
_CLAIM_LINES = {n[0]: wrap_claim(n[6], CLAIM_MAXW, CLAIM_FONT) for n in NODES}
_max_claim_lines = max(len(v) for v in _CLAIM_LINES.values())
_max_sym_lines = max(len(n[4]) for n in NODES)
SYM_TOP = 26                                   # 第一行符号的基线偏移(须 > BADGE_H/2 + 字高)
PATH_Y = SYM_TOP + _max_sym_lines * SYM_LINE_H  # 路径行基线
CLAIM_TOP = PATH_Y + 15                        # 首行论点基线
NODE_H = CLAIM_TOP + (_max_claim_lines - 1) * 12 + 10

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
NODE_LANE = {n[0]: n[1] for n in NODES}
NODE_COL = {n[0]: n[2] for n in NODES}
ORDER_OF = {n[0]: i + 1 for i, n in enumerate(NODES)}


def lane_label(i):
    """泳道标签 = 名字 + 流向箭头(蛇形三带里第二带是 ←、一三带是 →)。"""
    return LANES[i] + (" →" if LANE_DIR[i] > 0 else " ←")


# 跨带竖线附着在框宽中心：它只穿过**目标泳道**的标签行，故只核目标带的标签宽度
# (泳道标签从 x=16 起排，须在竖线左侧留出空隙，否则标签被走线压住——no_overlap 一项)。
CROSS_FRAC = 0.5
for _s, _d in EDGES:
    if NODE_LANE[_s] == NODE_LANE[_d]:
        continue
    _line_x = COLX[NODE_COL[_d]] + NODE_W * CROSS_FRAC
    _lb = lane_label(NODE_LANE[_d])
    assert 16 + cjk_text_width(_lb, 13) + 12 <= _line_x, (
        f"泳道标签『{_lb}』会被 x={_line_x:.0f} 的跨带走线压住——请缩短标签")
    _bw_d = cjk_text_width(NODE_BY_ID[_d][7], BADGE_FONT) + 2 * BADGE_PAD_X
    assert NODE_W * CROSS_FRAC + 6 <= NODE_W - 8 - _bw_d, (
        f"跨带走线会压到站牌『{NODE_BY_ID[_d][7]}』的胶囊")

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD
assert w <= 1500 and w / h <= 2.6, f"画布预算超标：{w}x{h}, {w / h:.2f}:1"


def badge(cx, cy, text, font=BADGE_FONT, pad_x=BADGE_PAD_X):
    """站牌徽标胶囊，居中挂在 (cx,cy)。宽度按 cjk_text_width 动态算(自然标题
    站牌比 §N.M 长得多，模板里的定宽 BADGE_W 会把长站牌文字挤出胶囊)。
    胶囊样式/配色/圆角高度仍是模板的不可变视觉语言。"""
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)；三条说明偏长，纵向各占一行堆叠，避免横排挤出画布
for li, (color, label) in enumerate(LEGEND):
    _row_y = TOP_PAD + TITLE_H + 14 + li * 14
    L.append(f'<rect x="{PAD_L}" y="{_row_y - 11}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 18}" y="{_row_y}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')

# 泳道背景 + 标签(带流向箭头指示蛇形方向) + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(lane_label(i))}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                 f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩：入口挂在第一站左侧，出口挂在末站右侧
_first, _last = NODES[0][0], NODES[-1][0]
ex, ey = NODE_XY[_first]; ey += NODE_H / 2
xx, xy = NODE_XY[_last]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("读者")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("下一部分")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)：同带内按该带流向做左右附着；跨带同列走竖直附着(下边中点 → 上边中点)。
for src, dst in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if NODE_LANE[src] != NODE_LANE[dst]:
        p1 = (xs_ + NODE_W * CROSS_FRAC, ys_ + NODE_H)
        p2 = (xd + NODE_W * CROSS_FRAC, yd)
    elif LANE_DIR[NODE_LANE[src]] > 0:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
    else:
        p1 = (xs_, ys_ + NODE_H / 2)
        p2 = (xd + NODE_W, yd + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 序号圆圈 + 真实符号 + 规范源码路径 + 一句论点 + 右上角站牌徽标)
for nid, lane, col, row, syms, path, claim, station in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="12" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    # 阅读序号：蛇形排布下「左上→右下」的默认约定不成立，序号显式给出看图顺序
    L.append(f'<circle cx="{x + ORD_R + 4:.1f}" cy="{y + ORD_R + 4:.1f}" r="{ORD_R}" '
             f'fill="{C_MAIN}"/>')
    L.append(f'<text x="{x + ORD_R + 4:.1f}" y="{y + ORD_R + 7.5:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#ffffff">'
             f'{ORDER_OF[nid]}</text>')
    # 符号行：字号按最宽一行自适应缩，保证不越框(序号圆圈占左上角，故留出 2×ORD_R 余量)
    sym_w_budget = NODE_W - 16 - 2 * ORD_R
    sym_size = SYM_FONT
    while max(cjk_text_width(s, sym_size) for s in syms) > sym_w_budget and sym_size > 8.0:
        sym_size -= 0.3
    for si, s in enumerate(syms):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + SYM_TOP + si * SYM_LINE_H:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{sym_size:.1f}" '
                 f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(s)}</text>')
    path_size = 8.3
    while mono_text_width(path, path_size) > NODE_W - 16 and path_size > 6.0:
        path_size -= 0.3
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + PATH_Y:.1f}" text-anchor="middle" '
             f'font-family="monospace" font-size="{path_size:.1f}" '
             f'fill="{C_NODE_PATH}">{esc(path)}</text>')
    for ci, cline in enumerate(_CLAIM_LINES[nid]):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + CLAIM_TOP + ci * 12:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{CLAIM_FONT}" '
                 f'fill="{C_NODE_SUB}">{esc(cline)}</text>')
    # 徽标右对齐钉在框内：居中挂角会让长站牌探出框外压到相邻节点
    _bw_station = cjk_text_width(station, BADGE_FONT) + 2 * BADGE_PAD_X
    badge_svg, _bw = badge(x + NODE_W - 8 - _bw_station / 2, y, station)
    L += badge_svg

# 底部「阅读路线」：模板要求的多条读法条(高亮=实线蓝 / 次要=虚线灰)。
# 蛇形版图里列号 ≠ 阅读序，故胶囊按「站序槽位」等分排布：同一站在各条路线里
# 上下对齐，缺席的站留空——一眼看出某条路线跳过了哪几站。
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         # 站序区间用半角数字写：圈号字体只到 ⑩，⑪/⑫ 在 rsvg 的回退字体里是豆腐块
         f'{esc("阅读路线（胶囊=图上站牌，按第 1 → 第 12 站的站序排；实线蓝=推荐 / 虚线灰=次要）")}</text>')
_name_w = max(cjk_text_width(r[0], 12.0) for r in ROUTES)
SLOT_L = 16 + _name_w + 14
SLOT_R = w - EDGE_MARGIN - 6
SLOT_W = (SLOT_R - SLOT_L) / len(STATION_ORDER)
_max_badge_w = max(cjk_text_width(s, ROUTE_BADGE_FONT) + 2 * ROUTE_BADGE_PAD_X for s in STATION_ORDER)
assert _max_badge_w <= SLOT_W, f"站牌胶囊 {_max_badge_w:.0f}px 放不进槽位 {SLOT_W:.0f}px"
SLOT_CX = [SLOT_L + i * SLOT_W + SLOT_W / 2 for i in range(len(STATION_ORDER))]

for ri, (rname, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12.0" '
             f'fill="{C_NODE_TITLE}">{esc(rname)}</text>')
    idxs = [STATION_ORDER.index(s) for s in stops]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{SLOT_CX[idxs[0]]:.1f}" y1="{ry:.1f}" x2="{SLOT_CX[idxs[-1]]:.1f}" '
             f'y2="{ry:.1f}" stroke="{C_MAIN if hi else C_ROUTE_DIM}" '
             f'stroke-width="{3 if hi else 1.5}"{dash}/>')
    for i, s in zip(idxs, stops):
        L += badge(SLOT_CX[i], ry, s, ROUTE_BADGE_FONT, ROUTE_BADGE_PAD_X)[0]

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
