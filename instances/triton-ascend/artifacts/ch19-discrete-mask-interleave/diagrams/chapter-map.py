#!/usr/bin/env python3
"""ch19「本章地图」——离散掩码改写 + 交错访存优化剖面：两条彼此独立的机制线
（kind=deep，skip_impl=true；本章为纯 C++ MLIR pass/工具函数，无精简版，图上只呈现
真实源码符号）。

本章是**自然标题章**（`## 一、…`中文数字标题，无 `## N.M` 编号）——按契约「自然标题章
禁用 §N.M 徽标，站牌改用标题词本身」，全部站牌摘自各节标题的精确子串（逐一在下面
NODES 定义处标注取自哪一节标题，自查时按此核对）。这与 ch16/17/18（同为自然标题章）
的处理手法一致。

模板改自 ch14 chapter-map.py 的成熟版本（数字标题章示例）：沿用其自适应符号字号
（cjk_text_width 逐字符估算 + 自动收缩循环）、monospace 路径行、阅读序编号圆圈、
路线槽位系统（STATION_ORDER + SLOT_CX）——这些都是几何/排版工具，与「§N.M vs 自然
标题」无关，可以直接复用；只是本章站牌文本改用标题子串而非 §N.M。

■ 两条机制线怎么摆放（关键设计取舍，供复核）：
  Lane0 = DiscreteMaskAccessConversionPass.cpp（§一~§八，离散掩码判定/拆分/三条
          改写/跨章打标签/驱动收尾）
  Lane1 = InterleaveOptimization.cpp（§九~§十二，交错访存优化，正文原话「相对独立
          的第二块」——两条 Lane 之间没有任何边，画布上也不共享列坐标语义）。

  Lane0 内部节点顺序是**简化叙述流**，不是逐字节字面调用链（做法与 ch14 文件头注释
  同一原则——那里 lattice→combine→transfer 是 parse 递归的兄弟分支而非串行调用，
  这里同理）：
    - `isDiscreteMask`(gate,§一)/`collectAndLeaves`(leaves,§二)/`decomposeAndMask`
      (decompose,§三) 三者按源码实际调用方向是 decompose 内部调用 leaves、且
      Load/Store/Atomic 三个 pattern 各自独立调用 gate 与 decompose（源码里
      `DiscreteMaskLoadConversion`/`StoreConversion`/`AtomicConversion` 三个
      matchAndRewrite 各自开头都重新跑一遍 `isDiscreteMask`+`decomposeAndMask`，
      不是「跑一次、三个 pattern 共享同一次调用结果」）。图上把 gate→leaves→
      decompose 画成一条共享前置链、只画一次，再统一扇出到三条改写——这是为了
      不把图变成 3×2=6 条重复边的蜘蛛网，牺牲了「严格按调用方向画」换来可读性，
      在此显式记录，不是笔误。
    - `runOnOperation`(driver,§八) 是这条 pass 真正的外部入口（PassManager 调用
      它，它再 `patterns.add<Load,Store,Atomic>()` 组装三个 pattern 并
      `applyPatternsAndFoldGreedily`）——但正文把它放在最后一节（§八）才讲「这些
      改写规则怎么被跑起来」，属于「先讲积木、后讲怎么拼」的讲解顺序，不代表
      源码执行顺序把它排在最后。图上把 driver 摆在 decompose 之后、三条改写
      之前，对应它在**执行时**确实是"assemble+drive"这三条改写的位置，与它在
      **正文里**排第八节不冲突（阅读序编号圆圈①~⑫仍按正文 §一~§十二 实际顺序
      标号，与节点在图上的 x/y 位置无关，见 NODE_ORDER）。
    - `store`(§五)/`atomic`(§六) 都会打 `DiscreteMask` 属性（attr,§七）；`load`
      (§四) 不会（正文明确「本章在两处打上它：离散 store 改写后的新 store（§五）、
      xchg 类无幺元的 atomic（§六）」）——所以 EDGES 里只有 store→attr、
      atomic→attr，没有 load→attr，这不是遗漏。

  Lane1 列序按"先讲的构件在左、消费构件的高层函数在右"排布（col0=expand/parity，
  col1=deinter/inter），与 Lane0"调用方在左"的习惯方向相反——这里的边画的是**数据流
  "喂给"方向**（expand→deinter、parity→deinter/inter），不是字面调用方向（实际调用
  方向反而是 deinter/inter 调用 expand/parity）；这样选是为了让 EXIT_NODE=inter 落在
  Lane1 最右列，出口桩箭头能从 inter 右边缘直接出画布，不必穿过 parity 的框
  （早前一版把 deinter/inter 摆在左列时，"预告 ch20"的出口线正好从 inter 右边缘
  横穿 parity 的整个节点框，Read PNG 时发现后改的——这是 no_overlap 自查的真实
  一次修正，不是假设性描述）。
  `DeinterleaveStatusOptimization`(deinter,§十一) 源码里直接调用
  `expandInterleaveMemRefType`(expand,§九) 与 `recountReinterpretCastOffset`
  (parity,§十)（正文 §十一源码块逐字可见这两处调用）；`InterleaveStatusOptimization`
  (inter,§十二) 的可见源码片段（L457-511）用到 `indexModeRecord.first/.second`
  （IndexMode 概念源自 §十），但该片段未显示它是否重新调用 `recountReinterpretCastOffset`
  或 `expandInterleaveMemRefType`（正文对 L370-456 做了省略）——为避免杜撰未见
  的调用，图上只画 parity→inter（IndexMode 概念依赖，有把握），不画 expand→inter
  （证据不足，宁可漏画不画错）。deinter 与 inter 之间**不画边**：正文说 inter
  「正是 §十一 deinterleave 的逆运算」，这是语义上的镜像关系，不是一方调用另一方，
  画一条箭头会误导成调用关系。

■ 跨章标注（exp-2026-07-18-04 硬规则：目标章号 > 本章号用「预告」，< 本章号用
  「回指」）：
  入口桩（绿，挂在 gate/§一）："回指 ch13"——ch13《MaskAnalysis》建立的
  `MaskState::parse`，本章 §一总闸直接复用它的成败当连续/离散判据（deps=ch13）。
  出口桩（橙，挂在 inter/§十二）："预告 ch20"——正文小结明确预告下一章讲
  「TritonAscend 方言与它的几条逃生舱」（outline-final.json 核实 ch20 标题
  《TritonAscend 方言与三条逃生舱》）。
  另有一条不走独立箭头的跨章标注：attr(§七) 节点自身「论点」文字里写明
  「回指 ch14（Unstructure 解包）/ch17（UseAnalysis）认领」——ch14、ch17 都 < ch19，
  按规则也是「回指」；但这不是"这次调用把控制权交还给更早章节"的返回关系，而是
  "本章产出的属性被更早出场、但在编译管线里实际运行在本章之后的两趟 pass 消费"，
  用整条橙色出口桩表达容易误读成"回指=返回上一步"，所以选择写进节点论点文字，
  不另画箭头/桩。

六项自查（渲染→Read PNG 亲眼看后如实记录）：见同目录 figure-manifest.json 该图
selfcheck。

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


_BREAK_AFTER = set("，；：、/ ,;()（）")


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
    "third_party/ascend/lib/DiscreteMaskAccessConversion · 判连续→拆掩码→三条改写→打跨章标签 · DiscreteMaskAccessConversionPass.cpp",
    "third_party/ascend/lib/Utils · 交错访存优化(相对独立第二块) · InterleaveOptimization.cpp",
]

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 省略前缀后的路径, 一句论点, 站牌(取自对应节标题的精确子串))
NODES = [
    ("gate", 0, 0, 0,
     ["isDiscreteMask"],
     "…/DiscreteMaskAccessConversionPass.cpp",
     "早退门：parse 成功→连续放行结构化路径(先 eraseInsertedOps 撤副作用)；parse 失败→离散、本 pass 接管",
     "连续与离散"),  # 取自 §一标题「一、连续与离散的分水岭：复用 MaskState 判连续」
    ("leaves", 0, 1, 0,
     ["collectAndLeaves"],
     "…/DiscreteMaskAccessConversionPass.cpp",
     "andi 两枝递归拆；broadcast(andi) 用分配律下推成 broadcast(a)&broadcast(b)，否则整体当叶子",
     "andi 树拍平"),  # 取自 §二标题「二、拆掩码：andi 树拍平 + broadcast 分配律下推」
    ("decompose", 0, 2, 0,
     ["decomposeAndMask"],
     "…/DiscreteMaskAccessConversionPass.cpp",
     "逐叶子跑 parse：能矩形化且 isMask()→contMask（收窄防越界），否则→discMask（逐元素选择）",
     "收窄防越界"),  # 取自 §三标题「三、混合掩码拆 contMask 与 discMask：收窄防越界」
    ("driver", 0, 3, 0,
     ["runOnOperation"],
     "…/DiscreteMaskAccessConversionPass.cpp",
     "RewritePatternSet 装 3 个 pattern，greedy 应用到收敛；收尾再跑 CSE+Canonicalize 清 parse 留下的死码",
     "Pass 驱动"),  # 取自 §八标题「八、Pass 驱动：三个 pattern greedy 应用 + 清死码」
    ("load", 0, 4, 0,
     ["DiscreteMaskLoadConversion"],
     "…/DiscreteMaskAccessConversionPass.cpp",
     "分流：contMask&&discMask 都在→安全 load(contMask)+select(combinedMask)；否则 fallback 全载+select(mask)",
     "安全全载"),  # 取自 §四标题「四、离散 Load 改写：安全全载 + select 屏蔽」
    ("store", 0, 4, 1,
     ["DiscreteMaskStoreConversion"],
     "…/DiscreteMaskAccessConversionPass.cpp",
     "读-改-写：load 目标原值→select 拼新值→store 回写；sync_block_lock/unlock 包成临界区，打 DiscreteMask",
     "读-改-写"),  # 取自 §五标题「五、离散 Store 改写：读-改-写 + 临界区序列化」
    ("atomic", 0, 4, 2,
     ["DiscreteMaskAtomicConversion"],
     "…/DiscreteMaskAccessConversionPass.cpp",
     "查 initMap 幺元表：未选中位置填幺元(参与运算不改结果)；xchg 无幺元，只打 DiscreteMask 后 failure()",
     "选幺元填充"),  # 取自 §六标题「六、离散 Atomic 改写：按运算类型选幺元填充」
    ("attr", 0, 5, 1,
     ['discreteMaskAttrName = "DiscreteMask"'],
     "…/include/Utils/Utils.h",
     "字符串属性、非算子；打在 store/atomic 新 op 上，回指 ch14（Unstructure 解包标量化）/ch17（UseAnalysis）认领",
     "跨章接头"),  # 取自 §七标题「七、DiscreteMask 属性：本章打上、下游消费的跨章接头」
    ("expand", 1, 0, 0,
     ["expandInterleaveMemRefType"],
     "…/InterleaveOptimization.cpp",
     "复制 memref 类型：末维 shape×2、末维 stride 归 1，静态 offset 归 0——stride=2 视图变连续 2N 描述",
     "末维翻倍"),  # 取自 §九标题「九、交错视图末维翻倍：把 stride=2 还原成连续 2N」
    ("parity", 1, 0, 1,
     ["recountReinterpretCastOffset"],
     "…/InterleaveOptimization.cpp",
     "常量 offset 断言只能是 0/1；值型 offset 靠 addi(_,1) 那个『+1』识别 ODD_MODE，否则 EVEN_MODE",
     "偶还是奇"),  # 取自 §十标题「十、偶还是奇：靠 offset 里的那个「+1」判定」
    ("deinter", 1, 1, 0,
     ["DeinterleaveStatusOptimization"],
     "…/InterleaveOptimization.cpp",
     "5 步：expand 造新 srcType→新 reinterpret_cast→alloc+copy 搬上片→to_tensor→extract_slice 隔一取一",
     "隔一取一"),  # 取自 §十一标题「十一、Deinterleave（load 侧）：翻倍搬回 + 片上隔一取一」
    ("inter", 1, 1, 1,
     ["InterleaveStatusOptimization"],
     "…/InterleaveOptimization.cpp",
     "偶/奇两条 materialize 按 offset 0/1、stride 2 插进 2N 空 tensor，单次 MaterializeInDestination 落盘",
     "交织成一次落盘"),  # 取自 §十二标题「十二、Interleave（store 侧）：两条 materialize 交织成一次落盘」
]
NODE_BY_ID = {n[0]: n for n in NODES}
# 阅读序①~⑫按正文 §一~§十二 实际出现顺序标号(与节点在图上的 x/y 位置无关——
# driver/§八在图上摆在 decompose 之后、三条改写之前，是执行时机位置，
# 阅读序仍按它在正文里排第 8 位标"⑧"，不随图上位置挪动)。
NODE_ORDER = ["gate", "leaves", "decompose", "load", "store", "atomic", "attr",
              "driver", "expand", "parity", "deinter", "inter"]
ENTRY_NODE, EXIT_NODE = "gate", "inter"

EDGES_MAIN = [  # 主线，实线蓝——Lane0 内是简化叙述流(非逐字节调用链，详见文件头注释)
    ("gate", "leaves"),
    ("leaves", "decompose"),
    ("decompose", "driver"),
    ("driver", "load"), ("driver", "store"), ("driver", "atomic"),
    ("store", "attr"), ("atomic", "attr"),  # load 不打 DiscreteMask 属性，无边
    # Lane1：expand/parity 是先讲的构件，deinter 确认调用了两者、画 expand→deinter、
    # parity→deinter(数据流"喂给"方向，非字面调用方向——deinter 才是调用方，见文件头
    # 注释)；inter 只确认依赖 parity(IndexMode)，不画 expand→inter(证据不足，正文对
    # inter 前半段 L370-456 做了省略，看不到它是否重新调用 expandInterleaveMemRefType)，
    # 也不画 deinter→inter(正文说 inter 是 deinter 的逆运算，语义镜像非调用关系)。
    ("expand", "deinter"), ("parity", "deinter"),
    ("parity", "inter"),
]
EDGES_SIDE = []  # 本章两条 Lane 之间无边(正文原话「相对独立的第二块」)

STATION_ORDER = [NODE_BY_ID[nid][7] for nid in NODE_ORDER]  # 站牌槽位=阅读序顺序
ROUTES = [  # (路线名, [站牌…]按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
    ("全览(按图上编号 1~12 顺序读)", STATION_ORDER, True),
    ("选读：连续/离散判据(呼应「读§一与§二」)",
     [NODE_BY_ID["gate"][7], NODE_BY_ID["leaves"][7]], False),
    ("选读：离散写多亏(呼应「直奔§五」)",
     [NODE_BY_ID["store"][7]], False),
    ("选读：交错优化独立块(呼应「从§九起」)",
     [NODE_BY_ID["expand"][7], NODE_BY_ID["parity"][7], NODE_BY_ID["deinter"][7], NODE_BY_ID["inter"][7]], False),
]
LEGEND = [
    ("#22c55e", "入口(回指 ch13)：MaskState::parse 判连续失败，才轮到本章的离散/交错两条优化路径接管"),
    ("#3b82f6", "主线：Lane0 总闸判定→拆分掩码→三条改写→跨章打标签；Lane1 交错优化按讲解顺序辅助连线"),
    ("#f97316", "出口(预告 ch20)：TritonAscend 方言与它的几条逃生舱"),
]
TITLE = "第 19 章 · 离散掩码改写 + 交错访存优化剖面(源码走线 + 讲解站牌)"
SUBNOTE = "节点路径省略公共前缀 third_party/ascend/(以 … 代替)；完整路径见正文行内夹注"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 195
COL_GAP, ROW_GAP = 20, 22
EDGE_MARGIN, STUB_W, STUB_H = 10, 50, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 14
LANE_LABEL_H, BAND_PAD = 22, 13
TOP_PAD, TITLE_H, SUBNOTE_H, LEGEND_H, BOTTOM_PAD = 12, 26, 16, 3 * 14.5 + 12, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_FONT, BADGE_PAD_X = 20, 10.5, 7
ROUTE_BADGE_FONT, ROUTE_BADGE_PAD_X = 10, 8
CLAIM_FONT = 8.6
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

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD
assert w <= 1500 and w / h <= 2.6, f"画布预算超标：{w}x{h}, {w / h:.2f}:1"


def badge(cx, cy, text, font=BADGE_FONT, pad_x=BADGE_PAD_X):
    """站牌胶囊，居中挂在 (cx,cy)，宽度按 cjk_text_width 动态算。"""
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
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#166534">{esc("回指 ch13")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#9a3412">{esc("预告 ch20")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 主线(实线蓝)——多条边汇入同一节点时(driver→load/store/atomic 的宿是同一节点?
# 不是，各自不同行，天然错开；store→attr 与 atomic→attr 才是真汇入同一节点，
# 按 y 偏移分开，避免重合看不出"汇合")。
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
    y_offset = (i - (n - 1) / 2) * 12 if n > 1 else 0
    p2 = (xd, yd + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
for src, dst in EDGES_SIDE:  # 本章为空，占位保留结构一致性
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1, p2 = (xs_ + NODE_W / 2, ys_ + NODE_H), (xd + NODE_W / 2, yd)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4" '
             f'marker-end="url(#mSide)"/>')

# 节点(圆角框 + 阅读序圆圈 + 符号 + 路径 + 论点 + 右上角站牌)
_ord_of = {nid: i + 1 for i, nid in enumerate(NODE_ORDER)}
for nid, lane, col, row, syms, path, claim, sec in NODES:
    x, y = NODE_XY[nid]
    oi = _ord_of[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="11" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<circle cx="{x + ORD_R + 4:.1f}" cy="{y + ORD_R + 4:.1f}" r="{ORD_R}" fill="{C_MAIN}"/>')
    L.append(f'<text x="{x + ORD_R + 4:.1f}" y="{y + ORD_R + 7.5:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#ffffff">{oi}</text>')
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
         f'{esc("阅读路线(胶囊=图上站牌；实线蓝=推荐 / 虚线灰=次要)")}</text>')
_name_w = max(cjk_text_width(r[0], 11) for r in ROUTES)
SLOT_L = 16 + _name_w + 14
SLOT_R = w - EDGE_MARGIN - 6
SLOT_W = (SLOT_R - SLOT_L) / len(STATION_ORDER)
_max_badge_w = max(cjk_text_width(s, ROUTE_BADGE_FONT) + 2 * ROUTE_BADGE_PAD_X for s in STATION_ORDER)
assert _max_badge_w <= SLOT_W, f"站牌胶囊 {_max_badge_w:.0f}px 放不进槽位 {SLOT_W:.0f}px"
SLOT_CX = [SLOT_L + i * SLOT_W + SLOT_W / 2 for i in range(len(STATION_ORDER))]

for ri, (rname, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11" '
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
