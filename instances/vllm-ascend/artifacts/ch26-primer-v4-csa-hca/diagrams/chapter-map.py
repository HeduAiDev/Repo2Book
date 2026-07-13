#!/usr/bin/env python3
"""第 26 章「本章地图」——DeepSeek-V4 混合注意力(CSA/HCA)论文推导剖面图。

2026-07-14 重绘缘由:writer 重构了本章骨架(全书统一体例:谱系回顾 → 一 动机 →
二 推导(2.1-2.5)→ 三 数值推演 → 四 落地一览),旧图挂的四个"源码装配"节点
(DeepseekV4Attention/Compressor/Indexer/HashEncoder/wo_a/wo_b/DeepseekV2DecoderLayer)
与现在的自然标题不再对应,且本章 dossier.json 顶层 kind=primer——
lint_chapter_map.py 对 primer 章的符号防杜撰改核 book/papers/<slug>/*.md 论文包 +
正文,不再核 dossier.json;这些源码类名在论文包与正文里都不出现(旧图 wo_a/wo_b
已被 lint 实测判杜撰)。本轮改为纯粹的"论文推导阅读地图"——不画源码调用链,
站牌与符号全部改用正文自然标题词与 arXiv 论文里的 Eq./关键词,均可在
narrative/chapter.md 原文逐字核到。

本章为自然标题章(chapter.md 无 `## N.M` 编号标题,只有"一/二/三/四"与
"2.1/2.2..."这类叙事内 `###` 三级编号)——按契约禁用 §N.M 徽标,站牌直接摘自
正文标题词本身(如"CSA 压缩"对应"### 2.1　CSA 压缩：...")。

三段折行(画布预算:宽 ≤1500 且宽高比 ≤2.6:1):
  CSA 支路(上)——2.1 压缩 → 2.2 稀疏;
  主线(中)——谱系回顾 → 动机 → 交错 → 支线(核注意力/滑窗sink/RoPE/mHC)→
    数值推演 → 落地一览;
  HCA 支路(下)——2.3(单节点,压得够狠不必再挑,直接汇入交错)。
CSA/HCA 两条支路都只跨相邻一个泳道汇入主线(CSA:上→中,HCA:下→中),不跨双泳道
长对角线(旧图 [FIX-ROUND-2] 教训:跨两泳道的直线会穿过中间泳道整列节点)。

节点预算 9(entry/motivation/csa_compress/csa_sparse/hca/interleave/sidetrack/
numerics/exit) ≤ 12。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版)——全角(ord>0x2E80)按
    1.0×size,半角按 0.58×size,求和。中英混排的图例/标签/站牌必须用这个,
    不能直接 0.58 * size * len(s)。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
# 泳道:0=CSA 支路(上)/1=主线(中)/2=HCA 支路(下)——CSA/HCA 都只与主线相邻,
# 互相不相邻,避免任何一条边跨两条泳道。
LANES = ["CSA 支路(2.1 压缩 → 2.2 稀疏)", "主线:谱系 → 动机 → 交错 → 支线 → 数值 → 落地", "HCA 支路(2.3)"]

# (节点id, 泳道下标, 列, 泳道内行号, 论文记号/关键词, 一行短语, 站牌(自然标题词))
NODES = [
    ("entry",         1, 0, 0, "arXiv:2606.19348",
     "MLA/DSA/CSA-HCA 三代各压一个正交因子", "谱系回顾"),
    ("motivation",    1, 1, 0, "1M 上下文",
     "KV 存量与核注意力 FLOPs 随 L 同步二次增长", "动机"),
    ("csa_compress",  0, 2, 0, "Eq.(11)-(12)",
     "窗口门控凸组合压 1 条,净压缩率 1/m", "CSA 压缩"),
    ("csa_sparse",    0, 3, 0, "Eq.(16)-(17)",
     "Lightning Indexer 打分,top-k 只精读常数条", "CSA 稀疏"),
    ("hca",           2, 2, 0, "Eq.(22)-(23)",
     "不重叠压缩 m'=128,全部块稠密 MQA", "HCA"),
    ("interleave",    1, 4, 0, "compress_ratios",
     "逐层开关表:4=CSA / 128=HCA / 0=稠密", "交错"),
    ("sidetrack",     1, 5, 0, "Eq.(18)/(27)/(8)",
     "MQA+分组输出投影;滑窗+sink;部分 RoPE;mHC 双随机残差", "支线"),
    ("numerics",      1, 6, 0, "27% / 10%",
     "示意参数验证账本自洽,hybrid 远低于 dense 基线", "数值推演"),
    ("exit",          1, 7, 0, "KVComp",
     "压缩器+索引器+逐层装配+mHC 算子对,外加 LSH+Hamming 近似选块", "落地一览"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;CSA/HCA 支路各自只跨相邻一个
           # 泳道汇入主线(0↔1、2↔1),不出现跨两泳道的长对角线。
    ("entry", "motivation"),
    ("motivation", "csa_compress"), ("motivation", "hca"),
    ("csa_compress", "csa_sparse"),
    ("csa_sparse", "interleave"), ("hca", "interleave"),
    ("interleave", "sidetrack"),
    ("sidetrack", "numerics"),
    ("numerics", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("通读全程", [(0, "谱系回顾"), (1, "动机"), (2, "CSA 压缩"), (3, "CSA 稀疏"),
                (4, "交错"), (5, "支线"), (6, "数值推演"), (7, "落地一览")], True),
    ("只读 CSA:怎么压/怎么选", [(2, "CSA 压缩"), (3, "CSA 稀疏")], False),
    ("只读 HCA + 为什么交错", [(2, "HCA"), (4, "交错")], False),
    ("只关心 27%/10% 那笔账", [(6, "数值推演")], False),
]
LEGEND = [("#22c55e", "入口:三代压缩谱系与乘积账动机"), ("#3b82f6", "章内主线:推导 → 数值验证 → 落地"),
          ("#f97316", "出口:落地一览,细节回指第 24/25 章")]
TITLE = "第 26 章 · DeepSeek-V4 混合注意力推导剖面(CSA/HCA 两条压缩律 + 交错互补)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 145, 72
# 节点内文字纵向布局:固定偏移量而非按 NODE_H 比例分配——比例分配在短语要
# 折 3 行时会把首行往上挤到贴上符号标题(见 [FIX-ROUND-2] 的重叠实测),改用
# "符号基线固定在顶部 20px 处,短语首行基线在符号基线之下留 13px 间距,
# 每行再加 11px 行距"这套固定几何,折几行都不会撞上符号标题。
SYMBOL_Y_OFFSET, PHRASE_GAP, PHRASE_LINE_H = 20, 13, 11
COL_GAP, ROW_GAP = 16, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H = 20  # 宽度改按文字动态算(见 badge_w),站牌是变长中文词,不能用定长常量

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


def wrap_phrase(phrase, max_w, font_size):
    """按 token(而非裸字符)折行:每个 CJK 字各自成一个可断点 token,连续的
    ASCII/数字/连字符游程(如 "top-k"、"Eq.(16)-(17)")作为一个不可再分的原子
    token——避免把英文词从中间切断(如把 "top-k" 拆成 "t" / "op-k" 两行,
    这在旧版按字符折行时出现过,读起来像两个不相关的碎片)。空格本身也当作
    一个 token 处理,折行时随所在行一起丢弃首尾空白。"""
    tokens, cur = [], ""
    for ch in phrase:
        if ord(ch) > 0x2E80 or ch == ' ':
            if cur:
                tokens.append(cur)
                cur = ""
            tokens.append(ch)
        else:
            cur += ch
    if cur:
        tokens.append(cur)

    lines, cur_line = [], ""
    for tok in tokens:
        trial = cur_line + tok
        if cjk_text_width(trial, font_size) > max_w and cur_line.strip():
            lines.append(cur_line.strip())
            cur_line = "" if tok == ' ' else tok
        else:
            cur_line = trial
    if cur_line.strip():
        lines.append(cur_line.strip())
    return lines


def badge_w(text):
    """站牌是变长中文词(如"数值推演"4 字 vs "HCA"3 字母),不能用定长
    BADGE_W——按 cjk_text_width 估算文字宽度再加左右各 8px 内边距。"""
    return cjk_text_width(text, 11) + 16


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy) —— 节点用它贴右上角,路线图例用它居中挂线上。
    宽度按文字动态算,但形状/配色/挂法与全书统一模板一致。"""
    bw = badge_w(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"读者从哪里进/从哪里出")
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("读者")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("第 24 章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝,先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点看不出"汇合"。
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

# 节点(圆角框 + 论文记号/关键词 + 一行短语 + 右上角站牌)
for nid, lane, col, row, symbol, phrase, station in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    symbol_y = y + SYMBOL_Y_OFFSET
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{symbol_y:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    # 短语较长,按 token 折行(见 wrap_phrase——不切断英文/公式词);首行基线固定在
    # 符号基线之下 PHRASE_GAP 处,不随行数上移,避免撞上符号标题。
    lines = wrap_phrase(phrase, NODE_W - 12, 9.5)[:3]
    for li, line in enumerate(lines):
        line_y = symbol_y + PHRASE_GAP + li * PHRASE_LINE_H
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{line_y:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    bw = badge_w(station)
    L += badge(x + NODE_W - bw / 2 + 8, y, station)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, station in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, station)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录,共 3 轮渲染,2026-07-14 重绘)
#   第 1 轮:claim_readable_10s=True(谱系→动机→CSA/HCA 分支→交错→支线→数值→
#     落地,10 秒可抓住脉络) numbers_match_spec=True(27%/10%、1M、Eq. 号均可在
#     正文逐字核到,primer 章符号防杜撰改核 book/papers/ch26-primer-v4-csa-hca/
#     的口径下无一处杜撰) no_overlap=False arrows_attached=True
#     cjk_rendered=True reading_order_clear=True —— Read PNG 发现两处重叠:
#     ①"CSA 稀疏"节点短语按裸字符折行把英文词从中间切断("Lightning
#     Indexer 打分,t" / "op-k 只精读常数条","top-k"被拆成"t"/"op-k"两截,
#     读起来像两个不相关碎片);②"支线"节点符号标题
#     "Eq.(18)-(19) / Eq.(27) / Eq.(8)"过长,溢出 145px 节点框,视觉上盖住了
#     "数值推演"节点的标题区。
#   第 2 轮修复:①新增 wrap_phrase() 按 token(CJK 单字自成一 token,连续
#     ASCII/连字符游程如"top-k"当一个不可再分的原子 token)折行,不再按裸字符
#     切;②"支线"symbol 精简为 "Eq.(18)/(27)/(8)"(仍是正文出现过的真实 Eq.
#     号,只是去掉长横杠范围)。重渲后 Read PNG 复核:①的切词问题已解决,但
#     发现新问题③——"支线"与"落地一览(KVComp)"两节点短语折到 3 行时,原按
#     NODE_H 比例居中定位短语首行,行数一多首行被向上挤到贴上符号标题基线,
#     两行文字重叠糊在一起(实测两处)。
#   第 3 轮修复:把节点内文字纵向布局从"按 NODE_H 比例分配"改成"固定偏移量"
#     ——符号基线固定在节点顶部下 20px(SYMBOL_Y_OFFSET),短语首行基线固定在
#     符号基线之下 13px(PHRASE_GAP),每行再加 11px 行距(PHRASE_LINE_H),
#     折几行都不会撞上符号标题;同时把 NODE_H 从 60 提到 72 给 3 行短语留出
#     余量。重渲后 Read PNG 逐节点复核:「CSA 压缩/CSA 稀疏/HCA/交错/支线/
#     数值推演/落地一览」七个内容节点 + entry/exit 两个接口桩,共 9 个节点
#     symbol 与最多 3 行 phrase 之间、相邻节点之间均无重叠;三条汇入
#     interleave/motivation 的箭头(CSA→交错、HCA→交错、动机→CSA/HCA 两支)
#     箭头端点清晰落在目标节点边框上,无悬空;中文/公式记号(m=4、m'=128、
#     27%/10%、arXiv 号)渲染正常;阅读顺序(CSA 支路居上、主线居中、HCA 支路
#     居下,自左而右,CSA/HCA 各自只跨相邻一条泳道汇入主线,无跨泳道长对角线
#     穿框)清楚。六项全 True。
#   画布:1472×656,宽高比 2.24:1(≤2.6 达标),节点数 9(≤12 达标)。
#   lint_chapter_map.py --require 与 lint_diagram_geometry.py 均 exit 0——
#     旧图 wo_a/wo_b 两处杜撰符号已随本轮重绘(改用论文 Eq. 号/compress_ratios/
#     KVComp 等在 book/papers/ch26-primer-v4-csa-hca/ 与正文均可核到的记号)
#     一并消除,不再出现杜撰符号；本章无 `## N.M` 编号标题(自然标题章),图上
#     全部站牌均取自正文标题词本身(谱系回顾/动机/CSA 压缩/CSA 稀疏/HCA/交错/
#     支线/数值推演/落地一览),零 §N.M 徽标。
#   [独立盲审](只看 PNG + chapter.md 标题列表,不看本文件其余注释)复述:
#     从绿色"读者"桩进入,先看谱系回顾(三代压缩谱系,MLA/DSA/CSA-HCA 各压一个
#     正交因子)与动机(1M 上下文的回看税)→ 分两支:上支 CSA 先压缩(2.1,
#     Eq.(11)-(12))再稀疏选块(2.2,Eq.(16)-(17));下支 HCA 直接重压缩后稠密
#     MQA(2.3,Eq.(22)-(23))→ 两支汇入"交错"(2.4,靠 compress_ratios 逐层
#     开关表决定每层挂哪套)→"支线"(2.5,核注意力/滑窗sink/部分RoPE/mHC 四件
#     套)→"数值推演"(三,27%/10% 账本示意)→"落地一览"(四,压缩器+索引器+
#     逐层装配+mHC 算子对,外加 KVComp 的 LSH+Hamming 近似选块)→ 橙色桩离开,
#     回指第 24 章。九个站牌与正文"谱系回顾/一 动机/2.1/2.2/2.3/2.4/2.5/
#     三/四"一一对应,四条底部阅读路线(通读全程/只读 CSA/只读 HCA+交错/只
#     关心效率账)与正文开篇的选读指引("跳读 2.1 与 2.2""读 2.3 与 2.4""直接
#     跳第三节")吻合。verdict=PASS,一轮通过,未启用第 2 轮盲审配额。
