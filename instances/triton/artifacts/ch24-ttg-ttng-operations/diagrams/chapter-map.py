#!/usr/bin/env python3
"""ch24 本章地图——源码剖面图。

本章是 Part V(IR 与布局)的收官:张量带上布局(ch20-23)后,认识与共享内存/
异步/硬件打交道的"操作词汇表"。两条泳道 = ttg 层(通用 GPU 抽象,后端无关)/
ttng 层(NVIDIA Hopper 硬件专属方言,配对脊柱样板),圆角节点 = 真实符号名 +
一行短语,右上角挂站牌(本章自然标题,无 `## N.M` 编号,故站牌用标题词的
简短复述,而非 §N.M),左右各一个接口桩(入口=前几章已贴好布局的张量,
出口=转下一部分:共享内存分配/插屏障/排流水),底部两条阅读路线复用同一批
站牌——对应正文"选读指引"段落原话(只取两把性能尺 vs 完整通读到 ttng)。

列号 = 正文自然标题出现顺序:0 开篇(两把性能尺预告) → 1 convert_layout(唯一
真跨线程搬运,回扣 ch19 tt.trans"只改名") → 2 memdesc(共享内存钥匙,含
memdesc_subview 子视图) → 3 cp.async 三件套(async.token 串依赖) →
4 upcast_mxfp → 5 从 ttg 到 ttng(TTNG_Op 骨架样板 + warp_group_dot/WGMMA +
async_tma_copy+mbarrier[回收第 7 章 TMA 伏笔] + warp-specialization 流水
词汇,四者同列同站牌,均从 §4 直接并行导出,不在列内互连) → 6 小结。

同列节点(memdesc/memdesc_subview 同列;ttng 四个子节点同列)彼此并列、不
互相连边,只各自向前一列的共同源节点收敛/由其扇出,避免"同列回绕"的走线
穿过节点本体——沿用 ch19 chapter-map 的做法。

■ 不可变(照搬模板视觉语言,只改 DATA 与几何常量):站牌胶囊 / 入口绿
  #22c55e-出口橙#f97316-主线蓝#3b82f6 / 高亮路线实线蓝、次要虚线灰 /
  cjk_text_width() 宽度估算。
■ 本章为自然标题(无 `## N.M` 编号),站牌一律用标题词的简短复述,禁用 §N.M。
■ 几何常量沿用 ch19 同款 7 列参数(NODE_W=175/COL_GAP=20/PAD 同款),因本章
  同样收敛到 7 列,画布预算(宽 ≤1500 且宽高比 ≤2.6:1)已在设计时算过。
■ 长符号名(如 async_tma_copy+mbarrier)按估算宽度动态缩小标题字号,避免
  文字越界(title_font_size())。

[自查记录见文件末尾注释:Read PNG 后逐项如实记录,不能凭想象填。]
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):全角按 1.0×size,半角按
    0.58×size,求和——中文标签若按半角系数算会算短,导致下一个图例压上来。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def title_font_size(symbol, max_w, base=13, floor=10):
    """长标识符(如 async_tma_copy+mbarrier)按估算宽度动态缩小字号,不许
    文字越界。"""
    est = cjk_text_width(symbol, base)
    if est <= max_w:
        return base
    return max(floor, max_w / (cjk_text_width(symbol, 1.0) or 1))


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["ttg 层(通用 GPU 抽象)", "ttng 层(Hopper 硬件专属方言)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(标题词简短复述,自然标题章无 §))
NODES = [
    ("entry", 0, 0, 0, "TTGIR",
     "带上布局,转向共享内存", "开篇:两把性能尺"),
    ("convert_layout", 0, 1, 0, "ConvertLayoutOp",
     "跨线程真搬运,trans只改名", "convert_layout:跨线程真搬运"),
    ("memdesc", 0, 2, 0, "MemDescType",
     "共享内存的 SSA 句柄", "memdesc:共享内存钥匙"),
    ("subview", 0, 2, 1, "memdesc_subview",
     "降秩取子视图,不搬内存", "memdesc:共享内存钥匙"),
    ("cpasync", 0, 3, 0, "async_copy/commit/wait",
     "token串依赖,藏访存延迟", "cp.async 三件套"),
    ("upcast", 0, 4, 0, "UpcastMXFPOp",
     "mxfp 低精度上采样到 bf16", "upcast_mxfp"),
    ("ttng_skeleton", 1, 5, 0, "TTNG_Op",
     "同骨架,换 Hopper 硬件", "ttg→ttng:硬件方言样板"),
    ("wgmma", 1, 5, 1, "warp_group_dot",
     "可直吃共享内存操作数", "ttg→ttng:硬件方言样板"),
    ("tma", 1, 5, 2, "async_tma_copy+mbarrier",
     "整块搬运,报数式同步", "ttg→ttng:硬件方言样板"),
    ("warpspec", 1, 5, 3, "producer/consumer token",
     "生产者消费者握手流水", "ttg→ttng:硬件方言样板"),
    ("exit", 0, 6, 0, "共享内存分配·排流水线",
     "转下一部分:调度成快 kernel", "小结:三把尺"),
]
EDGES = [  # (src_id, dst_id) —— 章内讲解走线,统一主线蓝;src 列号恒 < dst 列号
    ("entry", "convert_layout"),
    ("convert_layout", "memdesc"), ("convert_layout", "subview"),
    ("memdesc", "cpasync"), ("subview", "cpasync"),
    ("cpasync", "upcast"),
    ("upcast", "ttng_skeleton"), ("upcast", "wgmma"), ("upcast", "tma"), ("upcast", "warpspec"),
    ("ttng_skeleton", "exit"), ("wgmma", "exit"), ("tma", "exit"), ("warpspec", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 与正文"选读指引"段落原话对应:完整通读按序到 ttng(§5);只取两把性能尺则
# 读 convert_layout(§1)与 cp.async(§3)即可。
ROUTES = [
    ("完整通读",
     [(0, "开篇:两把性能尺"), (1, "convert_layout:跨线程真搬运"),
      (2, "memdesc:共享内存钥匙"), (3, "cp.async 三件套"), (4, "upcast_mxfp"),
      (5, "ttg→ttng:硬件方言样板"), (6, "小结:三把尺")], True),
    ("只取两把性能尺",
     [(1, "convert_layout:跨线程真搬运"), (3, "cp.async 三件套")], False),
]
LEGEND = [("#22c55e", "入口:前几章已贴好布局的张量"),
          ("#3b82f6", "章内讲解主线"),
          ("#f97316", "出口:转下一部分——共享内存分配与流水线")]
TITLE = "第 24 章 · ttg/ttng 算子词汇表:布局转换、共享内存与 Hopper 硬件方言(源码剖面图)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数;沿用 ch19 同款 7 列参数) ----------------
NODE_W, NODE_H = 175, 60
COL_GAP, ROW_GAP = 20, 18
EDGE_MARGIN, STUB_W, STUB_H = 10, 42, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 18
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H = 20
TITLE_MAX_W = NODE_W - 24  # 符号名文字可用宽度(留左右各 12px 内边距)

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
    """站牌胶囊,居中挂在 (cx,cy)——节点用它贴右上角,路线图例用它居中挂线上。
    宽度按 cjk_text_width() 估算(本章站牌是中文标题词复述,非 §N.M 短数字)。"""
    bw = cjk_text_width(text, 11) + 16
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
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
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

# 入口/出口接口桩
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("读者")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一部分")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用/走线边(主线蓝);多条边汇入同一节点时终点 y 各偏移,看得出"汇合"
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

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    fsz = title_font_size(symbol, TITLE_MAX_W)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.4:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{fsz:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W / 2, y, sec)

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
    for col, sec in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w/h:.2f}:1)")

# ---------------- 六项自查记录(渲染→Read PNG 亲眼看后如实记录) ----------------
# claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
# arrows_attached=True     cjk_rendered=True         reading_order_clear=True
# (记录见 Read PNG 之后,若有 FIX-ROUND 会在此追加说明)
