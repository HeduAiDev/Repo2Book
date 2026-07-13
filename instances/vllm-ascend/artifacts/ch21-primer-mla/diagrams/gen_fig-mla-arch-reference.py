#!/usr/bin/env python3
"""fig-mla-arch-reference —— MLA 完整数据流「参考架构图」(次要参考图,非头图)。

头图是压缩落差顿悟图 fig-mla-epiphany;本图相反——要准确、完整、可查:从 h_t 到 u_t
的全链路,三路下投影 / 缓存高亮 / 逐头上投影 / 解耦 RoPE 拼接 / 注意力 / 输出,每条边标维度。
忠实 DeepSeek-V2 (arXiv:2405.04434) §2.1 Eq(9)-(19) 与 §3.1.2 配置:
  d=5120, n_h=128, d_h=128, d_c=512, d'_c=1536, d_h^R=64。
  Eq9  c^KV = W^DKV h        (d_c=512)
  Eq10 k^C  = W^UK  c^KV     (逐头 d_h=128)
  Eq11 v    = W^UV  c^KV     (逐头 d_h=128)
  Eq12 c^Q  = W^DQ  h        (d'_c=1536)
  Eq13 q^C  = W^UQ  c^Q      (逐头 d_h=128)
  Eq14 q^R  = RoPE(W^QR c^Q) (逐头 d_h^R=64)
  Eq15 k^R  = RoPE(W^KR h)   (d_h^R=64, 全头共享)
  Eq16 q_i  = [q^C_i ; q^R_i]           (每头 192)
  Eq17 k_i  = [k^C_i ; k^R]  (k^R 共享)  (每头 192)
  Eq18 o_i  = Σ Softmax(q_i·k_j/√(d_h+d_h^R)) v_j   (每头 128)
  Eq19 u    = W^O [o_1;…;o_{n_h}]        (16384 → d=5120)
推理期只缓存 c^KV + k^R(共 d_c+d_h^R=576),其余现算——图上用加锁高亮框标出。
色=角色:绿=压缩潜向量/下投影,黄=查询内容,蓝=键内容,红=值,紫=位置/RoPE,灰=聚合/输出。
坐标全部由常量计算;文本全 esc();参考图无 trace 数字(维度=论文常量,豁免 spec.numbers)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(str(s))

def cw(c):
    o = ord(c)
    if o == 0x20: return 0.30
    if 0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F: return 1.0
    if c.isascii() and c.isalnum(): return 0.58
    return 0.5

def tw(s, size): return size * sum(cw(c) for c in s)

# ---------------- palette (语义色:色=角色) ----------------
ROLE = {
    "input": ("#f8fafc", "#334155", "#0f172a"),   # 中性:外部输入 h_t
    "comp":  ("#dcfce7", "#16a34a", "#15803d"),   # 绿:压缩潜向量 / 下投影
    "query": ("#fef08a", "#ca8a04", "#854d0e"),   # 黄:查询内容 q^C / W^UQ
    "key":   ("#dbeafe", "#2563eb", "#1d4ed8"),   # 蓝:键内容 k^C / W^UK
    "value": ("#fecaca", "#dc2626", "#991b1b"),   # 红:值 v / W^UV
    "rope":  ("#ede9fe", "#7c3aed", "#6d28d9"),   # 紫:位置 / RoPE
    "agg":   ("#e2e8f0", "#475569", "#0f172a"),   # 灰:聚合 / 输出
}
EDGE = {"comp": "#16a34a", "query": "#ca8a04", "key": "#2563eb",
        "value": "#dc2626", "rope": "#7c3aed", "agg": "#64748b"}
SLATE, MUT = "#0f172a", "#64748b"
CACHE_S, CACHE_F, CACHE_T = "#ea580c", "#fff7ed", "#9a3412"   # 橙:缓存高亮(非语义角色,专用)

L = []
def rect(x, y, w, h, fill, stroke, sw=1.5, rx=9, dash=None, op=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' fill-opacity="{op}"' if op is not None else ''
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}/>')

def text(x, y, s, size, fill, anchor="middle", weight=None):
    wt = f' font-weight="{weight}"' if weight else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="sans-serif" '
             f'font-size="{size}"{wt} fill="{fill}">{esc(s)}</text>')

def edge(p1, p2, color, sw=2.0, mid=None):
    mk = f'url(#a_{mid})' if mid else 'url(#a_agg)'
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{color}" stroke-width="{sw}" marker-end="{mk}"/>')

# ---------------- geometry ----------------
BH = 50            # 双行节点高
ATT_H = 58
INPUT_CY, L1_CY, L2_CY, L3_CY, L4_CY, L5_CY = 138, 250, 392, 512, 624, 726
TITLE_Y = 34

# 节点定义: id -> (cx, cy, title, sub, role, minw)
N = {
    "h":  (600, INPUT_CY, "h_t", "输入 · d = 5120", "input", 150),
    "cQ": (305, L1_CY, "c^Q", "查询潜向量 · 1536", "comp", 138),
    "kR": (560, L1_CY, "k^R", "位置键 · 64 · 全头共享", "rope", 168),
    "cKV":(878, L1_CY, "c^KV", "KV 潜向量 · 512", "comp", 146),
    "qC": (225, L2_CY, "q^C", "查询内容 · 每头 128", "query", 132),
    "qR": (388, L2_CY, "q^R", "查询位置 · 每头 64", "rope", 132),
    "kC": (700, L2_CY, "k^C", "键内容 · 每头 128", "key", 132),
    "v":  (1052, L2_CY, "v", "值 · 每头 128", "value", 132),
    "Q":  (306, L3_CY, "Q = q^C ⊕ q^R", "每头 128 + 64 = 192", "query", 176),
    "K":  (700, L3_CY, "K = k^C ⊕ k^R", "每头 128 + 64 = 192", "key", 176),
    "V":  (1052, L3_CY, "V = v", "每头 128", "value", 132),
    "att":(686, L4_CY, "多头注意力（128 头并行）", "Softmax(q·kᵀ / √192) · v", "agg", 340),
    "u":  (686, L5_CY, "u_t （本层输出）", "拼接 128 头 16384 → W^O → d = 5120", "agg", 320),
}
BOX = {}   # id -> (x, y, w, h)
for k, (cx, cy, ti, sub, role, minw) in N.items():
    h = ATT_H if k == "att" else BH
    w = max(minw, tw(ti, 14) + 26, tw(sub, 10.5) + 22)
    BOX[k] = (cx - w / 2, cy - h / 2, w, h)

def top(k):    x, y, w, h = BOX[k]; return (x + w / 2, y)
def bot(k):    x, y, w, h = BOX[k]; return (x + w / 2, y + h)
def topx(k, px): x, y, w, h = BOX[k]; return (px, y)      # 指定 x 进入顶边

# ---------------- canvas size ----------------
_content_r = max(BOX[k][0] + BOX[k][2] for k in BOX)
STRIP_L1 = "三路下投影从 h_t 各自压缩;推理期只把 c^KV 与 k^R 写进 KV 缓存(共 512+64=576),其余 K / V 现算。"
STRIP_L2 = "上投影逐头还原 q^C / k^C / v;位置项 q^R、k^R 单独走 RoPE,拼接成每头 192 维的 Q、K 与 128 维 V,注意力后经 W^O 投回 d=5120。"
OUTER = 40
_strip_r = OUTER + 16 + max(tw(STRIP_L1, 12.5), tw(STRIP_L2, 12.5)) + 16 + OUTER
W = max(_content_r + OUTER, _strip_r)
STRIP_H = 60
strip_top = L5_CY + BH / 2 + 30
H = strip_top + STRIP_H + OUTER - 10

# ---------------- SVG head + markers ----------------
L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}">')
mk = ['<defs>']
for role, col in EDGE.items():
    mk.append(f'<marker id="a_{role}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
              f'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{col}"/></marker>')
mk.append('</defs>')
L.append(''.join(mk))
L.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>')

# ---------------- 标题 ----------------
text(W / 2, TITLE_Y, "MLA 完整数据流:从 h_t 到 u_t —— 三路下投影 · 只缓存两根潜向量 · 逐头上投影 · 解耦 RoPE 拼接",
     15.5, SLATE, weight="bold")

# ---------------- 图例(色=角色) ----------------
legend = [("comp", "压缩潜向量 / 下投影"), ("query", "查询内容 q^C·W^UQ"),
          ("key", "键内容 k^C·W^UK"), ("value", "值 v·W^UV"),
          ("rope", "位置 / RoPE"), ("agg", "聚合 / 输出")]
lx, ly = OUTER, 60
for role, lab in legend:
    fill, stroke, _ = ROLE[role]
    itemw = 15 + 6 + tw(lab, 11) + 22
    if lx + itemw > W - OUTER:
        lx, ly = OUTER, ly + 22
    rect(lx, ly - 11, 15, 13, fill, stroke, sw=1.3, rx=3)
    text(lx + 21, ly, lab, 11, SLATE, anchor="start")
    lx += itemw
# 缓存高亮图例项
citemw = 24 + 6 + tw("推理期缓存(锁)", 11) + 22
if lx + citemw > W - OUTER:
    lx, ly = OUTER, ly + 22
rect(lx, ly - 11, 22, 13, CACHE_F, CACHE_S, sw=1.4, rx=3, dash="4 2")
text(lx + 28, ly, "推理期缓存(锁)", 11, CACHE_T, anchor="start")

# ---------------- 缓存高亮框(在节点之下先画:阴影 + 橙虚线区) ----------------
cx0 = BOX["kR"][0] - 14
cx1 = BOX["cKV"][0] + BOX["cKV"][2] + 14
cy0 = L1_CY - BH / 2 - 16
cy1 = L1_CY + BH / 2 + 16
rect(cx0 + 5, cy0 + 6, cx1 - cx0, cy1 - cy0, "#000000", "none", sw=0, rx=14, op=0.10)   # 阴影
rect(cx0, cy0, cx1 - cx0, cy1 - cy0, CACHE_F, CACHE_S, sw=2.2, rx=14, dash="7 4")

# ---------------- 边(先于节点画,箭头端点贴框边) ----------------
def bottom_at(k, px):
    x, y, w, h = BOX[k]; return (px, y + h)
def top_at(k, px):
    x, y, w, h = BOX[k]; return (px, y - 0.5)

# 下投影 (h_t → 三潜向量)
edge(bot("h"), top("cQ"), EDGE["comp"], mid="comp")
edge(bottom_at("h", 585), top("kR"), EDGE["rope"], mid="rope")
edge(bottom_at("h", 640), top("cKV"), EDGE["comp"], mid="comp")
# 上投影
edge(bottom_at("cQ", 285), top("qC"), EDGE["query"], mid="query")
edge(bottom_at("cQ", 330), top("qR"), EDGE["rope"], mid="rope")
edge(bottom_at("cKV", 830), top("kC"), EDGE["key"], mid="key")
edge(bottom_at("cKV", 930), top("v"), EDGE["value"], mid="value")
# 解耦拼接 (⊕)
edge(bot("qC"), top_at("Q", 270), EDGE["query"], mid="query")
edge(bot("qR"), top_at("Q", 345), EDGE["rope"], mid="rope")
edge(bot("kC"), top_at("K", 700), EDGE["key"], mid="key")
edge(bottom_at("kR", 560), top_at("K", 660), EDGE["rope"], mid="rope")   # k^R 共享 → K
edge(bot("v"), top("V"), EDGE["value"], mid="value")
# Q/K/V → 注意力
edge(bot("Q"), top_at("att", 560), EDGE["agg"], mid="agg")
edge(bot("K"), top_at("att", 686), EDGE["agg"], mid="agg")
edge(bot("V"), top_at("att", 812), EDGE["agg"], mid="agg")
# 注意力 → 输出
edge(bot("att"), top("u"), EDGE["agg"], mid="agg")

# ---------------- 节点 ----------------
for k, (cx, cy, ti, sub, role, minw) in N.items():
    x, y, w, h = BOX[k]
    fill, stroke, tcol = ROLE[role]
    rect(x, y, w, h, fill, stroke, sw=1.8)
    text(cx, cy - 4, ti, 14, tcol, weight="bold")
    text(cx, cy + 13, sub, 10.5, MUT)

# ---------------- 边标签(权重 + 维度,置于线侧空白) ----------------
def elabel(x, y, name, dim, col, anchor="middle"):
    text(x, y, name, 11.5, col, anchor=anchor, weight="bold")
    if dim:
        text(x, y + 14, dim, 10, MUT, anchor=anchor)

elabel(410, 184, "W^DQ", "d'_c=1536", EDGE["comp"], anchor="end")
elabel(525, 184, "W^KR·RoPE", "d_h^R=64", EDGE["rope"], anchor="end")
elabel(782, 184, "W^DKV", "d_c=512", EDGE["comp"], anchor="start")
elabel(214, 337, "W^UQ", "逐头 128", EDGE["query"], anchor="end")
elabel(372, 337, "W^QR·RoPE", "逐头 64", EDGE["rope"], anchor="start")
elabel(700, 333, "W^UK", "逐头 128", EDGE["key"], anchor="end")
elabel(1005, 333, "W^UV", "逐头 128", EDGE["value"], anchor="start")
elabel(706, 674, "W^O", "16384 → 5120", EDGE["agg"], anchor="start")

# ⊕ 合并标注(拼接算子)
for cxk in (306, 700):
    text(cxk, L3_CY - BH / 2 - 6, "⊕", 15, SLATE, weight="bold")
text(612, 470, "k^R 全头共享", 9.5, ROLE["rope"][2])

# ---------------- 缓存框:锁 + 说明(节点之上) ----------------
# 迷你锁(纯图形:环 + 体),画在两潜向量之间的留白中央上方
lockx, locky = 693, L1_CY - 8
L.append(f'<path d="M{lockx-7:.1f},{locky-4:.1f} v-5 a7,7 0 0 1 14,0 v5" fill="none" '
         f'stroke="{CACHE_S}" stroke-width="2.4"/>')
rect(lockx - 11, locky - 4, 22, 17, CACHE_S, CACHE_S, sw=0, rx=3)
text(lockx, locky + 30, "只缓存这两个", 11.5, CACHE_T, weight="bold")
text(lockx, locky + 44, "(其余现算)", 10, CACHE_T)

# ---------------- 左侧阶段标签 ----------------
for cy, lab in [(INPUT_CY, "输入"), (L1_CY, "① 下投影"), (L2_CY, "② 上投影"),
                (L3_CY, "③ 解耦拼接"), (L4_CY, "④ 注意力"), (L5_CY, "⑤ 输出")]:
    text(OUTER, cy + 4, lab, 11.5, MUT, anchor="start", weight="bold")

# ---------------- 底部要点条 ----------------
rect(OUTER, strip_top, W - 2 * OUTER, STRIP_H, "#fffbeb", CACHE_S, sw=1.3, rx=10)
text(OUTER + 16, strip_top + 24, STRIP_L1, 12.5, "#78350f", anchor="start", weight="bold")
text(OUTER + 16, strip_top + 45, STRIP_L2, 12.5, "#78350f", anchor="start")

L.append('</svg>')
out = Path(__file__).with_name("fig-mla-arch-reference.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W:.0f}x{H:.0f}), nodes={len(N)}")
