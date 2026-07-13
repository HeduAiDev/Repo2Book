#!/usr/bin/env python3
"""顿悟头图(§2.5 五步法):那一下 = 「O(L²) 没被消灭,只是换了个便宜约 9 倍的付账人」。
视觉主轴 = 落差对比(以为 256× vs 其实 8.69×)。
编码:高度 = 每对单价(到尺度:8192 是 73728 的约 1/9);宽度 = 当前 decode 步参与打分的对数
(灰=稠密每步全 L 对、琥珀=indexer 仍每步全 L 对 → 同为全宽;蓝=top-k 后 512 对,已放大以可见)。
关键顿悟:左上灰块(稠密全宽)与右下琥珀块(indexer 全宽)同宽——每步打分对数(O(L),累积 L 步成 O(L²))没缩,
只是换了付账人(灰→琥珀)且单价降到 1/9,这块固定地板把端到端从 256× 拖回 8.69×。
乘法公式一律用当前步真实的 L 对(非 L²):L×73728=9.66×10⁹、L×8192=1.07×10⁹,量纲自洽。
精确 log 柱状见 §六 fig32-cost-model,本图只打这一拳。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(str(s))

# ---- 数字(全部来自 figure-requests numbers,带溯源)----
UNIT_DENSE = 73728          # 主注意力单条 KV 乘加数 per_kv_dim
UNIT_IDX   = 8192           # indexer 单对打分 H^I×d^I=64×128
DENSE_TOTAL = "9.66×10⁹"    # 9663676416 = 131072×73728
MAIN_TOTAL  = "3.77×10⁷"    # 37748736   = 512×73728 (k=512)
IDX_TOTAL   = "1.07×10⁹"    # 1073741824 = 131072×64×128 (与 k 无关)
SPEEDUP_EXP = "256×"        # L/k = 131072/512
SPEEDUP_ACT = "8.69×"       # 9663676416/(37748736+1073741824)
RATIO       = "约 1/9"      # 8192/73728≈0.111
L, K = 131072, 512

# ---- 版式常量(零手写魔数,全部派生)----
PAD = 40
PANEL_W = 384
GAP = 132                      # 中央落差留白
w = PAD * 2 + PANEL_W * 2 + GAP
LX = PAD                       # 左面板 x
RX = PAD + PANEL_W + GAP       # 右面板 x
CX = LX + PANEL_W + GAP / 2    # 中央落差 x

H_TALL = 120                                  # 单价 73728 对应高度
SCALE = H_TALL / UNIT_DENSE
H_SHORT = round(UNIT_IDX * SCALE)             # 单价 8192 → 约 1/9 高度(到尺度)
SLIVER_W = 26                                 # top-k 后蓝条(真实约 1/256 宽,放大以可见)

TOP_TITLE = 30
LEGEND_Y = 78
B1 = 268                       # 第一行基线(块底)
B2 = 452                       # 第二行基线(块底)
BANNER_Y = 476
BANNER_H = 42
FOOT_Y = 552
h = FOOT_Y + 34

GREY, BLUE, AMBER = "#94a3b8", "#3b82f6", "#d97706"
INK, MUTE = "#0f172a", "#64748b"

L_ = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
      '<defs>'
      '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
      'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
      '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" '
      'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
      '</defs>',
      f'<rect width="{w}" height="{h}" fill="white"/>']

def txt(x, y, s, size=12, color=INK, weight="normal", anchor="start"):
    L_.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" '
              f'font-size="{size}" font-weight="{weight}" fill="{color}">{esc(s)}</text>')

# ---- 标题(那一下)----
txt(PAD, TOP_TITLE, "以为 top-k 把注意力砍 256×、端到端就该 256×——", 16.5, "#b91c1c", "bold")
txt(PAD, TOP_TITLE + 22, "其实 O(L²) 没死,只换了个便宜约 9 倍的付账人,端到端只有 8.69×",
    16.5, "#b45309", "bold")

# ---- 图例(3 种语义色)----
leg = [(GREY, "稠密基线"), (BLUE, "主注意力(top-k 砍到的)"), (AMBER, "indexer(不砍·固定地板)")]
lx = PAD
for c, name in leg:
    L_.append(f'<rect x="{lx}" y="{LEGEND_Y-11}" width="15" height="13" rx="2" fill="{c}"/>')
    txt(lx + 20, LEGEND_Y, name, 11.5, MUTE)
    lx += 20 + 7.2 * len(name) + 26

# ---- 面板标题 ----
txt(LX, B1 - H_TALL - 44, "以为(朴素心智模型)", 14, INK, "bold")
txt(RX, B1 - H_TALL - 44, "其实(真实成本账)", 14, INK, "bold")

def block(x, base, bw, bh, color, name, num=None, num_side="inside"):
    """块底对齐 base,向上长 bh。name 在块上方,num 在块内(宽块)或右侧(窄块)。"""
    y = base - bh
    L_.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="4" fill="{color}" '
              f'stroke="#1e293b" stroke-width="1"/>')
    txt(x, y - 8, name, 11.5, INK, "bold")
    if num is None:
        return
    if num_side == "inside" and bh >= 26:
        txt(x + bw / 2, y + bh / 2 + 5, num, 13, "white", "bold", "middle")
    else:  # 窄块或矮块:数字放右侧
        txt(x + bw + 8, base - 5, num, 12.5, INK, "bold")

# ================= 左面板:以为 =================
# 行1:稠密全宽块
block(LX, B1, PANEL_W, H_TALL, GREY, "稠密主注意力 · 每步 L 对",
      f"L 对 × {UNIT_DENSE} = {DENSE_TOTAL} MAC", "inside")
# 落差箭头:top-k
axl = LX + PANEL_W / 2
L_.append(f'<line x1="{axl}" y1="{B1+6}" x2="{axl}" y2="{B2-H_TALL-8}" '
          'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
txt(axl + 14, (B1 + B2 - H_TALL) / 2 - 6, f"top-k：{L}→{K} 对", 11.5, "#334155", "bold")
txt(axl + 14, (B1 + B2 - H_TALL) / 2 + 10, f"宽度砍 {SPEEDUP_EXP}", 11.5, "#334155")
# 行2:主注意力细条(top-k 后)
block(LX, B2, SLIVER_W, H_TALL, BLUE, "主注意力", f"{MAIN_TOTAL} MAC · 只 {K} 对", "side")
# 结论横幅
L_.append(f'<rect x="{LX}" y="{BANNER_Y}" width="{PANEL_W}" height="{BANNER_H}" rx="6" '
          f'fill="#fee2e2" stroke="#b91c1c" stroke-width="2"/>')
txt(LX + PANEL_W / 2, BANNER_Y + 27, f"以为 端到端 = {SPEEDUP_EXP}", 15, "#b91c1c", "bold", "middle")

# ================= 右面板:其实 =================
# 行1:同一条主注意力细条
block(RX, B1, SLIVER_W, H_TALL, BLUE, "主注意力(同一条)", f"{MAIN_TOTAL} MAC", "side")
# 加号
txt(RX + PANEL_W / 2, (B1 + B2 - H_SHORT) / 2 + 6, "＋", 26, MUTE, "bold", "middle")
# 行2:indexer 全宽但 1/9 高的固定地板(与左上灰块同宽 = O(L²) 没缩)
by = B2 - H_SHORT
L_.append(f'<rect x="{RX}" y="{by}" width="{PANEL_W}" height="{H_SHORT}" rx="3" fill="{AMBER}" '
          f'stroke="#1e293b" stroke-width="1"/>')
txt(RX, by - 26, f"lightning indexer · 仍每步 L 对 × {UNIT_IDX}({RATIO} 单价)", 11.5, INK, "bold")
txt(RX, by - 10, f"= {IDX_TOTAL} MAC 固定地板(与 k 无关)", 11.5, "#b45309", "bold")
# 结论横幅
L_.append(f'<rect x="{RX}" y="{BANNER_Y}" width="{PANEL_W}" height="{BANNER_H}" rx="6" '
          f'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
txt(RX + PANEL_W / 2, BANNER_Y + 27, f"其实 端到端 = {SPEEDUP_ACT}", 15, "#b45309", "bold", "middle")

# ---- 中央落差:256× → 8.69× ----
txt(CX, B1 - H_TALL + 4, "以为", 12, MUTE, "normal", "middle")
txt(CX, B1 - H_TALL + 34, SPEEDUP_EXP, 30, "#b91c1c", "bold", "middle")
L_.append(f'<line x1="{CX}" y1="{B1-H_TALL+50}" x2="{CX}" y2="{B1-H_TALL+92}" '
          'stroke="#b91c1c" stroke-width="3" marker-end="url(#ar)"/>')
txt(CX, B1 - H_TALL + 116, "其实", 12, MUTE, "normal", "middle")
txt(CX, B1 - H_TALL + 146, SPEEDUP_ACT, 30, "#b45309", "bold", "middle")
txt(CX, B2 - 34, "O(L²) 没被消灭", 12.5, INK, "bold", "middle")
txt(CX, B2 - 16, "只换了便宜", 12.5, INK, "normal", "middle")
txt(CX, B2 + 2, "约 9 倍的付账人", 12.5, INK, "bold", "middle")

# ---- 底注(诚实标注编码 + O(L²) 量纲桥)----
txt(w / 2, FOOT_Y,
    "高度=每对单价(到尺度:8192 是 73728 的约 1/9);宽度=当前 decode 步参与打分的对数——灰/琥珀同为每步全 L 对(体量没缩),"
    "蓝条 512 对真实约 1/256 宽、已放大以可见。",
    10.5, MUTE, "normal", "middle")
txt(w / 2, FOOT_Y + 18,
    "每步对 L 个候选打分 = O(L),L 步生成累积才成 O(L²)——图中乘法用的是当前步真实的 L 对(L×73728、L×8192),非 L²。精确 log 柱状见 §六 cost-model。",
    10.5, MUTE, "normal", "middle")

L_.append('</svg>')
out = Path(__file__).with_name("fig32-epiphany-tax-shift.svg")
out.write_text('\n'.join(L_), encoding="utf-8")
print(f"wrote {out}")
