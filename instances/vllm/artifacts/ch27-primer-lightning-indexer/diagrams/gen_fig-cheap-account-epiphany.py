#!/usr/bin/env python3
"""顿悟头图：稀疏注意力没少扫一对——O(L^2) 原封不动，只是换到便宜账户。
视觉主轴=落差反转：上半「以为：稀疏=少扫历史」贵账户满宽 85.9 亿对；
下半「其实：一对没少，换账户」便宜账户照扫满宽 85.9 亿对(0.25× 单价)，
贵账户塌缩成 1/32 细条(2.66 亿对)。红条宽度落差=32×，直接量在图上。
所有数字来自 traces/run_complexity.json big(L=131072,k=2048)。
坐标全部由常量/循环计算，文本全过 esc()。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(str(s))

# ---- 数据(全部溯源，见文件头) ----
DENSE_PAIRS   = 8590000128        # run_complexity big.main_dense_ops
SPARSE_PAIRS  = 266339328         # run_complexity big.main_sparse_ops
SPEEDUP       = 32.252090566211834  # run_complexity big.speedup
PRICE_RATIO   = 0.25              # run_complexity indexer_cost_ratio
SEL_NUM, SEL_DEN = 2048, 131072   # index_topk / L  → 1.56%
sliver_frac = SPARSE_PAIRS / DENSE_PAIRS   # ≈ 1/32.25

# ---- 版式常量 ----
W, H = 1040, 500
BX   = 300          # 条带起点 x（左侧留给「以为/其实」列）
BW   = 660          # 满宽条带宽度 = 全部 85.9 亿对
LX   = 44           # 左列标签 x
BH   = 62           # 条带高
sliver_w = BW * sliver_frac      # ≈ 20.5px 贵账户细条

C_EXP_FILL, C_EXP_STK = "#ef4444", "#b91c1c"   # 贵账户=红
C_CHP_FILL, C_CHP_STK = "#3b82f6", "#1d4ed8"   # 便宜账户=蓝
INK, MUTE = "#0f172a", "#64748b"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def txt(x, y, s, size=15, anchor="middle", fill=INK, bold=False, italic=False):
    b = ' font-weight="bold"' if bold else ''
    it = ' font-style="italic"' if italic else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
             f'font-family="sans-serif" font-size="{size}"{b}{it} fill="{fill}">{esc(s)}</text>')

def band_label(cx, cy, lines, fill=INK):
    n = len(lines)
    for i, (s, sz, bd) in enumerate(lines):
        y = cy - (n - 1) * 9 + i * 18 + 5
        txt(cx, y, s, size=sz, fill=fill, bold=bd)

# ---- 标题 ----
txt(W/2, 34, "一对都没少扫：省的不是「扫多少对」，是「单价」", size=21, bold=True)
txt(W/2, 58, "稀疏注意力照样给全部 (t,s) 对打分，O(L²) 原封不动——只是换到便宜账户结账",
    size=13.5, fill=MUTE)

# ================= 上半：以为 =================
yA = 96
# 左列
txt(LX, yA + BH/2 - 6, "以为", size=20, anchor="start", bold=True, fill=C_EXP_STK)
txt(LX, yA + BH/2 + 16, "稀疏 = 少扫历史", size=12.5, anchor="start", fill=MUTE)
# 满宽红条
L.append(f'<rect x="{BX}" y="{yA}" width="{BW}" height="{BH}" rx="6" '
         f'fill="{C_EXP_FILL}" fill-opacity="0.88" stroke="{C_EXP_STK}" stroke-width="1.5"/>')
band_label(BX + BW/2, yA + BH/2, [
    ("贵账户 · 主注意力算全部 85.9 亿对 (t,s)", 15, True),
    ("单价 = 全价", 12.5, False)], fill="white")

# ---- 反转箭头 + 「其实」枢纽 ----
midY = yA + BH + 40
L.append(f'<line x1="{BX + BW/2}" y1="{yA + BH + 6}" x2="{BX + BW/2}" y2="{midY + 14}" '
         f'stroke="#334155" stroke-width="2.4" marker-end="url(#a)"/>')
txt(BX + BW/2 + 12, midY + 4, "同样 85.9 亿对，账户重新分配", size=12.5, anchor="start", fill=MUTE, italic=True)

# ================= 下半：其实 =================
yB1 = midY + 30            # 便宜账户满宽条
gap = 18
yB2 = yB1 + BH + gap       # 贵账户细条 lane

# 左列
txt(LX, yB1 + BH + gap/2 - 4, "其实", size=20, anchor="start", bold=True, fill=C_CHP_STK)
txt(LX, yB1 + BH + gap/2 + 18, "一对没少，换账户", size=12.5, anchor="start", fill=MUTE)

# 便宜账户满宽蓝条（一对都没少扫）
L.append(f'<rect x="{BX}" y="{yB1}" width="{BW}" height="{BH}" rx="6" '
         f'fill="{C_CHP_FILL}" fill-opacity="0.85" stroke="{C_CHP_STK}" stroke-width="1.5"/>')
band_label(BX + BW/2, yB1 + BH/2, [
    ("便宜账户 · indexer 照扫全部 85.9 亿对", 15, True),
    ("单价 0.25×  ·  O(L²) 原封不动", 12.5, False)], fill="white")

# 贵账户 lane：满宽虚线幽灵(朴素满宽) + 实心细条(2.66 亿对)
LANE_H = 46
L.append(f'<rect x="{BX}" y="{yB2}" width="{BW}" height="{LANE_H}" rx="6" '
         f'fill="none" stroke="{C_EXP_STK}" stroke-width="1.3" stroke-dasharray="5 4" opacity="0.55"/>')
L.append(f'<rect x="{BX}" y="{yB2}" width="{sliver_w:.1f}" height="{LANE_H}" rx="3" '
         f'fill="{C_EXP_FILL}" stroke="{C_EXP_STK}" stroke-width="1.4"/>')
# 幽灵框内提示
txt(BX + BW - 8, yB2 + LANE_H/2 + 4, "（朴素稠密的贵账户满宽 · 已塌缩）",
    size=12, anchor="end", fill=C_EXP_STK)
# 细条标注（放右侧，细条太窄容不下字）
txt(BX + sliver_w + 10, yB2 - 6, "贵账户 · 只剩 2.66 亿对进主注意力",
    size=13, anchor="start", bold=True, fill=C_EXP_STK)

# 32× 落差量尺（细条右缘 → 满宽右缘）
scaleY = yB2 + LANE_H + 40
x1, x2 = BX + sliver_w, BX + BW
for xx in (x1, x2):
    L.append(f'<line x1="{xx:.1f}" y1="{scaleY-6}" x2="{xx:.1f}" y2="{scaleY+6}" '
             f'stroke="{MUTE}" stroke-width="1.4"/>')
L.append(f'<line x1="{x1:.1f}" y1="{scaleY}" x2="{x2:.1f}" y2="{scaleY}" '
         f'stroke="{MUTE}" stroke-width="1.4"/>')
txt((x1 + x2)/2, scaleY - 12, "红条宽度落差 ≈ 32×（85.9 亿 → 2.66 亿对）",
    size=13, bold=True, fill=INK)

# 闸门注解（放量尺之下，避开）
txt((x1 + x2)/2, scaleY + 26,
    "闸门：每 query 只留 top-2048 / 131072 ≈ 1.56% 的历史条目",
    size=12, fill=MUTE)

# ---- 图例 ----
lgY = H - 46
items = [("贵账户（全价，主注意力）", C_EXP_FILL, C_EXP_STK),
         ("便宜账户（0.25× 单价，indexer 打分）", C_CHP_FILL, C_CHP_STK)]
lx = BX
for label, fc, sc in items:
    L.append(f'<rect x="{lx}" y="{lgY-11}" width="16" height="16" rx="3" '
             f'fill="{fc}" fill-opacity="0.88" stroke="{sc}" stroke-width="1.2"/>')
    txt(lx + 22, lgY + 2, label, size=12.5, anchor="start", fill=INK)
    lx += 22 + len(label) * 13 + 30
# 溯源脚注
txt(W - 30, H - 14, "对数/加速比源自 traces/run_complexity.json（L=131072, k=2048）",
    size=11, anchor="end", fill=MUTE)

L.append('</svg>')
out = Path(__file__).with_name("fig-cheap-account-epiphany.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  sliver_w={sliver_w:.2f}px  ratio=1/{1/sliver_frac:.2f}")
