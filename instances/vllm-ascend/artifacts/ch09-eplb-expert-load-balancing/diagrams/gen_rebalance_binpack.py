#!/usr/bin/env python3
"""DefaultEplb.rebalance_experts：冗余副本 + 贪心装箱把偏斜负载铺平，5% 收益门槛判定是否落地。
数字来自 explainer/traces/rebalance.json（run_rebalance.py act_skew / skip_mild 两个场景）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


# ---- 数据（全部带 provenance：traces/rebalance.json） ----
NPU_LABELS = ["NPU0", "NPU1", "NPU2", "NPU3"]
HEAT_BEFORE = [100, 20, 20, 20]
PEAK_BEFORE, MEAN_BEFORE, IMB_BEFORE = 100, 40, 2.5
HEAT_AFTER = [60, 60, 20, 20]
PEAK_AFTER, IMB_AFTER = 60, 1.5
NUM_REDUNDANCY = 1
GATE_RATIO, GATE_VALUE, DROP_PCT, CHANGE = 0.95, 95, 40.0, 1
SKIP_PEAK_BEFORE, SKIP_PEAK_AFTER, SKIP_DROP_PCT, SKIP_CHANGE = 52, 50, 3.85, 0

W, H = 1360, 600
S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append('<defs>')
S.append('<marker id="a" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="9" markerHeight="7" '
          'orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
S.append('</defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')
S.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="22" font-weight="bold" '
         f'fill="#0f172a">DefaultEplb.rebalance_experts：冗余副本 + 贪心装箱，把偏斜负载铺平</text>')
S.append(f'<text x="{W/2}" y="66" text-anchor="middle" font-size="14" fill="#64748b">'
         f'峰值热度 {PEAK_BEFORE}→{PEAK_AFTER}（imbalance {IMB_BEFORE}→{IMB_AFTER}），降 {DROP_PCT:.0f}% 越过 5% 收益门槛 → change=1 落地迁移</text>')

BAR_W, BAR_GAP, SCALE = 60, 22, 1.4
CHART_H = 140  # heat=100 对应像素高度
BASE_Y = 340
N = len(NPU_LABELS)
TOTAL_BARS_W = N * BAR_W + (N - 1) * BAR_GAP


def draw_panel(x0, panel_w, title, subtitle, heats, hot_idx, info_line):
    cx0 = x0 + (panel_w - TOTAL_BARS_W) / 2
    S.append(f'<text x="{x0+panel_w/2}" y="{BASE_Y-CHART_H-46}" text-anchor="middle" font-size="17" '
              f'font-weight="bold" fill="#1e293b">{esc(title)}</text>')
    S.append(f'<text x="{x0+panel_w/2}" y="{BASE_Y-CHART_H-26}" text-anchor="middle" font-size="12.5" '
              f'fill="#64748b">{esc(subtitle)}</text>')
    # 基线
    S.append(f'<line x1="{x0+10}" y1="{BASE_Y}" x2="{x0+panel_w-10}" y2="{BASE_Y}" stroke="#94a3b8" stroke-width="1.5"/>')
    for i, heat in enumerate(heats):
        bx = cx0 + i * (BAR_W + BAR_GAP)
        bh = heat * SCALE
        by = BASE_Y - bh
        hot = i in hot_idx
        fill = "#fef3c7" if hot else "#e2e8f0"
        stroke = "#d97706" if hot else "#64748b"
        txtc = "#92400e" if hot else "#334155"
        S.append(f'<rect x="{bx}" y="{by}" width="{BAR_W}" height="{bh}" rx="4" fill="{fill}" '
                  f'stroke="{stroke}" stroke-width="{2 if hot else 1}"/>')
        S.append(f'<text x="{bx+BAR_W/2}" y="{by-8}" text-anchor="middle" font-size="14" '
                  f'font-weight="bold" fill="{txtc}">{heat}</text>')
        S.append(f'<text x="{bx+BAR_W/2}" y="{BASE_Y+20}" text-anchor="middle" font-size="12" '
                  f'fill="#475569">{esc(NPU_LABELS[i])}</text>')
    S.append(f'<text x="{x0+panel_w/2}" y="{BASE_Y+46}" text-anchor="middle" font-size="13" '
              f'font-weight="bold" fill="#1e293b">{esc(info_line)}</text>')


PANEL_A_X, PANEL_A_W = 40, 340
PANEL_C_X, PANEL_C_W = 980, 340
MID_X0, MID_X1 = PANEL_A_X + PANEL_A_W, PANEL_C_X

draw_panel(PANEL_A_X, PANEL_A_W, "偏斜（ACT 场景）", "expert0 独大，全灌到 NPU0",
           HEAT_BEFORE, {0}, f"peak={PEAK_BEFORE}  mean={MEAN_BEFORE}  imbalance={IMB_BEFORE}")
draw_panel(PANEL_C_X, PANEL_C_W, "铺平（装箱后）", "冗余副本分摊到两卡",
           HEAT_AFTER, {0, 1}, f"peak={PEAK_AFTER}  imbalance={IMB_AFTER}（降 {DROP_PCT:.0f}%）")

# ---- 中段：主箭头 + 两步骤（每步骤名 + 两行短说明，避免中点相撞） ----
arrow_y = BASE_Y - CHART_H / 2
S.append(f'<line x1="{MID_X0+8}" y1="{arrow_y}" x2="{MID_X1-8}" y2="{arrow_y}" '
          f'stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
mid_cx = (MID_X0 + MID_X1) / 2
STEPS = [
    (MID_X0, mid_cx, "① add_redundant", [f"复制冗余副本（num_redundancy={NUM_REDUNDANCY}）", "expert0：100 → 拆成两份各 50"]),
    (mid_cx, MID_X1, "② 贪心装箱", ["逐 expert / 副本", "放进当前总热度最低的卡"]),
]
for x_from, x_to, label, sub_lines in STEPS:
    scx = (x_from + x_to) / 2
    S.append(f'<line x1="{scx}" y1="{arrow_y-40}" x2="{scx}" y2="{arrow_y-6}" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="3,3"/>')
    S.append(f'<text x="{scx}" y="{arrow_y-88}" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#1e293b">{esc(label)}</text>')
    for li, line in enumerate(sub_lines):
        S.append(f'<text x="{scx}" y="{arrow_y-68+li*18}" text-anchor="middle" font-size="11.5" fill="#475569">{esc(line)}</text>')
S.append(f'<line x1="{mid_cx}" y1="{arrow_y-40}" x2="{mid_cx}" y2="{arrow_y+38}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2,3"/>')

# ---- 下段左：5% 收益门槛徽章（绿=落地） ----
badge_y = BASE_Y + 90
badge_gap = 40
badge_w, badge_h = (W - 2 * PANEL_A_X - badge_gap) / 2, 118
S.append(f'<rect x="{PANEL_A_X}" y="{badge_y}" width="{badge_w}" height="{badge_h}" rx="10" '
          f'fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
S.append(f'<text x="{PANEL_A_X+18}" y="{badge_y+30}" font-size="14" font-weight="bold" fill="#166534">'
          f'收益门槛：peak_after &lt; {GATE_RATIO}×peak_before ?</text>')
S.append(f'<text x="{PANEL_A_X+18}" y="{badge_y+56}" font-size="13.5" fill="#166534">'
          f'{PEAK_AFTER} &lt; {GATE_RATIO}×{PEAK_BEFORE} = {GATE_VALUE} → 成立，峰值降 {DROP_PCT:.0f}%（&gt;5% 门槛）</text>')
S.append(f'<text x="{PANEL_A_X+18}" y="{badge_y+82}" font-size="15" font-weight="bold" fill="#166534">'
          f'⇒ change={CHANGE}（落地迁移）</text>')

# ---- 下段右：SKIP 对照（灰=不搬） ----
skip_x = W - PANEL_A_X - badge_w
S.append(f'<rect x="{skip_x}" y="{badge_y}" width="{badge_w}" height="{badge_h}" rx="10" '
          f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>')
S.append(f'<text x="{skip_x+18}" y="{badge_y+30}" font-size="14" font-weight="bold" fill="#334155">'
          f'对照 · SKIP 场景（已近均衡）</text>')
S.append(f'<text x="{skip_x+18}" y="{badge_y+56}" font-size="13.5" fill="#334155">'
          f'peak {SKIP_PEAK_BEFORE}→{SKIP_PEAK_AFTER}，仅降 {SKIP_DROP_PCT:.2f}%（&lt;5% 门槛，不达标）</text>')
S.append(f'<text x="{skip_x+18}" y="{badge_y+82}" font-size="15" font-weight="bold" fill="#334155">'
          f'⇒ change={SKIP_CHANGE}（按兵不动，搬运成本不划算）</text>')

S.append('</svg>')
out_path = __file__.replace("gen_", "").replace(".py", ".svg")
open(out_path, "w").write("\n".join(S))
print(f"wrote {out_path}")
