#!/usr/bin/env python3
"""figure_id: davinci-onchip-memory-hierarchy
claim: 达芬奇片上是显式管理的多级 scratchpad:GM(DRAM)<->UB(192KB,vector 路径)/
L1<->L0A/L0B/L0C(cube 路径);搬运由 tl.load(GM->UB)/tl.store(UB->GM) 显式发起,
不是 GPU 的隐式 cache。
template: layout
numbers (spec.numbers, 逐条来源见 explainer.json):
  - UB 容量 = 192 KB = 1,572,864 bits        (programming_guide.md:180,272 逐字)
  - double-buffer 后可用 UB = ~96 KB          (192KB 减半，算术推导，derive_ch02.json)
  - GM->UB / UB->GM 搬运算子 = tl.load / tl.store (programming_guide.md:100-101)
  - L0A/L0B/L0C 职责: L0A 左矩阵/L0B 右矩阵/L0C 累加结果 (HPCA'21 paper-attributed, 未联网核实 —— 软性标注)
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


W, H = 1220, 720

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="arrowB" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="8" markerHeight="7" '
    'orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#1d4ed8"/></marker>'
    '<marker id="arrowG" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="8" markerHeight="7" '
    'orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---- title --------------------------------------------------------------
L.append(
    f'<text x="40" y="42" font-family="sans-serif" font-size="20" font-weight="bold" '
    f'fill="#0f172a">{esc("片上内存层级：显式搬运的多级 scratchpad（不是 GPU 隐式 cache）")}</text>'
)
L.append(
    f'<text x="40" y="66" font-family="sans-serif" font-size="13" fill="#475569">'
    f'{esc("vector 路径 GM↔UB（192KB）由 tl.load/tl.store 显式发起；cube 路径经 L1→L0A/L0B→L0C")}</text>'
)

COLORS = {"gm": "#e2e8f0", "ub": "#93c5fd", "l1": "#c4b5fd", "l0": "#a5b4fc"}

# ---- GM box (left, tall) -------------------------------------------------
GM_X, GM_Y, GM_W, GM_H = 50, 110, 220, 500
L.append(
    f'<rect x="{GM_X}" y="{GM_Y}" width="{GM_W}" height="{GM_H}" rx="14" '
    f'fill="{COLORS["gm"]}" stroke="#64748b" stroke-width="2"/>'
)
L.append(
    f'<text x="{GM_X + GM_W / 2}" y="{GM_Y + 34}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="16" font-weight="bold" fill="#0f172a">{esc("GM")}</text>'
)
L.append(
    f'<text x="{GM_X + GM_W / 2}" y="{GM_Y + 56}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" fill="#334155">{esc("Global Memory")}</text>'
)
L.append(
    f'<text x="{GM_X + GM_W / 2}" y="{GM_Y + 74}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#334155">{esc("(片外 DRAM)")}</text>'
)
L.append(
    f'<text x="{GM_X + GM_W / 2}" y="{GM_Y + GM_H - 20}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#475569">{esc("容量大 · 延迟高")}</text>'
)

# ---- Chip boundary (dashed, holds UB + cube path) ------------------------
CHIP_X = GM_X + GM_W + 90
CHIP_Y, CHIP_W, CHIP_H = 100, W - (GM_X + GM_W + 90) - 40, 520
L.append(
    f'<rect x="{CHIP_X}" y="{CHIP_Y}" width="{CHIP_W}" height="{CHIP_H}" rx="14" '
    f'fill="#f8fafc" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6,4"/>'
)
L.append(
    f'<text x="{CHIP_X + 16}" y="{CHIP_Y + 26}" font-family="sans-serif" font-size="14" '
    f'font-weight="bold" fill="#334155">{esc("片上（显式管理的 scratchpad）")}</text>'
)

# ---- UB box (vector path) -----------------------------------------------
UB_X = CHIP_X + 40
UB_Y = CHIP_Y + 50
UB_W, UB_H = 300, 150
L.append(
    f'<rect x="{UB_X}" y="{UB_Y}" width="{UB_W}" height="{UB_H}" rx="12" '
    f'fill="{COLORS["ub"]}" stroke="#1d4ed8" stroke-width="2"/>'
)
L.append(
    f'<text x="{UB_X + UB_W / 2}" y="{UB_Y + 30}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="16" font-weight="bold" fill="#0f172a">{esc("UB")}</text>'
)
L.append(
    f'<text x="{UB_X + UB_W / 2}" y="{UB_Y + 52}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" fill="#1e3a8a">{esc("Unified Buffer（vector 路径）")}</text>'
)
L.append(
    f'<text x="{UB_X + UB_W / 2}" y="{UB_Y + 78}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="14" font-weight="bold" fill="#1e3a8a">{esc("192 KB = 1,572,864 bits")}</text>'
)
L.append(
    f'<text x="{UB_X + UB_W / 2}" y="{UB_Y + 104}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" fill="#92400e">{esc("double-buffer 默认开 → 减半")}</text>'
)
L.append(
    f'<text x="{UB_X + UB_W / 2}" y="{UB_Y + 124}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="13" font-weight="bold" fill="#92400e">{esc("可用 ≈ 96 KB")}</text>'
)

# ---- tl.load / tl.store arrows between GM and UB -------------------------
gm_right_x = GM_X + GM_W
gm_load_y = UB_Y + 40
gm_store_y = UB_Y + UB_H - 30
L.append(
    f'<path d="M {gm_right_x} {gm_load_y} L {UB_X} {gm_load_y}" '
    f'fill="none" stroke="#1d4ed8" stroke-width="2.2" marker-end="url(#arrowB)"/>'
)
L.append(
    f'<text x="{(gm_right_x + UB_X) / 2}" y="{gm_load_y - 10}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#1d4ed8">{esc("tl.load")}</text>'
)
L.append(
    f'<path d="M {UB_X} {gm_store_y} L {gm_right_x} {gm_store_y}" '
    f'fill="none" stroke="#1d4ed8" stroke-width="2.2" marker-end="url(#arrowB)"/>'
)
L.append(
    f'<text x="{(gm_right_x + UB_X) / 2}" y="{gm_store_y + 22}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#1d4ed8">{esc("tl.store")}</text>'
)

# ---- cube path: L1 -> L0A/L0B -> L0C -------------------------------------
L1_X = CHIP_X + 40
L1_Y = UB_Y + UB_H + 60
L1_W, L1_H = 120, 90
L.append(
    f'<rect x="{L1_X}" y="{L1_Y}" width="{L1_W}" height="{L1_H}" rx="10" '
    f'fill="{COLORS["l1"]}" stroke="#6d28d9" stroke-width="2"/>'
)
L.append(
    f'<text x="{L1_X + L1_W / 2}" y="{L1_Y + 32}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="15" font-weight="bold" fill="#0f172a">{esc("L1")}</text>'
)
L.append(
    f'<text x="{L1_X + L1_W / 2}" y="{L1_Y + 54}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#4c1d95">{esc("cube 路径")}</text>'
)
L.append(
    f'<text x="{L1_X + L1_W / 2}" y="{L1_Y + 72}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="10" fill="#4c1d95">{esc("前置缓冲")}</text>'
)

L0_W, L0_H = 110, 78
L0_GAP = 16
L0A_X = L1_X + L1_W + 60
L0A_Y = L1_Y - 6
L0B_X = L0A_X
L0B_Y = L0A_Y + L0_H + L0_GAP
L0_labels = [("L0A", "左矩阵", L0A_X, L0A_Y), ("L0B", "右矩阵", L0B_X, L0B_Y)]
for name, role, lx, ly in L0_labels:
    L.append(
        f'<rect x="{lx}" y="{ly}" width="{L0_W}" height="{L0_H}" rx="10" '
        f'fill="{COLORS["l0"]}" stroke="#4338ca" stroke-width="2"/>'
    )
    L.append(
        f'<text x="{lx + L0_W / 2}" y="{ly + 28}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="14" font-weight="bold" fill="#0f172a">{esc(name)}</text>'
    )
    L.append(
        f'<text x="{lx + L0_W / 2}" y="{ly + 48}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" fill="#312e81">{esc(role)}</text>'
    )
    L.append(
        f'<path d="M {L1_X + L1_W} {L1_Y + L1_H / 2} L {lx} {ly + L0_H / 2}" '
        f'fill="none" stroke="#6d28d9" stroke-width="1.8" marker-end="url(#arrowG)"/>'
    )

L0C_X = L0A_X + L0_W + 60
L0C_Y = (L0A_Y + L0B_Y) / 2
L.append(
    f'<rect x="{L0C_X}" y="{L0C_Y}" width="{L0_W}" height="{L0_H}" rx="10" '
    f'fill="{COLORS["l0"]}" stroke="#4338ca" stroke-width="2"/>'
)
L.append(
    f'<text x="{L0C_X + L0_W / 2}" y="{L0C_Y + 28}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="14" font-weight="bold" fill="#0f172a">{esc("L0C")}</text>'
)
L.append(
    f'<text x="{L0C_X + L0_W / 2}" y="{L0C_Y + 48}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#312e81">{esc("累加结果")}</text>'
)
for _, _, lx, ly in L0_labels:
    L.append(
        f'<path d="M {lx + L0_W} {ly + L0_H / 2} L {L0C_X} {L0C_Y + L0_H / 2}" '
        f'fill="none" stroke="#6d28d9" stroke-width="1.8" marker-end="url(#arrowG)"/>'
    )

# GM <-> L1 dashed link (cube path also touches GM, less central to claim)
gm_l1_y = L1_Y + L1_H / 2
L.append(
    f'<path d="M {gm_right_x} {gm_l1_y} L {L1_X} {gm_l1_y}" '
    f'fill="none" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#arrowG)"/>'
)

# paper-attributed footnote for L0
L.append(
    f'<text x="{L1_X}" y="{L0B_Y + L0_H + 26}" font-family="sans-serif" font-size="10" '
    f'fill="#64748b" font-style="italic">'
    f'{esc("L0A/L0B/L0C 精确职责与容量为 HPCA’21 paper-attributed 口径，host 未联网核实，不写死数字")}</text>'
)

# ---- legend ---------------------------------------------------------------
LEGEND = [(COLORS["gm"], "GM（片外 DRAM）"), (COLORS["ub"], "UB（vector 路径，192KB）"),
          (COLORS["l1"], "L1（cube 前置缓冲）"), (COLORS["l0"], "L0A/L0B/L0C（cube 紧邻缓冲）")]
ly0 = CHIP_Y + CHIP_H + 30
for j, (color, label) in enumerate(LEGEND):
    lx = GM_X + j * 290
    L.append(f'<rect x="{lx}" y="{ly0}" width="18" height="18" rx="3" fill="{color}" stroke="#64748b"/>')
    L.append(
        f'<text x="{lx + 26}" y="{ly0 + 14}" font-family="sans-serif" font-size="12" '
        f'fill="#334155">{esc(label)}</text>'
    )

# ---- caption ----------------------------------------------------------------
L.append(
    f'<text x="{GM_X}" y="{ly0 + 44}" font-family="sans-serif" font-size="12" fill="#0f172a">'
    f'{esc("GPU 的 cache 自动帮你把数据端上来；达芬奇必须用 tl.load/tl.store 亲手搬——double-buffer 把可用 UB 从 192KB 砍到 96KB，正是下一节 tiling 成为硬件必然的直接原因。")}</text>'
)

L.append('</svg>')

out = Path(__file__).with_name("davinci-onchip-memory-hierarchy.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
