#!/usr/bin/env python3
"""figure_id: davinci-ai-core-structure
claim: 一个达芬奇 AI Core = 1 个 cube(矩阵)核 + 2 个 vector(向量)核 + scalar,
cube:vector 物理配比恒为 1:2(各单元的片上缓冲详见下一张片上内存层级图,本图不重复断言)。
template: layout
numbers (spec.numbers, 逐条来源见 explainer.json):
  - cube 核数/AI Core = 1                 (programming_guide.md:16)
  - vector 核数/AI Core = 2               (programming_guide.md:16)
  - cube:vector 配比 = 1:2                (programming_guide.md:14)
  - 脉动阵列维度(FP16 MAC) = 16x16x16     (HPCA'21 paper-attributed, 未联网核实 —— 软性标注)
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


W, H = 1180, 620

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs><marker id="arrow" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="8" '
    'markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#475569"/></marker></defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---- title -----------------------------------------------------------
L.append(
    f'<text x="40" y="44" font-family="sans-serif" font-size="20" font-weight="bold" '
    f'fill="#0f172a">{esc("达芬奇 AI Core 结构：1 个 cube + 2 个 vector + scalar")}</text>'
)
L.append(
    f'<text x="40" y="70" font-family="sans-serif" font-size="13" '
    f'fill="#475569">{esc("cube : vector 物理配比恒为 1 : 2（source-cited，非可调参数）——mix_mode / CV 融合都悬在这张分工图上")}</text>'
)

# ---- outer AI Core boundary -------------------------------------------
CORE_X, CORE_Y, CORE_W, CORE_H = 60, 100, W - 120, 400
L.append(
    f'<rect x="{CORE_X}" y="{CORE_Y}" width="{CORE_W}" height="{CORE_H}" rx="14" '
    f'fill="#f8fafc" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6,4"/>'
)
L.append(
    f'<text x="{CORE_X + 16}" y="{CORE_Y + 28}" font-family="sans-serif" font-size="15" '
    f'font-weight="bold" fill="#334155">{esc("一个 AI Core（computing core）")}</text>'
)

COLORS = {"cube": "#93c5fd", "vector": "#86efac", "scalar": "#e2e8f0"}

# ---- scalar unit (small, top-right of the core, controls the others) --
SC_W, SC_H = 200, 70
SC_X = CORE_X + CORE_W - SC_W - 30
SC_Y = CORE_Y + 46
L.append(
    f'<rect x="{SC_X}" y="{SC_Y}" width="{SC_W}" height="{SC_H}" rx="10" '
    f'fill="{COLORS["scalar"]}" stroke="#64748b" stroke-width="1.5"/>'
)
L.append(
    f'<text x="{SC_X + SC_W / 2}" y="{SC_Y + 28}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="14" font-weight="bold" fill="#0f172a">{esc("scalar")}</text>'
)
L.append(
    f'<text x="{SC_X + SC_W / 2}" y="{SC_Y + 48}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#334155">{esc("地址 / 循环 / 调度记账")}</text>'
)

# ---- cube unit (large, left) ------------------------------------------
CU_X, CU_Y, CU_W, CU_H = CORE_X + 30, CORE_Y + 90, 340, 260
L.append(
    f'<rect x="{CU_X}" y="{CU_Y}" width="{CU_W}" height="{CU_H}" rx="12" '
    f'fill="{COLORS["cube"]}" stroke="#1d4ed8" stroke-width="2"/>'
)
L.append(
    f'<text x="{CU_X + CU_W / 2}" y="{CU_Y + 34}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="16" font-weight="bold" fill="#0f172a">{esc("cube 核 × 1")}</text>'
)
L.append(
    f'<text x="{CU_X + CU_W / 2}" y="{CU_Y + 56}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" fill="#1e3a8a">{esc("脉动阵列（矩阵乘累加）")}</text>'
)
# small systolic-array grid motif inside the cube box (soft, not a literal size claim)
grid_n = 6
cell = 20
gx0 = CU_X + (CU_W - grid_n * cell) / 2
gy0 = CU_Y + 78
for r in range(grid_n):
    for c in range(grid_n):
        L.append(
            f'<rect x="{gx0 + c * cell}" y="{gy0 + r * cell}" width="{cell - 2}" height="{cell - 2}" '
            f'fill="#bfdbfe" stroke="#3b82f6" stroke-width="0.6"/>'
        )
L.append(
    f'<text x="{CU_X + CU_W / 2}" y="{gy0 + grid_n * cell + 22}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="10" fill="#64748b" font-style="italic">'
    f'{esc("脉动阵列维度（FP16 MAC）：16×16×16")}</text>'
)
L.append(
    f'<text x="{CU_X + CU_W / 2}" y="{gy0 + grid_n * cell + 38}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="10" fill="#64748b" font-style="italic">'
    f'{esc("约定俗成口径 · paper-attributed · 待联网核实，非本书断言")}</text>'
)

# ---- two vector units (right column) -----------------------------------
VU_W, VU_H = 320, 110
VU_X = CORE_X + CORE_W - VU_W - 30
VU_Y1 = SC_Y + SC_H + 24
VU_Y2 = VU_Y1 + VU_H + 24
for idx, vy in enumerate((VU_Y1, VU_Y2), start=1):
    L.append(
        f'<rect x="{VU_X}" y="{vy}" width="{VU_W}" height="{VU_H}" rx="12" '
        f'fill="{COLORS["vector"]}" stroke="#15803d" stroke-width="2"/>'
    )
    L.append(
        f'<text x="{VU_X + VU_W / 2}" y="{vy + 32}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="15" font-weight="bold" fill="#0f172a">{esc(f"vector 核 #{idx}")}</text>'
    )
    L.append(
        f'<text x="{VU_X + VU_W / 2}" y="{vy + 56}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" fill="#14532d">{esc("逐元素 / 归约计算")}</text>'
    )

# ---- scalar controls arrows down to cube and to vector column ----------
sc_bottom_x = SC_X + SC_W / 2
sc_bottom_y = SC_Y + SC_H
cube_top_x = CU_X + CU_W / 2
cube_top_y = CU_Y
L.append(
    f'<path d="M {sc_bottom_x - 40} {sc_bottom_y} L {cube_top_x} {cube_top_y}" '
    f'fill="none" stroke="#64748b" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#arrow)"/>'
)
L.append(
    f'<path d="M {SC_X + 20} {sc_bottom_y} L {VU_X + VU_W / 2} {VU_Y1}" '
    f'fill="none" stroke="#64748b" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#arrow)"/>'
)
L.append(
    f'<text x="{(sc_bottom_x + cube_top_x) / 2 - 90}" y="{(sc_bottom_y + cube_top_y) / 2 + 10}" '
    f'font-family="sans-serif" font-size="10" fill="#64748b">{esc("调度/控制（示意）")}</text>'
)

# ---- 1:2 ratio brace/annotation between cube and vector column ---------
ratio_x = (CU_X + CU_W + VU_X) / 2
ratio_y = CU_Y + CU_H / 2
L.append(
    f'<rect x="{ratio_x - 46}" y="{ratio_y - 22}" width="92" height="44" rx="10" '
    f'fill="#fef3c7" stroke="#d97706" stroke-width="1.6"/>'
)
L.append(
    f'<text x="{ratio_x}" y="{ratio_y - 2}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="16" font-weight="bold" fill="#92400e">{esc("1 : 2")}</text>'
)
L.append(
    f'<text x="{ratio_x}" y="{ratio_y + 16}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="10" fill="#92400e">{esc("cube : vector")}</text>'
)
L.append(
    f'<path d="M {CU_X + CU_W} {CU_Y + 40} L {ratio_x - 46} {ratio_y - 10}" '
    f'fill="none" stroke="#d97706" stroke-width="1.3"/>'
)
L.append(
    f'<path d="M {ratio_x + 46} {ratio_y - 10} L {VU_X} {VU_Y1 + 30}" '
    f'fill="none" stroke="#d97706" stroke-width="1.3"/>'
)
L.append(
    f'<path d="M {ratio_x + 46} {ratio_y + 10} L {VU_X} {VU_Y2 + 30}" '
    f'fill="none" stroke="#d97706" stroke-width="1.3"/>'
)

# ---- legend --------------------------------------------------------------
LEGEND = [("cube", "#93c5fd", "cube 核（矩阵乘）"), ("vector", "#86efac", "vector 核（向量计算）"),
          ("scalar", "#e2e8f0", "scalar（地址/调度）")]
ly = CORE_Y + CORE_H + 40
for j, (_, color, label) in enumerate(LEGEND):
    lx = CORE_X + j * 300
    L.append(f'<rect x="{lx}" y="{ly}" width="18" height="18" rx="3" fill="{color}" stroke="#64748b"/>')
    L.append(
        f'<text x="{lx + 26}" y="{ly + 14}" font-family="sans-serif" font-size="13" '
        f'fill="#334155">{esc(label)}</text>'
    )

# ---- caption --------------------------------------------------------------
L.append(
    f'<text x="{CORE_X}" y="{ly + 46}" font-family="sans-serif" font-size="12" '
    f'fill="#0f172a">{esc("每 AI Core 恒为 1 cube + 2 vector；GPU 的 SM 是通用工人堆叠，达芬奇是分工到人、各有各的工具台（片上缓冲详见下一张图）。")}</text>'
)

L.append('</svg>')

out = Path(__file__).with_name("davinci-ai-core-structure.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
