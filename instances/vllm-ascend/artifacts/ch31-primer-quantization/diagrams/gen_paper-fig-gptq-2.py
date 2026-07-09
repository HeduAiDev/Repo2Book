#!/usr/bin/env python3
"""paper-fig-gptq-2 —— 重绘自 arXiv:2210.17323 (GPTQ) Fig.2：
GPTQ 量化过程的空间结构：左图 Hessian 逆矩阵(Cholesky 分解)按对角块访问,
右图权重矩阵按块处理——块内白列在量化、蓝列等本地更新,块外(紫)要等整块
量化完才做一次性全局更新。informational structure 对齐原图(两个面板+对应
箭头),配色/字体套用本书视觉语言,非像素复制。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

_BOLD_BREAK = {"量"}
def btext(s):
    parts, buf = [], ""
    for ch in s:
        if ch in _BOLD_BREAK:
            if buf:
                parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
                buf = ""
            parts.append(f'<tspan font-weight="normal">{esc(ch)}</tspan>')
        else:
            buf += ch
    if buf:
        parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
    return "".join(parts)

TITLE = "GPTQ 按块量化：块内白列在算、蓝列等本地更新；块外要等整块量化完才做一次性全局更新"
SUBTITLE = "重绘自 arXiv:2210.17323 Fig.2：连续几列组成正在处理的块（Cholesky 存好的 Hessian 逆信息）"

# ---- 左面板：Hessian 逆（Cholesky 分解）阶梯网格 ----
N_STEP = 6          # 阶梯数（对角块数量的示意，非真实矩阵维度）
CUR_STEP = 3        # 高亮的“当前块”在第几阶（0-indexed）
SQ = 300            # 左面板方阵边长
STEP = SQ / N_STEP

# ---- 右面板：权重矩阵列（按块分区） ----
COL_W, COL_H, COL_GAP = 40, 300, 3
# 区段：done=已完成块（批量更新过）/ quant=块内已量化 / cur=正在量化的列 /
#       pend=块内待更新(本地) / wait=块外(等待批量更新)
ZONES = (["done"] * 3 + ["quant"] * 2 + ["cur"] * 1 + ["pend"] * 2 + ["wait"] * 4)
N_COL = len(ZONES)
ZONE_COLOR = {
    "done": "#e2e8f0", "quant": "#fcd34d", "cur": "#ffffff",
    "pend": "#93c5fd", "wait": "#ddd6fe",
}
ZONE_STROKE = {
    "done": "#94a3b8", "quant": "#b45309", "cur": "#ea580c",
    "pend": "#1e3a5f", "wait": "#7c3aed",
}
BLOCK_LO = 3  # 当前块在右面板的列区间 [BLOCK_LO, BLOCK_HI)
BLOCK_HI = 8

PAD, TOP = 40, 116
GAP_PANELS = 110
LEFT_W = SQ
RIGHT_W = N_COL * (COL_W + COL_GAP) - COL_GAP
W = PAD * 2 + LEFT_W + GAP_PANELS + RIGHT_W
PANEL_H = max(SQ, COL_H)
CAP_Y1 = TOP + PANEL_H + 34     # 面板下第 1 行说明（左：灰/浅蓝图例；右：block i 标签）
CAP_Y2 = TOP + PANEL_H + 52     # 面板下第 2 行说明（仅右面板有）
LEGEND_Y = TOP + PANEL_H + 96   # 图例行（five 色块）
FOOT_Y1 = LEGEND_Y + 46
FOOT_Y2 = FOOT_Y1 + 20
H = FOOT_Y2 + 20

LX0 = PAD
RX0 = PAD + LEFT_W + GAP_PANELS

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="1" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M10,0 L0,3 L10,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# ===== 左面板：Hessian 逆（阶梯网格）=====
L.append(f'<text x="{LX0+LEFT_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">'
          f'{esc("Hessian 逆（Cholesky 分解，按对角块访问）")}</text>')

for i in range(N_STEP):
    # 阶梯的每一“级”：左侧灰（下三角，已消耗/不再用），右侧浅蓝（上三角，已算好待用）
    step_x = LX0 + i * STEP
    step_y = TOP + i * STEP
    gray_hl = (i == CUR_STEP)
    if not gray_hl:
        # 本级左侧一格：灰色（已消耗）
        L.append(f'<rect x="{step_x}" y="{step_y}" width="{STEP}" height="{SQ-step_y+TOP-0}" '
                  f'fill="none"/>')
    # 灰色下三角（本级正下方到底部整块）
    L.append(f'<rect x="{step_x}" y="{step_y}" width="{STEP}" height="{TOP+SQ-step_y}" '
              f'fill="#94a3b8" opacity="0.55"/>')
    # 浅蓝上三角（本级正上方到顶部整块，覆盖在灰色之上再叠一层浅蓝在右侧）
    L.append(f'<rect x="{step_x}" y="{TOP}" width="{STEP}" height="{step_y-TOP+STEP}" '
              f'fill="#bfdbfe" opacity="0.9"/>')

# 当前块高亮：对角上第 CUR_STEP 级方块，粗边框
cbx = LX0 + CUR_STEP * STEP
cby = TOP + CUR_STEP * STEP
L.append(f'<rect x="{cbx}" y="{cby}" width="{STEP}" height="{STEP}" fill="#60a5fa" '
          f'stroke="#1d4ed8" stroke-width="3"/>')
# 当前块内再切一根白色细列＝正在量化的这一列
cur_col_w = STEP * 0.22
cur_col_x = cbx + STEP / 2 - cur_col_w / 2
L.append(f'<rect x="{cur_col_x}" y="{cby}" width="{cur_col_w}" height="{STEP}" '
          f'fill="white" stroke="#ea580c" stroke-width="2"/>')
L.append(f'<text x="{cbx+STEP/2}" y="{cby+STEP+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
          f'fill="#1d4ed8">{esc("当前块")}</text>')

L.append(f'<rect x="{LX0}" y="{TOP}" width="{SQ}" height="{SQ}" fill="none" '
          f'stroke="#334155" stroke-width="1.3"/>')
L.append(f'<text x="{LX0+8}" y="{CAP_Y1}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("灰＝已消耗(下三角)　浅蓝＝已算好待用(上三角)")}</text>')

# ===== 右面板：权重矩阵按块分区 =====
L.append(f'<text x="{RX0+RIGHT_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">'
          f'{esc("权重矩阵 / 块：block i 按列递归量化")}</text>')

for i, zone in enumerate(ZONES):
    x = RX0 + i * (COL_W + COL_GAP)
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W}" height="{COL_H}" '
              f'fill="{ZONE_COLOR[zone]}" stroke="{ZONE_STROKE[zone]}" stroke-width="1.2"/>')

# 当前块边框（粗边框框住 BLOCK_LO..BLOCK_HI 列）
bx0 = RX0 + BLOCK_LO * (COL_W + COL_GAP)
bx1 = RX0 + (BLOCK_HI - 1) * (COL_W + COL_GAP) + COL_W
L.append(f'<rect x="{bx0-3}" y="{TOP-3}" width="{bx1-bx0+6}" height="{COL_H+6}" '
          f'fill="none" stroke="#1d4ed8" stroke-width="3"/>')
L.append(f'<text x="{(bx0+bx1)/2}" y="{CAP_Y1}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
          f'fill="#1d4ed8">{esc("当前块 (block i)")}</text>')

L.append(f'<text x="{RX0+8}" y="{CAP_Y2}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("紫色区不参与任何本地计算，直到 block i 处理完才被一次性批量更新")}</text>')

# 双向箭头：连接左面板高亮块 与 右面板当前块，表明二者是同一件事
mid_y = TOP + PANEL_H / 2
ax0 = cbx + STEP + 6
ax1 = bx0 - 6
mid_arrow_y = TOP - 40
L.append(f'<line x1="{LX0+SQ+6}" y1="{TOP+SQ/2}" x2="{RX0-6}" y2="{TOP+COL_H/2}" '
          f'stroke="#64748b" stroke-width="2" marker-start="url(#b)" marker-end="url(#a)"/>')
L.append(f'<text x="{(LX0+SQ+RX0)/2}" y="{TOP+PANEL_H/2-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#64748b">{esc("同一个 block")}</text>')

# ===== 图例 =====
LEGEND = [
    ("done", "已完成块（已批量更新）"),
    ("quant", "块内已量化"),
    ("cur", "正在量化的列"),
    ("pend", "块内待更新（本地）"),
    ("wait", "块外（等待批量更新）"),
]
ly = LEGEND_Y
lx = PAD
for key, label in LEGEND:
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" '
              f'fill="{ZONE_COLOR[key]}" stroke="{ZONE_STROKE[key]}" stroke-width="1.3"/>')
    L.append(f'<text x="{lx+22}" y="{ly+13}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 22 + cjk_text_width(label, 11) + 30

# ===== 图注（结论，非画面描述）=====
L.append(f'<text x="{PAD}" y="{FOOT_Y1}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("批＝block 内连续几列一起懒惰批处理；块内按列递归、随时用 Cholesky 存好的逆信息本地补偿；")}</text>')
L.append(f'<text x="{PAD}" y="{FOOT_Y2}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("块外权重完全不动，等整块量化完才一次性做全局更新——这正是懒惰批省访存的来源。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-gptq-2.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
