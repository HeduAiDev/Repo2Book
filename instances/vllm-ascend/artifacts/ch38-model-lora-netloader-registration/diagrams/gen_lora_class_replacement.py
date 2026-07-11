#!/usr/bin/env python3
"""ch30 §30.2 全局类替换 trick：refresh_all_lora_classes() 把
_all_lora_classes 从 16 项内置类 splat 追加成 20 项（+4 昇腾类，追加于尾部），
from_layer 顺序遍历在第 17 项用严格类型相等命中昇腾 QKV 层——前 16 项判定逐字不变。
上半：before/after 元组对比（tick 网格表示项数，4 个昇腾类高亮）。
下半：from_layer 对 20 项元组的顺序遍历（前 16 步全 False、第 17 步命中）。
数字来自 traces/lora_class_replacement.json：16 / 20 / 4 / 17 / 1。
风格对齐同章 fig30-1-extension-points / netloader-flow。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# palette（与同章其它图一致）
R_FILL, R_STROKE, R_TC = "#f1f5f9", "#94a3b8", "#475569"   # 灰：内置类/未命中
P_FILL, P_STROKE, P_TC = "#f3e8ff", "#7c3aed", "#5b21b6"   # 紫：昇腾类/命中
G_FILL, G_STROKE, G_TC = "#dcfce7", "#16a34a", "#166534"   # 绿：结果

W, H = 1040, 630
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '<marker id="ap" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# title
L.append(f'<text x="{W/2}" y="42" text-anchor="middle" font-family="sans-serif" '
         f'font-size="26" font-weight="bold" fill="#1e293b">'
         f'{esc("全局类替换 trick：一行 splat 把候选池从 16 项扩到 20 项")}</text>')
L.append(f'<text x="{W/2}" y="70" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="#64748b">'
         f'{esc("追加到尾部 + 严格类型相等 → from_layer 前 16 项判定逐字不变，只在第 17 项命中")}</text>')


def rbox(x, y, w, h, fill, stroke, tc, lines, fs=15, mono=False, bold=True, rx=10):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    ff = "monospace" if mono else "sans-serif"
    fw = "bold" if bold else "normal"
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * (fs + 5) / 2 + fs / 2 - 2
    for i, ln in enumerate(lines):
        L.append(f'<text x="{x+w/2}" y="{y0+i*(fs+5)}" text-anchor="middle" '
                 f'font-family="{ff}" font-size="{fs}" font-weight="{fw}" '
                 f'fill="{tc}">{esc(ln)}</text>')


def arrow(x1, y1, x2, y2, marker="ag", stroke="#94a3b8", w=2):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
             f'stroke-width="{w}" marker-end="url(#{marker})"/>')


def alabel(x, y, lines, fill="#475569", fs=13):
    n = len(lines)
    y0 = y - (n - 1) * (fs + 3) / 2
    for i, ln in enumerate(lines):
        L.append(f'<text x="{x}" y="{y0+i*(fs+3)}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" font-weight="bold" '
                 f'fill="{fill}">{esc(ln)}</text>')


# ============ 上半：before / after 元组对比 ============
TOP_Y = 108
PANEL_H = 200
TICK = 26
TICK_GAP = 6

# ---- 左：vLLM 原始元组（16 项内置类）----
LX, LW = 40, 260
L.append(f'<rect x="{LX}" y="{TOP_Y}" width="{LW}" height="{PANEL_H}" rx="12" '
         f'fill="white" stroke="{R_STROKE}" stroke-width="1.5" stroke-dasharray="5,4"/>')
L.append(f'<text x="{LX+LW/2}" y="{TOP_Y+26}" text-anchor="middle" font-family="monospace" '
         f'font-size="14" font-weight="bold" fill="#334155">{esc("vLLM 原 _all_lora_classes")}</text>')
COLS_L = 8
grid_w_l = COLS_L * TICK + (COLS_L - 1) * TICK_GAP
gx0_l = LX + (LW - grid_w_l) / 2
gy0_l = TOP_Y + 44
for i in range(16):
    r, c = divmod(i, COLS_L)
    x = gx0_l + c * (TICK + TICK_GAP)
    y = gy0_l + r * (TICK + TICK_GAP)
    L.append(f'<rect x="{x}" y="{y}" width="{TICK}" height="{TICK}" rx="4" '
             f'fill="{R_FILL}" stroke="{R_STROKE}" stroke-width="1.5"/>')
L.append(f'<text x="{LX+LW/2}" y="{gy0_l+2*(TICK+TICK_GAP)+26}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="16" font-weight="bold" fill="{R_TC}">'
         f'{esc("16 项内置类")}</text>')
L.append(f'<text x="{LX+LW/2}" y="{gy0_l+2*(TICK+TICK_GAP)+46}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" fill="#64748b">'
         f'{esc("QKVParallelLinearWithLoRA 等")}</text>')

# ---- 中：refresh_all_lora_classes() 箭头 ----
AX1, AX2 = LX + LW + 14, LX + LW + 14 + 230
AMID_Y = TOP_Y + PANEL_H / 2
L.append(f'<text x="{(AX1+AX2)/2}" y="{AMID_Y-34}" text-anchor="middle" font-family="monospace" '
         f'font-size="13" font-weight="bold" fill="{P_TC}">{esc("refresh_all_lora_classes()")}</text>')
alabel((AX1 + AX2) / 2, AMID_Y + 30, ["元组 splat 追加 4 项", "（tuple(*old, *new)）"], fill=P_TC, fs=12)
arrow(AX1, AMID_Y, AX2, AMID_Y, marker="ap", stroke="#7c3aed", w=2.5)

# ---- 右：刷新后元组（20 项 = 16 不变 + 4 昇腾）----
RX, RW = AX2 + 14, 340
L.append(f'<rect x="{RX}" y="{TOP_Y}" width="{RW}" height="{PANEL_H}" rx="12" '
         f'fill="white" stroke="{P_STROKE}" stroke-width="1.5"/>')
L.append(f'<text x="{RX+RW/2}" y="{TOP_Y+26}" text-anchor="middle" font-family="monospace" '
         f'font-size="14" font-weight="bold" fill="#334155">{esc("刷新后：新元组 20 项")}</text>')
COLS_R = 10
grid_w_r = COLS_R * TICK + (COLS_R - 1) * TICK_GAP
gx0_r = RX + (RW - grid_w_r) / 2
gy0_r = TOP_Y + 44
for i in range(20):
    r, c = divmod(i, COLS_R)
    x = gx0_r + c * (TICK + TICK_GAP)
    y = gy0_r + r * (TICK + TICK_GAP)
    ascend = i >= 16   # 后 4 项 = 追加在尾部的昇腾类
    fill = P_FILL if ascend else R_FILL
    stroke = P_STROKE if ascend else R_STROKE
    L.append(f'<rect x="{x}" y="{y}" width="{TICK}" height="{TICK}" rx="4" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if ascend else 1.5}"/>')
L.append(f'<text x="{RX+RW/2}" y="{gy0_r+2*(TICK+TICK_GAP)+26}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="16" font-weight="bold" fill="#334155">'
         f'{esc("20 项 = 16 不变 + 4 昇腾（紫）")}</text>')
L.append(f'<text x="{RX+RW/2}" y="{gy0_r+2*(TICK+TICK_GAP)+46}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" fill="#64748b">'
         f'{esc("4 项追加于尾部，前 16 项对象与秩序不变")}</text>')

# ============ 过渡：连到下半 from_layer 遍历 ============
ARROW_Y1 = TOP_Y + PANEL_H + 6
ARROW_Y2 = ARROW_Y1 + 26
L.append(f'<line x1="{W/2}" y1="{ARROW_Y1}" x2="{W/2}" y2="{ARROW_Y2}" '
         f'stroke="#94a3b8" stroke-width="2" marker-end="url(#ag)"/>')
alabel(W / 2, ARROW_Y2 + 22, ["from_layer 顺序遍历这 20 项元组"], fill="#334155", fs=15)

SRC_Y = ARROW_Y2 + 40
SRC_W = 620
rbox((W - SRC_W) / 2, SRC_Y, SRC_W, 40, "#eef2ff", "#6366f1", "#3730a3",
     ["源层 = AscendQKVParallelLinear（packed_modules_list 长度 == 1）"], fs=14, mono=True)

# ============ 下半：三段遍历流 ============
ROW_Y = SRC_Y + 40 + 40
BOX_H = 130
GAP = 58

boxA_w = 268
boxB_w = 320
boxC_w = 244
total_w = boxA_w + GAP + boxB_w + GAP + boxC_w
start_x = (W - total_w) / 2

xA = start_x
xB = xA + boxA_w + GAP
xC = xB + boxB_w + GAP

rbox(xA, ROW_Y, boxA_w, BOX_H, R_FILL, R_STROKE, R_TC,
     ["第 1–16 项", "逐个 can_replace_layer(源层)", "→ False", "（内置类全不认 Ascend 类型）"], fs=14)
rbox(xB, ROW_Y, boxB_w, BOX_H, P_FILL, P_STROKE, P_TC,
     ["第 17 项", "AscendQKVParallelLinearWithLoRA", "type(源层) is AscendQKVParallelLinear", "and len == 1  →  True"], fs=14, mono=False)
rbox(xC, ROW_Y, boxC_w, BOX_H, G_FILL, G_STROKE, G_TC,
     ["结果", "包成对应 LoRA 层", "LoRA 生效"], fs=15)

cyA = ROW_Y + BOX_H / 2
arrow(xA + boxA_w, cyA, xB - 4, cyA, marker="ag", stroke="#94a3b8", w=2)
alabel((xA + boxA_w + xB - 4) / 2, cyA - 16, ["全 False"], fill="#475569", fs=12)
arrow(xB + boxB_w, cyA, xC - 4, cyA, marker="ap", stroke="#7c3aed", w=2.5)
alabel((xB + boxB_w + xC - 4) / 2, cyA - 16, ["命中"], fill=P_TC, fs=12)

L.append('</svg>')
svg = '\n'.join(L)
with open('ch30-lora-class-replacement.svg', 'w', encoding='utf-8') as f:
    f.write(svg)
print("wrote ch30-lora-class-replacement.svg", W, H)
