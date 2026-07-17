#!/usr/bin/env python3
"""flow 模板(自定义):distributed 布局四级计算层级 + shape/order 线性 id 铺法示例。
左:CTAsPerCGA->WarpsPerCTA->ThreadsPerWarp->ValuesPerThread 纵向链,顶两级用花括号
标注"共享:按 shape+order 连续铺线性 id",底两级标注"各编码自定义"。
右:该共享铺法的具体示例——shape=[4,4], order=[0,1] 的 4x4 线性 id 网格,
高亮首列(沿 dim0 连续)与首行(沿 dim1 跨 4)。
全坐标由循环/常量计算,零手写魔数;宽度按文本长度估算留够,不裁字。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    # 粗估:CJK 字符宽约 size*1.0,ASCII 约 size*0.56
    cjk = sum(1 for ch in s if ord(ch) > 0x2e7f)
    other = len(s) - cjk
    return cjk * size * 1.0 + other * size * 0.56

TITLE = "distributed 布局的四级计算层级"
LEVELS = ["CTAsPerCGA", "WarpsPerCTA", "ThreadsPerWarp", "ValuesPerThread"]
SHARED_LABEL = ["共享:按 shape+order", "连续铺线性 id"]
OWN_LABEL = ["各编码自定义", "(Blocked/Slice/Mma/", "DotOperand 分道)"]
GRID_TITLE = "共享铺法示例:shape=[4,4], order=[0,1]"
GRID_SUB = "order[0]=0:dim0 最快变化——沿 dim0 连续编号"
CAPTION1 = "四级从粗到细一次算定归属;顶两级(CTAsPerCGA/WarpsPerCTA)共享按 shape+order 连续铺 id 的规则(右图),"
CAPTION2 = "底两级(ThreadsPerWarp/ValuesPerThread)才由 Blocked/Slice/Mma/DotOperand 各自定义分发方式。"

# --- 左侧链条几何 ---
CHAIN_W, CHAIN_H, CHAIN_GAP, PAD, TOP = 190, 46, 30, 44, 96
chain_x = PAD
chain_y = [TOP + i * (CHAIN_H + CHAIN_GAP) for i in range(len(LEVELS))]
CHAIN_COLORS = ["#3b82f6", "#3b82f6", "#f59e0b", "#f59e0b"]  # 上两级蓝(共享),下两级橙(自定义)

brace_x = chain_x + CHAIN_W + 14
label_w = max(text_w(s, 12) for s in SHARED_LABEL + OWN_LABEL)
grid_x0 = brace_x + 30 + label_w + 50

# --- 右侧 4x4 网格几何 ---
GRID_N = 4
CELL = 50
grid_y0 = TOP + 70
grid_w = GRID_N * CELL
grid_h = GRID_N * CELL

right_text_w = max(text_w(GRID_TITLE, 13), text_w(GRID_SUB, 11), grid_w)
W = int(grid_x0 + right_text_w + PAD + 30)
chain_block_h = len(LEVELS) * (CHAIN_H + CHAIN_GAP) - CHAIN_GAP
row_arrow_y = grid_y0 + grid_h + 30
foot_h = 46
H = int(max(TOP + chain_block_h, row_arrow_y + 20) + foot_h + PAD)
cap_w = max(text_w(CAPTION1, 12), text_w(CAPTION2, 12))
W = int(max(W, PAD + cap_w + 10))

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
     '<marker id="c" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-16}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>']

# 左侧链条框 + 箭头
for i, name in enumerate(LEVELS):
    y = chain_y[i]
    color = CHAIN_COLORS[i]
    L.append(f'<rect x="{chain_x}" y="{y}" width="{CHAIN_W}" height="{CHAIN_H}" rx="8" '
              f'fill="white" stroke="{color}" stroke-width="2.5"/>')
    L.append(f'<text x="{chain_x+CHAIN_W/2}" y="{y+CHAIN_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="{color}">{esc(name)}</text>')
    if i < len(LEVELS) - 1:
        ax = chain_x + CHAIN_W / 2
        L.append(f'<line x1="{ax}" y1="{y+CHAIN_H}" x2="{ax}" y2="{chain_y[i+1]}" '
                  'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

# 花括号分组(用三次贝塞尔拼一个大括号形状)
def brace(x, y_top, y_bot, color):
    mid = (y_top + y_bot) / 2
    return (f'<path d="M {x} {y_top} '
            f'C {x+10} {y_top} {x+10} {mid-14} {x+18} {mid} '
            f'C {x+10} {mid+14} {x+10} {y_bot} {x} {y_bot}" '
            f'fill="none" stroke="{color}" stroke-width="2"/>')

top_group_top, top_group_bot = chain_y[0], chain_y[1] + CHAIN_H
bot_group_top, bot_group_bot = chain_y[2], chain_y[3] + CHAIN_H
L.append(brace(brace_x, top_group_top, top_group_bot, "#1d4ed8"))
mid1 = (top_group_top + top_group_bot) / 2
start1 = mid1 - (len(SHARED_LABEL) - 1) * 8
for k, line in enumerate(SHARED_LABEL):
    L.append(f'<text x="{brace_x+30}" y="{start1+k*16}" '
              f'font-family="sans-serif" font-size="12" fill="#1d4ed8">{esc(line)}</text>')

L.append(brace(brace_x, bot_group_top, bot_group_bot, "#b45309"))
mid2 = (bot_group_top + bot_group_bot) / 2
start2 = mid2 - (len(OWN_LABEL) - 1) * 8
for k, line in enumerate(OWN_LABEL):
    L.append(f'<text x="{brace_x+30}" y="{start2+k*16}" '
              f'font-family="sans-serif" font-size="12" fill="#b45309">{esc(line)}</text>')

# 右侧标题(留在网格正上方,不与任何箭头交叉)
L.append(f'<text x="{grid_x0}" y="{grid_y0-46}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">{esc(GRID_TITLE)}</text>')
L.append(f'<text x="{grid_x0}" y="{grid_y0-26}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(GRID_SUB)}</text>')

# 4x4 网格: id(row=dim0, col=dim1) = row + col*4
for r in range(GRID_N):
    for c in range(GRID_N):
        x = grid_x0 + c * CELL
        y = grid_y0 + r * CELL
        val = r + c * GRID_N
        is_first_col = (c == 0)
        is_first_row = (r == 0)
        if is_first_col and is_first_row:
            fill, stroke = "#93c5fd", "#1d4ed8"
        elif is_first_col:
            fill, stroke = "#dbeafe", "#1d4ed8"
        elif is_first_row:
            fill, stroke = "#fde68a", "#b45309"
        else:
            fill, stroke = "#f8fafc", "#94a3b8"
        L.append(f'<rect x="{x}" y="{y}" width="{CELL-4}" height="{CELL-4}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+(CELL-4)/2}" y="{y+(CELL-4)/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="14" fill="#0f172a">{val}</text>')

# 首列箭头(沿 dim0 连续,竖直向下,放在网格左侧留白区,不进入网格上方标题带)
col0_x = grid_x0 - 22
L.append(f'<line x1="{col0_x}" y1="{grid_y0+6}" x2="{col0_x}" y2="{grid_y0+grid_h-6}" '
          'stroke="#1d4ed8" stroke-width="2" marker-end="url(#b)"/>')
L.append(f'<text x="{col0_x-8}" y="{grid_y0+grid_h/2}" text-anchor="end" '
          f'font-family="sans-serif" font-size="11" fill="#1d4ed8">{esc("首列")}</text>')
L.append(f'<text x="{col0_x-8}" y="{grid_y0+grid_h/2+15}" text-anchor="end" '
          f'font-family="sans-serif" font-size="11" fill="#1d4ed8">{esc("0,1,2,3")}</text>')

# 首行箭头(沿 dim1 跨 4,水平向右),放在网格下方
row0_y = grid_y0 + grid_h + 26
L.append(f'<line x1="{grid_x0}" y1="{row0_y}" x2="{grid_x0+grid_w-CELL+4}" y2="{row0_y}" '
          'stroke="#b45309" stroke-width="2" marker-end="url(#c)"/>')
L.append(f'<text x="{grid_x0}" y="{row0_y+16}" font-family="sans-serif" font-size="11" '
          f'fill="#b45309">{esc("首行 0,4,8,12(每步跨 4)")}</text>')

foot_y1 = H - foot_h + 6
foot_y2 = foot_y1 + 18
L.append(f'<text x="{PAD}" y="{foot_y1}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(CAPTION1)}</text>')
L.append(f'<text x="{PAD}" y="{foot_y2}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(CAPTION2)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-distributed-hierarchy.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
