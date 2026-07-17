#!/usr/bin/env python3
"""fig-mma-operand-assembly (layout 模板,三段纵向:寄存器装配 / kWidth 拆分 / opIdx 重排序)
callMmaAmpere 按 (b,m,n,k) 从 ValueTableV2 凑 4A+2B+numMmaRets(C) 寄存器拼一条 mma.sync;
kWidth=8 时因单条 mma 操作数装不下,沿 K 方向 stride-4 拆成 4 个物理 mma。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PAD = 40
w = 1180

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 1">',  # h 占位,最后替换
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>']

y = 30
L.append(f'<text x="{PAD}" y="{y}" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("callMmaAmpere:从 ValueTableV2 按 (b,m,n,k) 凑寄存器,拼一条 mma.sync.m16n8k16")}</text>')
y += 22
L.append(f'<text x="{PAD}" y="{y}" font-family="sans-serif" font-size="12" fill="#64748b">'
          f'{esc("寄存器如何铺满 fragment 见第 27 章;这里只看『一次 mma 调用要凑齐哪些寄存器』")}</text>')

# ---------- Section 1: 寄存器装配 ----------
sec1_y = y + 30
L.append(f'<text x="{PAD}" y="{sec1_y}" font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#1e3a5f">{esc("① 寄存器装配")}</text>')
row_top = sec1_y + 22
CELL_W, CELL_H, CELL_GAP = 168, 58, 14

A_REGS = ["ha[{b,m,k}]", "ha[{b,m+1,k}]", "ha[{b,m,k+1}]", "ha[{b,m+1,k+1}]"]
B_REGS = ["hb[{b,n,k}]", "hb[{b,n,k+1}]"]

a_x0 = PAD
for i, lbl in enumerate(A_REGS):
    x = a_x0 + i * (CELL_W + CELL_GAP)
    L.append(f'<rect x="{x}" y="{row_top}" width="{CELL_W}" height="{CELL_H}" rx="8" '
              'fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CELL_W/2}" y="{row_top+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="#1e3a8a">{esc(f"A 片段 {i}")}</text>')
    L.append(f'<text x="{x+CELL_W/2}" y="{row_top+42}" text-anchor="middle" font-family="monospace" '
              f'font-size="11.5" fill="#1e40af">{esc(lbl)}</text>')

b_row_top = row_top + CELL_H + 20
b_x0 = a_x0 + (4 * (CELL_W + CELL_GAP) - 2 * (CELL_W + CELL_GAP)) / 2  # 居中对齐 4 格宽度
for i, lbl in enumerate(B_REGS):
    x = b_x0 + i * (CELL_W + CELL_GAP)
    L.append(f'<rect x="{x}" y="{b_row_top}" width="{CELL_W}" height="{CELL_H}" rx="8" '
              'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CELL_W/2}" y="{b_row_top+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="#92400e">{esc(f"B 片段 {i}")}</text>')
    L.append(f'<text x="{x+CELL_W/2}" y="{b_row_top+42}" text-anchor="middle" font-family="monospace" '
              f'font-size="11.5" fill="#92400e">{esc(lbl)}</text>')

# C 累加器(单格,标注复用)
c_row_top = b_row_top + CELL_H + 20
c_x = a_x0
C_W = CELL_W * 2 + CELL_GAP
L.append(f'<rect x="{c_x}" y="{c_row_top}" width="{C_W}" height="{CELL_H}" rx="8" '
          'fill="#dcfce7" stroke="#059669" stroke-width="1.5"/>')
L.append(f'<text x="{c_x+C_W/2}" y="{c_row_top+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#065f46">{esc("C 累加器:numMmaRets 个")}</text>')
L.append(f'<text x="{c_x+C_W/2}" y="{c_row_top+42}" text-anchor="middle" font-family="monospace" '
          f'font-size="11.5" fill="#065f46">{esc("std::to_string(i) 输入约束 = 输出约束(原地复用)")}</text>')

# 汇入 mma 指令框(右侧,纵向居中于三行)
mma_x = a_x0 + 4 * (CELL_W + CELL_GAP) + 30
mma_y = row_top
mma_w, mma_h = 260, c_row_top + CELL_H - row_top
L.append(f'<rect x="{mma_x}" y="{mma_y}" width="{mma_w}" height="{mma_h}" rx="10" '
          'fill="#ede9fe" stroke="#6d28d9" stroke-width="2"/>')
L.append(f'<text x="{mma_x+mma_w/2}" y="{mma_y+mma_h/2-8}" text-anchor="middle" font-family="monospace" '
          f'font-size="14" font-weight="bold" fill="#5b21b6">{esc("mma.sync.m16n8k16")}</text>')
L.append(f'<text x="{mma_x+mma_w/2}" y="{mma_y+mma_h/2+14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#5b21b6">{esc("4A + 2B + numMmaRets(C) → 1 条")}</text>')

# 三行箭头汇入
for (bx0, by, bw, stroke) in [
    (a_x0, row_top, 4 * (CELL_W + CELL_GAP) - CELL_GAP, "#2563eb"),
    (b_x0, b_row_top, 2 * (CELL_W + CELL_GAP) - CELL_GAP, "#d97706"),
    (c_x, c_row_top, C_W, "#059669"),
]:
    src_x = bx0 + bw
    src_y = by + CELL_H / 2
    L.append(f'<line x1="{src_x}" y1="{src_y}" x2="{mma_x}" y2="{mma_y+mma_h/2}" '
              f'stroke="{stroke}" stroke-width="1.4" marker-end="url(#a)" opacity="0.75"/>')

sec1_bottom = c_row_top + CELL_H

# ---------- Section 2: kWidth 拆分 ----------
sec2_y = sec1_bottom + 46
L.append(f'<text x="{PAD}" y="{sec2_y}" font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#1e3a5f">{esc("② kWidth=8 拆分:一次逻辑 mma → 4 个 stride-4 物理 mma")}</text>')
strip_y = sec2_y + 20
STRIP_W = w - PAD * 2 - 320
STRIP_H = 46
L.append(f'<rect x="{PAD}" y="{strip_y}" width="{STRIP_W}" height="{STRIP_H}" rx="6" '
          'fill="#f1f5f9" stroke="#64748b" stroke-width="1.3"/>')
seg_w = STRIP_W / 4
SEG_COLORS = ["#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6"]
for t in range(4):
    sx = PAD + t * seg_w
    L.append(f'<rect x="{sx}" y="{strip_y}" width="{seg_w}" height="{STRIP_H}" '
              f'fill="{SEG_COLORS[t]}" stroke="#64748b" stroke-width="1"/>')
    L.append(f'<text x="{sx+seg_w/2}" y="{strip_y+STRIP_H/2+5}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" font-weight="bold" fill="#0f172a">'
              f'{esc(f"mma #{t} (K+{t*4})")}</text>')
L.append(f'<text x="{PAD}" y="{strip_y+STRIP_H+18}" font-family="sans-serif" font-size="11" '
          f'fill="#334155">{esc("K 方向 stride 4 独立寻址,4 条物理 mma 覆盖 1 次逻辑 mma 的 K=8")}</text>')

# 拆分动机 note(右侧)
note_x = PAD + STRIP_W + 30
note_w = w - note_x - PAD
L.append(f'<rect x="{note_x}" y="{strip_y}" width="{note_w}" height="{STRIP_H+18}" rx="6" '
          'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.2"/>')
L.append(f'<text x="{note_x+12}" y="{strip_y+20}" font-family="sans-serif" font-size="11" '
          f'fill="#7f1d1d">{esc("拆分动机:")}</text>')
L.append(f'<text x="{note_x+12}" y="{strip_y+38}" font-family="monospace" font-size="11" '
          f'font-weight="bold" fill="#7f1d1d">{esc("kWidth·elemBitWidth")}</text>')
L.append(f'<text x="{note_x+12}" y="{strip_y+56}" font-family="monospace" font-size="11" '
          f'font-weight="bold" fill="#7f1d1d">{esc("= 8·16 = 128bit > 单 mma 操作数寄存器容量")}</text>')

sec2_bottom = strip_y + STRIP_H + 28

# ---------- Section 3: opIdx 重排序 ----------
sec3_y = sec2_bottom + 38
L.append(f'<text x="{PAD}" y="{sec3_y}" font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#1e3a5f">{esc("③ opIdx 重排序序列(把 dot 语义顺序映射到 mma 硬件期望顺序)")}</text>')

SEQ0 = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
SEQ1 = [0, 4, 1, 5, 2, 6, 3, 7]

def draw_seq(label, seq, top, color_fill, color_stroke):
    L.append(f'<text x="{PAD}" y="{top+14}" font-family="sans-serif" font-size="12" '
              f'font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    cell = 40
    x0 = PAD + 210
    for i, v in enumerate(seq):
        x = x0 + i * (cell + 3)
        L.append(f'<rect x="{x}" y="{top}" width="{cell}" height="26" rx="4" '
                  f'fill="{color_fill}" stroke="{color_stroke}" stroke-width="1"/>')
        L.append(f'<text x="{x+cell/2}" y="{top+18}" text-anchor="middle" font-family="monospace" '
                  f'font-size="12" fill="{color_stroke}">{esc(str(v))}</text>')
    return top + 26

y3a = sec3_y + 16
y3a_end = draw_seq("opIdx=0  si =", SEQ0, y3a, "#e0e7ff", "#4338ca")
y3b = y3a_end + 18
y3b_end = draw_seq("opIdx=1  si =", SEQ1, y3b, "#fef3c7", "#b45309")

foot_y = y3b_end + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" fill="#334155">'
          f'{esc("mma.sync 是固定操作数形状的硬件指令:降级的活就是按索引对号入座喂寄存器;")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="12.5" fill="#334155">'
          f'{esc("kWidth=8 装不下就沿 K 方向 stride-4 拆 4 条,Tensor Core 复用率不降。")}</text>')

h = foot_y + 46
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
L.insert(2, f'<rect width="{w}" height="{h}" fill="white"/>')
L.append('</svg>')

out = Path(__file__).with_name("fig-mma-operand-assembly.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  {w}x{h}")
