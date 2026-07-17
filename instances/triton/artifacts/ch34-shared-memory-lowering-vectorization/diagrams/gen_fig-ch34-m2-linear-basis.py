#!/usr/bin/env python3
"""state-table 模板:源码只存 2 的幂行的相位基{1:0,2:8,4:16},任意行的偏移由置位比特
基 XOR 合成,处处等于算术 vec*phase(r)——GF(2) 线性,全 8 行一致。
数据取自 explainer/traces/swizzle_phase.out。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_text_width(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

BASES = {1: 0, 2: 8, 4: 16}  # 源码只存这 3 个 2 的幂行的基
VEC = 8

def bits_of(r):
    return [b for b in (1, 2, 4) if r & b]

def compose(r):
    bs = bits_of(r)
    val = 0
    for b in bs:
        val ^= BASES[b]
    return val, bs

def phase_of(r):
    return (r // 2) % 4  # perPhase=2, maxPhase=4

ROWS = list(range(8))
TABLE = []
for r in ROWS:
    xor_val, bs = compose(r)
    ph = phase_of(r)
    arith = VEC * ph
    bits_str = "^".join(str(BASES[b]) for b in bs) if bs else "0"
    TABLE.append((r, format(r, "03b"), bits_str + f"={xor_val}" if bs else "0", arith, xor_val == arith))

BASIS_ROWS = {1, 2, 4}
DEMO_ROW = 6

TITLE = "相位在 GF(2) 上线性:置位比特基 XOR 合成 = 算术 vec·phase(r)"
SUBTITLE = "源码只存行 1/2/4 这 3 个基{0,8,16}(蓝);行 6=4+2 由 16^8=24 合成(橙),与 vec·phase(6)=8*3=24 相等;全 8 行核对一致"

COLS = ["行 r", "二进制", "置位比特→基 XOR 合成", "算术 vec·phase(r)", "线性一致?"]
COL_W = [70, 90, 210, 170, 110]
LABEL_W = 0
ROW_H, HEADER_H, TOP, PAD = 40, 46, 108, 32
NUM_COLS = len(COLS)
FOOT1 = "行 1/2/4 的列偏移基 = 0/8/16(蓝框);行 6=4+2:XOR 基 16^8=24,算术 vec·phase(6)=8*3=24,吻合(橙框)"
FOOT2 = "全 8 行『线性一致?』均 True(8/8)——这正是相位公式能塞进 LinearLayout 并与寄存器布局 invertAndCompose 复合的前提"
min_w_table = PAD * 2 + sum(COL_W)
min_w_text = PAD * 2 + max(cjk_text_width(SUBTITLE, 12), cjk_text_width(FOOT1, 12), cjk_text_width(FOOT2, 12))
w = max(min_w_table, min_w_text)
h = TOP + HEADER_H + ROW_H * len(TABLE) + PAD + 56

col_x = [PAD + sum(COL_W[:i]) for i in range(NUM_COLS)]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(TABLE))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W[j]-6}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W[j]-6)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, (r, binr, compose_str, arith, ok) in enumerate(TABLE):
    ry = row_y[i]
    is_basis = r in BASIS_ROWS
    is_demo = r == DEMO_ROW
    if is_demo:
        fill, stroke, tcolor = "#fff7ed", "#f97316", "#9a3412"
    elif is_basis:
        fill, stroke, tcolor = "#eff6ff", "#3b82f6", "#1e3a5f"
    else:
        fill, stroke, tcolor = "#f8fafc", "#cbd5e1", "#374151"
    vals = [str(r), binr, compose_str, str(arith), "True" if ok else "False"]
    for j, val in enumerate(vals):
        cx = col_x[j]
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W[j]-6}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if (is_basis or is_demo) else 1}"/>')
        weight = 'font-weight="bold" ' if (is_basis or is_demo) else ''
        fsz = 12 if len(val) <= 10 else 11
        L.append(f'<text x="{cx+(COL_W[j]-6)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="monospace" font-size="{fsz}" fill="{tcolor}" '
                  f'{weight}>{esc(val)}</text>')

foot_y = row_y[-1] + ROW_H + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc(FOOT1)}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc(FOOT2)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch34-m2-linear-basis.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
