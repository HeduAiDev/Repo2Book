#!/usr/bin/env python3
"""fig-m9-ctrl-word: layout 模板。
每条 SASS 指令后附的 64 位控制字里,bit41-57 编着 ptxas 的调度决策:
stall/yield/wr-barrier/rd-barrier/wait-mask 五个互不重叠的位段。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PAD = 40
TOP = 100

TITLE = "64 位控制字的调度位段:bit41-57 五个互不重叠字段,ptxas 在此写发射时序"
SUBTITLE = "python/triton/tools/disasm.py:L37-L41 —— parseCtrl"

# --- 顶部:64 位全景条,高亮 bit41-57 ---
FULL_BITS = 64
FULL_W = 12  # 每 bit 像素(全景条,仅示位置感)
full_bar_w = FULL_BITS * FULL_W
HI_LO, HI_HI = 41, 58  # [41, 58) 高亮区间(bit57 含,右开到 58)

# --- 下方:bit41-57 放大条,5 个字段 ---
FIELDS = [
    ("stall", 41, 4, "&0xf", "#fecaca", "#b91c1c", "#7f1d1d"),
    ("yield", 45, 1, "&0x1", "#fde68a", "#b45309", "#78350f"),
    ("wr-barrier", 46, 3, "&0x7", "#bbf7d0", "#15803d", "#166534"),
    ("rd-barrier", 49, 3, "&0x7", "#bfdbfe", "#1d4ed8", "#1e3a8a"),
    ("wait-mask", 52, 6, "&0x3f", "#e9d5ff", "#7e22ce", "#581c87"),
]
ZOOM_UNIT = 42  # 放大条:每 bit 像素(设最小宽度地板,保证短字段文字不裁剪)
FIELD_MIN_W = 130
FIELD_W = {name: max(nbits * ZOOM_UNIT, FIELD_MIN_W) for name, _, nbits, *_ in FIELDS}
zoom_bar_w = sum(FIELD_W.values())

w = PAD * 2 + max(full_bar_w, zoom_bar_w, 980)

elems = []


def add(s):
    elems.append(s)


# ---- 全景条 ----
full_x0 = PAD + (w - PAD * 2 - full_bar_w) / 2
add(f'<text x="{PAD:.0f}" y="{TOP-14:.0f}" font-family="sans-serif" font-size="13" '
    f'font-weight="bold" fill="#334155">64 位控制字全景(bit0..63)</text>')
add(f'<rect x="{full_x0:.0f}" y="{TOP:.0f}" width="{full_bar_w}" height="34" '
    'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.5"/>')
hi_x = full_x0 + HI_LO * FULL_W
hi_w = (HI_HI - HI_LO) * FULL_W
add(f'<rect x="{hi_x:.0f}" y="{TOP:.0f}" width="{hi_w:.0f}" height="34" '
    'fill="#fde68a" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{hi_x+hi_w/2:.0f}" y="{TOP+22:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11.5" font-weight="bold" fill="#78350f">bit41-57(调度信息,17 位)</text>')
add(f'<text x="{full_x0:.0f}" y="{TOP+50:.0f}" font-family="sans-serif" font-size="11" '
    f'fill="#94a3b8">bit0</text>')
add(f'<text x="{full_x0+full_bar_w:.0f}" y="{TOP+50:.0f}" text-anchor="end" '
    f'font-family="sans-serif" font-size="11" fill="#94a3b8">bit63</text>')

# 引出线连到放大条
zoom_top = TOP + 100
zoom_x0 = PAD + (w - PAD * 2 - zoom_bar_w) / 2
add(f'<line x1="{hi_x:.0f}" y1="{TOP+34:.0f}" x2="{zoom_x0:.0f}" y2="{zoom_top:.0f}" '
    'stroke="#b45309" stroke-width="1.5" stroke-dasharray="4,3"/>')
add(f'<line x1="{hi_x+hi_w:.0f}" y1="{TOP+34:.0f}" x2="{zoom_x0+zoom_bar_w:.0f}" y2="{zoom_top:.0f}" '
    'stroke="#b45309" stroke-width="1.5" stroke-dasharray="4,3"/>')

# ---- 放大条:5 个字段 ----
add(f'<text x="{PAD:.0f}" y="{zoom_top-14:.0f}" font-family="sans-serif" font-size="13" '
    f'font-weight="bold" fill="#334155">放大 bit41-57:五个互不重叠字段(各自掩码隔离)</text>')
x = zoom_x0
field_x = {}
for name, start, nbits, mask, fill, stroke, textc in FIELDS:
    fw = FIELD_W[name]
    field_x[name] = (x, fw)
    add(f'<rect x="{x:.0f}" y="{zoom_top:.0f}" width="{fw:.0f}" height="56" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    add(f'<text x="{x+fw/2:.0f}" y="{zoom_top+24:.0f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
        f'fill="{textc}">{esc(name)}</text>')
    end_bit = start + nbits - 1
    rng = f"bit{start}" if nbits == 1 else f"bit{start}-{end_bit}"
    add(f'<text x="{x+fw/2:.0f}" y="{zoom_top+42:.0f}" text-anchor="middle" '
        f'font-family="monospace" font-size="10.5" fill="{textc}">{esc(rng)} {esc(mask)}</text>')
    x += fw
zoom_bottom = zoom_top + 56

# ---- 实例解码表 ----
table_top = zoom_bottom + 60
add(f'<text x="{PAD:.0f}" y="{table_top-16:.0f}" font-family="sans-serif" font-size="13" '
    f'font-weight="bold" fill="#334155">实例解码(Triton v3.2.0 实测,target=sm_90)</text>')

COLS = ["SLINE(hex)", "stall", "yield", "wr-bar", "rd-bar", "wait-mask", "格式化"]
ROWS = [
    ["000e220000000800", "1", "1→'-'", "0", "7→'-'", "0→'--'", "--:-:0:-:1"],
    ["002fda000780c0ff", "13(0xd)", "0→'Y'", "7→'-'", "7→'-'", "2→'02'", "02:-:-:Y:d"],
]
col_w = [zoom_bar_w * 0.24, zoom_bar_w * 0.12, zoom_bar_w * 0.13, zoom_bar_w * 0.13,
         zoom_bar_w * 0.13, zoom_bar_w * 0.13, zoom_bar_w * 0.12]
row_h = 32
tx0 = zoom_x0

# 表头
cx = tx0
for c, cw in zip(COLS, col_w):
    add(f'<rect x="{cx:.0f}" y="{table_top:.0f}" width="{cw:.0f}" height="{row_h}" '
        'fill="#e2e8f0" stroke="#94a3b8"/>')
    add(f'<text x="{cx+cw/2:.0f}" y="{table_top+21:.0f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="11" font-weight="bold" '
        f'fill="#334155">{esc(c)}</text>')
    cx += cw
# 数据行
for r, row in enumerate(ROWS):
    ry = table_top + row_h * (r + 1)
    cx = tx0
    for v, cw in zip(row, col_w):
        add(f'<rect x="{cx:.0f}" y="{ry:.0f}" width="{cw:.0f}" height="{row_h}" '
            'fill="#ffffff" stroke="#cbd5e1"/>')
        add(f'<text x="{cx+cw/2:.0f}" y="{ry+21:.0f}" text-anchor="middle" '
            f'font-family="monospace" font-size="11" fill="#0f172a">{esc(v)}</text>')
        cx += cw

table_bottom = table_top + row_h * (len(ROWS) + 1)

note_lines = [
    "起始位 41/45/46/49/52 加各自宽度恰好首尾相接不交叠,一次 (enc>>shift)&mask 解码互不干扰。",
    "实例 0x002fda000780c0ff → 等待 2 号位屏障、让位(Y)、发射后停 13(0xd)拍——SASS 左列 02:-:-:Y:d 的来历。",
]
note_top = table_bottom + 30
note_h = 24 * len(note_lines) + 20
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+24+i*24:.0f}" font-family="sans-serif" '
        f'font-size="12.5" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-m9-ctrl-word.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
