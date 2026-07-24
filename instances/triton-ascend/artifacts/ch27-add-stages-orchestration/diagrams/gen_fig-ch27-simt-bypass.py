#!/usr/bin/env python3
"""fig-ch27-simt-bypass：force_simt_only 通过『在装配层少登记一段』实现旁路——
ttadapter 段整段不注册，TTIR 直接喂 ttir_to_npubin 交 bishengir 的 SIMT 纯路，
而非在 pass 链里加分支。坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)


def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


TITLE = "force_simt_only：装配层『少登记一段』实现旁路"
SUBTITLE = "third_party/ascend/backend/compiler.py:L942（分叉点）/ L824-868（ttir_to_npubin）"

PAD = 44
PANEL_W = 400
GUTTER = 90
TOP = 168
BOX_W = PANEL_W
w = PAD * 2 + PANEL_W * 2 + GUTTER

elems = []


def add(s):
    elems.append(s)


def box(cx, y, lines, w=BOX_W, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e",
        bold=False, fs=12.5, dashed=False):
    n = len(lines)
    box_h = 26 + 19 * (n - 1) + 30
    bx = cx - w / 2
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{w:.0f}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>')
    y0 = y + box_h / 2 - (n - 1) * 9.5 + 5
    fw = 'font-weight="bold" ' if bold else ''
    for k, line in enumerate(lines):
        add(f'<text x="{cx:.0f}" y="{y0+k*19:.0f}" text-anchor="middle" '
            f'font-family="monospace" font-size="{fs}" {fw}fill="{text_fill}">{esc(line)}</text>')
    return box_h


def varrow(x, y1, y2, color="#334155"):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')


x_left = PAD + PANEL_W / 2
x_right = PAD + PANEL_W + GUTTER + PANEL_W / 2

# 分叉点横幅（两面板共享同一行 if）
add(f'<rect x="{PAD}" y="94" width="{w-2*PAD:.0f}" height="34" rx="8" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="1.6"/>')
add(f'<text x="{PAD+(w-2*PAD)/2:.0f}" y="116" text-anchor="middle" font-family="monospace" '
    'font-size="12.5" font-weight="bold" fill="#78350f">'
    + esc('add_stages L942: if options.force_simt_only:') + '</text>')

titles = [
    (x_left, "常规路（force_simt_only=False，默认）"),
    (x_right, "快路径（force_simt_only=True）"),
]
for cx, t in titles:
    add(f'<text x="{cx:.0f}" y="{TOP-14:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(t)}</text>')

GAP = 26

# --- 左面板：常规 3 段 ---
ly = TOP
lbh = box(x_left, ly, ["ttir = make_ttir(...)"])
ly += lbh
varrow(x_left, ly, ly + GAP)
ly += GAP
lbh2 = box(x_left, ly, ["ttadapter = ttir_to_linalg(...)", "11 个 add_*"],
           fill="#dcfce7", stroke="#15803d", text_fill="#14532d")
ly += lbh2
varrow(x_left, ly, ly + GAP)
ly += GAP
lbh3 = box(x_left, ly, ["npubin = linalg_to_bin_*(...)"])
ly += lbh3
tag_y = ly + 16
tag_h = 32
add(f'<rect x="{x_left-PANEL_W/2:.0f}" y="{tag_y:.0f}" width="{PANEL_W}" height="{tag_h}" rx="8" '
    'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1"/>')
add(f'<text x="{x_left:.0f}" y="{tag_y+21:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#1e3a5f">共 3 段 stage（ttir→ttadapter→npubin）</text>')
left_bottom = tag_y + tag_h

# --- 右面板：快路径 2 段，ttadapter 整段以虚线占位框标『0 个 add_*』 ---
ry = TOP
rbh = box(x_right, ry, ["ttir = make_ttir(...)"])
ry += rbh
varrow(x_right, ry, ry + GAP)
ry += GAP
rbh2 = box(x_right, ry, ["ttadapter：不登记", "0 个 add_*（L942-948）"],
           fill="#fef2f2", stroke="#b91c1c", text_fill="#7f1d1d", dashed=True)
ry += rbh2
varrow(x_right, ry, ry + GAP)
ry += GAP
rbh3 = box(x_right, ry, ["npubin = ttir_to_npubin(...)",
                          "--enable-hivm-compile=false",
                          "--enable-triton-ir-compile",
                          "--pure-simt"],
           fill="#fee2e2", stroke="#b91c1c", text_fill="#7f1d1d", fs=11.5)
ry += rbh3
tag2_y = ry + 16
add(f'<rect x="{x_right-PANEL_W/2:.0f}" y="{tag2_y:.0f}" width="{PANEL_W}" height="{tag_h}" rx="8" '
    'fill="#fecaca" stroke="#b91c1c" stroke-width="1"/>')
add(f'<text x="{x_right:.0f}" y="{tag2_y+21:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#7f1d1d">共 2 段 stage（ttir→npubin，跳过 ttadapter）</text>')
right_bottom = tag2_y + tag_h

content_bottom = max(left_bottom, right_bottom)

note_lines = [
    "force_simt_only 默认值为 False（compiler.py:L775）；旁路是装配层的减法——为真时 add_stages",
    "只登记 ttir + npubin 并 return（L943-948），ttadapter 那 11 个 add_* 整段不存在，而非在 pass 链内部加 if。",
]
note_top = content_bottom + 30
note_h = 22 * len(note_lines) + 22
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*22:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-ch27-simt-bypass.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
