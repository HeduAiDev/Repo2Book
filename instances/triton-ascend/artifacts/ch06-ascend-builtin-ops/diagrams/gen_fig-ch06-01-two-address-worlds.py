#!/usr/bin/env python3
"""fig-ch06-01-two-address-worlds — layout 模板。
GM 侧在 Python 层只有裸指针、写不出 space=GM 的 buffer,
所以跨 GM 的带索引访问只能由四个 mem_ops 内建承担。

结构:
① .td 侧 7 档 address space 芯片条(沿用 ch05 的图式)
② pybind 边界:只导出 5 档,Zero/GM 不进 Python
③ 两个世界:左 GM(裸指针,无名字) / 右片上(5 档,有名字)
④ 接缝:4 个 mem_ops 内建横跨两侧,标出各自是否带 index_boundary
⑤ 对照:insert_slice/extract_slice —— 两端都在片上,不过这道缝

全部坐标由常量/循环计算,文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def text_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E7F else 0.58) for ch in s)


def fit(s, maxw, base, floor=9.0):
    size = base
    while size > floor and text_w(s, size) > maxw:
        size -= 0.5
    return size


W, H = 1500, 1080
PAD = 50

TITLE = "两个寻址世界的接缝:GM 只有地址,片上有名字"
SUB1 = ".td 定义 7 档 address space → pybind 只导出 5 档 → Zero 与 GM 根本不进 Python"
SUB2 = "四个 mem_ops 内建就是横跨两侧的唯一带索引通道:左边永远是 GM 裸指针,右边永远是 UB tile"

BLUE, ORANGE, GRAY, RED, GREEN = "#1d4ed8", "#c2410c", "#94a3b8", "#b91c1c", "#15803d"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     f'<marker id="gray" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{GRAY}"/></marker>'
     f'<marker id="blue" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{BLUE}"/></marker>'
     f'<marker id="red" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{RED}"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="19" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{W/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUB1)}</text>',
     f'<text x="{W/2}" y="77" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUB2)}</text>']

# ── ① .td 枚举 7 档:芯片条 ────────────────────────────────────────────
PANEL = (PAD, 94, W - 2 * PAD, 92)
px, py, pw, ph = PANEL
L.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" fill="#fbfbfc" '
         f'stroke="#cbd5e1" stroke-width="1.4"/>')
L.append(f'<text x="{px+16}" y="{py+20}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#475569">'
         f'{esc("① HIVMAttrs.td:L188-194 定义 7 档 address space")}</text>')

CHIP_Y, CHIP_H, CHIP_W, CHIP_DX = 130, 42, 178, 186
CHIP_X0 = 66
ENUM = [("Zero", 0, False), ("GM", 1, False), ("L1", 2, True), ("L0A", 3, True),
        ("L0B", 4, True), ("L0C", 5, True), ("UB", 6, True)]

chip_cx = {}
for i, (name, val, exported) in enumerate(ENUM):
    cx0 = CHIP_X0 + i * CHIP_DX
    ccx = cx0 + CHIP_W / 2
    chip_cx[name] = ccx
    if exported:
        fill, stroke, dash, nfill = "#f1f5f9", "#94a3b8", None, "#334155"
        mark, mfill = "✓ 导出", GREEN
    else:
        fill, stroke, dash, nfill = "#fef2f2", RED, "6,4", "#7f1d1d"
        mark, mfill = "✗ 未导出", RED
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<rect x="{cx0}" y="{CHIP_Y}" width="{CHIP_W}" height="{CHIP_H}" rx="8" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"{d}/>')
    L.append(f'<text x="{ccx}" y="{CHIP_Y+18}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{nfill}">{esc(f"{name} = {val}")}</text>')
    L.append(f'<text x="{ccx}" y="{CHIP_Y+34}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="9.5" font-weight="bold" fill="{mfill}">{esc(mark)}</text>')

# ── ② pybind 边界线 ────────────────────────────────────────────────────
BOUND_Y = 222
L.append(f'<text x="{W/2}" y="{BOUND_Y-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#0f172a">'
         f'{esc("② ascend_ir.cc:L412-417 的 pybind 边界:py::enum_ 只导出 5 档")}</text>')
L.append(f'<line x1="{PAD}" y1="{BOUND_Y}" x2="{W-PAD}" y2="{BOUND_Y}" stroke="#0f172a" '
         f'stroke-width="2.4" stroke-dasharray="12,6"/>')

# ── ③ 两个世界 ─────────────────────────────────────────────────────────
WORLD_Y = 272
L.append(f'<text x="{PAD}" y="{WORLD_Y}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#475569">{esc("③ 两个寻址世界")}</text>')

GM_BOX = (110, 300, 330, 150)
UB_BOX = (1060, 300, 330, 150)

gx, gy, gw, gh = GM_BOX
L.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="12" fill="#fef2f2" '
         f'stroke="{RED}" stroke-width="2.2" stroke-dasharray="8,5"/>')
for i, (txt, fs, bold) in enumerate([
        ("GM 世界(公共货场)", 15, True),
        ("Python 层只有裸指针", 12.5, False),
        ("写不出 space = GM 的 buffer", 12.5, False),
        ("(Zero / GM 不进 Python)", 11, False)]):
    b = ' font-weight="bold"' if bold else ''
    fill = "#7f1d1d" if bold else "#9a3412"
    L.append(f'<text x="{gx+gw/2}" y="{gy+34+i*28}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{fs}"{b} fill="{fill}">{esc(txt)}</text>')

ux, uy, uw, uh = UB_BOX
L.append(f'<rect x="{ux}" y="{uy}" width="{uw}" height="{uh}" rx="12" fill="#eff6ff" '
         f'stroke="{BLUE}" stroke-width="2.2"/>')
for i, (txt, fs, bold) in enumerate([
        ("片上世界(带标签的货架)", 15, True),
        ("UB tile:5 档 address space 之一", 12.5, False),
        ("Python 能直接 bl.alloc / 引用", 12.5, False),
        ("(L1/UB/L0A/L0B/L0C 都有名字)", 11, False)]):
    b = ' font-weight="bold"' if bold else ''
    fill = "#1e3a8a" if bold else "#1e40af"
    L.append(f'<text x="{ux+uw/2}" y="{uy+34+i*28}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{fs}"{b} fill="{fill}">{esc(txt)}</text>')

# ── ④ 接缝:4 个 mem_ops 内建 ──────────────────────────────────────────
BRIDGE_X0, BRIDGE_X1 = gx + gw, ux
BRIDGE_TOP, BRIDGE_ROW_H = 300, 37
BUILTINS = [
    ("index_put", True),
    ("gather_out_to_ub", True),
    ("scatter_ub_to_out", True),
    ("index_select_simd", False),
]
L.append(f'<text x="{(BRIDGE_X0+BRIDGE_X1)/2}" y="{BRIDGE_TOP-14}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#334155">'
         f'{esc("④ 接缝:4 个跨 GM 带索引内建(mem_ops.py:L40-636)")}</text>')
for i, (name, has_boundary) in enumerate(BUILTINS):
    ay = BRIDGE_TOP + 8 + i * BRIDGE_ROW_H
    color = BLUE if has_boundary else RED
    L.append(f'<line x1="{BRIDGE_X0+4}" y1="{ay}" x2="{BRIDGE_X1-4}" y2="{ay}" '
             f'stroke="{color}" stroke-width="2" marker-end="url(#{"blue" if has_boundary else "red"})"/>')
    mid = (BRIDGE_X0 + BRIDGE_X1) / 2
    L.append(f'<text x="{mid}" y="{ay-6}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" font-weight="bold" fill="{color}">{esc(name)}</text>')
    badge = "有 index_boundary" if has_boundary else "✗ 没有 index_boundary(不查越界)"
    bw = text_w(badge, 9.5) + 16
    L.append(f'<rect x="{mid-bw/2:.1f}" y="{ay+4}" width="{bw:.1f}" height="16" rx="8" '
             f'fill="{"#dbeafe" if has_boundary else "#fee2e2"}" stroke="{color}" stroke-width="1"/>')
    L.append(f'<text x="{mid}" y="{ay+15.5}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="9.5" font-weight="bold" fill="{color}">{esc(badge)}</text>')

# ── ⑤ 对照:insert_slice/extract_slice,两端都在片上 ────────────────────
PAIR_Y = 490
PAIR_BOX = (ux, PAIR_Y, uw, 96)
bx, by, bw2, bh2 = PAIR_BOX
L.append(f'<rect x="{bx}" y="{by}" width="{bw2}" height="{bh2}" rx="10" fill="#f8fafc" '
         f'stroke="{GRAY}" stroke-width="1.6"/>')
L.append(f'<text x="{bx+bw2/2}" y="{by+24}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#334155">'
         f'{esc("对照:insert_slice / extract_slice")}</text>')
L.append(f'<text x="{bx+bw2/2}" y="{by+46}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#475569">{esc("两端都在片上,落到上游 tensor 方言")}</text>')
L.append(f'<text x="{bx+bw2/2}" y="{by+66}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#475569">{esc("不碰 GM —— 不过这道缝")}</text>')
# 自环箭头:表示两端都在片上世界内部
loop_cx = bx - 18
L.append(f'<path d="M {bx} {by+18} C {loop_cx-34} {by+18} {loop_cx-34} {by+bh2-18} {bx} {by+bh2-18}" '
         f'fill="none" stroke="{GRAY}" stroke-width="1.8" marker-end="url(#gray)"/>')

# ── 底部脚注(数字出处) ──────────────────────────────────────────────
FOOT = [
    ".td 定义 7 档(Zero=0/GM=1/L1=2/L0A=3/L0B=4/L0C=5/UB=6):third_party/ascend/AscendNPU-IR/"
    "bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td:L188-194。",
    "pybind 只导出 5 档:third_party/ascend/ascend_ir.cc:L412-417 —— Zero 与 GM 不进 Python,"
    "kernel 里写不出 space=GM 的 buffer。",
    "4 个跨 GM 带索引内建:third_party/ascend/language/cann/extension/mem_ops.py:L40-636;"
    "导出清单见 extension/__init__.py:L67-79。",
    "带 index_boundary 的 3 个:mem_ops.py:L40/L180/L332 三处有该形参;index_select_simd(L485-521)没有,"
    "docstring 明写 does not check if index contains out-of-bounds values。",
    "insert_slice / extract_slice:third_party/ascend/language/cann/extension/vec_ops.py:L47-137;"
    "triton_ascend.cc:L52-116。",
]
FOOT_Y0 = 610
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{FOOT_Y0+i*22}" font-family="sans-serif" font-size="11" '
             f'fill="#64748b">{esc(ln)}</text>')

# ── 图例 ──────────────────────────────────────────────────────────────
LEG_Y = FOOT_Y0 + len(FOOT) * 22 + 30
LEG = [
    (BLUE, "带 index_boundary(3 个):越界不算地址,换成 other / 整条丢弃"),
    (RED, "没有 index_boundary(1 个,index_select_simd):越界照常按公式算地址"),
]
for i, (color, label) in enumerate(LEG):
    ly = LEG_Y + i * 26
    L.append(f'<line x1="{PAD}" y1="{ly-4}" x2="{PAD+34}" y2="{ly-4}" stroke="{color}" '
             f'stroke-width="3"/>')
    L.append(f'<text x="{PAD+46}" y="{ly}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(label)}</text>')

H_ACTUAL = LEG_Y + len(LEG) * 26 + 24
L.append('</svg>')
svg = '\n'.join(L)
svg = svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {H_ACTUAL}"')
svg = svg.replace(f'<rect width="{W}" height="{H}" fill="white"/>',
                  f'<rect width="{W}" height="{H_ACTUAL}" fill="white"/>')
out = Path(__file__).with_name('fig-ch06-01-two-address-worlds.svg')
out.write_text(svg, encoding='utf-8')
print(f'wrote {out} ({W}x{H_ACTUAL})')
