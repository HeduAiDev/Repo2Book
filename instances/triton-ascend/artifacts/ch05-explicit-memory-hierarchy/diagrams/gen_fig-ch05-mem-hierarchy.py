#!/usr/bin/env python3
"""fig-ch05-mem-hierarchy — layout 模板（一道 pybind 边界把 7 级地址空间筛成 5 级）。

修订史：
- 初版称「暴露并校验 4 级(UB/GM/L1/L0C)」——错。
- 二版改「3 级(UB/L1/L0C)」——仍错：L0C 从未被比较。
- 三版（本版）按源码四档重画：
  ① .td 定义 7 级：HIVMAttrs.td:L188-194（Zero=0/GM=1/L1=2/L0A=3/L0B=4/L0C=5/UB=6）
  ② pybind 只导出 5 级：ascend_ir.cc:L412-418 的 py::enum_ 只 .value() 了
     L1/UB/L0A/L0B/L0C —— Zero 与 GM 根本不进 Python
  ③ 边校验只认 2 级 UB/L1：全仓 .space 比较仅 5 处
     semantic.py:L104/L106、semantic.py:L123/L125、core.py:L300
  ④ L0C 仅契约：fixpipe docstring 要求 src 在 L0C，代码只 isinstance(src, tl.tensor)，
     而 tl.tensor 没有 .space 字段

全部坐标由常量/循环计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def text_w(s, size):
    """CJK 全角按 1.0em、其余按 0.58em 估宽。"""
    return sum(size * (1.0 if ord(ch) > 0x2E7F else 0.58) for ch in s)


def fit(s, maxw, base, floor=9.0):
    """自动缩字号直到文本宽度落进 maxw（防越界/压框）。"""
    size = base
    while size > floor and text_w(s, size) > maxw:
        size -= 0.5
    return size


W, H = 1500, 1000
PAD = 50

TITLE = "从 .td 到 Python：一道 pybind 边界，把 7 级地址空间筛成 5 级"
SUB1 = ".td 定义 7 级 → pybind 导出 5 级 → 搬运边真的比较的只有 2 级(UB / L1) → L0C 仅是 docstring 契约"
SUB2 = "写不出 space=GM 的 buffer，正是 GM↔UB 搬运要走 ch02 另一套机制的原因"

BLUE, ORANGE, GRAY, RED = "#1d4ed8", "#c2410c", "#94a3b8", "#b91c1c"

# ── 四档样式 ────────────────────────────────────────────────────────────
TIERS = {
    "checked": dict(fill="#dbeafe", stroke=BLUE, sw=2.6, dash=None,
                    name_fill="#1e3a8a", desc_fill="#1e40af",
                    pill="导出 + 边校验", pill_fill=BLUE,
                    pill_stroke=BLUE, pill_text="#ffffff", pill_dash=None),
    "contract": dict(fill="#fff7ed", stroke=ORANGE, sw=2.4, dash="7,5",
                     name_fill="#7c2d12", desc_fill="#9a3412",
                     pill="导出 + 契约，但不校验", pill_fill="#ffedd5",
                     pill_stroke=ORANGE, pill_text="#9a3412", pill_dash="4,3"),
    "exported": dict(fill="#f8fafc", stroke=GRAY, sw=1.6, dash=None,
                     name_fill="#334155", desc_fill="#64748b",
                     pill="导出，无校验", pill_fill="#e2e8f0",
                     pill_stroke="#cbd5e1", pill_text="#475569", pill_dash=None),
}

# ── ① .td 枚举 7 级：芯片条 ────────────────────────────────────────────
PANEL = (PAD, 96, W - 2 * PAD, 96)
CHIP_Y, CHIP_H, CHIP_W, CHIP_DX = 134, 44, 180, 188
CHIP_X0 = 66
# (名称, .td 值, 是否被 pybind 导出)
ENUM = [("Zero", 0, False), ("GM", 1, False), ("L1", 2, True), ("L0A", 3, True),
        ("L0B", 4, True), ("L0C", 5, True), ("UB", 6, True)]

# ── ② pybind 边界 ──────────────────────────────────────────────────────
BOUND_Y = 232

# ── ③ Python 语言层：导出的 5 级 ───────────────────────────────────────
ROW_B_Y, ROW_B_H = 320, 104
ROW_C_Y, ROW_C_H = 490, 78
ROW_D_Y, ROW_D_H = 614, 104

UB = (170, ROW_B_Y, 320, ROW_B_H)
L1 = (690, ROW_B_Y, 320, ROW_B_H)
L0A = (555, ROW_C_Y, 180, ROW_C_H)
L0B = (760, ROW_C_Y, 180, ROW_C_H)
L0C = (610, ROW_D_Y, 400, ROW_D_H)

BOXES = [
    (UB, "checked", "UB", "Unified Buffer(vector 侧)", "copy 的 src 必须是它；fixpipe 的 dst 必须是它"),
    (L1, "checked", "L1", "cube 输入缓冲", "copy 的 dst 可以是它；copy_from_ub_to_l1 只认它"),
    (L0A, "exported", "L0A", "cube 输入(语言层无搬运 API)", None),
    (L0B, "exported", "L0B", "cube 输入(语言层无搬运 API)", None),
    (L0C, "contract", "L0C", "cube 累加输出(Fractal NZ 布局)", "fixpipe 的 src「应当」在这里 —— 但没有任何一行代码检查"),
]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     f'<marker id="gray" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{GRAY}"/></marker>'
     f'<marker id="blue" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{BLUE}"/></marker>'
     f'<marker id="orange" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{ORANGE}"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="19" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{W/2}" y="60" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUB1)}</text>',
     f'<text x="{W/2}" y="80" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUB2)}</text>']

# ── ① 面板 + 7 枚芯片 ──────────────────────────────────────────────────
px, py, pw, ph = PANEL
L.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" fill="#fbfbfc" '
         f'stroke="#cbd5e1" stroke-width="1.4"/>')
L.append(f'<text x="{px+16}" y="{py+22}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#475569">'
         f'{esc("① MLIR / HIVM 方言侧：HIVMAttrs.td:L188-194 定义 7 级")}</text>')

chip_cx = {}
for i, (name, val, exported) in enumerate(ENUM):
    cx0 = CHIP_X0 + i * CHIP_DX
    ccx = cx0 + CHIP_W / 2
    chip_cx[name] = ccx
    if exported:
        fill, stroke, dash, nfill = "#f1f5f9", "#94a3b8", None, "#334155"
        mark, mfill = "✓ 已导出", "#15803d"
    else:
        fill, stroke, dash, nfill = "#fef2f2", RED, "6,4", "#7f1d1d"
        mark, mfill = "✗ 未导出", RED
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<rect x="{cx0}" y="{CHIP_Y}" width="{CHIP_W}" height="{CHIP_H}" rx="8" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"{d}/>')
    L.append(f'<text x="{ccx}" y="{CHIP_Y+19}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{nfill}">{esc(f"{name} = {val}")}</text>')
    L.append(f'<text x="{ccx}" y="{CHIP_Y+35}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="9.5" font-weight="bold" fill="{mfill}">{esc(mark)}</text>')
    # 已导出的：一小截向下的短线，穿过边界
    if exported:
        L.append(f'<line x1="{ccx}" y1="{CHIP_Y+CHIP_H}" x2="{ccx}" y2="{CHIP_Y+CHIP_H+18}" '
                 f'stroke="#94a3b8" stroke-width="1.4" marker-end="url(#gray)"/>')

# ── ② pybind 边界线 ────────────────────────────────────────────────────
L.append(f'<text x="{W/2}" y="{BOUND_Y-12}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#0f172a">'
         f'{esc("② pybind 边界：ascend_ir.cc:L412-418 的 py::enum_<hivm::AddressSpace> 只 .value() 了 5 个")}</text>')
L.append(f'<line x1="{PAD}" y1="{BOUND_Y}" x2="{W-PAD}" y2="{BOUND_Y}" stroke="#0f172a" '
         f'stroke-width="2.6" stroke-dasharray="12,6"/>')
L.append(f'<text x="{W/2}" y="{BOUND_Y+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#475569">'
         f'{esc("语言层能不能写出某一级，就取决于这几行 —— Zero 与 GM 根本不进 Python")}</text>')

# ── ③ 分区标题 ────────────────────────────────────────────────────────
L.append(f'<text x="{PAD}" y="272" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#475569">'
         f'{esc("③ Python 语言层：导出的 5 级")}</text>')


def pill(cx, cy, label, st):
    pw_, ph_ = text_w(label, 10.5) + 22, 21
    d = f' stroke-dasharray="{st["pill_dash"]}"' if st["pill_dash"] else ''
    L.append(f'<rect x="{cx-pw_/2:.1f}" y="{cy-ph_/2:.1f}" width="{pw_:.1f}" height="{ph_}" '
             f'rx="10.5" fill="{st["pill_fill"]}" stroke="{st["pill_stroke"]}" '
             f'stroke-width="1.2"{d}/>')
    L.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10.5" font-weight="bold" fill="{st["pill_text"]}">{esc(label)}</text>')


for (x, y, w, h), tier, name, desc, detail in BOXES:
    st = TIERS[tier]
    d = f' stroke-dasharray="{st["dash"]}"' if st["dash"] else ''
    cx, avail = x + w / 2, w - 20
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{st["fill"]}" '
             f'stroke="{st["stroke"]}" stroke-width="{st["sw"]}"{d}/>')
    L.append(f'<text x="{cx}" y="{y+24}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="{fit(name, avail, 14)}" font-weight="bold" '
             f'fill="{st["name_fill"]}">{esc(name)}</text>')
    L.append(f'<text x="{cx}" y="{y+43}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="{fit(desc, avail, 11)}" fill="{st["desc_fill"]}">{esc(desc)}</text>')
    if detail:
        L.append(f'<text x="{cx}" y="{y+62}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="{fit(detail, avail, 10.5)}" '
                 f'fill="{st["desc_fill"]}">{esc(detail)}</text>')
    pill(cx, y + h - 18, st["pill"], st)

# ── GM：MLIR 侧有、Python 里写不出 → 搬运只能走 ch02 的另一套机制 ──────
gm_x = chip_cx["GM"]
L.append(f'<line x1="{gm_x}" y1="{CHIP_Y+CHIP_H}" x2="{gm_x}" y2="{ROW_B_Y}" '
         f'stroke="{GRAY}" stroke-width="1.8" stroke-dasharray="6,5" '
         f'marker-end="url(#gray)" marker-start="url(#gray)"/>')
for i, ln in enumerate(("显式搬运 GM↔UB：走 ch02 的另一套机制",
                        "—— 因为 space=GM 的 buffer 根本写不出来")):
    L.append(f'<text x="{gm_x+16}" y="{272+i*18}" font-family="sans-serif" '
             f'font-size="10.5" fill="#64748b">{esc(ln)}</text>')

# ── L1 → L0A/L0B → L0C：硬件内部通路（灰虚线，语言层无 API）────────────
l1_cx = L1[0] + L1[2] / 2
l0c_cx = L0C[0] + L0C[2] / 2
for bx, bw in ((L0A[0], L0A[2]), (L0B[0], L0B[2])):
    L.append(f'<path d="M {l1_cx} {ROW_B_Y+ROW_B_H} L {bx+bw/2} {ROW_C_Y}" fill="none" '
             f'stroke="{GRAY}" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#gray)"/>')
    L.append(f'<path d="M {bx+bw/2} {ROW_C_Y+ROW_C_H} L {l0c_cx} {ROW_D_Y}" fill="none" '
             f'stroke="{GRAY}" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#gray)"/>')

# ── al.copy：UB → UB 自环（蓝）────────────────────────────────────────
loop_cx, loop_top, loop_bot = UB[0] - 6, ROW_B_Y + 16, ROW_B_Y + ROW_B_H - 16
L.append(f'<path d="M {loop_cx} {loop_top} C {loop_cx-46} {loop_top} {loop_cx-46} {loop_bot} '
         f'{loop_cx} {loop_bot}" fill="none" stroke="{BLUE}" stroke-width="2" '
         f'marker-end="url(#blue)"/>')
loop_ly = (loop_top + loop_bot) / 2
L.append(f'<text x="{loop_cx-72}" y="{loop_ly+4}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" font-weight="bold" fill="{BLUE}" '
         f'transform="rotate(-90 {loop_cx-72} {loop_ly+4})">{esc("al.copy")}</text>')

# ── al.copy：UB → L1（蓝，横向）───────────────────────────────────────
copy_y = ROW_B_Y + ROW_B_H / 2
copy_cx = (UB[0] + UB[2] + L1[0]) / 2
L.append(f'<line x1="{UB[0]+UB[2]}" y1="{copy_y}" x2="{L1[0]}" y2="{copy_y}" '
         f'stroke="{BLUE}" stroke-width="2.2" marker-end="url(#blue)"/>')
L.append(f'<text x="{copy_cx}" y="{copy_y-12}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="{BLUE}">{esc("al.copy: UB → {UB, L1}")}</text>')
L.append(f'<text x="{copy_cx}" y="{copy_y+20}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10" fill="{BLUE}">{esc("两端都被校验")}</text>')

# ── al.fixpipe：L0C → UB（橙，右侧长弧）────────────────────────────────
fx0, fy0 = L0C[0] + L0C[2], ROW_D_Y + ROW_D_H / 2
fx1, fy1 = UB[0] + UB[2] * 0.72, ROW_B_Y + ROW_B_H
ctrl_x = 1380
L.append(f'<path d="M {fx0} {fy0} C {ctrl_x} {fy0} {ctrl_x} {fy1+40} {fx1} {fy1}" fill="none" '
         f'stroke="{ORANGE}" stroke-width="2.4" marker-end="url(#orange)"/>')
# 标签放在弧线最右点（x≈1238）之外的空白右栏，横排，不用 rotate
for dy, fs, bold, txt in ((0, 12, ' font-weight="bold"', "al.fixpipe: L0C → UB"),
                          (22, 10.5, '', "只有到达端(UB)被校验")):
    L.append(f'<text x="1256" y="{580+dy}" font-family="sans-serif" font-size="{fs}"{bold} '
             f'fill="{ORANGE}">{esc(txt)}</text>')

# ── 左下两个说明框（蓝 = copy 两端都校验，橙 = fixpipe 只校验到达端）────
CALLOUTS = [
    (60, 476, 470, 104, BLUE, "#eff6ff", "al.copy(src, dst)：两端都真的比较 .space",
     [(0, "src.space != al.ascend_address_space.UB  → TypeError"),
      (0, "dst.space not in (al.ascend_address_space.L1,"),
      (18, "al.ascend_address_space.UB)  → TypeError")]),
    (60, 602, 470, 104, ORANGE, "#fff7ed", "al.fixpipe(src, dst, …)：只校验到达端",
     [(0, "dst.space != ascend_address_space.UB  → TypeError（到达端被拦）"),
      (0, "src 侧只有 isinstance(src, tl.tensor)：tl.tensor 没有 .space 字段，"),
      (0, "所以「src 必须在 L0C」拦不住 —— 靠人守约定，不靠语言层")]),
]
for cx0, cy0, cw, ch_, stroke, fill, head, lines in CALLOUTS:
    L.append(f'<rect x="{cx0}" y="{cy0}" width="{cw}" height="{ch_}" rx="9" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{cx0+16}" y="{cy0+25}" font-family="sans-serif" '
             f'font-size="{fit(head, cw-32, 12.5)}" font-weight="bold" '
             f'fill="{stroke}">{esc(head)}</text>')
    for i, (dx, ln) in enumerate(lines):
        L.append(f'<text x="{cx0+16+dx}" y="{cy0+48+i*20}" font-family="sans-serif" '
                 f'font-size="{fit(ln, cw-32-dx, 10.5)}" fill="#334155">{esc(ln)}</text>')

# ── 图例：左列四档、右列两类边 ────────────────────────────────────────
LEG_Y, LEG_DY = 780, 28
TIER_LEG = [
    (dict(fill="#fef2f2", stroke=RED, dash="6,4"),
     "① .td 有、pybind 未导出：Zero / GM —— Python 里根本写不出这一级"),
    (dict(fill=TIERS["checked"]["fill"], stroke=BLUE, dash=None),
     "② 导出 + 边校验：真的比较 .space、能抛 TypeError —— 只有 UB 与 L1"),
    (dict(fill=TIERS["contract"]["fill"], stroke=ORANGE, dash="7,5"),
     "③ 导出 + 契约但不校验：L0C 只写在 al.fixpipe 的 docstring 里"),
    (dict(fill=TIERS["exported"]["fill"], stroke=GRAY, dash=None),
     "④ 导出，但无校验：L0A / L0B —— 可传给 bl.alloc，但没有任何一条边比较它"),
]
for i, (sw_, label) in enumerate(TIER_LEG):
    ly = LEG_Y + i * LEG_DY
    d = f' stroke-dasharray="{sw_["dash"]}"' if sw_["dash"] else ''
    L.append(f'<rect x="{PAD}" y="{ly-11}" width="26" height="17" rx="5" fill="{sw_["fill"]}" '
             f'stroke="{sw_["stroke"]}" stroke-width="1.8"{d}/>')
    L.append(f'<text x="{PAD+36}" y="{ly+3}" font-family="sans-serif" font-size="11.5" '
             f'fill="#334155">{esc(label)}</text>')

EDGE_LEG = [(BLUE, None, "al.copy 覆盖边：UB → {UB, L1}"),
            (ORANGE, None, "al.fixpipe 覆盖边：L0C(名义)→ UB"),
            (GRAY, "5,4", "ch02 已建 / 硬件内部，本章不覆盖")]
for i, (color, dsh, label) in enumerate(EDGE_LEG):
    ly = LEG_Y + i * LEG_DY
    d = f' stroke-dasharray="{dsh}"' if dsh else ''
    L.append(f'<line x1="1050" y1="{ly-3}" x2="1080" y2="{ly-3}" stroke="{color}" '
             f'stroke-width="3"{d}/>')
    L.append(f'<text x="1090" y="{ly+3}" font-family="sans-serif" font-size="11.5" '
             f'fill="#334155">{esc(label)}</text>')

# ── 底部脚注 ──────────────────────────────────────────────────────────
FOOT = [
    ".td 定义 7 级（Zero=0 / GM=1 / L1=2 / L0A=3 / L0B=4 / L0C=5 / UB=6，HIVMAttrs.td:L188-194）"
    "→ pybind 导出 5 级（ascend_ir.cc:L412-418）→ 边校验 2 级（UB / L1）→ L0C 仅契约。",
    "全仓地址空间比较只有 5 处，全部只提 UB 与 L1：semantic.py:L104 / L106（copy_from_ub_to_l1）、"
    "semantic.py:L123 / L125（copy）、core.py:L300（fixpipe）。",
    "core.py:L152-163 的反射逐个包 ascend_ir.AddressSpace.__dict__，拿到的正是 pybind 导出的那 5 个；"
    "bl.alloc 原样透传 space，无白名单。",
    "基座 Triton 面对 GPU 时把这一层完全藏起来，由编译器代管共享内存。",
]
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{906+i*22}" font-family="sans-serif" font-size="11" '
             f'fill="#64748b">{esc(ln)}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch05-mem-hierarchy.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({W}x{H})')
