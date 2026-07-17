#!/usr/bin/env python3
"""fig-m5-amdgcn-hsaco-toolchain：工具链末端两后端同构分叉——共用
llvm.translate_to_asm 出汇编，AMD 走 assemble_amdgcn→ld.lld 出 hsaco，
NVIDIA 走 ptxas 出 cubin。flow 模板：单入口 -> 分叉两路径。
全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def esc_ml(s):
    return s.split("\n")


TITLE = "工具链末端：共用汇编入口，AMD/NVIDIA 分叉两条链"
SUBTITLE = "LLVM-IR → translate_to_asm（两后端共用）→ 之后各走各的汇编器/链接器"

BOX_W, BOX_H, HGAP, PAD, TOP = 260, 56, 46, 40, 100
LANE_GAP = 120

# 共享节点（单条主干）
SHARED = [
    ("LLVM-IR", "make_llir 产出（第 36/37 章骨架）"),
    ("translate_to_asm", "共用汇编入口（AMD/NVIDIA 各传各的 target triple/arch）"),
]

# 两条分支：每项 (标题, 副标签, 是否为终态)
AMD_CHAIN = [
    ("amdgcn 汇编文本", None, False),
    ("assemble_amdgcn", "AMD 汇编器", False),
    ("目标文件字节", "assemble_amdgcn 返回值", False),
    ("ld.lld -shared", "AMD 链接器", False),
    ("hsaco", "最终产物（ELF 共享对象）", True),
]
NVIDIA_CHAIN = [
    ("PTX 文本", None, False),
    ("ptxas", "NVIDIA 汇编器（出 cubin）", False),
    ("cubin", "最终产物", True),
]

w = PAD * 2 + BOX_W + HGAP + max(len(AMD_CHAIN), len(NVIDIA_CHAIN)) * 0  # placeholder, recompute below
# 横向流程：shared 纵向两个节点，然后画布右侧两条纵向并排 lane，各自横向流程？
# 采用更清晰的版式：shared 节点纵向居中在顶部，随后左右两条“斜下分叉”到两个 lane 的起点，
# 每条 lane 内部是纵向从上到下的流程（贴合原图 "amdgcn/hsaco vs ptxas/cubin" 的从属关系）。

LANE_TOP = TOP + 2 * (BOX_H + 50) + 40
n_max = max(len(AMD_CHAIN), len(NVIDIA_CHAIN))
w = PAD * 2 + BOX_W * 2 + LANE_GAP
N_FOOT_LINES = 4
FOOT_LINE_H = 16
foot_block_h = N_FOOT_LINES * FOOT_LINE_H + 26
lanes_bottom = LANE_TOP + n_max * (BOX_H + 34)
h = lanes_bottom + foot_block_h + PAD

cx_main = w / 2
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
     '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{cx_main}" y="{PAD}" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{cx_main}" y="{PAD+22}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# --- 共享主干（居中，纵向两个节点） ---
shared_y = []
for i, (name, sub) in enumerate(SHARED):
    y = TOP + i * (BOX_H + 50)
    shared_y.append(y)
    L.append(f'<rect x="{cx_main-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
    L.append(f'<text x="{cx_main}" y="{y+24}" text-anchor="middle" font-family="monospace" '
              f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    if sub:
        L.append(f'<text x="{cx_main}" y="{y+42}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" fill="#334155">{esc(sub)}</text>')
    if i > 0:
        py = shared_y[i - 1] + BOX_H
        L.append(f'<line x1="{cx_main}" y1="{py}" x2="{cx_main}" y2="{y-4}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

shared_bottom = shared_y[-1] + BOX_H
lane_amd_cx = PAD + BOX_W / 2
lane_nv_cx = w - PAD - BOX_W / 2

# 分叉线：从 shared 底部中点，斜向连到两条 lane 的首节点顶部
L.append(f'<line x1="{cx_main}" y1="{shared_bottom}" x2="{lane_amd_cx}" y2="{LANE_TOP-4}" '
          'stroke="#dc2626" stroke-width="1.8" marker-end="url(#ar)"/>')
L.append(f'<line x1="{cx_main}" y1="{shared_bottom}" x2="{lane_nv_cx}" y2="{LANE_TOP-4}" '
          'stroke="#15803d" stroke-width="1.8" marker-end="url(#ag)"/>')

def label_with_halo(x, y, text, color, anchor="middle"):
    tw = len(text) * 11 + 12  # 粗估宽度（含 CJK），留白背景避免压线
    L.append(f'<rect x="{x-tw/2 if anchor=="middle" else x-4}" y="{y-15}" width="{tw}" height="19" '
              f'rx="4" fill="white" opacity="0.9"/>')
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="{color}">{esc(text)}</text>')


label_with_halo((cx_main + lane_amd_cx) / 2 - 20, (shared_bottom + LANE_TOP) / 2 - 16,
                "AMD 分支", "#dc2626")
label_with_halo((cx_main + lane_nv_cx) / 2 + 20, (shared_bottom + LANE_TOP) / 2 - 16,
                "NVIDIA 分支", "#15803d")


def draw_lane(cx, chain, color):
    fill, stroke = color
    ys = []
    for i, (name, sub, is_final) in enumerate(chain):
        y = LANE_TOP + i * (BOX_H + 34)
        ys.append(y)
        bfill = fill if not is_final else stroke
        tfill = "#0f172a" if not is_final else "white"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{bfill}" stroke="{stroke}" stroke-width="{2.5 if is_final else 1.5}"/>')
        L.append(f'<text x="{cx}" y="{y+24}" text-anchor="middle" font-family="monospace" '
                  f'font-size="13" font-weight="bold" fill="{tfill}">{esc(name)}</text>')
        if sub:
            L.append(f'<text x="{cx}" y="{y+42}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="10" fill="{tfill}">{esc(sub)}</text>')
        if i > 0:
            py = ys[i - 1] + BOX_H
            mk = "ar" if stroke == "#dc2626" else "ag"
            L.append(f'<line x1="{cx}" y1="{py}" x2="{cx}" y2="{y-4}" '
                      f'stroke="{stroke}" stroke-width="1.5" marker-end="url(#{mk})"/>')


draw_lane(lane_amd_cx, AMD_CHAIN, ("#fee2e2", "#dc2626"))
draw_lane(lane_nv_cx, NVIDIA_CHAIN, ("#dcfce7", "#15803d"))

foot_y = h - foot_block_h + 16
FOOT_LINES = [
    "translate_to_asm 是两后端共用的汇编入口：third_party/amd/backend/compiler.py:L338，",
    "third_party/nvidia/backend/compiler.py:L324。",
    "AMD：assemble_amdgcn（L346）→ ld.lld -flavor gnu -shared（L353）出 hsaco；",
    "NVIDIA：ptxas（L341-353）直接出 cubin。",
]
assert len(FOOT_LINES) == N_FOOT_LINES
for k, line in enumerate(FOOT_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y+k*FOOT_LINE_H}" font-family="sans-serif" font-size="10.8" '
              f'fill="#64748b">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m5-amdgcn-hsaco-toolchain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
