#!/usr/bin/env python3
"""fig-m6-descent-to-ptx: flow 模板(纵向 3 级)。TTGIR → LLIR → PTX 最后两跳,
每跳标注真实 IR 地标。数字全部来自 _attn_fwd.{ttgir,llir,ptx}(spec.numbers)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "TTGIR 降两跳到硬件:make_llir → make_ptx"

STAGES = [
    ("TTGIR", "#475569", "#f1f5f9",
     ["3 块 memdesc:K/V 双缓冲(×2) + Q 常驻"]),
    ("LLVM IR", "#0369a1", "#e0f2fe",
     ["define ptx_kernel void @_attn_fwd(...)  ·  L9",
      "llvm.nvvm.read.ptx.sreg.ctaid.x / tid.x  ·  L10,L31",
      "共享内存 = 49152 字节 = 2×(2×64×64×2B) + 128×64×2B"]),
    ("PTX", "#b45309", "#fef3c7",
     [".target sm_120a, .reqntid 128(=4 warps×32)  ·  L6,L37",
      "mma.sync.aligned.m16n8k16...f32.f16.f16.f32  ·  L644 首条,全核 256 条",
      "ex2.approx.ftz.f32  ·  L1090 首条,全核 136 条",
      "cp.async.cg.shared.global + commit_group  ·  L251,L265,全核 48 条"]),
]
ARROW_LABELS = [
    "make_llir(allocate_shared_memory 给三块 memdesc 分配共享内存)",
    "make_ptx",
]

PAD, TOP, BOX_W = 40, 78, 780
VGAP = 58

def box_h(n_lines):
    return 32 + n_lines * 20 + 12

heights = [box_h(len(s[3])) for s in STAGES]
w = PAD * 2 + BOX_W
h = TOP + sum(heights) + VGAP * (len(STAGES) - 1) + 20 + 3 * 20 + 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

cx = PAD + BOX_W / 2
y = TOP
box_tops = []
for i, (name, stroke, fill, lines) in enumerate(STAGES):
    bh = heights[i]
    box_tops.append((y, bh))
    L.append(f'<rect x="{PAD}" y="{y}" width="{BOX_W}" height="{bh}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{cx}" y="{y+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="{stroke}">{esc(name)}</text>')
    for k, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{y+46+k*20}" text-anchor="middle" '
                  f'font-family="monospace" font-size="11" fill="#0f172a">{esc(ln)}</text>')
    if i < len(STAGES) - 1:
        gap_top = y + bh
        gap_bot = gap_top + VGAP
        lbl_y = gap_top + 20
        arrow_y1 = gap_top + 28
        arrow_y2 = gap_bot - 4
        L.append(f'<text x="{cx}" y="{lbl_y}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11.5" font-weight="bold" fill="#334155">{esc(ARROW_LABELS[i])}</text>')
        L.append(f'<line x1="{cx}" y1="{arrow_y1}" x2="{cx}" y2="{arrow_y2}" '
                  'stroke="#334155" stroke-width="2.2" marker-end="url(#a)"/>')
    y += bh + VGAP

foot_lines = [
    "结论:全链收口——一行 tl.dot 最终是 mma.sync.m16n8k16 的 Tensor Core 指令,一行 tl.math.exp2 是",
    "ex2.approx 硬件指令,tl.load+软件流水是 cp.async 异步拷贝;#mma versionMajor=2 → m16n8k16 是",
    "Ampere 系 MMA 形状,在 3.2.0↔3.6.0、Ampere/Hopper/Blackwell(fp16)上稳定。",
]
foot_y0 = y - VGAP + 40
for i, fl in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*20}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(fl)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m6-descent-to-ptx.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
