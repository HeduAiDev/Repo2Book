#!/usr/bin/env python3
"""fig-encoding-to-fragment (flow 模板,1-1 对应)
NvidiaMmaEncodingAttr/DotOperandEncodingAttr 的五个字段逐一对应 fragment 契约的一条要求,
每条连线标源码锚点。左列=编码字段,右列=fragment 契约项。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "编码字段 <-> fragment 契约:逐项对应,没有一个字段是自由发挥"
PAIRS = [
    ("versionMajor", "选哪一代 fragment 表\n(Volta/Ampere/Hopper)",
     "getMMAVersionSafe\nAccelerateMatmul.cpp:L26-L45"),
    ("instrShape = [16,8]", "单条 mma.sync 指令的\n『单位砖』",
     "TritonGPUAttrDefs.td\nL1100-L1103, L1130-L1137"),
    ("warpsPerCTA", "把单位砖平铺满\n整块输出 tile",
     "warpsPerTileV2\nAccelerateMatmul.cpp:L82-L104"),
    ("opIdx (0=A,1=B)", "kWidth 摆哪一维 /\nM-N 是否交换",
     "getContigPerThread .td:L1348-1359\ngetThreadsPerWarp Dialect.cpp:L2164-2175"),
    ("kWidth = 32/bitwidth", "每线程沿 K 的\n寄存器打包宽度",
     "TritonGPUAttrDefs.td:L1339-1341\n校验器 Dialect.cpp:L1076-1092"),
]

BOX_W, BOX_H = 260, 64
VGAP = 34
PAD = 50
TOP = 130
MID_GAP = 300  # 两列中间留给箭头+锚点标注

n = len(PAIRS)
w = PAD * 2 + BOX_W * 2 + MID_GAP
h = TOP + n * (BOX_H + VGAP) - VGAP + 70

left_x = PAD
right_x = PAD + BOX_W + MID_GAP
mid_cx = PAD + BOX_W + MID_GAP / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#6366f1"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="72" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">{esc("左:NvidiaMmaEncodingAttr / DotOperandEncodingAttr 字段    右:warp 级 mma.sync 的 fragment 硬件契约要求")}</text>')

L.append(f'<text x="{left_x+BOX_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#3730a3">{esc("编码字段")}</text>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#3730a3">{esc("fragment 契约要求")}</text>')

for i, (field, req, anchor) in enumerate(PAIRS):
    y = TOP + i * (BOX_H + VGAP)
    cy = y + BOX_H / 2

    # 左盒(字段)
    L.append(f'<rect x="{left_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>')
    L.append(f'<text x="{left_x+BOX_W/2}" y="{cy+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#312e81">{esc(field)}</text>')

    # 右盒(契约要求,多行)
    L.append(f'<rect x="{right_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#ecfdf5" stroke="#059669" stroke-width="2"/>')
    req_lines = req.split("\n")
    ry0 = cy - (len(req_lines) - 1) * 8
    for li, rl in enumerate(req_lines):
        L.append(f'<text x="{right_x+BOX_W/2}" y="{ry0+li*16+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="#065f46">{esc(rl)}</text>')

    # 箭头
    L.append(f'<line x1="{left_x+BOX_W}" y1="{cy}" x2="{right_x-6}" y2="{cy}" '
              'stroke="#6366f1" stroke-width="1.8" marker-end="url(#a)"/>')

    # 锚点标注(箭头上方,小字)
    anchor_lines = anchor.split("\n")
    ay0 = cy - 14 - (len(anchor_lines) - 1) * 12
    for li, al in enumerate(anchor_lines):
        L.append(f'<text x="{mid_cx}" y="{ay0+li*12}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#64748b">{esc(al)}</text>')

cap = "五字段命中前提:versionMajor 选代际、instrShape 定砖、warpsPerCTA 铺砖、opIdx 分 A/B、kWidth 定寄存器打包宽度。"
L.append(f'<text x="{PAD}" y="{h-16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(cap)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-encoding-to-fragment.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
