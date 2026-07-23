#!/usr/bin/env python3
"""fig-ch23-cube-datapath — tensor-flow 模板（flow 骨架 + 每条边标 pipe/说明）。
Cube 矩阵路径完整数据通路：L1 输入 ->(MTE1) L0A/L0B ->(M 乘累加) L0C 驻留累加 ->(FIX fixpipe) GM；
L0A/L0B 是 mmadL1 宏算子的内部缓冲，不出现在操作数上（用虚线内框圈出"内部"）。
全部坐标由循环/常量计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def text_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E7F else 0.58) for ch in s)

def fit(s, maxw, base, floor=9.0):
    size = base
    while size > floor and text_w(s, size) > maxw:
        size -= 0.5
    return size

TITLE = "Cube 矩阵路径：六级内存里用满五级的完整数据通路"
SUBTITLE = "L1 输入 →(MTE1) L0A/L0B(mmadL1 内部) →(M 乘累加) L0C 驻留累加 →(FIX,fixpipe) GM"

PAD, TOP = 50, 150
BOX_W, BOX_H = 190, 92
# 每条边的间隔单独给（边 1 紧邻内部虚线框右侧悬垂，需更宽间隔避免标签压框）
GAPS = [130, 190, 130]

STAGES = [
    ("L1(cbuf)", "输入 A/B 驻留", "#fef9c3", "#a16207", "#78350f", False),
    ("L0A/L0B(ca/cb)", "mmadL1 内部缓冲\n不出现在操作数上", "#f1f5f9", "#94a3b8", "#475569", True),
    ("L0C(cc)", "C 累加器驻留\nC += A×B（K 轮不回写）", "#fed7aa", "#c2410c", "#7c2d12", False),
    ("GM", "收尾搬出", "#e0e7ff", "#4338ca", "#3730a3", False),
]
EDGES = [
    ("MTE1", "L1 → L0A/L0B（宏算子内部流水）"),
    ("M（Cube 矩阵单元）", "乘累加 A×B → 写入 L0C"),
    ("FIX（fixpipe）", "L0C → GM，K 轮只搬这一次"),
]

n = len(STAGES)
w = PAD * 2 + n * BOX_W + sum(GAPS)
h = TOP + BOX_H + 300

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

xs_ = [PAD]
for i in range(1, n):
    xs_.append(xs_[-1] + BOX_W + GAPS[i-1])
cy = TOP + BOX_H / 2

# 内部缓冲的虚线外框（"mmadL1 内部"分组标注），画在方框之前作为背景
internal_i = 1
ix, iw = xs_[internal_i] - 16, BOX_W + 32
L.append(f'<rect x="{ix}" y="{TOP-34}" width="{iw}" height="{BOX_H+70}" rx="12" '
          f'fill="none" stroke="#cbd5e1" stroke-width="1.6" stroke-dasharray="7,5"/>')
L.append(f'<text x="{ix+iw/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#64748b">{esc("mmadL1 宏算子内部（不对外暴露）")}</text>')

for i, (name, desc, fill, stroke, tf, internal) in enumerate(STAGES):
    x = xs_[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+28}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(name, BOX_W-16, 16)}" font-weight="bold" fill="{tf}">{esc(name)}</text>')
    lines = desc.split("\n")
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{TOP+52+k*18}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{fit(ln, BOX_W-16, 11.5)}" fill="{tf}">{esc(ln)}</text>')

for i in range(n - 1):
    x1, x2 = xs_[i] + BOX_W, xs_[i+1]
    L.append(f'<line x1="{x1+4}" y1="{cy}" x2="{x2-4}" y2="{cy}" '
              f'stroke="#334155" stroke-width="2.2" marker-end="url(#a)"/>')
    name, desc = EDGES[i]
    L.append(f'<text x="{(x1+x2)/2}" y="{cy-16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(name, x2-x1-8, 12.5)}" font-weight="bold" fill="#334155">{esc(name)}</text>')
    L.append(f'<text x="{(x1+x2)/2}" y="{cy+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(desc, x2-x1-8, 9.5)}" fill="#64748b">{esc(desc)}</text>')

# provenance
prov_y = TOP + BOX_H + 70
PROV = [
    "HIVMMacroOps.td:L166；夹具 L11/L13 CHECK <cbuf>",
    "HIVMMacroOps.td:L62 MacroOpPipeTrait<PIPE_MTE1,PIPE_M>",
    "InferHIVMMemScope.cpp:L228-229；夹具 L7/L33 CHECK <cc>",
    "HIVMDMAOps.td:L272,L280；夹具 L57 fixpipe ins(cc) outs(gm)",
]
for i, p in enumerate(PROV):
    x = xs_[i] + BOX_W/2
    L.append(f'<text x="{x}" y="{prov_y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(p, BOX_W+90, 10)}" fill="#94a3b8">{esc(p)}</text>')

foot_y0 = prov_y + 46
FOOT = [
    "Cube 路径闭环：A/B 从 L1 经 MTE1 进内部工位 L0A/L0B，M 单元把乘积累加进 L0C 上常驻的 C",
    "（K 方向每轮 C+=A×B、不回写），整批算完由 fixpipe 一次搬回 GM —— 六级里用满五级，且 L0A/L0B 被宏算子藏在内部。",
]
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*22}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(ln)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch23-cube-datapath.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
